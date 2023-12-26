import os
import shutil
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from model import Decoder
from utils import normalize_pts, normalize_normals, SdfDataset, mkdir_p, isdir, showMeshReconstruction

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    best_loss = 2e10
    best_epoch = -1

    # create checkpoint folder
    if not isdir(args.checkpoint_folder):
        print("Creating new checkpoint folder " + args.checkpoint_folder)
        mkdir_p(args.checkpoint_folder)

    # default architecture in DeepSDF
    model = Decoder(args)
    model.to(device)
    print("=> Will use the (" + device.type + ") device.")

    # cudnn will optimize execution for our network
    cudnn.benchmark = True

    if args.evaluate:
        print("\nEvaluation only")
        path_to_resume_file = os.path.join(args.checkpoint_folder, args.resume_file)
        print("=> Loading training checkpoint '{}'".format(path_to_resume_file))
        checkpoint = torch.load(path_to_resume_file)
        model.load_state_dict(checkpoint['state_dict'])
        test_dataset = SdfDataset(phase='test', args=args)
        test(test_dataset, model, args)
        return

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    print("=> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # create dataset
    input_point_cloud = np.loadtxt(args.input_pts)
    training_points = normalize_pts(input_point_cloud[:, :3])
    training_normals = normalize_normals(input_point_cloud[:, 3:])
    n_points = training_points.shape[0]
    print("=> Number of points in input point cloud: %d" % n_points)

    # split dataset into train and validation set by args.train_split_ratio
    n_points_train = int(args.train_split_ratio * n_points)
    full_indices = np.arange(n_points)
    np.random.shuffle(full_indices)
    train_indices = full_indices[:n_points_train]
    val_indices = full_indices[n_points_train:]
    train_dataset = SdfDataset(points=training_points[train_indices], normals=training_normals[train_indices], args=args)
    val_dataset = SdfDataset(points=training_points[val_indices], normals=training_normals[val_indices], phase='val', args=args)

    # perform training!
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=args.gamma)

    for epoch in range(args.start_epoch, args.epochs):
        # breakpoint()
        train_loss = train(train_dataset, model, optimizer, args)
        val_loss = val(val_dataset, model, optimizer, args)
        scheduler.step()
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            best_epoch = epoch
        save_checkpoint({"epoch": epoch + 1, "state_dict": model.state_dict(), "best_loss": best_loss, "optimizer": optimizer.state_dict()},
                        is_best, checkpoint_folder=args.checkpoint_folder)
        print(f"Epoch {epoch+1:d}. train_loss: {train_loss:.8f}. val_loss: {val_loss:.8f}. Best Epoch: {best_epoch+1:d}. Best val loss: {best_loss:.8f}.")

