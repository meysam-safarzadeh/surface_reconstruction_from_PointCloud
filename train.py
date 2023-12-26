import os
import shutil
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from model import Decoder
from utils import normalize_pts, normalize_normals, SdfDataset, mkdir_p, isdir, showMeshReconstruction

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# function to save a checkpoint during training, including the best model so far
def save_checkpoint(state, is_best, checkpoint_folder='checkpoints/', filename='checkpoint.pth.tar'):
    checkpoint_file = os.path.join(checkpoint_folder, 'checkpoint_{}.pth.tar'.format(state['epoch']))
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, os.path.join(checkpoint_folder, 'model_best.pth.tar'))


def train(dataset, model, optimizer, args):
    model.train()  # switch to train mode
    loss_sum = 0.0
    loss_count = 0.0
    sigma = 0.1
    num_batch = len(dataset)
    for i in range(num_batch):
        data = dataset[i]  # a dict
        xyz_tensor = data['xyz'].to(device)
        sdf_gt_tensor = data['gt_sdf'].to(device)

        # Forward pass
        optimizer.zero_grad()
        sdf_pred_tensor = model(xyz_tensor)

        # Compute loss
        clamped_sdf_pred_tensor = torch.clamp(sdf_pred_tensor, -sigma, sigma)

        clamped_sdf_gt_tensor = torch.clamp(sdf_gt_tensor, -sigma, sigma)
        clamped_sdf_gt_tensor = clamped_sdf_gt_tensor.view(-1, 1)
        loss_tensor = torch.abs(clamped_sdf_pred_tensor - clamped_sdf_gt_tensor)
        loss = torch.sum(loss_tensor)
        # a = clamped_sdf_pred_tensor - clamped_sdf_gt_tensor
        # print(loss_tensor.shape)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update loss stats
        loss_sum += loss.item() * xyz_tensor.shape[0]
        loss_count += xyz_tensor.shape[0]

    return loss_sum / loss_count


# validation function
def val(dataset, model, optimizer, args):
    model.eval()  # switch to test mode
    loss_sum = 0.0
    loss_count = 0.0
    sigma = 0.1
    num_batch = len(dataset)
    for i in range(num_batch):
        data = dataset[i]  # a dict
        xyz_tensor = data['xyz'].to(device)
        sdf_gt_tensor = data['gt_sdf'].to(device)

        # Forward pass
        optimizer.zero_grad()
        sdf_pred_tensor = model(xyz_tensor)

        # Compute loss
        clamped_sdf_pred_tensor = torch.clamp(sdf_pred_tensor, -sigma, sigma)
        clamped_sdf_gt_tensor = torch.clamp(sdf_gt_tensor, -sigma, sigma)
        clamped_sdf_gt_tensor = clamped_sdf_gt_tensor.view(-1, 1)
        loss_tensor = torch.abs(clamped_sdf_pred_tensor - clamped_sdf_gt_tensor)
        loss = torch.sum(loss_tensor)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update loss stats
        loss_sum += loss.item() * xyz_tensor.shape[0]
        loss_count += xyz_tensor.shape[0]

    return loss_sum / loss_count


# testing function
def test(dataset, model, args):
    model.eval()  # switch to test mode
    num_batch = len(dataset)
    number_samples = dataset.number_samples
    grid_shape = dataset.grid_shape
    IF = np.zeros((number_samples, ))
    start_idx = 0
    for i in range(num_batch):
        data = dataset[i]  # a dict
        xyz_tensor = data['xyz'].to(device)
        this_bs = xyz_tensor.shape[0]
        end_idx = start_idx + this_bs
        with torch.no_grad():
            pred_sdf_tensor = model(xyz_tensor)
            pred_sdf_tensor = torch.clamp(pred_sdf_tensor, -args.clamping_distance, args.clamping_distance)
        pred_sdf = pred_sdf_tensor.cpu().squeeze().numpy()
        IF[start_idx:end_idx] = pred_sdf
        start_idx = end_idx
    IF = np.reshape(IF, grid_shape)

    verts, triangles = showMeshReconstruction(IF)
    with open('test.obj', 'w') as outfile:
        for v in verts:
            outfile.write( "v " + str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + "\n" )
        for f in triangles:
            outfile.write( "f " + str(f[0]+1) + " " + str(f[1]+1) + " " + str(f[2]+1) + "\n" )
    outfile.close()
    return


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepSDF')

    parser.add_argument("-e", "--evaluate", default=False, action="store_true",
                        help="Activate test mode - Evaluate model on val/test set (no training)")

    # paths you may want to adjust
    parser.add_argument("--input_pts", default="data/bunny-1000.pts", type=str, help="Input point cloud")
    parser.add_argument("--checkpoint_folder", default="checkpoints/", type=str, help="Folder to save checkpoints")
    parser.add_argument("--resume_file", default="model_best.pth.tar", type=str,
                        help="Path to retrieve latest checkpoint file relative to checkpoint folder")

    # hyperameters of network/options for training
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay/L2 regularization on weights")
    parser.add_argument("--lr", default=1e-4, type=float, help="Initial learning rate")
    parser.add_argument("--schedule", type=int, nargs="+", default=[40, 50],
                        help="Decrease learning rate at these milestone epochs.")
    parser.add_argument("--gamma", default=0.1, type=float,
                        help="Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestone epochs")
    parser.add_argument("--start_epoch", default=0, type=int, help="Start from specified epoch number")
    parser.add_argument("--epochs", default=100, type=int,
                        help="Number of epochs to train (when loading a previous model, it will train for an extra number of epochs)")
    parser.add_argument("--train_batch", default=512, type=int, help="Batch size for training")
    parser.add_argument("--train_split_ratio", default=0.8, type=float, help="ratio of training split")
    parser.add_argument("--N_samples", default=100, type=float,
                        help="for each input point, N samples are used for training or validation")
    parser.add_argument("--sample_std", default=0.05, type=float,
                        help="we perturb each surface point along normal direction with mean-zero Gaussian noise with the given standard deviation")
    parser.add_argument("--clamping_distance", default=0.1, type=float, help="clamping distance for sdf")

    # various options for testing and evaluation
    parser.add_argument("--test_batch", default=2048, type=int, help="Batch size for testing")
    parser.add_argument("--grid_N", default=128, type=int, help="construct a 3D NxNxN grid containing the point cloud")
    parser.add_argument("--max_xyz", default=1.0, type=float, help="largest xyz coordinates")

    args = parser.parse_args()
    print(args)
    main(args)