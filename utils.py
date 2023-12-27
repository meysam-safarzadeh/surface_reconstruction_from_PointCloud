import torch.utils.data as data
import numpy as np
import math
import torch
import os
import errno
import open3d as o3d
from skimage import measure


def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def isdir(dirname):
    return os.path.isdir(dirname)


def normalize_pts(input_pts):
    center_point = np.mean(input_pts, axis=0)
    center_point = center_point[np.newaxis, :]
    centered_pts = input_pts - center_point

    largest_radius = np.amax(np.sqrt(np.sum(centered_pts ** 2, axis=1)))
    normalized_pts = centered_pts / largest_radius  # / 1.03  if we follow DeepSDF completely

    return normalized_pts


def normalize_normals(input_normals):
    normals_magnitude = np.sqrt(np.sum(input_normals ** 2, axis=1))
    normals_magnitude = normals_magnitude[:, np.newaxis]

    normalized_normals = input_normals / normals_magnitude

    return normalized_normals


def showMeshReconstruction(IF):
    """
    calls marching cubes on the input implicit function sampled in the 3D grid
    and shows the reconstruction mesh
    Args:
        IF    : implicit function sampled at the grid points
    Returns:
        verts, triangles: vertices and triangles of the polygon mesh after iso-surfacing it at level 0
    """
    verts, triangles, normals, values = measure.marching_cubes(IF, 0)

    # Create an empty triangle mesh
    mesh = o3d.geometry.TriangleMesh()
    # Use mesh.vertex to access the vertices' attributes    
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    # Use mesh.triangle to access the triangles' attributes    
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
    return verts, triangles


def signed_distance(p, points, normals):
    """
    Computes the signed distance between a point p and a surface defined by a set of points and normals.

    Args:
        p (ndarray): 3D point.
        points (ndarray): 3D points that define the surface.
        normals (ndarray): 3D normals of the surface at the corresponding points.

    Returns:
        The signed distance between the point p and the surface.
    """

    diffs = p - points

    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    signs = np.sign(np.sum(diffs * normals, axis=1))
    sd = np.multiply(dists, signs)
    return sd


class SdfDataset(data.Dataset):
    def __init__(self, points=None, normals=None, phase='train', args=None):
        self.phase = phase

        if self.phase == 'test':
            self.bs = args.test_batch
            max_dimensions = np.ones((3,)) * args.max_xyz
            min_dimensions = -np.ones((3,)) * args.max_xyz

            bounding_box_dimensions = max_dimensions - min_dimensions  # compute the bounding box dimensions of the point cloud
            grid_spacing = max(bounding_box_dimensions) / (
                    args.grid_N - 9)  # each cell in the grid will have the same size
            X, Y, Z = np.meshgrid(list(
                np.arange(min_dimensions[0] - grid_spacing * 4, max_dimensions[0] + grid_spacing * 4, grid_spacing)),
                list(np.arange(min_dimensions[1] - grid_spacing * 4,
                               max_dimensions[1] + grid_spacing * 4,
                               grid_spacing)),
                list(np.arange(min_dimensions[2] - grid_spacing * 4,
                               max_dimensions[2] + grid_spacing * 4,
                               grid_spacing)))  # N x N x N
            self.grid_shape = X.shape
            self.samples_xyz = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).transpose()
            self.number_samples = self.samples_xyz.shape[0]
            self.number_batches = math.ceil(self.number_samples * 1.0 / self.bs)

        else:
            self.points = points
            self.normals = normals
            self.sample_std = args.sample_std
            self.bs = args.train_batch
            self.number_points = self.points.shape[0]
            self.number_samples = int(self.number_points * args.N_samples)
            self.number_batches = math.ceil(self.number_samples * 1.0 / self.bs)

            if phase == 'val':
                print('val init')
                self.samples_sdf = np.zeros((self.number_samples,))
                self.samples_xyz = np.zeros((self.number_samples, 3))

                for i in range(self.number_points):
                    # index = np.random.choice(self.number_points, 1)[0]
                    point = self.points[i, :]
                    normal = self.normals[i, :]

                    # Sample a point around the surface point along its normal direction
                    sample_point = (np.tile(point, (args.N_samples, 1)) +
                                    np.random.normal(0, self.sample_std, size=(args.N_samples, 3)) *
                                    np.tile(normal, (args.N_samples, 1)))
                    # print(sample_point.shape)

                    self.samples_xyz[(i) * args.N_samples:(i + 1) * args.N_samples, :] = sample_point
                    self.samples_sdf[(i) * args.N_samples:(i + 1) * args.N_samples] = signed_distance(sample_point,
                                                                                                      point,
                                                                                                      normal)

            if phase == 'train':
                print('train init')
                self.samples_sdf = np.zeros((self.number_samples,))
                self.samples_xyz = np.zeros((self.number_samples, 3))

                for i in range(self.number_points):
                    # index = np.random.choice(self.number_points, 1)[0]
                    point = self.points[i, :]
                    normal = self.normals[i, :]

                    # Sample a point around the surface point along its normal direction
                    sample_point = (np.tile(point, (args.N_samples, 1)) +
                                    np.random.normal(0, self.sample_std, size=(args.N_samples, 3)) *
                                    np.tile(normal, (args.N_samples, 1)))

                    self.samples_xyz[(i) * args.N_samples:(i + 1) * args.N_samples, :] = sample_point
                    self.samples_sdf[(i) * args.N_samples:(i + 1) * args.N_samples] = signed_distance(sample_point,
                                                                                                      point, normal)

                new_row_indices = np.random.permutation(self.samples_xyz.shape[0])

                # use advanced indexing to select the rows in the new order
                self.samples_xyz = self.samples_xyz[new_row_indices]
                self.samples_sdf = self.samples_sdf[new_row_indices]

    def __len__(self):
        return self.number_batches

    def __getitem__(self, idx):
        start_idx = idx * self.bs
        end_idx = min(start_idx + self.bs, self.number_samples)  # exclusive
        if self.phase == 'val':
            xyz = self.samples_xyz[start_idx:end_idx, :]
            gt_sdf = self.samples_sdf[start_idx:end_idx]

        elif self.phase == 'train':  # sample points on the fly
            xyz = self.samples_xyz[start_idx:end_idx, :]
            gt_sdf = self.samples_sdf[start_idx:end_idx]

        else:
            assert self.phase == 'test'
            xyz = self.samples_xyz[start_idx:end_idx, :]

        if self.phase == 'test':
            return {'xyz': torch.FloatTensor(xyz)}
        else:
            return {'xyz': torch.FloatTensor(xyz), 'gt_sdf': torch.FloatTensor(gt_sdf)}
