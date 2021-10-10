#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


# Part of the code is referred from: https://github.com/charlesq34/pointnet

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    # download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '../../datasets')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=768):
    pointcloud1 = pointcloud1.T
    pointcloud2 = pointcloud2.T
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y), n_jobs=1).fit(pointcloud1)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y), n_jobs=1).fit(pointcloud2)
    random_p2 = random_p1  # np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud1[idx1, :].T, pointcloud2[idx2, :].T


def pad_data(surface2, L):
    surface_temp_2 = np.zeros((4096))
    surface_temp_2[:len(surface2)] = surface2.squeeze()
    surface_temp_2[len(surface2):] = np.array([0] * (4096 - len(surface2))).squeeze()
    return surface_temp_2


def centeroid_(arr):
    """get the centroid of a pointcloud"""
    length = arr.shape[0]
    sum_x = np.mean(arr[:, 0])
    sum_y = np.mean(arr[:, 1])
    sum_z = np.mean(arr[:, 2])
    return sum_x, \
           sum_y, \
           sum_z


def find_nearest_vector(array, value):
    idx = np.array([np.linalg.norm(x + y + z) for (x, y, z) in np.abs(array - value)]).argmin()
    return idx


def vertebrae_surface(surface):
    L1 = np.argwhere(surface == 1).squeeze()
    L2 = np.argwhere(surface == 2).squeeze()
    L3 = np.argwhere(surface == 3).squeeze()
    L4 = np.argwhere(surface == 4).squeeze()
    L5 = np.argwhere(surface == 5).squeeze()
    return L1, L2, L3, L4, L5


def augment_data(flow, pos1, pos2):
    to_augment = np.random.random()
    angle_both = np.random.randint(0, 360)
    angle_target = np.random.randint(-20, 20)
    xyz = np.random.choice(["x", "y", "z"])
    r = Rotation.from_euler(xyz, angle_both * np.pi / 180)
    R = r.as_matrix()
    r2 = Rotation.from_euler(xyz, angle_target * np.pi / 180)
    R2 = r2.as_matrix()
    # rotation
    if to_augment > .5:
        pos1_centroid = centeroid_(pos1[:, :3])
        color1_centroid = centeroid_(pos1[:, 3:6])
        pos2_centroid = centeroid_(pos2[:, :3])
        color2_centroid = centeroid_(pos2[:, 3:6])

        pos1[:, :3] -= pos1_centroid
        pos2[:, :3] -= pos2_centroid
        pos1[:, 3:6] -= color1_centroid
        pos2[:, 3:6] -= color2_centroid

        # rotate both source and target:
        # pos1[:, :3] = np.dot(pos1[:, :3], R)
        # pos1[:, 3:6] = np.dot(pos1[:, 3:6], R)
        #
        # pos2[:, :3] = np.dot(pos2[:, :3], R)
        # pos2[:, 3:6] = np.dot(pos2[:, 3:6], R)

        # only target slightly rotated
        pos2[:, :3] = np.dot(pos2[:, :3], R2)
        pos2[:, 3:6] = np.dot(pos2[:, 3:6], R2)

        pos1[:, :3] += pos1_centroid
        pos2[:, :3] += pos2_centroid
        pos1[:, 3:6] += color1_centroid
        pos2[:, 3:6] += color2_centroid

        flow = pos2[:, :3] - pos1[:, :3]
    to_augment = np.random.random()
    # noise
    sample_x = np.random.normal(0, 0.002, pos2.shape[0])
    sample_y = np.random.normal(0, 0.002, pos2.shape[0])
    sample_z = np.random.normal(0, 0.002, pos2.shape[0])
    indexes = np.random.random(pos2.shape[0] // 3).astype(int)
    # adding noice
    if to_augment > .5:
        pos2[indexes, 0] += sample_x[indexes]
        pos2[indexes, 1] += sample_y[indexes]
        pos2[indexes, 2] += sample_z[indexes]
        pos2[indexes, 3] += sample_x[indexes]
        pos2[indexes, 4] += sample_y[indexes]
        pos2[indexes, 5] += sample_z[indexes]

        flow = pos2[:, :3] - pos1[:, :3]
    return flow


def read_numpy_file(fp):
    data = np.load(fp)
    pos1 = data["pc1"].astype('float32')
    pos2 = data["pc2"].astype('float32')
    flow = data["flow"].astype('float32')
    constraint = data["cstPts"].astype('int')
    return constraint, flow, pos1, pos2


def _get_spine_number(path: str):
    name: str = path.split('/')[-1].split('.')[0]
    num = name.split('_')[0][5:]
    return int(num)


class SceneflowDataset(Dataset):
    def __init__(self, npoints=4096, root='./spine_clouds', mode="train"):
        """
        :param npoints: number of points of input point clouds
        :param root: folder of data in .npz format
        :param train: mode can be any of the "train", "test" and "validation"
        """
        self.npoints = npoints
        self.mode = mode
        ##in case we want to test on different data
        # if self.train==False:
        #     self.root = "./spine_clouds"
        # else:
        self.root = root

        self.data_path = glob.glob(os.path.join(self.root, '*.npz'))
        train_spines = np.arange(1, 18)
        val_spines = [18, 19, 20]
        test_spines = [21, 22]
        # train
        if self.mode == "train":
            self.data_path = [path for path in self.data_path if _get_spine_number(path) in train_spines]
        # test
        elif self.mode == "test":
            self.data_path = [path for path in self.data_path if _get_spine_number(path) in test_spines]
        # validation
        elif self.mode == "validation":
            self.data_path = [path for path in self.data_path if _get_spine_number(path) in val_spines]
        else:
            raise Exception(f'dataset mode is {mode}. mode can be any of the "train", "test" and "validation"')

    def __getitem__(self, index):
        fn = self.data_path[index]
        with open(fn, 'rb') as fp:
            constraint, flow, pos1, pos2 = read_numpy_file(fp)

            # augmentation
            if self.mode == "train":
                flow = augment_data(flow, pos1, pos2)

        np.random.seed(100)

        L1, L2, L3, L4, L5, sample_idx1, sample_idx2, sample_idx3, sample_idx4, sample_idx5 = self.sample_vertebrae(
            pos1)
        # TODO: PROBLEM WITH POINT SELECTION WORK ON IT!!!!

        sample_idx_ = np.concatenate((sample_idx1, sample_idx2,
                                      sample_idx3, sample_idx4,
                                      sample_idx5), axis=0).astype(int)
        # take every 5th point so that every vertebra has equal number of points
        sample_idx_source = sample_idx_[::5]
        # make space for the constraint points
        sample_idx_source = np.delete(sample_idx_source, [10, 1200, 1201, 2000, 2001, 3000, 3001, 4000])
        # add the constraint points
        constraint_points = np.array([L1[constraint[0], ...],
                                      L2[constraint[1], ...],
                                      L2[constraint[2], ...],
                                      L3[constraint[3], ...],
                                      L3[constraint[4], ...],
                                      L4[constraint[5], ...],
                                      L4[constraint[6], ...],
                                      L5[constraint[7], ...]])

        sample_idx_source = np.concatenate((sample_idx_source, constraint_points), axis=0).astype(int)

        np.random.seed(20)
        L1, L2, L3, L4, L5, sample_idx1, sample_idx2, sample_idx3, sample_idx4, sample_idx5 = self.sample_vertebrae(
            pos2)

        sample_idx_ = np.concatenate((sample_idx1, sample_idx2, sample_idx3, sample_idx4, sample_idx5), axis=0)

        sample_idx_target = sample_idx_[::5]

        pos1_ = np.copy(pos1)[sample_idx_source, :3] #* 1e+3

        pos2_ = np.copy(pos2)[sample_idx_target, :3] #* 1e+3
        flow_ = np.copy(flow)[sample_idx_source, :] #* 1e+3

        color1 = np.copy(pos1)[sample_idx_source, 3:6] # * 1e+3
        color2 = np.copy(pos2)[sample_idx_target, 3:6] #* 1e+3

        surface1 = np.copy(pos1)[sample_idx_source, 6]
        # specific for vertebrae:
        L1, L2, L3, L4, L5 = vertebrae_surface(surface1)

        vertebrae_point_inx_src = [L1, L2, L3, L4, L5]
        surface2 = np.copy(pos2)[sample_idx_target, 6]
        L1, L2, L3, L4, L5 = vertebrae_surface(surface2)

        vertebrae_point_inx_tar = [L1, L2, L3, L4, L5]
        ########################################################################
        mask = np.ones([self.npoints])

        return pos1_, pos2_, color1, color2, flow_, mask, np.array([i for i in range(4095, 4095 - 8, -1)]), \
               vertebrae_point_inx_src, vertebrae_point_inx_tar, fn.split('/')[-1].split('.')[0]

    def sample_vertebrae(self, pos1):
        surface1 = np.copy(pos1)[:, 6]
        # specific for vertebrae: sampling 4096 points
        # and adding the 8 points for biomechanical constraint
        L1 = np.argwhere(surface1 == 1).squeeze()
        sample_idx1 = np.random.choice(L1, self.npoints, replace=False)
        L2 = np.argwhere(surface1 == 2).squeeze()
        sample_idx2 = np.random.choice(L2, self.npoints, replace=False)
        L3 = np.argwhere(surface1 == 3).squeeze()
        sample_idx3 = np.random.choice(L3, self.npoints, replace=False)
        L4 = np.argwhere(surface1 == 4).squeeze()
        sample_idx4 = np.random.choice(L4, self.npoints, replace=False)
        L5 = np.argwhere(surface1 == 5).squeeze()
        sample_idx5 = np.random.choice(L5, self.npoints, replace=False)
        return L1, L2, L3, L4, L5, sample_idx1, sample_idx2, sample_idx3, sample_idx4, sample_idx5

    def __len__(self):
        return len(self.data_path)


if __name__ == '__main__':
    # train = SceneflowDataset(1024)
    # test = SceneflowDataset(1024, 'test')
    # for data in train:
    #     print(data[0].shape)
    #     break
    # import mayavi.mlab as mlab

    dataset = SceneflowDataset(npoints=4096)
    data_loader = DataLoader(dataset, batch_size=2)
    # print(len(d))
    import time

    tic = time.time()
    for i, data in enumerate(data_loader):
        pc1, pc2, col1, col2, flow, mask, surface1, surface2 = data
        # print(surface1)
        # print(surface2)
        # position1 = np.where(surface1 == 1)[0]
        # ind1 = torch.tensor(position1).cuda()
        # position2 = np.where(surface2 == 1)[0]
        # ind2 = torch.tensor(position2).cuda()
        # print(pc1.shape)
        # pc1 = torch.tensor(pc1).cuda().transpose(2, 1).contiguous()
        # pc2 = torch.tensor(pc2).cuda().transpose(2, 1).contiguous()
        # a = torch.index_select(pc1, 2, ind1).cuda()
        # b = torch.index_select(pc2, 2, ind2).cuda()
        # print(a.shape, b.shape, flow.shape)
        print(pc1.shape, flow.shape)
        break

        # mlab.figure(bgcolor=(1, 1, 1))
        # mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], scale_factor=0.05, color=(1, 0, 0))
        # mlab.points3d(pc2[:, 0], pc2[:, 1], pc2[:, 2], scale_factor=0.05, color=(0, 1, 0))
        # input()
        #
        # mlab.figure(bgcolor=(1, 1, 1))
        # mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], scale_factor=0.05, color=(1, 0, 0))
        # mlab.points3d(pc2[:, 0], pc2[:, 1], pc2[:, 2], scale_factor=0.05, color=(0, 1, 0))
        # mlab.quiver3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], flow[:, 0], flow[:, 1], flow[:, 2], scale_factor=1,
        #               color=(0, 0, 1), line_width=0.2)
        # input()

    print(time.time() - tic)
    print(pc1.shape, type(pc1))
