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
    random_p2 = random_p1 #np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud1[idx1, :].T, pointcloud2[idx2, :].T


class ModelNet40(Dataset):
    def __init__(self, num_points, num_subsampled_points = 768, partition='train', gaussian_noise=False, unseen=False, factor=4):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.factor = factor
        self.num_subsampled_points = num_subsampled_points
        if num_points != num_subsampled_points:
            self.subsampled = True
        else:
            self.subsampled = False
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        # if self.gaussian_noise:
        #     pointcloud = jitter_pointcloud(pointcloud)
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        # 生成旋转矩阵
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        # 生成平移向量
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T
        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        if self.gaussian_noise:
            pointcloud1 = jitter_pointcloud(pointcloud1)
            pointcloud2 = jitter_pointcloud(pointcloud2)

        if self.subsampled:
            pointcloud1, pointcloud2 = farthest_subsample_points(pointcloud1, pointcloud2,
                                                                 num_subsampled_points=self.num_subsampled_points)

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return self.data.shape[0]


class SceneflowDataset(Dataset):
    def __init__(self, npoints=4096, root='./point_clouds', train=True):
        #train=1 take train part
        #train=2 take test part
        #train=0 take whole dataset
        self.npoints = npoints
        self.train = train
        if self.train==False:
            self.root = "./point_clouds_test"
        else:
            self.root = root

        self.datapath = glob.glob(os.path.join(self.root, '*.npz'))
        # if self.train == True:
        #     self.datapath = self.datapath[:-8]
        # else:
        #     self.datapath = self.datapath[-8:]
        # self.cache = {}
        # self.cache_size = 30000


    def __getitem__(self, index):
        # if index in self.cache:
        #     pos1_, pos2_, flow_ = self.cache[index]
        # else:
        fn = self.datapath[index]

        with open(fn, 'rb') as fp:
            data = np.load(fp)
            pos1 = data["pc1"].astype('float32')
            pos2 = data["pc2"].astype('float32')
            flow = data["flow"].astype('float32')

        # if len(self.cache) < self.cache_size:
        #     self.cache[index] = (pos1, pos2, flow)
        # print(pos1.shape, pos2.shape, flow.shape)
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        np.random.seed(100)
        if n1 >= self.npoints:
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
        else:
            sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, self.npoints - n1, replace=True)),
                                         axis=-1)
        if n2 >= self.npoints:
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)
        else:
            sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, self.npoints - n2, replace=True)),
                                         axis=-1)

        pos1_ = np.copy(pos1)[sample_idx1, :3]
        pos2_ = np.copy(pos2)[sample_idx2, :3]
        flow_ = np.copy(flow)[sample_idx1, :]

        color1 = np.copy(pos1)[sample_idx1, 3:6]
        color2 = np.copy(pos1)[sample_idx1, 3:6]
        # color1 = np.zeros([self.npoints, 3])
        # color2 = np.zeros([self.npoints, 3])
        mask = np.ones([self.npoints])
        surface1 = np.copy(pos1)[sample_idx1, 6]
        surface_temp_1 = np.zeros_like(surface1)
        surface1 = np.argwhere(surface1==1)
        surface_temp_1[:len(surface1)] = surface1.squeeze()
        surface_temp_1[len(surface1):] = np.array([surface1[0]]*(self.npoints-surface1.shape[0])).squeeze()

        surface2 = np.copy(pos2)[sample_idx2, 6]
        surface_temp_2 = np.zeros_like(surface2)
        surface2 = np.argwhere(surface2==1)
        surface_temp_2[:len(surface2)] = surface2.squeeze()
        surface_temp_2[len(surface2):] = np.array([surface2[0]] * (self.npoints - surface2.shape[0])).squeeze()

        return pos1_, pos2_, color1, color2, flow_, mask, surface_temp_1, surface_temp_2


    def __len__(self):
        return len(self.datapath)



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
        pc1, pc2, col1, col2, flow, mask, surface1, surface2= data
        position1 = np.where(surface1 == 1)[0]
        ind1 = torch.tensor(position1).cuda()
        position2 = np.where(surface2 == 1)[0]
        ind2 = torch.tensor(position2).cuda()
        print(pc1.shape)
        pc1 = torch.tensor(pc1).cuda().transpose(2, 1).contiguous()
        pc2 = torch.tensor(pc2).cuda().transpose(2, 1).contiguous()
        a = torch.index_select(pc1, 2, ind1).cuda()
        b = torch.index_select(pc2, 2, ind2).cuda()
        print(a.shape, b.shape, flow.shape)
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
