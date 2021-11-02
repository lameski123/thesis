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
from itertools import chain


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
    random_p2 = random_p1
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud1[idx1, :].T, pointcloud2[idx2, :].T


def pad_data(surface2, L):
    surface_temp_2 = np.zeros((4096))
    surface_temp_2[:len(surface2)] = surface2.squeeze()
    surface_temp_2[len(surface2):] = np.array([0] * (4096 - len(surface2))).squeeze()
    return surface_temp_2


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


def get_random_rotation():
    angle_target = np.random.randint(-20, 20)

    xyz = np.random.choice(["x", "y", "z"])

    r = Rotation.from_euler(xyz, angle_target * np.pi / 180)
    return r.as_matrix()


def apply_random_rotation(pc, rotation_center=np.array([0, 0, 0]), r=None):

    r = get_random_rotation() if r is None else r
    pc = pc - rotation_center
    rotated_pc = np.dot(pc, r)

    return rotated_pc + rotation_center


def augment_test(flow, pc1, pc2, tre_points, rotation, axis):
    # ###### Generating the arrays where to store the augmented data - the fourth dimension remains constant #######
    augmented_pc1 = np.zeros(pc1.shape)
    augmented_pc2 = np.zeros(pc2.shape)

    if pc1.shape[1] == 4:
        augmented_pc1[:, -1] = pc1[:, -1]
        pc1 = pc1[:, :3]

    if pc2.shape[1] == 4:
        augmented_pc2[:, -1] = pc2[:, -1]
        pc2 = pc2[:, :3]

    angle_target = rotation
    xyz = axis

    # ###### Augmenting the data #######
    # The ground truth position of the deformed source
    gt_target = pc1 + flow

    # rotate the source
    r = Rotation.from_euler(xyz, angle_target * np.pi / 180)
    r = r.as_matrix()
    pc1 = apply_random_rotation(pc1, r=r, rotation_center=np.mean(pc1, axis=0))
    tre_points[:, 0:3] = apply_random_rotation(tre_points[:, 0:3], r=r, rotation_center=np.mean(pc1, axis=0))

    # recompute the flow with the updated pc1 and gt_target
    flow = gt_target - pc1

    augmented_pc1[:, 0:3] = pc1
    augmented_pc2[:, 0:3] = pc2

    return flow, augmented_pc1, augmented_pc2, tre_points


def augment_data(flow, pc1, pc2, tre_points, augmentation_prob=0.5):
    # ###### Generating the arrays where to store the augmented data - the fourth dimension remains constant #######
    augmented_pc1 = np.zeros(pc1.shape)
    augmented_pc2 = np.zeros(pc2.shape)

    if pc1.shape[1] == 4:
        augmented_pc1[:, -1] = pc1[:, -1]
        pc1 = pc1[:, :3]

    if pc2.shape[1] == 4:
        augmented_pc2[:, -1] = pc2[:, -1]
        pc2 = pc2[:, :3]

    # ###### Augmenting the data #######
    # The ground truth position of the deformed source
    gt_target = pc1 + flow

    # rotate the source with a probability 0.5
    if np.random.random() < augmentation_prob:
        # rotate the source about its centroid randomly and update flow accordingly

        r = get_random_rotation()
        pc1 = apply_random_rotation(pc1, r=r, rotation_center=np.mean(pc1, axis=0))
        tre_points[:, 0:3] = apply_random_rotation(tre_points[:, 0:3], r=r, rotation_center=np.mean(pc1, axis=0))

    # rotate the target with a probability 0.5
    if np.random.random() < augmentation_prob:
        # apply the same rotation to ground truth target and pc2 (a rotation about the target centroid)
        # and update flow accordingly
        r = get_random_rotation()
        pc2 = apply_random_rotation(pc2, r=r, rotation_center=np.mean(pc2, axis=0))
        gt_target = apply_random_rotation(gt_target, r=r, rotation_center=np.mean(pc2, axis=0))

    # recompute the flow with the updated pc1 and gt_target
    flow = gt_target - pc1

    # add noise to the target points with a probability of 0.5
    if np.random.random() < augmentation_prob:
        pc2 = pc2 + np.random.normal(0, 2, pc2.shape)

    augmented_pc1[:, 0:3] = pc1
    augmented_pc2[:, 0:3] = pc2

    return flow, augmented_pc1, augmented_pc2, tre_points


def read_numpy_file(fp):
    data = np.load(fp)
    pos1 = data["pc1"].astype('float32')
    pos2 = data["pc2"].astype('float32')
    flow = data["flow"].astype('float32')
    constraint = data["ctsPts"].astype('int')
    return constraint, flow, pos1, pos2


def _get_spine_number(path: str):
    name = os.path.split(path)[-1]
    name = name.split("ts")[0]
    name = name.replace("raycasted", "")
    name = name.replace("full", "")
    name = name.replace("spine", "")
    name = name.replace("_", "")
    # if "raycasted" in name:
    #     num = name.split('_')[1][5:]
    # else:
    #     num = name.split('_')[0][5:]
    # # print(path, "  ", name, " ", num)
    try:
        return int(name)
    except:
        return -1


class SceneflowDataset(Dataset):
    def __init__(self, npoints=4096, root='/mnt/polyaxon/data1/Spine_Flownet/raycastedSpineClouds/', mode="train",
                 raycasted = False, augment=True, data_seed=0, **kwargs):
        """
        :param npoints: number of points of input point clouds
        :param root: folder of data in .npz format
        :param mode: mode can be any of the "train", "test" and "validation"
        :param raycasted: the data used is raycasted or full vertebras
        :param raycasted: the data used is raycasted or full vertebrae
        :param augment: if augment data for training
        :param data_seed: which permutation to use for slicing the dataset
        """

        if mode not in ["train", "val", "test"]:
            raise Exception(f'dataset mode is {mode}. mode can be any of the "train", "test" and "validation"')

        self.npoints = npoints
        self.mode = mode
        self.root = root
        self.raycasted = raycasted
        self.augment = augment
        self.data_path = glob.glob(os.path.join(self.root, '*.npz'))
        self.use_target_normalization_as_feature = True
        self.spine_splits = {"train": np.arange(1, 20), "val": [21], "test": [22]}
        # self.spine_splits = {"train": np.arange(1, 22), "val": [21], "test": np.arange(1, 22)}
        train_idx, val_idx, test_idx = self._get_sets_indices(seed=data_seed, )
        self.spine_splits = {"train": train_idx, "val": val_idx, "test": test_idx}
        self.data_path = [path for path in self.data_path if _get_spine_number(path) in self.spine_splits[self.mode]]

        if "augment_test" in kwargs.keys():
            self.augment_test = kwargs["augment_test"]
            self.test_rotation_degree = kwargs["test_rotation_degree"]
            self.test_rotation_axis = kwargs["test_rotation_axis"]
        else:
            self.augment_test = False
            self.test_rotation_degree = None
            self.test_rotation_axis = None

        # #in case we want to test on different data
        # if self.train==False:
        #     self.root = "./spine_clouds"
        # else:

        # train

    def _get_sets_indices(self, seed: int, num_spines=22):
        assert seed >= 0 and seed < 5, 'we have only 5 different sets for indices'
        indices = np.asarray([[10, 9, 0, 8, 3, 14, 2, 5, 12, 16, 15, 6, 1, 17, 13, 18, 7, 4, 11, 19, 20, 21],
                              [10, 2, 15, 19, 14, 6, 0, 7, 5, 18, 1, 9, 11, 12, 20, 16, 4, 3, 8, 17, 13, 21],
                              [13, 19, 15, 5, 0, 12, 11, 8, 3, 10, 1, 14, 9, 6, 4, 17, 7, 18, 16, 2, 20, 21],
                              [4, 11, 6, 10, 20, 13, 0, 12, 15, 14, 16, 9, 7, 2, 17, 3, 5, 8, 18, 1, 19, 21],
                              [9, 7, 15, 20, 17, 4, 10, 3, 0, 19, 1, 11, 14, 6, 13, 18, 16, 8, 12, 2, 5, 21]])
        indices += 1

        # np.random.seed(seed)
        # indices = np.random.permutation(num_spines)
        return indices[seed, :-3], indices[seed, -3:-1], indices[seed, -1:]

    def get_tre_points(self, filename):
        """
        Loading the points position for TRE error computation in testing. They are saved in the same folder as the
        data as spine_id + "_facet_targets.txt".
        :param filename: The input filename
        """
        # Example: filename = some_fold/raycasted_spine22_ts_7_0.npz

        # --> filename = raycasted_spine22_ts_7_0.npz
        filename = os.path.split(filename)[-1]

        # --> spine_id = spine22
        spine_id = [item for item in filename.split("_") if "spine" in item][0]

        # todo: remove this in future, only for wrongly named data
        spine_id = spine_id.replace("raycasted", "")
        spine_id = spine_id.replace("ts", "")

        # --> target_points_filepath = self.root/spine22_facet_targets.txt
        target_points_filepath = os.path.join(self.root, spine_id + "_facet_targets.txt")

        return np.loadtxt(target_points_filepath)

    def get_downsampled_idx(self, pc, random_seed, constraints=None, sample_each_vertebra=True):

        """
        :param pc: [Nx4] input point cloud, where:
            pc[i, 0:3] = (x, y, z) positions of the i-th point of the source point cloud
            pc[i, 4] = integer indicating the vertebral level the point i-th of the input point cloud belongs to

        :param random_seed: The random seed to search the random sample of points

        :param constraints: list of constraints idxs. Currently it is like:
            [L1.1, L2.1, L2.2, L3.1, L3.2, L4.1, L4.2, L5.1] where Lx.i is the i-th constraint point, lying on
            vertebra x

        :param sample_each_vertebra: A boolean that indicates if the sampling must be done separately for each vertebra,
            assuming that vertebral levels are indicated on the 4th column of the input point cloud (pc).
            If set to True, the script samples self.npoints/5 points from each vertebra

        :return The indexes of the input pc to be used to downsample the point cloud.
        """

        if constraints is not None and not sample_each_vertebra:
            raise NotImplementedError("Constraints are not supported if sample_each_vertebra is False")

        # 1. Down-sample the point cloud
        np.random.seed(random_seed)

        if sample_each_vertebra:

            # 1.a) L1, L2, L3, L4, L5 = indexes of vertebra 1, 2, 3, 4, 5
            # sample_idx1, sample_idx2, sample_idx3, sample_idx4, sample_idx5 = down-sampled indexes of vertebra
            L1, L2, L3, L4, L5, sample_idx1, sample_idx2, sample_idx3, sample_idx4, sample_idx5 = self.sample_vertebrae(
                pc)

            # 1.b) Concatenating the all the points together
            sample_idx_ = np.concatenate((sample_idx1, sample_idx2,
                                          sample_idx3, sample_idx4,
                                          sample_idx5), axis=0).astype(int)

        else:
            sample_idx_ = np.random.choice(pc, self.npoints, replace=False)
            return sample_idx_

        if constraints is None:
            return sample_idx_

        # 2. If constraints are also passed, then make space for the constraint points in the sample_idx_
        # points which will be deleted from the source point indexes to make space for the constraints

        # 2.a) Removing K points from the point cloud, with K = N constraints
        pc_lengths = [item.size for item in [sample_idx1, sample_idx2, sample_idx3, sample_idx4, sample_idx5]]
        pc_idx_centers = [np.sum(pc_lengths[0:i + 1]) - pc_lengths[i] // 2 for i in range(5)]
        constraints_per_vertebra = [len(np.where(pc[constraints, -1] == i)[0]) for i in range(1, 6)]
        delete_list = list(chain(*[range(center, center + item)
                                   for (center, item) in zip(pc_idx_centers, constraints_per_vertebra)]))
        sample_idx_ = np.delete(sample_idx_, delete_list)

        # 2.b) Adding the constraints points indexes in the end.
        sample_idx_ = np.concatenate((sample_idx_, constraints), axis=0).astype(int)

        # 2.c) As we have concatenated the constraints indexes at the end of the indexes array, the position of the
        # constraints in the downsampled point cloud will be at the end of it
        updated_constraints_idx = [i for i in range(len(sample_idx_) - len(constraints), len(sample_idx_))]

        return sample_idx_, updated_constraints_idx

    @staticmethod
    def get_centroid(input_pc):
        """
        :param input_pc: [Nx3] array of points
        """

        assert input_pc.shape[1] == 3
        centroid = np.mean(input_pc, axis=0)

        return centroid

    def normalize_data(self, source_pc, target_pc, tre_points = None):
        """
        The function normalizes the data according to the Fu paper:

        Given
        - vs_c = source centroid
        - vt_c = target centroid

        - vs_i = i-th point in the source point cloud
        - vt_i = i-th point in the target point cloud

        vs_i_norm = [vs_i - vs_c, vs_i - vt_c, label] =
        = [vs_i.x-vs_c.x, vs_i.y-vs_c.y, vs_i.z-vs_c.z, vs_i.x-vt_c.x, vs_i.y-vt_c.y, vs_i.z-vt_c.z, vs_i_label]

        vt_i_norm = [vt_i - vs_c, vt_i - vt_c, label] =
        = [vt_i.x-vs_c.x, vt_i.y-vs_c.y, vt_i.z-vs_c.z, vt_i.x-vs_c.x, vt_i.y-vs_c.y, vt_i.z-vs_c.z, vt_i_label]

        :param source_pc: [Nx3] array containing the coordinates and vertebra level of the source point cloud.
            Specifically: source_pc[i, 0:3] = (x, y, z) positions of the i-t point of the source point cloud

        :param target_pc: [Nx3] array containing the coordinates and vertebra level of the target point cloud.
            Specifically: target_pc[i, 0:3] = (x, y, z) positions of the i-t point of the target point cloud

        :param tre_points: Additional points to be transformed (e.g. to get the TRE error). This are only normalized
            wrt the source centroid
        """

        assert source_pc.shape[1] == target_pc.shape[1] == 3, "Input point clouds must have shape Nx3"

        vs_c = self.get_centroid(source_pc)  # source centroid
        vt_c = self.get_centroid(target_pc)  # target centroid

        vs_normalized = np.concatenate((source_pc - vs_c, source_pc - vt_c), axis=1)
        vt_normalized = np.concatenate((target_pc - vs_c, target_pc - vt_c), axis=1)

        if tre_points is not None:
            tre_points[:, 0:3] = tre_points[:, 0:3] - vs_c

        return vs_normalized, vt_normalized, tre_points

    def __getitem__(self, index):

        file_id = os.path.split(self.data_path[index])[-1].split(".")[0]
        constraint, flow, source_pc, target_pc = read_numpy_file(fp=self.data_path[index])

        # Getting the indexes to down-sample the source and target point clouds and the updated constraints indexes
        sample_idx_source, downsampled_constraints_idx = \
            self.get_downsampled_idx(pc=source_pc, random_seed=100, constraints=constraint, sample_each_vertebra=True)
        sample_idx_target = self.get_downsampled_idx(pc=target_pc, random_seed=20, sample_each_vertebra=True)

        # Down-sampling the point clouds
        downsampled_source_pc = source_pc[sample_idx_source, ...]
        downsampled_target_pc = target_pc[sample_idx_target, ...]
        downsampled_flow = flow[sample_idx_source, :]

        tre_points = self.get_tre_points(self.data_path[index])

        # augmentation in train
        if self.mode == "train" and self.augment:
            downsampled_flow, downsampled_source_pc, downsampled_target_pc, tre_points = \
                augment_data(downsampled_flow, downsampled_source_pc, downsampled_target_pc, tre_points,
                             augmentation_prob=0.5)

        if self.mode == "test" and self.augment_test:
            downsampled_flow, downsampled_source_pc, downsampled_target_pc, tre_points = \
                augment_test(flow=downsampled_flow,
                             pc1=downsampled_source_pc,
                             pc2=downsampled_target_pc,
                             tre_points=tre_points,
                             rotation=self.test_rotation_degree,
                             axis=self.test_rotation_axis)

        # Normalizing the point clouds - this returns a 6D vector (compared to Fu paper we remove the 7th dimension
        # as it is meaningless in our case). The normalization is not affecting the flow, as the normalization is only
        # applying a translation
        normalized_source_pc, normalized_target_pc, tre_points = \
            self.normalize_data(source_pc=downsampled_source_pc[..., :3],
                                target_pc=downsampled_target_pc[..., :3],
                                tre_points=tre_points)

        # augmentation in train
        if self.mode == "train" and self.augment:
            downsampled_flow, downsampled_source_pc, downsampled_target_pc, tre_points = augment_data(
                downsampled_flow, downsampled_source_pc, downsampled_target_pc, tre_points, augmentation_prob=1)

        if self.use_target_normalization_as_feature:
            pc1 = normalized_source_pc[..., :3]
            pc2 = normalized_target_pc[..., :3]
            feature1 = normalized_source_pc[..., 3:]
            feature2 = normalized_target_pc[..., 3:]
        else:
            pc1 = np.copy(normalized_source_pc)
            pc2 = np.copy(normalized_target_pc)
            feature1 = np.ones((normalized_source_pc.shape[0],))
            feature2 = np.ones((normalized_source_pc.shape[0],))

        # getting the vertebrae indexes needed to compute the losses
        L1_source, L2_source, L3_source, L4_source, L5_source = vertebrae_surface(downsampled_source_pc[..., 3])
        vertebrae_point_inx_src = [L1_source, L2_source, L3_source, L4_source, L5_source]

        mask = np.ones([self.npoints])

        # # If mode is test also evaluate the tre
        # if self.mode != "test":
        #     return pc1, pc2, feature1, feature2, downsampled_flow, mask, np.array(downsampled_constraints_idx), \
        #            vertebrae_point_inx_src, [], file_id

        # # Getting the tre points for test - this are the 3d coordinates of the target points for tre computation
        # tre_points = self.get_tre_points(self.data_path[index])

        return pc1, pc2, feature1, feature2, downsampled_flow, mask, np.array(downsampled_constraints_idx), \
               vertebrae_point_inx_src, [], file_id, tre_points

    def sample_vertebrae(self, pos1):

        # dividing by the number of vertebrae
        n_points = self.npoints // 5

        surface1 = np.copy(pos1)[:, 3]
        # specific for vertebrae: sampling 4096 points
        L1 = np.argwhere(surface1 == 1).squeeze()
        sample_idx1 = np.random.choice(L1, n_points, replace=False)
        L2 = np.argwhere(surface1 == 2).squeeze()
        sample_idx2 = np.random.choice(L2, n_points, replace=False)
        L3 = np.argwhere(surface1 == 3).squeeze()
        sample_idx3 = np.random.choice(L3, n_points, replace=False)
        L4 = np.argwhere(surface1 == 4).squeeze()
        sample_idx4 = np.random.choice(L4, n_points, replace=False)
        L5 = np.argwhere(surface1 == 5).squeeze()
        sample_idx5 = np.random.choice(L5, n_points, replace=False)
        return L1, L2, L3, L4, L5, sample_idx1, sample_idx2, sample_idx3, sample_idx4, sample_idx5

    def __len__(self):
        return len(self.data_path)

# if __name__ == '__main__':
#     # train = SceneflowDataset(1024)
#     # test = SceneflowDataset(1024, 'test')
#     # for data in train:
#     #     print(data[0].shape)
#     #     break
#     # import mayavi.mlab as mlab
#
#     dataset = SceneflowDataset(npoints=4096)
#     data_loader = DataLoader(dataset, batch_size=2)
#     # print(len(d))
#     import time
#
#     tic = time.time()
#     for i, data in enumerate(data_loader):
#         pc1, pc2, col1, col2, flow, mask, surface1, surface2 = data
#         # print(surface1)
#         # print(surface2)
#         # position1 = np.where(surface1 == 1)[0]
#         # ind1 = torch.tensor(position1).cuda()
#         # position2 = np.where(surface2 == 1)[0]
#         # ind2 = torch.tensor(position2).cuda()
#         # print(pc1.shape)
#         # pc1 = torch.tensor(pc1).cuda().transpose(2, 1).contiguous()
#         # pc2 = torch.tensor(pc2).cuda().transpose(2, 1).contiguous()
#         # a = torch.index_select(pc1, 2, ind1).cuda()
#         # b = torch.index_select(pc2, 2, ind2).cuda()
#         # print(a.shape, b.shape, flow.shape)
#         print(pc1.shape, flow.shape)
#         break
#
#         # mlab.figure(bgcolor=(1, 1, 1))
#         # mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], scale_factor=0.05, color=(1, 0, 0))
#         # mlab.points3d(pc2[:, 0], pc2[:, 1], pc2[:, 2], scale_factor=0.05, color=(0, 1, 0))
#         # input()
#         #
#         # mlab.figure(bgcolor=(1, 1, 1))
#         # mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], scale_factor=0.05, color=(1, 0, 0))
#         # mlab.points3d(pc2[:, 0], pc2[:, 1], pc2[:, 2], scale_factor=0.05, color=(0, 1, 0))
#         # mlab.quiver3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], flow[:, 0], flow[:, 1], flow[:, 2], scale_factor=1,
#         #               color=(0, 0, 1), line_width=0.2)
#         # input()
#
#     print(time.time() - tic)
#     print(pc1.shape, type(pc1))
