import time

from data import SceneflowDataset
import argparse
from constrained_cpd.BiomechanicalCPD import BiomechanicalCpd
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from test_utils.metrics import umeyama_absolute_orientation, pose_distance
import os
from sklearn.neighbors import KDTree


def get_closest_points(pc1, pc2):
    """
    returns the points of pc1 which are closest to pc2
    """
    kdtree=KDTree(pc1[:,:3])
    dist, ind =kdtree.query(pc2[:,:3], 1)
    ind = ind.flatten()
    points = pc1[ind, ...]

    return points


def visualize(iteration, error, X, Y, ax, file_id, vertebra, save_path):

    if not os.path.exists(os.path.join(save_path, str(file_id))):
        os.makedirs(os.path.join(save_path, str(file_id)))

    np.savetxt(os.path.join(save_path, str(file_id), "source_vert" + str(vertebra)) + ".txt", X)
    np.savetxt(os.path.join(save_path, str(file_id), "predicted_vert" + str(vertebra))+ ".txt", Y)

    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', alpha=0.1)
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', alpha=0.1)
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
              fontsize='x-large')
    # ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def get_connected_idxes(constraint):
    """
    From a list of connections as [idx_0, idx_1, idx_2, idx_3, ..., idx_m] it returns a list of tuples containing
    connecting indexes, assuming that the 2*i index in the list is connected with the 2*i+1 index.

    Example:
        constraint = [idx_0, idx_1, idx_2, idx_3, ..., idx_{2m}]
        returned value = [(idx_0, idx_1), (idx_2, idx_3), (idx_4, idx_5), ..., (idx_{2m-1}, idx_{2m})]
    """
    constrain_pairs = []
    for j in range(0, len(constraint) - 1, 2):
        constrain_pairs.append((constraint[j], constraint[j+1]))

    return constrain_pairs


def order_connection(item, vertebral_level_idxes):
    """
    Given  the connection item = (connection_index_1, connection_index_2), indicating a connection between two points
    (indexes) in a point cloud, the function first detects which of the connection points node belongs to the input
    vertebra (i.e. which index is contained in vertebral_level_idxes indicating the indexes of the points in the cloud
    belonging to a given vertebra). If the first point in the tuple is the one belonging to the input vertebra, the
    function returns the input item with the same order. Otherwise, it returns the input item with swapped elements,
    in a way that the first element in the return item (connection) is always the point belonging to the input
    vertebra
    """
    if item[0] in vertebral_level_idxes:
        return item

    return item[1], item[0]


def get_springs_from_vertebra(vertebral_level_idxes, constraints_pairs):
    """
    It returns the list of connection starting from the input vertebral level as a list of tuples like:
    [(idx_current_vertebra_level_0, idx_connected_vertebra_level_0),
    (idx_current_vertebra_level_1, idx_connected_vertebra_level_2),
                                ...,
    (idx_current_vertebra_level_n, idx_connected_vertebra_level_n)]
    """
    current_vertebra_springs = [item for item in constraints_pairs if item[0] in vertebral_level_idxes
                                or item[1] in vertebral_level_idxes]

    current_vertebra_springs = [order_connection(item, vertebral_level_idxes) for item in current_vertebra_springs]
    return current_vertebra_springs


def read_batch_data(data):
    pc1, pc2, color1, color2, flow, mask1, constraint, position1, position2, file_name = data
    pc1 = pc1.transpose(2, 1).contiguous().float()
    pc2 = pc2.transpose(2, 1).contiguous().float()
    color1 = color1.transpose(2, 1).contiguous().float()
    color2 = color2.transpose(2, 1).contiguous().float()
    flow = flow.transpose(2, 1).contiguous()
    mask1 = mask1.cuda().float()
    constraint = constraint.cuda()
    return color1, color2, constraint, flow, pc1, pc2, position1, file_name


def run_cpd(data_batch, save_path):
    # source_pc, target_pc, _, _, gt_flow, _, constraint, position1, _ = data_batch

    source_pc, target_pc, color1, color2, gt_flow, mask1, constraint, position1, position2, file_name = data_batch

    # pos1_, pos2_, color1, color2, flow_, mask, np.array([i for i in range(4095, 4095 - 8, -1)]), \
    # vertebrae_point_inx_src, vertebrae_point_inx_tar, fn.split('/')[-1].split('.')[0]

    constrain_pairs = get_connected_idxes(constraint)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    callback = partial(visualize, ax=ax, file_id=file_name, vertebra=0, save_path=save_path)
    ax.set_xlim([-40, 40])
    ax.set_ylim([-40, 40])
    ax.set_zlim([0, 200])

    # First iteration to alight the spines

    cpd_method = BiomechanicalCpd(target_pc=target_pc,
                                  source_pc=source_pc,
                                  max_iterations=5)

    TY, (_, R_reg, t_reg) = cpd_method.register(callback)
    plt.close(fig)

    source_pc = TY

    # 2. Generating the data for running the constrained CPD
    source_vertebrae_list = []
    gt_flow_list = []
    vertebrae_springs = []
    for vertebral_level_idxes in position1:

        # 2.a Extracting the points belonging to the first vertebra
        current_vertebra = source_pc[vertebral_level_idxes, ...]
        current_flow = gt_flow[vertebral_level_idxes, ...]
        source_vertebrae_list.append(current_vertebra)

        # 2.b Getting all the springs connections starting from the current vertebra
        current_vertebra_springs = get_springs_from_vertebra(vertebral_level_idxes, constrain_pairs)

        # 2.3 Generating the pairs: (current_vertebra_idx, constraint_position) where current_vertebra_idx
        # is the spring connection in the current_vertebra object and constraint_position is the position ([x, y, z]
        # position) of the point connected to the spring
        current_vertebra_connections = [(np.argwhere(vertebral_level_idxes == item[0]), source_pc[item[1]])
                                        for item in current_vertebra_springs]

        vertebrae_springs.append(current_vertebra_connections)
        gt_flow_list.append(current_flow)

    # 2.4 Iterate over all vertebrae and apply the constrained CPD
    for i, source_vertebra in enumerate(source_vertebrae_list):
        target_vertebra = get_closest_points(target_pc, source_vertebra)
        reg = BiomechanicalCpd(target_pc=target_vertebra,
                               source_pc=source_vertebra,
                               springs=vertebrae_springs[i])

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim([-40, 40])
        ax.set_ylim([-40, 40])
        ax.set_zlim([0, 200])

        callback = partial(visualize, ax=ax, file_id=file_name, vertebra=i, save_path=save_path)

        TY, (_, R_reg, t_reg) = reg.register(callback)
        plt.close(fig)

        R_gt, t_gt = umeyama_absolute_orientation(from_points=source_vertebra,
                                                  to_points=source_vertebra + gt_flow_list[i], fix_scaling=True)

        predicted_T = gt_T = np.eye(4)
        predicted_T[0:3, 0:3] = R_reg
        predicted_T[0:3, -1] = t_reg

        gt_T[0:3, 0:3] = R_gt
        gt_T[0:3, -1] = t_gt

        print(pose_distance(predicted_T, gt_T))


    # todo: save metrics

def main(dataset_path, save_path):
    test_set = SceneflowDataset(mode="test", root=dataset_path, raycasted=True)

    for data in test_set:

        run_cpd(data, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data generation testing')
    parser.add_argument('--dataset_path', type=str, default="./raycastedSpineClouds")
    parser.add_argument('--save_path', type=str, default="./raycastedCPDRes")

    args = parser.parse_args()

    main(args.dataset_path, args.save_path)
