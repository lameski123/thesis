import time

from data import SceneflowDataset
import argparse
from constrained_cpd.BiomechanicalCPD import BiomechanicalCpd
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from test_utils.metrics import umeyama_absolute_orientation, pose_distance, np_chamfer_distance
import os
from sklearn.neighbors import KDTree
from sklearn.metrics import mean_squared_error
import wandb


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


def preprocess_input(source_pc, gt_flow, position1, constrain_pairs, tre_points):

    vertebra_dict = []

    for i, vertebral_level_idxes in enumerate(position1):

        # 2.a Extracting the points belonging to the first vertebra
        current_vertebra = source_pc[vertebral_level_idxes, ...]
        current_flow = gt_flow[vertebral_level_idxes, ...]

        # 2.b Getting all the springs connections starting from the current vertebra
        current_vertebra_springs = get_springs_from_vertebra(vertebral_level_idxes, constrain_pairs)

        # 2.3 Generating the pairs: (current_vertebra_idx, constraint_position) where current_vertebra_idx
        # is the spring connection in the current_vertebra object and constraint_position is the position ([x, y, z]
        # position) of the point connected to the spring
        current_vertebra_connections = [(np.argwhere(vertebral_level_idxes == item[0]), source_pc[item[1]])
                                        for item in current_vertebra_springs]

        tre_point = tre_points[tre_points[:, -1] == i+1, :]

        vertebra_dict.append({'source': current_vertebra,
                              'gt_flow': current_flow,
                              'springs': current_vertebra_connections})

    return vertebra_dict

def visualize(iteration, error, X, Y, ax):

    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', alpha=0.1)
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', alpha=0.1)
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
              fontsize='x-large')
    # ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)

def get_fig_ax():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim([-40, 40])
    ax.set_ylim([-40, 40])
    ax.set_zlim([0, 200])

    return fig, ax


def run_registration(cpd_method, with_callback=False):
    if with_callback:
        fig, ax = get_fig_ax()
        callback = partial(visualize, ax=ax)
        TY, (_, R_reg, t_reg) = cpd_method.register(callback)
        plt.close(fig)
    else:
        TY, (_, R_reg, t_reg) = cpd_method.register()

    T = np.eye(4)
    T[0:3, 0:3] = np.transpose(R_reg)
    T[0:3, -1] = t_reg

    return TY, T


def run_cpd(data_batch, save_path, cpd_iterations=5, plot_iterations=True):

    # ##############################################################################################################
    # ############################################## Getting the data ##############################################
    # ##############################################################################################################
    source_pc, target_pc, color1, color2, gt_flow, mask1, constraint, position1, position2, file_name, tre_points\
        = data_batch
    constrain_pairs = get_connected_idxes(constraint)

    # Preprocessing and saving unprocessed data
    vertebra_dict = preprocess_input(source_pc, gt_flow, position1, constrain_pairs, tre_points)

    # ##############################################################################################################
    # ################################ 1.  1st CPD iteration on the full spine #####################################
    # ##############################################################################################################

    # 1.a First iteration to alight the spines
    cpd_method = BiomechanicalCpd(target_pc=target_pc, source_pc=source_pc, max_iterations=cpd_iterations)
    source_pc_it1, predicted_T_it1 = run_registration(cpd_method, with_callback=plot_iterations)

    # ##############################################################################################################
    # ################################ 2.  2nd CPD iteration on each vertebra ######################################
    # ##############################################################################################################

    # 2.a Getting the updated data to run the constrained CPD
    updated_source = source_pc_it1
    updated_gt_flow = source_pc + gt_flow - source_pc_it1  # the flow to move the source to target after iteration 1

    # 2.b Getting the updated pre-processed input data
    vertebra_dict_it1 = preprocess_input(updated_source, updated_gt_flow, position1, constrain_pairs, tre_points)

    # 2.c Iterate over all vertebrae and apply the constrained CPD
    result_iter2 = []

    full_source = []
    original_flow = []
    deformed_source = []
    tre_list = []
    for i, vertebra in enumerate(vertebra_dict_it1):

        # 2.e Running the constrained registration for the given vertebra
        reg = BiomechanicalCpd(target_pc=target_pc, source_pc=vertebra['source'], springs=vertebra['springs'],
                               max_iterations=cpd_iterations)
        _, _ = run_registration(reg, with_callback=plot_iterations)


def main(dataset_path, save_path, cpd_iterations):

    test_set = SceneflowDataset(mode="test",
                                root=dataset_path,
                                raycasted=True,
                                augment_test=False,
                                )

    results = []
    for i, data in enumerate(test_set):

        results.append(run_cpd(data_batch=data,
                               save_path=save_path,
                               cpd_iterations=cpd_iterations,
                               plot_iterations=True))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data generation testing')
    #parser.add_argument('--dataset_path', type=str, default="./raycastedSpineClouds")
    parser.add_argument('--dataset_path', type=str, default="E:/NAS/jane_project/flownet_data/nas_data/new_data_raycasted")

    args = parser.parse_args()
    #for cpd_iterations in range(10, 100, 10):

    main(dataset_path=args.dataset_path,
         save_path="",
         cpd_iterations=5)

