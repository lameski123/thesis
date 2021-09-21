import chamferdist
import os
import sys
import glob
import h5py
import numpy as np
import scipy
from sklearn.neighbors import NearestNeighbors



def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist

datapath_f = sorted(glob.glob(os.path.join("./Results", 'f_*.txt')))
datapath_r = sorted(glob.glob(os.path.join("./Results", 'r_*.txt')))
datapath_c = sorted(glob.glob(os.path.join("./Results", 'c_*.txt')))

registrations_f = datapath_f[:7]
targets_f = datapath_f[-7:]
registrations_r = datapath_r[:7]
targets_r = datapath_r[-7:]
registrations_c = datapath_r[:7]
targets_c = datapath_r[-7:]

print(registrations_f, registrations_r, registrations_c)
sum_f = 0
sum_r = 0
sum_c = 0
for i in range(8):
    x_f = np.loadtxt(datapath_f[i])
    y_f = np.loadtxt(datapath_f[-8+i])
    x_r = np.loadtxt(datapath_r[i])
    y_r = np.loadtxt(datapath_r[-8 + i])
    x_c = np.loadtxt(datapath_c[i])
    y_c = np.loadtxt(datapath_c[-8 + i])
    print("FLOW LOSS", "\t", "RIGID LOSS","\t", "CHAMFER LOSS")
    print(chamfer_distance(x_f, y_f, direction = "y_to_x"),"\t",
          chamfer_distance(x_r, y_r, direction = "y_to_x"),"\t",
          chamfer_distance(x_c, y_c, direction = "y_to_x"))
    sum_f+=chamfer_distance(x_f, y_f, direction = "y_to_x")
    sum_r+=chamfer_distance(x_r, y_r, direction = "y_to_x")
    sum_c+=chamfer_distance(x_c, y_c, direction = "y_to_x")
print("AVERAGE")
print(sum_f/8,"\t",sum_r/8, "\t", sum_c/8)
