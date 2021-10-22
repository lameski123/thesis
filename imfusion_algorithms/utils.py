import itertools
import numpy as np
from sklearn.neighbors import KDTree
import os
import matplotlib.pyplot as plt
os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']

import imfusion

def make_homogeneous(pc):
    if pc.shape[0] != 3:
        pc = np.transpose(pc)

    assert pc.shape[0] == 3

    return np.concatenate((pc, np.ones((1, pc.shape[1])) ), axis = 0 )


def indexes2pyhsicalspace(indexes_list, spacing, physical_size, T_data2world):
    indexes_array = np.array(indexes_list)
    spacing = np.array(spacing)

    # Points expressed in physical coordinates wrt to top left corner of the volume
    points_apex = np.multiply(indexes_array, np.expand_dims(spacing, axis=0))

    # # Adding half voxel, assuming that we consider voxel centers - This is only needed for very large voxels,
    # # otherwise is negligible
    points_apex = np.add(points_apex, np.expand_dims(spacing / 2, axis=0))

    # Point expressed in physical coordinates wrt to the volume center
    points_vol_center = np.add(points_apex, -np.expand_dims(physical_size / 2, axis=0))

    points_vol_center = make_homogeneous(points_vol_center)
    physical_points = np.matmul(T_data2world, points_vol_center)

    return physical_points


def grid2physicalspace(grid_size, spacing, T_data2world):

    dims = len(grid_size)
    physical_size = np.array([grid_size[i] * spacing[i] for i in range(dims)])
    spacing = np.array(spacing)

    a = [ list(range(0, size)) for size in grid_size]

    # getting the indexes list with the first dimension changing slowelier and last one changing faster
    indexes_list = list(itertools.product(*a))
    physical_points = indexes2pyhsicalspace(indexes_list, spacing, physical_size, T_data2world)

    physical_points = np.reshape(physical_points, grid_size.append(3))

    return physical_points


def get_closest_points(pc1, pc2):
    """
    returns the points of pc1 which are closest to pc2
    """
    kdtree=KDTree(pc1[:,:3])
    dist, ind =kdtree.query(pc2[:,:3], 1)
    ind = ind.flatten()
    points = pc1[ind, ...]

    return ind, points


def get_grid_indexes(grid_size):
    a = [ list(range(0, size)) for size in grid_size]

    # getting the indexes list with the first dimension changing slowelier and last one changing faster
    indexes_list = list(itertools.product(*a))
    return indexes_list


def find_intersected_voxels(vol_size, image_size, T_vol2world, T_img2world, vol_spacing, img_spacing):

    if len(image_size) == 2:
        image_size.append(1)

    vol_indexes = get_grid_indexes(vol_size)
    img_indexes = get_grid_indexes(image_size)

    vol_physical_size = np.array([vol_size[i] * vol_spacing[i] for i in range(3)])
    img_physical_size = np.array([image_size[i] * img_spacing[i] for i in range(3)])

    vol_points = indexes2pyhsicalspace(vol_indexes, vol_spacing, vol_physical_size, T_vol2world)
    img_points = indexes2pyhsicalspace(img_indexes, img_spacing, img_physical_size, T_img2world)

    ind, _ = get_closest_points(np.transpose(vol_points[0:3, ...]), np.transpose(img_points[0:3, ...]))

    vol_indexes = np.array(vol_indexes)
    vol_ind = vol_indexes[ind, ...]

    return np.array(img_indexes), np.array(vol_ind)


def volume2slice(volume, vol_spacing, T_vol2world, image_size, image_spacing, T_img2world):
    vol_grid = volume.shape
    img_indexes, vol_indexes = find_intersected_voxels(vol_grid, image_size, T_vol2world, T_img2world, vol_spacing, image_spacing)

    image = np.zeros(image_size)
    image[img_indexes[:, 0], img_indexes[:, 1], img_indexes[:, 2]] = volume[vol_indexes[:, 0], vol_indexes[:, 1], vol_indexes[:, 2]]

    return image




def main():
    imfusion.init()
    vol_path = "C:/Users/maria/OneDrive/Desktop/JaneSimulatedData/labelmap.imf"
    volume, = imfusion.open(vol_path)

    vol_spacing = volume[0].spacing
    vol_array = np.squeeze(np.array(volume))
    vol_size = [item for item in vol_array.shape]
    T_vol2world = np.linalg.inv(volume[0].matrix)

    us_path = "C:/Users/maria/OneDrive/Desktop/JaneSimulatedData/Ultrasound11.imf"
    us_sweep, = imfusion.open(us_path)

    # us_sweep = np.squeeze(np.array(us_sweep))

    for i, us in enumerate(us_sweep):
        us_array = np.squeeze(np.array(us))
        us_size = [item for item in us_array.shape]
        us_spacing = us.spacing
        T_img2world = us_sweep.matrix(i)

        image = volume2slice(vol_array, vol_spacing, T_vol2world, us_size, us_spacing, T_img2world)

        plt.imshow(image)
        plt.show()



main()
# size = [10, 5, 20]
# T_data2world = np.eye(4)
# T_data2world[0:3, -1] = [20, 0, 5]
# spacing = [1, 1, 1]
#
# volume = np.zeros([10, 10, 10])
# image_size = [5, 5]
# vol_spacing = [1, 1, 1]
# image_spacing = [1, 1, 1]
# T_vol2world = np.copy(T_data2world)
# T_img2world = T_data2world
#
# volume2slice(volume, vol_spacing, T_vol2world, image_size, image_spacing, T_img2world)







