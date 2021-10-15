from scipy.spatial import KDTree
import numpy as np
import math
import os
import visualization_utils as utils


class Point:
    def __init__(self, x, y, z, color):
        self.x = x
        self.y = y
        self.z = z
        self.color = color

    def _get_pt_as_array(self):
        return np.array([self.x, self.y, self.z])

    def get_closest_point_in_cloud(self, pc, filter_by_color=True):

        distances = np.array(
            [np.linalg.norm(x + y + z) for (x, y, z) in np.abs(pc[:, :3] - self._get_pt_as_array())])

        if not filter_by_color:
            idx = distances.argmin()
            return idx, pc[idx]

        if len(np.where(pc[:, 3] == self.color)) == 0:
            return None, None

        distances[pc[:, 3] != self.color] = np.max(distances) + 1

        idx = distances.argmin()

        return idx, pc[idx]


class Spring:
    def __init__(self, p1: Point, p2: Point, color1=None, color2=None):
        self.p1 = p1
        self.p2 = p2


list_files = "/Users/janelameski/Desktop/jane/sofa/SOFAZIPPED/install/bin/" + "txtFiles/"


def extract_spine_id(filename):

    filename = os.path.split(filename)[-1]

    return filename.split("_")[0]

def centeroidnp(arr):
    """get the centroid of a point cloud"""
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])
    return math.ceil(sum_x/length), \
            math.ceil(sum_y/length), \
            math.ceil(sum_z/length)

#%%


def create_7D(source_pc, source_center, target_center):
    """create a 7D pointcloud as explained in Fu et al."""
    v_s = np.zeros((len(source_pc), 7))
    for i in range(len(v_s)):
        v_ss = source_center - source_pc[i,:3]
        v_st = target_center - source_pc[i,:3]
        v_s[i,:3] = v_ss
        v_s[i,3:6] = v_st
        v_s[i,6] = source_pc[i,3]
    return v_s


def indexes2points(idxes_list, point_cloud, color=0):

    if point_cloud.shape[1] > 3:
        color = point_cloud[:, 3]
    else:
        color = np.ones([point_cloud.shape[0],])*color

    if isinstance(idxes_list, int) or isinstance(idxes_list, float):
        idxes_list = [idxes_list]

    constraints_points = []
    for item in idxes_list:
        if isinstance(item, tuple) or isinstance(item, list):
            assert all(isinstance(x, int) for x in item) or all(isinstance(x, float) for x in item)

            constraints_points.append(tuple(Point(x=point_cloud[idx, 0],
                                                  y=point_cloud[idx, 1],
                                                  z=point_cloud[idx, 2],
                                                  color=color[idx]) for idx in item))

        else:
            constraints_points.append(Point(x=point_cloud[item, 0],
                                             y=point_cloud[item, 1],
                                             z=point_cloud[item, 2],
                                             color=color[item]))

    if len(constraints_points) == 1:
        return constraints_points[0]

    return constraints_points


def points2indexes(point_list, point_cloud):

    idxes_list = []

    for item in point_list:
        if isinstance(item, tuple) or isinstance(item, list):
            assert all(isinstance(x, Point) for x in item)
            idxes_list.append(tuple(p.get_closest_point_in_cloud(point_cloud)[0] for p in item))

        else:
            idxes_list.append(item.get_closest_point_in_cloud(point_cloud[0]))

    return idxes_list


def obtain_file_id(p):
    """
    IMPORTANT: make sure that the path to files folder (p) does not contain "_" except in the name of the 
    files themselves
    """
    splits = p.split("_")
    # dealing with Source point clouds "spine1_vert10.txt"
    if len(splits) == 2:
        if splits[0].endswith("0"):
            s_id = splits[0][-7:] + splits[1][5]
        else:
            s_id = splits[0][-6:] + splits[1][5]
    # this is the case only for several target files such as spine1_vert2_1.txt
    elif len(splits) == 3 and splits[2].endswith(".txt"):
        s_id = splits[0][-6:] + "_" + splits[2][0]
        # for all other target files such as "spine1_vert5_1_0"
    else:
        if splits[0].endswith("0"):
            s_id = splits[0][-7:] + "_" + splits[2] + "_" + splits[3][0]
        else:
            s_id = splits[0][-6:] + "_" + splits[2] + "_" + splits[3][0]
    return s_id


def obtain_indices_raycasted_original_pc(spine_target, r_target):
    """
    find indices in spine_target w.r.t. r_target such that they are the closest points between the two 
    point clouds
    """
    kdtree = KDTree(spine_target[:, :3])
    dist, points = kdtree.query(r_target[:, :3], 1)

    return list(set(points))


def create_source_target_with_vertebra_label(source_pc, target_pc, vert):
    """
    source_pc: source point cloud
    target_pc: target point cloud
    vert: [1-5] for [L1-L5] vertebra respectively

    this function is to create source and target point clouds with label for each vertebra
    """

    source = np.ones((source_pc.shape[0], source_pc.shape[1] + 1))
    source[:, :3] = source_pc
    source[:, 3] = source[:, 3] * vert
    target = np.ones((target_pc.shape[0], target_pc.shape[1] + 1))
    target[:, :3] = target_pc
    target[:, 3] = target[:, 3] * vert

    return source, target


def create_source_target_flow_spine(source_pc, target_pc, vert):
    """
    source_pc: source point cloud
    target_pc: target point cloud
    vert: [1-5] for [L1-L5] vertebra respectively

    this function is to create source and target point clouds with 7D
    where the point clouds are centered.
    """

    source_pc, target_pc = create_source_target_with_vertebra_label(source_pc, target_pc, vert)

    centroid_source = centeroidnp(source_pc)
    centroid_target = centeroidnp(target_pc)

    source_7d = create_7D(source_pc, centroid_source, centroid_target)
    target_7d = create_7D(target_pc, centroid_source, centroid_target)

    flow = target_7d[:, :3] - source_7d[:, :3]

    return source_7d, target_7d, flow


def get_lumbar_vertebrae_dict(folder_path):
    """
    Given a timestamp folder, containing the 5 .txt files corresponding to the lumbar vertebrae, the function loads
    the vertebra and returns a dict containing the point clouds.
    Example, given the folder TestDataOrderingJane\txt_files\spine1\ts_0_0 containing the files (spine1_vert1_0.txt,
    spine1_vert2_0.txt, spine1_vert3_0.txt, spine1_vert4_0.txt, spine1_vert5_0.txt), the function returns a dict like
    {"vert1" : np.array(..), "vert2" : np.array(..), "vert3" : np.array(..), "vert4" : np.array(..),
    "vert5" : np.array(..)}, where the np.arrays are Nx3 arrays containing the 3D coordinates of the point clouds
    of each vertebra

    :param folder_path: str: The path to the folder containing the vertebra point clouds .txt files
    """

    vertebra_files = [item for item in os.listdir(folder_path) if "vert" in item]

    vertebrae_dict = dict()
    for vertebra in ["vert1", "vert2", "vert3", "vert4", "vert5"]:
        vert_file = [item for item in vertebra_files if vertebra in item]
        assert len(vert_file) == 1

        vertebrae_dict[vertebra] = np.loadtxt(os.path.join(folder_path, vert_file[0]))

    return vertebrae_dict


def load_biomechanical_constraints(spine_folder_path, source_vertebrae_dict):
    """
    Loads the biomechanical constraints and returns them as a list of tuples like
    [(idx_c1_1, idx_c1_2), (idx_c2_1, idx_c2_2), ..., (idx_cn_1, idx_cn_2)] where each tuple contains the index of the
    "starting" point connected to the spring and the index of the "ending" point connected to the spring:

    ci_1 _/\/\/\/\_ ci_2

    :param: spine_folder_path: str: The path containing the data for a given spine dataset
    """

    biomechanical_constraints_path = os.path.join(spine_folder_path,
                                                  extract_spine_id(spine_folder_path).replace("s", "S")
                                                  + "_biomechanical.txt")
    if not os.path.exists(biomechanical_constraints_path):
        return []

    # The biomechanical constraints are saved in an array on a single row, like:
    # idx_c1_1, idx_c1_2, idx_c2_1, idx_c2_2, ..., idx_cn_1, idx_cn_2
    biomechanical_constraints_array = np.squeeze(np.loadtxt(biomechanical_constraints_path))
    biomechanical_constraint_list = []
    for i in range(0, biomechanical_constraints_array.shape[0] - 1, 2):
        biomechanical_constraint_list.append(
            (int(biomechanical_constraints_array[i]), int(biomechanical_constraints_array[i + 1]) ))

    constraints_points = []
    dict_keys = [item for item in source_vertebrae_dict.keys()]

    for i in range(0, biomechanical_constraints_array.shape[0] - 1, 2):

        vert_name = dict_keys[int(i/2)]
        next_vert_name = dict_keys[int(i/2) + 1]
        p1 = indexes2points(int(biomechanical_constraints_array[i]),
                            point_cloud=source_vertebrae_dict[vert_name],
                            color=int(i/2) + 1)

        p2 = indexes2points(int(biomechanical_constraints_array[i+1]),
                            point_cloud=source_vertebrae_dict[next_vert_name],
                            color=int(i/2) + 2)

        constraints_points.append((p1, p2))

    return constraints_points


def preprocess_spine_data(spine_path):
    """
    Preprocess the data for a given spine dataset. Specifically, .. explain
    """
    spine_id = os.path.split(spine_path)[-1]

    # Get the folder containing the data relative to the un-deformed spine (source) and the list of folders
    # containing the deformed spine
    source_timestamp = "ts0"
    deformed_timestamps = [item for item in os.listdir(spine_path) if item != source_timestamp and "ts" in item]

    # Getting the source vertebrae dict, as {"vert1" : np.array(..), "vert2" : np.array(..),
    # "vert3" : np.array(..), "vert4" : np.array(..), "vert5" : np.array(..)}
    source_vertebrae = get_lumbar_vertebrae_dict(os.path.join(spine_path, source_timestamp))

    # Load the biomechanical constraints for the selected spine and get the Points list
    biomechanical_constraints = load_biomechanical_constraints(spine_path, source_vertebrae)

    # Iterate over all the deformed versions (folders) of the source spine and generate the data list
    data = []
    # todo: change this back!!
    for deformed_timestamp in deformed_timestamps[0:1]:
        deformed_vertebrae = get_lumbar_vertebrae_dict(os.path.join(spine_path, deformed_timestamp))

        preprocessed_source_vertebrae = []
        preprocessed_target_vertebrae = []

        # Preprocess the point clouds of each given vertebra and then concatenate the vertebrae in a single point cloud
        for i, vertebra in enumerate(["vert1", "vert2", "vert3", "vert4", "vert5"]):
            preprocessed_source_pc, preprocessed_target_pc = \
                create_source_target_with_vertebra_label(source_pc=source_vertebrae[vertebra],
                                                         target_pc=deformed_vertebrae[vertebra],
                                                         vert=i + 1)
            preprocessed_source_vertebrae.append(preprocessed_source_pc)
            preprocessed_target_vertebrae.append(preprocessed_target_pc)

        # Concatenating source and target vertebrae into a single spine point cloud
        preprocessed_source_spine = np.concatenate(preprocessed_source_vertebrae)
        preprocessed_target_spine = np.concatenate(preprocessed_target_vertebrae)

        # Append the generated source-target pair to the data list
        data.append({
            "spine_id": spine_id,
            "source_ts_id": source_timestamp,
            "target_ts_id": deformed_timestamp,
            "source_pc": preprocessed_source_spine,
            "target_pc": preprocessed_target_spine,
            "flow": preprocessed_target_spine[:, :3] - preprocessed_source_spine[:, :3],
            "biomechanical_constraint": biomechanical_constraints
        })

    return data


def get_ray_casted_data(data, raycasted_txt_path):
    # Loading the raycasted point clouds
    source_ray_casted_pc = np.loadtxt(os.path.join(raycasted_txt_path, data["spine_id"], data["source_ts_id"] + ".txt"))
    target_ray_casted_pc = np.loadtxt(os.path.join(raycasted_txt_path, data["spine_id"], data["target_ts_id"] + ".txt"))

    # Getting the biomechanical constraints (that will be used later)
    constraint_points, constraint_flows = [], []
    constraint_indexes = [ (c[0].get_closest_point_in_cloud(data["source_pc"])[0],
                            c[1].get_closest_point_in_cloud(data["source_pc"])[0]) for c in data["biomechanical_constraint"]]
    for (p1_idx, p2_idx) in constraint_indexes:
        p1_colored, p2_colored = data["source_pc"][p1_idx, :], data["source_pc"][p2_idx, :]
        p1_flow, p2_flow = data["flow"][p1_idx, :], data["flow"][p2_idx, :]

        constraint_points.append((p1_colored, p2_colored))
        constraint_flows.append((p1_flow, p2_flow))

    # Getting the indexes of the points in the source data which are closest to the ray_casted source points
    source_ray_casted_idxes = obtain_indices_raycasted_original_pc(spine_target=data["source_pc"],
                                                                   r_target=source_ray_casted_pc)
    data["source_pc"] = data["source_pc"][source_ray_casted_idxes]
    data["flow"] = data["flow"][source_ray_casted_idxes]

    # Getting the indexes of the points in the target data which are closest to the ray_casted target points
    target_ray_casted_idxes = obtain_indices_raycasted_original_pc(spine_target=data["target_pc"],
                                                                   r_target=target_ray_casted_pc)
    data["target_pc"] = data["target_pc"][target_ray_casted_idxes]

    # Adding the biomechanical constraints
    new_constraints_idx = []
    for (p1, p2), (flow1, flow2) in zip(constraint_points, constraint_flows):
        data["source_pc"] = np.concatenate((data["source_pc"], p1, p2), axis=1)
        data["flow"] = np.concatenate((data["flow"], flow1, flow2), axis=1)
        new_constraints_idx.append((data["source_pc"].shape[-2], data["source_pc"].shape[-1]))

    return data


def get_color_code(color_name):
    color_code_dict = {
        "dark_green" : "0 0.333 0 1",
        "yellow": "1 1 0 1",
        "default": "1 1 0 1"
    }

    if color_name in color_code_dict.keys():
        return color_code_dict[color_name]

    else:
        return color_code_dict["default"]


def save_for_sanity_check(data, save_dir):

    source_pc = data["source_pc"][:, :3]
    target_pc = data["target_pc"][:, :3]

    gt_target_pc = source_pc + data["flow"]

    save_folder_path = os.path.join(save_dir, data["spine_id"], data["target_ts_id"])
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # saving the point clouds
    # 1. Saving the full point clouds
    np.savetxt(os.path.join(save_folder_path, "full_source_pc.txt"), source_pc[:, :3])
    np.savetxt(os.path.join(save_folder_path, "full_target_pc.txt"), target_pc[:, :3])
    np.savetxt(os.path.join(save_folder_path, "full_gt_pc.txt"), gt_target_pc[:, :3])

    ps_list = [("full_source_pc", os.path.join(save_folder_path, "full_source_pc.txt"), get_color_code("dark_green")),
               ("full_target_pc", os.path.join(save_folder_path, "full_target_pc.txt"), get_color_code("yellow")),
               ("full_gt_pc", os.path.join(save_folder_path, "full_gt_pc.txt"),  get_color_code("yellow"))]

    imf_tree, imf_root = utils.get_empty_imfusion_ws()

    for i, (name, path, color) in enumerate(ps_list):

        imf_root = utils.add_block_to_xml(imf_root,
                                          parent_block_name="Annotations",
                                          block_name="point_cloud_annotation",
                                          param_dict={"referenceDataUid":"data" + str(i),
                                                      "name": str(name),
                                                      "color": str(color),
                                                      "labelText":"some",
                                                      "pointSize": "2"})

        imf_root = utils.add_block_to_xml(imf_root,
                                          parent_block_name="Algorithms",
                                          block_name="load_point_cloud",
                                          param_dict={"location": path,
                                                      "outputUids": "data" + str(i)})

    # Adding the biomechanical_constraints

    for i, (c1, c2) in enumerate(data["biomechanical_constraint"]):

        c1_idx, _ = c1.get_closest_point_in_cloud(data["source_pc"], filter_by_color=True)
        c2_idx, _ = c2.get_closest_point_in_cloud(data["source_pc"], filter_by_color=True)

        p1 = data["source_pc"][c1_idx, :3]
        p2 = data["source_pc"][c2_idx, :3]
        points = " ".join([str(item) for item in p1]) + " " + " ".join([str(item) for item in p2])
        imf_root = utils.add_block_to_xml(imf_root,
                                          parent_block_name="Annotations",
                                          block_name="segment_annotation",
                                          param_dict={"name": "constraint_" + str(i+1),
                                                      "points": points})

    utils.write_on_file(imf_tree, os.path.join(save_folder_path, "imf_ws.iws"))


def generate_npz_files(src_txt_pc_path, dst_npz_path, src_raycasted_pc_path="", ray_casted=False,
                       dst_sanity_check_data=""):

    if not os.path.exists(dst_npz_path):
        os.makedirs(dst_npz_path)

    # Iterate over all the patients (spine_id) in the dataset
    for spine_id in os.listdir(src_txt_pc_path):

        # Getting the dataset for the specific patient id (spine). It is a list of dict like:
        # [{"source_ts_id": ts0,
        #   "target_ts_id": ts_19_0,
        #   "source_pc": np.ndarray([])
        #   "target_pc": np.ndarray([])
        #   "biomechanical_constraint": np.ndarray([])}, ...]
        spine_data = preprocess_spine_data(os.path.join(src_txt_pc_path, spine_id))

        for data in spine_data:
            if ray_casted:
                data = get_ray_casted_data(data, src_raycasted_pc_path)

            save_for_sanity_check(data, dst_sanity_check_data)

            # convert biomechanical_constraint to a 1-d array, putting all the constraint on a single row - this needs
            # to be changed in future to be a list of tuple or similar format where it is clear which point belongs to 
            # the same connecting spring

            constraint_indexes = points2indexes(data["biomechanical_constraint"], data["source_pc"])
            flattened_constraints = [i for sub in constraint_indexes for i in sub]
            np.savez_compressed(file=os.path.join(dst_npz_path,
                                                  "raycasted" + spine_id + data["target_ts_id"] + ".npz"),
                                flow=data["flow"],
                                pc1=data["source_pc"],
                                pc2=data["target_pc"],
                                ctsPts=flattened_constraints)


generate_npz_files(src_txt_pc_path="C:\\Users\\maria\\OneDrive\\Desktop\\TestDataOrderingJane\\txt_files",
                   dst_npz_path="C:\\Users\\maria\\OneDrive\\Desktop\\TestDataOrderingJane\\npz_data",
                   ray_casted=False,
                   dst_sanity_check_data="C:\\Users\\maria\\OneDrive\\Desktop\\TestDataOrderingJane\\sanity_check")