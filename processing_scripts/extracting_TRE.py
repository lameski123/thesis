import xml.etree.ElementTree as ET
import os
import numpy as np
import copy
import warnings
import math

def cast_param(text):
    if text is None:
        return ""

    text = text.replace("\n", " ")
    text_list = text.split(" ")

    if len(text_list) == 1:
        try:
            casted_value = float(text.replace(" ", ""))
            return casted_value
        except ValueError:
            return text
    else:
        try:
            text_list = [item for item in text_list if item != ""]
            converted_list = [float(item.replace(" ", "")) for item in text_list]
            return np.array(converted_list)
        except ValueError:
            return text


def parse_alg_to_dict(param_dict, block):

    for sub_block in block:
        if sub_block.tag == 'param':
            param_dict[sub_block.attrib["name"]] = cast_param(sub_block.text)

        elif sub_block.tag == 'property':

            if sub_block.attrib["name"] in param_dict.keys():

                if not isinstance(param_dict[sub_block.attrib["name"]], list):
                    param_dict[sub_block.attrib["name"]] = [copy.deepcopy(param_dict[sub_block.attrib["name"]])]

                param_dict[sub_block.attrib["name"]].append(dict())
                parse_alg_to_dict(param_dict[sub_block.attrib["name"]][-1], sub_block)

            else:
                param_dict[sub_block.attrib["name"]] = dict()
                parse_alg_to_dict(param_dict[sub_block.attrib["name"]], sub_block)

    return param_dict


def get_file_id_from_filepath(ws_filepath: str):
    filepath = os.path.split(ws_filepath)[-1]
    file_id = filepath.spllit(".")[0]
    return file_id


def parse_points_list(imfusion_points_list):

    colored_points = np.zeros([len(imfusion_points_list), 4])
    for i, point in enumerate(imfusion_points_list):
        point_coordinates = point["points"]
        name = point["name"]

        print("name: ", name, " vertebra: ", name[1])
        vertebra = int(name[1])
        colored_point = np.concatenate((point_coordinates, np.array([vertebra])), axis=0)
        colored_points[i, :] = colored_point

    return colored_points


def write_poses_output(ws_filepath: str, output_filepath: str, centroid):

    tree = ET.parse(ws_filepath)
    root = tree.getroot()

    alg_dict = dict()
    alg_dict = parse_alg_to_dict(alg_dict, root)
    tre_points = parse_points_list(imfusion_points_list=alg_dict["Annotations"]["GlPoint"])

    # saving the label points as in a list
    tre_points = np.array(tre_points)
    tre_points[:, 0:3] = centroid -  tre_points[:, 0:3]

    np.savetxt(fname=output_filepath, X=tre_points)


#prepare data set
def centeroidnp(arr):
    """get the centroid of a point cloud"""
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])
    return math.ceil(sum_x/length), \
            math.ceil(sum_y/length), \
            math.ceil(sum_z/length)


def get_source_centroid(spine_path, spine_id):
    source_vertebrae = [np.loadtxt(os.path.join(spine_path,  spine_id + "_vert" + str(i) + "0.txt")) for i in range(1, 6)]
    source_pc = np.concatenate(source_vertebrae, axis=0)
    centroid = centeroidnp(source_pc)
    return centroid


def generate_tre_files(input_obj_dir, input_txt_dir, output_dir):
    spine_list = [item for item in os.listdir(input_obj_dir) if "spine" in item]

    for spine_id in spine_list:
        ts0_dir = os.path.join(input_obj_dir, spine_id, "ts0")

        input_ws_path = os.path.join(ts0_dir, "facet_annotations.iws")

        if not os.path.exists(input_ws_path):
            warnings.warn("No target face for spine id: " + spine_id)
            continue

        output_path = os.path.join(output_dir, spine_id + "_facet_targets.txt")

        source_pc_path = os.path.join(input_txt_dir, spine_id, "ts0")
        centroid = get_source_centroid(source_pc_path, spine_id)

        print(spine_id, centroid)

        write_poses_output(ws_filepath=input_ws_path,
                           output_filepath=output_path,
                           centroid=centroid
                           )


generate_tre_files(input_obj_dir = "E:/NAS/jane_project/obj_files",
                   output_dir = "E:/NAS/jane_project/tmp_db",
                   input_txt_dir="E:/NAS/jane_project/txt_files")
