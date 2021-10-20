import os
import numpy as np
import visualization_utils as vutils


def get_color_code(color_name):
    color_code_dict = {
        "black": [0, 0, 0],
        "red": [170, 0, 0],
        "dark_green": [0, 85, 0],
        "blue": [0, 0, 127],
        "yellow": [211, 230, 38],
        "default": "1 1 0 1"
    }

    if color_name in color_code_dict.keys():
        color_norm = [str(item/255) for item in color_code_dict[color_name]]
        color_str = " ".join(color_norm) + " 1"
        return color_str

    else:
        return color_code_dict["default"]


def add_spine_vertbyvert(imf_root, point_cloud, color_array, name, save_path, data_uid):

    vert_dict = {}
    for i in range(1, 5):
        vert_dict["vert" + str(i)] = point_cloud[color_array==i]

    vertebrae_colors = ["red", "black", "yellow", "blue", "dark_green"]

    vert_keys = [item for item in vert_dict.keys()]
    for i, vert in enumerate(vert_keys):

        vert_color = get_color_code(vertebrae_colors[i])
        filepath = os.path.join(save_path, name + "_vert" + str(i) + ".txt")
        np.savetxt(filepath, vert_dict[vert])
        imf_root = vutils.add_block_to_xml(imf_root,
                                           parent_block_name="Annotations",
                                           block_name="point_cloud_annotation",
                                           param_dict={"referenceDataUid": "data" + str(data_uid),
                                                       "name": str(name) + "_" + vert,
                                                       "color": vert_color,
                                                       "labelText":"some",
                                                       "pointSize": "4"})

        imf_root =vutils.add_block_to_xml(imf_root,
                                          parent_block_name="Algorithms",
                                          block_name="load_point_cloud",
                                          param_dict={"location": os.path.split(filepath)[-1],
                                                      "outputUids": "data" + str(data_uid)})

        data_uid += 1

    return imf_root, data_uid


def save_for_sanity_check(data_dir, file_id, save_root):
    """
    Saving the generated data in imfusion workspaces at specific location
    """

    source_pc = np.loadtxt( os.path.join(data_dir, "source_spine" + file_id + ".txt"))
    predicted_pc = np.loadtxt(os.path.join(data_dir, "predicted_spine" + file_id + ".txt"))
    gt_pc = np.loadtxt(os.path.join(data_dir, "gt_deformed_spine" + file_id + ".txt"))
    target_pc = np.loadtxt(os.path.join(data_dir, "target_spine" + file_id + ".txt"))
    color_array = source_pc[:, -1]

    source_pc = source_pc[:, 0:3]

    data_uid = 0
    imf_tree, imf_root = vutils.get_empty_imfusion_ws()
    os.makedirs(os.path.join(save_root, file_id))

    imf_root, data_uid = add_spine_vertbyvert(imf_root=imf_root,
                                              point_cloud=target_pc,
                                              color_array=np.ones((target_pc.shape[0],)),
                                              name="target_pc",
                                              save_path=os.path.join(save_root, file_id),
                                              data_uid=data_uid)

    data_uid += 1

    for pc, name in zip([source_pc, predicted_pc, gt_pc], ["source_pc", "predicted_pc", "gt_pc"]):

        imf_root, data_uid = add_spine_vertbyvert(imf_root=imf_root,
                                                  point_cloud=pc,
                                                  color_array = color_array,
                                                  name = name,
                                                  save_path=os.path.join(save_root, file_id),
                                                  data_uid = data_uid)

    vutils.write_on_file(imf_tree, os.path.join(save_root, file_id, "imf_ws.iws"))
    print("saved")


save_for_sanity_check(data_dir = "C:\\Repo\\thesis\\output\\flownet3d\\test_result",
     file_id = "8_18_0",
     save_root="temp_sanity_check")