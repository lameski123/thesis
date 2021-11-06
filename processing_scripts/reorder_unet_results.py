"""
This script has to be removed in the future once the data saving in unet is fixed
"""
import os
from shutil import move, rmtree


def fix_data(input_spine_folder):
    wrong_timestamp_folders = [item for item in os.listdir(input_spine_folder) if "_" not in item]

    for wrong_folder in wrong_timestamp_folders:

        if "spine" in wrong_folder:
            correct_dst_folder = wrong_folder.split("ts")[0] + "_ts" + wrong_folder.split("ts")[1]
        else:
            correct_dst_folder = wrong_folder[0:-1] + "_" + wrong_folder[-1]

        src_folder = os.path.join(input_spine_folder, wrong_folder)
        dst_folder = os.path.join(input_spine_folder, correct_dst_folder)

        print("\nsrc folder: ", src_folder)
        print("dst folder: ", dst_folder)

        for file in os.listdir(src_folder):
            move(os.path.join(src_folder, file), os.path.join(dst_folder, file))

        # Renaming all folders to be like ts+id
        rmtree(src_folder)

        if "spine" in wrong_folder:
            root_folder, folder_name = os.path.split(dst_folder)
            new_name = os.path.join(root_folder, folder_name.split("_")[1])
            print("renamed dst folder: ", new_name)
            os.rename(dst_folder, new_name)
        else:

            root_folder, folder_name = os.path.split(dst_folder)
            new_name = os.path.join(root_folder, "ts_"+folder_name)
            print("renamed dst folder: ", new_name)
            os.rename(dst_folder, new_name)

def main(dataset_path="E:/NAS/jane_project/unet_data/results_full/"):
    for spine in os.listdir(dataset_path):
        fix_data(os.path.join(dataset_path, spine))

main()