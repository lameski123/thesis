import os
from PIL import Image
from shutil import copy2
import numpy as np
import cv2


def ray_cast_image(image):
    rays = np.zeros_like(np.squeeze(np.squeeze(image)))
    j_range = range(image.shape[0])

    for i in range(image.shape[1]):
        for j in j_range:
            if image[j, i] != 12:
                rays[j, i] = 1
                break

    return rays


def generate_raycasted_db(db_path, output_dir):
    for split in ["train", "val", "test"]:

        save_dir = os.path.join(output_dir, split)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        split_path = os.path.join(db_path, split)

        us_images = [item for item in os.listdir(split_path) if "label" not in item]
        us_labels = [item for item in os.listdir(split_path) if "label" in item]

        for (image_name, label_name) in zip(us_images, us_labels):

            label = np.array(Image.open(os.path.join(split_path, label_name)))
            ray_casted_labels = ray_cast_image(label)

            # do not save empty images for training
            if np.sum(ray_casted_labels) == 0:
                continue

            ray_casted_labels = cv2.dilate(ray_casted_labels, np.ones((10, 10)), iterations=1)

            image_save_path = os.path.join(save_dir, image_name)
            label_save_path = os.path.join(save_dir, label_name)

            copy2(os.path.join(split_path, image_name), image_save_path)
            Image.fromarray(ray_casted_labels).save(label_save_path)


generate_raycasted_db(db_path="E:/NAS/jane_project/segmentation_network_data/full_labels",
                      output_dir="E:/NAS/jane_project/segmentation_network_data/ray_casted_labels")