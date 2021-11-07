import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import  Image
import random
import wandb

def main(data_path):

    # wandb.login(key="b79150d26e52618b08c56a9aef76185e04aa8d6c")
    # wandb.init(project='test-plot')
    #
    # res_list = [
    #     {"id": "data1",
    #      "TRE": 3.4,
    #      "distance": 2.2},
    #     {"id": "data2",
    #      "TRE": 2,
    #      "distance": 1.2},
    #     {"id": "data3",
    #      "TRE": 5,
    #      "distance": 5},
    #     {"id": "data4",
    #      "TRE": 2,
    #      "distance": 6}
    # ]
    #
    # current_res = res_list[0]
    # for test_metric in ["TRE", "distance"]:
    #
    #     table = wandb.Table(data=[[current_res["id"], current_res[test_metric]]], columns=["data", test_metric])
    #     wandb.log({test_metric + "-all": wandb.plot.bar(table, "data", test_metric)})
    us_images = [item for item in os.listdir(data_path) if "label" not in item]

    for image_name in us_images:
        suffix = image_name.split("_")[-1]
        label_name = image_name.replace(suffix, "label_" + suffix)

        image_path = os.path.join(data_path, image_name)
        labelpath = os.path.join(data_path, label_name)

        assert os.path.exists(image_path) and os.path.exists(labelpath), "non existing path: " + labelpath

        # image = np.array(Image.open(image_path))
        # label = np.array(Image.open(labelpath))
        #
        # if random.random() > 0.1:
        #     continue
        #
        # plt.subplot(1, 2, 1)
        # plt.imshow(image, cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(label)
        # plt.show()


main(data_path="E:/NAS/jane_project/unet_data/segmentation_network_data/ray_casted_labels/all_data")
print("done")
