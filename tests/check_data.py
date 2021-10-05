from data import SceneflowDataset
import argparse
import numpy as np


def check_data(data_batch):
    source_pc, target_pc, _, _, gt_flow, _, constraint, position1, _ = data_batch

    np.savetxt("source_pc.txt", source_pc)
    np.savetxt("target_pc.txt", target_pc)

    print("done saving")


def main(dataset_path):
    test_set = SceneflowDataset(npoints=4096, train=True, root=dataset_path)

    for data in test_set:
        check_data(data)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data generation testing')
    parser.add_argument('--dataset_path', type=str, default="../tmpDb")

    args = parser.parse_args()

    main(args.dataset_path)
