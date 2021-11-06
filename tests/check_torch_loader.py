from data import SceneflowDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt


def read_batch_data(data):

    if len(data) == 10:
        pc1, pc2, color1, color2, flow, mask1, constraint, position1, position2, file_name = data
    else:
        pc1, pc2, color1, color2, flow, mask1, constraint, position1, position2, file_name, tre_points = data

    pc1 = pc1.transpose(2, 1).contiguous().float()
    pc2 = pc2.transpose(2, 1).contiguous().float()

    if color1.size(1) == 3:
        color1 = color1.transpose(2, 1).contiguous().float()
        color2 = color2.transpose(2, 1).contiguous().float()

    else:
        color1 = color1.contiguous().float()
        color2 = color2.contiguous().float()
    flow = flow.transpose(2, 1).contiguous()
    mask1 = mask1.float()
    constraint = constraint

    if len(data) == 10:
        return color1, color2, constraint, flow, pc1, pc2, position1, file_name

    return color1, color2, constraint, flow, pc1, pc2, position1, file_name, tre_points

def knn(ref, query, k):
    ref_c =torch.stack([ref] * query.shape[-1], dim=0).permute(0, 2, 1).reshape(-1, 3).transpose(0, 1)
    query_c = torch.repeat_interleave(query, repeats=ref.shape[-1], dim=1)
    delta = query_c - ref_c
    distances = torch.sqrt(torch.pow(delta, 2).sum(dim=0))
    distances = distances.view(query.shape[-1], ref.shape[-1])
    sorted_dist, indices = torch.sort(distances, dim=-1)
    return sorted_dist[:, :k], indices[:, :k]

def estimate_target_color(source, target, source_color):

    estimated_colors = torch.zeros([target.size(0), target.size(2)])
    for i in range(source.size(0)):
        _, k_idx = knn(source[i, ...], target[i, ...], k=1)
        estimated_colors[i, ...] = source_color[-1, k_idx[:, 0]]

    return estimated_colors

def main(dataset_path, save_dir):
    train_set = SceneflowDataset(mode="train", root=dataset_path, raycasted=True, use_target_normalization_as_feature=False)

    train_loader = DataLoader(train_set, batch_size=4, drop_last=True, num_workers=0)

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):

        color1, color2, constraint, flow, pc1, pc2, position1, fn, tre_points = read_batch_data(data)

        for refinement_iteration in range(2):

            # First iteration using only the source-normalized position and a white color as feature for both source
            # and target
            if refinement_iteration == 0:
                source_color = color1 * 0 + 1
                target_color = color2 * 0 + 1

            # In the second iteration we assign proper colors
            else:
                source_color = color1
                target_color = estimate_target_color(pc1, pc2, source_color)

            source_1 = pc1.numpy()[0, ...]
            source_1_color = source_color.numpy()[0, ...]

            target_1 = pc2.numpy()[0, ...]
            target_1_color = target_color.numpy()[0, ...]

            print()

            plt.ioff()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(source_1[0, :], source_1[1, :], source_1[2, :], c = source_1_color)
            ax.scatter(target_1[0, :], target_1[1, :], target_1[2, :], c = target_1_color)
            plt.show()
            print()


main(#dataset_path="E:/NAS/jane_project/flownet_data/nas_data/new_data_raycasted",
    dataset_path="E:/NAS/jane_project/flownet_data/nas_data/new_data_raycasted",
     save_dir="C:/Users/maria/OneDrive/Desktop/data_check")
