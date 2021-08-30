from __future__ import print_function
import os
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from data import ModelNet40, SceneflowDataset
from model import FlowNet3D
import numpy as np
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from tqdm import tqdm
import sklearn
import chamferdist
import wandb
import itk
import argparse

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



if __name__ == "__main__":
    args = {"exp_name": "flownet3d", "emb_dims": 512, "num_points": 4096,
            "lr": 0.001, "momentum": 0.9, "seed": 100, "dropout": 0.5,
            "batch_size": 1, "test_batch_size": 1, "epochs": 300,
            "use_sgd": False, "eval": True, "cycle": False,
            "gaussian_noise": False}
    args = dotdict(args)

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    boardio = []
    _init_(args)

    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))



    net = FlowNet3D(args).cuda()

    net.load_state_dict(torch.load("./checkpoints/flownet3d/models/model_spine_loss_rigid_DATA_TR_dif_rand.best.t7"))
    net.eval()
    flow_pred = []
    flows = []
    pcs = []

    # for i in range(10):
    test_loader = DataLoader(
        SceneflowDataset(npoints=args.num_points, train=False),
        batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    for i, data in enumerate(test_loader):
        # wandb.init(config=args)
        pc1, pc2, color1, color2, flow, mask1, position1, position2 = data
        pc1 = pc1.cuda().transpose(2, 1).contiguous()
        pc2 = pc2.cuda().transpose(2, 1).contiguous()
        color1 = color1.cuda().transpose(2, 1).contiguous().float()
        color2 = color2.cuda().transpose(2, 1).contiguous().float()
        flow = flow.cuda().transpose(2, 1).contiguous()
        mask1 = mask1.cuda().float()
        flow_pred = net(pc1, pc2, color1, color2)
        pc1 = pc1.transpose(1, 2).detach().cpu().numpy()[0, :, :].squeeze()
        pc2 = pc2.transpose(1, 2).detach().cpu().numpy()[0, :, :].squeeze()
        flow = flow.transpose(1, 2).detach().cpu().numpy()[0, :, :].squeeze()
        color1 = color1.transpose(1, 2).detach().cpu().numpy()[0, :, :].squeeze()
        color2 = color2.transpose(1, 2).detach().cpu().numpy()[0, :, :].squeeze()
        flow_min = flow.min()
        flow_max = flow.max()
        flow_pred = flow_pred.transpose(1, 2).detach().cpu().numpy()[0, :, :].squeeze()
        np.savetxt("test_result_spine_"+str(i)+".txt", pc1[:,:3]+flow_pred)
        np.savetxt("test_source_spine_"+str(i)+".txt", pc1[:,:3])
        np.savetxt("test_target_spine_"+str(i)+".txt", pc2[:,:3])

        # break
        # wandb.log({
        #     "source": wandb.Object3D(
        #         {
        #             "type": "lidar/beta",
        #             "points": pc1[:, :3]
        #         }
        #     )})
        # pc1[:, :3] += flow_pred
        # wandb.log({
        #     "registered": wandb.Object3D(
        #         {
        #             "type": "lidar/beta",
        #             "points": pc1[:, :3]
        #         }
        #     )})
        # wandb.log({
        #     "target": wandb.Object3D(
        #         {
        #             "type": "lidar/beta",
        #             "points": pc2[:, 3:6]
        #         }
        #     )})
        # wandb.log({
        #     "sanity target check": wandb.Object3D(
        #         {
        #             "type": "lidar/beta",
        #             "points": pc1[:, :3] + flow
        #         }
        #     )})
        # wandb.join()

