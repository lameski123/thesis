#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

import utils
from data import SceneflowDataset
from model import FlowNet3D
from train import train


def main():
    parser = utils.create_parser()
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    np.random.seed(100)

    utils.create_paths(args)

    textio = utils.IOStream('checkpoints/' + "flownet3d" + '/run.log')
    textio.cprint(str(args))

    net = FlowNet3D(args).cuda()
    net.apply(utils.weights_init)

    wandb.init(project='spine_flownet', config=args)

    train_set = SceneflowDataset(npoints=4096, train=True, root=args.dataset_path)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=True)
    test_set = SceneflowDataset(npoints=4096, train=False, root=args.dataset_path)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,  drop_last=False)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    # wandb.watch(net, log_freq=100)
    train(args, net, train_loader, test_loader, textio)


if __name__ == '__main__':
    main()