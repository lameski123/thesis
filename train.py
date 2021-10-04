#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from data import SceneflowDataset
from model import FlowNet3D
from test import test_one_epoch


def train(args, net, train_loader, test_loader, textio):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    # scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)
    scheduler = StepLR(opt, 10, gamma=0.7)

    best_test_loss = np.inf
    for epoch in range(args.epochs):
        textio.cprint('==epoch: %d, learning rate: %f==' % (epoch, opt.param_groups[0]['lr']))
        train_losses = train_one_epoch(net, train_loader, opt, args.loss)
        textio.cprint('mean train EPE loss: %f' % train_losses['total_loss'])

        test_losses = test_one_epoch(net, test_loader, wandb_table=None)
        test_loss = test_losses['total_loss']
        textio.cprint('mean test loss: %f' % test_loss)
        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            textio.cprint('best test loss till now: %f' % test_loss)
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), f'{os.path.join(args.checkpoints_dir, args.exp_name)}/models/model_spine_bio.best.t7')
            else:
                torch.save(net.state_dict(), f'{os.path.join(args.checkpoints_dir, args.exp_name)}/models/model_spine_bio.best.t7')

        scheduler.step()
        wandb.log({'Train': train_losses, 'Validation': test_losses})

        args.lr = scheduler.get_last_lr()[0]


def train_one_epoch(net, train_loader, opt, loss_opt):
    net.train()
    total_loss = 0
    mse_loss_total, bio_loss_total, rig_loss_total, chamfer_loss_total = 0.0, 0.0, 0.0, 0.0
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        color1, color2, constraint, flow, pc1, pc2, position1, fn = utils.read_batch_data(data)
        batch_size = pc1.size(0)
        opt.zero_grad()
        flow_pred = net(pc1, pc2, color1, color2)
        bio_loss, chamfer_loss, loss, mse_loss, rig_loss = utils.calculate_loss(batch_size, constraint, flow, flow_pred,
                                                                                loss_opt, pc1, pc2, position1)
        mse_loss_total += mse_loss.item() / len(train_loader)
        bio_loss_total += bio_loss.item() / len(train_loader)
        rig_loss_total += rig_loss.item() / len(train_loader)
        chamfer_loss_total += chamfer_loss.item() / len(train_loader)
        total_loss += loss.item() / len(train_loader)

        loss.backward()
        opt.step()
        if i % 50 == 0:
            utils.plot_pointcloud(flow_pred, pc1, pc2)
        # for j in range(train_loader.batch_size):
        #     utils.plot_pointcloud(flow_pred[j:j+1, ...], pc1[j:j+1, ...], pc2[j:j+1, ...], fn[j:j+1])

    losses = {'total_loss': total_loss, 'mse_loss': mse_loss_total, 'biomechanical_loss': bio_loss_total,
              'rigid_loss': rig_loss_total, 'chamfer_loss': chamfer_loss_total}
    return losses


def main():
    parser = utils.create_parser()
    args = parser.parse_args()

    args = utils.update_args_for_cluster(args)

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    np.random.seed(100)

    utils.create_paths(args)

    textio = utils.IOStream(os.path.join(args.checkpoints_dir, 'run.log'))
    textio.cprint(str(args))

    net = FlowNet3D(args).cuda()
    net.apply(utils.weights_init)

    wandb.login(key=args.wandb_key)
    wandb.init(project='spine_flownet', config=args)

    train_set = SceneflowDataset(npoints=4096, train=True, root=args.dataset_path)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=True)
    test_set = SceneflowDataset(npoints=4096, train=False, root=args.dataset_path)
    test_loader = DataLoader(test_set, batch_size=1, drop_last=False)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    train(args, net, train_loader, test_loader, textio)


if __name__ == '__main__':
    main()
