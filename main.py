#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

import os

import chamferdist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from tqdm import tqdm

import utils
from data import SceneflowDataset
from model import FlowNet3D


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(exp_name):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + exp_name):
        os.makedirs('checkpoints/' + exp_name)
    if not os.path.exists('checkpoints/' + exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + exp_name + '/' + 'model.py.backup')
    os.system('cp data.py checkpoints' + '/' + exp_name + '/' + 'data.py.backup')


def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        # torch.nn.init.constant(m.weight.data, 1/1000)
        # nn.init.xavier_normal(m.weight.data)
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        # torch.nn.init.constant(m.weight.data, 1/1000)
        # nn.init.xavier_normal(m.weight.data)


def scene_flow_EPE_np(pred, labels, mask):
    error = np.sqrt(np.sum((pred - labels)**2, 2) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels*labels, 2) + 1e-20) # B,N
    acc1 = np.sum(np.logical_or((error <= 0.05)*mask, (error/gtflow_len <= 0.05)*mask), axis=1)
    acc2 = np.sum(np.logical_or((error <= 0.1)*mask, (error/gtflow_len <= 0.1)*mask), axis=1)

    mask_sum = np.sum(mask, 1)
    acc1 = acc1[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc1 = np.mean(acc1)
    acc2 = acc2[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc2 = np.mean(acc2)

    EPE = np.sum(error * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    EPE = np.mean(EPE)
    return EPE, acc1, acc2


def test_one_epoch(net, test_loader, loss_opt):
    net.eval()

    total_loss = 0
    total_epe = 0
    num_examples = 0
    chamfer = chamferdist.ChamferDistance()

    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        color1, color2, constraint, flow, pc1, pc2, position1 = read_batch_data(data)

        batch_size = pc1.size(0)
        num_examples += batch_size
        flow_pred = net(pc1, pc2, color1, color2)
        loss = F.mse_loss(flow_pred.float(), flow.float())
        if loss_opt == "biomechanical":
            for idx in range(batch_size):
                loss += biomechanical_loss(constraint, flow, flow_pred, idx, pc1)
        elif loss_opt == "rigidity":
            loss += rigidity_loss(flow, flow_pred, pc1, position1)
        elif loss_opt == "chamfer":
            loss += chamfer_loss(chamfer, flow, flow_pred, pc1, pc2)

        if i % 100 == 0:
            plot_pointcloud(flow_pred, pc1, pc2)

        total_loss += loss.item() * batch_size

    return total_loss * 1.0 / num_examples


def train_one_epoch(net, train_loader, opt, loss_opt):
    net.train()
    num_examples = 0
    total_loss = 0

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        color1, color2, constraint, flow, pc1, pc2, position1 = read_batch_data(data)

        batch_size = pc1.size(0)
        opt.zero_grad()
        num_examples += batch_size
        flow_pred = net(pc1, pc2, color1, color2)
        loss = F.mse_loss(flow_pred.float(), flow.float())
        if loss_opt == "biomechanical":
            for idx in range(batch_size):
                loss += utils.biomechanical_loss(constraint, flow, flow_pred, idx, pc1)
        elif loss_opt == "rigidity":
            loss += utils.rigidity_loss(flow, flow_pred, pc1, position1)
        elif loss_opt == "chamfer":
            loss += utils.chamfer_loss(flow, flow_pred, pc1, pc2)

        loss.backward()
        opt.step()
        total_loss += loss.item() * batch_size

        if i % 270 == 0:
            plot_pointcloud(flow_pred, pc1, pc2)

    return total_loss * 1.0 / num_examples


def plot_pointcloud(flow_pred, pc1, pc2):
    pc1 = pc1.transpose(1, 2).detach().cpu().numpy()[2, :, :].squeeze()
    pc2 = pc2.transpose(1, 2).detach().cpu().numpy()[2, :, :].squeeze()
    flow_pred = flow_pred.transpose(1, 2).detach().cpu().numpy()[2, :, :].squeeze()
    to_plot = np.zeros((pc1.shape[0] * 3, 6))
    to_plot[:pc1.shape[0], :3] = pc1[:, :3]
    to_plot[:pc1.shape[0], 3] = 255  # red
    to_plot[pc1.shape[0]:pc1.shape[0] * 2, :3] = pc1[:, :3] + flow_pred
    to_plot[pc1.shape[0]:pc1.shape[0] * 2, 4] = 255  # green
    to_plot[pc1.shape[0] * 2:, :3] = pc2[:, :3]
    to_plot[pc1.shape[0] * 2:, 5] = 255  # blue
    wandb.log({
        "training": wandb.Object3D({"type": "lidar/beta", "points": to_plot})
    })


def read_batch_data(data):
    pc1, pc2, color1, color2, flow, mask1, constraint, position1, position2 = data
    pc1 = pc1.cuda().transpose(2, 1).contiguous().float()
    pc2 = pc2.cuda().transpose(2, 1).contiguous().float()
    color1 = color1.cuda().transpose(2, 1).contiguous().float()
    color2 = color2.cuda().transpose(2, 1).contiguous().float()
    flow = flow.cuda().transpose(2, 1).contiguous()
    mask1 = mask1.cuda().float()
    constraint = constraint.cuda()
    return color1, color2, constraint, flow, pc1, pc2, position1


def test(args, net, test_loader, textio):

    test_loss, epe = test_one_epoch(net, test_loader, args.loss)

    textio.cprint('==FINAL TEST==')
    textio.cprint('mean test loss: %f\tEPE 3D: %f'%(test_loss, epe))


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
        textio.cprint('==epoch: %d, learning rate: %f=='%(epoch, opt.param_groups[0]['lr']))
        train_loss = train_one_epoch(net, train_loader, opt, args.loss)
        textio.cprint('mean train EPE loss: %f'%train_loss)

        test_loss = test_one_epoch(net, test_loader, args.loss)
        textio.cprint('mean test loss: %f'%(test_loss))
        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            textio.cprint('best test loss till now: %f'%test_loss)
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model_spine_bio.best.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model_spine_bio.best.t7' % args.exp_name)

        scheduler.step()
        wandb.log({"Train loss": train_loss})
        wandb.log({"Val loss": test_loss})
        # print(scheduler.get_last_lr())
        args.lr = scheduler.get_last_lr()[0]


def main():
    parser = utils.create_parser()
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    np.random.seed(100)

    _init_("flownet3d")

    textio = IOStream('checkpoints/' + "flownet3d" + '/run.log')
    textio.cprint(str(args))

    net = FlowNet3D(args).cuda()
    net.apply(weights_init)
    # net.load_state_dict(torch.load("./checkpoints/flownet3d/models/model_spine_kaiming_no_color.best.t7"))

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