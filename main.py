#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from data import SceneflowDataset
from model import FlowNet3D
import numpy as np
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from tqdm import tqdm
import sklearn
import chamferdist
import wandb
import itk
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized

T_co = TypeVar('T_co', covariant=True)



class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

mydict = {'val':'it works'}
nested_dict = {'val':'nested works too'}
mydict = dotdict(mydict)
mydict.val

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


def test_one_epoch(args, net, test_loader, loss_opt):
    net.eval()

    total_loss = 0
    total_epe = 0
    num_examples = 0
    chamfer = chamferdist.ChamferDistance()

    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        pc1, pc2, color1, color2, flow, mask1, constraint, position1, position2 = data

        pc1 = pc1.cuda().transpose(2, 1).contiguous().float()
        pc2 = pc2.cuda().transpose(2, 1).contiguous().float()
        color1 = color1.cuda().transpose(2, 1).contiguous().float()
        color2 = color2.cuda().transpose(2, 1).contiguous().float()
        flow = flow.cuda().contiguous()
        mask1 = mask1.cuda().float()
        constraint = constraint.cuda().long()

        batch_size = pc1.size(0)
        num_examples += batch_size
        flow_pred = net(pc1, pc2, color1, color2).permute(0,2,1)


        if loss_opt == "biomechanical":
            for idx in range(batch_size):
                source = pc1[idx, :, constraint[idx]].T
                predicted = pc1[idx, :, constraint[idx]].T + flow_pred[idx, constraint[idx], :]
                loss = F.mse_loss(flow_pred.float(), flow.float())
                for j in range(constraint.size(1) - 1):
                    loss += 1e-2*torch.abs(F.mse_loss(source[j,:], source[j + 1,:]) - \
                            F.mse_loss(predicted[j,:], predicted[j + 1,:]))

        elif loss_opt == "rigidity":
            source_dist1 = torch.Tensor().cuda()
            source_dist2 = torch.Tensor().cuda()

            predict_dist1 = torch.Tensor().cuda()
            predict_dist2 = torch.Tensor().cuda()

            for idx in range(pc1.shape[0]):
                for p1 in position1:
                    p1 = p1.type(torch.int).cuda()
                    source_dist1 = torch.cat((source_dist1, torch.index_select(pc1[idx, ...], 1, p1[idx, :])[..., None] \
                                              .expand(-1, -1, p1.size()[1]) \
                                              .reshape(3, -1).T), dim=0)

                    source_dist2 = torch.cat((source_dist2, torch.index_select(pc1[idx, ...], 1, p1[idx, :])[None, ...] \
                                              .expand(p1.size()[1], -1, -1) \
                                              .reshape(3, -1).T), dim=0)

                    predicted = pc1[idx, ...].T + flow_pred[idx, ...]
                    predict_dist1 = torch.cat((predict_dist1, torch.index_select(predicted.T, 1, p1[idx, :])[..., None] \
                                               .expand(-1, -1, p1.size()[1]) \
                                               .reshape(3, -1).T), dim=0)

                    predict_dist2 = torch.cat((predict_dist2, torch.index_select(predicted.T, 1, p1[idx, :])[None, ...] \
                                               .expand(p1.size()[1], -1, -1) \
                                               .reshape(3, -1).T), dim=0)

            loss = F.mse_loss(flow_pred.float(), flow.float()) + \
                       torch.abs(torch.sqrt(F.mse_loss(source_dist1, source_dist2)) -
                                 torch.sqrt(F.mse_loss(predict_dist1, predict_dist2))) / 5

        elif loss_opt == "chamfer":
            predicted = pc1.permute(0,2,1) + flow_pred
            loss = F.mse_loss(flow_pred.float(), flow.float()) + \
                   chamfer(predicted.type(torch.float), pc2.permute(0,2,1).type(torch.float), bidirectional=True) * 1e-7

        else:#flow loss
            loss = F.mse_loss(flow_pred.float(), flow.float())



        if i % 100 == 0:
            # print(i)
            pc1 = pc1.transpose(1, 2).detach().cpu().numpy()[0, :, :].squeeze()
            pc2 = pc2.transpose(1, 2).detach().cpu().numpy()[0, :, :].squeeze()
            flow_pred = flow_pred.detach().cpu().numpy()[0, :, :].squeeze()
            to_plot = np.zeros((pc1.shape[0] * 3, 6))
            to_plot[:pc1.shape[0], :3] = pc1[:, :3]
            to_plot[:pc1.shape[0], 3] = 255  # red
            to_plot[pc1.shape[0]:pc1.shape[0] * 2, :3] = pc1[:, :3] + flow_pred
            to_plot[pc1.shape[0]:pc1.shape[0] * 2, 4] = 255  # green
            to_plot[pc1.shape[0] * 2:, :3] = pc2[:, :3]
            to_plot[pc1.shape[0] * 2:, 5] = 255 #blue

            wandb.log({
                "validation": wandb.Object3D(
                    {
                        "type": "lidar/beta",
                        "points": to_plot

                    }
                )
            })

        total_loss += loss.item() * batch_size

    return total_loss * 1.0 / num_examples

def train_one_epoch(args, net, train_loader, opt, loss_opt):
    net.train()
    num_examples = 0
    total_loss = 0
    chamfer = chamferdist.ChamferDistance()
    for i, data in tqdm(enumerate(train_loader), total = len(train_loader)):

        pc1, pc2, color1, color2, flow, mask1,constraint, position1, position2 = data
        pc1 = pc1.cuda().transpose(2, 1).contiguous().float()
        pc2 = pc2.cuda().transpose(2, 1).contiguous().float()
        color1 = color1.cuda().transpose(2, 1).contiguous().float()
        color2 = color2.cuda().transpose(2, 1).contiguous().float()
        flow = flow.cuda().transpose(2, 1).contiguous()
        mask1 = mask1.cuda().float()
        constraint = constraint.cuda()

        batch_size = pc1.size(0)
        opt.zero_grad()
        num_examples += batch_size
        flow_pred = net(pc1, pc2, color1, color2)

        if loss_opt == "biomechanical":
            # print(pc1.size())
            # print(constraint.size())
            for idx in range(batch_size):
                source = pc1[idx, :, constraint[idx]]
                predicted = pc1[idx, :, constraint[idx]] + flow_pred[idx, :, constraint[idx]]
                loss = F.mse_loss(flow_pred.float(), flow.float())
                # print(predicted.size(), source.size())
                for j in range(0, constraint.size(1)-1,2):
                    loss += 1e-2*torch.abs(F.mse_loss(source[:,j], source[:,j+1]) -\
                            F.mse_loss(predicted[:,j], predicted[:,j+1]))

        elif loss_opt == "rigidity":
            source_dist1 = torch.Tensor().cuda()
            source_dist2 = torch.Tensor().cuda()

            predict_dist1 = torch.Tensor().cuda()
            predict_dist2 = torch.Tensor().cuda()

            for idx in range(pc1.shape[0]):
                for p1 in position1:
                    p1 = p1.type(torch.int).cuda()

                    source_dist1 = torch.cat((source_dist1, torch.index_select(pc1[idx, ...], 1, p1[idx, :])[..., None] \
                                              .expand(-1, -1, p1.size()[1]) \
                                              .reshape(3, -1).T), dim=0)

                    source_dist2 = torch.cat((source_dist2, torch.index_select(pc1[idx, ...], 1, p1[idx, :])[None, ...] \
                                              .expand(p1.size()[1], -1, -1) \
                                              .reshape(3, -1).T), dim=0)

                    predict_dist1 = torch.cat((predict_dist1, torch.index_select(pc1[idx, ...] +
                                                                                 flow_pred[idx, ...], 1, p1[idx, :])[
                        ..., None] \
                                               .expand(-1, -1, p1.size()[1]) \
                                               .reshape(3, -1).T), dim=0)

                    predict_dist2 = torch.cat((predict_dist2, torch.index_select(pc1[idx, ...] +
                                                                                 flow_pred[idx, ...], 1, p1[idx, :])[
                        None, ...] \
                                               .expand(p1.size()[1], -1, -1) \
                                               .reshape(3, -1).T), dim=0)
            loss = F.mse_loss(flow_pred.float(), flow.float()) + \
            torch.abs(torch.sqrt(F.mse_loss(source_dist1, source_dist2)) -
                         torch.sqrt(F.mse_loss(predict_dist1, predict_dist2)))/5

        elif loss_opt == "chamfer":
            predicted = pc1 + flow_pred
            loss = F.mse_loss(flow_pred.float(), flow.float()) + \
                chamfer(predicted.type(torch.float), pc2.type(torch.float), bidirectional=True)*1e-7
        else:#flow loss
            loss = F.mse_loss(flow_pred.float(), flow.float())

        loss.backward()

        opt.step()

        total_loss += loss.item() * batch_size
        # print(i)
        #Plot in wandb
        if i%270==0:
            pc1 = pc1.transpose(1, 2).detach().cpu().numpy()[2, :, :].squeeze()
            pc2 = pc2.transpose(1, 2).detach().cpu().numpy()[2, :, :].squeeze()
            flow_pred = flow_pred.transpose(1, 2).detach().cpu().numpy()[2, :, :].squeeze()
            to_plot = np.zeros((pc1.shape[0] * 3, 6))
            to_plot[:pc1.shape[0], :3] = pc1[:, :3]
            to_plot[:pc1.shape[0], 3] = 255  # red
            to_plot[pc1.shape[0]:pc1.shape[0] * 2, :3] = pc1[:, :3] + flow_pred
            to_plot[pc1.shape[0]:pc1.shape[0] * 2, 4] = 255  # green
            to_plot[pc1.shape[0] * 2:, :3] = pc2[:, :3]
            to_plot[pc1.shape[0] * 2:, 5] = 255 # blue
            wandb.log({
                "training": wandb.Object3D(
                    {
                        "type": "lidar/beta",
                        "points": to_plot

                    }
                )
            })

    return total_loss*1.0/num_examples



def test(args, net, test_loader, boardio, textio):

    test_loss, epe= test_one_epoch(args, net, test_loader, args.loss)

    textio.cprint('==FINAL TEST==')
    textio.cprint('mean test loss: %f\tEPE 3D: %f'%(test_loss, epe))


def train(args, net, train_loader, test_loader, boardio, textio):
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
        train_loss = train_one_epoch(args, net, train_loader, opt, args.loss)
        textio.cprint('mean train EPE loss: %f'%train_loss)

        test_loss = test_one_epoch(args, net, test_loader, args.loss)
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


if __name__ == '__main__':
    # main()
    # UNCOMMENT THIS SEGMENT IF YOU WANT TO USE TERMINAL TO RUN THE SCRIPT
    ###################################
    # parser = argparse.ArgumentParser(description='Spine Registration')
    #     parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
    #                         help='Name of the experiment')
    #     parser.add_argument('--model', type=str, default='flownet', metavar='N',
    #                         choices=['flownet'],
    #                         help='Model to use, [flownet]')
    #     parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
    #                         help='Dimension of embeddings')
    #     parser.add_argument('--num_points', type=int, default=4096,
    #                         help='Point Number [default: 4096]')
    #     parser.add_argument('--dropout', type=float, default=0.5, metavar='N',
    #                         help='Dropout ratio in transformer')
    #     parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size',
    #                         help='Size of batch)')
    #     parser.add_argument('--test_batch_size', type=int, default=4, metavar='batch_size',
    #                         help='Size of batch)')
    #     parser.add_argument('--epochs', type=int, default=100, metavar='N',
    #                         help='number of episode to train ')
    #     parser.add_argument('--use_sgd', action='store_true', default=False,
    #                         help='Use SGD')
    #     parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
    #                         help='learning rate (default: 0.001, 0.1 if using sgd)')
    #     parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
    #                         help='SGD momentum (default: 0.9)')
    #     parser.add_argument('--no_cuda', action='store_true', default=False,
    #                         help='enables CUDA training')
    #     parser.add_argument('--seed', type=int, default=100, metavar='S',
    #                         help='random seed (default: 100)')
    #     parser.add_argument('--eval', action='store_true', default=False,
    #                         help='evaluate the model')
    #     parser.add_argument('--cycle', type=bool, default=False, metavar='N',
    #                         help='Whether to use cycle consistency')
    #     parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
    #                         help='Wheter to add gaussian noise')
    #     parser.add_argument('--unseen', type=bool, default=False, metavar='N',
    #                         help='Whether to test on unseen category')
    #     parser.add_argument('--dataset', type=str, default='SceneflowDataset',
    #                         choices=['SceneflowDataset'], metavar='N',
    #                         help='dataset to use')
    #     parser.add_argument('--dataset_path', type=str, default='../../datasets/data_processed_maxcut_35_20k_2k_8192', metavar='N',
    #                         help='dataset to use')
    #     parser.add_argument('--model_path', type=str, default='', metavar='N',
    #                         help='Pretrained model path')
    #     parser.add_argument('--model_loss', type=str, default='biomechanical', metavar='N',
    #                         help='biomechanical(default), rigidity, chamfer or leave it empty("") only
    #                         for flow loss')
    #
    #     args = parser.parse_args()
    ################################
    # IF YOU WANT TO USE THE TERMINAL COMMANDS, COMMENT THE FOLLOWING BLOCK
    #################################
    args = {"exp_name":"flownet3d","emb_dims":512, "num_points":4096,
            "lr":0.001, "momentum":0.9,"seed":100, "dropout":0.5,
            "batch_size":4, "test_batch_size":4, "epochs":100,
            "use_sgd":False, "eval":False, "cycle":False,
            "gaussian_noise":False, "loss":"biomechanical"}
    args = dotdict(args)
    #################################

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    np.random.seed(100)

    # boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    boardio = []
    _init_("flownet3d")

    textio = IOStream('checkpoints/' + "flownet3d" + '/run.log')
    textio.cprint(str(args))

    dataset = SceneflowDataset(npoints=4096)

    net = FlowNet3D(args).cuda()
    net.apply(weights_init)
    # net.load_state_dict(torch.load("./checkpoints/flownet3d/models/model_spine_kaiming_no_color.best.t7"))

    wandb.init(config=args)

    train_loader = DataLoader(SceneflowDataset(npoints=4096, train=True),
        batch_size=args.batch_size, drop_last=True)
    test_loader = DataLoader(
        SceneflowDataset(npoints=4096, train=False),
        batch_size=args.batch_size,  drop_last=False)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    # wandb.watch(net, log_freq=100)
    train(args, net, train_loader, test_loader, boardio, textio)


