#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import copy
import os
from types import SimpleNamespace

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
from test import test_one_epoch, test, compute_test_metrics, get_color_array

args = None

def train(args, net, train_loader, val_loader, textio, two_stage_refinement=False):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    # scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)
    scheduler = StepLR(opt, 10, gamma=0.7)

    best_test_loss = np.inf
    best_net = None
    for epoch in range(args.epochs):
        textio.cprint('==epoch: %d, learning rate: %f==' % (epoch, opt.param_groups[0]['lr']))
        train_losses = train_one_epoch(net, train_loader, opt, args.loss, args, two_stage_refinement)
        textio.cprint('mean train EPE loss: %f' % train_losses['total_loss'])

        with torch.no_grad():
            test_losses = test_one_epoch(net, val_loader, args=args, wandb_table=None,
                                         two_stage_refinement=two_stage_refinement)
        test_loss = test_losses['TRE']
        textio.cprint('mean test loss: %f' % test_loss)
        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            best_net = copy.deepcopy(net)
            textio.cprint('best test loss till now: %f' % test_loss)
            if torch.cuda.device_count() > 1 and args.gpu_id == -1:
                torch.save(net.module.state_dict(), f'{args.checkpoints_dir}/models/model_spine_bio.best.t7')
            else:
                torch.save(net.state_dict(), f'{args.checkpoints_dir}/models/model_spine_bio.best.t7')

        scheduler.step()
        wandb.log({'Train': train_losses, 'Validation': test_losses, 'val_loss': test_losses[args.sweep_target_loss]})

        args.lr = scheduler.get_last_lr()[0]
    return best_net


def train_single_phase(net, loss_opt, loss_coeff, color1, color2, constraint, flow, pc1, pc2, position1):
    batch_size = pc1.size(0)
    flow_pred = net(pc1, pc2, color1, color2)
    bio_loss, chamfer_loss, loss, mse_loss, rig_loss = utils.calculate_loss(batch_size,
                                                                            constraint,
                                                                            flow,
                                                                            flow_pred,
                                                                            loss_opt,
                                                                            pc1,
                                                                            pc2,
                                                                            position1,
                                                                            loss_coeff)

    return flow_pred, loss, {'bio_loss': bio_loss,
                             'chamfer_loss': chamfer_loss,
                             'loss': loss,
                             'mse_loss': mse_loss,
                             'rig_loss': rig_loss
                             }


def train_one_epoch(net, train_loader, opt, loss_opt, args, two_stage_refinement = False):
    # two_stage_refinement = False
    net.train()
    total_loss = 0
    mse_loss_total, bio_loss_total, rig_loss_total, chamfer_loss_total, tre_total = 0.0, 0.0, 0.0, 0.0, 0.0
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):

        color1, color2, constraint, flow, pc1, pc2, position1, fn, tre_points = utils.read_batch_data(data)

        if two_stage_refinement and color1.shape[1] > 1:
            raise ValueError("Two stage refinement is only supported when 1-d colors are the input feature")

        batch_size = pc1.size(0)
        opt.zero_grad()

        # The first iteration will use the input color if only one stage is performed, otherwise it will assign the same
        # color to all points in the source and target point clouds
        flow_pred, loss, losses_dict = train_single_phase(net=net,
                                                          loss_opt=loss_opt,
                                                          loss_coeff=args.loss_coeff,
                                                          color1=color1*0+1 if two_stage_refinement else color1,
                                                          color2=color2*0+1 if two_stage_refinement else color2,
                                                          constraint=constraint,
                                                          flow=flow,
                                                          pc1=pc1,
                                                          pc2=pc2,
                                                          position1=position1)
        # Running the second stage prediction
        if two_stage_refinement:
            predicted_flow = flow_pred.clone().detach()
            updated_pc1 = pc1 + predicted_flow
            updated_gt_glow = flow - predicted_flow
            source_color = color1

            target_color = utils.estimate_target_color(updated_pc1, pc2, color1).cuda().contiguous().float()
            flow_pred_it2, loss_iter2,  losses_dict = train_single_phase(net=net,
                                                                         loss_opt=loss_opt,
                                                                         loss_coeff=args.loss_coeff,
                                                                         color1=source_color,
                                                                         color2=target_color,
                                                                         constraint=constraint,
                                                                         flow=updated_gt_glow,
                                                                         pc1=updated_pc1.clone().detach(),
                                                                         pc2=pc2.clone().detach(),
                                                                         position1=position1)

            overall_predicted_flow = flow_pred + flow_pred_it2
            loss += loss_iter2
        else:
            source_color = get_color_array(position1)
            overall_predicted_flow = flow_pred

        loss.backward()

        _, quaternion_distance, translation_distance, tre = compute_test_metrics(file_id=fn,
                                                                                 source_pc=pc1,
                                                                                 source_color = source_color if isinstance(source_color, np.ndarray) else torch.squeeze(source_color.detach(), dim=1),
                                                                                 gt_flow=flow,
                                                                                 estimated_flow=overall_predicted_flow.detach(),
                                                                                 tre_points=tre_points)

        mse_loss_total += losses_dict['mse_loss'].item() / len(train_loader)
        bio_loss_total += losses_dict['bio_loss'].item() / len(train_loader)
        rig_loss_total += losses_dict['rig_loss'].item() / len(train_loader)
        chamfer_loss_total += losses_dict['chamfer_loss'].item() / len(train_loader)
        total_loss += losses_dict['loss'].item() / len(train_loader)
        tre_total += tre / len(train_loader)

        # Backpropagate the loss


        opt.step()
        if i % 50 == 0 and args.wandb_sweep_id is None:  # plot only if not in sweep mode
            utils.plot_pointcloud(flow_pred, pc1, pc2)

        # computing refined loss

    losses = {'total_loss': total_loss, 'mse_loss': mse_loss_total, 'biomechanical_loss': bio_loss_total,
              'rigid_loss': rig_loss_total, 'chamfer_loss': chamfer_loss_total, 'TRE': tre_total}
    return losses


def run_experiment(args):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    np.random.seed(100)
    utils.create_paths(args)
    textio = utils.IOStream(os.path.join(args.checkpoints_dir, 'run.log'))
    textio.cprint(str(args))
    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)
    net = FlowNet3D(args).cuda()
    net.apply(utils.weights_init)

    #todo set as args entry
    train_set = SceneflowDataset(npoints=4096, mode="train", root=args.dataset_path,
                                 raycasted=args.use_raycasted_data, augment=not args.no_augmentation, data_seed=args.data_seed,
                                 use_target_normalization_as_feature=False)

    train_loader = DataLoader(train_set, batch_size=1, drop_last=True)
    val_set = SceneflowDataset(npoints=4096, mode="val", root=args.dataset_path,
                               raycasted=args.use_raycasted_data, data_seed=args.data_seed,
                               use_target_normalization_as_feature=False)
    val_loader = DataLoader(val_set, batch_size=1, drop_last=False)
    if torch.cuda.device_count() > 1 and args.gpu_id == -1:
        net = nn.DataParallel(net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    best_net = train(args, net, train_loader, val_loader, textio, two_stage_refinement=True)
    # test after training
    test(args, best_net, textio)


def train_wandb():
    global args
    with wandb.init(project='spine_flownet', config=args):
        config = wandb.config
        args = SimpleNamespace(**config)
        args = utils.update_args(args)
        print('-------------------config---------------------')
        print(args)

        run_experiment(args)


def main():
    global args
    parser = utils.create_parser()
    args = parser.parse_args()

    wandb.login(key=args.wandb_key)

    if args.wandb_sweep_id is not None:
        wandb.agent(args.wandb_sweep_id, train_wandb, count=args.wandb_sweep_count, project='spine_flownet')
    else:
        args = utils.update_args(args)

        wandb.init(project='spine_flownet', config=args)

        run_experiment(args)


if __name__ == '__main__':
    main()
