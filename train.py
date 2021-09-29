import numpy as np
import torch
import wandb
from torch import optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from tqdm import tqdm

import utils
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


def train_one_epoch(net, train_loader, opt, loss_opt):
    net.train()
    num_examples = 0
    total_loss = 0

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        color1, color2, constraint, flow, pc1, pc2, position1 = utils.read_batch_data(data)

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
            utils.plot_pointcloud(flow_pred, pc1, pc2)

    return total_loss * 1.0 / num_examples



