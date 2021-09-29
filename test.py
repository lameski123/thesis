from __future__ import print_function
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

import utils
from data import SceneflowDataset
from model import FlowNet3D


def test_one_epoch(net, test_loader, loss_opt):
    net.eval()

    total_loss = 0
    num_examples = 0

    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        color1, color2, constraint, flow, pc1, pc2, position1 = utils.read_batch_data(data)

        batch_size = pc1.size(0)
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

        if i % 100 == 0:
            utils.plot_pointcloud(flow_pred, pc1, pc2)

        total_loss += loss.item() * batch_size

    return total_loss * 1.0 / num_examples


def test(args, net, test_loader, textio):

    test_loss, epe = test_one_epoch(net, test_loader, args.loss)

    textio.cprint('==FINAL TEST==')
    textio.cprint('mean test loss: %f\tEPE 3D: %f'%(test_loss, epe))


if __name__ == "__main__":
    parser = utils.create_parser()
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    boardio = []
    utils.create_paths(args)

    textio = utils.IOStream('checkpoints/' + args.exp_name + '/run.log')
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
