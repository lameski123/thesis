from __future__ import print_function

import os

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

import utils
from data import SceneflowDataset
from model import FlowNet3D


def test_one_epoch(net, test_loader, loss_opt, save_results=False, args=None):
    net.eval()

    total_loss = 0
    num_examples = 0

    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        color1, color2, constraint, flow, pc1, pc2, position1 = utils.read_batch_data(data)

        batch_size = pc1.size(0)
        num_examples += batch_size
        flow_pred = net(pc1, pc2, color1, color2)
        loss = F.mse_loss(flow_pred.float(), flow.float())
        if "biomechanical" in loss_opt:
            for idx in range(batch_size):
                loss += utils.biomechanical_loss(constraint, flow, flow_pred, idx, pc1)[0]
        if "rigidity" in loss_opt:
            loss += utils.rigidity_loss(flow, flow_pred, pc1, position1)
        if "chamfer" in loss_opt:
            loss += utils.chamfer_loss(flow, flow_pred, pc1, pc2)

        if i % 100 == 0:
            utils.plot_pointcloud(flow_pred, pc1, pc2)

        total_loss += loss.item() * batch_size

        if save_results:
            result_path = os.path.join('checkpoints/', args.exp_name, 'test_result/')
            os.makedirs(result_path, exist_ok=True)
            n = pc1.shape[0]
            for j in range(n):
                idx = i*n + j
                np.savetxt(os.path.join(result_path, f"predicted_{idx}.txt"), (pc1[j, :, :] + flow_pred[j, :, :]).cpu())
                np.savetxt(os.path.join(result_path, f"source_{idx}.txt"), pc1[j, :, :].detach().cpu())
                np.savetxt(os.path.join(result_path, f"target_{idx}.txt"), pc2[j, :, :].detach().cpu())

    return total_loss * 1.0 / num_examples


def test(args, net, test_loader, textio):

    with torch.no_grad():
        test_loss = test_one_epoch(net, test_loader, args.loss, save_results=True, args=args)

    textio.cprint('==FINAL TEST==')
    textio.cprint(f'mean test loss: {test_loss}')


if __name__ == "__main__":

    parser = utils.create_parser()
    args = parser.parse_args()

    assert os.path.exists(args.model_path), f'model path {args.model_path} does not exist'

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    boardio = []
    utils.create_paths(args)

    wandb.login(key=args.wandb_key)
    wandb.init(project='spine_flownet', config=args)

    textio = utils.IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

    net = FlowNet3D(args).cuda()

    net.load_state_dict(torch.load(args.model_path))
    net.eval()
    flow_pred = []
    flows = []
    pcs = []

    test_set = SceneflowDataset(npoints=4096, train=False, root=args.dataset_path)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, drop_last=False)

    test(args, net, test_loader, textio)

