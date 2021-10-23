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


def test_one_epoch(net, test_loader, args, save_results=False, wandb_table: wandb.Table = None):
    net.eval()

    total_loss = 0
    mse_loss_total, bio_loss_total, rig_loss_total, chamfer_loss_total = 0.0, 0.0, 0.0, 0.0

    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        color1, color2, constraint, flow, pc1, pc2, position1, fn = utils.read_batch_data(data)

        batch_size = pc1.size(0)
        flow_pred = net(pc1, pc2, color1, color2)
        bio_loss, chamfer_loss, loss, mse_loss, rig_loss = utils.calculate_loss(batch_size, constraint, flow, flow_pred,
                                                                                ['all'], pc1, pc2, position1,
                                                                                args.loss_coeff)
        mse_loss_total += mse_loss.item() / len(test_loader)
        bio_loss_total += bio_loss.item() / len(test_loader)
        rig_loss_total += rig_loss.item() / len(test_loader)
        chamfer_loss_total += chamfer_loss.item() / len(test_loader)
        total_loss += loss.item() / len(test_loader)

        if save_results and args.wandb_sweep_id is None:
            result_path = os.path.join(args.checkpoints_dir, args.exp_name, 'test_result/')
            os.makedirs(result_path, exist_ok=True)
            for j in range(test_loader.batch_size):
                np.savetxt(os.path.join(result_path, f"predicted_{fn}.txt"), (pc1[j, :, :] + flow_pred[j, :, :]).cpu())
                np.savetxt(os.path.join(result_path, f"source_{fn}.txt"), pc1[j, :, :].detach().cpu())
                np.savetxt(os.path.join(result_path, f"target_{fn}.txt"), pc2[j, :, :].detach().cpu())
                utils.plot_pointcloud(flow_pred[j:j+1, ...], pc1[j:j+1, ...], pc2[j:j+1, ...], tag=fn[j], mode='test')


        if wandb_table is not None:
            for j in range(test_loader.batch_size):
                wandb_table.add_data(fn[j], mse_loss.item(), bio_loss.item(), chamfer_loss.item(), rig_loss.item())

    losses = {'total_loss': total_loss, 'mse_loss': mse_loss_total, 'biomechanical_loss': bio_loss_total,
              'rigid_loss': rig_loss_total, 'chamfer_loss': chamfer_loss_total}
    return losses


def main():
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

    test(args, net, textio)


def test(args, net, textio):

    test_set = SceneflowDataset(npoints=4096, mode="test", root=args.dataset_path)
    test_loader = DataLoader(test_set, batch_size=1, drop_last=False)

    test_data_at = wandb.Artifact("test_samples_" + str(wandb.run.id), type="predictions")
    columns = ['id', "mse loss", "biomechanical loss", "Chamfer loss", 'rigidity loss']
    test_table = wandb.Table(columns=columns)

    with torch.no_grad():
        test_loss = test_one_epoch(net, test_loader, args=args, save_results=True, wandb_table=test_table)

    textio.cprint('==FINAL TEST==')
    textio.cprint(f'mean test loss: {test_loss}')

    test_data_at.add(test_table, "test prediction")
    wandb.run.log_artifact(test_data_at)


if __name__ == "__main__":
    main()


