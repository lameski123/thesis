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

def wrap_img(input_image, displacement_field, output_image):

    Dimension = 3

    VectorComponentType = itk.F
    VectorPixelType = itk.Vector[VectorComponentType, Dimension]

    DisplacementFieldType = itk.Image[VectorPixelType, Dimension]

    PixelType = itk.UC
    ImageType = itk.Image[PixelType, Dimension]

    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(input_image)

    fieldReader = itk.ImageFileReader[DisplacementFieldType].New()
    fieldReader.SetFileName(displacement_field)
    fieldReader.Update()

    deformationField = fieldReader.GetOutput()

    warpFilter = itk.WarpImageFilter[ImageType, ImageType, DisplacementFieldType].New()

    interpolator = itk.LinearInterpolateImageFunction[ImageType, itk.D].New()

    warpFilter.SetInterpolator(interpolator)

    warpFilter.SetOutputSpacing(deformationField.GetSpacing())
    warpFilter.SetOutputOrigin(deformationField.GetOrigin())
    warpFilter.SetOutputDirection(deformationField.GetDirection())

    warpFilter.SetDisplacementField(deformationField)

    warpFilter.SetInput(reader.GetOutput())

    writer = itk.ImageFileWriter[ImageType].New()
    writer.SetInput(warpFilter.GetOutput())
    writer.SetFileName(output_image)

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

    net.load_state_dict(torch.load("./checkpoints/flownet3d/models/model_cross_val.best.t7"))
    net.eval()
    flow_pred = []
    flows = []
    pcs = []

    # for i in range(10):
    test_loader = DataLoader(
        SceneflowDataset(npoints=args.num_points, train=False),
        batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    # for i, data in enumerate(test_loader):
    data = next(iter(test_loader))
    pc1, pc2, color1, color2, flow, mask1 = data
    pcs.append(pc1)
    pcs.append(pc2)
    pc1 = pc1.cuda().transpose(2, 1).contiguous()
    pc2 = pc2.cuda().transpose(2, 1).contiguous()
    color1 = color1.cuda().transpose(2, 1).contiguous().float()
    color2 = color2.cuda().transpose(2, 1).contiguous().float()
    flow = flow.cuda()
    mask1 = mask1.cuda().float()
    flow_pred.append(net(pc1, pc2, color1, color2))
    flows.append(flow)


    flow_test = torch.transpose(flow_pred[0], dim0=2, dim1=1).detach().cpu().numpy().squeeze()
    flow_gt = flows[0].detach().cpu().numpy().squeeze()

    print(flow_test.shape, flow_gt.shape, pcs[0].numpy().squeeze().shape)

    # np.savetxt("./flow_cv_test.txt", flow_test)
    # print(flows[0].shape)
    np.savetxt("./source_transformed.txt", pcs[0]+flow_gt)
    # print(flow_test.min(), flow_test.max())
    # print(flow_gt.min(), flow_gt.max())
    # print("text saved!")


    # wrap_img("./1B00EF_source.txt", "./flow_test.txt", "./1B00EF_source_registerd.txt")

