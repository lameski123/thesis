import os

from torch import nn


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def read_batch_data(data):
    pc1, pc2, color1, color2, flow, mask1, constraint, position1, position2, file_name = data
    pc1 = pc1.cuda().transpose(2, 1).contiguous().float()
    pc2 = pc2.cuda().transpose(2, 1).contiguous().float()
    color1 = color1.cuda().transpose(2, 1).contiguous().float()
    color2 = color2.cuda().transpose(2, 1).contiguous().float()
    flow = flow.cuda().transpose(2, 1).contiguous()
    mask1 = mask1.cuda().float()
    constraint = constraint.cuda()
    return color1, color2, constraint, flow, pc1, pc2, position1, file_name


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


def create_paths(args):
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.join(args.checkpoints_dir, args.exp_name), exist_ok=True)
    os.makedirs(os.path.join(args.checkpoints_dir, args.exp_name, 'models'), exist_ok=True)


def update_args_for_cluster(args):
    try:
        from polyaxon_client.tracking import Experiment
        args.checkpoints_dir = Experiment().get_outputs_path()
        print("You are running on the cluster :)")
        print(args)
    except Exception as e:
        print(e)
        args.checkpoints_dir = 'checkpoints/' + "flownet3d/"
        print("You are Running on the local Machine")
        print(args)
    return args