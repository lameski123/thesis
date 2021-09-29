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
    pc1, pc2, color1, color2, flow, mask1, constraint, position1, position2 = data
    pc1 = pc1.cuda().transpose(2, 1).contiguous().float()
    pc2 = pc2.cuda().transpose(2, 1).contiguous().float()
    color1 = color1.cuda().transpose(2, 1).contiguous().float()
    color2 = color2.cuda().transpose(2, 1).contiguous().float()
    flow = flow.cuda().transpose(2, 1).contiguous()
    mask1 = mask1.cuda().float()
    constraint = constraint.cuda()
    return color1, color2, constraint, flow, pc1, pc2, position1


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
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')