import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='Spine Registration')
    parser.add_argument('--exp_name', type=str, default='flownet3d', metavar='N', help='Name of the experiment')
    parser.add_argument('--model', type=str, default='flownet', metavar='N', choices=['flownet'],
                        help='Model to use, [flownet]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--num_points', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='N', help='Dropout ratio in transformer')
    parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=4, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of episode to train')
    parser.add_argument('--use_sgd', action='store_true', default=False, help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=100, metavar='S', help='random seed (default: 100)')
    parser.add_argument('--eval', action='store_true', default=False, help='evaluate the model')
    parser.add_argument('--cycle', type=bool, default=False, metavar='N', help='Whether to use cycle consistency')
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N', help='Wheter to add gaussian noise')
    parser.add_argument('--dataset', type=str, default='SceneflowDataset', choices=['SceneflowDataset'], metavar='N',
                        help='dataset to use')
    parser.add_argument('--dataset_path', type=str, default='./spine_clouds', metavar='N', help='dataset to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')
    parser.add_argument('--loss', type=str, default='biomechanical', metavar='N',
                        help='biomechanical(default), rigidity, chamfer or leave it empty("") only for flow loss')
    parser.add_argument('--wandb-key', type=str, required=True, help='key to login to your wandb account')
    return parser
