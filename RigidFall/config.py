import argparse
import numpy as np
import torch


### build arguments
parser = argparse.ArgumentParser()
parser.add_argument('--env', default='RigidFall')
parser.add_argument('--stage', default='dy', help="dy: dynamics model")
parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--random_seed', type=int, default=42)

parser.add_argument('--time_step', type=int, default=0)
parser.add_argument('--dt', type=float, default=1. / 60.)
parser.add_argument('--n_instance', type=int, default=-1)

parser.add_argument('--nf_relation', type=int, default=150)
parser.add_argument('--nf_particle', type=int, default=150)
parser.add_argument('--nf_pos', type=int, default=150)
parser.add_argument('--nf_memory', type=int, default=150)
parser.add_argument('--mem_nlayer', type=int, default=2)
parser.add_argument('--nf_effect', type=int, default=150)

parser.add_argument('--outf', default='files')
parser.add_argument('--evalf', default='eval')
parser.add_argument('--dataf', default='data')

parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--verbose_data', type=int, default=0)
parser.add_argument('--verbose_model', type=int, default=0)
parser.add_argument('--eps', type=float, default=1e-6)

# for ablation study
parser.add_argument('--neighbor_radius', type=float, default=-1)
parser.add_argument('--neighbor_k', type=float, default=-1)

# use a flexible number of frames for each training iteration
parser.add_argument('--n_his', type=int, default=4)
parser.add_argument('--sequence_length', type=int, default=0)

# shape state:
# [x, y, z, x_last, y_last, z_last, quat(4), quat_last(4)]
parser.add_argument('--shape_state_dim', type=int, default=14)

# object attributes:
parser.add_argument('--attr_dim', type=int, default=0)

# object state:
parser.add_argument('--state_dim', type=int, default=0)

# relation attr:
parser.add_argument('--relation_dim', type=int, default=0)

# physics parameter
parser.add_argument('--physics_param_range', type=float, nargs=2, default=None)

# width and height for storing vision
parser.add_argument('--vis_width', type=int, default=160)
parser.add_argument('--vis_height', type=int, default=120)


'''
train
'''

parser.add_argument('--n_rollout', type=int, default=0)
parser.add_argument('--train_valid_ratio', type=float, default=0.9)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--log_per_iter', type=int, default=50)
parser.add_argument('--ckp_per_iter', type=int, default=1000)

parser.add_argument('--n_epoch', type=int, default=1000)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--optimizer', default='Adam', help='Adam|SGD')
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=1)

# data generation
parser.add_argument('--gen_data', type=int, default=0)
parser.add_argument('--gen_stat', type=int, default=0)
parser.add_argument('--gen_vision', type=int, default=0)

parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--resume_epoch', type=int, default=-1)
parser.add_argument('--resume_iter', type=int, default=-1)

# data augmentation
parser.add_argument('--augment_ratio', type=float, default=0.)


'''
eval
'''
parser.add_argument('--eval_epoch', type=int, default=-1, help='pretrained model')
parser.add_argument('--eval_iter', type=int, default=-1, help='pretrained model')
parser.add_argument('--eval_set', default='demo')

# visualization flog
parser.add_argument('--pyflex', type=int, default=1)
parser.add_argument('--vispy', type=int, default=1)


def gen_args():
    args = parser.parse_args()

    args.data_names = ['positions', 'shape_quats', 'scene_params']

    if args.env == 'RigidFall':
        args.env_idx = 3

        args.n_rollout = 5000  # The number of training trajectories
        args.time_step = 121

        # object states:
        # [x, y, z]
        args.state_dim = 3

        # object attr:
        # [particle, floor]
        args.attr_dim = 2

        args.neighbor_radius = 0.08
        args.neighbor_k = 20

        suffix = ''
        if args.n_instance == -1:
            args.n_instance = 3
        else:
            suffix += '_nIns_' + str(args.n_instance)

        args.physics_param_range = (-15., -5.)

        args.outf = 'YOUR_OUT_DIR' + args.outf + '_' + args.stage + suffix
        args.evalf = 'YOUR_OUR_DIR' + args.evalf + '_' + args.stage + suffix

        args.mean_p = np.array([0.14778039, 0.15373468, 0.10396217])
        args.std_p = np.array([0.27770899, 0.13548609, 0.15006677])
        args.mean_d = np.array([-1.91248869e-05, -2.05043765e-03, 2.10580908e-05])
        args.std_d = np.array([0.00468072, 0.00703023, 0.00304786])

    elif args.env == 'MassRope':
        args.env_idx = 9

        args.n_rollout = 3000
        args.time_step = 201

        # object states:
        # [x, y, z]
        args.state_dim = 3

        # object attr:
        # [particle, pin]
        args.attr_dim = 2

        args.neighbor_radius = 0.25
        args.neighbor_k = -1

        suffix = ''
        if args.n_instance == -1:
            args.n_instance = 2
        else:
            suffix += '_nIns_' + str(args.n_instance)

        args.physics_param_range = (0.25, 1.2)

        args.outf = 'YOUR_OUT_DIR' + args.outf + '_' + args.stage + suffix
        args.evalf = 'YOUR_OUT_DIR' + args.evalf + '_' + args.stage + suffix

        args.mean_p = np.array([0.06443707, 1.09444374, 0.04942945])
        args.std_p = np.array([0.45214754, 0.29002383, 0.41175843])
        args.mean_d = np.array([-0.00097918, -0.00033966, -0.00080952])
        args.std_d = np.array([0.02086366, 0.0145161, 0.01856096])

    else:
        raise AssertionError("Unsupported env")


    # path to data
    args.dataf = 'YOUR_DATA_DIR' + args.dataf + '_' + args.env


    # n_his
    args.outf += '_nHis%d' % args.n_his
    args.evalf += '_nHis%d' % args.n_his


    # data augmentation
    if args.augment_ratio > 0:
        args.outf += '_aug%.2f' % args.augment_ratio
        args.evalf += '_aug%.2f' % args.augment_ratio


    # evaluation checkpoints
    if args.stage in ['dy']:
        if args.eval_epoch > -1:
            args.evalf += '_dyEpoch_' + str(args.eval_epoch)
            args.evalf += '_dyIter_' + str(args.eval_iter)
        else:
            args.evalf += '_dyEpoch_best'

        args.evalf += '_%s' % args.eval_set


    return args
