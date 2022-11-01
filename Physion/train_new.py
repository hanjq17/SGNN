import os
import sys
import random
import logging
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import SGNN
from data_new import PhysicsFleXDataset, collate_fn
from utils import count_parameters, get_query_dir


def get_logger(save_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(save_dir + "/log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


data_root = get_query_dir("dpi_data_dir")
out_root = get_query_dir("out_dir")

parser = argparse.ArgumentParser()
parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--n_rollout', type=int, default=0)
parser.add_argument('--time_step', type=int, default=0)
parser.add_argument('--time_step_clip', type=int, default=0)
parser.add_argument('--dt', type=float, default=1./60.)
parser.add_argument('--training_fpt', type=float, default=1)

# parser.add_argument('--nf_relation', type=int, default=300)
# parser.add_argument('--nf_particle', type=int, default=200)
# parser.add_argument('--nf_effect', type=int, default=200)
parser.add_argument('--n_layer', type=int, default=1)
parser.add_argument('--p_step', type=int, default=4)
parser.add_argument('--hidden_dim', type=int, default=200)

parser.add_argument('--model_name', default='DPINet2')
parser.add_argument('--floor_cheat', type=int, default=0)
parser.add_argument('--env', default='')
parser.add_argument('--train_valid_ratio', type=float, default=0.9)
parser.add_argument('--outf', default='files')
parser.add_argument('--dataf', default='data')
parser.add_argument('--statf', default="")
parser.add_argument('--noise_std', type=float, default='0')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--gen_stat', type=int, default=0)

parser.add_argument('--subsample_particles', type=int, default=1)

parser.add_argument('--log_per_iter', type=int, default=1000)
parser.add_argument('--ckp_per_iter', type=int, default=10000)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--augment_worldcoord', type=int, default=0)

parser.add_argument('--verbose_data', type=int, default=0)
parser.add_argument('--verbose_model', type=int, default=0)

parser.add_argument('--n_instance', type=int, default=0)
parser.add_argument('--n_stages', type=int, default=0)
parser.add_argument('--n_his', type=int, default=0)

parser.add_argument('--n_epoch', type=int, default=1000)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--forward_times', type=int, default=2)

parser.add_argument('--resume_epoch', type=int, default=0)
parser.add_argument('--resume_iter', type=int, default=0)

# shape state:
# [x, y, z, x_last, y_last, z_last, quat(4), quat_last(4)]
parser.add_argument('--shape_state_dim', type=int, default=14)

# object attributes:
parser.add_argument('--attr_dim', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)

# object state:
parser.add_argument('--state_dim', type=int, default=0)
parser.add_argument('--position_dim', type=int, default=0)

# relation attr:
parser.add_argument('--relation_dim', type=int, default=0)

args = parser.parse_args()

phases_dict = dict()

random.seed(args.seed)
torch.manual_seed(args.seed)
print('Fix seed', args.seed)

model_saved_name_list = []
# torch.autograd.set_detect_anomaly(True)

# preparing phases_dict
if args.env == "TDWdominoes":
    args.n_rollout = None  # how many data
    # don't use, determined by data
    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3
    args.dt = 0.01

    # object attr:
    # [rigid, fluid, root_0]
    args.attr_dim = 3

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.n_instance = -1
    args.time_step = 301  # ??
    args.time_step_clip = 0
    args.n_stages = 4
    args.n_stages_types = ["leaf-leaf", "leaf-root", "root-root", "root-leaf"]

    args.neighbor_radius = 0.08

    phases_dict = dict()  # load from data
    # ["root_num"] = [[]]
    # phases_dict["instance"] = ["fluid"]
    # phases_dict["material"] = ["fluid"]
    args.outf = args.outf.strip()
    args.outf = os.path.join(out_root, 'dump/' + args.outf)

else:
    raise AssertionError("Unsupported env")

writer = SummaryWriter(os.path.join(args.outf, "log"))
data_root = os.path.join(data_root, "train")
args.data_root = data_root
if "," in args.dataf:
    # list of folder
    args.dataf = [os.path.join(data_root, tmp.strip()) for tmp in args.dataf.split(",") if tmp != ""]

else:
    args.dataf = args.dataf.strip()
    if "/" in args.dataf:
        args.dataf = 'data/' + args.dataf
    else:  # only prefix
        args.dataf = 'data/' + args.dataf + '_' + args.env
    os.system('mkdir -p ' + args.dataf)
os.system('mkdir -p ' + args.outf)

logger = get_logger(args.outf)
logger.info('Fix seed ' + str(args.seed))
# generate data
datasets = {phase: PhysicsFleXDataset(
    args, phase, phases_dict, args.verbose_data) for phase in ['train', 'valid']}

for phase in ['train', 'valid']:
    datasets[phase].load_data(args.env)

use_gpu = torch.cuda.is_available()
assert(use_gpu)

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
dataloaders = {x: torch.utils.data.DataLoader(
    datasets[x], batch_size=args.batch_size,
    shuffle=True if x == 'train' else False,
    # num_workers=args.num_workers,
    collate_fn=collate_fn)
    for x in ['train', 'valid']}

# define propagation network
if args.env == "TDWdominoes":

    if args.model_name == 'SGNN':
        args.noise_std = 3e-4
        # param here
        logger.info('layer {}, p_step {}, hidden {}, lr {}'.format(args.n_layer, args.p_step, args.hidden_dim, args.lr))
        model = SGNN(n_layer=args.n_layer, s_dim=4, hidden_dim=args.hidden_dim, activation=nn.SiLU(),
                     cutoff=0.08, gravity_axis=1, p_step=args.p_step)
    else:
        raise ValueError(f"no such model {args.model_name} for env {args.env}")
else:
    raise RuntimeError('Unknown env', args.env)

if use_gpu:
    model = model.cuda()

print('Using', args.model_name)
print("Number of parameters: %d" % count_parameters(model))
logger.info('Using ' + args.model_name)
logger.info("Number of parameters: %d" % count_parameters(model))

with open(os.path.join(args.outf, "args_stat.pkl"), 'wb') as f:
    import pickle
    pickle.dump(args, f)

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)

if args.resume_epoch > 0 or args.resume_iter > 0:
    # load local parameters
    args_load = model.load_local(os.path.join(args.outf, "args_stat.pkl"))
    args_current = vars(args)

    exempt_list = ["dataf", "lr", "num_workers", "resume_epoch", "resume_iter"]

    for key in args_load:
        if key in exempt_list:
            continue

        assert(args_load[key] == args_current[key]), f"{key} is mismatched in loaded args and current args: {args_load[key]} vs {args_current[key]}"

    # check args_load
    model_path = os.path.join(args.outf, 'net_epoch_%d_iter_%d.pth' % (args.resume_epoch, args.resume_iter))
    print("Loading saved ckp from %s" % model_path)
    logger.info("Loading saved ckp from %s" % model_path)

    checkpoint = torch.load(model_path)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        model.load_state_dict(torch.load(model_path))

# criterion
criterionMSE = nn.MSELoss()

# optimizer

optimizer.zero_grad()
if use_gpu:
    model = model.cuda()

st_epoch = args.resume_epoch if args.resume_epoch > 0 else 0
best_valid_loss = np.inf
train_iter = 0
current_loss = 0
best_valid_epoch = -1

max_nparticles = 0
for epoch in range(st_epoch, args.n_epoch):

    phases = ['train', 'valid'] if args.eval == 0 else ['valid']
    for phase in phases:
        import time

        model.train(phase == 'train')
        previous_run_time = time.time()
        start_time = time.time()

        losses = 0.
        for i, data in enumerate(dataloaders[phase]):

            # start_time = time.time()
            # print("previous run time", start_time - previous_run_time)
            x, v, h, obj_id, obj_type, v_target = data
            x = torch.Tensor(x)
            v = torch.Tensor(v)
            h = torch.Tensor(h)
            obj_id = torch.LongTensor(obj_id)
            v_target = torch.Tensor(v_target)
            data = [x, v, h, obj_id, v_target]

            with torch.set_grad_enabled(phase == 'train'):
                if use_gpu:
                    for d in range(len(data)):
                        if type(data[d]) == list:
                            for t in range(len(data[d])):
                                data[d][t] = Variable(data[d][t].cuda())
                        else:
                            data[d] = Variable(data[d].cuda())
                else:
                    for d in range(len(data)):
                        if type(data[d]) == list:
                            for t in range(len(data[d])):
                                data[d][t] = Variable(data[d][t])
                        else:
                            data[d] = Variable(data[d])

                x, v, h, obj_id, v_target = data
                predicted = model(x, v, h, obj_id, obj_type)
            label = v_target
            loss = criterionMSE(predicted, label) / args.forward_times
            current_loss = np.sqrt(loss.item() * args.forward_times)
            losses += np.sqrt(loss.item())
            if phase == 'train':
                train_iter += 1
                loss.backward()
                if i % args.forward_times == 0 and i!=0:
                    # update parameters every args.forward_times
                    optimizer.step()
                    optimizer.zero_grad()
            if i % args.log_per_iter == 0:
                n_relations = model.n_relation
                print('%s %s [%d/%d][%d/%d] n_relations: %d, cur loss: %.6f, average loss: %.6f' %
                      (phase, args.outf, epoch, args.n_epoch, i, len(dataloaders[phase]),
                       n_relations, current_loss, losses / (i + 1)))
                logger.info('%s %s [%d/%d][%d/%d] n_relations: %d, cur loss: %.6f, average loss: %.6f' %
                      (phase, args.outf, epoch, args.n_epoch, i, len(dataloaders[phase]),
                       n_relations, current_loss, losses / (i + 1)))
                print("total time:", time.time() - start_time)
                logger.info("total time: {:.3f}".format(time.time() - start_time))
                start_time = time.time()
                lr = get_lr(optimizer)
                if phase == "train":
                    writer.add_scalar(f'lr', lr, train_iter)
                writer.add_histogram(f'{phase}/label_x', label[:,0], train_iter)
                writer.add_histogram(f'{phase}/label_y', label[:,1], train_iter)
                writer.add_histogram(f'{phase}/label_z', label[:,2], train_iter)
                writer.add_histogram(f'{phase}/predicted_x', predicted[:,0], train_iter)
                writer.add_histogram(f'{phase}/predicted_y', predicted[:,1], train_iter)
                writer.add_histogram(f'{phase}/predicted_z', predicted[:,2], train_iter)
                writer.add_scalar(f'{phase}/loss', current_loss, train_iter)
            previous_run_time = time.time()

            if phase == 'train' and i > 0 and i % args.ckp_per_iter == 0:
                torch.save({'model_state_dict': model.state_dict()},
                           '%s/net_epoch_%d_iter_%d.pth' % (args.outf, epoch, i))

        print("total time:", time.time() - previous_run_time)
        logger.info("total time: {:.3f}".format(time.time() - previous_run_time))
        losses /= len(dataloaders[phase])
        if phase == 'valid':
            scheduler.step(losses)
            if losses < best_valid_loss:
                best_valid_loss = losses
                best_valid_epoch = epoch
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()},
                           '%s/net_best.pth' % (args.outf))
            logger.info("Validation: epoch {:4d}, cur val loss {:.6f}, best val loss {:.6f}".format(epoch,
                                                                                                    losses,
                                                                                                    best_valid_loss))
            logger.info("Best valid epoch {:4d}".format(best_valid_epoch))
            logger.info("current lr {:.6f}".format(float(get_lr(optimizer))))
            # check early stopping
            if epoch - best_valid_epoch >= 10:
                print('Early stopping with 10 epochs!')
                logger.info('Early stopping with 10 epochs!')
                print('Best valid loss {:.6f}, Best valid epoch {:4d}'.format(best_valid_loss, best_valid_epoch))
                logger.info('Best valid loss {:.6f}, Best valid epoch {:4d}'.format(best_valid_loss, best_valid_epoch))
                exit(0)

