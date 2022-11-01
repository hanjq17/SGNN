import os
import time
import sys
import copy

import multiprocessing as mp
# from progressbar import ProgressBar
import torch.nn as nn
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config import gen_args
from data_new import PhysicsFleXDataset, new_collate
from data_new import prepare_input, get_scene_info, get_env_group
from models.SGNN_batch import SGNN
from utils import make_graph, check_gradient, set_seed, AverageMeter, get_lr, Tee
from utils import count_parameters, my_collate


args = gen_args()
set_seed(args.random_seed)

os.system('mkdir -p ' + args.dataf)
os.system('mkdir -p ' + args.outf)

tee = Tee(os.path.join(args.outf, 'train.log'), 'w')


def get_logger(save_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(save_dir + "/log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
### training
logger = get_logger(args.outf)
# load training data

phases = ['train', 'valid'] if args.eval == 0 else ['valid']
datasets = {phase: PhysicsFleXDataset(args, phase) for phase in phases}

for phase in phases:
    if args.gen_data:
        datasets[phase].gen_data(args.env)
    else:
        datasets[phase].load_data(args.env)

dataloaders = {phase: DataLoader(
    datasets[phase],
    batch_size=args.batch_size,
    shuffle=True if phase == 'train' else False,
    num_workers=args.num_workers,
    collate_fn=new_collate) for phase in phases}

# create model and train
use_gpu = torch.cuda.is_available()
model = SGNN(n_layer=1, p_step=4, s_dim=2, hidden_dim=200, activation=nn.SiLU(), gravity_axis=1).cuda()

print("model #params: %d" % count_parameters(model))
logger.info("model #params: %d" % count_parameters(model))


# checkpoint to reload model from
model_path = None

# resume training of a saved model (if given)
if args.resume == 0:
    print("Randomly initialize the model's parameters")
    logger.info("Randomly initialize the model's parameters")

# optimizer
if args.stage == 'dy':
    params = model.parameters()
else:
    raise AssertionError("unknown stage: %s" % args.stage)

if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(
        params, lr=args.lr, betas=(args.beta1, 0.999))
elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=0.9)
else:
    raise AssertionError("unknown optimizer: %s" % args.optimizer)

# reduce learning rate when a metric has stopped improving
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)

# define loss
particle_dist_loss = torch.nn.L1Loss()

if use_gpu:
    model = model.cuda()

# log args
print(args)
logger.info(args)

# start training
st_epoch = args.resume_epoch if args.resume_epoch > 0 else 0
best_valid_loss = np.inf
cur_best_epoch = -1

mean_p = torch.FloatTensor(args.mean_p).cuda()
std_p = torch.FloatTensor(args.std_p).cuda()
mean_d = torch.FloatTensor(args.mean_d).cuda()
std_d = torch.FloatTensor(args.std_d).cuda()

for epoch in range(st_epoch, args.n_epoch):

    for phase in phases:

        model.train(phase == 'train')

        meter_loss = AverageMeter()
        meter_loss_raw = AverageMeter()

        meter_loss_ref = AverageMeter()
        meter_loss_nxt = AverageMeter()

        meter_loss_param = AverageMeter()


        # bar = ProgressBar(max_value=len(dataloaders[phase]))

        for i, data in enumerate(dataloaders[phase]):
            # each "data" is a trajectory of sequence_length time steps

            if args.stage == 'dy':
                attrs, particles, edge_index_inner, edge_index_inter, obj_id = data

                if use_gpu:
                    attrs = attrs.cuda()
                    particles = particles.cuda()
                    edge_index_inner = edge_index_inner.cuda()
                    edge_index_inter = edge_index_inter.cuda()
                    obj_id = obj_id.cuda()

                with torch.set_grad_enabled(phase == 'train'):
                    # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
                    state_cur = particles[:, :args.n_his]

                    v_norm = (state_cur[:, 1:] - state_cur[:, :-1] - mean_d) / std_d
                    x_norm = (state_cur[:, -1:] - mean_p) / std_p
                    x_norm = x_norm.squeeze(1)
                    v_norm = v_norm.squeeze(1)  # [B, N, 3]
                    h = attrs
                    B = h.shape[0]

                    pred_motion_norm = model(x_norm.reshape(-1, 3), v_norm.reshape(-1, 3), h.reshape(-1, 2), edge_index_inner,
                                             edge_index_inter, obj_id)
                    pred_motion_norm = pred_motion_norm.reshape(B, -1, 3)
                    pred_pos = state_cur[:, -1] + (pred_motion_norm * std_d + mean_d)

                    # concatenate the state of the shapes
                    # pred_pos (unnormalized): B x (n_p + n_s) x state_dim
                    n_particle = 64 * 3
                    gt_pos = particles[:, args.n_his]
                    pred_pos = torch.cat([pred_pos, gt_pos[:, n_particle:]], 1)

                    # gt_motion_norm (normalized): B x (n_p + n_s) x state_dim
                    # pred_motion_norm (normalized): B x (n_p + n_s) x state_dim
                    gt_motion = particles[:, args.n_his] - particles[:, args.n_his - 1]
                    # mean_d, std_d = model.stat[2:]
                    gt_motion_norm = (gt_motion - mean_d) / std_d
                    pred_motion_norm = torch.cat([pred_motion_norm, gt_motion_norm[:, n_particle:]], 1)

                    loss = F.l1_loss(pred_motion_norm[:, :n_particle], gt_motion_norm[:, :n_particle])
                    loss_raw = F.l1_loss(pred_pos, gt_pos)

                    meter_loss.update(loss.item(), B)
                    meter_loss_raw.update(loss_raw.item(), B)

                if i % args.log_per_iter == 0:
                    print()
                    print('%s epoch[%d/%d] iter[%d/%d] LR: %.6f, loss: %.6f (%.6f), loss_raw: %.8f (%.8f)' % (
                        phase, epoch, args.n_epoch, i, len(dataloaders[phase]), get_lr(optimizer),
                        loss.item(), meter_loss.avg, loss_raw.item(), meter_loss_raw.avg))
                    logger.info('%s epoch[%d/%d] iter[%d/%d] LR: %.6f, loss: %.6f (%.6f), loss_raw: %.8f (%.8f)' % (
                        phase, epoch, args.n_epoch, i, len(dataloaders[phase]), get_lr(optimizer),
                        loss.item(), meter_loss.avg, loss_raw.item(), meter_loss_raw.avg))


            # update model parameters
            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if phase == 'train' and i > 0 and i % args.ckp_per_iter == 0:
                model_path = '%s/net_epoch_%d_iter_%d.pth' % (args.outf, epoch, i)
                torch.save(model.state_dict(), model_path)


        print('%s epoch[%d/%d] Loss: %.6f, Best valid: %.6f' % (
            phase, epoch, args.n_epoch, meter_loss.avg, best_valid_loss))
        logger.info('%s epoch[%d/%d] Loss: %.6f, Best valid: %.6f' % (
            phase, epoch, args.n_epoch, meter_loss.avg, best_valid_loss))

        if phase == 'valid' and not args.eval:
            scheduler.step(meter_loss.avg)
            if meter_loss.avg < best_valid_loss:
                best_valid_loss = meter_loss.avg
                torch.save(model.state_dict(), '%s/net_best.pth' % (args.outf))
                cur_best_epoch = epoch
            if epoch >= cur_best_epoch + 10:
                print('Early stopping with 10 epochs')
                logger.info('Early stopping with 10 epochs')
                exit(0)
