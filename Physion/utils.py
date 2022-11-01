from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
import h5py
import os

import torch
from torch.autograd import Variable

import socket
import getpass
import yaml
import re
import os


def get_source_dir():
    return os.getcwd()


def get_query_dir(query_dir):
    paths_yaml_fn = os.path.join(get_source_dir(), 'paths.yaml')
    with open(paths_yaml_fn, 'r') as f:
        paths_config = yaml.load(f, Loader=yaml.Loader)
    return paths_config[query_dir]


def mkdir(path, is_assert=False):
    if is_assert:
        assert(not os.path.exists(path)), f"{path} exists, delete it first if you want to overwrite"
    if not os.path.exists(path):
        os.makedirs(path)


def rand_int(lo, hi):
    return np.random.randint(lo, hi)

def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_variable(tensor, use_gpu, requires_grad=False):
    if use_gpu:
        return Variable(torch.FloatTensor(tensor).cuda(),
                        requires_grad=requires_grad)
    else:
        return Variable(torch.FloatTensor(tensor),
                        requires_grad=requires_grad)


def visualize_point_clouds(point_clouds, c=['b', 'r'], view=None, store=False, store_path=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])
    frame.axes.yaxis.set_ticklabels([])
    frame.axes.zaxis.set_ticklabels([])

    for i in range(len(point_clouds)):
        points = point_clouds[i]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c[i], s=10, alpha=0.3)

    X, Y, Z = point_clouds[0][:, 0], point_clouds[0][:, 1], point_clouds[0][:, 2]

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    ax.grid(False)

    if view is None:
        view = 0, 0
    ax.view_init(view[0], view[1])
    plt.show()

    # plt.pause(5)

    if store:
        fig.savefig(store_path, bbox_inches='tight')


def quatFromAxisAngle(axis, angle):
    axis /= np.linalg.norm(axis)

    half = angle * 0.5
    w = np.cos(half)

    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two

    quat = np.array([axis[0], axis[1], axis[2], w])

    return quat


def quatFromAxisAngle_var(axis, angle):
    axis /= torch.norm(axis)

    half = angle * 0.5
    w = torch.cos(half)

    sin_theta_over_two = torch.sin(half)
    axis *= sin_theta_over_two

    quat = torch.cat([axis, w])
    # print("quat size", quat.size())

    return quat


class ChamferLoss(torch.nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def chamfer_distance(self, x, y):
        # x: [N, D]
        # y: [M, D]
        x = x.repeat(y.size(0), 1, 1)   # x: [M, N, D]
        x = x.transpose(0, 1)           # x: [N, M, D]
        y = y.repeat(x.size(0), 1, 1)   # y: [N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=2)    # dis: [N, M]
        dis_xy = torch.mean(torch.min(dis, dim=1)[0])   # dis_xy: mean over N
        dis_yx = torch.mean(torch.min(dis, dim=0)[0])   # dis_yx: mean over M

        return dis_xy + dis_yx

    def __call__(self, pred, label):
        return self.chamfer_distance(pred, label)
