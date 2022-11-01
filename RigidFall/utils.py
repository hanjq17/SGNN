from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable


def my_collate(batch):
    len_batch = len(batch[0])
    len_rel = 2

    ret = []
    for i in range(len_batch - len_rel):
        d = [item[i] for item in batch]
        if isinstance(d[0], int):
            d = torch.LongTensor(d)
        else:
            d = torch.FloatTensor(torch.stack(d))
        ret.append(d)

    # processing relations
    # R: B x seq_length x n_rel x (n_p + n_s)
    for i in range(len_rel):
        R = [item[-len_rel + i] for item in batch]
        max_n_rel = 0
        seq_length, _, N = R[0].size()
        for j in range(len(R)):
            max_n_rel = max(max_n_rel, R[j].size(1))
        for j in range(len(R)):
            r = R[j]
            r = torch.cat([r, torch.zeros(seq_length, max_n_rel - r.size(1), N)], 1)
            R[j] = r

        R = torch.FloatTensor(torch.stack(R))

        ret.append(R)

    return tuple(ret)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()

    def close(self):
        self.__del__()


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_gradient(step):
    def hook(grad):
        print(step, torch.mean(grad, 1)[:4])
    return hook


def add_log(fn, content, is_append=True):
    if is_append:
        with open(fn, "a+") as f:
            f.write(content)
    else:
        with open(fn, "w+") as f:
            f.write(content)


def rand_int(lo, hi):
    return np.random.randint(lo, hi)


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_var(tensor, use_gpu, requires_grad=False):
    if use_gpu:
        return Variable(torch.FloatTensor(tensor).cuda(),
                        requires_grad=requires_grad)
    else:
        return Variable(torch.FloatTensor(tensor),
                        requires_grad=requires_grad)


def make_graph(log, title, args):
    """make a loss graph"""
    plt.plot(log)
    plt.xlabel('iter')
    plt.ylabel('loss')

    title + '_loss_graph'
    plt.title(title)
    plt.savefig(os.path.join(args.logf, title + '.png'))
    plt.close()


def get_color_from_prob(prob, colors):
    # there's only one instance
    if len(colors) == 1:
        return colors[0] * prob
    elif len(prob) == 1:
        return colors * prob[0]
    else:
        res = np.zeros(4)
        for i in range(len(prob)):
            res += prob[i] * colors[i]
        return res


def create_instance_colors(n):
    # TODO: come up with a better way to initialize instance colors
    return np.array([
        [1., 0., 0., 1.],
        [0., 1., 0., 1.],
        [0., 0., 1., 1.],
        [1., 1., 0., 1.],
        [1., 0., 1., 1.]])[:n]


def convert_groups_to_colors(group, n_particles, n_rigid_instances, instance_colors, env=None):
    """
    Convert grouping to RGB colors of shape (n_particles, 4)
    :param grouping: [p_rigid, p_instance, physics_param]
    :return: RGB values that can be set as color densities
    """
    # p_rigid: n_instance
    # p_instance: n_p x n_instance
    p_rigid, p_instance = group[:2]

    p = p_instance

    colors = np.empty((n_particles, 4))

    for i in range(n_particles):
        colors[i] = get_color_from_prob(p[i], instance_colors)

    # print("colors", colors)
    return colors


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

    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    ax.grid(False)
    plt.show()

    if view is None:
        view = 0, 0
    ax.view_init(view[0], view[1])
    plt.draw()

    # plt.pause(5)

    if store:
        os.system('mkdir -p ' + store_path)
        fig.savefig(os.path.join(store_path, "vis.png"), bbox_inches='tight')

    '''
    for angle in range(0, 360, 2):
        ax.view_init(90, angle)
        plt.draw()
        # plt.pause(.001)

        if store:
            if angle % 100 == 0:
                print("Saving frame %d / %d" % (angle, 360))

            os.system('mkdir -p ' + store_path)
            fig.savefig(os.path.join(store_path, "%d.png" % angle), bbox_inches='tight')
    '''


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
        x = x.repeat(y.size(0), 1, 1)  # x: [M, N, D]
        x = x.transpose(0, 1)  # x: [N, M, D]
        y = y.repeat(x.size(0), 1, 1)  # y: [N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=2)  # dis: [N, M]
        dis_xy = torch.mean(torch.min(dis, dim=1)[0])  # dis_xy: mean over N
        dis_yx = torch.mean(torch.min(dis, dim=0)[0])  # dis_yx: mean over M

        return dis_xy + dis_yx

    def __call__(self, pred, label):
        return self.chamfer_distance(pred, label)


def get_l2_loss(g):
    num_particles = len(g)
    return torch.norm(num_particles - torch.norm(g, dim=1, keepdim=True))
