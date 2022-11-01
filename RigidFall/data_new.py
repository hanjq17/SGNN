import multiprocessing as mp
import os
import cv2
import time

import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
import scipy.misc
import scipy.spatial as spatial
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch_geometric.nn.pool import radius_graph

from utils import rand_int, rand_float


### from DPI

def store_data(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()


def load_data(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data


def combine_stat(stat_0, stat_1):
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt((std_0 ** 2 * n_0 + std_1 ** 2 * n_1 + \
                (mean_0 - mean) ** 2 * n_0 + (mean_1 - mean) ** 2 * n_1) / (n_0 + n_1))
    n = n_0 + n_1

    return np.stack([mean, std, n], axis=-1)


def init_stat(dim):
    # mean, std, count
    return np.zeros((dim, 3))


def normalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            stat[i][stat[i][:, 1] == 0, 1] = 1.
            s = Variable(torch.FloatTensor(stat[i]).cuda())

            stat_dim = stat[i].shape[0]
            n_rep = int(data[i].size(1) / stat_dim)
            data[i] = data[i].view(-1, n_rep, stat_dim)

            data[i] = (data[i] - s[:, 0]) / s[:, 1]

            data[i] = data[i].view(-1, n_rep * stat_dim)

    else:
        for i in range(len(stat)):
            stat[i][stat[i][:, 1] == 0, 1] = 1.

            stat_dim = stat[i].shape[0]
            n_rep = int(data[i].shape[1] / stat_dim)
            data[i] = data[i].reshape((-1, n_rep, stat_dim))

            data[i] = (data[i] - stat[i][:, 0]) / stat[i][:, 1]

            data[i] = data[i].reshape((-1, n_rep * stat_dim))

    return data


def denormalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            s = Variable(torch.FloatTensor(stat[i]).cuda())
            data[i] = data[i] * s[:, 1] + s[:, 0]
    else:
        for i in range(len(stat)):
            data[i] = data[i] * stat[i][:, 1] + stat[i][:, 0]

    return data


def calc_rigid_transform(XX, YY):
    X = XX.copy().T
    Y = YY.copy().T

    mean_X = np.mean(X, 1, keepdims=True)
    mean_Y = np.mean(Y, 1, keepdims=True)
    X = X - mean_X
    Y = Y - mean_Y
    C = np.dot(X, Y.T)
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(3)
    D[2, 2] = np.linalg.det(np.dot(Vt.T, U.T))
    R = np.dot(Vt.T, np.dot(D, U.T))
    T = mean_Y - np.dot(R, mean_X)

    '''
    YY_fitted = (np.dot(R, XX.T) + T).T
    print("MSE fit", np.mean(np.square(YY_fitted - YY)))
    '''

    return R, T


def normalize_scene_param(scene_params, param_idx, param_range, norm_range=(-1, 1)):
    normalized = np.copy(scene_params[param_idx])
    low, high = param_range
    if low == high:
        return normalized
    nlow, nhigh = norm_range
    normalized = nlow + (normalized - low) * (nhigh - nlow) / (high - low)
    return normalized


def gen_PyFleX(info):
    env, env_idx = info['env'], info['env_idx']
    thread_idx, data_dir, data_names = info['thread_idx'], info['data_dir'], info['data_names']
    n_rollout, time_step = info['n_rollout'], info['time_step']
    shape_state_dim, dt = info['shape_state_dim'], info['dt']

    gen_vision = info['gen_vision']
    vision_dir, vis_width, vis_height = info['vision_dir'], info['vis_width'], info['vis_height']

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

    # positions
    stats = [init_stat(3)]

    import pyflex
    pyflex.init()

    for i in range(n_rollout):

        if i % 10 == 0:
            print("%d / %d" % (i, n_rollout))

        rollout_idx = thread_idx * n_rollout + i
        rollout_dir = os.path.join(data_dir, str(rollout_idx))
        os.system('mkdir -p ' + rollout_dir)

        if env == 'RigidFall':
            g_low, g_high = info['physics_param_range']
            gravity = rand_float(g_low, g_high)
            print("Generated RigidFall rollout {} with gravity {} from range {} ~ {}".format(
                i, gravity, g_low, g_high))

            n_instance = 3
            draw_mesh = 1
            scene_params = np.zeros(n_instance * 3 + 3)
            scene_params[0] = n_instance
            scene_params[1] = gravity
            scene_params[-1] = draw_mesh

            low_bound = 0.09
            for j in range(n_instance):
                x = rand_float(0., 0.1)
                y = rand_float(low_bound, low_bound + 0.01)
                z = rand_float(0., 0.1)

                scene_params[j * 3 + 2] = x
                scene_params[j * 3 + 3] = y
                scene_params[j * 3 + 4] = z

                low_bound += 0.21

            pyflex.set_scene(env_idx, scene_params, thread_idx)
            pyflex.set_camPos(np.array([0.2, 0.875, 2.0]))

            n_particles = pyflex.get_n_particles()
            n_shapes = 1    # the floor

            positions = np.zeros((time_step, n_particles + n_shapes, 3), dtype=np.float32)
            shape_quats = np.zeros((time_step, n_shapes, 4), dtype=np.float32)

            for j in range(time_step):
                positions[j, :n_particles] = pyflex.get_positions().reshape(-1, 4)[:, :3]

                ref_positions = positions[0]

                for k in range(n_instance):
                    XX = ref_positions[64*k:64*(k+1)]
                    YY = positions[j, 64*k:64*(k+1)]

                    X = XX.copy().T
                    Y = YY.copy().T

                    mean_X = np.mean(X, 1, keepdims=True)
                    mean_Y = np.mean(Y, 1, keepdims=True)
                    X = X - mean_X
                    Y = Y - mean_Y
                    C = np.dot(X, Y.T)
                    U, S, Vt = np.linalg.svd(C)
                    D = np.eye(3)
                    D[2, 2] = np.linalg.det(np.dot(Vt.T, U.T))
                    R = np.dot(Vt.T, np.dot(D, U.T))
                    t = mean_Y - np.dot(R, mean_X)

                    YY_fitted = (np.dot(R, XX.T) + t).T
                    # print("MSE fit", np.mean(np.square(YY_fitted - YY)))

                    positions[j, 64*k:64*(k+1)] = YY_fitted

                if gen_vision:
                    pyflex.step(capture=True, path=os.path.join(rollout_dir, str(j) + '.tga'))
                else:
                    pyflex.step()

                data = [positions[j], shape_quats[j], scene_params]
                store_data(data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))

            if gen_vision:
                images = np.zeros((time_step, vis_height, vis_width, 3), dtype=np.uint8)
                for j in range(time_step):
                    img_path = os.path.join(rollout_dir, str(j) + '.tga')
                    img = scipy.misc.imread(img_path)[:, :, :3][:, :, ::-1]
                    img = cv2.resize(img, (vis_width, vis_height), interpolation=cv2.INTER_AREA)
                    images[j] = img
                    os.system('rm ' + img_path)

                store_data(['positions', 'images', 'scene_params'], [positions, images, scene_params],
                           os.path.join(vision_dir, str(rollout_idx) + '.h5'))

        elif env == 'MassRope':
            s_low, s_high = info['physics_param_range']
            stiffness = rand_float(s_low, s_high)
            print("Generated MassRope rollout {} with gravity {} from range {} ~ {}".format(
                i, stiffness, s_low, s_high))

            x = 0.
            y = 1.0
            z = 0.
            length = 0.7
            draw_mesh = 1.

            scene_params = np.array([x, y, z, length, stiffness, draw_mesh])

            pyflex.set_scene(env_idx, scene_params, 0)
            pyflex.set_camPos(np.array([0.13, 2.0, 3.2]))

            action = np.zeros(3)

            # the last particle is the pin, regarded as shape
            n_particles = pyflex.get_n_particles() - 1
            n_shapes = 1    # the mass at the top of the rope

            positions = np.zeros((time_step + 1, n_particles + n_shapes, 3), dtype=np.float32)
            shape_quats = np.zeros((time_step + 1, n_shapes, 4), dtype=np.float32)

            action = np.zeros(3)
            for j in range(time_step + 1):
                positions[j] = pyflex.get_positions().reshape(-1, 4)[:, :3]
                if j >= 1:
                    # append the action (position of the pin) to the previous time step
                    positions[j - 1, -1, :] = positions[j, -1, :]

                ref_positions = positions[0]

                # apply rigid projection to the rigid object
                # cube: [0, 81)
                # rope: [81, 95)
                # pin: [95, 96)
                XX = ref_positions[:81]
                YY = positions[j, :81]

                X = XX.copy().T
                Y = YY.copy().T

                mean_X = np.mean(X, 1, keepdims=True)
                mean_Y = np.mean(Y, 1, keepdims=True)
                X = X - mean_X
                Y = Y - mean_Y
                C = np.dot(X, Y.T)
                U, S, Vt = np.linalg.svd(C)
                D = np.eye(3)
                D[2, 2] = np.linalg.det(np.dot(Vt.T, U.T))
                R = np.dot(Vt.T, np.dot(D, U.T))
                t = mean_Y - np.dot(R, mean_X)

                YY_fitted = (np.dot(R, XX.T) + t).T

                positions[j, :81] = YY_fitted

                scale = 0.1
                action[0] += rand_float(-scale, scale) - positions[j, -1, 0] * 0.1
                action[2] += rand_float(-scale, scale) - positions[j, -1, 2] * 0.1

                if gen_vision:
                    pyflex.step(action * dt, capture=True, path=os.path.join(rollout_dir, str(j) + '.tga'))
                else:
                    pyflex.step(action * dt)

                if j >= 1:
                    data = [positions[j - 1], shape_quats[j - 1], scene_params]
                    store_data(data_names, data, os.path.join(rollout_dir, str(j - 1) + '.h5'))

            if gen_vision:
                images = np.zeros((time_step, vis_height, vis_width, 3), dtype=np.uint8)
                for j in range(time_step):
                    img_path = os.path.join(rollout_dir, str(j) + '.tga')
                    img = scipy.misc.imread(img_path)[:, :, :3][:, :, ::-1]
                    img = cv2.resize(img, (vis_width, vis_height), interpolation=cv2.INTER_AREA)
                    images[j] = img
                    os.system('rm ' + img_path)

                store_data(['positions', 'images', 'scene_params'], [positions, images, scene_params],
                           os.path.join(vision_dir, str(rollout_idx) + '.h5'))

        else:
            raise AssertionError("Unsupported env")

        # change dtype for more accurate stat calculation
        # only normalize positions
        datas = [positions[:time_step].astype(np.float64)]

        for j in range(len(stats)):
            stat = init_stat(stats[j].shape[0])
            stat[:, 0] = np.mean(datas[j], axis=(0, 1))[:]
            stat[:, 1] = np.std(datas[j], axis=(0, 1))[:]
            stat[:, 2] = datas[j].shape[0] * datas[j].shape[1]
            stats[j] = combine_stat(stats[j], stat)

    pyflex.clean()

    return stats


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def visualize_neighbors(anchors, queries, idx, neighbors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(queries[idx, 0], queries[idx, 1], queries[idx, 2], c='g', s=80)
    ax.scatter(anchors[neighbors, 0], anchors[neighbors, 1], anchors[neighbors, 2], c='r', s=80)
    ax.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2], alpha=0.2)
    axisEqual3D(ax)

    plt.show()


def find_relations_neighbor(pos, query_idx, anchor_idx, radius, order, var=False):
    if np.sum(anchor_idx) == 0:
        return []

    point_tree = spatial.cKDTree(pos[anchor_idx])
    neighbors = point_tree.query_ball_point(pos[query_idx], radius, p=order)

    '''
    for i in range(len(neighbors)):
        visualize_neighbors(pos[anchor_idx], pos[query_idx], i, neighbors[i])
    '''

    relations = []
    for i in range(len(neighbors)):
        count_neighbors = len(neighbors[i])
        if count_neighbors == 0:
            continue

        receiver = np.ones(count_neighbors, dtype=np.int) * query_idx[i]
        sender = np.array(anchor_idx[neighbors[i]])

        # receiver, sender, relation_type
        relations.append(np.stack([receiver, sender], axis=1))

    return relations


def find_k_relations_neighbor(k, positions, query_idx, anchor_idx, radius, order, var=False):
    """
    Same as find_relations_neighbor except that each point is only connected to the k nearest neighbors

    For each particle, only take the first min_neighbor neighbors, where
    min_neighbor = minimum number of neighbors among all particle's numbers of neighbors
    """
    if np.sum(anchor_idx) == 0:
        return []

    pos = positions.data.cpu().numpy() if var else positions

    point_tree = spatial.cKDTree(pos[anchor_idx])
    neighbors = point_tree.query_ball_point(pos[query_idx], radius, p=order)

    '''
    for i in range(len(neighbors)):
        visualize_neighbors(pos[anchor_idx], pos[query_idx], i, neighbors[i])
    '''

    relations = []
    min_neighbors = None
    for i in range(len(neighbors)):
        if min_neighbors is None:
            min_neighbors = len(neighbors[i])
        elif len(neighbors[i]) < min_neighbors:
            min_neighbors = len(neighbors[i])
        else:
            pass

    for i in range(len(neighbors)):
        receiver = np.ones(min_neighbors, dtype=np.int) * query_idx[i]
        sender = np.array(anchor_idx[neighbors[i][:min_neighbors]])

        # receiver, sender, relation_type
        relations.append(np.stack([receiver, sender], axis=1))

    return relations


def get_scene_info(data):
    """
    A subset of prepare_input() just to get number of particles
    for initialization of grouping
    """
    positions, shape_quats, scene_params = data
    n_shapes = shape_quats.shape[0]
    count_nodes = positions.shape[0]
    n_particles = count_nodes - n_shapes

    return n_particles, n_shapes, scene_params


def get_env_group(args, n_particles, scene_params, use_gpu=False):
    # n_particles (int)
    # scene_params: B x param_dim
    B = scene_params.shape[0]

    p_rigid = torch.zeros(B, args.n_instance)
    p_instance = torch.zeros(B, n_particles, args.n_instance)
    physics_param = torch.zeros(B, n_particles)

    if args.env == 'RigidFall':
        norm_g = normalize_scene_param(scene_params, 1, args.physics_param_range)
        physics_param[:] = torch.FloatTensor(norm_g).view(B, 1)

        p_rigid[:] = 1

        for i in range(args.n_instance):
            p_instance[:, 64 * i:64 * (i + 1), i] = 1

    elif args.env == 'MassRope':
        norm_stiff = normalize_scene_param(scene_params, 4, args.physics_param_range)
        physics_param[:] = torch.FloatTensor(norm_stiff).view(B, 1)

        n_rigid_particle = 81

        p_rigid[:, 0] = 1
        p_instance[:, :n_rigid_particle, 0] = 1
        p_instance[:, n_rigid_particle:, 1] = 1

    else:
        raise AssertionError("Unsupported env")

    if use_gpu:
        p_rigid = p_rigid.cuda()
        p_instance = p_instance.cuda()
        physics_param = physics_param.cuda()

    # p_rigid: B x n_instance
    # p_instance: B x n_p x n_instance
    # physics_param: B x n_p
    return [p_rigid, p_instance, physics_param]


def prepare_input(positions, n_particle, n_shape, args, var=False):
    # positions: (n_p + n_s) x 3

    verbose = args.verbose_data

    count_nodes = n_particle + n_shape

    if verbose:
        print("prepare_input::positions", positions.shape)
        print("prepare_input::n_particle", n_particle)
        print("prepare_input::n_shape", n_shape)

    ### object attributes
    attr = np.zeros((count_nodes, args.attr_dim))

    ##### add env specific graph components
    rels = []
    if args.env == 'RigidFall':
        # object attr:
        # [particle, floor]
        attr[n_particle, 1] = 1
        pos = positions.data.cpu().numpy() if var else positions

        # conncetion between floor and particles when they are close enough
        dis = pos[:n_particle, 1] - pos[n_particle, 1]
        nodes = np.nonzero(dis < args.neighbor_radius)[0]

        '''
        if verbose:
            visualize_neighbors(pos, pos, 0, nodes)
            print(np.sort(dis)[:10])
        '''

        floor = np.ones(nodes.shape[0], dtype=np.int) * n_particle
        rels += [np.stack([nodes, floor], axis=1)]

    elif args.env == 'MassRope':
        pos = positions.data.cpu().numpy() if var else positions
        dis = np.sqrt(np.sum((pos[n_particle] - pos[:n_particle])**2, 1))
        nodes = np.nonzero(dis < args.neighbor_radius)[0]

        '''
        if verbose:
            visualize_neighbors(pos, pos, 0, nodes)
            print(np.sort(dis)[:10])
        '''

        pin = np.ones(nodes.shape[0], dtype=np.int) * n_particle
        rels += [np.stack([nodes, pin], axis=1)]

    else:
        AssertionError("Unsupported env %s" % args.env)

    ##### add relations between leaf particles

    if args.env in ['RigidFall', 'MassRope']:
        queries = np.arange(n_particle)
        anchors = np.arange(n_particle)

    rels += find_relations_neighbor(pos, queries, anchors, args.neighbor_radius, 2, var)
    # rels += find_k_relations_neighbor(args.neighbor_k, pos, queries, anchors, args.neighbor_radius, 2, var)

    if len(rels) > 0:
        rels = np.concatenate(rels, 0)

    if verbose:
        print("Relations neighbor", rels.shape)

    n_rel = rels.shape[0]
    Rr = torch.zeros(n_rel, n_particle + n_shape)
    Rs = torch.zeros(n_rel, n_particle + n_shape)
    Rr[np.arange(n_rel), rels[:, 0]] = 1
    Rs[np.arange(n_rel), rels[:, 1]] = 1

    if verbose:
        print("Object attr:", np.sum(attr, axis=0))
        print("Particle attr:", np.sum(attr[:n_particle], axis=0))
        print("Shape attr:", np.sum(attr[n_particle:n_particle + n_shape], axis=0))

    if verbose:
        print("Particle positions stats")
        print("  Shape", positions.shape)
        print("  Min", np.min(positions[:n_particle], 0))
        print("  Max", np.max(positions[:n_particle], 0))
        print("  Mean", np.mean(positions[:n_particle], 0))
        print("  Std", np.std(positions[:n_particle], 0))

    if var:
        particle = positions
    else:
        particle = torch.FloatTensor(positions)

    if verbose:
        for i in range(count_nodes - 1):
            if np.sum(np.abs(attr[i] - attr[i + 1])) > 1e-6:
                print(i, attr[i], attr[i + 1])

    attr = torch.FloatTensor(attr)
    assert attr.size(0) == count_nodes
    assert attr.size(1) == args.attr_dim

    # attr: (n_p + n_s) x attr_dim
    # particle (unnormalized): (n_p + n_s) x state_dim
    # Rr, Rs: n_rel x (n_p + n_s)
    return attr, particle, Rr, Rs


class PhysicsFleXDataset(Dataset):

    def __init__(self, args, phase):
        self.args = args
        self.phase = phase
        self.data_dir = os.path.join(self.args.dataf, phase)
        self.vision_dir = self.data_dir + '_vision'
        self.stat_path = os.path.join(self.args.dataf, 'stat.h5')

        if args.gen_data:
            os.system('mkdir -p ' + self.data_dir)
        if args.gen_vision:
            os.system('mkdir -p ' + self.vision_dir)

        if args.env in ['RigidFall', 'MassRope']:
            self.data_names = ['positions', 'shape_quats', 'scene_params']
        else:
            raise AssertionError("Unsupported env")

        ratio = self.args.train_valid_ratio
        if phase == 'train':
            self.n_rollout = int(self.args.n_rollout * ratio)
        elif phase == 'valid':
            self.n_rollout = self.args.n_rollout - int(self.args.n_rollout * ratio)
        else:
            raise AssertionError("Unknown phase")

    def __len__(self):
        """
        Each data point is consisted of a whole trajectory
        """
        args = self.args
        return self.n_rollout * (args.time_step - args.sequence_length + 1)

    def load_data(self, name):
        print("Loading stat from %s ..." % self.stat_path)
        self.stat = load_data(self.data_names[:1], self.stat_path)
        # print(self.stat)
        # exit(0)

    def gen_data(self, name):
        # if the data hasn't been generated, generate the data
        print("Generating data ... n_rollout=%d, time_step=%d" % (self.n_rollout, self.args.time_step))

        infos = []
        for i in range(self.args.num_workers):
            info = {
                'env': self.args.env,
                'thread_idx': i,
                'data_dir': self.data_dir,
                'data_names': self.data_names,
                'n_rollout': self.n_rollout // self.args.num_workers,
                'time_step': self.args.time_step,
                'dt': self.args.dt,
                'shape_state_dim': self.args.shape_state_dim,
                'physics_param_range': self.args.physics_param_range,

                'gen_vision': self.args.gen_vision,
                'vision_dir': self.vision_dir,
                'vis_width': self.args.vis_width,
                'vis_height': self.args.vis_height}

            if self.args.env == 'RigidFall':
                info['env_idx'] = 3
            elif self.args.env == 'MassRope':
                info['env_idx'] = 9
            else:
                raise AssertionError("Unsupported env")

            infos.append(info)

        cores = self.args.num_workers
        pool = mp.Pool(processes=cores)
        data = pool.map(gen_PyFleX, infos)

        print("Training data generated, warpping up stats ...")

        if self.phase == 'train' and self.args.gen_stat:
            # positions [x, y, z]
            self.stat = [init_stat(3)]
            for i in range(len(data)):
                for j in range(len(self.stat)):
                    self.stat[j] = combine_stat(self.stat[j], data[i][j])
            store_data(self.data_names[:1], self.stat, self.stat_path)
        else:
            print("Loading stat from %s ..." % self.stat_path)
            self.stat = load_data(self.data_names[:1], self.stat_path)

    def __getitem__(self, idx):
        """
        Load a trajectory of length sequence_length
        """
        args = self.args

        offset = args.time_step - args.sequence_length + 1

        idx_rollout = idx // offset
        st_idx = idx % offset
        ed_idx = st_idx + args.sequence_length

        if args.stage in ['dy']:
            # load ground truth data
            attrs, particles, Rrs, Rss = [], [], [], []
            max_n_rel = 0
            for t in range(st_idx, ed_idx):
                # load data
                data_path = os.path.join(self.data_dir, str(idx_rollout), str(t) + '.h5')
                data = load_data(self.data_names, data_path)

                # load scene param
                if t == st_idx:
                    n_particle, n_shape, scene_params = get_scene_info(data)

                particles.append(data[0])

        '''
        add augmentation
        '''
        if args.stage in ['dy']:
            for t in range(args.sequence_length):
                if t == args.n_his - 1:
                    # set anchor for transforming rigid objects
                    particle_anchor = particles[t].copy()

                if t < args.n_his:
                    # add noise to observation frames - idx smaller than n_his
                    noise = np.random.randn(n_particle, 3) * args.std_d * args.augment_ratio
                    particles[t][:n_particle] += noise

                else:
                    # for augmenting rigid object,
                    # make sure the rigid transformation is the same before and after augmentation
                    if args.env == 'RigidFall':
                        for k in range(args.n_instance):
                            XX = particle_anchor[64*k:64*(k+1)]
                            XX_noise = particles[args.n_his - 1][64*k:64*(k+1)]

                            YY = particles[t][64*k:64*(k+1)]

                            R, T = calc_rigid_transform(XX, YY)

                            particles[t][64*k:64*(k+1)] = (np.dot(R, XX_noise.T) + T).T

                    elif args.env == 'MassRope':
                        n_rigid_particle = 81

                        XX = particle_anchor[:n_rigid_particle]
                        XX_noise = particles[args.n_his - 1][:n_rigid_particle]
                        YY = particles[t][:n_rigid_particle]

                        R, T = calc_rigid_transform(XX, YY)

                        particles[t][:n_rigid_particle] = (np.dot(R, XX_noise.T) + T).T


        else:
            AssertionError("Unknown stage %s" % args.stage)


        # attr: (n_p + n_s) x attr_dim
        # particles (unnormalized): seq_length x (n_p + n_s) x state_dim
        # scene_params: param_dim
        attr = torch.zeros(n_particle, 2)
        attr = attr[:64*3, :]
        particles = torch.FloatTensor(np.stack(particles))  # [T, N, 3]
        particles = particles[:, :64*3, :]
        mask = particles[1, :, 1] < args.neighbor_radius
        attr[mask, 1] = 1
        scene_params = torch.FloatTensor(scene_params)

        cur_x = particles[1, ...]
        obj_id = torch.zeros_like(attr)[..., 0].long()
        obj_id[:64] = 0
        obj_id[64:64*2] = 1
        obj_id[64*2:] = 2

        edge_index = radius_graph(cur_x, r=0.08, loop=False)  # [2, M]
        edge_index_inner_mask = obj_id[edge_index[0]] == obj_id[edge_index[1]]
        edge_index_inter_mask = obj_id[edge_index[0]] != obj_id[edge_index[1]]
        edge_index_inner = edge_index[..., edge_index_inner_mask]  # [2, M_in]
        edge_index_inter = edge_index[..., edge_index_inter_mask]  # [2, M_out]

        norm_g = normalize_scene_param(scene_params, 1, args.physics_param_range)
        attr[..., 0] = norm_g

        if args.stage in ['dy']:
            return attr, particles, edge_index_inner, edge_index_inter, obj_id


def new_collate(data):
    attr = [_[0] for _ in data]
    particles = [_[1] for _ in data]
    edge_index_inner = [_[2] for _ in data]
    edge_index_inter = [_[3] for _ in data]
    obj_id = [_[4] for _ in data]
    # attr, particles, edge_index_inner, edge_index_inter = data[0], data[1], data[2], data[3]
    N = 64 * 3
    attr = torch.stack(attr, dim=0)
    particles = torch.stack(particles, dim=0)
    edge_index_inner = torch.cat([edge_index_inner[_] + N * _ for _ in range(len(edge_index_inner))], dim=1)
    edge_index_inter = torch.cat([edge_index_inter[_] + N * _ for _ in range(len(edge_index_inter))], dim=1)
    obj_id = torch.cat([obj_id[_] + 3 * _ for _ in range(len(obj_id))])
    return attr, particles, edge_index_inner, edge_index_inter, obj_id
