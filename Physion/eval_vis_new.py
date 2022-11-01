import os
import cv2
import sys
import random
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy
import gzip
import pickle
import h5py

import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from data_new import load_data, load_data_dominoes, prepare_input, normalize, denormalize, recalculate_velocities, \
                 correct_bad_chair, remove_large_obstacles, subsample_particles_on_large_objects
from models import SGNN
from utils import mkdir, get_query_dir
from utils_geom import calc_rigid_transform

training_data_root = get_query_dir("training_data_dir")
testing_data_root = get_query_dir("testing_data_dir")

data_root = get_query_dir("dpi_data_dir")
model_root = get_query_dir("out_dir")
out_root = os.path.join(model_root, "eval")
random.seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--env', default='')
parser.add_argument('--time_step', type=int, default=0)
parser.add_argument('--time_step_clip', type=int, default=0)
parser.add_argument('--dt', type=float, default=1./60.)
parser.add_argument('--training_fpt', type=float, default=1)
parser.add_argument('--subsample', type=int, default=3000)

# parser.add_argument('--nf_relation', type=int, default=300)
# parser.add_argument('--nf_particle', type=int, default=200)
# parser.add_argument('--nf_effect', type=int, default=200)
parser.add_argument('--n_layer', type=int, default=1)
parser.add_argument('--p_step', type=int, default=4)
parser.add_argument('--hidden_dim', type=int, default=200)

parser.add_argument('--rand_rot', type=int, default=0)
parser.add_argument('--pred_only', type=int, default=1)

parser.add_argument('--modelf', default='files')
parser.add_argument('--dataf', default='data')
parser.add_argument('--evalf', default='eval')
parser.add_argument('--mode', default='valid')
parser.add_argument('--statf', default="")
parser.add_argument('--eval', type=int, default=1)
parser.add_argument('--gt_only', type=int, default=0)
parser.add_argument('--test_training_data_processing', type=int, default=0)
parser.add_argument('--ransac_on_pred', type=int, default=0)
parser.add_argument('--verbose_data', type=int, default=0)
parser.add_argument('--verbose_model', type=int, default=0)
parser.add_argument('--model_name', default='DPINet2')

parser.add_argument('--debug', type=int, default=0)

parser.add_argument('--n_instances', type=int, default=0)
parser.add_argument('--n_stages', type=int, default=0)
parser.add_argument('--n_his', type=int, default=0)
parser.add_argument('--augment_worldcoord', type=int, default=0)
parser.add_argument('--floor_cheat', type=int, default=0)
# shape state:
# [x, y, z, x_last, y_last, z_last, quat(4), quat_last(4)]
parser.add_argument('--shape_state_dim', type=int, default=14)

# object attributes:
parser.add_argument('--attr_dim', type=int, default=0)

# object state:
parser.add_argument('--state_dim', type=int, default=0)
parser.add_argument('--position_dim', type=int, default=0)

# relation attr:
parser.add_argument('--relation_dim', type=int, default=0)

#visualization
parser.add_argument('--interactive', type=int, default=0)
parser.add_argument('--saveavi', type=int, default=0)
parser.add_argument('--save_pred', type=int, default=1)

args = parser.parse_args()

phases_dict = dict()

if args.env == "TDWdominoes":
    args.n_rollout = 2# how many data
    data_names = ['positions', 'velocities']
    args.time_step = 200
    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3

    # object attr:
    # [rigid, fluid, root_0]
    args.attr_dim = 3
    args.dt = 0.01

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.n_instance = -1
    args.time_step_clip = 0
    args.n_stages = 4
    args.n_stages_types = ["leaf-leaf", "leaf-root", "root-root", "root-leaf"]

    args.neighbor_radius = 0.08
    args.gen_data = False

    phases_dict = dict()  # load from data
    model_name = copy.deepcopy(args.modelf)
    args.modelf = 'dump/' + args.modelf
    args.modelf = os.path.join(model_root, args.modelf)
else:
    raise AssertionError("Unsupported env")


gt_only = args.gt_only

evalf_root = os.path.join(out_root, args.evalf + '_' + args.env, model_name)
mkdir(os.path.join(out_root, args.evalf + '_' + args.env))
mkdir(evalf_root)

mode = args.mode
if mode == "train":
    hdf5_root = training_data_root
elif mode == "test":
    hdf5_root = testing_data_root
else:
    raise ValueError
data_root_ori = data_root
scenario = args.dataf
args.data_root = data_root

prefix = args.dataf
if gt_only:
    prefix += "_gtonly"

args.dataf = os.path.join(data_root, mode, args.dataf)
stat = [np.zeros((3, 3)), np.zeros((3, 3))]

if not gt_only:
    if args.statf:
        stat_path = os.path.join(data_root_ori, args.statf)
        print("Loading stored stat from %s" % stat_path)
        stat = load_data(data_names[:2], stat_path)
        for i in range(len(stat)):
            stat[i] = stat[i][-args.position_dim:, :]
            # print(data_names[i], stat[i].shape)

    use_gpu = torch.cuda.is_available()

    if args.model_name == 'SGNN':
        args.noise_std = 3e-4
        model = SGNN(n_layer=args.n_layer, s_dim=4, hidden_dim=args.hidden_dim, activation=nn.SiLU(),
                     cutoff=0.08, gravity_axis=1, p_step=args.p_step)
    else:
        raise ValueError(f"no such model {args.model_name} for env {args.env}" )

    if args.epoch == 0 and args.iter == 0:
        model_file = os.path.join(args.modelf, 'net_best.pth')
    else:
        model_file = os.path.join(args.modelf, 'net_epoch_%d_iter_%d.pth' % (args.epoch, args.iter))

    print("Loading network from %s" % model_file)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    criterionMSE = nn.MSELoss()

    if use_gpu:
        model.cuda()

mode = args.mode

# list all the args
# only evaluate on human data now

infos = np.arange(100)
data_name = args.dataf.split("/")[-1]

if args.save_pred:

    pred_gif_folder = os.path.join(evalf_root, mode + "-"+ scenario)

    if args.ransac_on_pred:
        pred_gif_folder = os.path.join(evalf_root, "ransacOnPred-" + mode +  scenario)
    mkdir(pred_gif_folder)
accs = []
recs = []

dt = args.training_fpt * args.dt

gt_preds = []
arg_names = [file for file in os.listdir(args.dataf) if not file.endswith("txt")]
trial_full_paths = []
for arg_name in arg_names:
    trial_full_paths.append(os.path.join(args.dataf, arg_name))


for trial_id, trial_name in enumerate(trial_full_paths):
    print('processing case', trial_id, 'total', len(trial_full_paths))

    gt_node_rs_idxs = []

    if "Support" in trial_name:
        max_timestep = 205
    elif "Link" in trial_name:
        max_timestep = 140
    elif "Contain" in trial_name:
        max_timestep = 125
    elif "Collide" in trial_name or "Drape" in trial_name:
        max_timestep = 55
    else:
        max_timestep = 105

    args.time_step = len([file for file in os.listdir(trial_name) if file.endswith(".h5")]) - 1

    print("Rollout %d / %d" % (trial_id, len(trial_full_paths)))

    timesteps = [t for t in range(0, args.time_step - int(args.training_fpt), int(args.training_fpt))]
    max_timestep = len(timesteps)
    total_nframes = max_timestep  # len(timesteps)

    if args.env == "TDWdominoes":
        pkl_path = os.path.join(trial_name, 'phases_dict.pkl')
        with open(pkl_path, "rb") as f:
            phases_dict = pickle.load(f)

    phases_dict["trial_dir"] = trial_name
    print(phases_dict["n_particles"])

    if args.test_training_data_processing:
        is_bad_chair = correct_bad_chair(phases_dict)
        # remove obstacles that are too big
        is_remove_obstacles = remove_large_obstacles(phases_dict)
        # downsample large object
        is_subsample = subsample_particles_on_large_objects(phases_dict, limit=args.subsample)
        # is_subsample = True
        print("trial_id", trial_id, "is_bad_chair", is_bad_chair, "is_remove_obstacles", is_remove_obstacles, "is_subsample", is_subsample)
        print("trial_name", trial_name)
    else:
        is_bad_chair = correct_bad_chair(phases_dict)
        # remove obstacles that are too big
        is_remove_obstacles = remove_large_obstacles(phases_dict, limit=args.subsample)
        # downsample large object
        # is_subsample = subsample_particles_on_large_objects(phases_dict, 4000)
    print(phases_dict["n_particles"])

    if args.rand_rot:
        print('Using Random Rotation!')
        temp_data_path = os.path.join(trial_name, str(0) + '.h5')
        temp_data = load_data_dominoes(data_names, temp_data_path, phases_dict)
        center = np.mean(temp_data[0], axis=0).reshape(1, -1)
        d = np.zeros(3)
        d[1] = 1
        x, y, z = d[0], d[1], d[2]
        theta = 0.5 * np.pi
        # theta = (np.random.rand(1) - 0.5) * 4
        cos, sin = np.cos(theta), np.sin(theta)
        ret = [
            [cos + (1 - cos) * x * x,
             (1 - cos) * x * y - sin * z,
             (1 - cos) * x * z + sin * y],
            [(1 - cos) * x * y + sin * z,
             cos + (1 - cos) * y * y,
             (1 - cos) * y * z - sin * x],
            [(1 - cos) * x * z - sin * y,
             (1 - cos) * y * z + sin * x,
             cos + (1 - cos) * z * z]
        ]
        Q = np.array(ret)

    for current_fid, step in enumerate(timesteps):
        data_path = os.path.join(trial_name, str(step) + '.h5')
        data_nxt_path = os.path.join(trial_name, str(step + int(args.training_fpt)) + '.h5')
        data = load_data_dominoes(data_names, data_path, phases_dict)
        if args.rand_rot:
            data[0] = np.matmul((data[0] - center), Q) + center
        data_nxt = load_data_dominoes(data_names, data_nxt_path, phases_dict)
        if args.rand_rot:
            data_nxt[0] = np.matmul((data_nxt[0] - center), Q) + center
        data_prev_path = os.path.join(trial_name, str(max(0, step - int(args.training_fpt))) + '.h5')
        data_prev = load_data_dominoes(data_names, data_prev_path, phases_dict)
        if args.rand_rot:
            data_prev[0] = np.matmul((data_prev[0] - center), Q) + center
        _, data, data_nxt = recalculate_velocities([data_prev, data, data_nxt], dt, data_names)
        attr, state, rels, n_particles, n_shapes, instance_idx = \
                prepare_input(data, stat, args, phases_dict, args.verbose_data)
        Ra, node_r_idx, node_s_idx, pstep, rels_types = rels[3], rels[4], rels[5], rels[6], rels[7]
        velocities_nxt = data_nxt[1]
        if step == 0:
            if args.env == "TDWdominoes":
                positions, velocities = data
                clusters = phases_dict["clusters"]
                n_shapes = 0
            else:
                raise AssertionError("Unsupported env")
            count_nodes = positions.shape[0]
            n_particles = count_nodes - n_shapes
            print("n_particles", n_particles)
            print("n_shapes", n_shapes)
            p_gt = np.zeros((total_nframes, n_particles + n_shapes, args.position_dim))
            s_gt = np.zeros((total_nframes, n_shapes, args.shape_state_dim))
            v_nxt_gt = np.zeros((total_nframes, n_particles + n_shapes, args.position_dim))
            p_pred = np.zeros((total_nframes, n_particles + n_shapes, args.position_dim))
        p_gt[current_fid] = positions[:, -args.position_dim:]
        v_nxt_gt[current_fid] = velocities_nxt[:, -args.position_dim:]
        positions = positions + velocities_nxt * dt

    n_actual_frames = len(timesteps)
    for step in range(n_actual_frames, total_nframes):
        p_gt[step] = p_gt[n_actual_frames - 1]
        # gt_node_rs_idxs.append(gt_node_rs_idxs[-1])

    if not gt_only:

        # model rollout
        start_timestep = 45  # 15
        start_id = 15  # 5
        data_path = os.path.join(trial_name, f'{start_timestep}.h5')

        data = load_data_dominoes(data_names, data_path, phases_dict)
        if args.rand_rot:
            data[0] = np.matmul((data[0] - center), Q) + center
        data_path_prev = os.path.join(trial_name, f'{int(start_timestep - args.training_fpt)}.h5')
        data_prev = load_data_dominoes(data_names, data_path_prev, phases_dict)
        if args.rand_rot:
            data_prev[0] = np.matmul((data_prev[0] - center), Q) + center
        _, data = recalculate_velocities([data_prev, data], dt, data_names)

        # node_rs_idxs = []
        for t in range(start_id):
            p_pred[t] = p_gt[t]
            # node_rs_idxs.append(gt_node_rs_idxs[t])

        for current_fid in range(total_nframes - start_id):
            if current_fid % 10 == 0:
                print("Step %d / %d" % (current_fid + start_id, total_nframes))
            p_pred[start_id + current_fid] = data[0]

            x = data[0]
            v = data[1]
            h = np.zeros((x.shape[0], 4))
            obj_id = np.zeros_like(x)[..., 0]
            obj_type = []
            instance_idx = phases_dict['instance_idx']
            for i in range(len(instance_idx) - 1):
                obj_id[instance_idx[i]:instance_idx[i + 1]] = i
                obj_type.append(phases_dict['material'][i])
                if phases_dict['material'][i] == 'rigid':
                    h[instance_idx[i]:instance_idx[i + 1], 0] = 1
                elif phases_dict['material'][i] in ['cloth', 'fluid']:
                    h[instance_idx[i]:instance_idx[i + 1], 1] = 1
            x = torch.Tensor(x)
            v = torch.Tensor(v)
            h = torch.Tensor(h)
            obj_id = torch.LongTensor(obj_id)

            buf = [x, v, h, obj_id]

            with torch.set_grad_enabled(False):
                if use_gpu:
                    for d in range(len(buf)):
                        if type(buf[d]) == list:
                            for t in range(len(buf[d])):
                                buf[d][t] = Variable(buf[d][t].cuda())
                        else:
                            buf[d] = Variable(buf[d].cuda())
                else:
                    for d in range(len(buf)):
                        if type(buf[d]) == list:
                            for t in range(len(buf[d])):
                                buf[d][t] = Variable(buf[d][t])
                        else:
                            buf[d] = Variable(buf[d])
                x, v, h, obj_id = buf
                vels = model(x, v, h, obj_id, obj_type)
                if args.debug:
                    data_nxt_path = os.path.join(trial_name, str(step + args.training_fpt) + '.h5')
                    data_nxt = normalize(load_data(data_names, data_nxt_path), stat)
                    label = Variable(torch.FloatTensor(data_nxt[1][:n_particles]).cuda())
                    # print(label)
                    loss = np.sqrt(criterionMSE(vels, label).item())
                    print(loss)

            vels = denormalize([vels.data.cpu().numpy()], [stat[1]])[0]

            if args.ransac_on_pred:
                positions_prev = data[0]
                predicted_positions = data[0] + vels * dt
                for obj_id in range(len(instance_idx) - 1):
                    st, ed = instance_idx[obj_id], instance_idx[obj_id + 1]
                    if phases_dict['material'][obj_id] == 'rigid':

                        pos_prev = positions_prev[st:ed]
                        pos_pred = predicted_positions[st:ed]

                        R, T = calc_rigid_transform(pos_prev, pos_pred)
                        refined_pos = (np.dot(R, pos_prev.T) + T).T

                        predicted_positions[st:ed, :] = refined_pos
                data[0] = predicted_positions
                data[1] = (predicted_positions - positions_prev)/dt
            else:
                data[0] = data[0] + vels * dt
                data[1][:, :args.position_dim] = vels

            if args.debug:
                data[0] = p_gt[current_fid + 1].copy()
                data[1][:, :args.position_dim] = v_nxt_gt[current_fid]

        import scipy
        spacing = 0.05
        st0, st1, st2 = instance_idx[0], instance_idx[1], instance_idx[2]
        obj_0_pos = p_pred[-1][st0:st1, :]
        obj_1_pos = p_pred[-1][st1:st2, :]

        sim_mat = scipy.spatial.distance_matrix(obj_0_pos, obj_1_pos, p=2)
        min_dist1=np.min(sim_mat)
        pred_target_contacting_zone = min_dist1 < spacing

        obj_0_pos = p_gt[-1][st0:st1, :]
        obj_1_pos = p_gt[-1][st1:st2, :]

        sim_mat = scipy.spatial.distance_matrix(obj_0_pos, obj_1_pos, p=2)
        min_dist2= np.min(sim_mat)
        gt_target_contacting_zone = min_dist2 < spacing * 0.8
        acc = int(gt_target_contacting_zone == pred_target_contacting_zone)

        accs.append(acc)
        print(args.dataf)
        print("gt vs pred:", gt_target_contacting_zone, pred_target_contacting_zone, min_dist2, min_dist1)
        print("accuracy:", np.mean(accs))

    # render in VisPy
    import vispy.scene
    from vispy import app
    from vispy.visuals import transforms
    from utils_vis import create_instance_colors, convert_groups_to_colors
    particle_size = 6.0
    n_instance = 5  # args.n_instance
    y_rotate_deg = 0
    vis_length = total_nframes


    def y_rotate(obj, deg=y_rotate_deg):
        tr = vispy.visuals.transforms.MatrixTransform()
        tr.rotate(deg, (0, 1, 0))
        obj.transform = tr

    def add_floor(v):
        # add floor
        floor_thickness = 0.025
        floor_length = 8.0
        w, h, d = floor_length, floor_length, floor_thickness
        b1 = vispy.scene.visuals.Box(width=w, height=h, depth=d, color=[0.8, 0.8, 0.8, 1], edge_color='black')
        # y_rotate(b1)
        v.add(b1)

        # adjust position of box
        mesh_b1 = b1.mesh.mesh_data
        v1 = mesh_b1.get_vertices()
        c1 = np.array([0., -floor_thickness*0.5, 0.], dtype=np.float32)
        mesh_b1.set_vertices(np.add(v1, c1))

        mesh_border_b1 = b1.border.mesh_data
        vv1 = mesh_border_b1.get_vertices()
        cc1 = np.array([0., -floor_thickness*0.5, 0.], dtype=np.float32)
        mesh_border_b1.set_vertices(np.add(vv1, cc1))

    c = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = c.central_widget.add_view()

    if "Collide" in trial_name:
        distance = 6.0
    elif "Support" in trial_name:
        distance = 6.0  # 6.0
    elif "Link" in trial_name:
        distance = 10.0
    elif "Drop" in trial_name:
        distance = 5.0
    elif "Drape" in trial_name:
        distance = 5.0
    else:
        distance = 3.0
    # 5
    view.camera = vispy.scene.cameras.TurntableCamera(fov=50, azimuth=80, elevation=30, distance=distance, up='+y')
    n_instance = len(phases_dict["instance"])
    # set instance colors
    instance_colors = create_instance_colors(n_instance)
    # render floor
    add_floor(view)

    # render particles
    p1 = vispy.scene.visuals.Markers()
    p1.antialias = 0  # remove white edge

    # y_rotate(p1)
    floor_pos = np.array([[0, -0.5, 0]])
    # line = vispy.scene.visuals.Line()
    view.add(p1)
    # view.add(line)
    # set animation
    t_step = 0

    '''
    set up data for rendering
    '''
    # 0 - p_pred: seq_length x n_p x 3
    # 1 - p_gt: seq_length x n_p x 3
    # 2 - s_gt: seq_length x n_s x 3
    print('p_pred', p_pred.shape)
    print('p_gt', p_gt.shape)
    print('s_gt', s_gt.shape)
    print('gt_node_rs_idx', len(gt_node_rs_idxs))

    # create directory to save images if not exist
    vispy_dir = os.path.join(pred_gif_folder, "vispy" + f"_{prefix}")

    os.system('mkdir -p ' + vispy_dir)

    def update(event):
        global p1
        global t_step
        global colors

        if t_step < vis_length:
            if t_step == 0:
                print("Rendering ground truth")
            t_actual = t_step
            colors = convert_groups_to_colors(
                phases_dict["instance_idx"],
                instance_colors=instance_colors, env=args.env)
            colors = np.clip(colors, 0., 1.)
            n_particle = phases_dict["instance_idx"][-1]
            p1.set_data(p_gt[t_actual, :n_particle], size=particle_size, edge_color='black', face_color=colors)
            # line.set_data(pos=np.concatenate([p_gt[t_actual, :], floor_pos], axis=0))
            # line.set_data(pos=np.concatenate([p_gt[t_actual, :], floor_pos], axis=0), connect=gt_node_rs_idxs[t_actual])
            # render for ground truth
            img = c.render()
            idx_episode = trial_id
            img_path = os.path.join(vispy_dir, "gt_{}_{}.png".format(str(idx_episode), str(t_actual)))
            vispy.io.write_png(img_path, img)

        elif not gt_only and vis_length <= t_step < vis_length * 2:
            if t_step == vis_length:
                print("Rendering prediction result")

            t_actual = t_step - vis_length

            colors = convert_groups_to_colors(
                phases_dict["instance_idx"],
                instance_colors=instance_colors, env=args.env)

            colors = np.clip(colors, 0., 1.)
            n_particle = phases_dict["instance_idx"][-1]
            p1.set_data(p_pred[t_actual, :n_particle], size=particle_size, edge_color='black', face_color=colors)
            # line.set_data(pos=np.concatenate([p_pred[t_actual, :n_particle], floor_pos], axis=0), connect=node_rs_idxs[t_actual])
            # line.set_data(pos=np.concatenate([p_pred[t_actual, :n_particle], floor_pos], axis=0))

            # render for perception result
            img = c.render()
            idx_episode = trial_id
            img_path = os.path.join(vispy_dir, "pred_{}_{}.png".format(str(idx_episode), str(t_actual)))
            vispy.io.write_png(img_path, img)

        else:
            # discarded frames
            pass

        # time forward
        t_step += 1

    # start animation

    if args.interactive:
        timer = app.Timer()
        timer.connect(update)
        timer.start(interval=1. / 60., iterations=vis_length * 2)
        c.show()
        app.run()

    else:
        for i in range(vis_length * 2):
            update(1)

    print("Render video for dynamics prediction")
    idx_episode = trial_id
    if args.saveavi:
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        out = cv2.VideoWriter(
            os.path.join(args.evalf, 'vid_%d_vispy.avi' % (idx_episode)),
            fourcc, 20, (800, 600))

        for step in range(vis_length):
            gt_path = os.path.join('vispy', 'gt_%d_%d.png' % (idx_episode, step))
            # pred_path = os.path.join(args.evalf, 'vispy', 'pred_%d_%d.png' % (idx_episode, step))
            gt = cv2.imread(gt_path)
            # pred = cv2.imread(pred_path)
            frame = np.zeros((600, 800, 3), dtype=np.uint8)
            frame[:, :800] = gt
            # frame[:, 800:] = pred
            out.write(frame)
        out.release()
    else:
        import imageio
        gt_imgs = []
        pred_imgs = []
        gt_paths = []
        pred_paths = []

        for step in range(vis_length):
            gt_path = os.path.join(vispy_dir, 'gt_%d_%d.png' % (idx_episode, step))
            gt_imgs.append(imageio.imread(gt_path))
            gt_paths.append(gt_path)
            if not gt_only:
                pred_path = os.path.join(vispy_dir, 'pred_%d_%d.png' % (idx_episode, step))
                pred_imgs.append(imageio.imread(pred_path))
                pred_paths.append(pred_path)

        if gt_only:
            imgs = gt_imgs
        elif args.pred_only:
            nimgs = len(gt_imgs)
            imgs = []
            for img_id in range(nimgs):
                imgs.append(pred_imgs[img_id])
        else:
            nimgs = len(gt_imgs)
            imgs = []
            for img_id in range(nimgs):
                imgs.append(np.concatenate([gt_imgs[img_id], pred_imgs[img_id]], axis=1))

        out = imageio.mimsave(
            os.path.join(pred_gif_folder, '%s_vid_%d_vispy.gif' % (prefix, idx_episode)),
            imgs, fps=20)
        print(os.path.join(pred_gif_folder, '%s_vid_%d_vispy.gif' % (prefix, idx_episode)))

        [os.remove(gt_path) for gt_path in gt_paths + pred_paths]

