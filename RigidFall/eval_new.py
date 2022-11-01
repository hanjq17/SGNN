import argparse
import copy
import os
import time
import cv2
import torch.nn as nn
import numpy as np
import scipy.misc
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn.pool import radius_graph
from models.SGNN_batch import SGNN

from config import gen_args
from data_new import normalize_scene_param
from data_new import load_data, get_scene_info
from data_new import get_env_group, prepare_input, denormalize

from utils import add_log, convert_groups_to_colors
from utils import create_instance_colors, set_seed, Tee, count_parameters
from data_new import calc_rigid_transform

args = gen_args()
set_seed(args.random_seed)

os.system('mkdir -p ' + args.evalf)
os.system('mkdir -p ' + os.path.join(args.evalf, 'render'))

tee = Tee(os.path.join(args.evalf, 'eval.log'), 'w')


### evaluating

data_names = args.data_names

use_gpu = torch.cuda.is_available()

# create model and load weights
model = SGNN(n_layer=1, p_step=4, s_dim=2, hidden_dim=200, activation=nn.SiLU(), gravity_axis=1).cuda()

print("model_kp #params: %d" % count_parameters(model))

if args.eval_epoch < 0:
    model_name = 'net_best.pth'
else:
    model_name = 'net_epoch_%d_iter_%d.pth' % (args.eval_epoch, args.eval_iter)

model_path = os.path.join(args.outf, model_name)
print("Loading network from %s" % model_path)

if args.stage == 'dy':
    pretrained_dict = torch.load(model_path)
    model_dict = model.state_dict()
    model.load_state_dict(pretrained_dict, strict=True)

else:
    AssertionError("Unsupported stage %s, using other evaluation scripts" % args.stage)

model.eval()


if use_gpu:
    model = model.cuda()

mean_p = torch.FloatTensor(args.mean_p).cuda()
std_p = torch.FloatTensor(args.std_p).cuda()
mean_d = torch.FloatTensor(args.mean_d).cuda()
std_d = torch.FloatTensor(args.std_d).cuda()

# infos = np.arange(10)
infos = np.arange(500)

loss_keeper_all = []
loss_raw_keeper_all = []

for idx_episode in range(len(infos)):

    print("Rollout %d / %d" % (idx_episode, len(infos)))

    B = 1
    n_particle, n_shape = 0, 0

    # ground truth
    datas = []
    p_gt = []
    s_gt = []
    for step in range(args.time_step):
        data_path = os.path.join(args.dataf, 'valid', str(infos[idx_episode]), str(step) + '.h5')

        data = load_data(data_names, data_path)

        if n_particle == 0 and n_shape == 0:
            n_particle, n_shape, scene_params = get_scene_info(data)
            scene_params = torch.FloatTensor(scene_params).unsqueeze(0)

        if args.verbose_data:
            print("n_particle", n_particle)
            print("n_shape", n_shape)

        datas.append(data)

        p_gt.append(data[0])
        s_gt.append(data[1])

    # p_gt: time_step x N x state_dim
    # s_gt: time_step x n_s x 4
    p_gt = torch.FloatTensor(np.stack(p_gt))
    s_gt = torch.FloatTensor(np.stack(s_gt))
    p_pred = torch.zeros(args.time_step, n_particle + n_shape, args.state_dim)

    # initialize particle grouping
    group_gt = get_env_group(args, n_particle, scene_params, use_gpu=use_gpu)

    print('scene_params:', group_gt[-1][0, 0].item())

    # memory: B x mem_nlayer x (n_particle + n_shape) x nf_memory
    # for now, only used as a placeholder
    # memory_init = model.init_memory(B, n_particle + n_shape)

    # model rollout
    loss = 0.
    loss_raw = 0.
    loss_counter = 0.
    loss_raw_keeper = []
    loss_keeper = []
    st_idx = args.n_his
    ed_idx = args.sequence_length

    with torch.set_grad_enabled(False):

        for step_id in range(st_idx, ed_idx):

            if step_id == st_idx:
                # state_cur (unnormalized): n_his x (n_p + n_s) x state_dim
                state_cur = p_gt[step_id - args.n_his:step_id]
                if use_gpu:
                    state_cur = state_cur.cuda()

            if step_id % 50 == 0:
                print("Step %d / %d" % (step_id, ed_idx))

            n_particle = 64 * 3
            attr = torch.zeros(n_particle, 2)
            attr = attr[:64 * 3, :]
            particles = state_cur.clone()  # [T, N, 3]
            particles = particles[:, :64 * 3, :]
            mask = particles[1, :, 1] < args.neighbor_radius
            attr[mask, 1] = 1
            scene_params = torch.FloatTensor(scene_params)

            cur_x = particles[1, ...]
            obj_id = torch.zeros_like(attr)[..., 0].long()
            obj_id[:64] = 0
            obj_id[64:64 * 2] = 1
            obj_id[64 * 2:] = 2

            edge_index = radius_graph(cur_x, r=0.08, loop=False)  # [2, M]
            edge_index_inner_mask = obj_id[edge_index[0]] == obj_id[edge_index[1]]
            edge_index_inter_mask = obj_id[edge_index[0]] != obj_id[edge_index[1]]
            edge_index_inner = edge_index[..., edge_index_inner_mask]  # [2, M_in]
            edge_index_inter = edge_index[..., edge_index_inter_mask]  # [2, M_out]

            norm_g = normalize_scene_param(scene_params[0], 1, args.physics_param_range)
            attr[..., 0] = norm_g
            if use_gpu:
                attr = attr.cuda()
                particles = particles.cuda()
                edge_index_inner = edge_index_inner.cuda()
                edge_index_inter = edge_index_inter.cuda()
                obj_id = obj_id.cuda()

            # t
            st_time = time.time()

            v_norm = (state_cur[1:] - state_cur[:-1] - mean_d) / std_d
            x_norm = (state_cur[-1:] - mean_p) / std_p
            x_norm = x_norm.squeeze(1)
            v_norm = v_norm.squeeze(1)  # [B, N, 3]
            h = attr
            pred_motion_norm = model(x_norm.reshape(-1, 3)[:n_particle], v_norm.reshape(-1, 3)[:n_particle],
                                     h.reshape(-1, 2)[:n_particle], edge_index_inner,
                                     edge_index_inter, obj_id)
            pred_motion_norm = pred_motion_norm.reshape(-1, 3)

            # use Kab
            next_pos = cur_x + (pred_motion_norm * std_d) + mean_d
            try:
                for obj_i in range(3):
                    R, T = calc_rigid_transform(cur_x.detach().cpu().numpy()[64*obj_i: 64*(obj_i+1)], next_pos.detach().cpu().numpy()[64*obj_i: 64*(obj_i+1)])
                    next_pos[64*obj_i: 64*(obj_i+1)] = torch.from_numpy((np.dot(R, cur_x[64*obj_i: 64*(obj_i+1)].detach().cpu().numpy().T) + T).T).cuda().float()
            except:
                print('svd does not converge')
                pass
            pred_motion_norm = (next_pos - cur_x - mean_d) / std_d


            pred_pos = state_cur[-1][:n_particle] + (pred_motion_norm * std_d + mean_d)

            # concatenate the state of the shapes
            # pred_pos (unnormalized): B x (n_p + n_s) x state_dim
            n_particle = 64 * 3

            # concatenate the state of the shapes
            # pred_pos (unnormalized): B x (n_p + n_s) x state_dim
            gt_pos = p_gt[step_id]
            if use_gpu:
                gt_pos = gt_pos.cuda()
            pred_pos = torch.cat([pred_pos, gt_pos[n_particle:]], 0)

            # gt_motion_norm (normalized): B x (n_p + n_s) x state_dim
            # pred_motion_norm (normalized): B x (n_p + n_s) x state_dim
            gt_motion = (p_gt[step_id] - p_gt[step_id - 1])
            if use_gpu:
                gt_motion = gt_motion.cuda()
            # mean_d, std_d = model.stat[2:]
            gt_motion_norm = (gt_motion - mean_d) / std_d
            pred_motion_norm = torch.cat([pred_motion_norm, gt_motion_norm[n_particle:]], 0)

            loss_cur = F.l1_loss(pred_motion_norm[:, :n_particle], gt_motion_norm[:, :n_particle])
            loss_cur_raw = F.l1_loss(pred_pos, gt_pos)

            loss += loss_cur
            loss_raw += loss_cur_raw
            loss_keeper.append(loss_cur.item())
            loss_raw_keeper.append(loss_cur_raw.item())
            loss_counter += 1

            # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
            state_cur = torch.cat([state_cur[1:], pred_pos.unsqueeze(0)], 0)
            state_cur = state_cur.detach()

            # record the prediction
            p_pred[step_id] = state_cur[-1].detach().cpu()

    loss_keeper_all.append(loss_keeper)
    loss_raw_keeper_all.append(loss_raw_keeper)

    '''
    print loss
    '''
    loss /= loss_counter
    loss_raw /= loss_counter
    print("loss: %.6f, loss_raw: %.10f" % (loss.item(), loss_raw.item()))


    '''
    visualization
    '''
    group_gt = [d.data.cpu().numpy()[0, ...] for d in group_gt]
    p_pred = p_pred.numpy()[st_idx:ed_idx]
    p_gt = p_gt.numpy()[st_idx:ed_idx]
    s_gt = s_gt.numpy()[st_idx:ed_idx]
    vis_length = ed_idx - st_idx

    if args.vispy:

        ### render in VisPy
        import vispy.scene
        from vispy import app
        from vispy.visuals import transforms

        particle_size = 0.01
        border = 0.025
        height = 1.3
        y_rotate_deg = -45.0


        def y_rotate(obj, deg=y_rotate_deg):
            tr = vispy.visuals.transforms.MatrixTransform()
            tr.rotate(deg, (0, 1, 0))
            obj.transform = tr

        def add_floor(v):
            # add floor
            floor_length = 3.0
            w, h, d = floor_length, floor_length, border
            b1 = vispy.scene.visuals.Box(width=w, height=h, depth=d, color=[0.8, 0.8, 0.8, 1], edge_color='black')
            y_rotate(b1)
            v.add(b1)

            # adjust position of box
            mesh_b1 = b1.mesh.mesh_data
            v1 = mesh_b1.get_vertices()
            c1 = np.array([0., -particle_size - border, 0.], dtype=np.float32)
            mesh_b1.set_vertices(np.add(v1, c1))

            mesh_border_b1 = b1.border.mesh_data
            vv1 = mesh_border_b1.get_vertices()
            cc1 = np.array([0., -particle_size - border, 0.], dtype=np.float32)
            mesh_border_b1.set_vertices(np.add(vv1, cc1))

        def update_box_states(boxes, last_states, curr_states):
            v = curr_states[0] - last_states[0]
            if args.verbose_data:
                print("box states:", last_states, curr_states)
                print("box velocity:", v)

            tr = vispy.visuals.transforms.MatrixTransform()
            tr.rotate(y_rotate_deg, (0, 1, 0))

            for i, box in enumerate(boxes):
                # use v to update box translation
                trans = (curr_states[i][0], curr_states[i][1], curr_states[i][2])
                box.transform = tr * vispy.visuals.transforms.STTransform(translate=trans)

        def translate_box(b, x, y, z):
            mesh_b = b.mesh.mesh_data
            v = mesh_b.get_vertices()
            c = np.array([x, y, z], dtype=np.float32)
            mesh_b.set_vertices(np.add(v, c))

            mesh_border_b = b.border.mesh_data
            vv = mesh_border_b.get_vertices()
            cc = np.array([x, y, z], dtype=np.float32)
            mesh_border_b.set_vertices(np.add(vv, cc))

        def add_box(v, w=0.1, h=0.1, d=0.1, x=0.0, y=0.0, z=0.0):
            """
            Add a box object to the scene view
            :param v: view to which the box should be added
            :param w: width
            :param h: height
            :param d: depth
            :param x: x center
            :param y: y center
            :param z: z center
            :return: None
            """
            # render background box
            b = vispy.scene.visuals.Box(width=w, height=h, depth=d, color=[0.8, 0.8, 0.8, 1], edge_color='black')
            y_rotate(b)
            v.add(b)

            # adjust position of box
            translate_box(b, x, y, z)

            return b

        def calc_box_init(x, z):
            boxes = []

            # floor
            boxes.append([x, z, border, 0., -particle_size / 2, 0.])

            # left wall
            boxes.append([border, z, (height + border), -particle_size / 2, 0., 0.])

            # right wall
            boxes.append([border, z, (height + border), particle_size / 2, 0., 0.])

            # back wall
            boxes.append([(x + border * 2), border, (height + border)])

            # front wall (disabled when colored)
            # boxes.append([(x + border * 2), border, (height + border)])

            return boxes

        def add_container(v, box_x, box_z):
            boxes = calc_box_init(box_x, box_z)
            visuals = []
            for b in boxes:
                if len(b) == 3:
                    visual = add_box(v, b[0], b[1], b[2])
                elif len(b) == 6:
                    visual = add_box(v, b[0], b[1], b[2], b[3], b[4], b[5])
                else:
                    raise AssertionError("Input should be either length 3 or length 6")
                visuals.append(visual)
            return visuals


        c = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
        view = c.central_widget.add_view()

        if args.env == 'RigidFall':
            view.camera = vispy.scene.cameras.TurntableCamera(fov=50, azimuth=45, elevation=20, distance=2, up='+y')
            # set instance colors
            instance_colors = create_instance_colors(args.n_instance)

            # render floor
            add_floor(view)

        if args.env == 'MassRope':
            view.camera = vispy.scene.cameras.TurntableCamera(fov=30, azimuth=0, elevation=20, distance=8, up='+y')

            # set instance colors
            n_string_particles = 15
            instance_colors = create_instance_colors(args.n_instance)

            # render floor
            add_floor(view)


        # render particles
        p1 = vispy.scene.visuals.Markers()
        p1.antialias = 0  # remove white edge

        y_rotate(p1)

        view.add(p1)

        # set animation
        t_step = 0


        '''
        set up data for rendering
        '''
        #0 - p_pred: seq_length x n_p x 3
        #1 - p_gt: seq_length x n_p x 3
        #2 - s_gt: seq_length x n_s x 3
        print('p_pred', p_pred.shape)
        print('p_gt', p_gt.shape)
        print('s_gt', s_gt.shape)

        # create directory to save images if not exist
        vispy_dir = args.evalf + "/vispy"
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
                    group_gt, n_particle, args.n_instance,
                    instance_colors=instance_colors, env=args.env)

                colors = np.clip(colors, 0., 1.)

                p1.set_data(p_gt[t_actual, :n_particle], edge_color='black', face_color=colors)

                # render for ground truth
                img = c.render()
                img_path = os.path.join(vispy_dir, "gt_{}_{}.png".format(str(idx_episode), str(t_actual)))
                vispy.io.write_png(img_path, img)


            elif vis_length <= t_step < vis_length * 2:
                if t_step == vis_length:
                    print("Rendering prediction result")

                t_actual = t_step - vis_length

                colors = convert_groups_to_colors(
                    group_gt, n_particle, args.n_instance,
                    instance_colors=instance_colors, env=args.env)

                colors = np.clip(colors, 0., 1.)

                p1.set_data(p_pred[t_actual, :n_particle], edge_color='black', face_color=colors)

                # render for perception result
                img = c.render()
                img_path = os.path.join(vispy_dir, "pred_{}_{}.png".format(str(idx_episode), str(t_actual)))
                vispy.io.write_png(img_path, img)

            else:
                # discarded frames
                pass

            # time forward
            t_step += 1

        for i in range(vis_length * 2):
            update(1)

        # render video for evaluating grouping result
        if args.stage in ['dy']:
            print("Render video for dynamics prediction")

            use_gif = True
            gt_only = True
            pred_only = False

            if use_gif:
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
                elif pred_only:
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
                    os.path.join(args.evalf, 'vid_%d_vispy.gif' % (idx_episode)),
                    imgs, fps=20)
                print(os.path.join(args.evalf, 'vid_%d_vispy.gif' % (idx_episode)))
                [os.remove(gt_path) for gt_path in gt_paths + pred_paths]

            else:

                print('saved to', os.path.join(args.evalf, 'vid_%d_vispy.avi' % (idx_episode)))

                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(
                    os.path.join(args.evalf, 'vid_%d_vispy.avi' % (idx_episode)),
                    fourcc, 20, (800 * 2, 600))

                for step in range(vis_length):
                    gt_path = os.path.join(args.evalf, 'vispy', 'gt_%d_%d.png' % (idx_episode, step))
                    pred_path = os.path.join(args.evalf, 'vispy', 'pred_%d_%d.png' % (idx_episode, step))

                    gt = cv2.imread(gt_path)
                    pred = cv2.imread(pred_path)

                    frame = np.zeros((600, 800 * 2, 3), dtype=np.uint8)
                    frame[:, :800] = gt
                    frame[:, 800:] = pred

                    out.write(frame)

                out.release()
