import torch
import torch.nn as nn
from torch_geometric.nn.pool import radius_graph
from .utils import get_fully_connected, SGNNMessagePassingNetwork
from torch_scatter import scatter


class SGNN(nn.Module):
    def __init__(self, n_layer, p_step, s_dim, hidden_dim=128, activation=nn.SiLU(), cutoff=0.10, gravity_axis=None):
        super(SGNN, self).__init__()
        self.cutoff = cutoff
        # initialize the networks
        self.embedding = nn.Linear(s_dim, hidden_dim)
        self.embedding1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.embedding2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.obj_g_p = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, 1),
            )
        self.particle_g_p = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, 1),
            )
        self.gravity_axis = gravity_axis
        self.n_relation = -1
        self.eta = 0.001
        self.local_interaction = SGNNMessagePassingNetwork(n_layer=n_layer, p_step=p_step,
                                                           node_f_dim=3, node_s_dim=hidden_dim,
                                                           edge_f_dim=1, edge_s_dim=0, hidden_dim=hidden_dim,
                                                           activation=activation, edge_readout=True)
        self.object_message_passing = SGNNMessagePassingNetwork(n_layer=n_layer, p_step=p_step,
                                                                node_f_dim=1 if self.gravity_axis is None else 2,
                                                                node_s_dim=hidden_dim,
                                                                edge_f_dim=2, edge_s_dim=hidden_dim, hidden_dim=hidden_dim,
                                                                activation=activation)
        self.object_to_particle = SGNNMessagePassingNetwork(n_layer=n_layer, p_step=p_step,
                                                            node_f_dim=4 if self.gravity_axis is None else 6,
                                                            node_s_dim=hidden_dim,
                                                            edge_f_dim=1, edge_s_dim=0, hidden_dim=hidden_dim,
                                                            activation=activation)

    def build_graph(self, x_p):
        edge_index = radius_graph(x_p, r=self.cutoff, loop=False)  # [2, M]
        self.n_relation = edge_index.shape[1]
        return edge_index

    def predictor(self, x_p, f_p, s_p, obj_id, obj_type, num_obj):
        # return the predicted velocity for each particle
        ret = torch.zeros_like(f_p)[..., 0]  # [N, 3]
        for i in range(num_obj):
            if obj_type[i] == 'rigid':
                ret[obj_id == i] = f_p[obj_id == i][..., 0]
            elif obj_type[i] in ['fluid', 'cloth']:
                ret[obj_id == i] = f_p[obj_id == i][..., 1]
            else:
                raise RuntimeError('Unknown object material')
        return ret

    def forward(self, x_p, v_p, h_p, obj_id, obj_type):
        try:
            h_p[x_p[..., 1] < 0.1, -1] = 1
        except:
            pass
        s_p = self.embedding(h_p)  # [N, H]

        f_o = scatter(torch.stack((x_p, v_p), dim=-1), obj_id, dim=0, reduce='mean')  # [N_obj, 3, x]
        s_o = scatter(s_p, obj_id, dim=0) * self.eta  # [N_obj, H]

        edge_index = self.build_graph(x_p)
        edge_index_inner_mask = obj_id[edge_index[0]] == obj_id[edge_index[1]]
        edge_index_inter_mask = obj_id[edge_index[0]] != obj_id[edge_index[1]]
        edge_index_inner = edge_index[..., edge_index_inner_mask]  # [2, M_in]
        edge_index_inter = edge_index[..., edge_index_inter_mask]  # [2, M_out]

        f_p = torch.stack((x_p, v_p), dim=-1)  # [N, 3, 2]
        f_p = torch.cat((f_p - f_o[obj_id], v_p.unsqueeze(-1)), dim=-1)  # [N, 3, 3]
        s_p = torch.cat((s_o[obj_id], s_p), dim=-1)  # [N, 2H]
        s_p = self.embedding1(s_p)  # [N, H]
        edge_attr_inter_f = (x_p[edge_index_inter[0]] - x_p[edge_index_inter[1]]).unsqueeze(-1)  # [M_out, 3, 1]
        edge_attr_f, edge_attr_s = self.local_interaction(f_p, s_p, edge_index_inter, edge_attr_inter_f)
        # [M_out, H], [M_out, 3, x]

        if self.gravity_axis is not None:
            g_o = torch.zeros_like(f_o)[..., 0]  # [N_obj, 3]
            g_o[..., self.gravity_axis] = 1
            g_o = g_o * self.obj_g_p(s_o)
            f_o = torch.cat((f_o, g_o.unsqueeze(-1)), dim=-1)  # [N_obj, 3, x+1]

        num_obj = obj_id.max() + 1  # N_obj
        edge_index_o = get_fully_connected(num_obj, device=obj_id.device, loop=True)  # [2, M_obj]
        edge_mapping = obj_id[edge_index_inter[0]] * num_obj + obj_id[edge_index_inter[1]]  # [M_out]
        edge_attr_o_f = scatter(edge_attr_f, edge_mapping, dim=0, reduce='mean', dim_size=num_obj ** 2)  # [M_obj, 3, x]
        edge_attr_o_s = scatter(edge_attr_s, edge_mapping, dim=0, reduce='mean', dim_size=num_obj ** 2)  # [M_obj, H]
        edge_pseudo = torch.ones(edge_attr_s.shape[0]).to(edge_attr_s.device)  # [M_, 1]
        count = scatter(edge_pseudo, edge_mapping, dim=0, reduce='sum', dim_size=num_obj ** 2)
        mask = count > 0
        edge_index_o, edge_attr_o_f, edge_attr_o_s = edge_index_o[..., mask], edge_attr_o_f[mask], edge_attr_o_s[mask]
        f_o_, s_o_ = self.object_message_passing(f_o[..., 1:], s_o, edge_index_o, edge_attr_o_f, edge_attr_o_s)  # [N_obj, 3, 2]

        edge_attr_inner_f = (x_p[edge_index_inner[0]] - x_p[edge_index_inner[1]]).unsqueeze(-1)  # [M_in, 3, 1]
        f_p_ = torch.cat((f_o_[obj_id], f_p), dim=-1)
        s_p_ = torch.cat((s_o_[obj_id], s_p), dim=-1)
        s_p_ = self.embedding2(s_p_)  # [N, H]

        if self.gravity_axis is not None:
            g_p = torch.zeros_like(f_p_)[..., 0]  # [N, 3]
            g_p[..., self.gravity_axis] = 1
            g_p = g_p * self.particle_g_p(s_p_)
            f_p_ = torch.cat((f_p_, g_p.unsqueeze(-1)), dim=-1)  # [N_obj, 3, x+1]

        f_p_, s_p_ = self.object_to_particle(f_p_, s_p_, edge_index_inner, edge_attr_inner_f)  # [N, 3, x], [N, H]
        v_out = self.predictor(x_p, f_p_, s_p_, obj_id, obj_type, num_obj)  # [N, 3]
        return v_out



