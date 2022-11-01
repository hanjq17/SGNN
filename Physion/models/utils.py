from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter


def get_fully_connected(num_obj, device, loop=True):
    row = torch.arange(num_obj, dtype=torch.long, device=device)
    col = torch.arange(num_obj, dtype=torch.long, device=device)
    row = row.view(-1, 1).repeat(1, num_obj).view(-1)
    col = col.repeat(num_obj)
    edge_index = torch.stack([row, col], dim=0)
    if not loop:
        edge_index = remove_self_loops(edge_index)
    return edge_index


class BaseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, residual=False, last_act=False, flat=False):
        super(BaseMLP, self).__init__()
        self.residual = residual
        if flat:
            activation = nn.Tanh()
            hidden_dim = 4 * hidden_dim
        if residual:
            assert output_dim == input_dim
        if last_act:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, output_dim),
                activation
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x):
        return self.mlp(x) if not self.residual else self.mlp(x) + x


class SGNNMessagePassingLayer(nn.Module):
    def __init__(self, node_f_dim, node_s_dim, edge_f_dim, edge_s_dim, hidden_dim, activation):
        super(SGNNMessagePassingLayer, self).__init__()
        self.node_f_dim, self.node_s_dim, self.edge_f_dim, self.edge_s_dim = node_f_dim, node_s_dim, edge_f_dim, edge_s_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.net = BaseMLP(input_dim=(node_f_dim * 2 + edge_f_dim) ** 2 + node_s_dim * 2 + edge_s_dim,
                           hidden_dim=hidden_dim,
                           output_dim=(node_f_dim * 2 + edge_f_dim) * node_f_dim + node_s_dim,
                           activation=activation,
                           residual=False,
                           last_act=False,
                           flat=False)
        self.self_net = BaseMLP(input_dim=(node_f_dim * 2) ** 2 + node_s_dim * 2,
                                hidden_dim=hidden_dim,
                                output_dim=node_f_dim * 2 * node_f_dim + node_s_dim,
                                activation=activation,
                                residual=False,
                                last_act=False,
                                flat=False)

    def forward(self, f, s, edge_index, edge_f=None, edge_s=None):
        if edge_index.shape[1] == 0:
            f_c, s_c = torch.zeros_like(f), torch.zeros_like(s)
        else:
            _f = torch.cat((f[edge_index[0]], f[edge_index[1]]), dim=-1)
            if edge_f is not None:
                _f = torch.cat((_f, edge_f), dim=-1)  # [M, 3, 2F+Fe]
            _s = torch.cat((s[edge_index[0]], s[edge_index[1]]), dim=-1)
            if edge_s is not None:
                _s = torch.cat((_s, edge_s), dim=-1)  # [M, 2S]
            _f_T = _f.transpose(-1, -2)
            f2s = torch.einsum('bij,bjk->bik', _f_T, _f)  # [M, (2F+Fe), (2F+Fe)]
            f2s = f2s.reshape(f2s.shape[0], -1)  # [M, (2F+Fe)*(2F+Fe)]
            f2s = F.normalize(f2s, p=2, dim=-1)
            f2s = torch.cat((f2s, _s), dim=-1)  # [M, (2F+Fe)*(2F+Fe)+2S+Se]
            c = self.net(f2s)  # [M, (2F+Fe)*F+H]
            # c = scatter(c, edge_index[0], dim=0, reduce='mean', dim_size=f.shape[0])  # [N, (2F+Fe)*F+H]
            f_c, s_c = c[..., :-self.hidden_dim], c[..., -self.hidden_dim:]  # [M, (2F+Fe)*F], [M, H]
            f_c = f_c.reshape(f_c.shape[0], _f.shape[-1], -1)  # [M, 2F+Fe, F]
            f_c = torch.einsum('bij,bjk->bik', _f, f_c)  # [M, 3, F]
            f_c = scatter(f_c, edge_index[0], dim=0, reduce='mean', dim_size=f.shape[0])  # [N, 3, F]
            s_c = scatter(s_c, edge_index[0], dim=0, reduce='mean', dim_size=f.shape[0])  # [N, H]
        # aggregate f_c and f
        temp_f = torch.cat((f, f_c), dim=-1)  # [N, 3, 2F]
        temp_f_T = temp_f.transpose(-1, -2)  # [N, 2F, 3]
        temp_f2s = torch.einsum('bij,bjk->bik', temp_f_T, temp_f)  # [N, 2F, 2F]
        temp_f2s = temp_f2s.reshape(temp_f2s.shape[0], -1)  # [N, 2F*2F]
        temp_f2s = F.normalize(temp_f2s, p=2, dim=-1)
        temp_f2s = torch.cat((temp_f2s, s, s_c), dim=-1)  # [N, 2F*2F+2S]
        temp_c = self.self_net(temp_f2s)  # [N, 2F*F+H]
        temp_f_c, temp_s_c = temp_c[..., :-self.hidden_dim], temp_c[..., -self.hidden_dim:]  # [N, 2F*F], [N, H]
        temp_f_c = temp_f_c.reshape(temp_f_c.shape[0], temp_f.shape[-1], -1)  # [N, 2F, F]
        temp_f_c = torch.einsum('bij,bjk->bik', temp_f, temp_f_c)  # [N, 3, F]
        f_out = temp_f_c
        s_out = temp_s_c
        return f_out, s_out
    

class SGNNEdgeReadoutLayer(nn.Module):
    def __init__(self, node_f_dim, node_s_dim, edge_f_dim, edge_s_dim, hidden_dim, activation, output_f_dim):
        super(SGNNEdgeReadoutLayer, self).__init__()
        self.node_f_dim, self.node_s_dim, self.edge_f_dim, self.edge_s_dim = node_f_dim, node_s_dim, edge_f_dim, edge_s_dim
        self.hidden_dim = hidden_dim
        self.output_f_dim = output_f_dim
        self.net = BaseMLP(input_dim=(node_f_dim * 2 + edge_f_dim) ** 2 + node_s_dim * 2 + edge_s_dim,
                           hidden_dim=hidden_dim,
                           output_dim=(node_f_dim * 2 + edge_f_dim) * output_f_dim + node_s_dim,
                           activation=activation,
                           residual=False,
                           last_act=False,
                           flat=False)

    def forward(self, f, s, edge_index, edge_f=None, edge_s=None):
        if edge_index.shape[1] == 0:
            f_c, s_c = torch.zeros(0, 3, self.output_f_dim).to(f.device), torch.zeros(0, self.node_s_dim).to(s.device)
            return f_c, s_c
        _f = torch.cat((f[edge_index[0]], f[edge_index[1]]), dim=-1)
        if edge_f is not None:
            _f = torch.cat((_f, edge_f), dim=-1)  # [M, 3, 2F+Fe]
        _s = torch.cat((s[edge_index[0]], s[edge_index[1]]), dim=-1)
        if edge_s is not None:
            _s = torch.cat((_s, edge_s), dim=-1)  # [M, 2S]
        _f_T = _f.transpose(-1, -2)
        f2s = torch.einsum('bij,bjk->bik', _f_T, _f)  # [M, (2F+Fe), (2F+Fe)]
        f2s = f2s.reshape(f2s.shape[0], -1)  # [M, (2F+Fe)*(2F+Fe)]
        f2s = F.normalize(f2s, p=2, dim=-1)
        f2s = torch.cat((f2s, _s), dim=-1)  # [M, (2F+Fe)*(2F+Fe)+2S+Se]
        c = self.net(f2s)  # [M, (2F+Fe)*F_out+H]
        f_c, s_c = c[..., :-self.hidden_dim], c[..., -self.hidden_dim:]  # [M, (2F+Fe)*F_out], [M, H]
        f_c = f_c.reshape(f_c.shape[0], _f.shape[-1], -1)  # [M, 2F+Fe, F_out]
        f_c = torch.einsum('bij,bjk->bik', _f, f_c)  # [M, 3, F_out]
        return f_c, s_c


class SGNNMessagePassingNetwork(nn.Module):
    def __init__(self, n_layer, p_step, node_f_dim, node_s_dim, edge_f_dim, edge_s_dim, hidden_dim,
                 activation, edge_readout=False):
        super(SGNNMessagePassingNetwork, self).__init__()
        self.networks = nn.ModuleList()
        self.n_layer = n_layer
        self.p_step = p_step
        for i in range(self.n_layer):
            self.networks.append(SGNNMessagePassingLayer(node_f_dim, node_s_dim, edge_f_dim,
                                                           edge_s_dim, hidden_dim, activation))
        self.edge_readout = edge_readout
        if edge_readout:
            self.readout = SGNNEdgeReadoutLayer(node_f_dim, node_s_dim, edge_f_dim, edge_s_dim, hidden_dim,
                                                  activation, output_f_dim=2)

    def forward(self, f, s, edge_index, edge_f=None, edge_s=None):
        for i in range(self.p_step):
            f, s = self.networks[0](f, s, edge_index, edge_f, edge_s)
        if self.edge_readout:  # edge-level readout
            fe, se = self.readout(f, s, edge_index, edge_f, edge_s)
            return fe, se
        else:  # node-level output
            return f, s

