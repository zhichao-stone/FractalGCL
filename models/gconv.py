import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, global_add_pool



class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(self.make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(self.make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim)
        )

    def make_gin_conv(self, input_dim, out_dim):
        return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))

    def forward(self, x, edge_index, batch, project: bool = False):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        g = torch.cat([global_add_pool(z, batch) for z in zs], dim=1)

        if project:
            g: torch.Tensor = self.project(g)

        return g


class ConcatModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ConcatModel, self).__init__()
        self.gconv = GConv(input_dim, hidden_dim, num_layers)

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim * 2, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim)
        )

    def forward(self, x1, edge_index1, batch1, x2, edge_index2, batch2):
        g1 = self.gconv(x1, edge_index1, batch1, project=True)
        g2 = self.gconv(x2, edge_index2, batch2, project=True)

        g: torch.Tensor = self.project(torch.cat([g1, g2], dim=-1))
        return g