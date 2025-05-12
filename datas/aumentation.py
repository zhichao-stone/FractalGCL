import random
from typing import List, Dict, Tuple

import numpy as np
import torch
import GCL.augmentors as A



def renormalization_graph_random_center(
    x: torch.Tensor, 
    edge_index: torch.Tensor,  
    adj: torch.Tensor, 
    nadj: torch.Tensor, 
    radius: int, 
    device: torch.device, 
    min_edges: int = 1
):
    if radius <= 0:
        return x, edge_index
    
    num_nodes = edge_index.max().item() + 1

    # cluster supernodes
    center_nodes = np.zeros(num_nodes, dtype=int)

    # values = torch.ones(edge_index.size(-1))
    # Adj = torch.sparse_coo_tensor(edge_index, values, size=(num_nodes, num_nodes)).to(device)
    # N_Adj = Adj.clone().to(device)
    # for _ in range(1, radius):
    #     N_Adj = torch.matmul(N_Adj, Adj) + N_Adj
    # N_Adj = N_Adj.to_dense().int()
    # N_Adj = N_Adj | torch.eye(num_nodes, dtype=torch.int32, device=device)
    Adj = adj.clone().to(device)
    N_Adj = nadj.clone().to_dense().int().to(device) | torch.eye(num_nodes, dtype=torch.int32, device=device)

    all_nodes = list(range(num_nodes))
    remaining_nodes = set(all_nodes)
    num_supernodes = 0
    # supernode_size = []

    while remaining_nodes:
        c = random.choice(list(remaining_nodes))

        balls = torch.where(N_Adj[c])[0].cpu().tolist()
        center_nodes[balls] = num_supernodes

        remaining_nodes = remaining_nodes.difference(balls)

        # supernode_size.append(len(balls))
        num_supernodes += 1

        N_Adj[balls, :] = 0
        N_Adj[:, balls] = 0

    # calculate features of supernodes
    supernodes_features = torch.zeros((num_supernodes, x.size(-1)), device=device)
    for n, c in enumerate(center_nodes):
        # supernodes_features[c] += graph.ndata["feat"][n] / supernode_size[c]
        supernodes_features[c] += x[n]

    # calculate supernode edges
    S = torch.sparse_coo_tensor(torch.tensor([center_nodes.tolist(), all_nodes]), torch.ones(num_nodes), size=(num_supernodes, num_nodes)).to(device)
    A = torch.matmul(torch.matmul(S, Adj), S.T).to_dense()
    A = A - torch.diag_embed(A.diag()).to(device)
    renorm_edges = torch.where(A >= min_edges)

    renorm_edges = torch.stack(renorm_edges)

    return supernodes_features, renorm_edges


class FractalAugmentor:
    def __init__(self, 
        drop_ratio: float = 0.2, 
        aug_fractal_threshold: float = 0.95, 
        renorm_min_edges: int = 1, 
        device: torch.device = torch.device("cuda")
    ) -> None:
        self.drop_ratio = drop_ratio
        self.aug_fractal_threshold = aug_fractal_threshold
        self.renorm_min_edges = renorm_min_edges
        self.device = device

        self.default_aug = A.NodeDropping(self.drop_ratio)
        self.non_fractal_aug_types = ["drop_node", "simple_random_walk"]
        self.nadjs: Dict[int, List[torch.Tensor]] = {}

    def merge_batch(self, batch_x: List[torch.FloatTensor], batch_edge_index: List[torch.LongTensor]):
        edge_index, batch = torch.tensor(batch_edge_index[0]), [0 for _ in range(batch_x[0].size()[0])]
        bias = len(batch_x[0])
        for i in range(1, len(batch_x)):
            batch += [i for _ in range(batch_x[i].size()[0])]
            edge_index = torch.cat((edge_index, batch_edge_index[i]+bias), dim=-1)
            bias += len(batch_x[i])
        x = torch.cat(batch_x)
        return x, edge_index, torch.tensor(batch, dtype=torch.int64)

    def calculate_nadj(self, gid: int, edge_index: torch.Tensor, max_r: int):
        if gid not in self.nadjs:
            num_nodes = edge_index.max().item() + 1
            values = torch.ones(edge_index.size(-1)).to(self.device)
            adj = torch.sparse_coo_tensor(edge_index, values, size=(num_nodes, num_nodes)).to(self.device)
            self.nadjs[gid] = [adj]

        current_r = len(self.nadjs[gid]) - 1
        if current_r < max_r - 1:
            Adj = self.nadjs[gid][0]
            N_Adj = self.nadjs[gid][current_r].clone().to_dense().to(self.device)
            while current_r < max_r - 1:
                N_Adj = torch.matmul(N_Adj, Adj) + N_Adj
                self.nadjs[gid].append(N_Adj.clone().to(self.device))
                current_r += 1

    def augment(self, 
        x: torch.FloatTensor,
        edge_index: torch.LongTensor, 
        batch: torch.LongTensor, 
        fractalities: List[float], 
        diameters: List[int], 
        gids: List[int], 
        aug_type: str, 
        aug_num: int = 2, 
    ):
        batch_size = len(diameters)
        batch_x: List[torch.Tensor] = []
        batch_edge_index: List[torch.Tensor] = []
        for i in range(batch_size):
            index = torch.where(batch == i)[0].tolist()
            min_node, max_node = index[0], index[-1]
            batch_x.append(x[index].to(self.device))
            indices = torch.nonzero((edge_index[0] >= min_node) & (edge_index[0] <= max_node)).squeeze().tolist()
            batch_edge: torch.Tensor = edge_index.T[indices] - min_node
            batch_edge_index.append(batch_edge.T.to(self.device))

        aug_xs, aug_edge_indexs = [[] for _ in range(aug_num)], [[] for _ in range(aug_num)]
        for i in range(batch_size):
            x, edge_index = batch_x[i], batch_edge_index[i]
            r2, diameter = fractalities[i], diameters[i]
            gid = gids[i]

            if aug_type not in self.non_fractal_aug_types:
                if r2 > 0.0 and aug_type in ["renorm_rc_prob", "mix_sep"]:
                    radius = 1
                    self.calculate_nadj(gid, edge_index, radius)
                    adj, nadj = self.nadjs[gid][0], self.nadjs[gid][radius-1]

                    prob = r2 if aug_type == "renorm_rc_prob" else 0.5
                    if random.random() < prob:
                        for i in range(aug_num):
                            aug_x, aug_edge_index = renormalization_graph_random_center(x, edge_index, adj, nadj, radius, self.device, self.renorm_min_edges)
                            aug_xs[i].append(aug_x.to(self.device))
                            aug_edge_indexs[i].append(aug_edge_index.to(self.device))
                    else:
                        for i in range(aug_num):
                            aug_x, aug_edge_index = self.default_aug(x, edge_index)[:2]
                            aug_xs[i].append(aug_x.to(self.device))
                            aug_edge_indexs[i].append(aug_edge_index.to(self.device))
                else:
                    if r2 >= self.aug_fractal_threshold:
                        if aug_type == "renorm_rc_rr":
                            radius = random.choice(list(range(1, max(2, (diameter//2)//2))))
                        elif aug_type == "renorm_rc":
                            radius = 1
                        else:
                            raise NotImplementedError(f"Augmentation method {aug_type} is not supported!")

                        self.calculate_nadj(gid, edge_index, radius)
                        adj, nadj = self.nadjs[gid][0], self.nadjs[gid][radius-1]

                        for i in range(aug_num):
                            aug_x, aug_edge_index = renormalization_graph_random_center(x, edge_index, adj, nadj, radius, self.device, self.renorm_min_edges)
                            aug_xs[i].append(aug_x.to(self.device))
                            aug_edge_indexs[i].append(aug_edge_index.to(self.device))
                    else:
                        for i in range(aug_num):
                            aug_x, aug_edge_index = self.default_aug(x, edge_index)[:2]
                            aug_xs[i].append(aug_x.to(self.device))
                            aug_edge_indexs[i].append(aug_edge_index.to(self.device))
            else:
                if aug_type == "drop_node":
                    augmentor = A.NodeDropping(self.drop_ratio)
                elif aug_type == "simple_random_walk":
                    augmentor = A.RWSampling(num_seeds=1000, walk_length=10)
                else:
                    raise NotImplementedError(f"Augmentation method {aug_type} is not supported!")

                for i in range(aug_num):
                    aug_x, aug_edge_index = augmentor(x, edge_index)[:2]
                    aug_xs[i].append(aug_x.to(self.device))
                    aug_edge_indexs[i].append(aug_edge_index.to(self.device))

        aug_graphs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for i in range(aug_num):
            aug_x, aug_edge_index, aug_batch = self.merge_batch(aug_xs[i], aug_edge_indexs[i])
            aug_graphs.append((aug_x, aug_edge_index, aug_batch.to(self.device)))
        return aug_graphs