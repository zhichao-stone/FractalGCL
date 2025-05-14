import random
from typing import List, Dict, Tuple

import numpy as np
import networkx as nx
import torch
import GCL.augmentors as A

from .fractal import compute_box_dimension



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
    center_nodes: np.ndarray = np.zeros(num_nodes, dtype=int)
    # adjacency matrix for N-hop
    Adj = adj.clone().to(device)
    N_Adj = nadj.clone().int().to(device) | torch.eye(num_nodes, dtype=torch.int32, device=device)

    all_nodes = list(range(num_nodes))
    remaining_nodes = set(all_nodes)
    num_supernodes = 0

    while remaining_nodes:
        c = random.choice(list(remaining_nodes))
        balls = torch.where(N_Adj[c])[0].cpu().tolist()
        center_nodes[balls] = num_supernodes

        remaining_nodes = remaining_nodes.difference(balls)
        num_supernodes += 1

        N_Adj[balls, :] = 0
        N_Adj[:, balls] = 0

    # calculate features of supernodes
    supernodes_features = torch.zeros((num_supernodes, x.size(-1)), device=device)
    for n, c in enumerate(center_nodes):
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
        edges, batches = [], []
        bias = 0
        for i in range(len(batch_x)):
            graph_size = batch_x[i].size(0)
            edges.append(batch_edge_index[i] + bias)
            batches.append((torch.ones(graph_size) * i).long().to(self.device))
            bias += graph_size
        x:torch.Tensor = torch.cat(batch_x, dim=0)
        edge_index, batch = torch.cat(edges, dim=-1), torch.cat(batches, dim=-1)
        return x.to(self.device), edge_index.to(self.device), batch.to(self.device)

    def split_batch(self, x: torch.FloatTensor, edge_index: torch.LongTensor, batch: torch.Tensor, batch_size: int):
        batch_x_list: List[torch.Tensor] = []
        batch_edge_index_list: List[torch.Tensor] = []
        for i in range(batch_size):
            index = torch.where(batch == i)[0].tolist()
            min_node, max_node = index[0], index[-1]
            batch_x_list.append(x[index].to(self.device))
            indices = torch.nonzero((edge_index[0] >= min_node) & (edge_index[0] <= max_node)).squeeze().tolist()
            batch_edge: torch.Tensor = edge_index.T[indices] - min_node
            batch_edge_index_list.append(batch_edge.T.to(self.device))
        return batch_x_list, batch_edge_index_list

    def calculate_nadj(self, gid: int, edge_index: torch.Tensor, max_r: int):
        if gid not in self.nadjs:
            num_nodes = edge_index.max().item() + 1
            values = torch.ones(edge_index.size(-1)).to(self.device)
            adj = torch.sparse_coo_tensor(edge_index, values, size=(num_nodes, num_nodes)).to_dense().int().to(self.device)
            adj = (adj | adj.T).float() # undirected, 2025-05-14
            self.nadjs[gid] = [adj]

        current_r = len(self.nadjs[gid]) - 1
        if current_r < max_r - 1:
            Adj = self.nadjs[gid][0]
            N_Adj = self.nadjs[gid][current_r].clone().to(self.device)
            while current_r < max_r - 1:
                N_Adj = torch.matmul(N_Adj, Adj) + N_Adj
                self.nadjs[gid].append(N_Adj.clone().to(self.device))
                current_r += 1

    def compute_aug_diameter_dimension(self, edge_index: torch.Tensor):
        G = nx.Graph(edge_index.T.int().tolist())
        if G.number_of_edges() == 0:
            aug_diameter, aug_dimension = 0, 0.0
        else:
            lcc = max(nx.connected_components(nx.to_undirected(G.copy())), key=len)
            lccG: nx.Graph = G.subgraph(lcc)
            aug_diameter: int = nx.diameter(lccG)
            _, aug_dimension = compute_box_dimension(G, aug_diameter, self.device)
        return aug_diameter, aug_dimension

    def merge_graph(self, 
        x1: torch.Tensor, edge_index1: torch.Tensor, 
        x2: torch.Tensor, edge_index2: torch.Tensor, 
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ns1 = int(x1.size(0))
        x1, x2 = x1.to(self.device), x2.to(self.device)
        edge_index1, edge_index2 = edge_index1.to(self.device), edge_index2.to(self.device)

        x = torch.cat([x1, x2], dim=0)
        edge_index = torch.cat([edge_index1, edge_index2 + ns1], dim=-1)
        return x, edge_index

    def augment(self, 
        x: torch.FloatTensor,
        edge_index: torch.LongTensor, 
        batch: torch.LongTensor, 
        fractalities: torch.Tensor, 
        diameters: torch.Tensor, 
        dimensions: torch.Tensor, 
        gids: torch.Tensor, 
        aug_type: str, 
        aug_num: int = 2, 
        compute_dimension: bool = False, 
        merge_graph: bool = False, 
    ):  
        aug_graphs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        aug_dimensions: List[torch.Tensor] = [dimensions.clone() for _ in range(aug_num)]
        aug_diameters: List[torch.Tensor] = [diameters.clone() for _ in range(aug_num)]

        if aug_type in self.non_fractal_aug_types:
            if aug_type == "drop_node":
                augmentor = A.NodeDropping(self.drop_ratio)
            elif aug_type == "simple_random_walk":
                augmentor = A.RWSampling(num_seeds=1000, walk_length=10)
            else:
                raise NotImplementedError(f"Augmentation method {aug_type} is not supported!")

            for i in range(aug_num):
                aug_x, aug_edge_index = augmentor(x, edge_index)[:2]
                aug_graphs.append((aug_x.to(self.device), aug_edge_index.to(self.device), batch.clone().to(self.device)))

            if compute_dimension:
                aug_dimensions = [torch.zeros_like(dimensions) for _ in range(aug_num)]

            return aug_graphs, aug_dimensions, aug_diameters
            
        else:
            aug_xs: List[List[torch.Tensor]] = [[] for _ in range(aug_num)]
            aug_edge_indexs: List[List[torch.Tensor]] = [[] for _ in range(aug_num)]

            # split whole batch graph into graphs according to batch
            batch_size = len(diameters)
            batch_x_list, batch_edge_index_list = self.split_batch(x, edge_index, batch, batch_size)

            if aug_type in ["renorm_rc_prob", "mix_sep"]:
                for bi in range(batch_size):
                    batch_x, batch_edge_index = batch_x_list[bi], batch_edge_index_list[bi]
                    gid, r2, diameter = int(gids[bi]), fractalities[bi], int(diameters[bi])
                    if r2 > 0.0:
                        radius = 1
                        self.calculate_nadj(gid, batch_edge_index, radius)
                        adj, nadj = self.nadjs[gid][0], self.nadjs[gid][radius - 1]

                        prob = r2 if aug_type == "renorm_rc_prob" else 0.5
                        if random.random() < prob:
                            for i in range(aug_num):
                                aug_x, aug_edge_index = renormalization_graph_random_center(batch_x, batch_edge_index, adj, nadj, radius, self.device, self.renorm_min_edges)
                                if merge_graph:
                                    aug_x, aug_edge_index = self.merge_graph(aug_x, aug_edge_index, batch_x, batch_edge_index)

                                aug_xs[i].append(aug_x.to(self.device))
                                aug_edge_indexs[i].append(aug_edge_index.to(self.device))
                        else:
                            for i in range(aug_num):
                                aug_x, aug_edge_index = self.default_aug(batch_x, batch_edge_index)[:2]
                                aug_xs[i].append(aug_x.to(self.device))
                                aug_edge_indexs[i].append(aug_edge_index.to(self.device))
                    else:
                        for i in range(aug_num):
                            aug_x, aug_edge_index = self.default_aug(batch_x, batch_edge_index)[:2]
                            aug_xs[i].append(aug_x.to(self.device))
                            aug_edge_indexs[i].append(aug_edge_index.to(self.device))

            elif aug_type in ["renorm_rc", "renorm_rc_rr"]:

                for bi in range(batch_size):
                    batch_x, batch_edge_index = batch_x_list[bi], batch_edge_index_list[bi]
                    gid, r2, diameter = int(gids[bi]), fractalities[bi], int(diameters[bi])

                    if r2 >= self.aug_fractal_threshold:
                        if aug_type == "renorm_rc_rr":
                            radius = random.choice(list(range(1, max(2, (diameter//2)//2))))
                        elif aug_type == "renorm_rc":
                            radius = 1
                        else:
                            raise NotImplementedError(f"Augmentation method {aug_type} is not supported!")

                        self.calculate_nadj(gid, batch_edge_index, radius)
                        adj, nadj = self.nadjs[gid][0], self.nadjs[gid][radius-1]

                        for i in range(aug_num):
                            aug_x, aug_edge_index = renormalization_graph_random_center(batch_x, batch_edge_index, adj, nadj, radius, self.device, self.renorm_min_edges)
                            if merge_graph:
                                aug_x, aug_edge_index = self.merge_graph(aug_x, aug_edge_index, batch_x, batch_edge_index)

                            aug_xs[i].append(aug_x.to(self.device))
                            aug_edge_indexs[i].append(aug_edge_index.to(self.device))
                    else:
                        for i in range(aug_num):
                            aug_x, aug_edge_index = self.default_aug(batch_x, batch_edge_index)[:2]
                            aug_xs[i].append(aug_x.to(self.device))
                            aug_edge_indexs[i].append(aug_edge_index.to(self.device))
            else:
                raise NotImplementedError(f"Augmentation method {aug_type} is not supported!")

            if compute_dimension:
                for i in range(aug_num):
                    for bi in range(batch_size):
                        aug_diameter, aug_dimension = self.compute_aug_diameter_dimension(aug_edge_indexs[i][bi])
                        aug_diameters[i][bi] = aug_diameter
                        aug_dimensions[i][bi] = aug_dimension

            for i in range(aug_num):
                aug_x, aug_edge_index, aug_batch = self.merge_batch(aug_xs[i], aug_edge_indexs[i])
                aug_graphs.append((aug_x, aug_edge_index, aug_batch))

            return aug_graphs, aug_dimensions, aug_diameters