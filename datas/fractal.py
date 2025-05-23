import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



def get_single_center_covers(G: nx.Graph, nadj: torch.Tensor, device: torch.device = torch.device("cuda")):
    '''
    Use box counting algorithm to get covers that each cover has only one center.
    '''
    box_count = 0
    radj = nadj.clone().to(device)
    uncovered = set(G.nodes)
    while uncovered:
        cov_size = torch.sum(radj, dim=-1)
        if torch.max(cov_size) <= 1:
            break
        center = int(torch.argmax(cov_size).cpu())
        cover = torch.where(radj[center])[0].cpu().tolist()
        radj[cover, :] = 0
        radj[:, cover] = 0
        cover = set(cover)
        uncovered -= cover
        box_count += 1
    box_count += len(uncovered)
    return box_count


def get_double_center_covers_by_networkx(G: nx.Graph, radius: int):
    '''
    Use box counting algorithm to get covers that each cover has two adjacent centers.
    Use function of networkx.
    '''
    box_count = 0
    uncovered = set(G.nodes)
    while uncovered:
        best_pair, best_union = None, set()
        for u in uncovered:
            for v in G.neighbors(u):
                if v not in uncovered:
                    continue
                cov_u = set(nx.single_source_shortest_path_length(G, u, cutoff=radius).keys())
                cov_v = set(nx.single_source_shortest_path_length(G, v, cutoff=radius).keys())
                union_cov = (cov_u | cov_v) & uncovered
                if len(union_cov) > len(best_union):
                    best_pair, best_union = [u, v], union_cov

        if best_pair:
            uncovered -= best_union
            box_count += 1
        else:
            best_cov = set()
            for node in uncovered:
                cov = set(nx.single_source_shortest_path_length(G, node, cutoff=radius).keys()) & uncovered
                if len(cov) > len(best_cov):
                    best_cov = cov
            uncovered -= best_cov
            box_count += 1
            if len(best_cov) <= 1:
                break
    box_count += len(uncovered)
    return box_count


def get_double_center_covers_by_matrix(G: nx.Graph, nadj: torch.Tensor, device: torch.device = torch.device("cuda")):
    '''
    Use box counting algorithm to get covers that each cover has two adjacent centers.
    Use Torch's matrix multiplication operation for acceleration.
    '''
    box_count = 0
    radj = nadj.clone().to(device)
    uncovered = set(G.nodes)
    while uncovered:
        best_pair, best_cov_size = None, 0
        for u in uncovered:
            v_list = [v for v in G.neighbors(u) if v in uncovered]
            if v_list:
                cov_mat = radj[v_list, :] | radj[u]
                cov_size = torch.sum(cov_mat, dim=-1)
                v_idx = int(torch.argmax(cov_size).cpu())
                v = v_list[v_idx]
                if cov_size[v_idx] > best_cov_size:
                    best_pair, best_cov_size = [u, v], int(cov_size[v_idx])
        
        if best_pair:
            u, v = best_pair
            best_union = torch.where(radj[u] | radj[v])[0].cpu().tolist()
            radj[best_union, :] = 0
            radj[:, best_union] = 0
            best_union = set(best_union)
            uncovered -= best_union
            box_count += 1
        else:
            cov_size = torch.sum(radj, dim=-1)
            if torch.max(cov_size) <= 1:
                break
            center = int(torch.argmax(cov_size).cpu())
            cover = torch.where(radj[center])[0].cpu().tolist()
            radj[cover, :] = 0
            radj[:, cover] = 0
            cover = set(cover)
            uncovered -= cover
            box_count += 1
    box_count += len(uncovered)
    return box_count


def compute_box_dimension(G: nx.Graph, diameter: int, device: torch.device = torch.device("cuda")):
    # calculate adjacency matrix
    G = G.copy().to_undirected()
    edges = torch.tensor(list(G.edges())).T
    num_nodes = int(torch.max(edges)) + 1
    values = torch.ones(G.number_of_edges())

    I = torch.eye(num_nodes, dtype=torch.int32, device=device)
    adj = torch.sparse_coo_tensor(edges, values, size=(num_nodes, num_nodes)).to_dense().int().to(device)
    adj = (adj | adj.T).float()
    nadj = adj.clone().to(device)
    current_r = 1

    # compute box dimension
    max_d = max(1, diameter//2)
    d_values, N_box_values = [], []

    # for d in range(2, max_d+1):
    for d in range(1, max_d + 1):
        r = d // 2
        while current_r < r:
            nadj = torch.matmul(nadj, adj) + nadj
            current_r += 1
        nadj[nadj > 0] = 1
        radj = nadj.clone().to_dense().int().to(device) | I

        if d & 1 == 0:  # When the diameter is even, the box is centered around a single node.
            box_count = get_single_center_covers(G, radj, device)
        else:   # When the diameter is odd, the box is centered around two adjacent nodes
            if r <= 5:
                box_count = get_double_center_covers_by_networkx(G, r)
            else:
                box_count = get_double_center_covers_by_matrix(G, radj, device)

        d_values.append(d)
        N_box_values.append(box_count)

    if len(d_values) < 2:
        return 0.0, 0.0
    
    # linear regression
    log_l = np.log(np.array(d_values)).reshape(-1, 1)
    log_N_box = np.log(np.array(N_box_values))

    reg = LinearRegression().fit(log_l, log_N_box)
    box_dimension = - float(reg.coef_[0])
    r2 = float(reg.score(log_l, log_N_box))
    
    return r2, box_dimension