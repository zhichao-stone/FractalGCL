import os
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_single_center_covers(G: nx.Graph, nadj: torch.Tensor):
    box_count = 0
    radj = nadj.clone().to(DEVICE)
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

def get_double_center_covers_by_matrix(G: nx.Graph, nadj: torch.Tensor):
    box_count = 0
    radj = nadj.clone().to(DEVICE)
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


def compute_box_dimension(G: nx.Graph, diameter: int, plot_path: str = ""):
    # calculate adjacency matrix
    edges = torch.tensor(list(G.edges())).T
    num_nodes = int(torch.max(edges)) + 1
    values = torch.ones(G.number_of_edges())

    I = torch.eye(num_nodes, dtype=torch.int32, device=DEVICE)
    adj = torch.sparse_coo_tensor(edges, values, size=(num_nodes, num_nodes)).to(DEVICE)
    nadj = adj.clone().to_dense().to(DEVICE)
    current_r = 1

    # compute box dimension
    max_d = max(1, diameter//2)
    d_values, N_box_values = [], []

    for d in range(2, max_d+1):
        r = d // 2
        while current_r < r:
            nadj = torch.matmul(nadj, adj) + nadj
            current_r += 1
        nadj[nadj > 0] = 1

        radj = nadj.clone().to_dense().int().to(DEVICE)
        radj = radj | I

        if d & 1 == 0:  # single center
            box_count = get_single_center_covers(G, radj)
        else:           # double center
            if r <= 5:
                box_count = get_double_center_covers_by_networkx(G, r)
            else:
                box_count = get_double_center_covers_by_matrix(G, radj)

        d_values.append(d)
        N_box_values.append(box_count)

    if len(d_values) < 2:
        return 0.0, 0.0, 0.0, 0.0
    
    log_l = np.log(np.array(d_values)).reshape(-1, 1)
    log_N_box = np.log(np.array(N_box_values))

    reg = LinearRegression().fit(log_l, log_N_box)
    m, b = reg.coef_[0], reg.intercept_
    R2 = reg.score(log_l, log_N_box)
    box_dimension = -m
    G_logy_fit_1 = m * log_l.flatten() + b

    if plot_path:
        plt.figure(figsize=(10, 6))
        plt.loglog(d_values, N_box_values, "o", label="Data points")
        plt.loglog(d_values, np.exp(G_logy_fit_1), label=f"slope={m:.2f}\nR²={R2:.2f}", linestyle="--")
        plt.xlabel("X (log scale)")
        plt.ylabel("Y (log scale)")
        plt.title(f"Log-Log Plot, R²={R2:.3f}, dim_B={box_dimension:.3f}")
        plt.savefig(plot_path)
        plt.clf()
    
    return m, b, R2, box_dimension


def get_fractal_dimension(G:nx.Graph, count_diameter_less_nine: bool = True, plot_path: str = ""):
    # get the largest connected subgraph
    largest_cc = max(nx.connected_components(nx.to_undirected(G.copy())), key=len)
    lcc_G:nx.Graph = G.subgraph(largest_cc).copy()
    diameter = nx.diameter(lcc_G)

    # get fractal dimension
    if not count_diameter_less_nine and diameter <= 9:
        regression_1 = {"Can Test Fractality": False}
    else:
        slope_1, intercept_1, R2_1, box_dimension_1 = compute_box_dimension(G, diameter, plot_path)
        if box_dimension_1 > 0:
            regression_1 = {"Slope": slope_1, "Intercept": intercept_1, "R²": R2_1, "Box_Dimension": box_dimension_1}
        else:
            regression_1 = {"Can Test Fractality": False}

    regression_result = {
        "Statistics of Graph": {
            "Nodes": G.number_of_nodes(),
            "Edges": G.number_of_edges(),
            "Diameter": diameter,
        },
        "Linear Regression": {
            "Origin Graph": regression_1
        }
    }

    return regression_result



if __name__ == "__main__":
    import json
    import argparse
    from torch_geometric.datasets import TUDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="citeseer")
    parser.add_argument("--renew", action="store_true")
    args = parser.parse_args()
    DATA = str(args.data).upper()
    save_dir = "fractal_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"linear_regression_{DATA.lower()}.json")

    if not os.path.exists(save_path) or args.renew:
        root = os.path.join(os.path.expanduser("~"), "datasets")
        dataset = TUDataset(root, name=DATA)
        num_classes = dataset.num_classes

        # get fractal dimensions
        from tqdm import tqdm
        print(f"number of graphs: {len(dataset)} , number of classes: {num_classes}")
        regression_results = []
        index = 0

        for g in tqdm(dataset):
            edge_index: torch.Tensor = g.edge_index.T
            G = nx.Graph(edge_index.int().tolist())
            result = get_fractal_dimension(G)
            regression_results.append(result)
            index += 1

        with open(save_path, "w", encoding="utf-8") as fw:
            json.dump(regression_results, fw, ensure_ascii=False, indent=4)
    else:
        with open(save_path, "r", encoding="utf-8") as fr:
            regression_results = json.load(fr)

    # statistic
    total = len(regression_results)
    is_fractal_num = 0
    r2_distribution = {f"{k*0.05:.2f}": 0 for k in range(1, 20)}
    diameter_distribution = {}
    for r in regression_results:
        res = r["Linear Regression"]["Origin Graph"]
        d = r["Statistics of Graph"]["Diameter"]
        if d not in diameter_distribution:
            diameter_distribution[d] = 1
        else:
            diameter_distribution[d] += 1
        if "R²" in res:
            is_fractal_num += 1
            r2_interval = int(abs(res["R²"]) / 0.05)
            if r2_interval == 20:
                r2_interval -= 1
            for k in range(1, r2_interval+1):
                r2_distribution[f"{k*0.05:.2f}"] += 1
    
    statistic = {
        "total graphs": total, 
        "proportion of fractal graphs": f"num={is_fractal_num} | proportion={is_fractal_num/total:.2%}", 
        "help": "The following distribution's format: \"Key: Number | Proportion\"", 
        "fractality distribution": {
            f">= {k}": f"{v:4d} | {v/total:.2%}" for k, v in r2_distribution.items()
        }, 
        "diameter distribution": {
            f"{k:3d}": f"{diameter_distribution[k]:4d} | {diameter_distribution[k]/total:.2%}" for k in sorted(list(diameter_distribution.keys()))
        }
    }
    statistic_dir = os.path.join("fractal_results", "statistic")
    if not os.path.exists(statistic_dir):
        os.makedirs(statistic_dir)
    with open(os.path.join(statistic_dir, f"statistic_{DATA.lower()}.json"), "w", encoding="utf-8") as fw:
        json.dump(statistic, fw, ensure_ascii=False, indent=4)

    print(f"=============== Statistic of Fractality of {DATA.upper()} ===============")
    print(f"# Proportion of Fractal Graphs: {is_fractal_num} / {total} = {is_fractal_num/total:.2%}")
    print(f"# Fractality Distribution:")
    for k, v in r2_distribution.items():
        print(f"\t>= {k} : {v:4d} / {total} = {v/total:.2%}")
    print()

