import os
import torch
import networkx as nx

from datas import compute_box_dimension


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_fractal_dimension(G:nx.Graph, count_diameter_less_nine: bool = True):
    # get the largest connected subgraph
    largest_cc = max(nx.connected_components(nx.to_undirected(G.copy())), key=len)
    lcc_G: nx.Graph = G.subgraph(largest_cc).copy()
    diameter = nx.diameter(lcc_G)

    # get fractal dimension
    if not count_diameter_less_nine and diameter <= 9:
        r2, dimension = 0.0, 0.0
    else:
        r2, dimension = compute_box_dimension(G, diameter, DEVICE)
        if r2 <= 0:
            r2, dimension = 0.0, 0.0

    regression_result = {
        "statistics of graph": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "diameter": diameter,
        },
        "fractality": {
            "r²": r2, 
            "dimension": dimension
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
        r2 = r["fractality"]["r²"]
        d = r["statistics of graph"]["diameter"]
        if d not in diameter_distribution:
            diameter_distribution[d] = 1
        else:
            diameter_distribution[d] += 1
        
        if r2 > 0:
            is_fractal_num += 1
        r2_interval = int(abs(r2) / 0.05)
        if r2_interval == 20:
            r2_interval -= 1
        for k in range(1, r2_interval+1):
            r2_distribution[f"{k*0.05:.2f}"] += 1
    
    statistic = {
        "total graphs": total, 
        "help": "The format: \"Number | Proportion ( Numer / Total)\"",
        "proportion of fractal graphs": f"{is_fractal_num} | {is_fractal_num/total:.2%}",  
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

