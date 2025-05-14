import os
import argparse
import random
import torch
import numpy as np
import logging
from typing import List, Tuple



class ExpLogger:
    def __init__(self, name, dir: str = "log") -> None:
        if not os.path.exists(dir):
            os.makedirs(dir)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(os.path.join(dir, f"{name}.log"))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def get_logger(self):
        return self.logger


class AdditionalRequirements:
    def __init__(self, args) -> None:
        self.compute_dimension = args.compute_dimension
        self.sum_embeddding = args.sum_embedding
        self.concat_graph = args.concat_graph


def get_args():
    aug_type_choices = ["renorm_rc", "renorm_rc_rr", "renorm_rc_prob", "mix_sep", "drop_node", "simple_random_walk"]
    parser = argparse.ArgumentParser()
    # base environment
    parser.add_argument("--random_seed", type=int, default=42, help="random seed for training")
    parser.add_argument("--cpu", action="store_true", help="whether to use cpu")
    # data setting
    parser.add_argument("--data", type=str, default="MUTAG", help="The name of TUDataset. See https://chrsmrrs.github.io/datasets/docs/datasets for detail.")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for loading data")
    parser.add_argument("--aug_type", type=str, default="renorm_rc", choices=aug_type_choices, help=f"type of augment method, seleted from {aug_type_choices}")
    parser.add_argument("--aug_num", type=int, default=2, choices=[1, 2], help="num of augmented graphs, either 1 or 2.")
    parser.add_argument("--aug_ratio", type=float, default=0.2, help="drop ratio for default augmentation (drop node)")
    parser.add_argument("--aug_threshold", type=float, default=0.9, help="threshold of fractal r2 for renormalization")
    parser.add_argument("--renorm_min_edges", type=int, default=1, help="the minimum edge num to be regard as edge between supernodes when renormalizing")
    parser.add_argument("--compute_dimension", action="store_true", help="whether to compute real dimension of augmented graphs")
    parser.add_argument("--concat_graph", action="store_true")
    # model setting
    parser.add_argument("--postfix", type=str, default="", help="postfix of model name for better differentiation")
    parser.add_argument("--gconv_num_layers", type=int, default=2, help="num of layers of GConv for features")
    parser.add_argument("--gconv_hidden_dim", type=int, default=64, help="hidden dim of GConv for features")
    parser.add_argument("--mlp_num_layers", type=int, default=2, help="num of layers of MLP for classification")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="hidden dim of MLP for classification")
    parser.add_argument("--sum_embedding", action="store_true")
    # loss setting
    parser.add_argument("--alpha", type=float, default=0.1, help="weight of gaussian noise of fractal dimension")
    parser.add_argument("--temperature", type=float, default=0.4, help="temperature of loss function")
    parser.add_argument("--sigma", type=float, default=0.1, help="sigma of gaussian noise of fractal dimension")
    # training setting
    parser.add_argument("--save_dir", type=str, default="save_model", help="directory of saved model")
    parser.add_argument("--pretrain_max_epochs", type=int, default=100, help="max training epochs when pretraining")
    parser.add_argument("--pretrain_lr", type=float, default=0.01, help="learning rate of pretraining")
    parser.add_argument("--pretrain_wd", type=float, default=0.0, help="weight decay of pretraining")
    parser.add_argument("--finetune_max_epochs", type=int, default=100, help="max training epochs when finetuning")
    parser.add_argument("--finetune_lr", type=float, default=0.001, help="learning rate of finetuning")
    parser.add_argument("--finetune_wd", type=float, default=1e-5, help="weight decay of finetuning")
    parser.add_argument("--force_train", action="store_true", help="whether to force pretraining, even when there are saved model")
    parser.add_argument("--folds", type=int, default=10, help="k of k-fold cross validation")
    parser.add_argument("--num_repeat_exp", type=int, default=5, help="num of repeated experiment")
    args = parser.parse_args()
    return args


def set_random_seed(random_seed: int):
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)  # set seed for cpu
    torch.cuda.manual_seed(random_seed)  # set seed for current gpu
    torch.cuda.manual_seed_all(random_seed)  # set seed for all gpu


def statistic_results_single_epoch(
    epoch: int, 
    accs: List[float], 
    logger: logging.Logger, 
    folds: int = 10, 
    detail: bool = True, 
    time_cost: float = None,
):
    if detail:
        logger.info(f"=============== Epoch {epoch:03d} ===============")
        logger.info(f"# Test Accs of All Folds: {[round(a, 4) for a in accs]}")

        accs = sorted(accs)
        logger.info(f"Acc Statistic: min={accs[0]:.4f} , max={accs[-1]:.4f}")
        
        medium_index = (folds-1) // 2
        logger.info(f"Acc Statistic: medium_l={accs[medium_index]:.4f} , medium_r={accs[-medium_index]:.4f} , medium={(accs[medium_index]+accs[-medium_index])/2:.4f}")
        
        mean, std = np.mean(accs), np.std(accs)
        logger.info(f"Average Accs of {folds} Folds: {mean:.4f}, Std: {std:.4f}{f', Time: {time_cost:.2f} s' if time_cost else ''}\n")
    else:
        mean, std = np.mean(accs), np.std(accs)
        logger.info(f"# Epoch: {epoch:03d} | Average Accs of {folds} Folds: {mean:.4f}, Std: {std:.4f}{f', Time: {time_cost:.2f} s' if time_cost else ''}")

    return mean, std


def concat_graph(
    x1: torch.Tensor, edge_index1: torch.Tensor, batch1: torch.Tensor, 
    x2: torch.Tensor, edge_index2: torch.Tensor, batch2: torch.Tensor, 
    device: torch.device = torch.device("cuda")
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bs1, bs2 = int(max(batch1)) + 1, int(max(batch2)) + 1
    assert bs1 == bs2
    ns1 = int(x1.size(0))
    x1, x2 = x1.to(device), x2.to(device)
    edge_index1, edge_index2 = edge_index1.to(device), edge_index2.to(device)
    batch1, batch2 = batch1.to(device), batch2.to(device)
    x = torch.cat([x1, x2], dim=0)
    edge_index = torch.cat([edge_index1, edge_index2 + ns1], dim=-1)
    batch = torch.cat([batch1, batch2], dim=-1)
    return x, edge_index, batch