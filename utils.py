import os
import argparse
import random
import torch
import numpy as np
import logging
from typing import List



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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="MUTAG")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--random_seed", type=int, default=12306)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--aug_type", type=str, default="renorm_rc")
    parser.add_argument("--aug_num", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="save_model")
    parser.add_argument("--force_train", action="store_true")
    args = parser.parse_args()
    return args


def set_random_seed(random_seed: int):
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)  # set seed for cpu
    torch.cuda.manual_seed(random_seed)  # set seed for current gpu
    torch.cuda.manual_seed_all(random_seed)  # set seed for all gpu

def statistic_results_single_epoch(epoch: int, accs: List[float], logger: logging.Logger, folds: int = 10, detail: bool = True):
    if detail:
        logger.info(f"=============== Epoch {epoch:03d} ===============")
        logger.info(f"# Test Accs of All Folds: {[round(a, 4) for a in accs]}")

        accs = sorted(accs)
        logger.info(f"Acc Statistic: min={accs[0]:.4f} , max={accs[-1]:.4f}")
        
        medium_index = (folds-1) // 2
        logger.info(f"Acc Statistic: medium_l={accs[medium_index]:.4f} , medium_r={accs[-medium_index]:.4f} , medium={(accs[medium_index]+accs[-medium_index])/2:.4f}")
        
        mean, std = np.mean(accs), np.std(accs)
        logger.info(f"Average Accs of {folds} Folds: {mean:.4f}, Std: {std:.4f}\n")
    else:
        mean, std = np.mean(accs), np.std(accs)
        logger.info(f"# Epoch: {epoch:03d} | Average Accs of {folds} Folds: {mean:.4f}, Std: {std:.4f}")

    return mean, std