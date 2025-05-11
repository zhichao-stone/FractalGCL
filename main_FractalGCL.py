import warnings
warnings.filterwarnings("ignore")

import os
import random
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from datas.dataload import *
from datas.aumentation import FractalAugmentor
from models import GConv
from models.loss import *
from evaluate.eval import test_accuracy
from logger import Logger


def train(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: optim.Optimizer, 
    augmentor: FractalAugmentor, 
    aug_type: str, 
    loss_fn: FractalGCLLoss, 
    device: torch.device = torch.device("cuda"), 
    aug_num: int = 2
):
    model.train()
    epoch_loss = 0.0
    for data, fractalities, diameters, dimensions, gids in dataloader:
        data = data.to(device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), device=device)

        optimizer.zero_grad()
        if aug_num == 2:
            (x1, edge_index1, batch1), (x2, edge_index2, batch2) = augmentor.augment(data.x, data.edge_index, data.batch, fractalities, diameters, gids, aug_type, aug_num)
        else:
            x1, edge_index1, batch1 = data.x, data.edge_index, data.batch
            x2, edge_index2, batch2 = augmentor.augment(data.x, data.edge_index, data.batch, fractalities, diameters, gids, aug_type, aug_num)[0]
        g1 = model(x1, edge_index1, batch1, project=True)
        g2 = model(x2, edge_index2, batch2, project=True)

        loss = loss_fn(g1, g2, dimensions, diameters)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="MUTAG")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--aug_type", type=str, default="renorm_rc")
    parser.add_argument("--aug_num", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="save_model")
    parser.add_argument("--force_train", action="store_true")
    args = parser.parse_args()
    device = torch.device(f"cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    folds = args.folds
    max_epoch = args.epoch
    aug_type = args.aug_type
    batch_size: int = args.batch_size
    data_name = str(args.data).upper()
    logger = Logger(name=f"FractalGCL_{data_name}_{aug_type}").get_logger()

    logger.info(f"Training FractalGCL on {data_name} with {aug_type}, {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # random setting
    random_seed = args.random_seed
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)  # set seed for cpu
    torch.cuda.manual_seed(random_seed)  # set seed for current gpu
    torch.cuda.manual_seed_all(random_seed)  # set seed for all gpu

    # load
    root = os.path.join(os.path.expanduser("~"), "datasets")
    fractal_result_path = os.path.join("fractal_results", f"linear_regression_{data_name.lower()}.json")
    fractal_results = load_json(fractal_result_path) if os.path.exists(fractal_result_path) else []
    tudataset = TUDataset(root, name=data_name)
    dataset = FractalDataset(tudataset, fractal_results)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    input_dim = max(tudataset.num_features, 1)

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"FractalGCL_{data_name}_{aug_type}.pt")

    # model
    model = GConv(
        input_dim=input_dim, 
        hidden_dim=64, 
        num_layers=2
    ).to(device)

    finetune_dataloader = DataLoader(tudataset, batch_size=batch_size)

    # train
    if not os.path.exists(save_path) or args.force_train:
        augmentor = FractalAugmentor(
            drop_ratio=0.2, 
            aug_fractal_threshold=0.95, 
            renorm_min_edges=1, 
            device=device
        )
        loss_fn = FractalGCLLoss(temperature=0.5, alpha=0.05, sigma=0.1)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        epoch_accs = []
        with tqdm(total=max_epoch, desc="pretrain") as pbar:
            for epoch in range(1, max_epoch+1):
                st = time.time()
                loss = train(model, dataloader, optimizer, augmentor, aug_type, loss_fn, device, aug_num=args.aug_num)
                pbar.set_postfix({"loss": round(loss, 4), "time": round(time.time() - st, 2)})

                if epoch % 5 == 0:
                    test_acc = test_accuracy(model, finetune_dataloader, folds=folds, device=device)
                    logger.info(f"# Epoch: {epoch} | test acc: {test_acc:.4f}")
                    epoch_accs.append(test_acc)

                pbar.update()
                
        torch.save(model.state_dict(), save_path)
        best_epoch, best_acc = max(enumerate(epoch_accs), key=lambda x:x[1])
        logger.info(f"# Final Results: {best_acc:.4f} , epoch={(best_epoch+1)*5}\n\n")
    else:
        model.load_state_dict(torch.load(save_path))

        # test and statistic
        test_acc = test_accuracy(model, dataloader, folds=folds, device=device)
        logger.info(f"# Best CV Acc: {test_acc:.4f}")
