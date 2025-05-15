import warnings
warnings.filterwarnings("ignore")

import os
import random
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from datas.dataload import *
from datas.aumentation import FractalAugmentor
from models import GConv
from models.loss import *
from evaluate import test_accuracy_SVC
from utils import *



def train(
    model: GConv, 
    dataloader: DataLoader, 
    optimizer: optim.Optimizer, 
    augmentor: FractalAugmentor, 
    aug_type: str, 
    loss_fn: FractalGCLLoss, 
    key_arguments: KeyArguments, 
    device: torch.device = torch.device("cuda")
):
    model.train()
    epoch_loss = 0.0
    for data, fractalities, diameters, dimensions, gids in dataloader:
        data = data.to(device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), device=device)

        optimizer.zero_grad()
        if key_arguments.num_aug_graphs == 2:
            aug_graphs, aug_dimesions, aug_diameters = augmentor.augment(
                data.x, data.edge_index, data.batch, 
                fractalities, diameters, dimensions, gids, 
                aug_type, key_arguments.num_aug_graphs, 
                key_arguments.compute_dimension
            )
            (x1, edge_index1, batch1), (x2, edge_index2, batch2) = aug_graphs
            aug_dims1, aug_dims2 = aug_dimesions
            aug_d1, aug_d2 = aug_diameters
        else:
            x1, edge_index1, batch1, aug_dims1, aug_d1 = data.x, data.edge_index, data.batch, dimensions, diameters
            aug_graphs, aug_dimesions = augmentor.augment(
                data.x, data.edge_index, data.batch, 
                fractalities, diameters, dimensions, gids, 
                aug_type, key_arguments.num_aug_graphs, 
                key_arguments.compute_dimension
            )
            x2, edge_index2, batch2 = aug_graphs[0]
            aug_dims2, aug_d2 = aug_dimesions[0], aug_diameters[0]

        if not key_arguments.only_renorm:
            x1, edge_index1, batch1 = concat_graph(x1, edge_index1, batch1, data.x, data.edge_index, data.batch, device)
            x2, edge_index2, batch2 = concat_graph(x2, edge_index2, batch2, data.x, data.edge_index, data.batch, device)
        
        g1 = model(x1, edge_index1, batch1, project=True)
        g2 = model(x2, edge_index2, batch2, project=True)

        loss = loss_fn(g1, g2, aug_dims1, aug_dims2, aug_d1, aug_d2)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss




if __name__ == "__main__":
    # base arguments
    args = get_args()
    device = torch.device(f"cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    folds = args.folds
    aug_type = args.aug_type
    batch_size: int = args.batch_size
    data_name = str(args.data).upper()
    model_name = f"FractalGCL_{data_name}_{aug_type}"
    model_post_fix = args.postfix
    if model_post_fix:
        model_name += f"_{model_post_fix}"
    logger = ExpLogger(name=model_name).get_logger()

    logger.info(f"Training FractalGCL on {data_name} with {aug_type}{' and ' + model_post_fix if model_post_fix else ''}, {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # random setting
    random_seed = args.random_seed
    set_random_seed(random_seed)

    # load
    root = os.path.join(os.path.expanduser("~"), "datasets")
    fractal_result_path = os.path.join("fractal_results", f"linear_regression_{data_name.lower()}.json")
    fractal_results = load_json(fractal_result_path) if os.path.exists(fractal_result_path) else []
    tudataset = TUDataset(root, name=data_name)
    dataset = FractalDataset(tudataset, fractal_results)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    input_dim = max(tudataset.num_features, 1)
    num_classes = tudataset.num_classes

    # model
    model = GConv(
        input_dim=input_dim, 
        hidden_dim=args.gconv_hidden_dim, 
        num_layers=args.gconv_num_layers
    ).to(device)

    # Pretrain
    save_dir = os.path.join(args.save_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not args.force_train:
        for root, dirs, files in os.walk(save_dir):
            epochs = sorted([int(os.path.splitext(f)[0].replace("epoch", "")) for f in files if f.endswith(".pt")])
            need_train = (len(epochs) == 0) or (epochs[-1] < args.pretrain_max_epochs)
            save_model_file_name = f"epoch{epochs[-1]}.pt" if epochs else ""
            current_epoch = epochs[-1] if epochs else 0
    else:
        need_train = True
        current_epoch, save_model_file_name = 0, ""

    if need_train:
        if save_model_file_name:
            model.load_state_dict(torch.load(os.path.join(save_dir, save_model_file_name)))

        key_arguments = KeyArguments(args)

        augmentor = FractalAugmentor(
            drop_ratio=args.aug_ratio, 
            aug_fractal_threshold=args.aug_threshold, 
            renorm_min_edges=args.renorm_min_edges, 
            device=device
        )
        pure_dataloader = DataLoader(tudataset, batch_size=batch_size)

        loss_fn = FractalGCLLoss(temperature=args.temperature, alpha=args.alpha, sigma=args.sigma)
        optimizer = optim.Adam(model.parameters(), lr=args.pretrain_lr, weight_decay=args.pretrain_wd)

        # epoch_accs = []
        time_cost = 0.0
        max_epochs = args.pretrain_max_epochs
        with tqdm(total=max_epochs-current_epoch, desc="pretrain") as pbar:
            for epoch in range(current_epoch+1, max_epochs+1):
                st = time.time()
                loss = train(
                    model, 
                    dataloader, 
                    optimizer, 
                    augmentor, 
                    aug_type, 
                    loss_fn, 
                    key_arguments=key_arguments,  
                    device=device
                )
                epoch_time_cost = time.time() - st
                time_cost += epoch_time_cost
                pbar.set_postfix({"loss": round(loss, 4), "time": round(epoch_time_cost, 2)})

                if epoch % 5 == 0:
                    torch.save(model.state_dict(), os.path.join(save_dir, f"epoch{epoch}.pt"))

                pbar.update()
        logger.info(f"# Pretraining Time Cost: {time_cost} s")
                
        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch{epoch}.pt"))

    # Multiple Experiment: K-Fold Finetune and Test
    accs, acc_epochs = [], []
    if args.num_repeat_exp > 1:
        seeds = random.sample(list(range(100000)), args.num_repeat_exp)
    else:
        seeds = [random_seed]
    for i, seed in enumerate(seeds):
        logger.info(f"=============== Seed {i+1}: {seed} ===============")
        set_random_seed(seed)
        best_acc, best_acc_std, best_acc_epoch = 0.0, 0.0, -1
        test_time_cost = 0.0
        for root, dirs, files in os.walk(save_dir):
            epochs = sorted([int(os.path.splitext(f)[0].replace("epoch", "")) for f in files if f.endswith(".pt")])
            for epoch in epochs:
                epoch_st = time.time()
                model.load_state_dict(torch.load(os.path.join(save_dir, f"epoch{epoch}.pt")))

                # finetune, test and statistic
                test_accs = test_accuracy_SVC(model, dataloader, folds=folds, device=device)
                epoch_time_cost = time.time() - epoch_st
                mean, std = statistic_results_single_epoch(epoch, test_accs, logger, folds, detail=False, time_cost=epoch_time_cost)
                test_time_cost += epoch_time_cost

                if mean > best_acc:
                    best_acc, best_acc_std, best_acc_epoch = mean, std, epoch

        logger.info(f"=============== Final Result of Seed {i+1} ===============")
        logger.info(f"Best 10-fold Result: acc={best_acc:.4f} , std={best_acc_std:.4f} , epoch={best_acc_epoch:03d} , time_cost={test_time_cost:.2f} s\n")
        accs.append(best_acc)
        acc_epochs.append(best_acc_epoch)
    
    mean, std = np.mean(accs), np.std(accs)
    logger.info(f"=============== Result of Multiple Experiment ===============")
    logger.info(f"All Results: accs={[round(a, 4) for a in accs]} , epochs={acc_epochs}")
    logger.info(f"Average Result: acc={mean:.4f} , std={std:.4f}")

