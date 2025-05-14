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
from models import GConv, MLP
from models.loss import *
from evaluate import k_fold, split_data, split_batches, get_features, test_accuracy_SVC, test_accuracy
from utils import *



def train(
    model: GConv, 
    dataloader: DataLoader, 
    optimizer: optim.Optimizer, 
    augmentor: FractalAugmentor, 
    aug_type: str, 
    loss_fn: FractalGCLLoss, 
    requirements: AdditionalRequirements, 
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
            aug_graphs, aug_dimesions, aug_diameters = augmentor.augment(data.x, data.edge_index, data.batch, fractalities, diameters, gids, aug_type, aug_num, requirements.compute_dimension)
            (x1, edge_index1, batch1), (x2, edge_index2, batch2) = aug_graphs
            aug_dims1, aug_dims2 = aug_dimesions
            aug_d1, aug_d2 = aug_diameters
        else:
            x1, edge_index1, batch1, aug_dims1, aug_d1 = data.x, data.edge_index, data.batch, dimensions, diameters
            aug_graphs, aug_dimesions = augmentor.augment(data.x, data.edge_index, data.batch, fractalities, diameters, gids, aug_type, aug_num, requirements.compute_dimension)[0]
            x2, edge_index2, batch2 = aug_graphs[0]
            aug_dims2, aug_d2 = aug_dimesions[0], aug_diameters[0]

        if requirements.concat_graph:
            x1, edge_index1, batch1 = concat_graph(x1, edge_index1, batch1, data.x, data.edge_index, data.batch, device)
            x2, edge_index2, batch2 = concat_graph(x2, edge_index2, batch2, data.x, data.edge_index, data.batch, device)

        g1 = model(x1, edge_index1, batch1, project=True)
        g2 = model(x2, edge_index2, batch2, project=True)

        if requirements.sum_embeddding:
            g = model(data.x, data.edge_index, data.batch, project=True)
            g1, g2 = g1 + g, g2 + g

        loss = loss_fn(g1, g2, aug_dims1, aug_dims2, aug_d1, aug_d2)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss


def train_mlp(
    model: GConv, 
    optimizer: optim.Optimizer, 
    features: torch.Tensor, 
    labels: torch.Tensor, 
    batch_size: int = 128, 
    device: torch.device = torch.device("cuda")
):
    model.train()
    features, labels = features.to(device), labels.to(device)
    batches = split_batches(features.size(0), batch_size, shuffle=True)
    epoch_loss = 0.0
    for batch in batches:
        x, y = features[batch], labels[batch]
        optimizer.zero_grad()
        scores = model(x)
        loss: torch.Tensor = nn.CrossEntropyLoss()(scores, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().cpu().item()

    epoch_loss = epoch_loss / len(batches)

    return epoch_loss


def finetune(
    model: nn.Module, 
    dataloader: DataLoader, 
    num_classes: int, 
    mlp_num_layers: int = 2, 
    mlp_hidden_dim: int = 64, 
    folds: int = 10, 
    learning_rate: float = 0.001, 
    weight_decay: float = 1e-5, 
    max_epochs: int = 100, 
    semi_ratio: float = 1.0, 
    device: torch.device = torch.device("cuda")
):
    # get features
    features, labels = get_features(model, dataloader, device)

    # finetune
    if folds > 1:
        splits = k_fold(features.size()[0], folds=folds)
    else:
        splits = [split_data(features.size()[0])]

    accs = []
    for i in range(folds):
        split = splits[i].dict()
        x_train, x_val, x_test, y_train, y_val, y_test = [obj[split[key]] for obj in [features, labels] for key in ["train", "valid", "test"]]
        x_train, y_train = torch.cat([x_train, x_val], dim=0), torch.cat([y_train, y_val], dim=0)
        mlp = MLP(
            input_dim=features.size(-1), 
            hidden_dim=mlp_hidden_dim, 
            output_dim=num_classes, 
            num_layers=mlp_num_layers
        ).to(device)
        optimizer = optim.Adam(mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)
        best_acc = 0.0
        for epoch in range(max_epochs):
            train_mlp(mlp, optimizer, x_train, y_train, batch_size, device)
            test_acc = test_accuracy(mlp, x_test, y_test, batch_size, device)
            if test_acc > best_acc:
                best_acc = test_acc
        accs.append(best_acc)

    return accs


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
    gconv_num_layers = args.gconv_num_layers
    gconv_hidden_dim = args.gconv_hidden_dim
    model = GConv(
        input_dim=input_dim, 
        hidden_dim=gconv_hidden_dim, 
        num_layers=gconv_num_layers
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

        requirements = AdditionalRequirements(args)

        augmentor = FractalAugmentor(
            drop_ratio=args.aug_ratio, 
            aug_fractal_threshold=args.aug_threshold, 
            renorm_min_edges=args.renorm_min_edges, 
            device=device
        )
        pure_dataloader = DataLoader(tudataset, batch_size=batch_size)

        loss_fn = FractalGCLLoss(temperature=args.temperature, alpha=args.alpha, sigma=args.sigma)
        optimizer = optim.Adam(model.parameters(), lr=args.pretrain_lr, weight_decay=args.pretrain_wd)

        epoch_accs = []
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
                    requirements=requirements, 
                    device=device, 
                    aug_num=args.aug_num)
                pbar.set_postfix({"loss": round(loss, 4), "time": round(time.time() - st, 2)})

                if epoch % 5 == 0:
                    test_accs = test_accuracy_SVC(model, pure_dataloader, folds=folds, device=device)
                    test_acc = np.mean(test_accs)
                    logger.info(f"# Epoch: {epoch} | test acc: {test_acc:.4f}")
                    epoch_accs.append(test_acc)
                    torch.save(model.state_dict(), os.path.join(save_dir, f"epoch{epoch}.pt"))

                pbar.update()
                
        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch{epoch}.pt"))
        best_epoch, best_acc = max(enumerate(epoch_accs), key=lambda x:x[1])
        best_epoch = (best_epoch + 1) * 5
        logger.info(f"# Final Results: {best_acc:.4f} , epoch={best_epoch:3d}\n\n")

    # Multiple Experiment: K-Fold Finetune and Test
    accs, acc_epochs = [], []
    if args.num_repeat_exp > 1:
        seeds = random.sample(list(range(100000)), args.num_repeat_exp)
    else:
        seeds = [random_seed]
    for i, seed in enumerate(seeds):
        logger.info(f"=============== Seed{i+1} {seed} ===============")
        set_random_seed(seed)
        best_acc, best_acc_std, best_acc_epoch = 0.0, 0.0, -1
        for root, dirs, files in os.walk(save_dir):
            epochs = sorted([int(os.path.splitext(f)[0].replace("epoch", "")) for f in files if f.endswith(".pt")])
            for epoch in epochs:
                model.load_state_dict(torch.load(os.path.join(save_dir, f"epoch{epoch}.pt")))

                # finetune, test and statistic
                test_accs = finetune(
                    model=model, 
                    dataloader=dataloader, 
                    num_classes=num_classes, 
                    mlp_num_layers=args.mlp_num_layers, 
                    mlp_hidden_dim=args.mlp_hidden_dim, 
                    folds=folds, 
                    learning_rate=args.finetune_lr, 
                    weight_decay=args.finetune_wd, 
                    max_epochs=args.finetune_max_epochs, 
                    device=device
                )
                mean, std = statistic_results_single_epoch(epoch, test_accs, logger, folds, detail=False)
                
                if mean > best_acc:
                    best_acc, best_acc_std, best_acc_epoch = mean, std, epoch
        logger.info(f"=============== Final Result ===============")
        logger.info(f"Best 10-fold Result: acc={best_acc:.4f} , std={best_acc_std:.4f} , epoch={best_acc_epoch:03d}\n")
        accs.append(best_acc)
        acc_epochs.append(best_acc_epoch)
    
    mean, std = np.mean(accs), np.std(accs)
    logger.info(f"=============== Result of Multiple Experiment ===============")
    logger.info(f"All Results: accs={[round(a, 4) for a in accs]} , epochs={acc_epochs}")
    logger.info(f"Average Result: acc={mean:.4f} , std={std:.4f}")

