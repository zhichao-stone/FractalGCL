from tqdm import tqdm
from typing import List, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

from GCL.eval import SVMEvaluator
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from .split import *



def test_f1_score(
    model: nn.Module, 
    dataloader: DataLoader, 
    folds: int = 1, 
    device: torch.device = torch.device("cuda")
):
    model.eval()
    x, y = [], []
    with torch.no_grad():
        for iterdata in dataloader:
            if isinstance(iterdata, (List, Tuple)):
                data = iterdata[0].to(device)
            else:
                data = iterdata.to(device)
                
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), device=device)
            
            g = model(data.x, data.edge_index, data.batch)
            x.append(g)
            y.append(data.y)
    x, y = torch.cat(x, dim=0), torch.cat(y, dim=0)
    
    if folds > 1:
        splits = k_fold(x.size()[0], folds=folds)
    else:
        splits = [split_data(x.size()[0])]

    micro_f1s, macro_f1s = [], []
    for i in tqdm(range(folds), desc=f"{folds}-folds"):
        result = SVMEvaluator(linear=True)(x, y, splits[i].dict())
        micro_f1s.append(result["micro_f1"])
        macro_f1s.append(result["macro_f1"])

    return micro_f1s, macro_f1s


def test_accuracy(
    model: nn.Module, 
    dataloader: DataLoader, 
    evaluator = LinearSVC(), 
    eval_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}, 
    folds: int = 10, 
    device: torch.device = torch.device("cuda")
):
    model.eval()
    x, y = [], []
    with torch.no_grad():
        for iterdata in dataloader:
            if isinstance(iterdata, (List, Tuple)):
                data = iterdata[0].to(device)
            else:
                data = iterdata.to(device)

            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), device=device)
            
            g = model(data.x, data.edge_index, data.batch)
            x.append(g)
            y.append(data.y)
    x, y = torch.cat(x, dim=0), torch.cat(y, dim=0)

    x, y = [obj.detach().cpu().numpy() for obj in [x, y]]
    classifier = GridSearchCV(evaluator, eval_params, cv=folds, scoring="accuracy")
    classifier.fit(x, y)
    acc = float(classifier.best_score_)

    return acc