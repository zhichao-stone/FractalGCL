import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from .split import *


def get_features(
    model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device = torch.device("cuda")
):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for iterdata in dataloader:
            if isinstance(iterdata, (List, Tuple)):
                data = iterdata[0].to(device)
            else:
                data = iterdata.to(device)
                
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), device=device)
            
            feat: torch.Tensor = model(data.x, data.edge_index, data.batch)
            features.append(feat.detach().cpu())
            labels.append(data.y)
    features, labels = torch.cat(features, dim=0).to(device), torch.cat(labels, dim=0).to(device)
    return features, labels


def test_accuracy_SVC(
    model: nn.Module, 
    dataloader: DataLoader, 
    evaluator = SVC(kernel="rbf"), 
    eval_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}, 
    folds: int = 10, 
    device: torch.device = torch.device("cuda")
):
    features, labels = get_features(model, dataloader, device)

    if folds > 1:
        splits = k_fold(features.size()[0], folds=folds)
    else:
        splits = [split_data(features.size()[0])]
    
    accs = []
    for i in range(folds):
        split = splits[i].dict()
        x_train, x_val, x_test, y_train, y_val, y_test = [obj[split[key]].detach().cpu().numpy() for obj in [features, labels] for key in ["train", "valid", "test"]]
        x_train, y_train = np.concatenate([x_train, x_val], axis=0), np.concatenate([y_train, y_val], axis=0)
        classifier = GridSearchCV(evaluator, eval_params, cv=5, scoring="accuracy")
        classifier.fit(x_train, y_train)
        accs.append(accuracy_score(y_test, classifier.predict(x_test)))

    return accs


def test_accuracy_model(
    model: nn.Module, 
    features: torch.Tensor, 
    labels: torch.Tensor, 
    batch_size: int = 128, 
    device: torch.device = torch.device("cuda")
):
    model.eval()
    num_samples = features.size(0)
    batches = split_batches(num_samples, batch_size)
    acc = 0
    with torch.no_grad():
        features, labels = features.to(device), labels.to(device)
        for batch in batches:
            feat, label = features[batch], labels[batch]
            scores = model(feat)

            _, preds = torch.max(scores, dim=-1)
            correct = torch.sum(preds == label).item()
            acc += correct
    acc = acc / num_samples

    return acc