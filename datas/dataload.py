import warnings
warnings.filterwarnings("ignore")
import json
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold



def load_json(path: str, **kwargs):
    encoding = kwargs.pop("encoding", "utf-8")
    mode = kwargs.pop("mode", "r")
    with open(path, mode, encoding=encoding, **kwargs) as fr:
        data = json.load(fr)
    return data

def dump_json(obj, path: str, **kwargs):
    encoding = kwargs.pop("encoding", "utf-8")
    indent = kwargs.pop("indent", 4)
    mode = kwargs.pop("mode", "w")
    with open(path, mode, encoding=encoding, **kwargs) as fw:
        json.dump(obj, fw, ensure_ascii=False, indent=indent)

class FractalAttr:
    def __init__(self, 
        box_dimension: float, 
        fractality: float, 
        diameter: int
    ) -> None:
        self.fractality = fractality
        self.diameter = diameter
        self.dimension = box_dimension

    def __str__(self):
        return f"FractalAttr(fractality={self.fractality} , diameter={self.diameter} , dimension={self.dimension})"


def process_fractal_results(fractal_results: List[Dict[str, str]], dataset_size: int):
    fractal_attrs = []
    if fractal_results is None or len(fractal_results) == 0:
        fractal_attrs = [FractalAttr(
            box_dimension=0.0, 
            fractality=0.0, 
            diameter=0
        ) for _ in range(dataset_size)]
    else:
        for r in fractal_results:
            diameter = r["statistics of graph"]["diameter"]
            res = r["fractality"]
            fractality = res["r²"]
            box_dimension = res["dimension"]
            fractal_attrs.append(FractalAttr(
                box_dimension=box_dimension, 
                fractality=fractality, 
                diameter=diameter
            ))

    return fractal_attrs


def k_fold(num_samples: int, folds: int = 10):
    train_indices: List[List[int]] = []
    valid_indices: List[List[int]] = []
    test_indices: List[List[int]] = []

    skf = StratifiedKFold(folds, shuffle=True)
    for _, idxs in skf.split(torch.zeros(num_samples), torch.ones(num_samples)):
        test_indices.append(torch.tensor([int(idx) for idx in idxs]))
    
    valid_indices = [test_indices[i-1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(num_samples, dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[valid_indices[i]] = 0
        idx_train = train_mask.nonzero().view(-1)
    
        train_indices.append(torch.tensor([int(idx) for idx in idx_train]))

    return train_indices, valid_indices, test_indices


class FractalDataset(Dataset):
    def __init__(self, 
        dataset: Dataset,
        fractal_results: List[Dict[str, str]] = []
    ) -> None:
        super(FractalDataset, self).__init__()
        self.dataset = dataset
        self.fractals = process_fractal_results(fractal_results, len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fractal_attr = self.fractals[index]
        return self.dataset[index], fractal_attr.fractality, fractal_attr.diameter, fractal_attr.dimension, index