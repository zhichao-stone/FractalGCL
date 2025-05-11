from typing import List, Dict

import torch
from sklearn.model_selection import StratifiedKFold



class Split:
    def __init__(self, train, valid, test) -> None:
        self.train = torch.tensor(train)
        self.valid = torch.tensor(valid)
        self.test = torch.tensor(test)

    def dict(self) -> Dict[str, torch.Tensor]:
        return {
            "train": self.train, 
            "valid": self.valid, 
            "test": self.test
        }


def split_data(num_samples: int, train_ratio: float = 0.8, test_ratio: float = 0.1):
    assert train_ratio + test_ratio <= 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    val_size = num_samples - train_size - test_size
    indices = torch.randperm(num_samples)
    return Split(
        train=indices[ : train_size], 
        valid=indices[train_size : train_size+val_size], 
        test=indices[train_size+val_size : ]
    )


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

    splits = []
    for i in range(folds):
        splits.append(Split(
            train=train_indices[i], 
            valid=valid_indices[i], 
            test=test_indices[i]
        ))

    return splits