import numpy as np
import torch


def relative_euclidean_dist(X: torch.Tensor, X_hat: torch.Tensor):
    return (X - X_hat).norm(2, dim=1) / X.norm(2, dim=1)


def get_most_recurrent(items: np.ndarray):
    label, count = "", 0
    for item in np.unique(items):
        N = (items == item).sum()
        if N > count:
            count = N
            label = item
    return label
