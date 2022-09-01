import torch


def relative_euclidean_dist(X: torch.Tensor, X_hat: torch.Tensor):
    return (X - X_hat).norm(2, dim=1) / X.norm(2, dim=1)
