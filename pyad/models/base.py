import torch
import numpy as np

from typing import Tuple, Optional
from abc import abstractmethod
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader


activation_map = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leakyRelu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid
}


def create_net_layers(in_dim, out_dim, hidden_dims, activation="relu", bias=True, dropout=0.):
    layers = []
    assert 0. <= dropout <= 1., "`dropout` must be inclusively between 0 and 1"
    for i in range(len(hidden_dims)):
        layers.append(
            nn.Linear(in_dim, hidden_dims[i], bias=bias)
        )
        if dropout > 0.:
            layers.append(
                nn.Dropout(dropout)
            )
        layers.append(
            activation_map[activation]()
        )
        in_dim = hidden_dims[i]
    layers.append(
        nn.Linear(hidden_dims[-1], out_dim, bias=bias)
    )
    return layers


class BaseModule(nn.Module):
    def __init__(
            self,
            in_features: int,
            n_instances: int,
            lr: float,
            device: str = None,
            weight_decay: float = 0.,
            **kwargs
    ):
        super(BaseModule, self).__init__()
        self.in_features = in_features
        self.n_instances = n_instances
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = None
        self.scheduler = None
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.current_epoch = None

    @abstractmethod
    def print_name(self) -> str:
        pass

    @abstractmethod
    def training_step(self, X: torch.Tensor, y: torch.Tensor = None):
        pass

    @abstractmethod
    def score(self, X: torch.Tensor, y: torch.Tensor = None):
        pass

    @abstractmethod
    def compute_loss(self, outputs: torch.Tensor, y: torch.Tensor, **kwargs):
        pass

    @abstractmethod
    def get_hparams(self) -> dict:
        pass

    def predict_step(self, X: torch.Tensor):
        scores = self.score(X)
        return scores

    def on_train_epoch_start(self):
        pass

    def on_train_epoch_end(self):
        pass

    def on_test_model_eval(self):
        pass

    def on_before_fit(self, dataset: DataLoader):
        pass

    @abstractmethod
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.StepLR]]:
        pass

    def get_params(self) -> dict:
        shared_params = {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "device": self.device
        }
        specific_params = self.get_hparams()
        return dict(
            **shared_params,
            **specific_params
        )

    def eval_step(self):
        self.eval()
        self.on_test_model_eval()

    def predict(self, dataset):
        all_scores, all_ys, all_labels = [], [], []
        self.eval_step()
        with torch.no_grad():
            for batch in dataset:
                X, y, labels = batch
                y = y.detach().cpu().tolist()
                X = X.to(self.device).float()
                scores = self.predict_step(X).detach().cpu().tolist()
                # all_scores = np.hstack((all_scores, scores))
                # all_ys = np.hstack((all_ys, y))
                # all_labels = np.hstack((all_labels, labels))
                all_scores.extend(scores)
                all_ys.extend(y)
                all_labels.extend(labels)
        self.train(mode=True)
        return np.array(all_scores), np.array(all_ys, dtype=np.int8), np.array(all_labels)
