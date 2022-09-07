import numpy as np
import torch
import torch.nn.functional as F

from collections import OrderedDict
from pyad.models.base import BaseModule, create_net_layers
from pyad.utilities.cli import MODEL_REGISTRY
from torch.utils.data import DataLoader
from torch import nn
from typing import List


@MODEL_REGISTRY
class DeepSVDD(BaseModule):

    def __init__(
            self,
            feature_dim: int,
            hidden_dims: List[int],
            activation="relu",
            eps: float = 0.1,
            radius=None,
            **kwargs):
        super(DeepSVDD, self).__init__(**kwargs)
        # model parameters
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.eps = eps
        # computed parameters
        self.center = nn.Parameter(
            torch.empty(self.feature_dim, dtype=torch.float, device=self.device)
        )
        self.radius = radius
        self._build_network()

    def _build_network(self):
        self.net = nn.Sequential(
            *create_net_layers(
                in_dim=self.in_features,
                out_dim=self.feature_dim,
                hidden_dims=self.hidden_dims,
                activation=self.activation
            )
        ).to(self.device)

    def print_name(self) -> str:
        return "DSVDD"

    def get_hparams(self) -> dict:
        return dict(
            feature_dim=self.feature_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            eps=self.eps
        )

    def init_center_c(self, train_loader: DataLoader):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data.
           Code taken from https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/optim/deepSVDD_trainer.py"""
        n_samples = 0
        c = torch.empty(self.feature_dim, dtype=torch.float, device=self.device)

        self.net.eval()
        self.eval()
        with torch.no_grad():
            for sample in train_loader:
                # get the inputs of the batch
                X, _, _ = sample
                X = X.to(self.device).float()
                outputs = self.net(X)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        self.train(mode=True)
        self.net.train(mode=True)

        if c.isnan().sum() > 0:
            raise Exception("NaN value encountered during init_center_c")

        if torch.allclose(c, torch.zeros_like(c)):
            raise Exception("Center c initialized at 0")

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < self.eps) & (c < 0)] = -self.eps
        c[(abs(c) < self.eps) & (c > 0)] = self.eps

        return nn.Parameter(c)

    def on_before_fit(self, dataloader: DataLoader):
        print("Initializing center ...")
        center = self.init_center_c(dataloader).to(self.device)
        self.center = center

    def score(self, X: torch.Tensor, y: torch.Tensor = None, labels: torch.Tensor = None):
        assert torch.allclose(self.center, torch.zeros_like(self.center)) is False, "center not initialized"
        outputs = self.net(X)
        if self.center.device != outputs.device:
            self.center = self.center.to(outputs.device)
        return torch.sum((outputs - self.center) ** 2, dim=1)

    def compute_loss(self, outputs: torch.Tensor, **kwargs):
        loss = torch.sum((outputs - self.center) ** 2, dim=1)
        return loss.mean()

    def training_step(self, X: torch.Tensor, y: torch.Tensor = None, labels: torch.Tensor = None):
        outputs = self.net(X)
        loss = self.compute_loss(outputs)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
        return optimizer, None


@MODEL_REGISTRY
class DROCC(BaseModule):

    def __init__(
            self,
            lamb: float = 1.,
            radius: float = None,
            gamma: float = 2.,
            n_classes: int = 1,
            n_hidden_nodes: int = 20,
            only_ce_epochs: int = 50,
            ascent_step_size: float = 0.01,
            ascent_num_steps: int = 50,
            **kwargs
    ):
        """
        Implements architecture presented in `DROCC: Deep Robust One-Class Classification` by Goyal et al. published
        in 2020 (https://arxiv.org/pdf/2002.12718.pdf). Most of the implementation is adapted directly from the original
        GitHub repository: https://github.com/microsoft/EdgeML

        Parameters
        ----------
        lamb: float
            weight for the adversarial loss
        radius: float
            radius of the hypersphere
        gamma: float
            used to fit the maximum volume of the hypersphere (gamma * radius)
        n_classes: int
            number of different classes (should always be one)
        n_hidden_nodes: int
        only_ce_epochs: int
            number of training epochs where only the binary cross-entropy loss is considered
        ascent_step_size: float
            step size during gradient ascent
        ascent_num_steps: int
            number of gradient ascent steps
        kwargs
        """
        super(DROCC, self).__init__(**kwargs)
        # models parameters
        self.lamb = lamb
        self.gamma = gamma
        self.n_classes = n_classes
        self.n_hidden_nodes = n_hidden_nodes
        self.only_ce_epochs = only_ce_epochs
        self.ascent_step_size = ascent_step_size
        self.ascent_num_steps = ascent_num_steps
        # computed parameters
        self.radius = radius or np.sqrt(self.in_features) / 2
        self._build_network()

    def _build_network(self):
        self.feature_extractor = nn.Sequential(
            OrderedDict([
                ('fc', nn.Linear(self.in_features, self.n_hidden_nodes)),
                ('relu1', torch.nn.ReLU(inplace=True))])
        ).to(self.device)
        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(self.n_hidden_nodes, self.n_classes))
            ])
        ).to(self.device)

    def print_name(self) -> str:
        return "DROCC"

    def get_hparams(self) -> dict:
        return dict(
            lamb=self.lamb,
            gamma=self.gamma,
            n_classes=self.n_classes,
            n_hidden_nodes=self.n_hidden_nodes,
            only_ce_epochs=self.only_ce_epochs,
            ascent_step_size=self.ascent_step_size,
            ascent_num_steps=self.ascent_num_steps,
            radius=self.radius,
        )

    def forward(self, X: torch.Tensor):
        features = self.feature_extractor(X)
        logits = self.classifier(features.view(-1, self.n_hidden_nodes))
        return logits

    def score(self, X: torch.Tensor, y: torch.Tensor = None, labels: torch.Tensor = None):
        logits = self.forward(X)
        logits = logits.squeeze(dim=1)
        return logits

    def on_train_epoch_start(self) -> None:
        # Placeholders for the two losses
        self.epoch_adv_loss = torch.tensor([0]).float().to(self.device)  # AdvLoss
        self.epoch_ce_loss = 0  # Cross entropy Loss

    def compute_loss(self, outputs: torch.Tensor, **kwargs):
        X, y = kwargs.get("X"), kwargs.get("y")
        # cross entropy loss
        ce_loss = F.binary_cross_entropy_with_logits(outputs, y)
        self.epoch_ce_loss += ce_loss

        if self.current_epoch >= self.only_ce_epochs:
            data = X[y == 0]
            # AdvLoss
            adv_loss = self.one_class_adv_loss(data).float()
            self.epoch_adv_loss += adv_loss
            loss = ce_loss + adv_loss * self.lamb
        else:
            # If only CE based training has to be done
            loss = ce_loss
        return loss

    def one_class_adv_loss(self, X: torch.Tensor):
        """Computes the adversarial loss:
        1) Sample points initially at random around the positive training
            data points
        2) Gradient ascent to find the most optimal point in set N_i(r)
            classified as +ve (label=0). This is done by maximizing
            the CE loss wrt label 0
        3) Project the points between spheres of radius R and gamma * R
            (set N_i(r))
        4) Pass the calculated adversarial points through the model,
            and calculate the CE loss wrt target class 0
        Parameters
        ----------
        X: torch.Tensor
            Batch of data to compute loss on.
        """
        batch_size = len(X)

        # Randomly sample points around the training data
        # We will perform SGD on these to find the adversarial points
        x_adv = torch.randn(X.shape).to(self.device).detach().requires_grad_()
        x_adv_sampled = x_adv + X
        for step in range(self.ascent_num_steps):
            with torch.enable_grad():
                new_targets = torch.zeros(batch_size, 1).to(self.device)
                if new_targets.squeeze().ndim > 0:
                    new_targets = torch.squeeze(new_targets)
                else:
                    new_targets = torch.zeros(batch_size).to(self.device)
                # new_targets = torch.zeros(batch_size, 1).to(self.device)
                # new_targets = torch.squeeze(new_targets)
                new_targets = new_targets.to(torch.float)

                logits = self.forward(x_adv_sampled)
                logits = torch.squeeze(logits, dim=1)

                new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)
                grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
                grad_norm = torch.norm(grad, p=2, dim=tuple(range(1, grad.dim())))
                grad_norm = grad_norm.view(-1, *[1] * (grad.dim() - 1))
                grad_norm[grad_norm == 0.0] = 10e-10
                grad_normalized = grad / grad_norm

            with torch.no_grad():
                x_adv_sampled.add_(self.ascent_step_size * grad_normalized)

            if (step + 1) % 10 == 0:
                # Project the normal points to the set N_i(r)
                h = x_adv_sampled - X
                norm_h = torch.sqrt(
                    torch.sum(h ** 2, dim=tuple(range(1, h.dim())))
                )
                alpha = torch.clamp(
                    norm_h, self.radius, self.gamma * self.radius
                ).to(self.device)
                # Make use of broadcast to project h
                proj = (alpha / norm_h).view(-1, *[1] * (h.dim() - 1))
                h = proj * h
                x_adv_sampled = X + h  # These adv_points are now on the surface of hyper-sphere

        adv_pred = self.forward(x_adv_sampled)
        adv_pred = torch.squeeze(adv_pred, dim=1)
        adv_loss = F.binary_cross_entropy_with_logits(adv_pred, (new_targets + 1))

        return adv_loss

    def training_step(self, X: torch.Tensor, y: torch.Tensor = None, labels: torch.Tensor = None):
        # Extract the logits for cross entropy loss
        logits = self.score(X)
        loss = self.compute_loss(logits, X=X, y=y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
        return optimizer, None  # [optimizer], [scheduler]
