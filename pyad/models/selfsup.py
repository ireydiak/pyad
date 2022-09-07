import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pyad.loss.criterion import TripletCenterLoss
from pyad.models.base import BaseModule
from pyad.utilities.cli import MODEL_REGISTRY
from torch.optim.lr_scheduler import StepLR
from typing import List, Tuple


def create_network(
        input_dim: int,
        hidden_dims: list,
        bias: bool = True,
        act_fn: nn.Module = nn.ReLU,
        batch_norm: bool = False
) -> list:
    net_layers = []
    for i in range(len(hidden_dims) - 1):
        net_layers.append(
            nn.Linear(input_dim, hidden_dims[i], bias=bias)
        )
        if batch_norm:
            net_layers.append(
                nn.BatchNorm1d(hidden_dims[i], affine=False)
            )
        net_layers.append(
            act_fn()
        )
        input_dim = hidden_dims[i]

    net_layers.append(
        nn.Linear(input_dim, hidden_dims[-1], bias=bias)
    )
    return net_layers


@MODEL_REGISTRY
class GOAD(BaseModule):

    """
    *** Code largely inspired by the original GOAD repository (https://github.com/lironber/GOAD) ***
    """
    def __init__(self,
                 n_transforms: int,
                 feature_dim: int,
                 num_hidden_nodes: int,
                 batch_size: int,
                 n_layers: int = 0,
                 eps: float = 0.,
                 lamb: float = 0.1,
                 margin: float = 1.,
                 **kwargs):
        super(GOAD, self).__init__(**kwargs)
        # model params
        self.n_transforms = n_transforms
        self.feature_dim = feature_dim
        self.num_hidden_nodes = num_hidden_nodes
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.eps = eps
        self.lamb = lamb
        self.margin = margin
        # build neural network
        self._build_network()
        # Cross entropy loss
        self.ce_loss = nn.CrossEntropyLoss()
        # Triplet loss
        self.tc_loss = TripletCenterLoss(margin=self.margin)
        # Transformation matrix
        trans_matrix = torch.randn(
            (self.n_transforms, self.in_features, self.feature_dim),
            dtype=torch.float, device=self.device
        )
        self.trans_matrix = nn.Parameter(trans_matrix)
        # Hypersphere centers
        centers = torch.empty(
            (self.num_hidden_nodes, self.n_transforms),
            dtype=torch.float, device=self.device
        )
        self.centers = torch.nn.Parameter(centers, requires_grad=False)
        # used to compute hypersphere centers
        self.n_batch = 0

    def _build_network(self):
        trunk_layers = [
            nn.Conv1d(self.feature_dim, self.num_hidden_nodes, kernel_size=1, bias=False)
        ]
        for i in range(0, self.n_layers):
            trunk_layers.append(
                nn.Conv1d(self.num_hidden_nodes, self.num_hidden_nodes, kernel_size=1, bias=False),
            )
            if i < self.n_layers - 1:
                trunk_layers.append(
                    nn.LeakyReLU(0.2, inplace=True),
                )
            else:
                trunk_layers.append(
                    nn.Conv1d(self.num_hidden_nodes, self.num_hidden_nodes, kernel_size=1, bias=False),
                )
        self.trunk = nn.Sequential(
            *trunk_layers
        ).to(self.device)
        self.head = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.num_hidden_nodes, self.n_transforms, kernel_size=1, bias=True),
        ).to(self.device)

    def print_name(self) -> str:
        return "GOAD"

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.5, 0.999)
        )
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.9)

        return optimizer, None  # [optimizer], [scheduler]

    def forward(self, X: torch.Tensor):
        # (batch_size, num_hidden_nodes, n_transforms)
        tc = self.trunk(X)
        # (batch_size, n_transforms, n_transforms)
        logits = self.head(tc)
        return tc, logits

    def score(self, X: torch.Tensor, y: torch.Tensor = None, labels: torch.Tensor = None):
        diffs = ((X.unsqueeze(2) - self.centers) ** 2).sum(-1)
        diffs_eps = self.eps * torch.ones_like(diffs)
        diffs = torch.max(diffs, diffs_eps)
        logp_sz = torch.nn.functional.log_softmax(-diffs, dim=2)
        score = -torch.diagonal(logp_sz, 0, 1, 2).sum(dim=1)
        return score

    def predict_step(self, X: torch.Tensor, y: torch.Tensor = None, labels: torch.Tensor = None):
        # Apply affine transformations
        X_augmented = torch.vstack(
            [X @ t for t in self.trans_matrix]
        ).reshape(X.shape[0], self.feature_dim, self.n_transforms)
        # Forward pass & reshape
        zs, fs = self.forward(X_augmented)
        zs = zs.permute(0, 2, 1)
        # Compute anomaly score
        scores = self.score(zs)
        return scores

    def training_step(self, X: torch.Tensor, y: torch.Tensor = None, labels: torch.Tensor = None):
        # transformation labels
        labels = torch.arange(
            self.n_transforms
        ).unsqueeze(0).expand((len(X), self.n_transforms)).long().to(self.device)

        # Apply affine transformations
        X_augmented = torch.vstack(
            [X @ t for t in self.trans_matrix]
        ).reshape(X.shape[0], self.feature_dim, self.n_transforms).to(self.device)
        # Forward pass
        tc_zs, logits = self.forward(X_augmented)
        # Update enters estimates
        self.centers += tc_zs.mean(0).data
        # Update batch count for computing centers means
        self.n_batch += 1
        # Compute loss
        loss = self.compute_loss(logits, labels=labels, tc_zs=tc_zs)

        return loss

    def compute_loss(self, outputs, **kwargs):
        labels = kwargs.get("labels")
        tc_zs = kwargs.get("tc_zs")
        ce_loss = self.ce_loss(outputs, labels)
        tc_loss = self.tc_loss(tc_zs)
        loss = self.lamb * tc_loss + ce_loss
        return loss

    def on_train_epoch_start(self) -> None:
        centers = torch.zeros(
            (self.num_hidden_nodes, self.n_transforms),
            dtype=torch.float, device=self.device
        )
        self.centers = torch.nn.Parameter(centers, requires_grad=False)
        self.n_batch = 0

    def on_train_epoch_end(self) -> None:
        centers = (self.centers.mT / self.n_batch).unsqueeze(0).float().to(self.device)
        self.centers = torch.nn.Parameter(centers, requires_grad=False)

    def get_hparams(self):
        return dict(
            n_transforms=self.n_transforms,
            feature_dim=self.feature_dim,
            num_hidden_nodes=self.num_hidden_nodes,
            n_layers=self.n_layers,
            eps=self.eps,
            lamb=self.lamb,
            margin=self.margin,
        )


@MODEL_REGISTRY
class NeuTraLAD(BaseModule):
    """
    *** Code largely inspired by the original repository: https://github.com/boschresearch/NeuTraL-AD/ ***
    """
    def __init__(
            self,
            n_transforms: int,
            trans_type: str,
            temperature: float,
            trans_hidden_dims: List[int],
            enc_hidden_dims: List[int],
            use_batch_norm: bool = False,
            **kwargs
    ):
        super(NeuTraLAD, self).__init__(**kwargs)
        # Model parameters
        self.n_transforms = n_transforms
        self.trans_type = trans_type
        self.temperature = temperature
        self.trans_hidden_dims = trans_hidden_dims
        self.enc_hidden_dims = enc_hidden_dims
        self.latent_dim = enc_hidden_dims[-1]
        self.use_batch_norm = use_batch_norm
        # Loss Module
        self.cosim = nn.CosineSimilarity()
        # Encoder and Transformation layers
        self._build_network()
        # Transforms
        self.masks = self._create_masks()

    def print_name(self) -> str:
        return "NeuTraLAD"

    def get_hparams(self) -> dict:
        return dict(
            n_transforms=self.n_transforms,
            trans_type=self.trans_type,
            temperature=self.temperature,
            trans_hidden_dims=self.trans_hidden_dims,
            enc_hidden_dims=self.enc_hidden_dims,
            use_batch_norm=self.use_batch_norm
        )

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, StepLR]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
        return optimizer, scheduler

    def _build_network(self):
        if self.trans_hidden_dims[-1] != self.in_features:
            self.trans_hidden_dims.append(self.in_features)
        # Encoder
        enc_layers = create_network(
            self.in_features, self.enc_hidden_dims,
            bias=False, batch_norm=self.use_batch_norm
        )
        # for some reason, this NeuTraLAD performs better when the last layer is the activation function
        # enc_layers = enc_layers[:-1]
        enc = nn.Sequential(
            *enc_layers
        ).to(self.device)
        self.enc = enc

    def _create_masks(self):
        masks = nn.ModuleList()
        for k_i in range(self.n_transforms):
            layers = create_network(self.in_features, self.trans_hidden_dims, bias=False)
            if self.trans_type == "mul":
                layers.append(nn.Sigmoid())
            masks.append(
                nn.Sequential(*layers).to(self.device)
            )
        return masks

    def forward(self, X: torch.Tensor):
        X_augmented = torch.empty(X.shape[0], self.n_transforms, X.shape[-1]).to(X.device)
        for k in range(self.n_transforms):
            mask = self.masks[k](X)
            if self.trans_type == "mul":
                X_augmented[:, k] = mask * X
            else:
                X_augmented[:, k] = mask + X
        X_augmented = torch.cat((
            X.unsqueeze(1), X_augmented
        ), 1).to(self.device)
        X_augmented = X_augmented.reshape(-1, X.shape[-1])

        emb = self.enc(X_augmented)
        # (batch_size, n_transforms + 1, latent_dim)
        emb = emb.reshape(X.shape[0], self.n_transforms + 1, self.latent_dim)

        return emb

    def compute_loss(self, outputs: torch.Tensor, **kwargs):
        logits = F.normalize(outputs, p=2, dim=-1)
        emb_ori = logits[:, 0]
        emb_trans = logits[:, 1:]
        batch_size, n_transforms, latent_dim = logits.shape

        sim_matrix = torch.exp(
            torch.matmul(
                logits,
                logits.permute(0, 2, 1) / self.temperature
            )
        )
        mask = (torch.ones_like(sim_matrix).to(self.device) - torch.eye(n_transforms).unsqueeze(0).to(
            self.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size, n_transforms, -1)
        trans_matrix = sim_matrix[:, 1:].sum(-1)

        pos_sim = torch.exp(
            torch.sum(emb_trans * emb_ori.unsqueeze(1), -1) / self.temperature
        )
        K = n_transforms - 1
        scale = 1 / np.abs(K * np.log(1. / K))

        loss = torch.log(
            trans_matrix - torch.log(pos_sim)
        ) * scale

        return loss.sum(1)

    def training_step(self, X: torch.Tensor, y: torch.Tensor = None, labels: torch.Tensor = None):
        emb = self(X)
        loss = self.compute_loss(emb)
        loss = loss.mean()
        return loss

    def score(self, X: torch.Tensor, y: torch.Tensor = None, labels: torch.Tensor = None):
        emb = self(X)
        scores = self.compute_loss(emb)
        return scores

    def _computeH_ij(self, Z):
        hij = F.cosine_similarity(Z.unsqueeze(1), Z.unsqueeze(0), dim=2)
        exp_hij = torch.exp(
            hij / self.temperature
        )
        return exp_hij

    def _computeBatchH_ij(self, Z):
        hij = F.cosine_similarity(Z.unsqueeze(2), Z.unsqueeze(1), dim=3)
        exp_hij = torch.exp(
            hij / self.temperature
        )
        return exp_hij

    def _computeH_x_xk(self, z, zk):
        hij = F.cosine_similarity(z.unsqueeze(0), zk)
        exp_hij = torch.exp(
            hij / self.temperature
        )
        return exp_hij

    def _computeBatchH_x_xk(self, z, zk):
        hij = F.cosine_similarity(z.unsqueeze(1), zk, dim=2)
        exp_hij = torch.exp(
            hij / self.temperature
        )
        return exp_hij

    def _computeX_k(self, X):
        X_t_s = []

        def transform(trans_type, x):
            if trans_type == 'res':
                return lambda mask, X: mask(X) + X
            else:
                return lambda mask, X: mask(X) * X

        t_function = transform(self.trans_type, X)
        for k in range(self.n_transforms):
            X_t_k = t_function(self.masks[k], X)
            X_t_s.append(X_t_k)
        X_t_s = torch.stack(X_t_s, dim=0)

        return X_t_s
