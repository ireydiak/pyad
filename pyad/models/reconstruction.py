import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

from pyad.loss.criterion import EntropyLoss
from pyad.models.base import BaseModule, create_net_layers
from typing import List, Tuple, Optional

from pyad.models.memory_module import MemoryUnit
from pyad.utilities.cli import MODEL_REGISTRY


@MODEL_REGISTRY
class AutoEncoder(BaseModule):
    def __init__(
            self,
            hidden_dims: List[int],
            latent_dim: int,
            activation: str,
            reg: float = 0.5,
            **kwargs
    ):
        super(AutoEncoder, self).__init__(**kwargs)
        self.encoder = None
        self.decoder = None
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.activation = activation
        self.reg = reg
        self._build_network()

    def _build_network(self):
        self.encoder = nn.Sequential(
            *create_net_layers(
                self.in_features,
                self.latent_dim,
                self.hidden_dims,
                activation=self.activation
            )
        ).to(self.device)
        self.decoder = nn.Sequential(
            *create_net_layers(
                self.latent_dim,
                self.in_features,
                list(reversed(self.hidden_dims)),
                activation=self.activation
            )
        ).to(self.device)

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.StepLR]]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer, None

    def print_name(self) -> str:
        return "autoencoder"

    def get_hparams(self):
        params = {
            "hidden_dims": "[" + ",".join(map(lambda x: str(x), self.hidden_dims)) + "]",
            "latent_dim": self.latent_dim,
            "activation": self.activation,
            "reg": self.reg,
        }
        return params

    def forward(self, X: torch.Tensor):
        emb = self.encoder(X)
        X_hat = self.decoder(emb)
        return X_hat

    def compute_loss(self, outputs: torch.Tensor, **kwargs):
        X = kwargs.get("X")
        emb = kwargs.get("emb")
        loss = ((X - outputs) ** 2).sum(axis=-1).mean() + self.reg * emb.norm(2, dim=1).mean()
        return loss

    def score(self, X: torch.Tensor, y: torch.Tensor = None, labels: torch.Tensor = None) -> torch.Tensor:
        X_hat = self(X)
        score = ((X - X_hat) ** 2).sum(axis=-1)
        return score

    def training_step(self, X: torch.Tensor, y: torch.Tensor = None, labels: torch.Tensor = None):
        emb = self.encoder(X)
        X_hat = self.decoder(emb)
        loss = self.compute_loss(X_hat, X=X_hat, emb=emb)
        return loss


@MODEL_REGISTRY
class MemAE(BaseModule):

    """
    *** Code largely inspired by the original repository for MemAE (https://github.com/donggong1/memae-anomaly-detection/) ***
    """
    def __init__(
            self,
            mem_dim: int,
            latent_dim: int,
            enc_hidden_dims: List[int],
            shrink_thresh: float,
            alpha: float,
            activation="relu",
            **kwargs
    ):
        super(MemAE, self).__init__(**kwargs)
        # model params
        self.mem_dim = mem_dim
        self.latent_dim = latent_dim
        self.enc_hidden_dims = enc_hidden_dims
        self.shrink_thresh = shrink_thresh
        self.alpha = alpha
        self.activation = activation
        # loss modules
        self.recon_loss_fn = nn.MSELoss().to(self.device)
        self.entropy_loss_fn = EntropyLoss().to(self.device)
        # build neural networks
        self._build_network()

    def _build_network(self):
        # encoder-decoder network
        self.encoder = nn.Sequential(*create_net_layers(
            in_dim=self.in_features,
            out_dim=self.latent_dim,
            hidden_dims=self.enc_hidden_dims,
            activation=self.activation
        )).to(self.device)
        # xavier_init(self.encoder)
        self.decoder = nn.Sequential(*create_net_layers(
            in_dim=self.latent_dim,
            out_dim=self.in_features,
            hidden_dims=list(reversed(self.enc_hidden_dims)),
            activation=self.activation
        )).to(self.device)
        # xavier_init(self.decoder)
        # memory module
        self.mem_rep = MemoryUnit(
            self.mem_dim,
            self.latent_dim,
            self.shrink_thresh,
        ).to(self.device)

    def print_name(self) -> str:
        return "memae"

    def compute_loss(self, outputs: torch.Tensor, **kwargs):
        X, W_hat = kwargs.get("X"), kwargs.get("W_hat")
        R = self.recon_loss_fn(X, outputs)
        E = self.entropy_loss_fn(W_hat)
        loss = R + (self.alpha * E)
        return loss

    def forward(self, X: torch.Tensor):
        f_e = self.encoder(X)
        f_mem, att = self.mem_rep(f_e)
        f_d = self.decoder(f_mem)
        return f_d, att

    def training_step(self, X: torch.Tensor, y: torch.Tensor = None, labels: torch.Tensor = None):
        X_hat, W_hat = self(X)
        return self.compute_loss(X_hat, X=X, W_hat=W_hat)

    def score(self, X: torch.Tensor, y: torch.Tensor = None, labels: torch.Tensor = None):
        X_hat, _ = self.forward(X)
        return torch.sum((X - X_hat) ** 2, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
        return optimizer, None  # [optimizer], [scheduler]

    def get_hparams(self) -> dict:
        return dict(
            mem_dim=self.mem_dim,
            latent_dim=self.latent_dim,
            enc_hidden_dims=self.enc_hidden_dims,
            shrink_thresh=self.shrink_thresh,
            alpha=self.alpha,
            activation=self.activation
        )
