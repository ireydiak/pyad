import torch

from pyad.models.base import BaseModule
from pyad.utilities import weights_init_xavier
from pyad.utilities.cli import MODEL_REGISTRY
from torch.autograd import Variable
from torch import nn


@MODEL_REGISTRY
class ALAD(BaseModule):

    def __init__(
            self,
            latent_dim: int,
            out_dim: int = 128,
            negative_slope: float = 0.2,
            **kwargs
    ):
        super(ALAD, self).__init__(**kwargs)
        # model parameters
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.negative_slope = negative_slope
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        self._build_network()

    def _build_network(self):
        self.D_xz = DiscriminatorXZ(
            self.in_features,
            self.out_dim,
            self.latent_dim,
            negative_slope=self.negative_slope,
            p=0.5
        ).to(self.device)
        self.D_xx = DiscriminatorXX(
            self.in_features,
            self.out_dim,
            negative_slope=self.negative_slope,
            p=0.2
        ).to(self.device)
        self.D_zz = DiscriminatorZZ(
            self.latent_dim,
            self.latent_dim,
            negative_slope=self.negative_slope,
            p=0.2
        ).to(self.device)
        self.G = Generator(
            self.latent_dim,
            self.in_features,
            negative_slope=1e-4
        ).to(self.device)
        self.E = Encoder(
            self.in_features,
            self.latent_dim
        ).to(self.device)
        self.D_xz.apply(weights_init_xavier)
        self.D_xx.apply(weights_init_xavier)
        self.D_zz.apply(weights_init_xavier)
        self.G.apply(weights_init_xavier)
        self.E.apply(weights_init_xavier)

    def get_hparams(self) -> dict:
        return dict(
            latent_dim=self.latent_dim,
            out_dim=self.out_dim,
            negative_slope=self.negative_slope
        )

    def print_name(self) -> str:
        return "ALAD"

    def compute_loss(self, outputs: torch.Tensor, **kwargs):
        pass

    def score(self, X: torch.Tensor, y: torch.Tensor = None, labels: torch.Tensor = None):
        _, feature_real = self.D_xx(X, X)
        _, feature_gen = self.D_xx(X, self.G(self.E(X)))
        return torch.linalg.norm(feature_real - feature_gen, 2, keepdim=False, dim=1)

    def forward(self, X: torch.Tensor):
        # Encoder
        z_gen = self.E(X)

        # Generator
        z_real = Variable(
            torch.randn((X.size(0), self.latent_dim), device=self.device),
            requires_grad=False
        )
        x_gen = self.G(z_real)

        # DiscriminatorXZ
        out_truexz, _ = self.D_xz(X, z_gen)
        out_fakexz, _ = self.D_xz(x_gen, z_real)

        # DiscriminatorZZ
        out_truezz, _ = self.D_zz(z_real, z_real)
        out_fakezz, _ = self.D_zz(z_real, self.E(self.G(z_real)))

        # DiscriminatorXX
        out_truexx, _ = self.D_xx(X, X)
        out_fakexx, _ = self.D_xx(X, self.G(self.E(X)))

        return out_truexz, out_fakexz, out_truezz, out_fakezz, out_truexx, out_fakexx

    def compute_d_loss(self, X: torch.Tensor):
        # Labels
        y_true = Variable(torch.zeros(X.size(0), 1)).to(self.device)
        y_fake = Variable(torch.ones(X.size(0), 1)).to(self.device)
        # Forward pass
        out_truexz, out_fakexz, out_truezz, out_fakezz, out_truexx, out_fakexx = self.forward(X)
        # Compute discriminator losses
        loss_dxz = self.criterion(out_truexz, y_true) + self.criterion(out_fakexz, y_fake)
        loss_dzz = self.criterion(out_truezz, y_true) + self.criterion(out_fakezz, y_fake)
        loss_dxx = self.criterion(out_truexx, y_true) + self.criterion(out_fakexx, y_fake)
        loss_d = loss_dxz + loss_dzz + loss_dxx

        return loss_d

    def compute_g_loss(self, X: torch.Tensor):
        # Labels
        y_true = Variable(torch.zeros(X.size(0), 1)).to(self.device)
        y_fake = Variable(torch.ones(X.size(0), 1)).to(self.device)
        # Forward pass
        out_truexz, out_fakexz, out_truezz, out_fakezz, out_truexx, out_fakexx = self.forward(X)
        # Compute generator losses
        loss_gexz = self.criterion(out_fakexz, y_true.clone()) + self.criterion(out_truexz, y_fake.clone())
        loss_gezz = self.criterion(out_fakezz, y_true.clone()) + self.criterion(out_truezz, y_fake.clone())
        loss_gexx = self.criterion(out_fakexx, y_true.clone()) + self.criterion(out_truexx, y_fake.clone())
        cycle_consistency = loss_gexx + loss_gezz
        loss_g = loss_gexz + cycle_consistency

        return loss_g

    def training_step(self, X: torch.Tensor, y: torch.Tensor = None, labels: torch.Tensor = None):
        # # manual optimization: fetch optimizers
        # g_opt, d_opt = self.optimizers()

        X_g, X_d = X.to(self.device).float(), X.clone().to(self.device).float()

        # Forward pass
        loss_g = self.compute_g_loss(X_g)
        loss_d = self.compute_d_loss(X_d)

        return loss_g, loss_d
        # # Optimize generator
        # g_opt.zero_grad()
        # self.manual_backward(loss_g)
        # g_opt.step()
        #
        # # Optimize discriminator
        # d_opt.zero_grad()
        # self.manual_backward(loss_d)
        # d_opt.step()
        #
        # # log both losses
        # self.log_dict({
        #     "g_loss": loss_g.item(),
        #     "d_loss": loss_d.item()
        # })

    def configure_optimizers(self):
        optim_ge = torch.optim.Adam(
            list(self.G.parameters()) + list(self.E.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )
        optim_d = torch.optim.Adam(
            list(self.D_xz.parameters()) +
            list(self.D_zz.parameters()) +
            list(self.D_xx.parameters()),
            lr=self.lr,
            betas=(0.5, 0.999)
        )
        return (optim_ge, optim_d), None


class Encoder(nn.Module):
    def __init__(
            self,
            in_features: int,
            latent_dim: int,
            negative_slope: float = 0.2
    ):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.fc_1 = nn.Sequential(
            nn.Linear(in_features, latent_dim * 2),
            nn.LeakyReLU(negative_slope),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def forward(self, X):
        return self.fc_1(X)


class Generator(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            feature_dim: int,
            negative_slope: float = 1e-4
    ):
        super(Generator, self).__init__()
        self.fc_1 = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim)
        )

    def forward(self, Z):
        return self.fc_1(Z)


class DiscriminatorZZ(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            negative_slope: float = 0.2,
            p: float = 0.5,
            n_classes: int = 1
    ):
        super(DiscriminatorZZ, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.negative_slope = negative_slope
        self.p = p
        self.n_classes = n_classes
        self._build_network()

    def _build_network(self):
        self.fc_1 = nn.Sequential(
            nn.Linear(2 * self.in_features, self.out_features),
            nn.LeakyReLU(self.negative_slope),
            nn.Dropout(self.p)
        )
        self.fc_2 = nn.Linear(self.out_features, self.n_classes)

    def forward(self, Z, rec_Z):
        ZZ = torch.cat((Z, rec_Z), dim=1)
        mid_layer = self.fc_1(ZZ)
        logits = self.fc_2(mid_layer)
        return logits, mid_layer


class DiscriminatorXX(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            negative_slope: float = 0.2,
            p: float = 0.5,
            n_classes: int = 1
    ):
        super(DiscriminatorXX, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes
        self.negative_slope = negative_slope
        self.p = p
        self._build_network()

    def _build_network(self):
        self.fc_1 = nn.Sequential(
            nn.Linear(self.in_features * 2, self.out_features),
            nn.LeakyReLU(self.negative_slope),
            nn.Dropout(self.p)
        )
        self.fc_2 = nn.Linear(
            self.out_features, self.n_classes
        )

    def forward(self, X, rec_X):
        XX = torch.cat((X, rec_X), dim=1)
        mid_layer = self.fc_1(XX)
        logits = self.fc_2(mid_layer)
        return logits, mid_layer


class DiscriminatorXZ(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            latent_dim: int,
            negative_slope: float = 0.2,
            p: float = 0.5,
            n_classes: int = 1
    ):
        super(DiscriminatorXZ, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.negative_slope = negative_slope
        self.p = p
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self._build_network()

    def _build_network(self):
        # Inference over x
        self.fc_1x_linear = nn.Linear(self.in_features, self.out_features)
        self.fc_1x_bnorm = nn.BatchNorm1d(self.out_features)
        self.fc_1x_lrelu = nn.LeakyReLU(self.negative_slope)
        # Inference over z
        self.fc_1z = nn.Sequential(
            nn.Linear(self.latent_dim, self.out_features),
            nn.LeakyReLU(self.negative_slope),
            nn.Dropout(self.p)
        )
        # Joint inference
        self.fc_1xz = nn.Sequential(
            nn.Linear(2 * self.out_features, self.out_features),
            nn.LeakyReLU(self.negative_slope),
            nn.Dropout(self.p)
        )
        self.fc_2xz = nn.Linear(self.out_features, self.n_classes)

    def forward_xz(self, xz):
        mid_layer = self.fc_1xz(xz)
        logits = self.fc_2xz(mid_layer)
        return logits, mid_layer

    def forward(self, X, Z):
        x = self.fc_1x_linear(X)
        if x.shape[0] > 1:
            x = self.fc_1x_bnorm(x)
        x = self.fc_1x_lrelu(x)
        z = self.fc_1z(Z)
        xz = torch.cat((x, z), dim=1)
        logits, mid_layer = self.forward_xz(xz)
        return logits, mid_layer
