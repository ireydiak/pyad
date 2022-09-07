import torch
import numpy as np

from minisom import MiniSom
from torch import nn
from torch.utils.data import DataLoader
from .reconstruction import AutoEncoder   # keep unused import for registry
from pyad.models.base import BaseModule, create_net_layers
from typing import List, Any, Tuple, Optional
from pyad.utilities.cli import MODEL_REGISTRY
from pyad.utilities.math import relative_euclidean_dist


@MODEL_REGISTRY
class DSEBM(BaseModule):

    def __init__(
            self,
            fc_1_out: int,
            fc_2_out: int,
            batch_size: int,
            score_metric="reconstruction",
            b_prime: torch.Tensor = None,
            **kwargs
    ):
        super(DSEBM, self).__init__(**kwargs)
        score_metrics_opts = {"reconstruction", "energy"}
        # energy or reconstruction-based anomaly score function
        assert score_metric in score_metrics_opts, "unknown `score_metric` %s, please select %s" % (
            score_metric, score_metrics_opts)
        # loss function
        self.criterion = nn.BCEWithLogitsLoss()
        # model params
        self.batch_size = batch_size
        self.fc_1_out = fc_1_out
        self.fc_2_out = fc_2_out
        self.score_metric = score_metric
        # create optimizable param
        self.b_prime = b_prime or torch.nn.Parameter(
            torch.empty(self.batch_size, self.in_features, dtype=torch.float, device=self.device)
        )
        torch.nn.init.xavier_normal_(self.b_prime)
        # build neural network
        self._build_network()

    def print_name(self) -> str:
        return "DSEBM-e" if self.score_metric == "energy" else "DSEBM-r"

    def _build_network(self):
        # TODO: Make model more flexible. Users should be able to set the number of layers
        self.fc_1 = nn.Linear(self.in_features, self.fc_1_out).to(self.device)
        self.fc_2 = nn.Linear(self.fc_1_out, self.fc_2_out).to(self.device)
        self.softp = nn.Softplus().to(self.device)
        self.bias_inv_1 = torch.nn.Parameter(torch.Tensor(self.fc_1_out)).to(self.device)
        self.bias_inv_2 = torch.nn.Parameter(torch.Tensor(self.in_features)).to(self.device)
        torch.nn.init.xavier_normal_(self.fc_2.weight)
        torch.nn.init.xavier_normal_(self.fc_1.weight)
        self.fc_1.bias.data.zero_()
        self.fc_2.bias.data.zero_()
        self.bias_inv_1.data.zero_()
        self.bias_inv_2.data.zero_()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.5, 0.999)
        )
        return optimizer, None

    def random_noise_like(self, X: torch.Tensor):
        return torch.normal(mean=0., std=1., size=X.shape).float().to(self.device)

    def forward(self, X: torch.Tensor):
        output = self.softp(self.fc_1(X))
        output = self.softp(self.fc_2(output))

        # inverse layer
        output = self.softp((output @ self.fc_2.weight) + self.bias_inv_1)
        output = self.softp((output @ self.fc_1.weight) + self.bias_inv_2)

        return output

    def energy(self, X, X_hat):
        diff = self.b_prime.shape[0] - X.shape[0]
        if diff > 0:
            energy = 0.5 * torch.sum(torch.square(X - self.b_prime[:X.shape[0]])) - torch.sum(X_hat)
        else:
            energy = 0.5 * torch.sum(torch.square(X - self.b_prime)) - torch.sum(X_hat)

        return energy

    def compute_loss(self, outputs: torch.Tensor, **kwargs):
        X = kwargs.get("X")
        out = torch.square(X - outputs)
        out = torch.sum(out, dim=-1)
        out = torch.mean(out)
        return out

    def score(self, X: torch.Tensor, y: torch.Tensor = None, labels: torch.Tensor = None):
        # TODO: add support for multi-score
        if self.score_metric == "energy":
            # Evaluation of the score based on the energy
            with torch.no_grad():
                diff = self.b_prime.shape[0] - X.shape[0]
                if diff > 0:
                    flat = X - self.b_prime[:X.shape[0]]
                else:
                    flat = X - self.b_prime
                out = self.forward(X)
                energies = 0.5 * torch.sum(torch.square(flat), dim=1) - torch.sum(out, dim=1)
                scores = energies
        else:
            # Evaluation of the score based on the reconstruction error
            torch.set_grad_enabled(True)
            X.requires_grad_()
            out = self.forward(X)
            energy = self.energy(X, out)
            dEn_dX = torch.autograd.grad(energy, X)[0]
            rec_errs = torch.linalg.norm(dEn_dX, 2, keepdim=False, dim=1)
            scores = rec_errs
        return scores

    def training_step(self, X: torch.Tensor, y: torch.Tensor = None, labels: torch.Tensor = None):
        # add noise to input data
        noise = self.random_noise_like(X).to(self.device)
        X_noise = X + noise
        X.requires_grad_()
        X_noise.requires_grad_()
        # forward pass on noisy input
        out_noise = self.forward(X_noise)
        energy_noise = self.energy(X_noise, out_noise)
        # compute gradient
        dEn_dX = torch.autograd.grad(energy_noise, X_noise, retain_graph=True, create_graph=True)
        fx_noise = (X_noise - dEn_dX[0])

        return self.compute_loss(fx_noise, X=X)

    def on_test_model_eval(self) -> None:
        torch.set_grad_enabled(True)

    def get_hparams(self) -> dict:
        return dict(
            fc_1_out=self.fc_1_out,
            fc_2_out=self.fc_2_out,
            score_metric=self.score_metric
        )


@MODEL_REGISTRY
class DAGMM(BaseModule):

    def __init__(
            self,
            n_mixtures: int,
            ae_hidden_dims: List[int],
            gmm_hidden_dims: List[int],
            ae_activation: str = "relu",
            gmm_activation: str = "tanh",
            latent_dim: int = 1,
            lamb_1: float = 0.1,
            lamb_2: float = 0.005,
            reg_covar: float = 1e-12,
            dropout_rate: float = 0.5,
            **kwargs: Any
    ):
        super(DAGMM, self).__init__(**kwargs)
        # model parameters
        self.n_mixtures = n_mixtures
        self.ae_hidden_dims = ae_hidden_dims
        self.gmm_hidden_dims = gmm_hidden_dims
        self.ae_activation = ae_activation
        self.gmm_activation = gmm_activation
        self.latent_dim = latent_dim
        self.lamb_1 = lamb_1
        self.lamb_2 = lamb_2
        self.reg_covar = reg_covar
        self.dropout_rate = dropout_rate
        # computed parameters
        self.phi = torch.nn.Parameter(
            torch.empty((self.n_mixtures,), dtype=torch.float, device=self.device)
        )
        self.mu = torch.nn.Parameter(
            torch.empty((self.n_mixtures, self.latent_dim + 2), dtype=torch.float, device=self.device)
        )
        self.cov_mat = torch.nn.Parameter(
            torch.empty(
                (self.n_mixtures, self.latent_dim + 2, self.latent_dim + 2),
                dtype=torch.float, device=self.device
            )
        )
        self.cosim = nn.CosineSimilarity().to(self.device)
        self.softmax = nn.Softmax(dim=1).to(self.device)
        # build network
        self._build_network()

    def _build_network(self):
        # GMM network
        gmm_layers = create_net_layers(
            in_dim=self.latent_dim + 2,
            out_dim=self.n_mixtures,
            activation=self.gmm_activation,
            hidden_dims=self.gmm_hidden_dims,
            dropout=self.dropout_rate
        )
        self.gmm_net = nn.Sequential(
            *gmm_layers
        ).to(self.device)
        # AutoEncoder network
        self.encoder = nn.Sequential(
            *create_net_layers(
                self.in_features,
                self.latent_dim,
                self.ae_hidden_dims,
                activation=self.ae_activation
            )
        ).to(self.device)
        self.decoder = nn.Sequential(
            *create_net_layers(
                self.latent_dim,
                self.in_features,
                list(reversed(self.ae_hidden_dims)),
                activation=self.ae_activation
            )
        ).to(self.device)

    def print_name(self) -> str:
        return "DAGMM"

    def forward(self, X: torch.Tensor, return_gamma_hat=True):
        # computes the z vector of the original paper (p.4), that is
        # :math:`z = [z_c, z_r]` with
        #   - :math:`z_c = h(x; \theta_e)`
        #   - :math:`z_r = f(x, x')`
        emb = self.encoder(X)
        X_hat = self.decoder(emb)
        rel_euc_dist = relative_euclidean_dist(X, X_hat)
        cosim = self.cosim(X, X_hat)
        z = torch.cat(
            [emb, rel_euc_dist.unsqueeze(-1), cosim.unsqueeze(-1)],
            dim=1
        ).to(self.device)
        if return_gamma_hat:
            # compute gmm net output, that is
            #   - p = MLP(z, \theta_m) and
            #   - \gamma_hat = softmax(p)
            gamma_hat = self.gmm_net(z)
            gamma_hat = self.softmax(gamma_hat)
        else:
            gamma_hat = None
        return emb, X_hat, z, gamma_hat

    def forward_estimation_net(self, Z: torch.Tensor):
        gamma_hat = self.gmm_net(Z)
        gamma_hat = self.softmax(gamma_hat)
        return gamma_hat

    def training_step(self, X: torch.Tensor, y: torch.Tensor = None, labels: torch.Tensor = None):
        # forward pass:
        # - embed and reconstruct original sample
        # - create new feature matrix from embeddings and reconstruction error
        # - pass later input to GMM-MLP
        z_c, x_prime, z, gamma_hat = self.forward(X)
        # estimate phi, mu and covariance matrices
        phi, mu, cov_mat = self.compute_params(z, gamma_hat)
        # estimate energy
        energy_result, pen_cov_mat = self.estimate_sample_energy(
            z, phi, mu, cov_mat
        )
        self.phi = torch.nn.Parameter(phi)
        self.mu = torch.nn.Parameter(mu)
        self.cov_mat = torch.nn.Parameter(cov_mat)
        loss = self.compute_loss(x_prime, X=X, energy=energy_result, pen_cov_mat=pen_cov_mat)
        return loss

    def compute_loss(self, outputs: torch.Tensor, **kwargs):
        X, energy, pen_cov_mat = kwargs.get("X"), kwargs.get("energy"), kwargs.get("pen_cov_mat")
        rec_err = (X - outputs) ** 2
        loss = rec_err.mean() + (self.lamb_1 / self.n_instances) * energy + self.lamb_2 * pen_cov_mat
        return loss

    def weighted_log_sum_exp(self, x, weights, dim):
        """
        Inspired by https://discuss.pytorch.org/t/moving-to-numerically-stable-log-sum-exp-leads-to-extremely-large-loss-values/61938

        Parameters
        ----------
        x
        weights
        dim

        Returns
        -------

        """
        m, idx = torch.max(x, dim=dim, keepdim=True)
        return m.squeeze(dim) + torch.log(torch.sum(torch.exp(x - m) * (weights.unsqueeze(2)), dim=dim))

    def compute_params(self, z: torch.Tensor, gamma_hat: torch.Tensor):
        r"""
        Estimates the parameters of the GMM.
        Implements the following formulas (p.5):
            :math:`\hat{\phi_k} = \sum_{i=1}^N \frac{\hat{\gamma_{ik}}}{N}`
            :math:`\hat{\mu}_k = \frac{\sum{i=1}^N \hat{\gamma_{ik} z_i}}{\sum{i=1}^N \hat{\gamma_{ik}}}`
            :math:`\hat{\Sigma_k} = \frac{
                \sum{i=1}^N \hat{\gamma_{ik}} (z_i - \hat{\mu_k}) (z_i - \hat{\mu_k})^T}
                {\sum{i=1}^N \hat{\gamma_{ik}}
            }`

        The second formula was modified to use matrices instead:
            :math:`\hat{\mu}_k = (I * \Gamma)^{-1} (\gamma^T z)`

        Parameters
        ----------
        z: N x D matrix (n_samples, n_features)
        gamma_hat: N x K matrix (n_samples, n_mixtures)


        Returns
        -------

        """
        N = z.shape[0]

        # gamma
        gamma_sum = gamma_hat.sum(dim=0)
        # gamma_sum /= gamma_sum.sum()

        # \phi \in (n_mixtures,)
        phi = gamma_sum / N

        # \mu \in (n_mixtures, z_dim)
        mu = torch.sum(gamma_hat.unsqueeze(-1) * z.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(-1)
        # mu = torch.linalg.inv(torch.diag(gamma_sum)) @ (gamma.T @ z)

        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)
        cov_mat = mu_z.unsqueeze(-1) @ mu_z.unsqueeze(-2)
        cov_mat = gamma_hat.unsqueeze(-1).unsqueeze(-1) * cov_mat
        cov_mat = torch.sum(cov_mat, dim=0) / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov_mat

    def estimate_sample_energy(self, z, phi=None, mu=None, cov_mat=None, average_energy=True):
        if phi is None:
            phi = self.phi
        if mu is None:
            mu = self.mu
        if cov_mat is None:
            cov_mat = self.cov_mat

        # Avoid non-invertible covariance matrix by adding small values (self.reg_covar)
        d = z.shape[1]
        cov_mat = cov_mat + (torch.eye(d)).to(self.device) * self.reg_covar
        # N x K x D
        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)

        # scaler
        inv_cov_mat = torch.cholesky_inverse(torch.linalg.cholesky(cov_mat))
        # inv_cov_mat = torch.linalg.inv(cov_mat)
        det_cov_mat = torch.linalg.cholesky(2 * np.pi * cov_mat)
        det_cov_mat = torch.diagonal(det_cov_mat, dim1=1, dim2=2)
        det_cov_mat = torch.prod(det_cov_mat, dim=1)

        exp_term = torch.matmul(mu_z.unsqueeze(-2), inv_cov_mat)
        exp_term = torch.matmul(exp_term, mu_z.unsqueeze(-1))
        exp_term = - 0.5 * exp_term.squeeze()

        # Applying log-sum-exp stability trick
        # https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        if exp_term.ndim == 1:
            exp_term = exp_term.unsqueeze(0)
        max_val = torch.max(exp_term.clamp(min=0), dim=1, keepdim=True)[0]
        exp_result = torch.exp(exp_term - max_val)

        log_term = phi * exp_result
        log_term /= det_cov_mat
        log_term = log_term.sum(axis=-1)

        # energy computation
        energy_result = - max_val.squeeze() - torch.log(log_term + self.reg_covar)

        if average_energy:
            energy_result = energy_result.mean()

        # penalty term
        cov_diag = torch.diagonal(cov_mat, dim1=1, dim2=2)
        pen_cov_mat = (1 / cov_diag).sum()

        return energy_result, pen_cov_mat

    def score(self, X: torch.Tensor, y: torch.Tensor = None, labels: torch.Tensor = None):
        _, _, z, _ = self.forward(X)
        energy, _ = self.estimate_sample_energy(z, average_energy=False)
        return energy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer, None

    def get_hparams(self) -> dict:
        return dict(
            n_mixtures=self.n_mixtures,
            ae_hidden_dims=self.ae_hidden_dims,
            gmm_hidden_dims=self.gmm_hidden_dims,
            ae_activation=self.ae_activation,
            gmm_activation=self.gmm_activation,
            latent_dim=self.latent_dim,
            lamb_1=self.lamb_1,
            lamb_2=self.lamb_2,
            reg_covar=self.reg_covar,
            dropout_rate=self.dropout_rate
        )


default_som_args = {
    "x": 32,
    "y": 32,
    "lr": 0.6,
    "neighborhood_function": "bubble",
    "n_epoch": 500,
    "n_som": 1
}


@MODEL_REGISTRY
class SOMDAGMM(BaseModule):

    def __init__(
            self,
            n_soms: int,
            dagmm: DAGMM,
            **kwargs):
        super(SOMDAGMM, self).__init__(**kwargs)
        self.n_soms = n_soms
        self.dagmm = dagmm
        self.som_args = None
        self.soms = None
        self.dagmm.mu = torch.nn.Parameter(
            torch.empty(
                (self.dagmm.n_mixtures, self.dagmm.latent_dim + 4),
                dtype=torch.float, device=self.device
            )
        )
        self.dagmm.cov_mat = torch.nn.Parameter(
            torch.empty(
                (self.dagmm.n_mixtures, self.dagmm.latent_dim + 4, self.dagmm.latent_dim + 4),
                dtype=torch.float, device=self.device
            )
        )
        self._build_network()

    def _build_network(self):
        # set these values according to the used dataset
        # Use 0.6 for KDD; 0.8 for IDS2018 with babel as neighborhood function as suggested in the paper.
        grid_length = int(np.sqrt(5 * np.sqrt(self.n_instances))) // 2
        grid_length = 32 if grid_length > 32 else grid_length
        self.som_args = {
            "x": grid_length,
            "y": grid_length,
            "lr": 0.6,
            "neighborhood_function": "bubble",
            "n_epoch": 8000,
            "n_som": self.n_soms
        }
        self.soms = [MiniSom(
            self.som_args['x'], self.som_args['y'], self.in_features,
            neighborhood_function=self.som_args['neighborhood_function'],
            learning_rate=self.som_args['lr']
        )] * self.som_args.get('n_som', 1)
        # Replace DAGMM's GMM network
        gmm_layers = create_net_layers(
            in_dim=self.n_soms * 2 + self.dagmm.latent_dim + 2,
            out_dim=self.dagmm.n_mixtures,
            activation=self.dagmm.gmm_activation,
            hidden_dims=[10],
            dropout=self.dagmm.dropout_rate
        )
        self.dagmm.gmm_hidden_dims = [10]
        self.dagmm.gmm_net = nn.Sequential(
            *gmm_layers
        ).to(self.device)

    def print_name(self) -> str:
        return "SOM-DAGMM"

    def score(self, X: torch.Tensor, y: torch.Tensor = None, labels: torch.Tensor = None):
        _, _, _, Z, _ = self(X)
        energy, _ = self.dagmm.estimate_sample_energy(Z, average_energy=False)
        return energy

    def get_hparams(self) -> dict:
        dagmm_params = self.dagmm.get_hparams()
        params = dict(
            **dagmm_params,
            n_soms=self.n_soms
        )
        for k, v in self.som_args.items():
            params[f'SOM-{k}'] = v
        return params

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.StepLR]]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer, None

    def on_before_fit(self, dataset: DataLoader):
        assert dataset.dataset.y.sum() == 0, "found anomalies in the training set, aborting"
        self.train_som(dataset.dataset.X)

    def train_som(self, X: torch.Tensor):
        # SOM-generated low-dimensional representation
        for i in range(len(self.soms)):
            self.soms[i].train(X, self.som_args["n_epoch"])

    def forward(self, X: torch.Tensor):
        # DAGMM's latent feature, the reconstruction error and gamma
        emb = self.dagmm.encoder(X)
        X_hat = self.dagmm.decoder(emb)
        cosim = self.dagmm.cosim(X, X_hat)
        rel_euc_dist = relative_euclidean_dist(X, X_hat)
        z_r = torch.cat([emb, rel_euc_dist.unsqueeze(-1), cosim.unsqueeze(-1)], dim=1)
        # Concatenate SOM's features with DAGMM's
        z_r_s = []
        for i in range(len(self.soms)):
            z_s_i = [self.soms[i].winner(x) for x in X.cpu()]
            z_s_i = [[x, y] for x, y in z_s_i]
            z_s_i = torch.from_numpy(np.array(z_s_i)).to(z_r.device)  # / (default_som_args.get('x')+1)
            z_r_s.append(z_s_i)
        z_r_s.append(z_r)
        Z = torch.cat(z_r_s, dim=1).to(self.device)

        # estimation network
        gamma = self.dagmm.forward_estimation_net(Z)

        return emb, X_hat, cosim, Z, gamma

    def training_step(self, X: torch.Tensor, y: torch.Tensor = None, labels: torch.Tensor = None):
        _, X_hat, _, Z, gamma_hat = self(X)
        phi, mu, cov_mat = self.dagmm.compute_params(Z, gamma_hat)
        energy_result, pen_cov_mat = self.dagmm.estimate_sample_energy(
            Z, phi, mu, cov_mat
        )
        self.dagmm.phi = torch.nn.Parameter(phi)
        self.dagmm.mu = torch.nn.Parameter(mu)
        self.dagmm.cov_mat = torch.nn.Parameter(cov_mat)
        return self.compute_loss(X_hat, X=X, energy=energy_result, Sigma=pen_cov_mat)

    def compute_loss(self, outputs: torch.Tensor, **kwargs):
        X, energy, Sigma = kwargs.get("X"), kwargs.get("energy"), kwargs.get("Sigma")
        rec_loss = ((X - outputs) ** 2).mean()
        sample_energy = self.dagmm.lamb_1 * energy
        penalty_term = self.dagmm.lamb_2 * Sigma

        return rec_loss + sample_energy + penalty_term
