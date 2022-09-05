import math

import torch
from torch import nn
from pyad.models.base import BaseModule, activation_map
from typing import List, Tuple


def create_layers(in_features: int, hidden_dims: List[int], act_fn: torch.nn.Module):
    layers = []
    in_dim = in_features
    for i in range(len(hidden_dims)):
        layers.append(
            nn.Linear(in_dim, hidden_dims[i])
        )
        layers.append(
            act_fn()
        )
        in_dim = hidden_dims[i]
    layers.append(
        nn.Linear(in_dim, in_features)
    )
    return layers


class CouplingLayer(nn.Module):

    def __init__(
            self,
            in_features: int,
            mask: torch.Tensor,
            hidden_dims: List[int],
            scale_act_fn: str = "tanh",
            translate_act_fn: str = "relu"
    ):
        super(CouplingLayer, self).__init__()
        self.in_features = in_features
        self.mask = mask
        self.hidden_dims = hidden_dims
        self.scale_act_fn = activation_map[scale_act_fn]
        self.translate_act_fn = activation_map[translate_act_fn]
        self._build_network()

    def _build_network(self):
        scale_layers = create_layers(self.in_features, self.hidden_dims, self.scale_act_fn)
        self.scale_net = nn.Sequential(
            *scale_layers
        )
        self.translate_net = nn.Sequential(
            *create_layers(self.in_features, self.hidden_dims, self.translate_act_fn)
        )

    def forward(self, X: torch.Tensor, mode="forward") -> Tuple[torch.Tensor, torch.Tensor]:
        masked_X = X * self.mask

        log_s = self.scale_net(X) * (1 - self.mask)
        t = self.translate_net(X) * (1 - self.mask)

        if mode == "forward":
            s = torch.exp(log_s)
            outputs = masked_X * s + t, log_s.sum(-1, keepdim=True)
        else:
            s = torch.exp(-log_s)
            outputs = (masked_X - t) * s, -log_s.sum(-1, keepdim=True)

        return outputs


class SequentialFlow(nn.Sequential):

    def __init__(self, *args: nn.Module):
        super(SequentialFlow, self).__init__(*args)
        self.in_features = None

    def forward(
            self,
            inputs: torch.Tensor,
            logdets: torch.Tensor = None,
            mode: str = "forward"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.in_features = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        if mode == "forward":
            for module in self._modules.values():
                outputs, logdet = module(inputs, mode=mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                outputs, logdet = module(inputs, mode=mode)
                logdets += logdet

        return outputs, logdets

    def log_probs(self, inputs: torch.Tensor) -> torch.Tensor:
        u, log_jacob = self(inputs)
        log_probs = -0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)
        log_probs = log_probs.sum(-1, keepdim=True)
        log_probs = (log_probs + log_jacob).sum(-1, keepdim=True)
        return log_probs

    def sample(self, n_samples: int = None, noise=None):
        device = next(self.parameters()).device
        if noise is None:
            noise = torch.Tensor(n_samples, self.in_features).normal_()
            noise = noise.to(device)
        samples = self.forward(noise, mode="inverse")
        return samples[0]
