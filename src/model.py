"""LSTM autoencoder for multivariate sensor anomaly detection."""

from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return self.output(decoded)
