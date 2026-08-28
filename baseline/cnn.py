"""Fixed-grid convolutional baseline the latent Bloch operator is measured against.

The baseline keeps the geometry encoder and drops the operator: one linear
head turns the latent vector into a whole band structure at once, one row per
wave vector of the sweep the split was solved on, in that order.

The sweep is the irreducible wedge, kx from 0 to 1 in steps of 0.1 with
0 <= ky <= kx, which is K = 66 wave vectors::

    (0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.2, 0.0), (0.2, 0.1), (0.2, 0.2),
    (0.3, 0.0), ... (1.0, 0.8), (1.0, 0.9), (1.0, 1.0)

Row 0 is therefore Gamma, where the acoustic branches vanish, and the last
row is the M corner.

This module contains:
    - CNNBaselineConfig
    - CNNBaseline
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from model.bloch_operator import GeometryEncoder


@dataclass(frozen=True)
class CNNBaselineConfig:
    """Hyperparameters for the fixed-grid convolutional baseline.

    Attributes
    ----------
    in_channels : int
        Channels in the geometry image.
    latent_dim : int
        Width of the latent geometry vector.
    widths : tuple[int, ...]
        Channel width of each stride-2 ConvBlock stage of the encoder.
    n_wave_vectors : int
        Wave vectors K the split was solved on, one head row each.
    n_bands : int
        Lowest bands predicted per wave vector.
    n_acoustic_modes : int
        Leading bands held at zero at Gamma, the first wave vector.
    frequency_scale : float
        Hertz per unit of normalized frequency. Match the operator it is
        compared against, since one loss reads both.
    """

    in_channels: int = 1
    latent_dim: int = 256
    widths: tuple[int, ...] = (32, 64, 128, 256)
    n_wave_vectors: int = 66
    n_bands: int = 6
    n_acoustic_modes: int = 2
    frequency_scale: float = 40_000.0


class CNNBaseline(nn.Module):
    """Predict a whole band structure in one pass, on a fixed wave-vector sweep.

    The encoder is the one BlochOperator uses, so the two models differ only
    in what sits above the latent vector. The head emits normalized
    frequencies: their squares are the eigenvalues the loss supervises.

    Parameters
    ----------
    config : CNNBaselineConfig
        Encoder shape, sweep length, retained bands and frequency scale.
    """

    def __init__(self, config: CNNBaselineConfig = CNNBaselineConfig()) -> None:
        super().__init__()

        self.config: CNNBaselineConfig = config

        self.encoder = GeometryEncoder(
            in_channels=config.in_channels,
            latent_dim=config.latent_dim,
            widths=config.widths,
        )
        self.head = nn.Linear(
            in_features=config.latent_dim,
            out_features=config.n_wave_vectors * config.n_bands,
        )

        # The acoustic branches vanish at Gamma
        mask = torch.ones(config.n_wave_vectors, config.n_bands)
        mask[0, :config.n_acoustic_modes] = 0.0
        self.register_buffer("acoustic_mask", mask)

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        """Predict the bands of every wave vector in the sweep.

        Parameters
        ----------
        images : Tensor
            Geometry images, shape (B, C, H, W).

        Returns
        -------
        dict[str, Tensor]
            latent          (B, latent_dim)
            eigenvalues     (B, K, n_bands)
            frequencies     (B, K, n_bands), in hertz
        """
        latent = self.encoder(images)

        normalized = self.acoustic_mask * self.head(latent).reshape(
            len(images), self.config.n_wave_vectors, self.config.n_bands,
        )

        return {
            "latent": latent,
            "eigenvalues": normalized.square(),
            "frequencies": self.config.frequency_scale * normalized.abs(),
        }
