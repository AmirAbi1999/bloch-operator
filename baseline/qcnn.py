"""Two-branch convolutional baseline answering any wave vector it is asked for.

One branch is the GeometryEncoder BlochOperator uses, the other a perceptron
on each wave vector. The head reads a geometry's latent vector joined to a
wave vector's embedding and predicts the geometry's bands at that wave vector::

    encoder       (B, C, H, W)                   -> (B, latent_dim)
    wave_branch   (B, K, 2)                      -> (B, K, wave_dim)
    head          (B, K, latent_dim + wave_dim)  -> (B, K, n_bands)

This module contains:
    - QCNNBaselineConfig
    - QCNNBaseline
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from model.bloch_operator import GeometryEncoder
from model.utils import prepare_wavevectors


@dataclass(frozen=True)
class QCNNBaselineConfig:
    """Hyperparameters for the two-branch convolutional baseline.

    Attributes
    ----------
    in_channels : int
        Channels in the geometry image.
    latent_dim : int
        Width of the latent geometry vector.
    widths : tuple[int, ...]
        Channel width of each stride-2 ConvBlock stage of the encoder.
    wave_dim : int
        Width of every layer of the wave-vector branch.
    hidden_dim : int
        Width of the hidden layers of the head.
    n_bands : int
        Lowest bands predicted per wave vector.
    n_acoustic_modes : int
        Leading bands held at zero at Gamma.
    frequency_scale : float
        Hertz per unit of normalized frequency. Match the operator it is
        compared against, since one loss reads both.
    """

    in_channels: int = 1
    latent_dim: int = 256
    widths: tuple[int, ...] = (32, 64, 128, 256)
    wave_dim: int = 32
    hidden_dim: int = 200
    n_bands: int = 6
    n_acoustic_modes: int = 2
    frequency_scale: float = 40_000.0


class QCNNBaseline(nn.Module):
    """Predict the bands of each geometry at each wave vector it is asked for.

    At the defaults the wave-vector branch and the head hold about as many
    parameters as the operator's three heads at r = 12, so the two models
    differ in structure rather than in size. The head emits normalized
    frequencies, unsorted: their squares are the eigenvalues the loss
    supervises.

    Parameters
    ----------
    config : QCNNBaselineConfig
        Encoder shape, branch and head widths, retained bands and frequency
        scale.
    """

    def __init__(self, config: QCNNBaselineConfig = QCNNBaselineConfig()) -> None:
        super().__init__()

        self.config: QCNNBaselineConfig = config

        self.encoder = GeometryEncoder(
            in_channels=config.in_channels,
            latent_dim=config.latent_dim,
            widths=config.widths,
        )
        self.wave_branch = nn.Sequential(
            nn.Linear(2, config.wave_dim),
            nn.GELU(),
            nn.Linear(config.wave_dim, config.wave_dim),
            nn.GELU(),
            nn.Linear(config.wave_dim, config.wave_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(config.latent_dim + config.wave_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.n_bands),
        )

        self.register_buffer(
            "acoustic_bands",
            torch.arange(config.n_bands) < config.n_acoustic_modes,
        )

    def forward(self, images: Tensor, wave_vectors: Tensor) -> dict[str, Tensor]:
        """Predict the lowest bands for each geometry and wave vector.

        Parameters
        ----------
        images : Tensor
            Geometry images, shape (B, C, H, W).
        wave_vectors : Tensor
            Nondimensional wave vectors, shape (K, 2) or (B, K, 2).

        Returns
        -------
        dict[str, Tensor]
            latent          (B, latent_dim)
            eigenvalues     (B, K, n_bands)
            frequencies     (B, K, n_bands), in hertz

        Raises
        ------
        ValueError
            If wave_vectors is neither (K, 2) nor (B, K, 2) against the batch.
        """
        latent = self.encoder(images)
        wave_vectors = prepare_wavevectors(wave_vectors, len(images)).to(latent)

        joint = torch.cat([
            latent.unsqueeze(1).expand(-1, wave_vectors.shape[1], -1),
            self.wave_branch(wave_vectors),
        ], dim=-1)

        # The acoustic branches vanish at Gamma, which the splits hold exactly
        at_gamma = (wave_vectors == 0).all(dim=-1, keepdim=True)
        normalized = self.head(joint).masked_fill(at_gamma & self.acoustic_bands, 0.0)

        return {
            "latent": latent,
            "eigenvalues": normalized.square(),
            "frequencies": self.config.frequency_scale * normalized.abs(),
        }
