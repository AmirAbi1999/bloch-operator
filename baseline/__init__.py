"""Convolutional baselines the latent Bloch operator is compared against."""

from .cnn import CNNBaseline, CNNBaselineConfig
from .qcnn import QCNNBaseline, QCNNBaselineConfig

__all__ = [
    "CNNBaseline",
    "CNNBaselineConfig",
    "QCNNBaseline",
    "QCNNBaselineConfig",
]
