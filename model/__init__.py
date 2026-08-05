"""Structure-preserving latent Bloch operator: network, loss, and symmetry tools."""

from .bloch_operator import (
    AcousticNullHead,
    BlochOperator,
    BlochOperatorConfig,
    ConvBlock,
    GeometryEncoder,
    MatrixHead,
)
from .loss import EigenvalueSupervisedLoss
from .utils import d4_orbit, gamma_diagnostics, random_d4_transform

__all__ = [
    "AcousticNullHead",
    "BlochOperator",
    "BlochOperatorConfig",
    "ConvBlock",
    "GeometryEncoder",
    "MatrixHead",
    "EigenvalueSupervisedLoss",
    "d4_orbit",
    "gamma_diagnostics",
    "random_d4_transform",
]
