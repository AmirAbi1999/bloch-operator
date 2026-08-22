"""COMSOL process automation: per-worker sessions and dataset sweeps."""

from .comsol_runner import ComsolRunError, ComsolRunner, ComsolTags, SweepValues
from .dataset_builder import ComsolDatasetBuilder

__all__ = [
    "ComsolRunError",
    "ComsolRunner",
    "ComsolTags",
    "SweepValues",
    "ComsolDatasetBuilder",
]
