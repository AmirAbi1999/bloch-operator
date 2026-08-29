"""Configuration for one Bloch operator training run.

A TrainingConfig describes a run as data: where the splits live, how the
operator is shaped, what the loss and the schedule look like, and where the
run is written. It builds nothing; train.py wires what it describes.

This module contains:
    - DataConfig
    - LossConfig
    - OptimConfig
    - RuntimeConfig
    - TrainingConfig
    - set_seed
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from baseline import CNNBaselineConfig
from model import BlochOperatorConfig

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DataConfig:
    """Where the splits live and how they are batched.

    Attributes
    ----------
    root : pathlib.Path
        Directory holding the train, val and test split directories.
    eval_batch_size : int
        Geometries per batch when scoring. Scoring reads far more wave
        vectors per geometry, so it takes the smaller batch.
    num_workers : int
        Loader worker processes; above 0 the caller needs a main guard.
    pin_memory : bool
        Stage batches in pinned memory, which the trainer's non-blocking
        copies need; a run on the cpu pins nothing.
    d4_augmentation : bool
        Supervise every wave vector on its whole D4 orbit, which is what
        carries the rest of the Brillouin zone into a run solved over the
        irreducible wedge alone.
    """

    root: Path = ROOT / "dataset"
    batch_size: int = 32
    eval_batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    d4_augmentation: bool = True


@dataclass(frozen=True)
class LossConfig:
    """Shape of the eigenvalue loss.

    Attributes
    ----------
    huber_beta : float
        Transition between the quadratic and the linear region of the
        loss, in log-eigenvalue units.
    band_weights : tuple[float, ...], optional
        One weight per retained band, or None to weight them equally.
    """

    huber_beta: float = 1.0e-2
    band_weights: tuple[float, ...] | None = None


@dataclass(frozen=True)
class OptimConfig:
    """Optimizer and the rate schedule the run is trained on.

    Attributes
    ----------
    scheduler : {"cosine", "plateau", "none"}
        Rate policy: a cosine decay over the epochs the warmup leaves, a
        cut whenever the validation loss stalls, or one flat rate.
    warmup_epochs : int
        Epochs spent ramping up to the full rate before the policy takes
        over; 0 opens at the full rate. The plateau policy takes none.
    warmup_start_factor : float
        Fraction of learning_rate the warmup opens at.
    min_learning_rate : float
        Floor the cosine decays towards, and the floor below which the
        plateau stops cutting.
    plateau_factor : float
        Factor the plateau multiplies the rate by when it cuts.
    plateau_patience : int
        Epochs the validation loss may stall before the plateau cuts.
        Keep it under patience, or the run stops before the cut.
    grad_clip : float, optional
        Gradient-norm ceiling; None measures the norm without clipping.
    patience : int, optional
        Epochs without a lower validation loss before the run stops.
    """

    epochs: int = 200
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    betas: tuple[float, float] = (0.9, 0.999)
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.1
    min_learning_rate: float = 1.0e-6
    plateau_factor: float = 0.5
    plateau_patience: int = 10
    grad_clip: float | None = 1.0
    patience: int | None = 25


@dataclass(frozen=True)
class RuntimeConfig:
    """Where the run happens and what it leaves behind.

    Attributes
    ----------
    device : str
        Device name, or "auto" to take cuda when it is available.
    deterministic : bool
        Ask cuDNN for reproducible kernels, at some speed.
    resume : pathlib.Path, optional
        Checkpoint the run continues from.
    """

    device: str = "auto"
    seed: int = 0
    deterministic: bool = False
    output_dir: Path = ROOT / "runs"
    resume: Path | None = None


@dataclass(frozen=True)
class TrainingConfig:
    """One training run, described as data."""

    data: DataConfig = field(default_factory=DataConfig)
    model: BlochOperatorConfig | CNNBaselineConfig = field(
        default_factory=BlochOperatorConfig
    )
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self) -> None:
        """Reject a run nothing downstream rejects in time."""
        if self.model.n_acoustic_modes >= self.model.n_bands:
            raise ValueError("n_acoustic_modes must leave one optical band.")

        if not 0 <= self.optim.warmup_epochs < self.optim.epochs:
            raise ValueError("warmup_epochs must lie in [0, epochs).")

        if self.optim.scheduler == "plateau" and self.optim.warmup_epochs:
            raise ValueError("The plateau scheduler takes no warmup.")


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch, and optionally pin cuDNN.

    Parameters
    ----------
    seed : int
        Seed handed to all three generators; torch covers cuda with it.
    deterministic : bool
        Ask cuDNN for reproducible kernels and stop it benchmarking, at
        some speed. The eigensolver is deterministic either way.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
