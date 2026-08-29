"""Training run driving one TrainingConfig from config to fitted model.

The config describes the run and this module builds it. Edit the
TrainingConfig main builds to run something other than the default.

This module contains:
    - SCHEDULERS
    - resolve_device
    - build_scheduler
    - build_trainer
    - build_loader
    - git_commit
    - save_run_config
    - main
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    ReduceLROnPlateau,
    SequentialLR,
)
from torch.utils.data import DataLoader

from baseline import CNNBaseline
from config import OptimConfig, TrainingConfig, set_seed
from model import BlochOperator, BlochOperatorConfig, EigenvalueSupervisedLoss
from training import Trainer, make_dataloader

log = logging.getLogger(__name__)

SCHEDULERS = ("cosine", "plateau", "none")


def resolve_device(name: str) -> torch.device:
    """Resolve "auto" to cuda when there is one, and to cpu otherwise.

    Resolved on use rather than when the config is written, so a saved run
    stays portable between machines.
    """
    if name != "auto":
        return torch.device(name)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_scheduler(
    optimizer: Optimizer,
    optim: OptimConfig,
) -> LRScheduler | None:
    """Build the rate policy OptimConfig names.

    Returns
    -------
    torch.optim.lr_scheduler.LRScheduler or None
        Scheduler the trainer steps once per epoch, or None when the run
        holds one rate throughout.
    """
    if optim.scheduler not in SCHEDULERS:
        raise ValueError(f"scheduler must be one of {SCHEDULERS}.")

    if optim.scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            factor=optim.plateau_factor,
            patience=optim.plateau_patience,
            min_lr=optim.min_learning_rate,
        )

    schedulers: list[LRScheduler] = []

    # A fraction of the rate to open with, since an early step that
    # inflates the encoder is squared by D = H^H @ H before the loss sees it
    if optim.warmup_epochs:
        schedulers.append(LinearLR(
            optimizer,
            start_factor=optim.warmup_start_factor,
            total_iters=optim.warmup_epochs,
        ))

    # The cosine takes the epochs the warmup leaves, so the two span the run
    if optim.scheduler == "cosine":
        schedulers.append(CosineAnnealingLR(
            optimizer,
            T_max=optim.epochs - optim.warmup_epochs,
            eta_min=optim.min_learning_rate,
        ))

    if len(schedulers) < 2:
        return schedulers[0] if schedulers else None

    return SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=[optim.warmup_epochs],
    )


def build_trainer(config: TrainingConfig) -> Trainer:
    """Build the model, the loss, the optimizer and the trainer.

    Returns
    -------
    training.Trainer
        Trainer ready to fit, resumed from runtime.resume when it is set.
    """
    optim = config.optim
    operator = isinstance(config.model, BlochOperatorConfig)
    model = BlochOperator(config.model) if operator else CNNBaseline(config.model)

    optimizer = AdamW(
        model.parameters(),
        lr=optim.learning_rate,
        betas=optim.betas,
        weight_decay=optim.weight_decay,
    )

    weights = config.loss.band_weights
    trainer = Trainer(
        model,
        EigenvalueSupervisedLoss(
            frequency_scale=config.model.frequency_scale,
            huber_beta=config.loss.huber_beta,
            band_weights=None if weights is None else torch.tensor(weights),
        ),
        optimizer,
        resolve_device(config.runtime.device),
        scheduler=build_scheduler(optimizer, optim),
        grad_clip=optim.grad_clip,
        # Only the operator reads the wave vectors, so only it can be
        # supervised on their orbit or answered at a Gamma of its own
        d4_augmentation=config.data.d4_augmentation and operator,
        gamma_check=operator,
        patience=optim.patience,
        output_dir=config.runtime.output_dir,
    )

    if config.runtime.resume is not None:
        trainer.load_checkpoint(config.runtime.resume)

    return trainer


def build_loader(config: TrainingConfig, split: str) -> DataLoader:
    """Open one split directory under data.root as a loader.

    Only the training split is shuffled, and only it reads batch_size;
    the rest read eval_batch_size.

    Returns
    -------
    torch.utils.data.DataLoader
        Loader yielding image, wave-vector and frequency batches, the
        frequencies cut to the operator's own n_bands.
    """
    data = config.data
    training = split == "train"
    device = resolve_device(config.runtime.device)

    return make_dataloader(
        data.root / split,
        batch_size=data.batch_size if training else data.eval_batch_size,
        shuffle=training,
        num_workers=data.num_workers,
        # Pinning buys the non-blocking copies something only on cuda
        pin_memory=data.pin_memory and device.type == "cuda",
        n_bands=config.model.n_bands,
    )


def git_commit() -> str | None:
    """Name the commit the run was launched from, dirty tree included.

    Returns
    -------
    str or None
        Commit description, or None outside a repository.
    """
    try:
        head = subprocess.run(
            ("git", "describe", "--always", "--dirty"),
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return head.stdout.strip()


def save_run_config(config: TrainingConfig, trainer: Trainer) -> Path:
    """Write the whole run configuration beside its history and checkpoints.

    Returns
    -------
    pathlib.Path
        The file that was written.
    """
    # Spelled out, since the torch stub declares no __str__ for a device
    device = trainer.device
    resolved = device.type if device.index is None else f"{device.type}:{device.index}"

    record = {
        "model": {"name": type(trainer.model).__name__, **asdict(config.model)},
        "data": asdict(config.data),
        "loss": asdict(config.loss),
        "optim": asdict(config.optim),
        "runtime": asdict(config.runtime),
        "run": {
            "started": datetime.now().isoformat(timespec="seconds"),
            "device": resolved,
            "d4_augmentation": trainer.d4_augmentation,
            "gamma_check": trainer.gamma_check,
            "parameters": sum(p.numel() for p in trainer.model.parameters()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "commit": git_commit(),
        },
    }

    path = trainer.output_dir / "config.json"
    # Paths are the only member JSON cannot carry on its own
    path.write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8",
    )

    return path


def main() -> None:
    """Seed one run, open the two splits, and fit."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = TrainingConfig()
    set_seed(config.runtime.seed, config.runtime.deterministic)
    log.info("Training on %s, writing %s",
             resolve_device(config.runtime.device), config.runtime.output_dir)

    trainer = build_trainer(config)

    save_run_config(config, trainer)

    trainer.fit(
        build_loader(config, "train"),
        build_loader(config, "val"),
        config.optim.epochs,
    )


if __name__ == "__main__":
    main()
