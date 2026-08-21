"""Supervised training loop for the latent Bloch operator.

A Trainer fits one BlochOperator against solved band frequencies one epoch
at a time: optimize over a training split, score a validation split, step
the scheduler and checkpoint. Every run leaves history.csv, last.pt and
best.pt in a single output directory.

This module contains:
    - Trainer
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader

from model.bloch_operator import BlochOperator
from model.utils import d4_orbit, gamma_diagnostics

from .metrics import MetricTracker

log = logging.getLogger(__name__)


class Trainer:
    """Fit a BlochOperator against solved band frequencies.

    One epoch trains over a split, scores a second one, steps the
    scheduler, appends a row to history.csv and writes last.pt, keeping
    best.pt at the lowest validation loss seen. Loaders are passed per
    call, so the same trainer scores whichever split it is handed.

    The loss reads the model's eigenvalues and the metrics its frequencies.
    Both are (B, K, N), so swapping them raises nothing and trains nothing.
    A batch whose loss or gradient norm is not finite is skipped and counted.

    Parameters
    ----------
    model : model.BlochOperator
        Model returning the eigenvalues and frequencies of one batch.
    criterion : torch.nn.Module
        Loss called as criterion(eigenvalues, target_frequencies).
    optimizer : torch.optim.Optimizer
        Optimizer over the model parameters.
    device : str or torch.device
        Device the model, the loss and every batch are moved to.
    scheduler : torch.optim.lr_scheduler.LRScheduler, optional
        Stepped once per epoch, on the validation loss when it is a
        ReduceLROnPlateau.
    grad_clip : float, optional
        Gradient-norm ceiling; None measures the norm without clipping.
    d4_augmentation : bool
        Supervise every wave vector on its whole D4 orbit, at eight times
        the eigenproblems per batch.
    patience : int, optional
        Epochs without a lower validation loss before the run stops.
    output_dir : str or pathlib.Path
        Directory the history and the checkpoints are written to.
    """

    def __init__(self,
                 model: BlochOperator,
                 criterion: nn.Module,
                 optimizer: Optimizer,
                 device: str | torch.device = "cpu",
                 *,
                 scheduler: LRScheduler | None = None,
                 grad_clip: float | None = 1.0,
                 d4_augmentation: bool = True,
                 patience: int | None = None,
                 output_dir: str | Path = "runs",
    )->None:
        self.device = torch.device(device)
        self.model: BlochOperator = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = float("inf") if grad_clip is None else grad_clip
        self.d4_augmentation = d4_augmentation
        self.patience = patience

        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output_dir / "history.csv"

        # Last completed epoch: carried across fit calls, and restored by
        # load_checkpoint, so the next one to run is always self.epoch + 1
        self.epoch = 0
        self.best_loss = float("inf")

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int,
    ) -> list[dict[str, float]]:
        """Train, score and checkpoint one epoch at a time.

        Parameters
        ----------
        train_loader, val_loader : torch.utils.data.DataLoader
            Loaders over the training and validation splits.
        epochs : int
            Epochs to run, counted from wherever the trainer left off.

        Returns
        -------
        list[dict[str, float]]
            One row per completed epoch, as written to history.csv.
        """
        history: list[dict[str, float]] = []
        improved_at = self.epoch

        # Fixed on entry, since the loop advances self.epoch as it goes
        epoch_range = range(self.epoch + 1, self.epoch + 1 + epochs)

        for epoch in epoch_range:
            start = time.perf_counter()

            # Read before the scheduler steps: the rate the epoch used
            learning_rate = self.optimizer.param_groups[0]["lr"]

            train = self.train_epoch(train_loader)
            validation = self.validate_epoch(val_loader)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(validation["loss"])
            elif self.scheduler is not None:
                self.scheduler.step()

            row = self.record_epoch(
                epoch, learning_rate, time.perf_counter() - start,
                train, validation,
            )
            history.append(row)

            log.info(
                "Epoch %3d/%d | train %.4e | val %.4e | val MAPE %6.2f%% | "
                "lr %.2e | %.1fs%s",
                epoch, epoch_range.stop - 1, row["train_loss"], row["val_loss"],
                row["val_mape"], learning_rate, row["epoch_time_s"],
                f" | {train['skipped']} skipped" if train["skipped"] else "",
            )

            self.epoch = epoch
            self.save_checkpoint("last.pt", epoch, validation)

            if validation["loss"] < self.best_loss:
                self.best_loss = validation["loss"]
                improved_at = epoch
                self.save_checkpoint("best.pt", epoch, validation)

            elif self.patience is not None and epoch - improved_at >= self.patience:
                log.info("Stopping at epoch %d: no gain in %d epochs.",
                         epoch, self.patience)
                break

        return history

    def train_epoch(
            self,
            loader: DataLoader,
    ) -> dict[str, Any]:
        """Run one optimization pass over the training split.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            Loader yielding image, wave-vector and frequency batches.

        Returns
        -------
        dict[str, Any]
            loss, the sample-weighted mean; metrics, the MetricTracker
            result; grad_norm, the mean pre-clip norm; and skipped, the
            batches that never reached a step.

        Raises
        ------
        RuntimeError
            If no batch completed at all.
        """
        self.model.train()
        tracker = MetricTracker()

        loss_sum = grad_sum = 0.0
        n_samples = n_steps = skipped = 0

        for batch in loader:
            images, wave_vectors, targets = (
                tensor.to(self.device, non_blocking=True) for tensor in batch
            )

            # The orbit widens the wave-vector axis, (B, K, 2) -> (B, 8K, 2),
            # against each target held over the eight images of its own q.
            if self.d4_augmentation:
                wave_vectors = d4_orbit(wave_vectors).flatten(1, 2)
                targets = targets.repeat_interleave(8, dim=1)

            self.optimizer.zero_grad(set_to_none=True)

            output = self.model(images, wave_vectors)
            loss = self.criterion(output["eigenvalues"], targets)

            if not torch.isfinite(loss):
                skipped += 1
                continue

            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip,
            )
            if not torch.isfinite(grad_norm):
                self.optimizer.zero_grad(set_to_none=True)
                skipped += 1
                continue

            self.optimizer.step()

            n_batch = images.shape[0]
            loss_sum += loss.item() * n_batch
            grad_sum += grad_norm.item()
            n_samples += n_batch
            n_steps += 1

            tracker.update(output["frequencies"].detach(), targets)

        if not n_samples:
            raise RuntimeError(f"No training batch completed: {skipped} skipped.")

        return {
            "loss": loss_sum / n_samples,
            "metrics": tracker.compute(),
            "grad_norm": grad_sum / n_steps,
            "skipped": skipped,
        }

    def validate_epoch(
            self,
            loader: DataLoader,
    ) -> dict[str, Any]:
        """Score one split without touching the parameters.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            Loader yielding image, wave-vector and frequency batches.

        Returns
        -------
        dict[str, Any]
            loss, the sample-weighted mean; metrics, the MetricTracker
            result; and gamma, the acoustic residual and first optical
            eigenvalue at the Gamma point.

        Raises
        ------
        RuntimeError
            If the loader yielded no batch.
        """
        self.model.eval()
        tracker = MetricTracker()

        loss_sum = 0.0
        n_samples = 0
        gamma: dict[str, float] = {}

        with torch.inference_mode():
            for index, batch in enumerate(loader):
                images, wave_vectors, targets = (
                    tensor.to(self.device, non_blocking=True) for tensor in batch
                )

                output = self.model(images, wave_vectors)
                loss = self.criterion(output["eigenvalues"], targets)

                loss_sum += loss.item() * images.shape[0]
                n_samples += images.shape[0]
                tracker.update(output["frequencies"], targets)

                if not index:
                    diagnostics = gamma_diagnostics(self.model, images)
                    gamma = {
                        "gamma_acoustic": diagnostics["max_acoustic_eigenvalue"].item(),
                        "gamma_optical": diagnostics["min_first_optical_eigenvalue"].item(),
                    }

        if not n_samples:
            raise RuntimeError("The validation loader yielded no batch.")

        return {
            "loss": loss_sum / n_samples,
            "metrics": tracker.compute(),
            "gamma": gamma,
        }

    def save_checkpoint(self,
                        filename: str,
                        epoch: int,
                        validation: dict[str, Any],
    ) -> Path:
        """Write the run state and one epoch's full metrics.

        Parameters
        ----------
        filename : str
            Name of the file inside the output directory.
        epoch : int
            Epoch the state was reached at.
        validation : dict[str, Any]
            Result of validate_epoch, stored whole: its per-band tensors
            are recoverable from nowhere else.

        Returns
        -------
        pathlib.Path
            The file that was written.
        """
        state: dict[str, Any] = {
            "epoch": epoch,
            "best_loss": self.best_loss,
            "val_loss": validation["loss"],
            "val_metrics": validation["metrics"],
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        # The BlochOperatorConfig the model was built from, flattened to a
        # dict: it records the operator_dim, n_bands and frequency_scale the
        # weights above assume, so a checkpoint found on its own still says
        # what shape of model to load it back into.
        state["model_config"] = asdict(self.model.config)

        path = self.output_dir / filename
        torch.save(state, path)

        return path

    def load_checkpoint(self,
                        path: str | Path,
    ) -> int:
        """Restore a run and report the epoch it resumes at.

        Parameters
        ----------
        path : str or pathlib.Path
            Checkpoint written by save_checkpoint.

        Returns
        -------
        int
            First epoch the next fit call will run.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        self.epoch = int(state["epoch"])
        self.best_loss = float(state.get("best_loss", state["val_loss"]))
        log.info("Resuming %s at epoch %d.", Path(path).name, self.epoch + 1)

        return self.epoch + 1

    def record_epoch(self,
                     epoch: int,
                     learning_rate: float,
                     elapsed: float,
                     train: dict[str, Any],
                     validation: dict[str, Any],
    ) -> dict[str, float]:
        """Flatten one epoch into a row and append it to history.csv.

        Every metric is prefixed by the split it came from, so mae reads as
        train_mae and val_mae, and the Gamma-point diagnostics keep their
        own names. Only the scalars go into the row; the per-band tensors
        are written whole into the checkpoint instead.

        Parameters
        ----------
        epoch : int
            Epoch the row describes.
        learning_rate : float
            Rate the epoch was trained at.
        elapsed : float
            Wall-clock seconds the epoch took.
        train, validation : dict[str, Any]
            Results of train_epoch and validate_epoch.

        Returns
        -------
        dict[str, float]
            The row as it was written.
        """
        row: dict[str, float] = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "epoch_time_s": elapsed,
            "train_loss": train["loss"],
            "val_loss": validation["loss"],
            "grad_norm": train["grad_norm"],
            "skipped": train["skipped"],
        }

        for prefix, result in (("train", train), ("val", validation)):
            row.update({
                f"{prefix}_{name}": value
                for name, value in result["metrics"].items()
                if isinstance(value, float)
            })

        row.update(validation["gamma"])

        with self.history_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if not file.tell():
                writer.writeheader()
            writer.writerow(row)

        return row
