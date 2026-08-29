"""Scoring a trained Bloch operator against a solved split.

load_checkpoint rebuilds a model from a checkpoint alone, which records
the BlochOperatorConfig its weights assume. evaluate runs one split
through that model and returns its metrics, optionally with the loss and
the frequencies.

This module contains:
    - load_checkpoint
    - evaluate
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from baseline import CNNBaseline, CNNBaselineConfig
from model.bloch_operator import BlochOperator, BlochOperatorConfig
from training.metrics import MetricTracker

MODELS = {
    "BlochOperator": (BlochOperator, BlochOperatorConfig),
    "CNNBaseline": (CNNBaseline, CNNBaselineConfig),
}


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Rebuild a trained model from a checkpoint.

    Parameters
    ----------
    path : str or pathlib.Path
        Checkpoint written by Trainer.save_checkpoint.
    device : str or torch.device
        Device the model is moved to.

    Returns
    -------
    torch.nn.Module
        Model in eval mode, of the class the checkpoint names and the
        shape it stores. Take n_bands and frequency_scale from
        model.config, not from a TrainingConfig: the two need not
        describe the same run.

    Raises
    ------
    KeyError
        If the checkpoint records no model name, config or weights, or
        names a model that is not one of MODELS.
    """
    device = torch.device(device)
    state = torch.load(path, map_location=device, weights_only=False)

    missing = [
        key
        for key in ("model_name", "model_config", "model_state_dict")
        if key not in state
    ]
    if missing:
        raise KeyError(
            f"Checkpoint {Path(path).name} is missing: {', '.join(missing)}."
        )

    name = state["model_name"]
    if name not in MODELS:
        raise KeyError(
            f"Checkpoint {Path(path).name} holds a {name}, "
            f"which is none of: {', '.join(MODELS)}."
        )

    model_class, config_class = MODELS[name]
    model = model_class(config_class(**state["model_config"]))
    model.load_state_dict(state["model_state_dict"])

    return model.to(device).eval()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device = "cpu",
    *,
    criterion: nn.Module | None = None,
    relative_floor: float = 1.0,
    collect_predictions: bool = True,
) -> dict[str, Any]:
    """Score one split without touching the parameters.

    Parameters
    ----------
    model : torch.nn.Module
        Model called as model(images, wave_vectors), returning the
        frequencies of one batch.
    loader : torch.utils.data.DataLoader
        Loader yielding image, wave-vector and frequency batches.
    device : str or torch.device
        Device the model and every batch are moved to.
    criterion : torch.nn.Module, optional
        Loss called as criterion(eigenvalues, target_frequencies), as in
        the trainer. None returns the metrics alone.
    relative_floor : float
        Smallest target frequency, in hertz, carrying a relative error.
    collect_predictions : bool
        Return the frequencies with the metrics, as one array per split
        held in memory.

    Returns
    -------
    dict[str, Any]
        metrics               the MetricTracker result
        loss                  sample-weighted mean, given a criterion
        predictions, targets  (n_geometries, K, n_bands) in hertz, collected
        wave_vectors          (n_geometries, K, 2), when collected

    Raises
    ------
    ValueError
        If the loader yielded no batch, from MetricTracker.compute.

    Notes
    -----
    No D4 augmentation, unlike the training pass: the split is scored on
    the wave vectors it was solved on.
    """
    device = torch.device(device)
    model = model.to(device).eval()
    tracker = MetricTracker(relative_floor=relative_floor)

    if criterion is not None:
        criterion = criterion.to(device)

    predicted_batches: list[Tensor] = []
    target_batches: list[Tensor] = []
    wave_vector_batches: list[Tensor] = []

    loss_sum = 0.0
    n_samples = 0

    with torch.inference_mode():
        for batch in loader:
            images, wave_vectors, targets = (
                tensor.to(device, non_blocking=True) for tensor in batch
            )

            output = model(images, wave_vectors)
            predicted = output["frequencies"]
            tracker.update(predicted, targets)

            if criterion is not None:
                loss = criterion(output["eigenvalues"], targets)
                loss_sum += loss.item() * images.shape[0]
                n_samples += images.shape[0]

            if collect_predictions:
                predicted_batches.append(predicted.cpu())
                target_batches.append(targets.cpu())
                wave_vector_batches.append(wave_vectors.cpu())

    result: dict[str, Any] = {"metrics": tracker.compute()}

    if criterion is not None:
        result["loss"] = loss_sum / n_samples

    if collect_predictions:
        result["predictions"] = torch.cat(predicted_batches)
        result["targets"] = torch.cat(target_batches)
        result["wave_vectors"] = torch.cat(wave_vector_batches)

    return result
