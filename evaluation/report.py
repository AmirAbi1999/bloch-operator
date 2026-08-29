"""Writing one evaluation of a Bloch operator to disk.

Nothing here runs the model: evaluate supplies the metrics and the
frequencies, and this module writes them as tables::

    output_dir/
    |-- metrics.json               whole-split scalars and the shape scored
    |-- per_band_metrics.csv       one row per retained band
    |-- per_geometry_metrics.csv   one row per case
    `-- cases/case{i}.csv          one row per wave vector and mode

One metric is one header everywhere it lands, MAE (Hz) in metrics.json
and in the tables alike. The per-case tables are long format.

This module contains:
    - save_evaluation_report
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from training.metrics import MetricTracker

log = logging.getLogger(__name__)


def save_evaluation_report(
    metrics: dict[str, Any],
    predictions: Tensor,
    targets: Tensor,
    wave_vectors: Tensor,
    case_ids: Sequence[int],
    output_dir: str | Path,
    *,
    n_worst_cases: int | None = 20,
) -> None:
    """Write every table of one evaluation into a single directory.

    Parameters
    ----------
    metrics : dict[str, Any]
        Result from evaluate, already keyed by the header each metric is
        written under. Its scalars go to metrics.json and its per-band
        entries to their own table, so a loss can be passed in beside
        them.
    predictions, targets : Tensor
        Frequencies in hertz, each (B, K, n_bands).
    wave_vectors : Tensor
        Wave vectors the split was scored on, (B, K, 2).
    case_ids : sequence of int
        Case id of each geometry, in split order.
    output_dir : str or pathlib.Path
        Root of the layout in the module docstring.
    n_worst_cases : int, optional
        Cases written to cases/, the worst by MAPE. None writes one file
        per geometry.

    Raises
    ------
    KeyError
        If the relative floor, or a whole-split or per-band metric, is
        absent.
    ValueError
        If the tensors do not share the layout above, if the case ids do
        not count the geometries, or if the predictions carry gradients.
    """
    _validate_layout(predictions, targets, wave_vectors, case_ids)

    if "Relative floor (Hz)" not in metrics:
        raise KeyError(
            "metrics carries no Relative floor (Hz); pass the result of "
            "evaluate, which records the floor it scored under."
        )
    relative_floor = float(metrics["Relative floor (Hz)"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predicted = predictions.cpu()
    target = targets.cpu().to(predicted.dtype)
    wave_vectors = wave_vectors.cpu()
    error = predicted - target

    n_geometries, n_wave_vectors, n_bands = predicted.shape

    scalars = _scalar_metrics({
        **metrics,
        "Geometries": n_geometries,
        "Wave vectors": n_wave_vectors,
        "Bands": n_bands,
    })
    (output_dir / "metrics.json").write_text(
        json.dumps(scalars, indent=2) + "\n", encoding="utf-8",
    )

    geometry = _per_geometry_table(error, target, case_ids, relative_floor)
    _per_band_table(metrics).to_csv(output_dir / "per_band_metrics.csv", index=False)
    geometry.to_csv(output_dir / "per_geometry_metrics.csv", index=False)

    worst = (
        list(range(n_geometries))
        if n_worst_cases is None or n_worst_cases >= n_geometries
        else geometry["MAPE (%)"].nlargest(n_worst_cases).index.tolist()
    )
    _write_case_tables(
        predicted[worst],
        target[worst],
        wave_vectors[worst],
        [case_ids[position] for position in worst],
        output_dir / "cases",
        relative_floor,
    )

    log.info("Evaluation report -> %s", output_dir)


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float | int | None]:
    """Keep the scalar metrics, in the form JSON can carry.

    Parameters
    ----------
    metrics : dict[str, Any]
        Metrics to filter. A tensor holding one value is a scalar.

    Returns
    -------
    dict[str, float or int or None]
        Scalars in input order, a nan written as None.
    """
    scalars: dict[str, float | int | None] = {}

    for name, value in metrics.items():
        if isinstance(value, Tensor):
            if value.numel() != 1:
                continue
            value = value.item()

        if isinstance(value, bool):  # a flag is not a metric, and bool is an int
            continue
        if isinstance(value, int):
            scalars[name] = value
        elif isinstance(value, float):
            scalars[name] = value if math.isfinite(value) else None

    return scalars


def _per_band_table(metrics: dict[str, Any]) -> pd.DataFrame:
    """Lay the per-band metrics out as one row per band.

    Parameters
    ----------
    metrics : dict[str, Any]
        MetricTracker result, read for its per-band entries.

    Returns
    -------
    pandas.DataFrame
        One row per band, the lowest first.

    Raises
    ------
    KeyError
        If a per-band metric is absent.
    ValueError
        If the per-band metrics do not share one length.
    """
    names = ("Band MAE (Hz)", "Band RMSE (Hz)", "Band MAPE (%)", "Band R2")

    missing = [name for name in names if name not in metrics]
    if missing:
        raise KeyError(f"Missing per-band metrics: {', '.join(missing)}.")

    values = {name: torch.as_tensor(metrics[name]).flatten() for name in names}
    n_bands = values["Band MAE (Hz)"].numel()

    if any(value.numel() != n_bands for value in values.values()):
        raise ValueError("The per-band metrics do not share one length.")

    # One row is one band already, so the columns drop the Band prefix
    return pd.DataFrame({
        "Band": range(1, n_bands + 1),
        "MAE (Hz)": values["Band MAE (Hz)"].numpy(),
        "RMSE (Hz)": values["Band RMSE (Hz)"].numpy(),
        "MAPE (%)": values["Band MAPE (%)"].numpy(),
        "R2": values["Band R2"].numpy(),
    })


def _per_geometry_table(
    error: Tensor,
    target: Tensor,
    case_ids: Sequence[int],
    relative_floor: float,
) -> pd.DataFrame:
    """Summarize the error of each geometry over its own bands.

    Parameters
    ----------
    error, target : Tensor
        Signed error and the targets it was taken against, each
        (B, K, n_bands).
    case_ids : sequence of int
        Case id of each geometry, in split order.
    relative_floor : float
        Smallest target frequency, in hertz, carrying a relative error.

    Returns
    -------
    pandas.DataFrame
        One row per case. A case whose targets carry no spread reads nan
        under R2.
    """
    absolute = error.abs()
    squared = error.square()

    # Scored against each case's own spread, not the split's
    deviation = (
        (target - target.mean(dim=(1, 2), keepdim=True))
        .square()
        .sum(dim=(1, 2))
    )

    return pd.DataFrame({
        "Case": [int(case) for case in case_ids],
        "MAE (Hz)": absolute.mean(dim=(1, 2)).numpy(),
        "RMSE (Hz)": squared.mean(dim=(1, 2)).sqrt().numpy(),
        "MAPE (%)": _percentage_error(
            absolute, target, relative_floor, dims=(1, 2),
        ).numpy(),
        "R2": MetricTracker.r_squared(squared.sum(dim=(1, 2)), deviation).numpy(),
        # The worst mode decides whether a case keeps its gaps, and a
        # mean over every wave vector hides it
        "Max Error (Hz)": absolute.amax(dim=(1, 2)).numpy(),
    })


def _write_case_tables(
    predicted: Tensor,
    target: Tensor,
    wave_vectors: Tensor,
    case_ids: Sequence[int],
    output_dir: Path,
    relative_floor: float,
) -> None:
    """Write one long-format table of predictions per geometry.

    Parameters
    ----------
    predicted, target : Tensor
        Frequencies in hertz, each (B, K, n_bands).
    wave_vectors : Tensor
        Wave vectors the split was scored on, (B, K, 2).
    case_ids : sequence of int
        Case id of each geometry, naming its file.
    output_dir : pathlib.Path
        Directory the tables are written into.
    relative_floor : float
        Smallest target frequency, in hertz, carrying a relative error.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_wave_vectors, n_bands = predicted.shape[1:]

    # Rows run wave vector by wave vector, the modes of one wave vector
    # together, the order assign_band_index reads them back in
    modes = torch.arange(1, n_bands + 1).repeat(n_wave_vectors)

    for position, case in enumerate(case_ids):
        points = wave_vectors[position]
        case_target = target[position].reshape(-1)
        case_predicted = predicted[position].reshape(-1)
        absolute = (case_predicted - case_target).abs()

        pd.DataFrame({
            "kx": points[:, 0].repeat_interleave(n_bands).numpy(),
            "ky": points[:, 1].repeat_interleave(n_bands).numpy(),
            "Mode": modes.numpy(),
            "Target (Hz)": case_target.numpy(),
            "Predicted (Hz)": case_predicted.numpy(),
            "Abs Error (Hz)": absolute.numpy(),
            "APE (%)": _percentage_error(
                absolute, case_target, relative_floor, dims=(),
            ).numpy(),
        }).to_csv(output_dir / f"case{int(case)}.csv", index=False)


def _percentage_error(
    absolute: Tensor,
    target: Tensor,
    relative_floor: float,
    *,
    dims: tuple[int, ...],
) -> Tensor:
    """Score the error against the target it was taken against.

    Parameters
    ----------
    absolute, target : Tensor
        Absolute error and target frequencies in hertz, one shape.
    relative_floor : float
        Smallest target frequency carrying a relative error. A target at
        or below it, such as an acoustic mode at Gamma, is dropped rather
        than divided by.
    dims : tuple[int, ...]
        Dimensions averaged over; the empty tuple averages nothing.

    Returns
    -------
    Tensor
        Percentage error, nan wherever nothing was scored.
    """
    scored = target > relative_floor
    percentage = torch.full_like(target, torch.nan)
    percentage[scored] = 100.0 * absolute[scored] / target[scored]

    return percentage.nanmean(dim=dims) if dims else percentage


def _validate_layout(
    predictions: Tensor,
    targets: Tensor,
    wave_vectors: Tensor,
    case_ids: Sequence[int],
) -> None:
    """Reject results that do not share one layout.

    Parameters
    ----------
    predictions, targets : Tensor
        Frequencies in hertz, each expected to be (B, K, n_bands).
    wave_vectors : Tensor
        Wave vectors, expected to be (B, K, 2).
    case_ids : sequence of int
        Case id of each geometry.

    Raises
    ------
    ValueError
        If the predictions still require gradients, if the two frequency
        tensors do not share one shape (B, K, n_bands), if the wave vectors do
        not stand beside them, or if the case ids do not count the
        geometries.
    """
    if predictions.requires_grad:
        raise ValueError("Detach predictions before writing a report.")
    if predictions.ndim != 3 or predictions.shape != targets.shape:
        raise ValueError(
            "predictions and targets must share one shape (B, K, n_bands); "
            f"received {tuple(predictions.shape)} and {tuple(targets.shape)}."
        )
    if wave_vectors.shape != (*predictions.shape[:2], 2):
        raise ValueError(
            "wave_vectors must have shape (B, K, 2) beside the predictions; "
            f"received {tuple(wave_vectors.shape)}."
        )
    if len(case_ids) != predictions.shape[0]:
        raise ValueError(
            f"{len(case_ids)} case ids were given for "
            f"{predictions.shape[0]} geometries."
        )
