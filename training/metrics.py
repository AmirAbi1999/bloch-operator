"""Evaluation Metrics for bloch-operator frequency prediction.

Expected tensor shape::

    predictions : (B, K, n_bands)
    targets     : (B, K, n_bands)

B geometries in a batch, K wave vectors, n_bands bands retained of each.

Usage
-----
tracker = MetricTracker()

for predictions, targets in ...:
    tracker.update(predictions.detach(), targets)

metrics = tracker.compute()

Notes
-----
The tracker accumulates sufficient statistics across all batches, so metrics
such as RMSE and R² are computed globally rather than averaged batch-wise.

This module does not detach tensors or disable autograd. During training,
pass detached predictions to metric functions. During validation/test,
compute metrics inside "torch.inference_mode()" or "torch.no_grad()".

This module contains:
    - MetricTracker
"""

from __future__ import annotations

import torch
from torch import Tensor


class MetricTracker:
    """Accumulate exact band-frequency errors across batches.

    Parameters
    ----------
    relative_floor : float
        Smallest target frequency, in hertz, carrying a relative error.

    Notes
    -----
     A target at or below
    relative_floor, such as an acoustic mode at Gamma, carries no meaningful
    relative error and is dropped from the relative metrics only.
    """

    # 1 - residual / deviation, and nan wherever the targets carry no
    # spread for the score to be read against.
    r_squared = staticmethod(lambda residual, deviation: torch.where(
        deviation > 0.0,
        1.0 - residual / deviation,
        torch.nan,
    ))

    def __init__(self, relative_floor: float = 1.0) -> None:
        self.relative_floor = float(relative_floor)

        self.n_geometries = 0
        self.band_sums = torch.zeros(())
        self.geometry_mape: list[Tensor] = []

    def reset(self) -> None:
        """Drop every statistic accumulated so far."""
        self.n_geometries = 0
        self.band_sums = torch.zeros(())
        self.geometry_mape.clear()

    def update(self, predicted: Tensor, target: Tensor) -> None:
        """Accumulate the statistics of one batch.

        Parameters
        ----------
        predicted : Tensor
            Predicted frequencies in hertz, shape (B, K, n_bands).
        target : Tensor
            Target frequencies in hertz, shape (B, K, n_bands).

        Raises
        ------
        ValueError
            If the predictions still require gradients, or if the two
            tensors do not share one shape (B, K, n_bands).
        """
        self._validate(predicted, target)

        target = target.to(device=predicted.device, dtype=predicted.dtype)
        error = predicted - target
        absolute = error.abs()
        squared = error.square()

        # Score against the target itself, and only where it carries a scale
        scored = target > self.relative_floor
        percentage = torch.where(
            scored,
            100.0 * absolute / target.clamp_min(self.relative_floor),
            torch.nan,
        )
        # One stacked sum over the batch, in the order compute() unpacks it
        self.n_geometries += target.shape[0]
        self.band_sums = self.band_sums + torch.stack((
            absolute.sum(0),
            squared.sum(0),
            percentage.nansum(0),
            scored.to(target.dtype).sum(0),
            target.sum(0),
            target.square().sum(0),
        )).cpu()

        # Quantiles and per-geometry scores are exact only over the samples
        self.geometry_mape.append(percentage.nanmean((1, 2)).cpu())

    def compute(self) -> dict[str, float | Tensor]:
        """Return metrics over all accumulated batches.

        Returns
        -------
        dict[str, float | Tensor]
            MAE (Hz), RMSE (Hz)             float, over every band
            MAPE (%), R2                    float, over every band
            Band MAE (Hz), Band RMSE (Hz)   (n_bands,)
            Band MAPE (%), Band R2          (n_bands,)
            P90 MAPE (%), P95 MAPE (%)      float, tails over the geometries

            Every key is the header the metric is written under, so a
            row or a column takes the name it came in with.

            A metric with nothing scored under it reads nan.

        Raises
        ------
        ValueError
            If no batch has been accumulated.
        """
        if not self.n_geometries:
            raise ValueError("No batches have been added.")

        # Every sum holds one entry per (wave vector, band) pair
        (absolute_sum, squared_sum, percentage_sum,
         scored_count, target_sum, target_square_sum) = self.band_sums

        n_wave_vectors, n_bands = absolute_sum.shape
        n_values = self.n_geometries * n_wave_vectors * n_bands
        n_per_band = self.n_geometries * n_wave_vectors

        geometry_mape = torch.cat(self.geometry_mape)

        # Quantiles of the scored bands, then of the per-geometry scores
        quantiles = torch.tensor([0.90, 0.95], dtype=geometry_mape.dtype)
        geometry_p90_mape, geometry_p95_mape = (
            geometry_mape.nanquantile(quantiles).tolist()
        )

        # Squared deviation of the targets, over every band and per band
        deviation = target_square_sum.sum() - target_sum.sum().square() / n_values
        band_deviation = (
            target_square_sum.sum(0)
            - target_sum.sum(0).square() / n_per_band
        )

        return {
            "MAE (Hz)": (absolute_sum.sum() / n_values).item(),
            "RMSE (Hz)": (squared_sum.sum() / n_values).sqrt().item(),
            "MAPE (%)": (percentage_sum.sum() / scored_count.sum()).item(),
            "R2": self.r_squared(squared_sum.sum(), deviation).item(),
            "Band MAE (Hz)": absolute_sum.sum(0) / n_per_band,
            "Band RMSE (Hz)": (squared_sum.sum(0) / n_per_band).sqrt(),
            "Band MAPE (%)": percentage_sum.sum(0) / scored_count.sum(0),
            "Band R2": self.r_squared(squared_sum.sum(0), band_deviation),
            "P90 MAPE (%)": geometry_p90_mape,
            "P95 MAPE (%)": geometry_p95_mape,
        }

    @staticmethod
    def _validate(predicted: Tensor, target: Tensor) -> None:
        """Reject a batch that still carries gradients or the wrong shape.

        Parameters
        ----------
        predicted : Tensor
            Predicted frequencies in hertz.
        target : Tensor
            Target frequencies in hertz.

        Raises
        ------
        ValueError
            If the predictions still require gradients, or if the two
            tensors do not share one shape (B, K, n_bands).
        """
        if predicted.requires_grad:
            raise ValueError("Detach predictions before updating metrics.")
        if predicted.ndim != 3 or predicted.shape != target.shape:
            raise ValueError(
                "predictions and targets must share one shape (B, K, n_bands); "
                f"received {tuple(predicted.shape)} and {tuple(target.shape)}."
            )
