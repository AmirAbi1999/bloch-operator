"""Evaluation Metrics for bloch-operator frequency prediction.

Expected tensor shape:
    predictions : (B, K, N)
    targets     : (B, K, N)

B = geometries, K = wave vectors, N = Frequency bands.

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

__all__ = ["MetricTracker"]


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
        self.percentage_samples: list[Tensor] = []
        self.geometry_mape: list[Tensor] = []
        self.geometry_r2: list[Tensor] = []

    def reset(self) -> None:
        """Drop every statistic accumulated so far."""
        self.n_geometries = 0
        self.band_sums = torch.zeros(())
        self.percentage_samples.clear()
        self.geometry_mape.clear()
        self.geometry_r2.clear()

    def update(self,
               predicted: Tensor,
               target: Tensor,
    ) -> None:
        """Accumulate the statistics of one batch.

        Parameters
        ----------
        predicted : Tensor
            Predicted frequencies in hertz, shape (B, K, N).
        target : Tensor
            Target frequencies in hertz, shape (B, K, N).

        Raises
        ------
        ValueError
            If the predictions still require gradients, or if the two
            tensors do not share one shape (B, K, N).
        """

        self._validate(predicted, target)

        target = target.to(device=predicted.device,
                           dtype=predicted.dtype
        )
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
        symmetric = (
            200.0 * absolute
            / (predicted + target).clamp_min(self.relative_floor)
        )

        # One stacked sum over the batch, in the order compute() unpacks it
        self.n_geometries += target.shape[0]
        self.band_sums = self.band_sums + torch.stack((
            absolute.sum(0),
            squared.sum(0),
            percentage.nansum(0),
            symmetric.sum(0),
            scored.to(target.dtype).sum(0),
            target.sum(0),
            target.square().sum(0),
        )).cpu()

        # Quantiles and per-geometry scores are exact only over the samples
        deviation = (
                target - target.mean((1, 2), keepdim=True)
        ).square().sum((1, 2))

        self.percentage_samples.append(percentage.flatten(0, 1).cpu())
        self.geometry_mape.append(percentage.nanmean((1, 2)).cpu())
        self.geometry_r2.append(self.r_squared(squared.sum((1, 2)), deviation).cpu())

    def compute(self) -> dict[str, float | Tensor]:
        """Return metrics over all accumulated batches.

        Returns
        -------
        dict[str, float | Tensor]
            mae, rmse, mape, smape, r2       float, over every band
            p90_ape, p95_ape                 float, tails of the percentage error
            per_band_mae, per_band_rmse      (N, )
            per_band_mape, per_band_r2       (N, )
            per_band_p90_ape                 (N, )
            per_band_p95_ape                 (N, )
            per_geometry_mape                (B, )
            per_geometry_r2                  (B, )
            per_geometry_p90_mape            float, tails over the geometries
            per_geometry_p95_mape            float
            per_geometry_p90_r2              float
            per_geometry_p95_r2              float

            A metric with nothing scored under it reads nan.

        Raises
        ------
        ValueError
            If no batch has been accumulated.
        """

        if not self.n_geometries:
            raise ValueError("No batches have been added.")

        # Every sum holds one entry per (wave vector, band) pair
        (absolute_sum, squared_sum, percentage_sum, symmetric_sum,
         scored_count, target_sum, target_square_sum) = self.band_sums

        n_wave_vectors, n_bands = absolute_sum.shape
        n_values = self.n_geometries * n_wave_vectors * n_bands
        n_per_band = self.n_geometries * n_wave_vectors

        percentage_samples = torch.cat(self.percentage_samples)
        geometry_mape = torch.cat(self.geometry_mape)
        geometry_r2 = torch.cat(self.geometry_r2)

        # Quantiles of the scored bands, then of the per-geometry scores
        quantiles = torch.tensor([0.90, 0.95], dtype=percentage_samples.dtype)
        p90_ape, p95_ape = percentage_samples.nanquantile(quantiles).tolist()
        band_p90_ape, band_p95_ape = percentage_samples.nanquantile(quantiles, dim=0)
        geometry_p90_mape, geometry_p95_mape = geometry_mape.nanquantile(quantiles).tolist()
        geometry_p90_r2, geometry_p95_r2 = geometry_r2.nanquantile(quantiles).tolist()

        # Squared deviation of the targets, over every band and per band
        deviation = target_square_sum.sum() - target_sum.sum().square() / n_values
        band_deviation = (
            target_square_sum.sum(0)
            - target_sum.sum(0).square() / n_per_band
        )

        return {
            "mae": (absolute_sum.sum() / n_values).item(),
            "rmse": (squared_sum.sum() / n_values).sqrt().item(),
            "mape": (percentage_sum.sum() / scored_count.sum()).item(),
            "smape": (symmetric_sum.sum() / n_values).item(),
            "r2": self.r_squared(squared_sum.sum(), deviation).item(),
            "p90_ape": p90_ape,
            "p95_ape": p95_ape,
            "per_band_mae": absolute_sum.sum(0) / n_per_band,
            "per_band_rmse": (squared_sum.sum(0) / n_per_band).sqrt(),
            "per_band_mape": percentage_sum.sum(0) / scored_count.sum(0),
            "per_band_r2": self.r_squared(squared_sum.sum(0), band_deviation),
            "per_band_p90_ape": band_p90_ape,
            "per_band_p95_ape": band_p95_ape,
            "per_geometry_mape": geometry_mape,
            "per_geometry_r2": geometry_r2,
            "per_geometry_p90_mape": geometry_p90_mape,
            "per_geometry_p95_mape": geometry_p95_mape,
            "per_geometry_p90_r2": geometry_p90_r2,
            "per_geometry_p95_r2": geometry_p95_r2,
        }

    @staticmethod
    def _validate(
            predicted: Tensor,
            target: Tensor,
    ) -> None:
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
            tensors do not share one shape (B, K, N).
        """
        if predicted.requires_grad:
            raise ValueError("Detach predictions before updating metrics.")
        if predicted.ndim != 3 or predicted.shape != target.shape:
            raise ValueError(
                "predictions and targets must share one shape (B, K, N); "
                f"received {tuple(predicted.shape)} and {tuple(target.shape)}."
            )
