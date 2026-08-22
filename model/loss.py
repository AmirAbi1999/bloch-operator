"""Loss functions for supervising the latent Bloch eigenspectrum.

Smooth L1 loss between the predicted eigenvalues and the target frequencies
carried to the same scale, lambda = (f / frequency_scale)^2, with both sides
taken under log1p and an optional weight per band.

This module contains:
    - EigenvalueSupervisedLoss
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class EigenvalueSupervisedLoss(nn.Module):
    """Compare predicted eigenvalues with squared normalized target frequencies.

    Parameters
    ----------
    frequency_scale : float
        Hertz per unit of normalized frequency, used to convert target
        frequencies to eigenvalues.
    huber_beta : float
        Transition point between the quadratic and linear regions of the
        Smooth L1 loss.
    band_weights : Tensor, optional
        One weight per band, applied to the elementwise loss.
    """

    def __init__(
        self,
        frequency_scale: float = 1000.0,
        huber_beta: float = 1.0e-2,
        band_weights: Tensor | None = None,
    ) -> None:
        super().__init__()

        self.frequency_scale = float(frequency_scale)
        self.huber_beta = float(huber_beta)

        if band_weights is None:
            self.register_buffer("band_weights", None)
        else:
            self.register_buffer(
                "band_weights",
                band_weights.detach().clone().float(),
            )

    def forward(
        self,
        predicted_eigenvalues: Tensor,
        target_frequencies: Tensor,
    ) -> Tensor:
        """Compute the elementwise weighted Smooth L1 loss.

        Parameters
        ----------
        predicted_eigenvalues : Tensor
            Predicted eigenvalues, shape [..., n_bands].
        target_frequencies : Tensor
            Target frequencies in hertz, shape [..., n_bands].

        Returns
        -------
        Tensor
            Scalar mean loss.

        Raises
        ------
        ValueError
            If band_weights does not contain one value per band.
        """
        target_frequencies = target_frequencies.to(
            device=predicted_eigenvalues.device,
            dtype=predicted_eigenvalues.dtype,
        )
        target_lambda = (
                target_frequencies / self.frequency_scale
        ).square()

        prediction = torch.log1p(
            torch.clamp_min(predicted_eigenvalues, 0.0)
        )
        target = torch.log1p(torch.clamp_min(target_lambda, 0.0))

        element_loss = F.smooth_l1_loss(
            prediction,
            target,
            beta=self.huber_beta,
            reduction="none",
        )

        if self.band_weights is not None:
            weights = self.band_weights.to(
                dtype=element_loss.dtype,
                device=element_loss.device,
            )
            if weights.numel() != element_loss.shape[-1]:
                raise ValueError(
                    "band_weights must contain one value per band."
                )
            element_loss = element_loss * weights

        return element_loss.mean()
