"""Wave-vector broadcasting, symmetry augmentation and diagnostic utilities."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


def prepare_wavevectors(wave_vectors: Tensor, batch_size: int) -> Tensor:
    """Broadcast wave vectors to one set per geometry, shape (B, K, 2).

    Parameters
    ----------
    wave_vectors : Tensor
        Wave vectors shared by the batch, shape (K, 2) or (1, K, 2), or
        given per geometry, shape (B, K, 2).
    batch_size : int
        Geometries in the batch.

    Returns
    -------
    Tensor
        Wave vectors with shape (B, K, 2), a view of the input.

    Raises
    ------
    ValueError
        If wave_vectors is neither (K, 2) nor (B, K, 2), or holds a set
        count that is neither 1 nor batch_size.
    """
    if wave_vectors.ndim not in (2, 3) or wave_vectors.shape[-1] != 2:
        raise ValueError(
            "wave_vectors must have shape (K, 2) or (B, K, 2); "
            f"received {tuple(wave_vectors.shape)}."
        )
    if wave_vectors.ndim == 3 and wave_vectors.shape[0] not in (1, batch_size):
        raise ValueError(
            "wave_vectors must hold one set per geometry or one for the batch; "
            f"received {wave_vectors.shape[0]} for a batch of {batch_size}."
        )

    return wave_vectors.expand(batch_size, -1, -1)


def d4_orbit(wave_vectors: Tensor) -> Tensor:
    """Return the eight D4-related versions of each two-dimensional wave vector.

    Parameters
    ----------
    wave_vectors : Tensor
        Wave vectors with shape (..., 2).

    Returns
    -------
    Tensor
        Wave-vector orbit with shape (..., 8, 2).

    Raises
    ------
    ValueError
        If wave_vectors does not have shape (..., 2).
    """
    if wave_vectors.ndim < 1 or wave_vectors.shape[-1] != 2:
        raise ValueError(
            "wave_vectors must have shape (..., 2); "
            f"received {tuple(wave_vectors.shape)}."
        )
    qx = wave_vectors[..., 0]
    qy = wave_vectors[..., 1]

    return torch.stack(
        (
            torch.stack((+qx, +qy), -1),
            torch.stack((+qx, -qy), -1),
            torch.stack((-qx, +qy), -1),
            torch.stack((-qx, -qy), -1),
            torch.stack((+qy, +qx), -1),
            torch.stack((+qy, -qx), -1),
            torch.stack((-qy, +qx), -1),
            torch.stack((-qy, -qx), -1),
        ),
        dim=-2
    )


def random_d4_transform(
    wave_vectors: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Apply one randomly selected D4 transform to every input wave vector.

    Parameters
    ----------
    wave_vectors : Tensor
        Wave vectors with shape (..., 2).
    generator : torch.Generator, optional
        Generator driving the transform selection.

    Returns
    -------
    Tensor
        Transformed wave vectors with shape (..., 2).
    """
    orbit = d4_orbit(wave_vectors)
    index_shape = wave_vectors.shape[:-1]

    indices = torch.randint(
        low=0,
        high=8,
        size=index_shape,
        device=wave_vectors.device,
        generator=generator,
    )

    gather_index = indices[..., None, None].expand(
        *indices.shape,
        1,
        2,
    )
    return torch.gather(orbit, dim=-2, index=gather_index).squeeze(-2)


@torch.no_grad()
def gamma_diagnostics(
    model: nn.Module,
    images: Tensor,
    tolerance: float = 1.0e-8,
) -> dict[str, Tensor]:
    """Inspect the acoustic eigenvalues produced at the Gamma point.

    Parameters
    ----------
    model : torch.nn.Module
        Model evaluated at Gamma.
    images : Tensor
        Geometry images with shape (B, C, H, W).
    tolerance : float
        Largest acoustic eigenvalue still counted as zero.

    Returns
    -------
    dict[str, Tensor]
        gamma_eigenvalues, shape (B, n_bands); max_acoustic_eigenvalue,
        min_first_optical_eigenvalue and acoustic_constraint_passed, each
        a scalar.

    Raises
    ------
    ValueError
        If the model retains no band beyond the acoustic modes.
    """
    gamma = torch.zeros(
        images.shape[0],
        1,
        2,
        dtype=images.dtype,
        device=images.device,
    )
    values = model(images, gamma)["eigenvalues"][:, 0, :]

    config: Any = model.config
    n_acoustic = config.n_acoustic_modes
    if values.shape[-1] <= n_acoustic:
        raise ValueError(
            "The model must retain at least one band beyond the acoustic modes."
        )

    acoustic = values[:, :n_acoustic]
    first_optical = values[:, n_acoustic]

    return {
        "gamma_eigenvalues": values,
        "max_acoustic_eigenvalue": acoustic.max(),
        "min_first_optical_eigenvalue": first_optical.min(),
        "acoustic_constraint_passed": acoustic.max() <= tolerance,
    }
