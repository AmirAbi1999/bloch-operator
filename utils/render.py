"""Rasterization of harmonically perturbed cavity unit cells.

A cavity is described in polar coordinates by a circle whose radius is
modulated by cosine harmonics of order 4*n, so the boundary keeps the D4
symmetry of the square unit cell:

    r(theta) = R * (1 + sum_i ai*cos(4*ni*theta))

R is set from cavity_fraction, the share of the unit-cell area occupied by
the cavity. render_cell samples that boundary on a square pixel grid and
returns the solid mask consumed by the BlochOperator geometry encoder.

A geometry sweep names these parameters one scalar column at a time, since
that is what a COMSOL model declares them as. render reads one
such row back into a cavity, and is what a dataset build renders with.

This module contains:
    - CavityGeometry
    - render_cell
    - render
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["CavityGeometry", "render_cell", "render"]


@dataclass(frozen=True)
class CavityGeometry:
    """Shape parameters of a single harmonically perturbed cavity.

    Attributes
    ----------
    harmonic_orders : tuple[int, ...]
        Order ni of each cosine harmonic, applied as 4*ni*theta.
    harmonic_amplitudes : tuple[float, ...]
        Relative amplitude ai of each harmonic, one per order.
    cavity_fraction : float
        Share of the unit-cell area occupied by the cavity.
    """

    harmonic_orders: tuple[int, ...]
    harmonic_amplitudes: tuple[float, ...]
    cavity_fraction: float

    def __post_init__(self) -> None:
        """Reject parameters that do not describe a valid cavity.

        Raises
        ------
        ValueError
            If cavity_fraction is outside (0, 1), if the orders and the
            amplitudes do not pair up, if an order is not positive, or if
            the amplitudes are large enough to drive the boundary radius to
            zero.
        """
        if not 0.0 < self.cavity_fraction < 1.0:
            raise ValueError("cavity_fraction must lie strictly between 0 and 1.")

        if len(self.harmonic_orders) != len(self.harmonic_amplitudes):
            raise ValueError(
                f"{len(self.harmonic_orders)} harmonic orders were given "
                f"against {len(self.harmonic_amplitudes)} amplitudes."
            )

        if any(order <= 0 for order in self.harmonic_orders):
            raise ValueError("Harmonic orders must be positive.")

        if sum(abs(amplitude) for amplitude in self.harmonic_amplitudes) >= 1.0:
            raise ValueError(
                "Harmonic amplitudes must sum to less than 1 in magnitude."
            )


def _cavity_boundary(theta: np.ndarray, geom: CavityGeometry) -> np.ndarray:
    """Evaluate the cavity boundary radius at the given polar angles.

    Parameters
    ----------
    theta : numpy.ndarray
        Polar angles in radians, any shape.
    geom : CavityGeometry
        Harmonic orders, amplitudes and area fraction of the cavity.

    Returns
    -------
    numpy.ndarray
        Boundary radius in unit-cell lengths, same shape as theta.
    """
    # The harmonics add area of their own, so the unmodulated radius is
    # corrected for them to hold cavity_fraction fixed
    correction = 0.5 * sum(
        amplitude ** 2 for amplitude in geom.harmonic_amplitudes
    )
    radius = np.sqrt(geom.cavity_fraction / (np.pi * (1 + correction)))

    modulation = sum(
        amplitude * np.cos(4 * order * theta)
        for amplitude, order in zip(geom.harmonic_amplitudes, geom.harmonic_orders)
    )

    return radius * (1 + modulation)


def render_cell(
    geom: CavityGeometry,
    *,
    n_pixels: int = 128,
) -> np.ndarray:
    """Rasterize one cavity unit cell into a binary solid mask.

    Parameters
    ----------
    geom : CavityGeometry
        Shape parameters of the cavity carved out of the unit cell.
    n_pixels : int
        Side length of the square pixel grid.

    Returns
    -------
    numpy.ndarray
        Mask with shape (n_pixels, n_pixels) and dtype uint8, 1 in the
        solid and 0 inside the cavity.

    Raises
    ------
    ValueError
        If n_pixels is not positive.
    """
    if n_pixels <= 0:
        raise ValueError("n_pixels must be positive.")

    # Pixel centers, spanning one unit cell in units of pixels
    coords = np.linspace(-n_pixels / 2 + 1 / 2.0, n_pixels / 2 - 1 / 2.0, n_pixels)
    y, x = np.ogrid[:n_pixels, :n_pixels]
    x = coords[x]
    y = coords[::-1][y]

    # Compare each pixel against the boundary radius at its own angle
    r = np.hypot(x, y)
    theta = np.arctan2(x, y)
    r_boundary = n_pixels * _cavity_boundary(theta, geom)

    return (r > r_boundary).astype(np.uint8)


def render(
    values: Mapping[str, Any],
    *,
    n_pixels: int = 128,
) -> np.ndarray:
    """Rasterize the cavity one geometry-sweep row describes.

    Harmonics are numbered from 1 in the keys, harmonic_order1 beside
    harmonic_amplitude1, and are read until the numbering stops. A numbered
    order whose amplitude is absent raises rather than dropping a harmonic
    the case was solved with.

    Parameters
    ----------
    values : Mapping[str, Any]
        Parameter values of one case, cavity columns included.
    n_pixels : int
        Side length of the square pixel grid.

    Returns
    -------
    numpy.ndarray
        Mask with shape (n_pixels, n_pixels) and dtype uint8, 1 in the
        solid and 0 inside the cavity.

    Raises
    ------
    KeyError
        If cavity_fraction is absent, or a numbered order has no amplitude
        beside it.
    ValueError
        If the row does not describe a valid cavity, or n_pixels is not
        positive.
    """
    orders: list[int] = []
    amplitudes: list[float] = []
    index = 1

    while f"harmonic_order{index}" in values:
        orders.append(int(values[f"harmonic_order{index}"]))
        amplitudes.append(float(values[f"harmonic_amplitude{index}"]))
        index += 1

    return render_cell(
        CavityGeometry(
            harmonic_orders=tuple(orders),
            harmonic_amplitudes=tuple(amplitudes),
            cavity_fraction=float(values["cavity_fraction"]),
        ),
        n_pixels=n_pixels,
    )
