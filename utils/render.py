"""Rasterization of harmonically perturbed cavity unit cells.

A cavity is described in polar coordinates by a circle whose radius is
modulated by two cosine harmonics of order 4*n, so the boundary keeps the
D4 symmetry of the square unit cell:

    r(theta) = R * (1 + a1*cos(4*n1*theta) + a2*cos(4*n2*theta))

R is set from cavity_fraction, the share of the unit-cell area occupied by
the cavity. render_cell samples that boundary on a square pixel grid and
returns the solid mask consumed by the BlochOperator geometry encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CavityGeometry", "render_cell"]



@dataclass(frozen=True)
class CavityGeometry:
    """Shape parameters of a single harmonically perturbed cavity.

    Attributes
    ----------
    harmonic_order1 : int
        Order n1 of the first cosine harmonic, applied as 4*n1*theta.
    harmonic_order2 : int
        Order n2 of the second cosine harmonic, applied as 4*n2*theta.
    harmonic_amplitude1 : float
        Relative amplitude a1 of the first harmonic.
    harmonic_amplitude2 : float
        Relative amplitude a2 of the second harmonic.
    cavity_fraction : float
        Share of the unit-cell area occupied by the cavity.
    """

    harmonic_order1: int
    harmonic_order2: int
    harmonic_amplitude1: float
    harmonic_amplitude2: float
    cavity_fraction: float

    def __post_init__(self)->None:
        """Reject parameters that do not describe a valid cavity.

        Raises
        ------
        ValueError
            If cavity_fraction is outside (0, 1), if either harmonic order
            is not positive, or if the harmonic amplitudes are large enough
            to drive the boundary radius to zero.
        """
        if not 0.0 < self.cavity_fraction < 1.0:
            raise ValueError("cavity_fraction must lie strictly between 0 and 1.")

        if self.harmonic_order1 <= 0 or self.harmonic_order2 <= 0:
            raise ValueError("Harmonic orders must be positive.")

        if abs(self.harmonic_amplitude1) + abs(self.harmonic_amplitude2) >= 1.0:
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
    radius = np.sqrt(
        geom.cavity_fraction
        / (np.pi * (1 + 0.5 * (geom.harmonic_amplitude1 ** 2 + geom.harmonic_amplitude2 ** 2)))
    )

    return radius * (
        1
        + geom.harmonic_amplitude1 * np.cos(4 * geom.harmonic_order1 * theta)
        + geom.harmonic_amplitude2 * np.cos(4 * geom.harmonic_order2 * theta)
    )


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

    # Pixel centres, spanning one unit cell in units of pixels.
    coords = np.linspace(-n_pixels / 2 + 1 / 2.0, n_pixels / 2 - 1 / 2.0, n_pixels)
    y, x = np.ogrid[:n_pixels, :n_pixels]
    x = coords[x]
    y = coords[::-1][y]

    # Compare each pixel against the boundary radius at its own angle.
    r = np.hypot(x, y)
    theta = np.arctan2(x, y)
    r_boundary = n_pixels * _cavity_boundary(theta, geom)

    return (r > r_boundary).astype(np.uint8)
