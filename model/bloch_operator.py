"""BlochOperator: structure-preserving latent Bloch operator for a
finite-thickness 2D elastic unit cell that is periodic in x and y.

The network learns three real latent factors:

    H(qx, qy) = G0 + i*qx*Gx + i*qy*Gy
    D(qx, qy) = H(qx, qy)^H * H(qx, qy)

D is Hermitian positive semidefinite by construction. Its lowest eigenvalues
represent squared normalized frequencies.

At Gamma, qx = qy = 0:

    D(0, 0) = G0^T G0

For a connected, traction-free 2D solid unit cell with periodicity only in
x and y, there are two acoustic zero modes at Gamma. G0 is therefore
parameterized to have exactly a two-dimensional right nullspace.

This module contains:
    - ConvBlock
    - GeometryEncoder
    - MatrixHead
    - AcousticNullHead
    - BlochOperatorConfig
    - BlochOperator
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class ConvBlock(nn.Module):
    """Stride-2 convolutional downsampling block.

    Parameters
    ----------
    in_channels : int
        Channels entering the block.
    out_channels : int
        Channels produced by the block.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Downsample a feature map by a factor of two.

        Parameters
        ----------
        x : Tensor
            Feature map with shape (B, in_channels, H, W).

        Returns
        -------
        Tensor
            Feature map with shape (B, out_channels, H // 2, W // 2).
        """
        return self.block(x)


class GeometryEncoder(nn.Module):
    """Convert a binary or grayscale geometry image into a latent geometry vector.

    Parameters
    ----------
    in_channels : int
        Channels in the geometry image.
    latent_dim : int
        Width of the latent geometry vector.
    widths : tuple[int, ...]
        Channel width of each stride-2 ConvBlock stage.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 256,
        widths: tuple[int, ...] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()

        blocks: list[ConvBlock] = []
        current = in_channels
        for width in widths:
            blocks.append(ConvBlock(current, width))
            current = width

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=current, out_features=latent_dim),
            nn.SiLU(),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Encode geometry images.

        Parameters
        ----------
        images : Tensor
            Geometry images with shape (B, C, H, W).

        Returns
        -------
        Tensor
            Latent tensor with shape (B, latent_dim).
        """
        images = self.features(images)
        images = self.pool(images)
        images = self.projection(images)
        return images


class MatrixHead(nn.Module):
    """Predict an unconstrained square matrix from the latent geometry vector.

    Parameters
    ----------
    latent_dim : int
        Width of the latent geometry vector.
    matrix_dim : int
        Side length of the predicted square matrix.
    """

    def __init__(self, latent_dim: int = 256, matrix_dim: int = 16) -> None:
        super().__init__()

        self.matrix_dim = matrix_dim

        self.linear = nn.Linear(
            in_features=latent_dim,
            out_features=matrix_dim * matrix_dim,
        )

    def forward(self, latent: Tensor) -> Tensor:
        """Predict a batch of square matrices.

        Parameters
        ----------
        latent : Tensor
            Latent geometry vector with shape (B, latent_dim).

        Returns
        -------
        Tensor
            Matrices with shape (B, matrix_dim, matrix_dim).
        """
        batch_size = latent.shape[0]
        matrix = self.linear(latent)
        matrix = matrix.reshape(batch_size, self.matrix_dim, self.matrix_dim)

        return matrix


class AcousticNullHead(nn.Module):
    """Predict G0 with a prescribed number of exact acoustic null directions.

    The first "n_acoustic_modes columns" are fixed to zero.

    Parameters
    ----------
    latent_dim : int
        Width of the latent geometry vector.
    matrix_dim : int
        Side length of the predicted G0 matrix.
    n_acoustic_modes : int
        Number of leading columns fixed to zero, one per acoustic null
        direction.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        matrix_dim: int = 16,
        n_acoustic_modes: int = 2,
    ) -> None:
        super().__init__()
        self.matrix_dim = matrix_dim
        self.n_acoustic_modes = n_acoustic_modes
        self.active_dim = matrix_dim - n_acoustic_modes

        self.active_head = nn.Linear(
            in_features=latent_dim,
            out_features=matrix_dim * self.active_dim,
        )

    def forward(self, latent: Tensor) -> Tensor:
        """Predict the constrained Gamma-point factor G0.

        Parameters
        ----------
        latent : Tensor
            Latent geometry vector with shape (B, latent_dim).

        Returns
        -------
        Tensor
            G0 with shape (B, matrix_dim, matrix_dim).
        """
        batch_size = latent.shape[0]

        active_block = self.active_head(latent).reshape(
            batch_size,
            self.matrix_dim,
            self.active_dim,
        )

        zero_columns = torch.zeros(
            batch_size,
            self.matrix_dim,
            self.n_acoustic_modes,
            dtype=latent.dtype,
            device=latent.device,
        )

        g0 = torch.cat([zero_columns, active_block], dim=-1)

        return g0


@dataclass(frozen=True)
class BlochOperatorConfig:
    """Hyperparameters for the latent Bloch operator.

    Attributes
    ----------
    in_channels : int
        Channels in the geometry image.
    latent_dim : int
        Width of the latent geometry vector.
    operator_dim : int
        Side length r of the operator matrices G0, Gx and Gy.
    n_bands : int
        Number of lowest bands retained, must not exceed operator_dim.
    n_acoustic_modes : int
        Number of exact acoustic zero modes imposed on G0 at Gamma.
    frequency_scale : float
        Hertz per unit of normalized frequency, used to convert
        eigenvalues to frequencies.
    use_float64 : bool
        Run the eigensolver in double precision.
    """

    in_channels: int = 1
    latent_dim: int = 256
    operator_dim: int = 12
    n_bands: int = 6
    n_acoustic_modes: int = 2
    frequency_scale: float = 1000.0
    use_float64: bool = True


class BlochOperator(nn.Module):
    """The model encodes a geometry image into a latent vector and predicts
    three real factor matrices G0, Gx, and Gy. For each nondimensional wave
    vector (qx, qy), it constructs

        H = G0 + i*qx*Gx + i*qy*Gy
        D = H.mH @ H

    and computes the lowest eigenvalues of the Hermitian positive-
    semidefinite matrix D using torch.linalg.eigvalsh.

    Parameters
    ----------
    config : BlochOperatorConfig
        Model dimensions, number of retained bands, acoustic modes,
        frequency scale, and eigensolver precision.

    Raises
    ------
    ValueError
        If config.n_bands exceeds config.operator_dim.
    """

    def __init__(
        self,
        config: BlochOperatorConfig = BlochOperatorConfig(),
    ) -> None:
        super().__init__()

        if config.n_bands > config.operator_dim:
            raise ValueError(
                "Number of bands must not exceed the operator dimension."
            )

        self.config: BlochOperatorConfig = config

        # Geometry image -> latent geometry vector z
        self.encoder = GeometryEncoder(
            in_channels=config.in_channels,
            latent_dim=config.latent_dim,
        )

        self.g0 = AcousticNullHead(
            latent_dim=config.latent_dim,
            matrix_dim=config.operator_dim,
            n_acoustic_modes=config.n_acoustic_modes,
        )

        self.gx = MatrixHead(
            latent_dim=config.latent_dim,
            matrix_dim=config.operator_dim,
        )

        self.gy = MatrixHead(
            latent_dim=config.latent_dim,
            matrix_dim=config.operator_dim,
        )

        self.register_buffer(
            "frequency_scale",
            torch.tensor(float(config.frequency_scale)),
        )

    def operators(self, images: Tensor) -> dict[str, Tensor]:
        """Predict geometry-dependent factor matrices.

        Parameters
        ----------
        images : Tensor
            Geometry images with shape (B, C, H, W).

        Returns
        -------
        dict[str, Tensor]
            latent, shape (B, latent_dim); G0, Gx and Gy, each with shape
            (B, matrix_dim, matrix_dim).
        """
        latent = self.encoder(images)
        return {
            "latent": latent,
            "G0": self.g0(latent),
            "Gx": self.gx(latent),
            "Gy": self.gy(latent),
        }

    @staticmethod
    def prepare_wavevectors(wave_points: Tensor, batch_size: int) -> Tensor:
        """Convert wave-vector input to shape (B, K, 2).

        Parameters
        ----------
        wave_points : Tensor
            Wave vectors with shape (K, 2) or (B, K, 2).
        batch_size : int
            Batch size the wave vectors are broadcast against.

        Returns
        -------
        Tensor
            Wave vectors with shape (B, K, 2).

        Raises
        ------
        ValueError
            If wave_points is neither two- nor three-dimensional, if its
            last dimension is not 2, or if its batch dimension is neither
            1 nor batch_size.
        """
        if wave_points.ndim == 2:
            if wave_points.shape[-1] != 2:
                raise ValueError("wave-vectors are in 2-pairs")

            wave_points = wave_points.unsqueeze(0).expand(batch_size, -1, -1)

        elif wave_points.ndim == 3:
            if wave_points.shape[-1] != 2:
                raise ValueError("wave-vectors are in 2-pairs")
            if wave_points.shape[0] == 1 and batch_size > 1:
                wave_points = wave_points.expand(batch_size, -1, -1)
            elif wave_points.shape[0] != batch_size:
                raise ValueError(
                    "The batch dimension of q_points must match images."
                )

        else:
            raise ValueError("wave-vectors should be 2-dimensional")

        return wave_points

    def forward(
        self,
        images: Tensor,
        wave_vectors: Tensor,
    ) -> dict[str, Tensor]:
        """Predict the lowest Bloch bands for each geometry and wave vector.

        Parameters
        ----------
        images : Tensor
            Geometry images, shape (batch, channels, height, width).
        wave_vectors : Tensor
            Nondimensional wave vectors, shape (p, 2) or (batch, p, 2).

        Returns
        -------
        dict[str, Tensor]
            latent          (batch, latent_dim)
            G0, Gx, Gy      (batch, r, r)
            H               (batch, p, r, r), complex
            D               (batch, p, r, r), complex Hermitian PSD
            eigenvalues     (batch, p, n_bands)
            frequencies     (batch, p, n_bands)
        """
        batch_size = images.size(0)
        wave_vectors = self.prepare_wavevectors(wave_vectors, batch_size)

        # Predict geometry-dependent matrices
        predicted_operators = self.operators(images)
        latent = predicted_operators["latent"]
        g0 = predicted_operators["G0"]
        gx = predicted_operators["Gx"]
        gy = predicted_operators["Gy"]

        # Choose eigensolver precision
        real_dtype = (
            torch.float64
            if self.config.use_float64
            else g0.dtype
        )
        complex_dtype = (
            torch.complex128
            if real_dtype == torch.float64
            else torch.complex64
        )

        # Convert the matrices to eigensolver precision
        g0_eig = g0.to(dtype=real_dtype)
        gx_eig = gx.to(dtype=real_dtype)
        gy_eig = gy.to(dtype=real_dtype)
        wave_vectors_eig = wave_vectors.to(device=g0.device, dtype=real_dtype)

        # Extract vx and vy
        wx = wave_vectors_eig[..., 0, None, None]
        wy = wave_vectors_eig[..., 1, None, None]

        # Insert wave-vector dimension into G matrices
        g0_complex = g0_eig.to(complex_dtype).unsqueeze(1)
        gx_complex = gx_eig.to(complex_dtype).unsqueeze(1)
        gy_complex = gy_eig.to(complex_dtype).unsqueeze(1)

        # Construct the Bloch operator H
        h_operator = (
            g0_complex
            + 1j * wx * gx_complex
            + 1j * wy * gy_complex
        )

        # Construct the Hermitian PSD operator
        d_operator = h_operator.mH @ h_operator

        # Solve the eigenvalue problem
        all_eigenvalues = torch.linalg.eigvalsh(d_operator).real
        all_eigenvalues = torch.clamp(all_eigenvalues, min=0.0)

        # Retain the first requested bands
        eigenvalues = all_eigenvalues[..., : self.config.n_bands]

        # Convert eigenvalues to frequencies
        frequencies = (
            self.frequency_scale
            * torch.sqrt(eigenvalues)
        )

        return {
            "latent": latent,
            "G0": g0,
            "Gx": gx,
            "Gy": gy,
            "H": h_operator,
            "D": d_operator,
            "eigenvalues": eigenvalues,
            "frequencies": frequencies,
        }
