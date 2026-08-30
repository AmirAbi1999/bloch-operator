"""PyTorch dataset and DataLoader over a dataset built by ComsolDatasetBuilder.

Expected directory structure::
------------------------------
    dataset/
    |-- geometry.csv          one row per case
    |-- sweep.csv             that case's wave vectors, as JSON arrays
    |-- data/case{i}.csv      raw response table per case
    `-- images/case{i}.npy    rendered unit-cell mask per case

This module contains:
    - BlochDataset
    - make_dataloader
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


class BlochDataset(Dataset):
    """Band structures of rendered unit cells, one item per geometry case.

    A case is one unit-cell geometry. geometry.csv names it, images/ holds
    the mask it was rasterized into, sweep.csv holds the wave vectors it
    was solved on, and data/ holds the eigenfrequencies that came back.
    One item is those three views of the same case::

        image         (1, H, W)
        wave_vectors  (K, 2)
        frequencies   (K, n_bands)

    A response table holds one row per solved band, ordered by wave vector,
    so its rows are read as K equal groups and the lowest n_bands of each
    are kept. Items are read on demand: one CSV per case, flat memory.

    Cases missing a sweep row, a response table or an image are dropped
    when the dataset is opened. K is constant within a dataset, so the
    default collate batches a split without padding.

    Parameters
    ----------
    path : str or pathlib.Path
        Dataset root holding geometry.csv, sweep.csv, data/ and images/.
    n_bands : int
        Bands kept per wave vector, counted up from the lowest.
    n_wave_vectors : int, optional
        Wave vectors kept per case, or None to keep every one it was
        solved on.
    seed : int
        Seed of the permutation the kept wave vectors are drawn from.
    wave_columns : sequence of str
        The two sweep.csv columns holding the wave-vector components.
    frequency_column : str
        Response-table column holding the band frequencies.

    Raises
    ------
    FileNotFoundError
        If geometry.csv or sweep.csv is missing.
    ValueError
        If n_bands or n_wave_vectors is not a positive integer, if
        wave_columns does not name exactly two columns, if either table
        lacks a column this class reads, if no case owns a complete set of
        files, or if wave vectors are kept where the cases were not solved
        on the same ones.
    """

    def __init__(
        self,
        path: str | Path,
        n_bands: int = 16,
        n_wave_vectors: int | None = None,
        seed: int = 0,
        wave_columns: Sequence[str] = ("kx", "ky"),
        frequency_column: str = "Frequency (Hz)",
    ) -> None:
        if not isinstance(n_bands, int) or n_bands < 1:
            raise ValueError(f"n_bands must be a positive integer, got {n_bands}.")

        if n_wave_vectors is not None and (
            not isinstance(n_wave_vectors, int) or n_wave_vectors < 1
        ):
            raise ValueError(
                f"n_wave_vectors must be a positive integer, got {n_wave_vectors}."
            )

        self.wave_columns = tuple(wave_columns)
        if len(self.wave_columns) != 2:
            raise ValueError(
                f"wave_columns must name 2 columns, got {len(self.wave_columns)}."
            )

        self.path: Path = Path(path)
        self.n_bands = n_bands
        self.frequency_column = frequency_column
        self.data_dir = self.path / "data"
        self.images_dir = self.path / "images"

        geometry = self._read_table(self.path / "geometry.csv", ["case"])
        sweep = self._read_table(self.path / "sweep.csv", ["case", *self.wave_columns])

        # Case ids are the join key, not row positions: they start at 1
        self.sweep = sweep.astype({"case": int}).set_index("case")
        self.cases = self._complete_cases(geometry["case"].astype(int))

        # Keeping every wave vector indexes as a slice, which copies nothing
        self.wave_index: Tensor | slice = (
            slice(None)
            if n_wave_vectors is None
            else self._select_wave_vectors(n_wave_vectors, seed)
        )

    def __len__(self) -> int:
        """Return the number of usable cases."""
        return len(self.cases)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        """Load the image, wave vectors and band frequencies of one case.

        Parameters
        ----------
        index : int
            Position in the case list, which is not the case id.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            image, shape (1, H, W); wave_vectors, shape (K, 2); and
            frequencies in hertz, shape (K, n_bands). K is the number kept.
        """
        case = int(self.cases[index])
        wave_vectors = self._load_wave(case)
        frequencies = self._load_frequency(case, len(wave_vectors))

        # The case is read as it was solved, then cut to the kept positions
        return (
            self._load_image(case),
            wave_vectors[self.wave_index],
            frequencies[self.wave_index],
        )

    def _complete_cases(self, cases: pd.Series) -> np.ndarray:
        """Keep the cases owning a sweep row, a response table and an image.

        Parameters
        ----------
        cases : pandas.Series
            Case ids listed in geometry.csv.

        Returns
        -------
        numpy.ndarray
            Usable case ids, in the order geometry.csv lists them.

        Raises
        ------
        ValueError
            If no case owns a complete set of files.
        """
        complete = np.array([
            case
            for case in cases
            if case in self.sweep.index
            and (self.data_dir / f"case{case}.csv").is_file()
            and (self.images_dir / f"case{case}.npy").is_file()
        ], dtype=np.int64)

        if complete.size < len(cases):
            log.warning(
                "Dropping %d of %d cases in %s: no sweep row, no response "
                "table, or no image.",
                len(cases) - complete.size, len(cases), self.path,
            )
        if complete.size == 0:
            raise ValueError(f"No case in {self.path} has a complete set of files.")

        return complete

    def _select_wave_vectors(self, n_wave_vectors: int, seed: int) -> Tensor:
        """Choose which of a case's wave vectors are kept.

        Parameters
        ----------
        n_wave_vectors : int
            Wave vectors kept per case.
        seed : int
            Seed of the permutation the kept positions are a prefix of.

        Returns
        -------
        Tensor
            Positions into one case's wave vectors, in solved order.

        Raises
        ------
        ValueError
            If the cases were solved on fewer than were asked for.
        """

        n_solved = len(self._load_wave(int(self.cases[0])))
        if n_wave_vectors > n_solved:
            raise ValueError(
                f"{self.path} solves {n_solved} wave vectors per case, fewer "
                f"than the requested n_wave_vectors={n_wave_vectors}."
            )

        kept = np.random.default_rng(seed).permutation(n_solved)[:n_wave_vectors]

        # Sorted back into solved order, the order a fixed-grid model reads
        # its rows in
        return torch.from_numpy(np.sort(kept))

    def _load_image(self, case: int) -> Tensor:
        """Load one rendered unit-cell mask.

        Parameters
        ----------
        case : int
            Case id.

        Returns
        -------
        Tensor
            Mask with shape (1, H, W), 1 in the solid and 0 in the cavity.
        """
        mask = np.load(self.images_dir / f"case{case}.npy")

        return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

    def _load_wave(self, case: int) -> Tensor:
        """Read one case's wave vectors from the sweep table.

        Parameters
        ----------
        case : int
            Case id.

        Returns
        -------
        Tensor
            Wave vectors with shape (K, 2), in the order sweep.csv lists them.
        """
        row = self.sweep.loc[case]
        components = [json.loads(row[column]) for column in self.wave_columns]

        return torch.from_numpy(np.column_stack(components).astype(np.float32))

    def _load_frequency(self, case: int, n_wave_vectors: int) -> Tensor:
        """Read one case's band frequencies, grouped by wave vector.

        Parameters
        ----------
        case : int
            Case id.
        n_wave_vectors : int
            Wave vectors K solved for this case, one row group each.

        Returns
        -------
        Tensor
            Frequencies in hertz with shape (K, n_bands).

        Raises
        ------
        ValueError
            If the frequency column is absent, if the rows are not a whole
            number of bands per wave vector, or if the case solved fewer
            bands than were requested.
        """
        path = self.data_dir / f"case{case}.csv"
        data = pd.read_csv(path)

        if self.frequency_column not in data.columns:
            raise ValueError(f"{path.name} has no '{self.frequency_column}' column.")

        frequencies = data[self.frequency_column].map(self._real).to_numpy()
        n_solved, remainder = divmod(frequencies.size, n_wave_vectors)

        if remainder:
            raise ValueError(
                f"{path.name} holds {frequencies.size} rows, which is not a "
                f"whole number of bands for {n_wave_vectors} wave vectors."
            )
        if n_solved < self.n_bands:
            raise ValueError(
                f"{path.name} holds {n_solved} bands per wave vector, fewer "
                f"than the requested n_bands={self.n_bands}."
            )

        frequencies = frequencies.reshape(n_wave_vectors, n_solved)[:, :self.n_bands]

        return torch.from_numpy(frequencies.astype(np.float32))

    @staticmethod
    def _read_table(path: Path, required: Sequence[str]) -> pd.DataFrame:
        """Read one dataset table and check the columns this class reads.

        Parameters
        ----------
        path : pathlib.Path
            File to read.
        required : sequence of str
            Column names that must be present.

        Returns
        -------
        pandas.DataFrame
            Parsed table.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If a required column is missing from it.
        """
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        table = pd.read_csv(path)
        missing = [column for column in required if column not in table.columns]
        if missing:
            raise ValueError(f"Missing columns in {path.name}: {', '.join(missing)}.")

        return table

    @staticmethod
    def _real(value: Any) -> float:
        """Return the real part of one frequency cell.

        Parameters
        ----------
        value : Any
            Cell value, either a number or a complex string.

        Returns
        -------
        float
            Real part of the value.
        """
        # COMSOL writes complex values as "1.2+3.4i", Python expects "1.2+3.4j"
        return complex(re.sub(r"i$", "j", str(value).strip())).real


def make_dataloader(
    path: str | Path,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    generator: torch.Generator | None = None,
    **dataset_kwargs: Any,
) -> DataLoader:
    """Create a DataLoader for one dataset split.

    Parameters
    ----------
    path : str or pathlib.Path
        Dataset root passed to BlochDataset.
    batch_size : int
        Cases per batch.
    shuffle : bool
        Reshuffle the cases every epoch.
    num_workers : int
        Loader worker processes; 0 loads in the calling process.
    pin_memory : bool
        Stage batches in pinned memory, which a non-blocking copy to a
        cuda device needs to overlap the transfer with compute.
    generator : torch.Generator, optional
        Drives the shuffle and the worker seeds. Without one, they come
        from the global torch generator, which a differently sized model
        leaves in a different state.
    **dataset_kwargs
        Forwarded to BlochDataset.

    Returns
    -------
    torch.utils.data.DataLoader
        Loader yielding image, wave-vector and frequency batches.
    """
    dataset = BlochDataset(path, **dataset_kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
