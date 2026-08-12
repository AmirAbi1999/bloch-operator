"""Parallel COMSOL geometry sweeps with per-worker solver sessions.

Each worker process starts one COMSOL client, loads the model once, and
reuses that session for many cases; solver objects never cross process
boundaries. The parent process performs all shared file writes and all
rendering, so partial results are always consistent on disk even if a
worker dies mid-sweep.

Case identity comes from the sweeps themselves: geometry_sweep and
auxiliary_sweep each carry a "case" column, and the two are joined on it.

This module contains:
    - ComsolDatasetBuilder
"""

from __future__ import annotations

import atexit
import json
import logging
import multiprocessing
import os
import pathlib
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import MISSING, fields
from typing import Any

import numpy as np
import pandas as pd

from utils import CavityGeometry, render_cell

from .comsol_runner import ComsolRunner, ComsolTags

log = logging.getLogger(__name__)
_worker_runner: ComsolRunner | None = None

# Geometry-sweep column name to the type CavityGeometry expects for it
_CAVITY_CASTS: dict[str, type] = {
    "harmonic_order1": int,
    "harmonic_order2": int,
    "harmonic_amplitude1": float,
    "harmonic_amplitude2": float,
    "cavity_fraction": float,
}


def _physical_cores() -> int:
    """Return the number of physical CPU cores on this machine.

    Solver threads should be sized against physical cores: os.cpu_count
    reports logical processors, which double-counts every core on a machine
    with simultaneous multithreading and hides real oversubscription.

    Returns
    -------
    int
        Physical core count, falling back to the logical count when psutil
        is unavailable or cannot determine it.
    """
    try:
        import psutil
    except ImportError:
        return os.cpu_count() or 1

    return psutil.cpu_count(logical=False) or os.cpu_count() or 1


def _peak_rss_mb() -> float | None:
    """Return the peak resident set size of the calling process, in MiB.

    Windows exposes a true high-water mark; elsewhere the current resident
    size stands in for it, which tracks closely because neither the JVM nor
    the COMSOL kernel returns memory to the operating system between cases.

    Returns
    -------
    float or None
        Peak memory in mebibytes, None when psutil is not installed.
    """
    try:
        import psutil
    except ImportError:
        return None

    info = psutil.Process().memory_info()
    return round(getattr(info, "peak_wset", info.rss) / 1024 ** 2, 1)


def _close_worker() -> None:
    """Release the worker-local COMSOL model and client."""
    global _worker_runner

    if _worker_runner is not None:
        try:
            _worker_runner.close(disconnect=True)
        except Exception as e:
            log.warning("COMSOL worker cleanup failed: %s", e)
        _worker_runner = None


def _init_worker(
    model_path: str,
    cores: int,
    tags: ComsolTags,
    rebuild_geometry: bool,
    rebuild_mesh: bool,
    clear_results_table: bool,
) -> None:
    """Start one COMSOL client and runner inside a worker process.

    Parameters
    ----------
    model_path : str
        COMSOL .mph model file loaded into this worker's client.
    cores : int
        Solver threads granted to the client.
    tags : ComsolTags
        COMSOL node tags forwarded to the runner.
    rebuild_geometry, rebuild_mesh, clear_results_table : bool
        Forwarded to ComsolRunner.
    """
    import mph

    global _worker_runner

    client = mph.start(cores=cores)
    model = client.load(model_path)
    _worker_runner = ComsolRunner(
        mph_client=client,
        mph_model=model,
        tags=tags,
        rebuild_geometry=rebuild_geometry,
        rebuild_mesh=rebuild_mesh,
        clear_results_table=clear_results_table,
    )
    atexit.register(_close_worker)


def _run_case(
    case: int,
    comsol_params: dict[str, str],
    auxiliary_sweep: Mapping[str, Sequence[float] | np.ndarray] | None,
) -> dict[str, Any]:
    """Run one geometry case using the worker-local runner.

    Solver failures are captured rather than raised, so one bad case never
    tears down the pool.

    Parameters
    ----------
    case : int
        Case id taken from the geometry sweep.
    comsol_params : dict[str, str]
        COMSOL parameter name to value string, units included.
    auxiliary_sweep : Mapping[str, Sequence[float] or numpy.ndarray], optional
        Auxiliary Sweep inputs for this case, None to disable the sweep.

    Returns
    -------
    dict[str, Any]
        case, response (None when the case failed), status, elapsed_s,
        worker_pid, peak_rss_mb and error.
    """
    started = time.perf_counter()

    try:
        if _worker_runner is None:
            raise RuntimeError("COMSOL worker is not initialized.")

        response = _worker_runner.run(
            comsol_params,
            auxiliary_sweep=auxiliary_sweep,
        )
        status, error = "ok", ""

    except Exception as e:
        response = None
        status, error = "failed", f"{type(e).__name__}: {e}"

    return {
        "case": case,
        "response": response,
        "status": status,
        "elapsed_s": time.perf_counter() - started,
        "worker_pid": os.getpid(),
        "peak_rss_mb": _peak_rss_mb(),
        "error": error,
    }


class ComsolDatasetBuilder:
    """Run a geometry sweep with independent COMSOL worker processes.

    Output layout::

        output_dir/
        |-- simulation_log.csv    one row per case: geometry, status, timing
        |-- data/case{i}.csv      raw response table per case
        `-- images/case{i}.npy    rendered unit-cell mask per case

    Each worker owns one COMSOL client (one license seat) running
    cores_per_client solver threads, so keep workers * cores_per_client at
    or below the physical core count and within your license limits. The
    log records worker_pid and peak_rss_mb per case; group by worker_pid
    and take the maximum to size the next sweep against available memory.

    Parameters
    ----------
    model_path : str or pathlib.Path
        COMSOL .mph model file, loaded once per worker.
    geometry_sweep : pandas.DataFrame
        One row per case. "case" holds the integer case id, and every other
        column is numeric and named after a COMSOL parameter. When
        render_pixels is set, the cavity columns listed in _CAVITY_CASTS
        must be present as well.
    auxiliary_sweep : pandas.DataFrame, optional
        Compact table with one row per case. "case" joins the row to the
        geometry sweep, and every other column is a COMSOL Auxiliary Sweep
        input whose cell holds an ordered one-dimensional sequence. Cells
        may also be JSON array strings, for example "[0.0, 0.5, 1.0]", so a
        table read straight from a CSV works unchanged. Cases without a row
        run with the Auxiliary Sweep disabled.
    output_dir : str or pathlib.Path
        Root of the output layout above.
    units : str, sequence, or mapping
        COMSOL unit string(s) appended to each value: one string for all
        columns, a sequence in column order, or a mapping by column name.
    workers, cores_per_client : int
        Worker processes, and solver threads per COMSOL client.
    tags : ComsolTags
        COMSOL node tags forwarded to each worker's runner.
    rebuild_geometry, rebuild_mesh, clear_results_table : bool
        Forwarded to ComsolRunner.
    render_pixels : int, optional
        Side length of the rendered unit-cell mask. Rendering runs in the
        parent process before any case is dispatched; leave as None to skip
        it entirely.

    Raises
    ------
    FileNotFoundError
        If model_path does not point at a file.
    ValueError
        If geometry_sweep is empty, has duplicate or non-unique case ids,
        lacks a "case" column, lacks the cavity columns while rendering is
        enabled, or if workers, cores_per_client or render_pixels are not
        positive.
    TypeError
        If geometry_sweep column names are not strings or its parameter
        columns are not numeric.
    """

    def __init__(self,
                 model_path: str | pathlib.Path,
                 geometry_sweep: pd.DataFrame,
                 auxiliary_sweep: pd.DataFrame | None = None,
                 output_dir: str | pathlib.Path = "output",
                 units: str | Sequence[str] | Mapping[str, str] = "[mm]",
                 *,
                 workers: int = 1,
                 cores_per_client: int = 1,
                 tags: ComsolTags = ComsolTags(),
                 rebuild_geometry: bool = True,
                 rebuild_mesh: bool = True,
                 clear_results_table: bool = True,
                 render_pixels: int | None = None,
    )->None:
        self.model_path: pathlib.Path = pathlib.Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"COMSOL model not found: {self.model_path}")

        if geometry_sweep.empty:
            raise ValueError("geometry_sweep must contain at least one case.")
        if geometry_sweep.columns.has_duplicates:
            raise ValueError("geometry_sweep column names must be unique.")
        if not all(isinstance(column, str) for column in geometry_sweep.columns):
            raise TypeError("geometry_sweep column names must be strings.")
        if "case" not in geometry_sweep.columns:
            raise ValueError("geometry_sweep must contain a 'case' column.")

        self.geometry_sweep = geometry_sweep.reset_index(drop=True).copy()
        self.geometry_sweep["case"] = self.geometry_sweep["case"].astype(int)
        if self.geometry_sweep["case"].duplicated().any():
            raise ValueError("geometry_sweep case ids must be unique.")

        # Every column except the case id is a COMSOL parameter
        self.param_names: list[str] = [
            column
            for column in self.geometry_sweep.columns.astype(str).tolist()
            if column != "case"
        ]
        non_numeric = [
            column
            for column in self.param_names
            if not pd.api.types.is_numeric_dtype(self.geometry_sweep[column])
        ]
        if non_numeric:
            raise TypeError(f"geometry_sweep columns must be numeric: {non_numeric}.")
        if workers < 1 or cores_per_client < 1:
            raise ValueError("workers and cores_per_client must be positive.")
        if render_pixels is not None and render_pixels < 1:
            raise ValueError("render_pixels must be positive.")

        available_cores = _physical_cores()
        if workers * cores_per_client > available_cores:
            log.warning(
                "Requested %d solver threads (%d workers x %d cores) but only "
                "%d physical cores are available; reduce workers or "
                "cores_per_client to avoid oversubscription.",
                workers * cores_per_client, workers, cores_per_client,
                available_cores,
            )

        self.units = self._normalise_units(units)
        self.auxiliary_sweep = self._normalise_auxiliary_sweep(auxiliary_sweep)
        self.auxiliary_columns = self.auxiliary_sweep.columns.tolist()
        self.workers = workers
        self.cores_per_client = cores_per_client
        self.tags = tags
        self.rebuild_geometry = rebuild_geometry
        self.rebuild_mesh = rebuild_mesh
        self.clear_results_table = clear_results_table
        self.render_pixels = render_pixels

        if self.render_pixels is not None:
            self._validate_render_columns()

        root = pathlib.Path(output_dir)
        self.data_dir = root / "data"
        self.images_dir = root / "images"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.render_pixels is not None:
            self.images_dir.mkdir(parents=True, exist_ok=True)

        self.log_cols = [
            "case", *self.param_names,
            "status", "elapsed_s", "worker_pid", "peak_rss_mb", "error",
        ]
        self.log_path = root / "simulation_log.csv"

    def run(self) -> pd.DataFrame:
        """Run every case, persist results, and return the simulation log.

        Unit cells are rendered first, in this process, so the geometry
        images exist whether the solution later succeeds. Failed cases
        are logged and leave no response file; a broken worker pool marks
        its unfinished cases as failed instead of raising.

        Returns
        -------
        pandas.DataFrame
            Simulation log sorted by case: case, the geometry columns,
            status, elapsed_s, worker_pid, peak_rss_mb and error.
        """
        self._init_log()
        tasks = self._make_tasks()
        n_total = len(tasks)
        n_done = 0
        context = multiprocessing.get_context("spawn")

        # Render in the parent, before any worker is handed a case
        if self.render_pixels is not None:
            self._render_cases(tasks)

        init_args = (
            str(self.model_path),
            self.cores_per_client,
            self.tags,
            self.rebuild_geometry,
            self.rebuild_mesh,
            self.clear_results_table,
        )

        with ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=init_args,
        ) as executor:
            futures = {
                executor.submit(
                    _run_case,
                    task["case"],
                    task["comsol_params"],
                    task["auxiliary_sweep"],
                ): task
                for task in tasks
            }

            for future in as_completed(futures):
                task = futures[future]

                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "case": task["case"],
                        "response": None,
                        "status": "failed",
                        "elapsed_s": 0.0,
                        "worker_pid": None,
                        "peak_rss_mb": None,
                        "error": f"{type(e).__name__}: {e}",
                    }

                n_done += 1
                log.info(
                    "Case %d %s in %.1fs (%d/%d)",
                    result["case"], result["status"], result["elapsed_s"],
                    n_done, n_total,
                )
                self._save_result(result, task["geometry_values"])

        return self._finalise_log()

    def _normalise_units(self,
                         units: str | Sequence[str] | Mapping[str, str],
    ) -> list[str]:
        """Expand the units argument into one unit string per parameter.

        Parameters
        ----------
        units : str, sequence, or mapping
            One string for all parameters, a sequence in column order, or a
            mapping by column name.

        Returns
        -------
        list[str]
            Unit strings aligned with param_names.

        Raises
        ------
        ValueError
            If a mapping does not cover exactly the parameter columns, or a
            sequence has the wrong length.
        """
        columns = self.param_names

        if isinstance(units, str):
            return [units] * len(columns)

        if isinstance(units, Mapping):
            missing = [column for column in columns if column not in units]
            extra = [name for name in units if name not in columns]
            if missing or extra:
                raise ValueError(
                    f"Unit names do not match geometry columns; "
                    f"missing={missing}, extra={extra}."
                )
            return [units[column] for column in columns]

        unit_list = list(units)
        if len(unit_list) != len(columns):
            raise ValueError(
                f"units length {len(unit_list)} does not match "
                f"number of parameters {len(columns)}."
            )
        return unit_list

    def _normalise_auxiliary_sweep(self,
                                   auxiliary_sweep: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Index the compact per-case Auxiliary Sweep table by case.

        Parameters
        ----------
        auxiliary_sweep : pandas.DataFrame, optional
            Table carrying a "case" column and one column per sweep input.

        Returns
        -------
        pandas.DataFrame
            Sweep values as numeric tuples, indexed by case; empty when no
            table was supplied.

        Raises
        ------
        ValueError
            If the table lacks a "case" column or repeats a case id.
        """
        if auxiliary_sweep is None:
            return pd.DataFrame(index=pd.Index([], name="case"))

        table = auxiliary_sweep.copy()

        if "case" not in table.columns:
            raise ValueError("auxiliary_sweep must contain a 'case' column.")

        table["case"] = table["case"].astype(int)
        table = table.set_index("case").sort_index()
        if not table.index.is_unique:
            raise ValueError("auxiliary_sweep must contain one row per case.")

        return table.map(self._parse_sweep_values)

    @staticmethod
    def _parse_sweep_values(
            value: Any,
    ) -> tuple[float, ...]:
        """Parse one sweep cell into a numeric tuple.

        Parameters
        ----------
        value : Any
            Any iterable of numbers, or a JSON array string as produced by
            reading the table back from a CSV.

        Returns
        -------
        tuple[float, ...]
            Sweep values in their original order.
        """
        if isinstance(value, str):
            value = json.loads(value)
        return tuple(float(item) for item in value)

    def _validate_render_columns(self) -> None:
        """Confirm the geometry sweep can be turned into a CavityGeometry.

        Raises
        ------
        ValueError
            If a cavity column without a default is missing from the sweep.
        """
        required = [
            field.name
            for field in fields(CavityGeometry)
            if field.default is MISSING
        ]
        missing = [name for name in required if name not in self.param_names]
        if missing:
            raise ValueError(
                f"render_pixels is set but geometry_sweep lacks the cavity "
                f"columns: {', '.join(missing)}."
            )

    def _make_tasks(self) -> list[dict[str, Any]]:
        """Pair every geometry row with its Auxiliary Sweep values.

        Returns
        -------
        list[dict[str, Any]]
            One task per case: case, geometry_values, comsol_params and
            auxiliary_sweep.
        """
        tasks: list[dict[str, Any]] = []

        for _, row in self.geometry_sweep.iterrows():
            case = int(row["case"])
            geometry = row.drop("case")

            if case not in self.auxiliary_sweep.index:
                auxiliary = None
            else:
                case_sweep = self.auxiliary_sweep.loc[case]
                auxiliary = {
                    name: np.asarray(case_sweep[name], dtype=float)
                    for name in self.auxiliary_columns
                }

            tasks.append({
                "case": case,
                "geometry_values": geometry.to_dict(),
                "comsol_params": self._geometry_to_comsol_params(geometry),
                "auxiliary_sweep": auxiliary,
            })

        return tasks

    def _geometry_to_comsol_params(self, geometry: pd.Series) -> dict[str, str]:
        """Format one geometry row as COMSOL parameter value strings.

        Parameters
        ----------
        geometry : pandas.Series
            Parameter values of one case, without the case id.

        Returns
        -------
        dict[str, str]
            Parameter name to value string with its unit appended.
        """
        return {
            name: f"{float(value):.10g} {unit}".strip()
            for name, value, unit in zip(self.param_names, geometry.to_numpy(), self.units)
        }

    def _render_cases(self, tasks: list[dict[str, Any]]) -> None:
        """Rasterize every case and store the masks beside the responses.

        A geometry that cannot be rendered is reported and skipped, so an
        unrenderable row never blocks the solver sweep.

        Parameters
        ----------
        tasks : list[dict[str, Any]]
            Tasks produced by _make_tasks.
        """
        for task in tasks:
            case = task["case"]
            try:
                cavity = self._to_cavity_geometry(task["geometry_values"])
                mask = render_cell(cavity, n_pixels=self.render_pixels)
                np.save(self.images_dir / f"case{case}.npy", mask)

            except Exception as e:
                log.warning("Render failed for case %d: %s", case, e)

        log.info("Geometry images -> %s", self.images_dir)

    @staticmethod
    def _to_cavity_geometry(geometry_values: Mapping[str, Any]) -> CavityGeometry:
        """Build a CavityGeometry from the cavity columns of one case.

        Parameters
        ----------
        geometry_values : Mapping[str, Any]
            Parameter values of one case, cavity columns included.

        Returns
        -------
        CavityGeometry
            Shape parameters accepted by render_cell.
        """
        return CavityGeometry(**{
            name: caster(geometry_values[name])
            for name, caster in _CAVITY_CASTS.items()
            if name in geometry_values
        })

    def _init_log(self) -> None:
        """Create an empty simulation log with its header row."""
        pd.DataFrame(columns=self.log_cols).to_csv(self.log_path, index=False)
        log.info("Simulation log -> %s", self.log_path)

    def _save_result(self,
                     result: dict[str, Any],
                     geometry_values: Mapping[str, Any],
    ) -> None:
        """Write one case's response table and append its log row.

        Parameters
        ----------
        result : dict[str, Any]
            Result dictionary produced by _run_case.
        geometry_values : Mapping[str, Any]
            Parameter values of this case, held in the parent process.
        """
        case = result["case"]
        response = result["response"]

        if response is not None:
            response.to_csv(self.data_dir / f"case{case}.csv", index=False)

        self._append_log(result, geometry_values)

    def _append_log(self,
                    result: dict[str, Any],
                    geometry_values: Mapping[str, Any],
    ) -> None:
        """Append one row to the simulation log.

        Parameters
        ----------
        result : dict[str, Any]
            Result dictionary produced by _run_case.
        geometry_values : Mapping[str, Any]
            Parameter values of this case, held in the parent process.
        """
        row = {"case": result["case"], **geometry_values}
        row.update({
            "status": result["status"],
            "elapsed_s": round(result["elapsed_s"], 3),
            "worker_pid": result["worker_pid"],
            "peak_rss_mb": result["peak_rss_mb"],
            "error": result["error"],
        })
        pd.DataFrame([row], columns=self.log_cols).to_csv(
            self.log_path,
            mode="a",
            index=False,
            header=False,
        )

    @staticmethod
    def _read_csv(path: pathlib.Path) -> pd.DataFrame:
        """Read a CSV into a DataFrame.

        Parameters
        ----------
        path : pathlib.Path
            File to read.

        Returns
        -------
        pandas.DataFrame
            Parsed table.
        """
        read_csv: Any = pd.read_csv
        return read_csv(path)

    def _finalise_log(self) -> pd.DataFrame:
        """Sort the simulation log by case and rewrite it in place.

        Rows are appended in completion order, which is not case order once
        more than one worker is running.

        Returns
        -------
        pandas.DataFrame
            Simulation log sorted by case.
        """
        simulation_log = self._read_csv(self.log_path).sort_values("case")
        simulation_log.to_csv(self.log_path, index=False)

        # High-water mark across workers, for sizing the next sweep
        peaks = simulation_log["peak_rss_mb"].dropna()
        if not peaks.empty:
            log.info(
                "Peak worker memory %.1f MiB across %d workers",
                peaks.max(),
                simulation_log["worker_pid"].dropna().nunique(),
            )

        return simulation_log.reset_index(drop=True)
