"""Dataset build driving one geometry table and its Auxiliary Sweep.

The two input tables and everything the build derives from them share
one directory, so a dataset can be inspected, copied or deleted on its
own::

    dataset/val/
    |-- geometry.csv          one row per case
    |-- sweep.csv             that case's wave-pairs, as JSON arrays
    |-- simulation_log.csv    one row per case: geometry, status, timing
    |-- data/case{i}.csv      raw response table per case
    `-- images/case{i}.npy    rendered unit-cell mask per case

Point DATASET_DIR at another directory to build a different dataset.

This module contains:
    - preflight
    - summarize
    - main
"""

from __future__ import annotations

import json
import logging
import pathlib

import pandas as pd

from api import ComsolDatasetBuilder, ComsolRunner, ComsolTags

log = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parent
MODEL_PATH = ROOT / "2DIrregularSolid.mph"
DATASET_DIR = ROOT / "dataset" / "val"
GEOMETRY_PATH = DATASET_DIR / "geometry.csv"
SWEEP_PATH = DATASET_DIR / "sweep.csv"

TAGS = ComsolTags()

RENDER_PIXELS = 128
WORKERS = 16
CORES_PER_CLIENT = 1


def preflight(
    geometry_sweep: pd.DataFrame,
    auxiliary_sweep: pd.DataFrame,
) -> None:
    """Check the two tables agree, then validate the model once.

    Parameters
    ----------
    geometry_sweep : pandas.DataFrame
        Table whose columns must exist as COMSOL parameters.
    auxiliary_sweep : pandas.DataFrame
        Table whose columns must exist as study-step sweep inputs, and
        which must carry a row for every geometry case.

    Raises
    ------
    ValueError
        If a geometry case has no sweep row, or a geometry column has no
        matching COMSOL parameter.
    ComsolRunError
        If any configured node tag is missing from the model.
    """
    cases = geometry_sweep["case"]
    unswept = sorted(cases[~cases.isin(auxiliary_sweep["case"])])
    if unswept:
        raise ValueError(
            f"{len(unswept)} geometry cases have no Auxiliary Sweep row: "
            f"{unswept[:5]}."
        )

    # Sweep rows outside the geometry table are never solved
    swept = auxiliary_sweep.loc[auxiliary_sweep["case"].isin(cases), "kx"]
    pairs = [len(json.loads(cell)) for cell in swept]
    log.info(
        "%d cases x %d-%d wave-pairs: %d solves",
        len(cases), min(pairs), max(pairs), sum(pairs),
    )

    import mph

    log.info("Preflight: starting one COMSOL session")
    client = mph.start(cores=1)
    model = None

    try:
        model = client.load(str(MODEL_PATH))

        # Raises ComsolRunError if any ComsolTags entry is missing
        ComsolRunner(mph_model=model, mph_client=client, tags=TAGS)
        parameters = set(model.parameters(evaluate=False))

        undefined = [
            column
            for column in geometry_sweep.columns
            if column != "case" and column not in parameters
        ]
        if undefined:
            raise ValueError(
                f"The model defines no parameter for geometry columns: "
                f"{', '.join(undefined)}; it defines {sorted(parameters)}."
            )

        log.info(
            "Preflight passed: %d model parameters, sweep inputs %s",
            len(parameters),
            [column for column in auxiliary_sweep.columns if column != "case"],
        )

    finally:
        if model is not None:
            client.remove(model)
        client.disconnect()


def summarize(
    simulation_log: pd.DataFrame,
    builder: ComsolDatasetBuilder,
) -> None:
    """Report status counts, timings, memory and file counts.

    Parameters
    ----------
    simulation_log : pandas.DataFrame
        Simulation log returned by ComsolDatasetBuilder.run.
    builder : ComsolDatasetBuilder
        Builder that produced the log, read for its output directories.
    """
    elapsed = simulation_log["elapsed_s"]
    n_ok = int((simulation_log["status"] == "ok").sum())

    # peak_rss_mb is a worker high-water mark, so reduce it per worker
    per_worker = (
        simulation_log
        .dropna(subset=["worker_pid"])
        .groupby("worker_pid")["peak_rss_mb"]
        .max()
    )

    log.info("Status: %s", simulation_log["status"].value_counts().to_dict())
    log.info(
        "Case time: mean %.1fs, max %.1fs, %.1f solver-hours total",
        elapsed.mean(), elapsed.max(), elapsed.sum() / 3600.0,
    )
    log.info(
        "Peak memory: %.1f MiB across %d workers",
        per_worker.sum(), len(per_worker),
    )
    log.info(
        "Files: %d of %d response tables, %d of %d geometry images",
        len(list(builder.data_dir.glob("case*.csv"))), n_ok,
        len(list(builder.images_dir.glob("case*.npy"))), len(simulation_log),
    )


def main() -> None:
    """Load both tables, validate them, run the sweep, and report."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    geometry_sweep = pd.read_csv(GEOMETRY_PATH)
    auxiliary_sweep = pd.read_csv(SWEEP_PATH)
    log.info("Building %s", DATASET_DIR)

    preflight(geometry_sweep, auxiliary_sweep)

    builder = ComsolDatasetBuilder(
        MODEL_PATH,
        geometry_sweep,
        auxiliary_sweep,
        output_dir=DATASET_DIR,
        units="",
        workers=WORKERS,
        cores_per_client=CORES_PER_CLIENT,
        tags=TAGS,
        render_pixels=RENDER_PIXELS,
    )

    summarize(builder.run(), builder)


if __name__ == "__main__":
    main()
