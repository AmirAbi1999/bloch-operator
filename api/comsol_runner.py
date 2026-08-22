"""Process-local driver for one COMSOL model via the mph bridge.

A ComsolRunner wraps a loaded model and executes the evaluation pipeline

    set parameters -> configure auxiliary sweep -> rebuild geometry and
    mesh -> solve -> evaluate derived values -> read results table

for one parameter set at a time. Every step that touches the Java side is
wrapped, so any failure surfaces as a ComsolRunError naming the step and
the parameter set that produced it.

The runner is strictly process-local: the Java bridge behind mph cannot be
pickled or shared, so each worker process must create, and close, its own
runner.

This module contains:
    - SweepValues
    - ComsolTags
    - ComsolRunError
    - ComsolRunner
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import mph

log = logging.getLogger(__name__)

# Auxiliary Sweep inputs: one ordered value list per sweep input name
SweepValues = Mapping[str, Sequence[float] | np.ndarray]


@dataclass(frozen=True)
class ComsolTags:
    """COMSOL model node tags, overridden when a model uses different names.

    Attributes
    ----------
    comp : str
        Tag of the component holding the geometry and mesh.
    geom : str
        Tag of the geometry sequence.
    mesh : str
        Tag of the mesh sequence.
    study : str
        Tag of the study that drives the solver.
    study_step : str
        Tag of the study step carrying the auxiliary sweep.
    derived_values : str
        Tag of the derived-values node evaluated after the solution.
    table : str
        Tag of the results table the derived values are written to.
    """

    comp: str = "comp1"
    geom: str = "geom1"
    mesh: str = "mesh1"
    study: str = "std1"
    study_step: str = "eig"
    derived_values: str = "gev1"
    table: str = "tbl1"


class ComsolRunError(RuntimeError):
    """Raised when a COMSOL pipeline step fails.

    Parameters
    ----------
    step : str
        Pipeline step that failed, for example "Mesh Run".
    message : str
        Description of the failure, usually the message of the underlying
        Java exception.
    params : Mapping[str, str], optional
        Parameter set in effect when the step failed, retained for
        reporting.
    """

    def __init__(
        self,
        step: str,
        message: str,
        params: Mapping[str, str] | None = None,
    ) -> None:
        self.step = step
        self.params = dict(params) if params is not None else {}
        super().__init__(f"{step}: failed {message}")


class ComsolRunner:
    """Execute a full COMSOL evaluation pipeline for a given parameter set.

    A runner belongs to the process in which it was created and must not be
    shared between threads or processes.

    Parameters
    ----------
    mph_model : mph.Model
        Loaded mph Model object, required.
    mph_client : mph.Client, optional
        Client owning mph_model, retained so close can release the model
        and disconnect.
    tags : ComsolTags
        COMSOL node tag names, use the defaults or supply custom ones.
    rebuild_geometry : bool
        Re-run the geometry sequence before each solve.
    rebuild_mesh : bool
        Re-run the mesher before each solve.
    clear_results_table : bool
        Wipe the results table before writing new derived values.

    Raises
    ------
    ValueError
        If mph_model is None.
    ComsolRunError
        If any configured node tag is missing from the model.
    """

    def __init__(
        self,
        mph_model: mph.Model,
        mph_client: mph.Client | None = None,
        tags: ComsolTags = ComsolTags(),
        *,
        rebuild_geometry: bool = True,
        rebuild_mesh: bool = True,
        clear_results_table: bool = True,
    ) -> None:
        if mph_model is None:
            raise ValueError("mph_model must be provided (loaded COMSOL model).")

        self.client: mph.Client | None = mph_client
        self.model = mph_model
        self.java_model = mph_model.java
        self.tags = tags
        self.rebuild_geometry = rebuild_geometry
        self.rebuild_mesh = rebuild_mesh
        self.clear_results_table = clear_results_table
        self._parameter_names: set[str] = set()
        self._owner_pid = os.getpid()
        self._closed = False

        self._validate_model()

    def __call__(
        self,
        params: Mapping[str, str],
        auxiliary_sweep: SweepValues | None = None,
    ) -> pd.DataFrame:
        """Alias for run.

        Parameters
        ----------
        params : Mapping[str, str]
            COMSOL parameter name to value string.
        auxiliary_sweep : Mapping[str, Sequence[float] or numpy.ndarray], optional
            Auxiliary Sweep input names mapped to their ordered value lists.

        Returns
        -------
        pandas.DataFrame
            Derived-values table extracted from the COMSOL results node.
        """
        return self.run(params, auxiliary_sweep=auxiliary_sweep)

    def run(
        self,
        params: Mapping[str, str],
        *,
        auxiliary_sweep: SweepValues | None = None,
    ) -> pd.DataFrame:
        """Run the full pipeline and return the results table.

        Parameters
        ----------
        params : Mapping[str, str]
            COMSOL parameter name to value string, for example {"d": "5[mm]"}.
        auxiliary_sweep : Mapping[str, Sequence[float] or numpy.ndarray], optional
            Auxiliary Sweep input names mapped to their ordered value lists.
            All lists must have the same length because the runner uses
            specified combinations. These are study inputs, not entries in
            "params". Omit to switch the sweep off.

        Returns
        -------
        pandas.DataFrame
            Derived-values table extracted from the COMSOL results node.

        Raises
        ------
        RuntimeError
            If the runner is closed or used outside its owning process.
        ComsolRunError
            If any pipeline step fails.
        """
        self._check_state()
        log.debug("COMSOL run starting: %s", dict(params))

        # Push the requested inputs into the model
        self._set_parameters(params)
        self._set_auxiliary_sweep(auxiliary_sweep)

        # Rebuild the discretization, then solve
        self._rebuild(params)
        self._solve(params)

        # Evaluate the derived values and collect them
        self._run_derived_values(params)

        return self._read_table(params)

    def close(self, disconnect: bool = False) -> None:
        """Remove the model and optionally disconnect its COMSOL client.

        Parameters
        ----------
        disconnect : bool
            Also disconnect the client once the model has been removed.

        Raises
        ------
        RuntimeError
            If close is called outside the owning process.
        ComsolRunError
            If the model cannot be removed or the client cannot be
            disconnected.
        """
        if self._closed:
            return

        if os.getpid() != self._owner_pid:
            raise RuntimeError(
                "ComsolRunner must be closed in the process where it was created."
            )

        client = self.client
        try:
            if client is not None:
                client.remove(self.model)

                if disconnect and getattr(client, "port", None):
                    client.disconnect()

        except Exception as e:
            raise ComsolRunError(step="Cleanup", message=str(e)) from e

        finally:
            self._closed = True

    def _check_state(self) -> None:
        """Confirm the runner is open and owned by the calling process.

        Raises
        ------
        RuntimeError
            If the runner is closed or belongs to another process.
        """
        if self._closed:
            raise RuntimeError("ComsolRunner has been closed.")

        if os.getpid() != self._owner_pid:
            raise RuntimeError(
                "ComsolRunner must be used in the process where it was created."
            )

    def _validate_model(self) -> None:
        """Check parameter and node tags before changing the model.

        Raises
        ------
        ComsolRunError
            If a configured tag does not resolve to a node in the model.
        """
        node_type, tag = "component", self.tags.comp
        try:
            self._parameter_names = set(self.model.parameters(evaluate=False))
            component = self.java_model.component(tag)

            if self.rebuild_geometry:
                node_type, tag = "geometry", self.tags.geom
                component.geom(tag)

            if self.rebuild_mesh:
                node_type, tag = "mesh", self.tags.mesh
                component.mesh(tag)

            node_type, tag = "study", self.tags.study
            study = self.java_model.study(tag)

            node_type, tag = "study step", self.tags.study_step
            study.feature(tag)

            node_type, tag = "derived values", self.tags.derived_values
            self.java_model.result().numerical(tag)

            node_type, tag = "table", self.tags.table
            self.java_model.result().table(tag)

        except Exception as e:
            raise ComsolRunError(
                step="Model Validation",
                message=f"{node_type} tag '{tag}': {e}",
            ) from e

    def _set_parameters(self, params: Mapping[str, str]) -> None:
        """Push each parameter value into the COMSOL model's parameter node.

        Parameters
        ----------
        params : Mapping[str, str]
            COMSOL parameter name to value string.

        Raises
        ------
        ComsolRunError
            If a name is not defined in the model, if a value is not a
            string, or if the assignment fails on the Java side.
        """
        unknown = sorted(set(params).difference(self._parameter_names))
        if unknown:
            raise ComsolRunError(
                step="Parameter Validation",
                message=f"parameters are not defined: {', '.join(unknown)}",
                params=params,
            )

        non_strings = sorted(
            key for key, value in params.items() if not isinstance(value, str)
        )
        if non_strings:
            raise ComsolRunError(
                step="Parameter Validation",
                message=f"parameter values must be strings: {', '.join(non_strings)}",
                params=params,
            )

        try:
            for key, value in params.items():
                self.java_model.param().set(key, value)

        except Exception as e:
            raise ComsolRunError(
                step="Parameter Set",
                message=str(e),
                params=params,
            ) from e

    def _set_auxiliary_sweep(
        self,
        auxiliary_sweep: SweepValues | None = None,
    ) -> None:
        """Configure specified combinations for the Auxiliary Sweep.

        Parameters
        ----------
        auxiliary_sweep : Mapping[str, Sequence[float] or numpy.ndarray], optional
            Sweep input names mapped to their ordered value lists. An empty
            or omitted mapping switches the sweep off.

        Raises
        ------
        ComsolRunError
            If a name is empty, if a value list is not a nonempty
            one-dimensional array of finite numbers, if the lists differ in
            length, or if the assignment fails on the Java side.
        """
        try:
            step = (
                self.java_model
                .study(self.tags.study)
                .feature(self.tags.study_step)
            )

            if not auxiliary_sweep:
                step.set("useparam", "off")
                return

            names = [name.strip() for name in auxiliary_sweep]
            arrays = [
                np.asarray(values, dtype=float)
                for values in auxiliary_sweep.values()
            ]

            if any(not name for name in names):
                raise ValueError("Sweep parameter names cannot be empty.")

            if any(values.ndim != 1 or values.size == 0 for values in arrays):
                raise ValueError("Sweep values must be nonempty 1D arrays.")

            if any(not np.all(np.isfinite(values)) for values in arrays):
                raise ValueError("Sweep values must be finite.")

            if len({values.size for values in arrays}) != 1:
                raise ValueError("Sweep value lists must have equal lengths.")

            # COMSOL expects one space-separated value string per sweep name
            value_strings = [
                " ".join(f"{float(value):.16g}" for value in values)
                for values in arrays
            ]

            step.set("useparam", "on")
            step.set("sweeptype", "sparse")
            step.set("pname", names)
            step.set("plistarr", value_strings)
            step.set("punit", [""] * len(names))

        except Exception as e:
            raise ComsolRunError(
                step="Auxiliary Sweep Configuration",
                message=str(e),
            ) from e

    def _rebuild(self, params: Mapping[str, str]) -> None:
        """Re-run geometry and mesh sequences as configured.

        Parameters
        ----------
        params : Mapping[str, str]
            Parameter set in effect, attached to any raised error.

        Raises
        ------
        ComsolRunError
            If the geometry or the mesh sequence fails.
        """
        if self.rebuild_geometry:
            try:
                self.java_model.component(self.tags.comp).geom(self.tags.geom).run()

            except Exception as e:
                raise ComsolRunError(
                    step="Geometry Run",
                    message=str(e),
                    params=params,
                ) from e

        if self.rebuild_mesh:
            try:
                self.java_model.component(self.tags.comp).mesh(self.tags.mesh).run()

            except Exception as e:
                raise ComsolRunError(
                    step="Mesh Run",
                    message=str(e),
                    params=params,
                ) from e

    def _solve(self, params: Mapping[str, str]) -> None:
        """Execute the study, which triggers the solver.

        Parameters
        ----------
        params : Mapping[str, str]
            Parameter set in effect, attached to any raised error.

        Raises
        ------
        ComsolRunError
            If the study fails to run.
        """
        try:
            self.java_model.study(self.tags.study).run()

        except Exception as e:
            raise ComsolRunError(
                step="Study Run",
                message=str(e),
                params=params,
            ) from e

    def _run_derived_values(self, params: Mapping[str, str]) -> None:
        """Evaluate the derived-values node and write results to the table.

        Parameters
        ----------
        params : Mapping[str, str]
            Parameter set in effect, attached to any raised error.

        Raises
        ------
        ComsolRunError
            If the table cannot be cleared or the evaluation fails.
        """
        try:
            derived_values_node = self.java_model.result().numerical(
                self.tags.derived_values
            )
            derived_values_node.set("table", self.tags.table)

            if self.clear_results_table:
                results_table = self.java_model.result().table(self.tags.table)
                if not hasattr(results_table, "clearTableData"):
                    raise AttributeError(
                        "Results table does not support clearTableData()."
                    )
                results_table.clearTableData()

            derived_values_node.setResult()

        except Exception as e:
            raise ComsolRunError(
                step="Derived Values",
                message=str(e),
                params=params,
            ) from e

    def _read_table(self, params: Mapping[str, str]) -> pd.DataFrame:
        """Read the COMSOL results table into a DataFrame.

        Parameters
        ----------
        params : Mapping[str, str]
            Parameter set in effect, attached to any raised error.

        Returns
        -------
        pandas.DataFrame
            Table contents, with placeholder column names when the headers
            cannot be matched to the data.

        Raises
        ------
        ComsolRunError
            If the table cannot be read or does not hold a two-dimensional
            block of numbers.
        """
        try:
            results_table = self.java_model.result().table(self.tags.table)
            data = np.asarray(results_table.getReal(), dtype=float)
            headers = []

            # Java headers arrive as strings or as nested character sequences
            if hasattr(results_table, "getColumnHeaders"):
                raw_headers = list(results_table.getColumnHeaders())

                for header in raw_headers:
                    if isinstance(header, (tuple, list)):
                        header = "".join(map(str, header)).strip()
                    else:
                        header = str(header).strip()
                    headers.append(header)

            # Restore the row-by-column layout collapsed by degenerate results
            if data.size == 0:
                data = np.empty((0, len(headers)), dtype=float)
            elif data.ndim == 1:
                data = data.reshape(-1, 1) if len(headers) == 1 else data.reshape(1, -1)
            elif data.ndim != 2:
                raise ValueError("Result table must be one- or two-dimensional.")

            if len(headers) != data.shape[1]:
                headers = [f"col{i}" for i in range(data.shape[1])]

            return pd.DataFrame(data, columns=headers)

        except Exception as e:
            raise ComsolRunError(
                step="Table Read",
                message=str(e),
                params=params,
            ) from e
