"""Typed artifact references for one completed stage run.

`cfd_get_run` hands an agent field names that are exactly the *next* stage's
parameter names, so the chain is copied rather than guessed. Two facts are
needed for that, and they come from different places:

* **Which parameters exist** is derived from the stage registry, which is the
  only place a stage's inputs are declared. `CONSUMER_PARAMETERS` below is
  computed from `StageInputBinding.parameter_name`, so a stage cannot gain an
  input this layer has never heard of.
* **What a stage's own files are called** is *not* in the registry.
  `StageInputBinding.staged_path` is the name a file is mounted as for the
  *consumer* - the flow stage reads its mesh as `mesh.npz` no matter what the
  import stage called it - while the import entrypoint writes
  `f"{msh_path.stem}_imported_mesh.npz"`
  (`experiments/gmsh/run_import_gmsh_mesh.py:287`). Producer names therefore
  live in `STAGE_PRODUCTS`, read off the entrypoints, and `_validate_products`
  plus `mcp-server/tests/test_registry_drift.py` keep that table honest
  against the registry.

The table is also what scopes `outputs` to the stage that ran. All five
entrypoints write a `summary.json`; only the flow stage's is a
`flow_summary_path`. A single global parameter table would answer an import
run with `flow_summary_path = <import-run>/summary.json`, a value that
validates as a key and then fails inside `experiments/gmsh/_flow_coupling.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from posixpath import basename, splitext
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from backend.app.stage_registry import STAGE_REGISTRY


def _build_consumer_parameters() -> Mapping[str, frozenset[str]]:
    """Every stage input parameter, mapped to the stages that consume it."""

    consumers: dict[str, set[str]] = {}
    for stage_id, definition in STAGE_REGISTRY.items():
        for binding in definition.inputs:
            consumers.setdefault(binding.parameter_name, set()).add(stage_id)
    return MappingProxyType(
        {name: frozenset(stages) for name, stages in consumers.items()}
    )


CONSUMER_PARAMETERS: Mapping[str, frozenset[str]] = _build_consumer_parameters()


def _staged_basename(stage_id: str, parameter_name: str) -> str:
    for binding in STAGE_REGISTRY[stage_id].inputs:
        if binding.parameter_name == parameter_name:
            return basename(binding.staged_path)
    raise ValueError(
        f"Stage {stage_id!r} has no input parameter {parameter_name!r}; "
        "microfluidics_mcp.outputs cannot derive its produced filenames."
    )


# `run_import_gmsh_mesh.py:287` names the mesh archive after the stem of the
# `--msh` file it was handed, and `stage_commands._import_argv` hands it the
# staged input. Deriving the stem instead of writing "input_imported_mesh.npz"
# keeps the two ends together if the registry ever stages the mesh elsewhere.
_IMPORTED_MESH_NPZ = (
    splitext(_staged_basename("stage_import", "mesh_path"))[0] + "_imported_mesh.npz"
)


@dataclass(frozen=True)
class StageProducts:
    """What one stage can hand to a later stage.

    `produced` maps a next-stage parameter to the basename *this* stage's
    entrypoint writes for it. `echoed` names this stage's own input parameters
    that a later stage also needs and that no stage produces - the original
    `.msh` reaches `stage_thermal` this way, because only the agent ever had
    it.
    """

    produced: Mapping[str, str] = field(default_factory=dict)
    echoed: frozenset[str] = frozenset()


# Read off the entrypoints, not the registry. File:line for every name:
#
#   stage_import   run_import_gmsh_mesh.py:287           <stem>_imported_mesh.npz
#   stage_flow     run_gmsh_tetra_flow_debug.py:13035    summary.json
#                  run_gmsh_tetra_flow_debug.py:1779     flow_coupling_metadata.json
#                  run_gmsh_tetra_flow_debug.py:1770     final_corrected_face_flux.npy
#                  run_gmsh_tetra_flow_debug.py:1775     face_to_cells.npy
#                  run_gmsh_tetra_flow_debug.py:1777     cell_volumes.npy
#
# `stage_transport`, `stage_thermal`, and `stage_reactive_transport` are
# terminal: they write summaries,
# regime audits and their own fields (`run_gmsh_tetra_transport_debug.py:5209`,
# `run_gmsh_tetra_thermal_debug.py:949`), but no stage consumes any of it, so
# they legitimately produce nothing for this layer. Their `summary.json` in
# particular is not a `flow_summary_path`.
STAGE_PRODUCTS: Mapping[str, StageProducts] = MappingProxyType(
    {
        "stage_import": StageProducts(
            produced=MappingProxyType({"mesh_npz_path": _IMPORTED_MESH_NPZ}),
            echoed=frozenset({"mesh_path"}),
        ),
        "stage_flow": StageProducts(
            produced=MappingProxyType(
                {
                    "flow_summary_path": "summary.json",
                    "flow_coupling_metadata_path": "flow_coupling_metadata.json",
                    "flow_face_flux_path": "final_corrected_face_flux.npy",
                    "flow_face_to_cells_path": "face_to_cells.npy",
                    "flow_cell_volumes_path": "cell_volumes.npy",
                }
            ),
            echoed=frozenset({"mesh_npz_path"}),
        ),
        "stage_transport": StageProducts(),
        "stage_thermal": StageProducts(),
        "stage_reactive_transport": StageProducts(
            echoed=frozenset({"reactive_case_path"})
        ),
    }
)


def _validate_products() -> None:
    """Fail loudly on an edit to this file the registry cannot support.

    The opposite direction - the registry growing an input no stage produces
    or echoes - is a test
    (`test_registry_drift.test_every_stage_input_can_be_supplied_by_a_stage`),
    so adding a stage does not turn into an ImportError with no test name
    attached to it.
    """

    for stage_id, products in STAGE_PRODUCTS.items():
        if stage_id not in STAGE_REGISTRY:
            raise ValueError(
                f"STAGE_PRODUCTS names {stage_id!r}, which is not a stage in "
                "backend.app.stage_registry.STAGE_REGISTRY."
            )
        overlap = sorted(set(products.produced) & products.echoed)
        if overlap:
            raise ValueError(
                f"Stage {stage_id!r} both produces and echoes {overlap}; one "
                "parameter cannot have two sources."
            )
        own_inputs = {
            binding.parameter_name for binding in STAGE_REGISTRY[stage_id].inputs
        }
        for parameter in sorted(products.echoed):
            if parameter not in own_inputs:
                raise ValueError(
                    f"Stage {stage_id!r} cannot echo {parameter!r}: it is not "
                    "one of that stage's own inputs."
                )
        for parameter in sorted(set(products.produced) | products.echoed):
            if parameter not in CONSUMER_PARAMETERS:
                raise ValueError(
                    f"Stage {stage_id!r} offers {parameter!r}, which no stage "
                    "in STAGE_REGISTRY consumes."
                )


_validate_products()


def supplied_parameters() -> frozenset[str]:
    """Every parameter some stage can put in `outputs`."""

    supplied: set[str] = set()
    for products in STAGE_PRODUCTS.values():
        supplied.update(products.produced)
        supplied.update(products.echoed)
    return frozenset(supplied)


@dataclass(frozen=True)
class StageOutputs:
    """Next-stage parameters resolved from one completed run."""

    outputs: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)
    artifact_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "outputs": dict(self.outputs),
            "missing": list(self.missing),
            "ambiguous": list(self.ambiguous),
            "artifact_count": self.artifact_count,
        }


def resolve_stage_outputs(
    *,
    experiment_id: str | None,
    artifacts: Sequence[str],
    request_parameters: Mapping[str, Any],
) -> StageOutputs:
    """Map one stage's artifacts and echoed inputs onto next-stage parameters.

    Only the parameters `experiment_id` can legitimately supply are considered:
    an artifact of one stage never stands in for a parameter it cannot serve.
    Two artifacts sharing a produced basename are reported as ambiguous rather
    than picked arbitrarily - a silently wrong coupling file yields a plausible
    but wrong field. A run whose stage is not known yet (no result payload)
    resolves nothing rather than guessing.
    """

    products = STAGE_PRODUCTS.get(str(experiment_id or ""))
    if products is None:
        return StageOutputs(artifact_count=len(artifacts))

    by_basename: dict[str, list[str]] = {}
    for key in artifacts:
        by_basename.setdefault(basename(key), []).append(key)

    outputs: dict[str, str] = {}
    missing: list[str] = []
    ambiguous: list[str] = []
    for parameter in sorted(set(products.produced) | products.echoed):
        expected = products.produced.get(parameter)
        if expected is not None:
            candidates = by_basename.get(expected, [])
            if len(candidates) == 1:
                outputs[parameter] = candidates[0]
            elif candidates:
                ambiguous.append(parameter)
            else:
                missing.append(parameter)
            continue
        echoed = request_parameters.get(parameter)
        if isinstance(echoed, str) and echoed:
            outputs[parameter] = echoed
        else:
            missing.append(parameter)

    return StageOutputs(
        outputs=outputs,
        missing=missing,
        ambiguous=ambiguous,
        artifact_count=len(artifacts),
    )
