"""Every filename asserted here is the one an entrypoint really writes.

The names come from `experiments/gmsh/`, not from the registry's staged paths
and not from a fixture's imagination:

* `input_imported_mesh.npz` — `run_import_gmsh_mesh.py:287` writes
  `f"{msh_path.stem}_imported_mesh.npz"`, and `stage_commands._import_argv`
  passes the staged `/job/work/input.msh` as `--msh`, so the stem is `input`.
  The registry's `mesh.npz` is what the *flow* stage mounts that file as; no
  producer ever writes it.
* `summary.json`, `flow_coupling_metadata.json`,
  `final_corrected_face_flux.npy`, `face_to_cells.npy`, `cell_volumes.npy` —
  `run_gmsh_tetra_flow_debug.py` lines 13035 and 1770-1779.
"""

from __future__ import annotations

from microfluidics_mcp.outputs import (
    CONSUMER_PARAMETERS,
    STAGE_PRODUCTS,
    resolve_stage_outputs,
)

IMPORT_MESH_NPZ = "input_imported_mesh.npz"


def test_produced_names_are_the_names_the_entrypoints_write() -> None:
    assert STAGE_PRODUCTS["stage_import"].produced == {
        "mesh_npz_path": IMPORT_MESH_NPZ
    }
    assert STAGE_PRODUCTS["stage_flow"].produced == {
        "flow_summary_path": "summary.json",
        "flow_coupling_metadata_path": "flow_coupling_metadata.json",
        "flow_face_flux_path": "final_corrected_face_flux.npy",
        "flow_face_to_cells_path": "face_to_cells.npy",
        "flow_cell_volumes_path": "cell_volumes.npy",
    }


def test_the_staged_name_is_not_mistaken_for_a_produced_name() -> None:
    """`mesh.npz` is how the flow stage mounts its input, and nothing writes
    it. Expecting it from an import run is what left `mesh_npz_path`
    permanently `missing` after every real import."""

    produced = {
        name
        for products in STAGE_PRODUCTS.values()
        for name in products.produced.values()
    }

    assert "mesh.npz" not in produced
    assert "input.msh" not in produced


def test_terminal_stages_produce_nothing_for_the_next_stage() -> None:
    for stage_id in ("stage_transport", "stage_thermal"):
        assert STAGE_PRODUCTS[stage_id].produced == {}
        assert STAGE_PRODUCTS[stage_id].echoed == frozenset()

    assert STAGE_PRODUCTS["stage_reactive_transport"].produced == {}
    assert STAGE_PRODUCTS["stage_reactive_transport"].echoed == frozenset(
        {"reactive_case_path"}
    )


def test_consumer_parameters_come_from_the_registry() -> None:
    assert CONSUMER_PARAMETERS["mesh_path"] == frozenset(
        {"stage_import", "stage_thermal"}
    )
    assert CONSUMER_PARAMETERS["flow_summary_path"] == frozenset(
        {"stage_thermal", "stage_reactive_transport"}
    )


def test_import_run_exposes_mesh_npz_and_echoes_source_mesh() -> None:
    resolved = resolve_stage_outputs(
        experiment_id="stage_import",
        artifacts=[
            f"data/inputs/cfd/import-1/{IMPORT_MESH_NPZ}",
            "data/inputs/cfd/import-1/summary.json",
        ],
        request_parameters={"mesh_path": "data/inputs/gmsh/pipe.msh"},
    )

    assert resolved.outputs["mesh_npz_path"] == (
        f"data/inputs/cfd/import-1/{IMPORT_MESH_NPZ}"
    )
    assert resolved.outputs["mesh_path"] == "data/inputs/gmsh/pipe.msh"
    assert resolved.missing == []
    assert resolved.ambiguous == []
    assert resolved.artifact_count == 2


def test_an_import_summary_is_never_offered_as_the_flow_summary() -> None:
    """All five entrypoints write a `summary.json`. Only the flow stage's is a
    `flow_summary_path`; the import run's would validate as a key and then
    fail inside `experiments/gmsh/_flow_coupling.py`."""

    resolved = resolve_stage_outputs(
        experiment_id="stage_import",
        artifacts=[
            f"data/inputs/cfd/import-1/{IMPORT_MESH_NPZ}",
            "data/inputs/cfd/import-1/summary.json",
        ],
        request_parameters={"mesh_path": "data/inputs/gmsh/pipe.msh"},
    )

    assert "flow_summary_path" not in resolved.outputs
    assert "flow_summary_path" not in resolved.missing


def test_a_transport_run_offers_nothing_despite_its_summary() -> None:
    resolved = resolve_stage_outputs(
        experiment_id="stage_transport",
        artifacts=[
            "data/inputs/cfd/transport-1/summary.json",
            "data/inputs/cfd/transport-1/final_concentration.npy",
        ],
        request_parameters={"mesh_npz_path": f"import-1/{IMPORT_MESH_NPZ}"},
    )

    assert resolved.outputs == {}
    assert resolved.missing == []
    assert resolved.artifact_count == 2


def test_flow_run_exposes_every_downstream_input() -> None:
    prefix = "data/inputs/cfd/flow-1"
    resolved = resolve_stage_outputs(
        experiment_id="stage_flow",
        artifacts=[
            f"{prefix}/summary.json",
            f"{prefix}/flow_coupling_metadata.json",
            f"{prefix}/final_corrected_face_flux.npy",
            f"{prefix}/face_to_cells.npy",
            f"{prefix}/cell_volumes.npy",
            f"{prefix}/logs/run.log",
        ],
        request_parameters={
            "mesh_npz_path": f"data/inputs/cfd/import-1/{IMPORT_MESH_NPZ}"
        },
    )

    assert resolved.outputs["flow_summary_path"] == f"{prefix}/summary.json"
    assert (
        resolved.outputs["flow_face_to_cells_path"] == f"{prefix}/face_to_cells.npy"
    )
    assert resolved.outputs["mesh_npz_path"] == (
        f"data/inputs/cfd/import-1/{IMPORT_MESH_NPZ}"
    )
    # stage_thermal's mesh_path is not a flow parameter at all, so it is not
    # this run's business: it comes from the import run that echoed it.
    assert "mesh_path" not in resolved.outputs
    assert resolved.missing == []


def test_a_flow_run_that_produced_nothing_reports_its_own_parameters_missing() -> None:
    resolved = resolve_stage_outputs(
        experiment_id="stage_flow",
        artifacts=[],
        request_parameters={},
    )

    assert resolved.outputs == {}
    assert resolved.missing == sorted(
        set(STAGE_PRODUCTS["stage_flow"].produced) | {"mesh_npz_path"}
    )


def test_duplicate_basenames_are_reported_instead_of_guessed() -> None:
    resolved = resolve_stage_outputs(
        experiment_id="stage_flow",
        artifacts=[
            "data/inputs/cfd/flow-1/summary.json",
            "data/inputs/cfd/flow-1/nested/summary.json",
        ],
        request_parameters={},
    )

    assert "flow_summary_path" not in resolved.outputs
    assert "flow_summary_path" in resolved.ambiguous


def test_an_unknown_stage_resolves_nothing_rather_than_guessing() -> None:
    resolved = resolve_stage_outputs(
        experiment_id="",
        artifacts=["data/inputs/cfd/flow-1/summary.json"],
        request_parameters={"mesh_npz_path": "in/mesh.npz"},
    )

    assert resolved.outputs == {}
    assert resolved.missing == []
    assert resolved.artifact_count == 1
