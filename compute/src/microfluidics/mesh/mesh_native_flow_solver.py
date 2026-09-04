"""Minimal mesh-native incompressible flow foundation.

Projection scheme implemented in this module:
1) Predictor:
   u* = BC( u + dt * nu * Laplace(u) )
2) Poisson:
   Laplace(p) = div(u*) / dt
3) Correction:
   u_new = u* - alpha * dt * grad(p)

Discrete consistency choice:
- Poisson RHS is computed from face-flux divergence.
- Correction is applied on face fluxes first, then mapped back to cell velocity.
This keeps `div` and `grad` operators aligned inside the projection stage.

The native solver is intentionally limited to creeping-flow / Stokes-like regimes.
It does not attempt to model the convective term.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from microfluidics.cpu_threads import configure_torch_cpu_threads
from microfluidics.mesh.mesh_builder import MeshDataModel
from microfluidics.mesh.mesh_fields import MeshScalarField, MeshVectorField
from microfluidics.mesh.mesh_flow_fields import MeshFlowFields
from microfluidics.mesh.mesh_geometry import MeshGeometryModel3D
from microfluidics.mesh.mesh_linear_systems import PoissonSystem, solve_poisson_pcg
from microfluidics.mesh.mesh_operators import MeshTopology, build_mesh_topology
from microfluidics.mesh.mesh_scalar_solver import identify_boundary_cells


def _maybe_import_torch():
    try:
        import torch  # type: ignore

        configure_torch_cpu_threads(torch)
        return torch
    except ModuleNotFoundError:
        return None


def _select_backend(cfg) -> tuple[str, str, object | None]:
    torch_mod = _maybe_import_torch()
    for candidate in cfg.backend_preference:
        if candidate == "torch-cuda":
            if torch_mod is not None and bool(torch_mod.cuda.is_available()):
                return "torch-cuda", "cuda:0", torch_mod
            continue
        if candidate == "torch-cpu":
            if torch_mod is not None:
                return "torch-cpu", "cpu", torch_mod
            continue
        if candidate == "numpy":
            return "numpy", "cpu", None
    if torch_mod is not None and bool(torch_mod.cuda.is_available()):
        return "torch-cuda", "cuda:0", torch_mod
    if torch_mod is not None:
        return "torch-cpu", "cpu", torch_mod
    return "numpy", "cpu", None


def _to_numpy_array(values, backend: str) -> np.ndarray:
    if backend.startswith("torch"):
        return values.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(values, dtype=np.float64)


def _safe_ratio(numerator: float, denominator: float) -> float:
    denom = max(float(denominator), 1e-30)
    return float(numerator / denom)


def _compatibility_correct_rhs_numpy(rhs: np.ndarray) -> tuple[np.ndarray, float]:
    """Project Neumann Poisson RHS to zero-mean subspace for solvability."""
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    rhs_mean = float(np.mean(rhs_arr))
    return rhs_arr - rhs_mean, rhs_mean


def _compatibility_correct_rhs_torch(rhs, torch_mod):
    """Project Neumann Poisson RHS to zero-mean subspace for solvability."""
    rhs_mean = torch_mod.mean(rhs)
    return rhs - rhs_mean, float(rhs_mean.item())


def _build_divergence_region_masks(
    mesh: MeshDataModel,
    geometry: MeshGeometryModel3D,
    boundary_cells: Dict[str, np.ndarray],
    boundary_face_cells: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    centers = np.asarray(mesh.cell_centers, dtype=np.float64)
    x = centers[:, 0]
    y = centers[:, 1]
    cfg = geometry.config
    half_inlet = 0.5 * cfg.inlet_width
    half_outlet = 0.5 * cfg.outlet_width
    patch = max(float(cfg.patch_depth), 1e-12)

    inlet_zone = np.zeros(centers.shape[0], dtype=bool)
    for key in ("left_inlet", "right_inlet"):
        ids = boundary_cells.get(key, np.empty((0,), dtype=np.int64))
        if ids.size > 0:
            inlet_zone[ids] = True
    inlet_zone |= (np.abs(y) <= 1.1 * half_inlet) & (
        np.abs(x) >= 0.55 * cfg.inlet_length
    )

    junction_zone = (
        (np.abs(x) <= 0.75 * half_outlet)
        & (y >= -0.75 * half_inlet)
        & (y <= max(2.0 * half_inlet, cfg.cavity_offset_from_junction + patch))
    )

    outlet_near_zone = (np.abs(x) <= 1.05 * half_outlet) & (
        y >= cfg.outlet_length - max(2.0 * patch, 0.15 * cfg.outlet_length)
    )
    outlet_ids = boundary_cells.get("outlet", np.empty((0,), dtype=np.int64))
    if outlet_ids.size > 0:
        outlet_near_zone[outlet_ids] = True

    bulk_vertical_branch = (
        (np.abs(x) <= 0.95 * half_outlet)
        & (y >= max(2.0 * patch, 1.5 * half_inlet))
        & (y <= cfg.outlet_length - max(2.0 * patch, 0.15 * cfg.outlet_length))
    )

    masks = {
        "inlet_adjacent": np.zeros(centers.shape[0], dtype=bool),
        "outlet_adjacent": np.zeros(centers.shape[0], dtype=bool),
        "inlet_zone": inlet_zone,
        "junction_zone": junction_zone,
        "outlet_near_zone": outlet_near_zone,
        "bulk_vertical_branch": bulk_vertical_branch,
    }
    if boundary_face_cells["left_inlet"].size > 0:
        masks["inlet_adjacent"][boundary_face_cells["left_inlet"]] = True
    if boundary_face_cells["right_inlet"].size > 0:
        masks["inlet_adjacent"][boundary_face_cells["right_inlet"]] = True
    if boundary_face_cells["outlet"].size > 0:
        masks["outlet_adjacent"][boundary_face_cells["outlet"]] = True
    for key, mask in list(masks.items()):
        if int(np.count_nonzero(mask)) == 0:
            masks[key] = np.ones(centers.shape[0], dtype=bool)
    return masks


def _extract_boundary_face_cells(
    topology: MeshTopology,
    boundary_cells: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    left = boundary_cells["left_inlet"]
    right = boundary_cells["right_inlet"]
    outlet = boundary_cells["outlet"]
    left_face = left[topology.neighbor_ids[left, 0] < 0] if left.size else left
    right_face = right[topology.neighbor_ids[right, 1] < 0] if right.size else right
    outlet_face = (
        outlet[topology.neighbor_ids[outlet, 3] < 0] if outlet.size else outlet
    )
    return {
        "left_inlet": left_face,
        "right_inlet": right_face,
        "outlet": outlet_face,
    }


def _extract_inlet_layer_cells(
    topology: MeshTopology,
    boundary_face_cells: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    left_adj = boundary_face_cells["left_inlet"]
    right_adj = boundary_face_cells["right_inlet"]

    left_first_all = (
        topology.neighbor_ids[left_adj, 1]
        if left_adj.size
        else np.empty((0,), dtype=np.int64)
    )
    right_first_all = (
        topology.neighbor_ids[right_adj, 0]
        if right_adj.size
        else np.empty((0,), dtype=np.int64)
    )
    left_first = np.unique(left_first_all[left_first_all >= 0]).astype(np.int64)
    right_first = np.unique(right_first_all[right_first_all >= 0]).astype(np.int64)

    left_second_all = (
        topology.neighbor_ids[left_first, 1]
        if left_first.size
        else np.empty((0,), dtype=np.int64)
    )
    right_second_all = (
        topology.neighbor_ids[right_first, 0]
        if right_first.size
        else np.empty((0,), dtype=np.int64)
    )
    left_second = np.unique(left_second_all[left_second_all >= 0]).astype(np.int64)
    right_second = np.unique(right_second_all[right_second_all >= 0]).astype(np.int64)

    first_all = np.unique(np.concatenate([left_first, right_first])).astype(np.int64)
    second_all = np.unique(np.concatenate([left_second, right_second])).astype(np.int64)
    return {
        "left_first": left_first,
        "right_first": right_first,
        "first_all": first_all,
        "left_second": left_second,
        "right_second": right_second,
        "second_all": second_all,
    }


def _extract_inlet_layer_pairs(
    topology: MeshTopology,
    boundary_face_cells: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    left_adj = boundary_face_cells["left_inlet"]
    right_adj = boundary_face_cells["right_inlet"]

    left_pairs = np.empty((0, 2), dtype=np.int64)
    if left_adj.size:
        left_next = topology.neighbor_ids[left_adj, 1]
        valid = left_next >= 0
        if np.any(valid):
            left_pairs = np.stack([left_adj[valid], left_next[valid]], axis=1).astype(
                np.int64
            )

    right_pairs = np.empty((0, 2), dtype=np.int64)
    if right_adj.size:
        right_next = topology.neighbor_ids[right_adj, 0]
        valid = right_next >= 0
        if np.any(valid):
            right_pairs = np.stack(
                [right_adj[valid], right_next[valid]], axis=1
            ).astype(np.int64)

    return {"left": left_pairs, "right": right_pairs}


def _extract_interface_12_pairs(
    topology: MeshTopology,
    inlet_layer_cells: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    left_first = inlet_layer_cells["left_first"]
    right_first = inlet_layer_cells["right_first"]
    left_second = inlet_layer_cells["left_second"]
    right_second = inlet_layer_cells["right_second"]

    left_pairs = np.empty((0, 2), dtype=np.int64)
    if left_first.size > 0:
        left_next = topology.neighbor_ids[left_first, 1]
        valid = left_next >= 0
        if left_second.size > 0:
            valid &= np.isin(left_next, left_second)
        if np.any(valid):
            left_pairs = np.stack([left_first[valid], left_next[valid]], axis=1).astype(
                np.int64
            )

    right_pairs = np.empty((0, 2), dtype=np.int64)
    if right_first.size > 0:
        right_next = topology.neighbor_ids[right_first, 0]
        valid = right_next >= 0
        if right_second.size > 0:
            valid &= np.isin(right_next, right_second)
        if np.any(valid):
            right_pairs = np.stack(
                [right_first[valid], right_next[valid]], axis=1
            ).astype(np.int64)

    return {"left": left_pairs, "right": right_pairs}


def _extract_outlet_layer_cells(
    topology: MeshTopology,
    boundary_face_cells: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    outlet_adj = boundary_face_cells["outlet"]
    first_all = (
        topology.neighbor_ids[outlet_adj, 2]
        if outlet_adj.size
        else np.empty((0,), dtype=np.int64)
    )
    first = np.unique(first_all[first_all >= 0]).astype(np.int64)
    second_all = (
        topology.neighbor_ids[first, 2]
        if first.size
        else np.empty((0,), dtype=np.int64)
    )
    second = np.unique(second_all[second_all >= 0]).astype(np.int64)
    return {
        "adjacent": outlet_adj.astype(np.int64),
        "first": first,
        "second": second,
    }


def _extract_outlet_layer_pairs(
    topology: MeshTopology,
    outlet_layer_cells: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    adjacent = outlet_layer_cells["adjacent"]
    first = outlet_layer_cells["first"]
    second = outlet_layer_cells["second"]

    pair_01 = np.empty((0, 2), dtype=np.int64)
    if adjacent.size:
        next_ids = topology.neighbor_ids[adjacent, 2]
        valid = next_ids >= 0
        if first.size:
            valid &= np.isin(next_ids, first)
        if np.any(valid):
            pair_01 = np.stack([adjacent[valid], next_ids[valid]], axis=1).astype(
                np.int64
            )

    pair_12 = np.empty((0, 2), dtype=np.int64)
    if first.size:
        next_ids = topology.neighbor_ids[first, 2]
        valid = next_ids >= 0
        if second.size:
            valid &= np.isin(next_ids, second)
        if np.any(valid):
            pair_12 = np.stack([first[valid], next_ids[valid]], axis=1).astype(np.int64)

    return {"01": pair_01, "12": pair_12}


def _boundary_flux_stats_numpy(
    fluxes: np.ndarray,
    context: "MeshNativeFlowContext",
) -> Dict[str, float]:
    b = context.boundary_face_cells
    left_flux = (
        float(np.sum(fluxes[b["left_inlet"], 0])) if b["left_inlet"].size else 0.0
    )
    right_flux = (
        float(np.sum(fluxes[b["right_inlet"], 1])) if b["right_inlet"].size else 0.0
    )
    outlet_flux = float(np.sum(fluxes[b["outlet"], 3])) if b["outlet"].size else 0.0
    net_flux = left_flux + right_flux + outlet_flux
    return {
        "left_inlet": left_flux,
        "right_inlet": right_flux,
        "outlet": outlet_flux,
        "net": float(net_flux),
    }


def _regional_divergence_stats_numpy(
    divergence: np.ndarray,
    region_masks: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    div = np.asarray(divergence, dtype=np.float64).reshape(-1)
    stats: Dict[str, Dict[str, float]] = {}
    for name, mask in region_masks.items():
        active = np.asarray(mask, dtype=bool).reshape(-1)
        if active.shape[0] != div.shape[0] or not np.any(active):
            values = np.abs(div)
            count = int(div.shape[0])
        else:
            values = np.abs(div[active])
            count = int(np.count_nonzero(active))
        stats[name] = {
            "max_abs": float(np.max(values)) if values.size else 0.0,
            "mean_abs": float(np.mean(values)) if values.size else 0.0,
            "count": float(count),
        }
    return stats


def _inlet_zone_stats(region_stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    inlet = region_stats.get("inlet_zone", {})
    return {
        "max_abs": float(inlet.get("max_abs", 0.0)),
        "mean_abs": float(inlet.get("mean_abs", 0.0)),
    }


def _masked_abs_stats(values: np.ndarray, mask_ids: np.ndarray) -> Dict[str, float]:
    if mask_ids.size == 0:
        return {"max_abs": 0.0, "mean_abs": 0.0, "count": 0.0}
    sample = np.abs(np.asarray(values, dtype=np.float64)[mask_ids])
    if sample.size == 0:
        return {"max_abs": 0.0, "mean_abs": 0.0, "count": 0.0}
    return {
        "max_abs": float(np.max(sample)),
        "mean_abs": float(np.mean(sample)),
        "count": float(sample.size),
    }


def _unique_concat_ids(*arrays: np.ndarray) -> np.ndarray:
    non_empty = [
        np.asarray(arr, dtype=np.int64).reshape(-1) for arr in arrays if arr.size > 0
    ]
    if not non_empty:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.concatenate(non_empty)).astype(np.int64)


def _combine_samples(*arrays: np.ndarray) -> np.ndarray:
    non_empty = [
        np.asarray(arr, dtype=np.float64).reshape(-1) for arr in arrays if arr.size > 0
    ]
    if not non_empty:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(non_empty).astype(np.float64, copy=False)


def _sample_stats(values: np.ndarray) -> Dict[str, float]:
    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    if sample.size == 0:
        return {"max_abs": 0.0, "mean_abs": 0.0, "count": 0.0}
    abs_sample = np.abs(sample)
    return {
        "max_abs": float(np.max(abs_sample)),
        "mean_abs": float(np.mean(abs_sample)),
        "count": float(abs_sample.size),
    }


def _masked_component_stats(
    values: np.ndarray,
    ids: np.ndarray,
    component: int,
) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    idx = np.asarray(ids, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return {"max": 0.0, "mean": 0.0}
    comp = arr[idx, component]
    return {"max": float(np.max(comp)), "mean": float(np.mean(comp))}


def _regional_scalar_abs_stats(
    values: np.ndarray,
    region_masks: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    stats: Dict[str, Dict[str, float]] = {}
    for name, mask in region_masks.items():
        active = np.asarray(mask, dtype=bool).reshape(-1)
        if active.shape[0] != arr.shape[0] or not np.any(active):
            sample = np.abs(arr)
            count = int(arr.shape[0])
        else:
            sample = np.abs(arr[active])
            count = int(np.count_nonzero(active))
        stats[name] = {
            "max_abs": float(np.max(sample)) if sample.size else 0.0,
            "mean_abs": float(np.mean(sample)) if sample.size else 0.0,
            "count": float(count),
        }
    return stats


def _outlet_flux_stage_stats(
    *,
    flux_star: np.ndarray,
    flux_corrected: np.ndarray,
    flux_reconstructed: np.ndarray,
    flux_final: np.ndarray,
    context: "MeshNativeFlowContext",
) -> Dict[str, float]:
    outlet_ids = context.boundary_face_cells["outlet"]
    if outlet_ids.size == 0:
        return {
            "outlet_flux_star": 0.0,
            "outlet_flux_corrected": 0.0,
            "outlet_flux_reconstructed": 0.0,
            "outlet_flux_final": 0.0,
        }
    return {
        "outlet_flux_star": float(np.sum(flux_star[outlet_ids, 3])),
        "outlet_flux_corrected": float(np.sum(flux_corrected[outlet_ids, 3])),
        "outlet_flux_reconstructed": float(np.sum(flux_reconstructed[outlet_ids, 3])),
        "outlet_flux_final": float(np.sum(flux_final[outlet_ids, 3])),
    }


def _outlet_near_divergence_chain(
    *,
    div_star: np.ndarray,
    div_corrected_flux: np.ndarray,
    div_reconstructed_flux: np.ndarray,
    div_final: np.ndarray,
    context: "MeshNativeFlowContext",
) -> Dict[str, Dict[str, float]]:
    mask = context.region_masks.get(
        "outlet_near_zone", np.zeros_like(div_star, dtype=bool)
    )
    ids = np.flatnonzero(np.asarray(mask, dtype=bool))
    return {
        "star": _masked_abs_stats(div_star, ids),
        "corrected_flux": _masked_abs_stats(div_corrected_flux, ids),
        "reconstructed_flux": _masked_abs_stats(div_reconstructed_flux, ids),
        "final": _masked_abs_stats(div_final, ids),
    }


def _layer_stream_samples(
    *,
    layer_ids: np.ndarray,
    direction: int,
    velocity: np.ndarray,
    pressure: np.ndarray,
    fluxes: np.ndarray,
    context: "MeshNativeFlowContext",
) -> Dict[str, np.ndarray]:
    ids = np.asarray(layer_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        empty = np.empty((0,), dtype=np.float64)
        return {"grad": empty, "jump": empty, "mismatch": empty}
    neigh = context.topology.neighbor_ids
    next_ids = neigh[ids, direction]
    valid = next_ids >= 0
    if not np.any(valid):
        empty = np.empty((0,), dtype=np.float64)
        return {"grad": empty, "jump": empty, "mismatch": empty}

    ids_valid = ids[valid]
    next_valid = next_ids[valid]
    grad = np.abs(
        (pressure[next_valid] - pressure[ids_valid]) / float(context.topology.dx)
    )
    jump = np.linalg.norm(velocity[next_valid] - velocity[ids_valid], axis=1)
    normals = fluxes / context.face_areas
    n_curr = normals[ids_valid, direction]
    n_next = normals[next_valid, direction]
    mismatch = np.abs(n_next - n_curr)
    return {"grad": grad, "jump": jump, "mismatch": mismatch}


def _layer_by_layer_diagnostics(
    *,
    velocity: np.ndarray,
    pressure: np.ndarray,
    fluxes: np.ndarray,
    divergence: np.ndarray,
    context: "MeshNativeFlowContext",
) -> Dict[str, Dict[str, object]]:
    layers = context.inlet_layer_cells
    bfaces = context.boundary_face_cells
    layer0_left = bfaces["left_inlet"]
    layer0_right = bfaces["right_inlet"]
    layer1_left = layers["left_first"]
    layer1_right = layers["right_first"]
    layer2_left = layers["left_second"]
    layer2_right = layers["right_second"]

    def _layer_entry(
        name: str, left_ids: np.ndarray, right_ids: np.ndarray
    ) -> Dict[str, object]:
        layer_ids = _unique_concat_ids(left_ids, right_ids)
        div_stats = _masked_abs_stats(divergence, layer_ids)
        left_dir = 1
        right_dir = 0
        left_samples = _layer_stream_samples(
            layer_ids=left_ids,
            direction=left_dir,
            velocity=velocity,
            pressure=pressure,
            fluxes=fluxes,
            context=context,
        )
        right_samples = _layer_stream_samples(
            layer_ids=right_ids,
            direction=right_dir,
            velocity=velocity,
            pressure=pressure,
            fluxes=fluxes,
            context=context,
        )
        grad = _combine_samples(left_samples["grad"], right_samples["grad"])
        jump = _combine_samples(left_samples["jump"], right_samples["jump"])
        mismatch = _combine_samples(left_samples["mismatch"], right_samples["mismatch"])
        return {
            "name": name,
            "divergence": div_stats,
            "pressure_gradient_to_next": _sample_stats(grad),
            "velocity_jump_to_next": _sample_stats(jump),
            "normal_flux_mismatch_to_next": _sample_stats(mismatch),
        }

    return {
        "layer0_inlet_adjacent": _layer_entry(
            "layer0_inlet_adjacent", layer0_left, layer0_right
        ),
        "layer1_first_interior": _layer_entry(
            "layer1_first_interior", layer1_left, layer1_right
        ),
        "layer2_second_interior": _layer_entry(
            "layer2_second_interior",
            layer2_left,
            layer2_right,
        ),
    }


def _interface_12_samples(
    *,
    pairs: np.ndarray,
    direction: int,
    opposite_direction: int,
    velocity: np.ndarray,
    pressure: np.ndarray,
    fluxes: np.ndarray,
    context: "MeshNativeFlowContext",
    flux_reference: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    if pairs.size == 0:
        empty = np.empty((0,), dtype=np.float64)
        return {
            "flux": empty,
            "flux_correction": empty,
            "flux_mismatch": empty,
            "pgrad": empty,
            "jump": empty,
        }
    src = pairs[:, 0]
    dst = pairs[:, 1]
    shared_flux = fluxes[src, direction]
    mismatch = np.abs(shared_flux + fluxes[dst, opposite_direction])
    grad = np.abs((pressure[dst] - pressure[src]) / float(context.topology.dx))
    jump = np.linalg.norm(velocity[dst] - velocity[src], axis=1)
    corr = np.empty((0,), dtype=np.float64)
    if flux_reference is not None:
        corr = np.abs(shared_flux - flux_reference[src, direction])
    return {
        "flux": np.abs(shared_flux),
        "flux_correction": corr,
        "flux_mismatch": mismatch,
        "pgrad": grad,
        "jump": jump,
    }


def _interface_12_diagnostics(
    *,
    velocity: np.ndarray,
    pressure: np.ndarray,
    fluxes: np.ndarray,
    divergence: np.ndarray,
    context: "MeshNativeFlowContext",
    flux_reference: np.ndarray | None = None,
) -> Dict[str, Dict[str, float]]:
    pairs = context.interface_12_pairs
    left = _interface_12_samples(
        pairs=pairs["left"],
        direction=1,
        opposite_direction=0,
        velocity=velocity,
        pressure=pressure,
        fluxes=fluxes,
        context=context,
        flux_reference=flux_reference,
    )
    right = _interface_12_samples(
        pairs=pairs["right"],
        direction=0,
        opposite_direction=1,
        velocity=velocity,
        pressure=pressure,
        fluxes=fluxes,
        context=context,
        flux_reference=flux_reference,
    )
    flux_all = _combine_samples(left["flux"], right["flux"])
    corr_all = _combine_samples(left["flux_correction"], right["flux_correction"])
    mismatch_all = _combine_samples(left["flux_mismatch"], right["flux_mismatch"])
    pgrad_all = _combine_samples(left["pgrad"], right["pgrad"])
    jump_all = _combine_samples(left["jump"], right["jump"])
    layer2_ids = context.inlet_layer_cells["second_all"]
    return {
        "interface_12_flux": _sample_stats(flux_all),
        "interface_12_flux_correction": _sample_stats(corr_all),
        "interface_12_flux_mismatch": _sample_stats(mismatch_all),
        "layer2_divergence": _masked_abs_stats(divergence, layer2_ids),
        "layer2_pressure_gradient": _sample_stats(pgrad_all),
        "layer2_velocity_jump": _sample_stats(jump_all),
    }


def _outlet_interface_samples(
    *,
    pairs: np.ndarray,
    direction: int,
    opposite_direction: int,
    fluxes: np.ndarray,
    flux_reference: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    if pairs.size == 0:
        empty = np.empty((0,), dtype=np.float64)
        return {
            "flux": empty,
            "flux_correction": empty,
            "flux_mismatch": empty,
        }
    src = pairs[:, 0]
    dst = pairs[:, 1]
    shared_flux = fluxes[src, direction]
    mismatch = np.abs(shared_flux + fluxes[dst, opposite_direction])
    corr = np.empty((0,), dtype=np.float64)
    if flux_reference is not None:
        corr = np.abs(shared_flux - flux_reference[src, direction])
    return {
        "flux": np.abs(shared_flux),
        "flux_correction": corr,
        "flux_mismatch": mismatch,
    }


def _outlet_band_diagnostics(
    *,
    velocity: np.ndarray,
    pressure: np.ndarray,
    fluxes: np.ndarray,
    divergence: np.ndarray,
    context: "MeshNativeFlowContext",
    flux_reference: np.ndarray | None = None,
) -> Dict[str, object]:
    layers = context.outlet_layer_cells
    pairs = context.outlet_layer_pairs
    layer0 = np.asarray(layers["adjacent"], dtype=np.int64)
    layer1 = np.asarray(layers["first"], dtype=np.int64)
    layer2 = np.asarray(layers["second"], dtype=np.int64)

    s0 = _layer_stream_samples(
        layer_ids=layer0,
        direction=2,
        velocity=velocity,
        pressure=pressure,
        fluxes=fluxes,
        context=context,
    )
    s1 = _layer_stream_samples(
        layer_ids=layer1,
        direction=2,
        velocity=velocity,
        pressure=pressure,
        fluxes=fluxes,
        context=context,
    )
    s2 = _layer_stream_samples(
        layer_ids=layer2,
        direction=2,
        velocity=velocity,
        pressure=pressure,
        fluxes=fluxes,
        context=context,
    )
    pgrad_all = _combine_samples(s0["grad"], s1["grad"], s2["grad"])
    jump_all = _combine_samples(s0["jump"], s1["jump"], s2["jump"])
    mismatch_all = _combine_samples(s0["mismatch"], s1["mismatch"], s2["mismatch"])

    iface_01 = _outlet_interface_samples(
        pairs=pairs["01"],
        direction=2,
        opposite_direction=3,
        fluxes=fluxes,
        flux_reference=flux_reference,
    )
    iface_12 = _outlet_interface_samples(
        pairs=pairs["12"],
        direction=2,
        opposite_direction=3,
        fluxes=fluxes,
        flux_reference=flux_reference,
    )
    iface_flux = _combine_samples(iface_01["flux"], iface_12["flux"])
    iface_corr = _combine_samples(
        iface_01["flux_correction"], iface_12["flux_correction"]
    )
    iface_mismatch = _combine_samples(
        iface_01["flux_mismatch"], iface_12["flux_mismatch"]
    )

    return {
        "outlet_layer0_divergence": _masked_abs_stats(divergence, layer0),
        "outlet_layer1_divergence": _masked_abs_stats(divergence, layer1),
        "outlet_layer2_divergence": _masked_abs_stats(divergence, layer2),
        "outlet_pressure_gradient": _sample_stats(pgrad_all),
        "outlet_velocity_jump": _sample_stats(jump_all),
        "outlet_flux_mismatch": _sample_stats(mismatch_all),
        "outlet_interface_flux": _sample_stats(iface_flux),
        "outlet_interface_flux_correction": _sample_stats(iface_corr),
        "outlet_interface_flux_mismatch": _sample_stats(iface_mismatch),
    }


def _apply_layer2_local_response_limiter_numpy(
    velocity: np.ndarray,
    flux_target: np.ndarray,
    context: "MeshNativeFlowContext",
    base_blend: float = 0.18,
) -> np.ndarray:
    out = np.array(velocity, copy=True)
    alpha0 = float(np.clip(base_blend, 0.0, 1.0))
    if alpha0 <= 0.0:
        return out
    normals = flux_target / context.face_areas
    pairs = context.interface_12_pairs

    n_cells = out.shape[0]
    solid = np.zeros(n_cells, dtype=bool)
    wall_ids = context.boundary_cells["wall"]
    cavity_ids = context.boundary_cells["cavity_wall"]
    if wall_ids.size:
        solid[wall_ids] = True
    if cavity_ids.size:
        solid[cavity_ids] = True

    def _apply_side(pair_ids: np.ndarray, direction: int) -> None:
        if pair_ids.size == 0:
            return
        src = pair_ids[:, 0]
        dst = pair_ids[:, 1]
        valid = (~solid[src]) & (~solid[dst])
        if not np.any(valid):
            return
        src = src[valid]
        dst = dst[valid]
        n_face = normals[src, direction]
        if direction == 1:
            desired = 2.0 * n_face - out[src, 0]
        else:
            desired = -2.0 * n_face - out[src, 0]
        delta = desired - out[dst, 0]
        jump = np.abs(out[dst, 0] - out[src, 0])
        jump_scale = np.maximum(np.abs(out[src, 0]), 0.5 * context.cfg.inlet_speed)
        jump_scale = np.maximum(jump_scale, 1e-12)
        damp = 1.0 / (1.0 + (jump / jump_scale))
        alpha = alpha0 * damp
        out[dst, 0] = out[dst, 0] + alpha * delta
        out[dst, 1:] *= 1.0 - 0.10 * alpha[:, None]

    _apply_side(pairs["left"], direction=1)
    _apply_side(pairs["right"], direction=0)
    return out


def _apply_outlet_local_response_limiter_numpy(
    velocity: np.ndarray,
    flux_target: np.ndarray,
    context: "MeshNativeFlowContext",
    face_blend: float = 0.22,
    inner_blend: float = 0.16,
) -> np.ndarray:
    out = np.array(velocity, copy=True)
    alpha_face = float(np.clip(face_blend, 0.0, 1.0))
    alpha_inner = float(np.clip(inner_blend, 0.0, 1.0))
    if alpha_face <= 0.0 and alpha_inner <= 0.0:
        return out
    normals = flux_target / context.face_areas
    layers = context.outlet_layer_cells
    pairs = context.outlet_layer_pairs

    n_cells = out.shape[0]
    solid = np.zeros(n_cells, dtype=bool)
    wall_ids = context.boundary_cells["wall"]
    cavity_ids = context.boundary_cells["cavity_wall"]
    if wall_ids.size:
        solid[wall_ids] = True
    if cavity_ids.size:
        solid[cavity_ids] = True

    outlet_ids = np.asarray(layers["adjacent"], dtype=np.int64)
    if outlet_ids.size and alpha_face > 0.0:
        valid = ~solid[outlet_ids]
        if np.any(valid):
            ids = outlet_ids[valid]
            desired = normals[ids, 3]
            delta = desired - out[ids, 1]
            jump = np.abs(out[ids, 1])
            scale = np.maximum(0.5 * context.cfg.inlet_speed, np.abs(desired))
            scale = np.maximum(scale, 1e-12)
            damp = 1.0 / (1.0 + jump / scale)
            alpha = alpha_face * damp
            out[ids, 1] = out[ids, 1] + alpha * delta
            out[ids, 0] *= 1.0 - 0.08 * alpha
            out[ids, 2] *= 1.0 - 0.08 * alpha

    def _apply_pairs(pair_ids: np.ndarray, blend: float, target_mix: float) -> None:
        if pair_ids.size == 0 or blend <= 0.0:
            return
        src = pair_ids[:, 0]
        dst = pair_ids[:, 1]
        valid = (~solid[src]) & (~solid[dst])
        if not np.any(valid):
            return
        src = src[valid]
        dst = dst[valid]
        n_face = normals[src, 2]
        desired_flux = -2.0 * n_face - out[src, 1]
        mix = float(np.clip(target_mix, 0.0, 1.0))
        desired = mix * desired_flux + (1.0 - mix) * out[src, 1]
        delta = desired - out[dst, 1]
        jump = np.abs(out[dst, 1] - out[src, 1])
        scale = np.maximum(0.5 * context.cfg.inlet_speed, np.abs(out[src, 1]))
        scale = np.maximum(scale, 1e-12)
        damp = 1.0 / (1.0 + jump / scale)
        alpha = blend * damp
        out[dst, 1] = out[dst, 1] + alpha * delta
        out[dst, 0] *= 1.0 - 0.06 * alpha
        out[dst, 2] *= 1.0 - 0.06 * alpha

    _apply_pairs(pairs["01"], alpha_inner, 0.65)
    _apply_pairs(pairs["12"], 0.75 * alpha_inner, 0.50)
    return out


def _apply_layer2_local_response_limiter_torch(
    velocity,
    flux_target,
    context: "MeshNativeFlowContext",
    base_blend: float = 0.18,
):
    torch_mod = context.torch_module
    td = context.torch_data
    out = velocity.clone()
    alpha0 = float(np.clip(base_blend, 0.0, 1.0))
    if alpha0 <= 0.0:
        return out
    normals = flux_target / td["face_areas"]

    n_cells = int(out.shape[0])
    solid = torch_mod.zeros((n_cells,), dtype=torch_mod.bool, device=out.device)
    wall_ids = td["boundary_cells"]["wall"]
    cavity_ids = td["boundary_cells"]["cavity_wall"]
    if int(wall_ids.numel()) > 0:
        solid[wall_ids] = True
    if int(cavity_ids.numel()) > 0:
        solid[cavity_ids] = True

    def _apply_side(pair_ids, direction: int) -> None:
        if int(pair_ids.numel()) == 0:
            return
        src = pair_ids[:, 0]
        dst = pair_ids[:, 1]
        valid = (~solid[src]) & (~solid[dst])
        if not bool(torch_mod.any(valid)):
            return
        idx = torch_mod.nonzero(valid, as_tuple=False).reshape(-1)
        src = src[idx]
        dst = dst[idx]
        n_face = normals[src, direction]
        if direction == 1:
            desired = 2.0 * n_face - out[src, 0]
        else:
            desired = -2.0 * n_face - out[src, 0]
        delta = desired - out[dst, 0]
        jump = torch_mod.abs(out[dst, 0] - out[src, 0])
        jump_scale = torch_mod.maximum(
            torch_mod.abs(out[src, 0]),
            torch_mod.full_like(jump, 0.5 * context.cfg.inlet_speed),
        )
        jump_scale = torch_mod.maximum(jump_scale, torch_mod.full_like(jump, 1e-12))
        damp = 1.0 / (1.0 + (jump / jump_scale))
        alpha = alpha0 * damp
        out[dst, 0] = out[dst, 0] + alpha * delta
        out[dst, 1:] = out[dst, 1:] * (1.0 - 0.10 * alpha.unsqueeze(-1))

    _apply_side(td["interface_12_pairs"]["left"], direction=1)
    _apply_side(td["interface_12_pairs"]["right"], direction=0)
    return out


def _apply_outlet_local_response_limiter_torch(
    velocity,
    flux_target,
    context: "MeshNativeFlowContext",
    face_blend: float = 0.22,
    inner_blend: float = 0.16,
):
    torch_mod = context.torch_module
    td = context.torch_data
    out = velocity.clone()
    alpha_face = float(np.clip(face_blend, 0.0, 1.0))
    alpha_inner = float(np.clip(inner_blend, 0.0, 1.0))
    if alpha_face <= 0.0 and alpha_inner <= 0.0:
        return out
    normals = flux_target / td["face_areas"]

    n_cells = int(out.shape[0])
    solid = torch_mod.zeros((n_cells,), dtype=torch_mod.bool, device=out.device)
    wall_ids = td["boundary_cells"]["wall"]
    cavity_ids = td["boundary_cells"]["cavity_wall"]
    if int(wall_ids.numel()) > 0:
        solid[wall_ids] = True
    if int(cavity_ids.numel()) > 0:
        solid[cavity_ids] = True

    outlet_ids = td["outlet_layer_cells"]["adjacent"]
    if int(outlet_ids.numel()) > 0 and alpha_face > 0.0:
        valid = ~solid[outlet_ids]
        if bool(torch_mod.any(valid)):
            idx = torch_mod.nonzero(valid, as_tuple=False).reshape(-1)
            ids = outlet_ids[idx]
            desired = normals[ids, 3]
            delta = desired - out[ids, 1]
            jump = torch_mod.abs(out[ids, 1])
            scale = torch_mod.maximum(
                torch_mod.full_like(jump, 0.5 * context.cfg.inlet_speed),
                torch_mod.abs(desired),
            )
            scale = torch_mod.maximum(scale, torch_mod.full_like(scale, 1e-12))
            damp = 1.0 / (1.0 + jump / scale)
            alpha = alpha_face * damp
            out[ids, 1] = out[ids, 1] + alpha * delta
            out[ids, 0] = out[ids, 0] * (1.0 - 0.08 * alpha)
            out[ids, 2] = out[ids, 2] * (1.0 - 0.08 * alpha)

    def _apply_pairs(pair_ids, blend: float, target_mix: float) -> None:
        if int(pair_ids.numel()) == 0 or blend <= 0.0:
            return
        src = pair_ids[:, 0]
        dst = pair_ids[:, 1]
        valid = (~solid[src]) & (~solid[dst])
        if not bool(torch_mod.any(valid)):
            return
        idx = torch_mod.nonzero(valid, as_tuple=False).reshape(-1)
        src = src[idx]
        dst = dst[idx]
        n_face = normals[src, 2]
        desired_flux = -2.0 * n_face - out[src, 1]
        mix = float(np.clip(target_mix, 0.0, 1.0))
        desired = mix * desired_flux + (1.0 - mix) * out[src, 1]
        delta = desired - out[dst, 1]
        jump = torch_mod.abs(out[dst, 1] - out[src, 1])
        scale = torch_mod.maximum(
            torch_mod.full_like(jump, 0.5 * context.cfg.inlet_speed),
            torch_mod.abs(out[src, 1]),
        )
        scale = torch_mod.maximum(scale, torch_mod.full_like(scale, 1e-12))
        damp = 1.0 / (1.0 + jump / scale)
        alpha = blend * damp
        out[dst, 1] = out[dst, 1] + alpha * delta
        out[dst, 0] = out[dst, 0] * (1.0 - 0.06 * alpha)
        out[dst, 2] = out[dst, 2] * (1.0 - 0.06 * alpha)

    _apply_pairs(td["outlet_layer_pairs"]["01"], alpha_inner, 0.65)
    _apply_pairs(td["outlet_layer_pairs"]["12"], 0.75 * alpha_inner, 0.50)
    return out


def _inlet_layer_chain_diagnostics(
    velocity: np.ndarray,
    pressure: np.ndarray,
    fluxes: np.ndarray,
    divergence: np.ndarray,
    context: "MeshNativeFlowContext",
) -> Dict[str, object]:
    pairs = context.inlet_layer_pairs
    layers = context.inlet_layer_cells
    dx = float(context.topology.dx)
    bfaces = context.boundary_face_cells

    left_pairs = pairs["left"]
    right_pairs = pairs["right"]
    left_faces = bfaces["left_inlet"]
    right_faces = bfaces["right_inlet"]

    left_interior_flux = (
        float(np.sum(fluxes[left_pairs[:, 0], 1])) if left_pairs.size else 0.0
    )
    right_interior_flux = (
        float(np.sum(fluxes[right_pairs[:, 0], 0])) if right_pairs.size else 0.0
    )
    left_face_flux = float(np.sum(fluxes[left_faces, 0])) if left_faces.size else 0.0
    right_face_flux = float(np.sum(fluxes[right_faces, 1])) if right_faces.size else 0.0

    left_face_inlet_sign = -left_face_flux
    right_face_inlet_sign = -right_face_flux
    left_first_inlet_sign = left_interior_flux
    right_first_inlet_sign = right_interior_flux

    left_imposed = float(context.imposed_inlet_flux.get("left_inlet", 0.0))
    right_imposed = float(context.imposed_inlet_flux.get("right_inlet", 0.0))
    left_face_mismatch = left_face_flux - left_imposed
    right_face_mismatch = right_face_flux - right_imposed
    left_first_vs_face_mismatch = left_first_inlet_sign - left_face_inlet_sign
    right_first_vs_face_mismatch = right_first_inlet_sign - right_face_inlet_sign

    grad_samples = []
    jump_samples = []
    if left_pairs.size:
        left_adj = left_pairs[:, 0]
        left_inner = left_pairs[:, 1]
        grad_samples.append(np.abs((pressure[left_inner] - pressure[left_adj]) / dx))
        jump_samples.append(
            np.linalg.norm(velocity[left_inner] - velocity[left_adj], axis=1)
        )
    if right_pairs.size:
        right_adj = right_pairs[:, 0]
        right_inner = right_pairs[:, 1]
        grad_samples.append(np.abs((pressure[right_inner] - pressure[right_adj]) / dx))
        jump_samples.append(
            np.linalg.norm(velocity[right_inner] - velocity[right_adj], axis=1)
        )

    grad_all = (
        np.concatenate(grad_samples)
        if grad_samples
        else np.empty((0,), dtype=np.float64)
    )
    jump_all = (
        np.concatenate(jump_samples)
        if jump_samples
        else np.empty((0,), dtype=np.float64)
    )
    inlet_adjacent_ids = _unique_concat_ids(left_faces, right_faces)

    return {
        "inlet_face_flux_left": left_face_flux,
        "inlet_face_flux_right": right_face_flux,
        "inlet_face_flux_total": float(left_face_flux + right_face_flux),
        "inlet_face_flux_inlet_sign_left": left_face_inlet_sign,
        "inlet_face_flux_inlet_sign_right": right_face_inlet_sign,
        "inlet_face_flux_inlet_sign_total": float(
            left_face_inlet_sign + right_face_inlet_sign
        ),
        "inlet_face_flux_mismatch_left": float(left_face_mismatch),
        "inlet_face_flux_mismatch_right": float(right_face_mismatch),
        "inlet_face_flux_mismatch_max_abs": float(
            max(abs(left_face_mismatch), abs(right_face_mismatch))
        ),
        "first_interior_layer_flux_left": left_interior_flux,
        "first_interior_layer_flux_right": right_interior_flux,
        "first_interior_layer_flux_total": float(
            left_interior_flux + right_interior_flux
        ),
        "first_interior_layer_flux_inlet_sign_left": left_first_inlet_sign,
        "first_interior_layer_flux_inlet_sign_right": right_first_inlet_sign,
        "first_interior_layer_flux_inlet_sign_total": float(
            left_first_inlet_sign + right_first_inlet_sign
        ),
        "first_interior_layer_flux_mismatch_left": float(left_first_vs_face_mismatch),
        "first_interior_layer_flux_mismatch_right": float(right_first_vs_face_mismatch),
        "first_interior_layer_flux_mismatch_max_abs": float(
            max(abs(left_first_vs_face_mismatch), abs(right_first_vs_face_mismatch))
        ),
        "inlet_face_adjacent_divergence": _masked_abs_stats(
            divergence, inlet_adjacent_ids
        ),
        "first_interior_layer_divergence": _masked_abs_stats(
            divergence, layers["first_all"]
        ),
        "second_interior_layer_divergence": _masked_abs_stats(
            divergence, layers["second_all"]
        ),
        "first_interior_layer_pressure_gradient": {
            "max_abs": float(np.max(grad_all)) if grad_all.size else 0.0,
            "mean_abs": float(np.mean(grad_all)) if grad_all.size else 0.0,
            "count": float(grad_all.size),
        },
        "inlet_to_first_layer_velocity_jump": {
            "max": float(np.max(jump_all)) if jump_all.size else 0.0,
            "mean": float(np.mean(jump_all)) if jump_all.size else 0.0,
            "count": float(jump_all.size),
        },
    }


@dataclass(frozen=True)
class MeshNativeFlowConfig:
    dt: float = 5e-4
    viscosity: float = 1e-6
    inlet_speed: float = 0.15
    steps: int = 40
    pressure_max_iters: int = 600
    pressure_tol: float = 1e-9
    jacobi_omega: float = 0.65
    projection_relaxation: float = 1.0
    progress_every: int = 10
    backend_preference: tuple[str, ...] = ("torch-cuda", "torch-cpu", "numpy")
    warn_on_cpu: bool = True
    enable_safeguards: bool = True
    blowup_abs_limit: float = 1e12
    max_poisson_residual: float = 1e14
    max_divergence: float = 1e12
    post_projection_enforce_inlets: bool = False
    post_projection_enforce_outlet: bool = False
    post_projection_enforce_walls: bool = False
    max_supported_reynolds: float = 1.0
    allow_unsupported_reynolds: bool = False


@dataclass(frozen=True)
class MeshNativeFlowContext:
    mesh: MeshDataModel
    geometry: MeshGeometryModel3D
    topology: MeshTopology
    boundary_cells: Dict[str, np.ndarray]
    boundary_face_cells: Dict[str, np.ndarray]
    cfg: MeshNativeFlowConfig
    gauge_index: int
    cell_volumes: np.ndarray
    face_areas: np.ndarray
    face_distances: np.ndarray
    backend: str
    device: str
    torch_module: object | None
    torch_data: Dict[str, object] | None
    region_masks: Dict[str, np.ndarray]
    imposed_inlet_flux: Dict[str, float]
    inlet_layer_cells: Dict[str, np.ndarray]
    inlet_layer_pairs: Dict[str, np.ndarray]
    interface_12_pairs: Dict[str, np.ndarray]
    outlet_layer_cells: Dict[str, np.ndarray]
    outlet_layer_pairs: Dict[str, np.ndarray]
    estimated_reynolds: float


def _estimate_native_flow_reynolds(
    geometry: MeshGeometryModel3D,
    cfg: MeshNativeFlowConfig,
) -> float:
    hydraulic_diameter = max(float(geometry.config.hydraulic_diameter), 0.0)
    viscosity = max(float(cfg.viscosity), 1e-30)
    return float(abs(float(cfg.inlet_speed)) * hydraulic_diameter / viscosity)


def _compute_imposed_inlet_flux(
    *,
    boundary_face_cells: Dict[str, np.ndarray],
    face_areas: np.ndarray,
    inlet_speed: float,
) -> Dict[str, float]:
    left_ids = boundary_face_cells["left_inlet"]
    right_ids = boundary_face_cells["right_inlet"]
    left_area = float(np.sum(face_areas[left_ids, 0])) if left_ids.size else 0.0
    right_area = float(np.sum(face_areas[right_ids, 1])) if right_ids.size else 0.0
    left_flux = -inlet_speed * left_area
    right_flux = -inlet_speed * right_area
    return {
        "left_inlet": float(left_flux),
        "right_inlet": float(right_flux),
        "left_area": float(left_area),
        "right_area": float(right_area),
        "total_inlet": float(left_flux + right_flux),
    }


def _inlet_flux_mismatch(
    boundary_flux: Dict[str, float],
    imposed_inlet_flux: Dict[str, float],
) -> Dict[str, float]:
    left_imposed = float(imposed_inlet_flux.get("left_inlet", 0.0))
    right_imposed = float(imposed_inlet_flux.get("right_inlet", 0.0))
    left_reconstructed = float(boundary_flux.get("left_inlet", 0.0))
    right_reconstructed = float(boundary_flux.get("right_inlet", 0.0))
    left_delta = left_reconstructed - left_imposed
    right_delta = right_reconstructed - right_imposed
    return {
        "left_imposed": left_imposed,
        "right_imposed": right_imposed,
        "left_reconstructed": left_reconstructed,
        "right_reconstructed": right_reconstructed,
        "left_delta": float(left_delta),
        "right_delta": float(right_delta),
        "max_abs_delta": float(max(abs(left_delta), abs(right_delta))),
        "mean_abs_delta": float(0.5 * (abs(left_delta) + abs(right_delta))),
    }


@dataclass
class MeshNativeFlowState:
    mesh: MeshDataModel
    velocity: object
    pressure: object
    face_flux: object | None = None
    backend: str = "numpy"
    step: int = 0
    diagnostics_history: list[dict] | None = None

    def __post_init__(self) -> None:
        if self.diagnostics_history is None:
            self.diagnostics_history = []

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        velocity_np = _to_numpy_array(self.velocity, self.backend)
        pressure_np = _to_numpy_array(self.pressure, self.backend)
        return velocity_np, pressure_np

    @property
    def fields(self) -> MeshFlowFields:
        velocity_np, pressure_np = self.to_numpy()
        return MeshFlowFields(
            velocity=MeshVectorField(
                mesh=self.mesh,
                values=velocity_np,
                name="velocity_native",
            ),
            pressure=MeshScalarField(
                mesh=self.mesh,
                values=pressure_np,
                name="pressure_native",
            ),
        )


def _build_torch_data(
    topology: MeshTopology,
    boundary_cells: Dict[str, np.ndarray],
    boundary_face_cells: Dict[str, np.ndarray],
    inlet_layer_cells: Dict[str, np.ndarray],
    inlet_layer_pairs: Dict[str, np.ndarray],
    interface_12_pairs: Dict[str, np.ndarray],
    outlet_layer_cells: Dict[str, np.ndarray],
    outlet_layer_pairs: Dict[str, np.ndarray],
    cell_volumes: np.ndarray,
    face_areas: np.ndarray,
    face_distances: np.ndarray,
    device: str,
    torch_mod,
) -> Dict[str, object]:
    torch_device = torch_mod.device(device)
    neighbor_ids = torch_mod.tensor(
        topology.neighbor_ids,
        dtype=torch_mod.int64,
        device=torch_device,
    )
    has_neigh = neighbor_ids >= 0
    coeffs = torch_mod.tensor(
        1.0 / (face_distances * face_distances),
        dtype=torch_mod.float64,
        device=torch_device,
    )
    offdiag = torch_mod.zeros(
        (topology.n_cells, 6),
        dtype=torch_mod.float64,
        device=torch_device,
    )
    for face_idx in range(6):
        offdiag[:, face_idx] = torch_mod.where(
            has_neigh[:, face_idx],
            coeffs[face_idx],
            torch_mod.zeros(
                topology.n_cells, dtype=torch_mod.float64, device=torch_device
            ),
        )
    diag = torch_mod.sum(offdiag, dim=1)
    safe_diag = torch_mod.where(diag > 0.0, diag, torch_mod.ones_like(diag))
    boundary_torch = {
        key: torch_mod.tensor(ids, dtype=torch_mod.int64, device=torch_device)
        for key, ids in boundary_cells.items()
    }
    boundary_face_torch = {
        key: torch_mod.tensor(ids, dtype=torch_mod.int64, device=torch_device)
        for key, ids in boundary_face_cells.items()
    }
    inlet_layer_cells_torch = {
        key: torch_mod.tensor(ids, dtype=torch_mod.int64, device=torch_device)
        for key, ids in inlet_layer_cells.items()
    }
    inlet_layer_pairs_torch = {
        key: torch_mod.tensor(ids, dtype=torch_mod.int64, device=torch_device)
        for key, ids in inlet_layer_pairs.items()
    }
    interface_12_pairs_torch = {
        key: torch_mod.tensor(ids, dtype=torch_mod.int64, device=torch_device)
        for key, ids in interface_12_pairs.items()
    }
    outlet_layer_cells_torch = {
        key: torch_mod.tensor(ids, dtype=torch_mod.int64, device=torch_device)
        for key, ids in outlet_layer_cells.items()
    }
    outlet_layer_pairs_torch = {
        key: torch_mod.tensor(ids, dtype=torch_mod.int64, device=torch_device)
        for key, ids in outlet_layer_pairs.items()
    }
    return {
        "device": torch_device,
        "neighbor_ids": neighbor_ids,
        "has_neigh": has_neigh,
        "offdiag": offdiag,
        "diag": diag,
        "safe_diag": safe_diag,
        "cell_volumes": torch_mod.tensor(
            cell_volumes,
            dtype=torch_mod.float64,
            device=torch_device,
        ),
        "face_areas": torch_mod.tensor(
            face_areas,
            dtype=torch_mod.float64,
            device=torch_device,
        ),
        "face_distances": torch_mod.tensor(
            face_distances,
            dtype=torch_mod.float64,
            device=torch_device,
        ),
        "boundary_cells": boundary_torch,
        "boundary_face_cells": boundary_face_torch,
        "inlet_layer_cells": inlet_layer_cells_torch,
        "inlet_layer_pairs": inlet_layer_pairs_torch,
        "interface_12_pairs": interface_12_pairs_torch,
        "outlet_layer_cells": outlet_layer_cells_torch,
        "outlet_layer_pairs": outlet_layer_pairs_torch,
    }


def build_mesh_native_flow_context(
    mesh: MeshDataModel,
    geometry: MeshGeometryModel3D,
    cfg: MeshNativeFlowConfig,
) -> MeshNativeFlowContext:
    if cfg.dt <= 0:
        raise ValueError("dt must be positive.")
    if cfg.steps <= 0:
        raise ValueError("steps must be positive.")
    if cfg.viscosity < 0:
        raise ValueError("viscosity must be non-negative.")
    if cfg.inlet_speed < 0:
        raise ValueError("inlet_speed must be non-negative.")
    if cfg.max_supported_reynolds <= 0:
        raise ValueError("max_supported_reynolds must be positive.")
    if not (0.0 < cfg.jacobi_omega <= 1.0):
        raise ValueError("jacobi_omega must be in (0, 1].")
    if not (0.0 < cfg.projection_relaxation <= 1.0):
        raise ValueError("projection_relaxation must be in (0, 1].")
    if not np.isclose(float(cfg.projection_relaxation), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError(
            "mesh_native_flow_solver requires projection_relaxation=1.0 on the "
            "supported creeping-flow / Stokes-like path."
        )

    estimated_reynolds = _estimate_native_flow_reynolds(geometry, cfg)
    if not cfg.allow_unsupported_reynolds and estimated_reynolds > float(
        cfg.max_supported_reynolds
    ):
        raise ValueError(
            "mesh_native_flow_solver omits the convective term and only supports "
            "creeping-flow / Stokes-like use. "
            f"Estimated Reynolds number {estimated_reynolds:.3f} exceeds the "
            f"configured limit {cfg.max_supported_reynolds:.3f}. "
            "Lower inlet_speed, increase viscosity, or set "
            "allow_unsupported_reynolds=True for research-only debugging."
        )

    topology = build_mesh_topology(mesh)
    boundary_cells = identify_boundary_cells(mesh, geometry)
    if boundary_cells["left_inlet"].size == 0:
        raise ValueError("No cells tagged for left_inlet boundary.")
    if boundary_cells["right_inlet"].size == 0:
        raise ValueError("No cells tagged for right_inlet boundary.")
    if boundary_cells["outlet"].size == 0:
        raise ValueError("No cells tagged for outlet boundary.")
    boundary_face_cells = _extract_boundary_face_cells(topology, boundary_cells)
    if boundary_face_cells["left_inlet"].size == 0:
        raise ValueError("No left_inlet face-adjacent cells found for velocity BC.")
    if boundary_face_cells["right_inlet"].size == 0:
        raise ValueError("No right_inlet face-adjacent cells found for velocity BC.")
    if boundary_face_cells["outlet"].size == 0:
        raise ValueError("No outlet face-adjacent cells found for outlet BC.")
    inlet_layer_cells = _extract_inlet_layer_cells(topology, boundary_face_cells)
    inlet_layer_pairs = _extract_inlet_layer_pairs(topology, boundary_face_cells)
    interface_12_pairs = _extract_interface_12_pairs(topology, inlet_layer_cells)
    outlet_layer_cells = _extract_outlet_layer_cells(topology, boundary_face_cells)
    outlet_layer_pairs = _extract_outlet_layer_pairs(topology, outlet_layer_cells)
    gauge_index = int(boundary_cells["outlet"][0])

    cell_volumes = np.full(mesh.cells.shape[0], topology.dx * topology.dy * topology.dz)
    face_areas = np.repeat(
        np.array(
            [
                topology.dy * topology.dz,
                topology.dy * topology.dz,
                topology.dx * topology.dz,
                topology.dx * topology.dz,
                topology.dx * topology.dy,
                topology.dx * topology.dy,
            ],
            dtype=np.float64,
        )[None, :],
        mesh.cells.shape[0],
        axis=0,
    )
    face_distances = np.array(
        [
            topology.dx,
            topology.dx,
            topology.dy,
            topology.dy,
            topology.dz,
            topology.dz,
        ],
        dtype=np.float64,
    )

    backend, device, torch_mod = _select_backend(cfg)
    torch_data = None
    if backend.startswith("torch"):
        torch_data = _build_torch_data(
            topology=topology,
            boundary_cells=boundary_cells,
            boundary_face_cells=boundary_face_cells,
            inlet_layer_cells=inlet_layer_cells,
            inlet_layer_pairs=inlet_layer_pairs,
            interface_12_pairs=interface_12_pairs,
            outlet_layer_cells=outlet_layer_cells,
            outlet_layer_pairs=outlet_layer_pairs,
            cell_volumes=cell_volumes,
            face_areas=face_areas,
            face_distances=face_distances,
            device=device,
            torch_mod=torch_mod,
        )
    region_masks = _build_divergence_region_masks(
        mesh=mesh,
        geometry=geometry,
        boundary_cells=boundary_cells,
        boundary_face_cells=boundary_face_cells,
    )
    imposed_inlet_flux = _compute_imposed_inlet_flux(
        boundary_face_cells=boundary_face_cells,
        face_areas=face_areas,
        inlet_speed=cfg.inlet_speed,
    )
    return MeshNativeFlowContext(
        mesh=mesh,
        geometry=geometry,
        topology=topology,
        boundary_cells=boundary_cells,
        boundary_face_cells=boundary_face_cells,
        cfg=cfg,
        gauge_index=gauge_index,
        cell_volumes=cell_volumes,
        face_areas=face_areas,
        face_distances=face_distances,
        backend=backend,
        device=device,
        torch_module=torch_mod,
        torch_data=torch_data,
        region_masks=region_masks,
        imposed_inlet_flux=imposed_inlet_flux,
        inlet_layer_cells=inlet_layer_cells,
        inlet_layer_pairs=inlet_layer_pairs,
        interface_12_pairs=interface_12_pairs,
        outlet_layer_cells=outlet_layer_cells,
        outlet_layer_pairs=outlet_layer_pairs,
        estimated_reynolds=estimated_reynolds,
    )


def _outlet_zero_gradient_numpy(
    values: np.ndarray,
    topology: MeshTopology,
    outlet_ids: np.ndarray,
) -> np.ndarray:
    out = np.array(values, copy=True)
    if outlet_ids.size == 0:
        return out
    upstream = topology.neighbor_ids[outlet_ids, 2]
    has_upstream = upstream >= 0
    if np.any(has_upstream):
        out[outlet_ids[has_upstream], :] = out[upstream[has_upstream], :]
    return out


def _apply_inlet_first_layer_relaxation_numpy(
    values: np.ndarray,
    context: MeshNativeFlowContext,
    blend: float = 0.35,
) -> np.ndarray:
    out = np.array(values, copy=True)
    pairs = context.inlet_layer_pairs
    neigh = context.topology.neighbor_ids
    n_cells = out.shape[0]
    solid = np.zeros(n_cells, dtype=bool)
    wall_ids = context.boundary_cells["wall"]
    cavity_ids = context.boundary_cells["cavity_wall"]
    if wall_ids.size:
        solid[wall_ids] = True
    if cavity_ids.size:
        solid[cavity_ids] = True
    alpha = float(np.clip(blend, 0.0, 1.0))
    keep = 1.0 - alpha

    left_pairs = pairs["left"]
    if left_pairs.shape[0] > 0:
        adj = left_pairs[:, 0]
        inner = left_pairs[:, 1]
        valid_inner = ~solid[inner]
        if not np.any(valid_inner):
            inner = np.empty((0,), dtype=np.int64)
            adj = np.empty((0,), dtype=np.int64)
        else:
            inner = inner[valid_inner]
            adj = adj[valid_inner]
    if left_pairs.shape[0] > 0 and inner.size > 0:
        inner_next = neigh[inner, 1]
        has_next = inner_next >= 0
        if np.any(has_next):
            has_next &= ~solid[inner_next]
        target_x = out[adj, 0].copy()
        if np.any(has_next):
            target_x[has_next] = 0.5 * (
                out[adj[has_next], 0] + out[inner_next[has_next], 0]
            )
        out[inner, 0] = keep * out[inner, 0] + alpha * target_x
        out[inner, 1:] = keep * out[inner, 1:]

    right_pairs = pairs["right"]
    if right_pairs.shape[0] > 0:
        adj = right_pairs[:, 0]
        inner = right_pairs[:, 1]
        valid_inner = ~solid[inner]
        if not np.any(valid_inner):
            inner = np.empty((0,), dtype=np.int64)
            adj = np.empty((0,), dtype=np.int64)
        else:
            inner = inner[valid_inner]
            adj = adj[valid_inner]
    if right_pairs.shape[0] > 0 and inner.size > 0:
        inner_next = neigh[inner, 0]
        has_next = inner_next >= 0
        if np.any(has_next):
            has_next &= ~solid[inner_next]
        target_x = out[adj, 0].copy()
        if np.any(has_next):
            target_x[has_next] = 0.5 * (
                out[adj[has_next], 0] + out[inner_next[has_next], 0]
            )
        out[inner, 0] = keep * out[inner, 0] + alpha * target_x
        out[inner, 1:] = keep * out[inner, 1:]

    return out


def _enforce_inlet_band_projection_consistency_numpy(
    velocity: np.ndarray,
    flux_target: np.ndarray,
    context: MeshNativeFlowContext,
    blend: float = 0.45,
) -> np.ndarray:
    out = np.array(velocity, copy=True)
    alpha = float(np.clip(blend, 0.0, 1.0))
    keep = 1.0 - alpha
    neigh = context.topology.neighbor_ids
    normals = flux_target / context.face_areas

    n_cells = out.shape[0]
    solid = np.zeros(n_cells, dtype=bool)
    wall_ids = context.boundary_cells["wall"]
    cavity_ids = context.boundary_cells["cavity_wall"]
    if wall_ids.size:
        solid[wall_ids] = True
    if cavity_ids.size:
        solid[cavity_ids] = True

    def _apply_side(layer_ids: np.ndarray, direction: int) -> None:
        ids = np.asarray(layer_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return
        next_ids = neigh[ids, direction]
        valid = (next_ids >= 0) & (~solid[ids]) & (~solid[next_ids])
        if not np.any(valid):
            return
        src = ids[valid]
        dst = next_ids[valid]
        n_face = normals[src, direction]
        if direction == 1:
            desired = 2.0 * n_face - out[src, 0]
        else:
            desired = -2.0 * n_face - out[src, 0]
        out[dst, 0] = keep * out[dst, 0] + alpha * desired
        out[dst, 1:] = keep * out[dst, 1:]

    _apply_side(context.boundary_face_cells["left_inlet"], direction=1)
    _apply_side(context.boundary_face_cells["right_inlet"], direction=0)
    return out


def _apply_velocity_bc_numpy(
    values: np.ndarray,
    context: MeshNativeFlowContext,
    *,
    enforce_inlets: bool,
    enforce_outlet: bool,
    enforce_walls: bool,
) -> np.ndarray:
    out = np.array(values, copy=True)
    bc = context.boundary_cells
    bc_face = context.boundary_face_cells
    if enforce_walls:
        if bc["wall"].size > 0:
            out[bc["wall"], :] = 0.0
        if bc["cavity_wall"].size > 0:
            out[bc["cavity_wall"], :] = 0.0
    if enforce_inlets:
        if bc_face["left_inlet"].size > 0:
            out[bc_face["left_inlet"], 0] = context.cfg.inlet_speed
            out[bc_face["left_inlet"], 1:] = 0.0
        if bc_face["right_inlet"].size > 0:
            out[bc_face["right_inlet"], 0] = -context.cfg.inlet_speed
            out[bc_face["right_inlet"], 1:] = 0.0
        out = _apply_inlet_first_layer_relaxation_numpy(out, context)
    if enforce_outlet:
        out = _outlet_zero_gradient_numpy(
            out,
            context.topology,
            context.boundary_face_cells["outlet"],
        )
    return out


def _outlet_zero_gradient_torch(values, context: MeshNativeFlowContext):
    torch_mod = context.torch_module
    td = context.torch_data
    out = values.clone()
    outlet = td["boundary_face_cells"]["outlet"]
    if int(outlet.numel()) == 0:
        return out
    upstream = td["neighbor_ids"][outlet, 2]
    valid = upstream >= 0
    if bool(torch_mod.any(valid)):
        out[outlet[valid], :] = out[upstream[valid], :]
    return out


def _apply_inlet_first_layer_relaxation_torch(
    values,
    context: MeshNativeFlowContext,
    blend: float = 0.35,
):
    td = context.torch_data
    torch_mod = context.torch_module
    out = values.clone()
    n_cells = int(out.shape[0])
    solid = torch_mod.zeros((n_cells,), dtype=torch_mod.bool, device=out.device)
    wall_ids = td["boundary_cells"]["wall"]
    cavity_ids = td["boundary_cells"]["cavity_wall"]
    if int(wall_ids.numel()) > 0:
        solid[wall_ids] = True
    if int(cavity_ids.numel()) > 0:
        solid[cavity_ids] = True
    alpha = float(np.clip(blend, 0.0, 1.0))
    keep = 1.0 - alpha

    adj = td["inlet_layer_pairs"]["left"][:0, 0]
    inner = td["inlet_layer_pairs"]["left"][:0, 1]
    left_pairs = td["inlet_layer_pairs"]["left"]
    if int(left_pairs.numel()) > 0:
        adj = left_pairs[:, 0]
        inner = left_pairs[:, 1]
        valid_inner = ~solid[inner]
        if bool(torch_mod.any(valid_inner)):
            idx_valid = torch_mod.nonzero(valid_inner, as_tuple=False).reshape(-1)
            adj = adj[idx_valid]
            inner = inner[idx_valid]
        else:
            inner = inner[:0]
            adj = adj[:0]
    if int(inner.numel()) > 0:
        inner_next = td["neighbor_ids"][inner, 1]
        has_next = inner_next >= 0
        if bool(torch_mod.any(has_next)):
            has_next = has_next & (~solid[inner_next])
        target_x = out[adj, 0].clone()
        if bool(torch_mod.any(has_next)):
            idx = torch_mod.nonzero(has_next, as_tuple=False).reshape(-1)
            target_x[idx] = 0.5 * (out[adj[idx], 0] + out[inner_next[idx], 0])
        out[inner, 0] = keep * out[inner, 0] + alpha * target_x
        out[inner, 1:] = keep * out[inner, 1:]

    adj = td["inlet_layer_pairs"]["right"][:0, 0]
    inner = td["inlet_layer_pairs"]["right"][:0, 1]
    right_pairs = td["inlet_layer_pairs"]["right"]
    if int(right_pairs.numel()) > 0:
        adj = right_pairs[:, 0]
        inner = right_pairs[:, 1]
        valid_inner = ~solid[inner]
        if bool(torch_mod.any(valid_inner)):
            idx_valid = torch_mod.nonzero(valid_inner, as_tuple=False).reshape(-1)
            adj = adj[idx_valid]
            inner = inner[idx_valid]
        else:
            inner = inner[:0]
            adj = adj[:0]
    if int(inner.numel()) > 0:
        inner_next = td["neighbor_ids"][inner, 0]
        has_next = inner_next >= 0
        if bool(torch_mod.any(has_next)):
            has_next = has_next & (~solid[inner_next])
        target_x = out[adj, 0].clone()
        if bool(torch_mod.any(has_next)):
            idx = torch_mod.nonzero(has_next, as_tuple=False).reshape(-1)
            target_x[idx] = 0.5 * (out[adj[idx], 0] + out[inner_next[idx], 0])
        out[inner, 0] = keep * out[inner, 0] + alpha * target_x
        out[inner, 1:] = keep * out[inner, 1:]

    return out


def _enforce_inlet_band_projection_consistency_torch(
    velocity,
    flux_target,
    context: MeshNativeFlowContext,
    blend: float = 0.45,
):
    torch_mod = context.torch_module
    td = context.torch_data
    out = velocity.clone()
    alpha = float(np.clip(blend, 0.0, 1.0))
    keep = 1.0 - alpha
    normals = flux_target / td["face_areas"]
    neigh = td["neighbor_ids"]

    n_cells = int(out.shape[0])
    solid = torch_mod.zeros((n_cells,), dtype=torch_mod.bool, device=out.device)
    wall_ids = td["boundary_cells"]["wall"]
    cavity_ids = td["boundary_cells"]["cavity_wall"]
    if int(wall_ids.numel()) > 0:
        solid[wall_ids] = True
    if int(cavity_ids.numel()) > 0:
        solid[cavity_ids] = True

    def _apply_side(layer_ids, direction: int) -> None:
        if int(layer_ids.numel()) == 0:
            return
        next_ids = neigh[layer_ids, direction]
        valid = (next_ids >= 0) & (~solid[layer_ids]) & (~solid[next_ids])
        if not bool(torch_mod.any(valid)):
            return
        idx = torch_mod.nonzero(valid, as_tuple=False).reshape(-1)
        src = layer_ids[idx]
        dst = next_ids[idx]
        n_face = normals[src, direction]
        if direction == 1:
            desired = 2.0 * n_face - out[src, 0]
        else:
            desired = -2.0 * n_face - out[src, 0]
        out[dst, 0] = keep * out[dst, 0] + alpha * desired
        out[dst, 1:] = keep * out[dst, 1:]

    _apply_side(td["boundary_face_cells"]["left_inlet"], direction=1)
    _apply_side(td["boundary_face_cells"]["right_inlet"], direction=0)
    return out


def _apply_velocity_bc_torch(
    values,
    context: MeshNativeFlowContext,
    *,
    enforce_inlets: bool,
    enforce_outlet: bool,
    enforce_walls: bool,
):
    td = context.torch_data
    out = values.clone()
    bc = td["boundary_cells"]
    bc_face = td["boundary_face_cells"]
    if enforce_walls:
        if int(bc["wall"].numel()) > 0:
            out[bc["wall"], :] = 0.0
        if int(bc["cavity_wall"].numel()) > 0:
            out[bc["cavity_wall"], :] = 0.0
    if enforce_inlets:
        if int(bc_face["left_inlet"].numel()) > 0:
            out[bc_face["left_inlet"], 0] = context.cfg.inlet_speed
            out[bc_face["left_inlet"], 1:] = 0.0
        if int(bc_face["right_inlet"].numel()) > 0:
            out[bc_face["right_inlet"], 0] = -context.cfg.inlet_speed
            out[bc_face["right_inlet"], 1:] = 0.0
        out = _apply_inlet_first_layer_relaxation_torch(out, context)
    if enforce_outlet:
        out = _outlet_zero_gradient_torch(out, context)
    return out


def initialize_mesh_native_flow_state(
    context: MeshNativeFlowContext,
) -> MeshNativeFlowState:
    n_cells = context.mesh.cells.shape[0]
    if context.backend.startswith("torch"):
        torch_mod = context.torch_module
        device = context.torch_data["device"]
        velocity = torch_mod.zeros((n_cells, 3), dtype=torch_mod.float64, device=device)
        pressure = torch_mod.zeros(n_cells, dtype=torch_mod.float64, device=device)
        velocity = _apply_velocity_bc_torch(
            velocity,
            context,
            enforce_inlets=True,
            enforce_outlet=True,
            enforce_walls=True,
        )
        face_flux = compute_face_fluxes(velocity, context)
        return MeshNativeFlowState(
            mesh=context.mesh,
            velocity=velocity,
            pressure=pressure,
            face_flux=face_flux,
            backend=context.backend,
            step=0,
            diagnostics_history=[],
        )
    velocity = np.zeros((n_cells, 3), dtype=np.float64)
    pressure = np.zeros(n_cells, dtype=np.float64)
    velocity = _apply_velocity_bc_numpy(
        velocity,
        context,
        enforce_inlets=True,
        enforce_outlet=True,
        enforce_walls=True,
    )
    face_flux = compute_face_fluxes(velocity, context)
    return MeshNativeFlowState(
        mesh=context.mesh,
        velocity=velocity,
        pressure=pressure,
        face_flux=face_flux,
        backend="numpy",
        step=0,
        diagnostics_history=[],
    )


def apply_velocity_boundary_conditions(
    fields: MeshFlowFields,
    context: MeshNativeFlowContext,
) -> MeshFlowFields:
    values = _apply_velocity_bc_numpy(
        fields.velocity.values,
        context,
        enforce_inlets=True,
        enforce_outlet=True,
        enforce_walls=True,
    )
    fields.velocity.values = values
    return fields


def _axis_second_derivative_numpy(
    values: np.ndarray,
    minus_neigh: np.ndarray,
    plus_neigh: np.ndarray,
    inv_h2: float,
) -> np.ndarray:
    second = np.zeros(values.shape[0], dtype=np.float64)
    has_minus = minus_neigh >= 0
    has_plus = plus_neigh >= 0
    both = has_minus & has_plus
    second[both] = (
        values[plus_neigh[both]] - 2.0 * values[both] + values[minus_neigh[both]]
    ) * inv_h2
    only_plus = (~has_minus) & has_plus
    only_minus = has_minus & (~has_plus)
    second[only_plus] = (values[plus_neigh[only_plus]] - values[only_plus]) * inv_h2
    second[only_minus] = (values[minus_neigh[only_minus]] - values[only_minus]) * inv_h2
    return second


def _laplace_vector_numpy(values: np.ndarray, topology: MeshTopology) -> np.ndarray:
    inv_dx2 = 1.0 / (topology.dx * topology.dx)
    inv_dy2 = 1.0 / (topology.dy * topology.dy)
    inv_dz2 = 1.0 / (topology.dz * topology.dz)
    out = np.zeros_like(values)
    for comp in range(3):
        v = values[:, comp]
        out[:, comp] = (
            _axis_second_derivative_numpy(
                v,
                topology.neighbor_ids[:, 0],
                topology.neighbor_ids[:, 1],
                inv_dx2,
            )
            + _axis_second_derivative_numpy(
                v,
                topology.neighbor_ids[:, 2],
                topology.neighbor_ids[:, 3],
                inv_dy2,
            )
            + _axis_second_derivative_numpy(
                v,
                topology.neighbor_ids[:, 4],
                topology.neighbor_ids[:, 5],
                inv_dz2,
            )
        )
    return out


def _axis_second_derivative_torch(
    values, minus_neigh, plus_neigh, inv_h2: float, torch_mod
):
    second = torch_mod.zeros_like(values)
    has_minus = minus_neigh >= 0
    has_plus = plus_neigh >= 0
    both = has_minus & has_plus
    only_plus = (~has_minus) & has_plus
    only_minus = has_minus & (~has_plus)
    if bool(torch_mod.any(both)):
        idx = torch_mod.nonzero(both, as_tuple=False).reshape(-1)
        second[idx] = (
            values[plus_neigh[idx]] - 2.0 * values[idx] + values[minus_neigh[idx]]
        ) * inv_h2
    if bool(torch_mod.any(only_plus)):
        idx = torch_mod.nonzero(only_plus, as_tuple=False).reshape(-1)
        second[idx] = (values[plus_neigh[idx]] - values[idx]) * inv_h2
    if bool(torch_mod.any(only_minus)):
        idx = torch_mod.nonzero(only_minus, as_tuple=False).reshape(-1)
        second[idx] = (values[minus_neigh[idx]] - values[idx]) * inv_h2
    return second


def _laplace_vector_torch(values, context: MeshNativeFlowContext):
    torch_mod = context.torch_module
    td = context.torch_data
    neigh = td["neighbor_ids"]
    inv_dx2 = 1.0 / (context.topology.dx * context.topology.dx)
    inv_dy2 = 1.0 / (context.topology.dy * context.topology.dy)
    inv_dz2 = 1.0 / (context.topology.dz * context.topology.dz)
    out = torch_mod.zeros_like(values)
    for comp in range(3):
        v = values[:, comp]
        out[:, comp] = (
            _axis_second_derivative_torch(
                v, neigh[:, 0], neigh[:, 1], inv_dx2, torch_mod
            )
            + _axis_second_derivative_torch(
                v, neigh[:, 2], neigh[:, 3], inv_dy2, torch_mod
            )
            + _axis_second_derivative_torch(
                v, neigh[:, 4], neigh[:, 5], inv_dz2, torch_mod
            )
        )
    return out


def compute_face_normal_velocity(
    values: np.ndarray, topology: MeshTopology
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("values must be shaped (n_cells, 3).")
    if arr.shape[0] != topology.n_cells:
        raise ValueError("values row count must match topology.n_cells.")
    normal_vel = np.zeros((topology.n_cells, 6), dtype=np.float64)
    face_comp = (0, 0, 1, 1, 2, 2)
    face_sign = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    for face_idx in range(6):
        comp = face_comp[face_idx]
        sign = face_sign[face_idx]
        neigh = topology.neighbor_ids[:, face_idx]
        has_neigh = neigh >= 0
        face_val = np.zeros(topology.n_cells, dtype=np.float64)
        face_val[has_neigh] = 0.5 * (arr[has_neigh, comp] + arr[neigh[has_neigh], comp])
        face_val[~has_neigh] = arr[~has_neigh, comp]
        normal_vel[:, face_idx] = sign * face_val
    return normal_vel


def _compute_face_normal_velocity_torch(values, context: MeshNativeFlowContext):
    torch_mod = context.torch_module
    td = context.torch_data
    n_cells = int(values.shape[0])
    normal_vel = torch_mod.zeros((n_cells, 6), dtype=values.dtype, device=values.device)
    face_comp = (0, 0, 1, 1, 2, 2)
    face_sign = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    for face_idx in range(6):
        comp = face_comp[face_idx]
        neigh = td["neighbor_ids"][:, face_idx]
        has = td["has_neigh"][:, face_idx]
        face_val = values[:, comp].clone()
        if bool(torch_mod.any(has)):
            idx = torch_mod.nonzero(has, as_tuple=False).reshape(-1)
            face_val[idx] = 0.5 * (values[idx, comp] + values[neigh[idx], comp])
        normal_vel[:, face_idx] = face_sign[face_idx] * face_val
    return normal_vel


def _apply_face_boundary_normal_bc_numpy(
    normal_vel: np.ndarray,
    values: np.ndarray,
    context: MeshNativeFlowContext,
) -> np.ndarray:
    out = np.array(normal_vel, copy=True)
    missing = context.topology.neighbor_ids < 0
    out[missing] = 0.0
    bc = context.boundary_face_cells

    if bc["left_inlet"].size > 0:
        ids = bc["left_inlet"]
        out[ids, 0] = -values[ids, 0]
    if bc["right_inlet"].size > 0:
        ids = bc["right_inlet"]
        out[ids, 1] = values[ids, 0]
    if bc["outlet"].size > 0:
        ids = bc["outlet"]
        out[ids, 3] = values[ids, 1]
    return out


def _apply_face_boundary_normal_bc_torch(
    normal_vel, values, context: MeshNativeFlowContext
):
    td = context.torch_data
    out = normal_vel.clone()
    missing = ~td["has_neigh"]
    out[missing] = 0.0
    bc = td["boundary_face_cells"]

    if int(bc["left_inlet"].numel()) > 0:
        ids = bc["left_inlet"]
        out[ids, 0] = -values[ids, 0]
    if int(bc["right_inlet"].numel()) > 0:
        ids = bc["right_inlet"]
        out[ids, 1] = values[ids, 0]
    if int(bc["outlet"].numel()) > 0:
        ids = bc["outlet"]
        out[ids, 3] = values[ids, 1]
    return out


def compute_face_fluxes(values, context: MeshNativeFlowContext):
    if context.backend.startswith("torch") and context.torch_module is not None:
        if isinstance(values, context.torch_module.Tensor):
            normal_vel = _compute_face_normal_velocity_torch(values, context)
            normal_vel = _apply_face_boundary_normal_bc_torch(
                normal_vel, values, context
            )
            return normal_vel * context.torch_data["face_areas"]
    values_np = np.asarray(values, dtype=np.float64)
    normal_vel = compute_face_normal_velocity(values_np, context.topology)
    normal_vel = _apply_face_boundary_normal_bc_numpy(normal_vel, values_np, context)
    return normal_vel * context.face_areas


def _apply_flux_projection_closure_numpy(
    fluxes: np.ndarray,
    context: MeshNativeFlowContext,
) -> np.ndarray:
    out = np.array(fluxes, copy=True)
    neigh = context.topology.neighbor_ids
    has_neigh = neigh >= 0
    out[~has_neigh] = 0.0
    bc_face = context.boundary_face_cells

    left_ids = bc_face["left_inlet"]
    if left_ids.size > 0:
        out[left_ids, 0] = -context.cfg.inlet_speed * context.face_areas[left_ids, 0]
    right_ids = bc_face["right_inlet"]
    if right_ids.size > 0:
        out[right_ids, 1] = -context.cfg.inlet_speed * context.face_areas[right_ids, 1]
    return out


def _apply_flux_projection_closure_torch(fluxes, context: MeshNativeFlowContext):
    td = context.torch_data
    out = fluxes.clone()
    has_neigh = td["has_neigh"]
    out[~has_neigh] = 0.0
    bc_face = td["boundary_face_cells"]

    left_ids = bc_face["left_inlet"]
    if int(left_ids.numel()) > 0:
        out[left_ids, 0] = -context.cfg.inlet_speed * td["face_areas"][left_ids, 0]
    right_ids = bc_face["right_inlet"]
    if int(right_ids.numel()) > 0:
        out[right_ids, 1] = -context.cfg.inlet_speed * td["face_areas"][right_ids, 1]
    return out


def divergence_from_face_fluxes(fluxes, context: MeshNativeFlowContext):
    if context.backend.startswith("torch") and context.torch_module is not None:
        if isinstance(fluxes, context.torch_module.Tensor):
            return (
                context.torch_module.sum(fluxes, dim=1)
                / context.torch_data["cell_volumes"]
            )
    arr = np.asarray(fluxes, dtype=np.float64)
    if arr.shape != (context.topology.n_cells, 6):
        raise ValueError("fluxes must be shaped (n_cells, 6).")
    return np.sum(arr, axis=1) / context.cell_volumes


def _velocity_from_face_fluxes_numpy(
    fluxes: np.ndarray,
    context: MeshNativeFlowContext,
) -> np.ndarray:
    normals = fluxes / context.face_areas
    neigh = context.topology.neighbor_ids
    has_minus_x = neigh[:, 0] >= 0
    has_plus_x = neigh[:, 1] >= 0
    has_minus_y = neigh[:, 2] >= 0
    has_plus_y = neigh[:, 3] >= 0
    has_minus_z = neigh[:, 4] >= 0
    has_plus_z = neigh[:, 5] >= 0

    def _component(
        minus_idx: int, plus_idx: int, has_minus: np.ndarray, has_plus: np.ndarray
    ) -> np.ndarray:
        plus = normals[:, plus_idx]
        minus = normals[:, minus_idx]
        comp = 0.5 * (plus - minus)
        only_minus_boundary = (~has_minus) & has_plus
        only_plus_boundary = has_minus & (~has_plus)
        both_boundary = (~has_minus) & (~has_plus)
        if np.any(only_minus_boundary):
            comp[only_minus_boundary] = -minus[only_minus_boundary]
        if np.any(only_plus_boundary):
            comp[only_plus_boundary] = plus[only_plus_boundary]
        if np.any(both_boundary):
            comp[both_boundary] = 0.5 * (plus[both_boundary] - minus[both_boundary])
        return comp

    velocity = np.zeros((context.topology.n_cells, 3), dtype=np.float64)
    velocity[:, 0] = _component(0, 1, has_minus_x, has_plus_x)
    velocity[:, 1] = _component(2, 3, has_minus_y, has_plus_y)
    velocity[:, 2] = _component(4, 5, has_minus_z, has_plus_z)
    return velocity


def _velocity_from_face_fluxes_torch(fluxes, context: MeshNativeFlowContext):
    torch_mod = context.torch_module
    td = context.torch_data
    normals = fluxes / context.torch_data["face_areas"]
    neigh = td["neighbor_ids"]
    has_minus_x = neigh[:, 0] >= 0
    has_plus_x = neigh[:, 1] >= 0
    has_minus_y = neigh[:, 2] >= 0
    has_plus_y = neigh[:, 3] >= 0
    has_minus_z = neigh[:, 4] >= 0
    has_plus_z = neigh[:, 5] >= 0

    def _component(minus_idx: int, plus_idx: int, has_minus, has_plus):
        plus = normals[:, plus_idx]
        minus = normals[:, minus_idx]
        centered = 0.5 * (plus - minus)
        left_value = -minus
        right_value = plus
        comp = torch_mod.where((~has_minus) & has_plus, left_value, centered)
        comp = torch_mod.where(has_minus & (~has_plus), right_value, comp)
        return comp

    velocity = context.torch_module.zeros(
        (context.topology.n_cells, 3),
        dtype=fluxes.dtype,
        device=fluxes.device,
    )
    velocity[:, 0] = _component(0, 1, has_minus_x, has_plus_x)
    velocity[:, 1] = _component(2, 3, has_minus_y, has_plus_y)
    velocity[:, 2] = _component(4, 5, has_minus_z, has_plus_z)
    return velocity


def _pressure_flux_correction_numpy(
    pressure: np.ndarray,
    context: MeshNativeFlowContext,
) -> np.ndarray:
    neigh = context.topology.neighbor_ids
    has = neigh >= 0
    correction = np.zeros((context.topology.n_cells, 6), dtype=np.float64)
    for face_idx in range(6):
        active = has[:, face_idx]
        if not np.any(active):
            continue
        nb = neigh[active, face_idx]
        grad_n = (pressure[nb] - pressure[active]) / context.face_distances[face_idx]
        correction[active, face_idx] = (
            context.cfg.dt * context.face_areas[active, face_idx] * grad_n
        )
    return correction


def _pressure_flux_correction_torch(pressure, context: MeshNativeFlowContext):
    torch_mod = context.torch_module
    td = context.torch_data
    neigh = td["neighbor_ids"]
    has = td["has_neigh"]
    correction = torch_mod.zeros(
        (context.topology.n_cells, 6),
        dtype=pressure.dtype,
        device=pressure.device,
    )
    for face_idx in range(6):
        active = has[:, face_idx]
        if not bool(torch_mod.any(active)):
            continue
        idx = torch_mod.nonzero(active, as_tuple=False).reshape(-1)
        nb = neigh[idx, face_idx]
        grad_n = (pressure[nb] - pressure[idx]) / td["face_distances"][face_idx]
        correction[idx, face_idx] = (
            context.cfg.dt * td["face_areas"][idx, face_idx] * grad_n
        )
    return correction


def _poisson_residual_torch(pressure, rhs, context: MeshNativeFlowContext) -> float:
    torch_mod = context.torch_module
    td = context.torch_data
    lhs = torch_mod.zeros_like(pressure)
    for face_idx in range(6):
        has = td["has_neigh"][:, face_idx]
        if bool(torch_mod.any(has)):
            idx = torch_mod.nonzero(has, as_tuple=False).reshape(-1)
            nb = td["neighbor_ids"][idx, face_idx]
            coeff = td["offdiag"][idx, face_idx]
            lhs[idx] = lhs[idx] + coeff * (pressure[nb] - pressure[idx])
    return float(torch_mod.max(torch_mod.abs(lhs - rhs)).item())


def _poisson_residual_numpy(
    pressure: np.ndarray,
    rhs: np.ndarray,
    offdiag: np.ndarray,
    neigh: np.ndarray,
    has_neigh: np.ndarray,
) -> float:
    lhs = np.zeros_like(pressure)
    for face_idx in range(6):
        has = has_neigh[:, face_idx]
        lhs[has] += offdiag[has, face_idx] * (
            pressure[neigh[has, face_idx]] - pressure[has]
        )
    return float(np.max(np.abs(lhs - rhs)))


def _apply_poisson_laplace_numpy(
    pressure: np.ndarray,
    context: "MeshNativeFlowContext",
) -> np.ndarray:
    neigh = context.topology.neighbor_ids
    has_neigh = neigh >= 0
    coeffs = 1.0 / (context.face_distances * context.face_distances)
    laplace = np.zeros(context.topology.n_cells, dtype=np.float64)
    for face_idx in range(6):
        active = has_neigh[:, face_idx]
        if not np.any(active):
            continue
        nb = neigh[active, face_idx]
        laplace[active] += coeffs[face_idx] * (pressure[nb] - pressure[active])
    return laplace


def projection_operator_consistency_diagnostics(
    context: "MeshNativeFlowContext",
) -> Dict[str, object]:
    centers = np.asarray(context.mesh.cell_centers, dtype=np.float64)
    x = centers[:, 0]
    y = centers[:, 1]
    z = centers[:, 2]
    x0, x1 = float(np.min(x)), float(np.max(x))
    y0, y1 = float(np.min(y)), float(np.max(y))
    z0, z1 = float(np.min(z)), float(np.max(z))
    sx = (x - x0) / max(x1 - x0, 1e-12)
    sy = (y - y0) / max(y1 - y0, 1e-12)
    sz = (z - z0) / max(z1 - z0, 1e-12)
    pressure_probe = (
        np.sin(2.1 * np.pi * sx)
        + 0.7 * np.cos(1.7 * np.pi * sy)
        + 0.4 * np.sin(1.3 * np.pi * sz)
        + 0.2 * sx * sy
        - 0.1 * sy * sz
    ).astype(np.float64)
    laplace_from_poisson = _apply_poisson_laplace_numpy(pressure_probe, context)
    grad_flux_dt = _pressure_flux_correction_numpy(pressure_probe, context)
    div_grad = divergence_from_face_fluxes(grad_flux_dt, context) / context.cfg.dt
    diff = laplace_from_poisson - div_grad

    near_boundary = np.zeros(context.topology.n_cells, dtype=bool)
    for ids in context.boundary_cells.values():
        if ids.size:
            near_boundary[ids] = True
    interior = ~near_boundary
    interior_ids = np.flatnonzero(interior)
    near_boundary_ids = np.flatnonzero(near_boundary)
    regional = _regional_scalar_abs_stats(diff, context.region_masks)
    return {
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "interior": _masked_abs_stats(diff, interior_ids),
        "near_boundary": _masked_abs_stats(diff, near_boundary_ids),
        "regional": regional,
    }


def _solve_poisson_jacobi_torch(rhs, pressure_guess, context: MeshNativeFlowContext):
    torch_mod = context.torch_module
    td = context.torch_data
    omega = float(context.cfg.jacobi_omega)
    pressure = pressure_guess.clone()
    for iteration in range(1, context.cfg.pressure_max_iters + 1):
        neighbor_term = torch_mod.zeros_like(pressure)
        for face_idx in range(6):
            has = td["has_neigh"][:, face_idx]
            if bool(torch_mod.any(has)):
                idx = torch_mod.nonzero(has, as_tuple=False).reshape(-1)
                nb = td["neighbor_ids"][idx, face_idx]
                coeff = td["offdiag"][idx, face_idx]
                neighbor_term[idx] = neighbor_term[idx] + coeff * pressure[nb]
        candidate = (neighbor_term - rhs) / td["safe_diag"]
        pressure_new = pressure + omega * (candidate - pressure)
        pressure_new[context.gauge_index] = 0.0
        delta = torch_mod.max(torch_mod.abs(pressure_new - pressure))
        pressure = pressure_new
        if float(delta.item()) <= context.cfg.pressure_tol:
            residual = _poisson_residual_torch(pressure, rhs, context)
            return pressure, iteration, residual
    residual = _poisson_residual_torch(pressure, rhs, context)
    return pressure, context.cfg.pressure_max_iters, residual


def _solve_poisson_jacobi_numpy(
    rhs: np.ndarray,
    pressure_guess: np.ndarray,
    context: MeshNativeFlowContext,
):
    neigh = context.topology.neighbor_ids
    has_neigh = neigh >= 0
    coeffs = 1.0 / (context.face_distances * context.face_distances)
    offdiag = np.zeros((context.topology.n_cells, 6), dtype=np.float64)
    for face_idx in range(6):
        offdiag[has_neigh[:, face_idx], face_idx] = coeffs[face_idx]
    diag = np.sum(offdiag, axis=1)
    safe_diag = np.where(diag > 0.0, diag, 1.0)
    omega = float(context.cfg.jacobi_omega)
    pressure = np.array(pressure_guess, copy=True)
    iterations = context.cfg.pressure_max_iters
    for iteration in range(1, context.cfg.pressure_max_iters + 1):
        neighbor_term = np.zeros_like(pressure)
        for face_idx in range(6):
            has = has_neigh[:, face_idx]
            neighbor_term[has] += (
                offdiag[has, face_idx] * pressure[neigh[has, face_idx]]
            )
        candidate = (neighbor_term - rhs) / safe_diag
        pressure_new = pressure + omega * (candidate - pressure)
        pressure_new[context.gauge_index] = 0.0
        delta = float(np.max(np.abs(pressure_new - pressure)))
        pressure = pressure_new
        if delta <= context.cfg.pressure_tol:
            iterations = iteration
            break
    residual = _poisson_residual_numpy(pressure, rhs, offdiag, neigh, has_neigh)
    return pressure, iterations, residual


def _solve_poisson_pcg_numpy(
    rhs: np.ndarray,
    pressure_guess: np.ndarray,
    context: MeshNativeFlowContext,
):
    neigh = context.topology.neighbor_ids
    has_neigh = neigh >= 0
    coeffs = 1.0 / (context.face_distances * context.face_distances)
    offdiag = np.zeros((context.topology.n_cells, 6), dtype=np.float64)
    for face_idx in range(6):
        offdiag[has_neigh[:, face_idx], face_idx] = coeffs[face_idx]
    diag = np.sum(offdiag, axis=1)
    fixed_mask = np.zeros((context.topology.n_cells,), dtype=bool)
    fixed_mask[context.gauge_index] = True
    fixed_values = np.zeros((context.topology.n_cells,), dtype=np.float64)
    system = PoissonSystem(
        diag=diag,
        offdiag=offdiag,
        rhs_dirichlet=np.zeros((context.topology.n_cells,), dtype=np.float64),
    )
    return solve_poisson_pcg(
        rhs,
        context.topology,
        system,
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
        initial_guess=np.asarray(pressure_guess, dtype=np.float64),
        max_iters=context.cfg.pressure_max_iters,
        tol=context.cfg.pressure_tol,
    )


def _solve_poisson_pcg_torch(rhs, pressure_guess, context: MeshNativeFlowContext):
    torch_mod = context.torch_module
    pressure_np, iterations, residual = _solve_poisson_pcg_numpy(
        _to_numpy_array(rhs, context.backend),
        _to_numpy_array(pressure_guess, context.backend),
        context,
    )
    pressure = torch_mod.tensor(
        pressure_np,
        dtype=pressure_guess.dtype,
        device=pressure_guess.device,
    )
    pressure[context.gauge_index] = 0.0
    return pressure, iterations, residual


def _projection_from_flux_numpy(
    predicted_velocity: np.ndarray,
    pressure_guess: np.ndarray,
    context: MeshNativeFlowContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    flux_star = compute_face_fluxes(predicted_velocity, context)
    div_star = divergence_from_face_fluxes(flux_star, context)
    rhs_raw = div_star / context.cfg.dt
    rhs, rhs_mean_raw = _compatibility_correct_rhs_numpy(rhs_raw)
    pressure, iterations, poisson_res = _solve_poisson_pcg_numpy(
        rhs, pressure_guess, context
    )
    flux_correction = _pressure_flux_correction_numpy(pressure, context)
    flux_new = flux_star - flux_correction
    flux_new = _apply_flux_projection_closure_numpy(flux_new, context)
    div_new_flux = divergence_from_face_fluxes(flux_new, context)
    velocity_reconstructed_raw = _velocity_from_face_fluxes_numpy(flux_new, context)
    flux_from_reconstructed_raw = compute_face_fluxes(
        velocity_reconstructed_raw, context
    )
    div_from_reconstructed_raw = divergence_from_face_fluxes(
        flux_from_reconstructed_raw, context
    )
    flux_reconstruction_delta = flux_new - flux_from_reconstructed_raw
    div_reconstruction_delta = div_new_flux - div_from_reconstructed_raw
    velocity_new = np.array(velocity_reconstructed_raw, copy=True)
    outlet_adj_ids = np.flatnonzero(
        context.region_masks.get("outlet_adjacent", np.zeros_like(div_star, dtype=bool))
    )
    outlet_vy_after_correction = _masked_component_stats(
        velocity_reconstructed_raw,
        outlet_adj_ids,
        component=1,
    )
    outlet_vy_after_reconstruction = _masked_component_stats(
        velocity_new,
        outlet_adj_ids,
        component=1,
    )
    correction_vec = velocity_new - predicted_velocity
    region_star = _regional_divergence_stats_numpy(div_star, context.region_masks)
    region_corrected = _regional_divergence_stats_numpy(
        div_new_flux, context.region_masks
    )
    bflux_star = _boundary_flux_stats_numpy(flux_star, context)
    bflux_corrected = _boundary_flux_stats_numpy(flux_new, context)
    inlet_mismatch_star = _inlet_flux_mismatch(bflux_star, context.imposed_inlet_flux)
    inlet_mismatch_corrected = _inlet_flux_mismatch(
        bflux_corrected,
        context.imposed_inlet_flux,
    )
    inlet_flux_star = {
        "left_inlet": float(bflux_star["left_inlet"]),
        "right_inlet": float(bflux_star["right_inlet"]),
        "total_inlet": float(bflux_star["left_inlet"] + bflux_star["right_inlet"]),
    }
    inlet_flux_corrected = {
        "left_inlet": float(bflux_corrected["left_inlet"]),
        "right_inlet": float(bflux_corrected["right_inlet"]),
        "total_inlet": float(
            bflux_corrected["left_inlet"] + bflux_corrected["right_inlet"]
        ),
    }
    vol_div_star = float(np.sum(div_star * context.cell_volumes))
    vol_div_corrected = float(np.sum(div_new_flux * context.cell_volumes))
    flux_delta_cell_max = np.max(np.abs(flux_reconstruction_delta), axis=1)
    flux_delta_cell_mean = np.mean(np.abs(flux_reconstruction_delta), axis=1)
    flux_reconstruction_region_max = _regional_scalar_abs_stats(
        flux_delta_cell_max,
        context.region_masks,
    )
    flux_reconstruction_region_mean = _regional_scalar_abs_stats(
        flux_delta_cell_mean,
        context.region_masks,
    )
    div_reconstruction_region = _regional_scalar_abs_stats(
        div_reconstruction_delta,
        context.region_masks,
    )
    outlet_flux_stages = _outlet_flux_stage_stats(
        flux_star=flux_star,
        flux_corrected=flux_new,
        flux_reconstructed=flux_from_reconstructed_raw,
        flux_final=flux_new,
        context=context,
    )
    outlet_near_div_chain = _outlet_near_divergence_chain(
        div_star=div_star,
        div_corrected_flux=div_new_flux,
        div_reconstructed_flux=div_from_reconstructed_raw,
        div_final=div_new_flux,
        context=context,
    )
    inlet_layer_star = _inlet_layer_chain_diagnostics(
        velocity=predicted_velocity,
        pressure=pressure_guess,
        fluxes=flux_star,
        divergence=div_star,
        context=context,
    )
    inlet_layer_corrected = _inlet_layer_chain_diagnostics(
        velocity=velocity_new,
        pressure=pressure,
        fluxes=flux_new,
        divergence=div_new_flux,
        context=context,
    )
    layer_band_star = _layer_by_layer_diagnostics(
        velocity=predicted_velocity,
        pressure=pressure_guess,
        fluxes=flux_star,
        divergence=div_star,
        context=context,
    )
    layer_band_corrected = _layer_by_layer_diagnostics(
        velocity=velocity_new,
        pressure=pressure,
        fluxes=flux_new,
        divergence=div_new_flux,
        context=context,
    )
    interface_12_star = _interface_12_diagnostics(
        velocity=predicted_velocity,
        pressure=pressure_guess,
        fluxes=flux_star,
        divergence=div_star,
        context=context,
        flux_reference=None,
    )
    interface_12_corrected = _interface_12_diagnostics(
        velocity=velocity_new,
        pressure=pressure,
        fluxes=flux_new,
        divergence=div_new_flux,
        context=context,
        flux_reference=flux_star,
    )
    outlet_band_star = _outlet_band_diagnostics(
        velocity=predicted_velocity,
        pressure=pressure_guess,
        fluxes=flux_star,
        divergence=div_star,
        context=context,
        flux_reference=None,
    )
    outlet_band_corrected = _outlet_band_diagnostics(
        velocity=velocity_new,
        pressure=pressure,
        fluxes=flux_new,
        divergence=div_new_flux,
        context=context,
        flux_reference=flux_star,
    )
    diag = {
        "divergence_star_max_abs": float(np.max(np.abs(div_star))),
        "divergence_star_l2": float(np.sqrt(np.mean(div_star * div_star))),
        "divergence_projected_flux_max_abs": float(np.max(np.abs(div_new_flux))),
        "divergence_projected_flux_l2": float(
            np.sqrt(np.mean(div_new_flux * div_new_flux))
        ),
        "divergence_projected_max_abs": float(np.max(np.abs(div_new_flux))),
        "divergence_projected_l2": float(np.sqrt(np.mean(div_new_flux * div_new_flux))),
        "projection_reduction_factor_max_abs": _safe_ratio(
            float(np.max(np.abs(div_new_flux))),
            float(np.max(np.abs(div_star))),
        ),
        "projection_reduction_factor_l2": _safe_ratio(
            float(np.sqrt(np.mean(div_new_flux * div_new_flux))),
            float(np.sqrt(np.mean(div_star * div_star))),
        ),
        "poisson_iterations": int(iterations),
        "poisson_residual_max_abs": float(poisson_res),
        "pressure_solver": "pcg_diag",
        "pressure_gauge_value": float(pressure[context.gauge_index]),
        "max_correction_magnitude": float(
            np.max(np.linalg.norm(correction_vec, axis=1))
        ),
        "poisson_rhs_mean_raw": float(rhs_mean_raw),
        "poisson_rhs_mean_corrected": float(np.mean(rhs)),
        "poisson_rhs_l2": float(np.sqrt(np.mean(rhs * rhs))),
        "projection_chain_flux_reconstruction_mismatch_max_abs": float(
            np.max(np.abs(flux_reconstruction_delta))
        ),
        "projection_chain_flux_reconstruction_mismatch_mean_abs": float(
            np.mean(np.abs(flux_reconstruction_delta))
        ),
        "projection_chain_divergence_reconstruction_mismatch_max_abs": float(
            np.max(np.abs(div_reconstruction_delta))
        ),
        "projection_chain_divergence_reconstruction_mismatch_mean_abs": float(
            np.mean(np.abs(div_reconstruction_delta))
        ),
        "projection_chain_flux_reconstruction_mismatch_region_max": flux_reconstruction_region_max,
        "projection_chain_flux_reconstruction_mismatch_region_mean": (
            flux_reconstruction_region_mean
        ),
        "projection_chain_divergence_reconstruction_mismatch_region": div_reconstruction_region,
        "outlet_flux_stages": outlet_flux_stages,
        "outlet_near_divergence_chain": outlet_near_div_chain,
        "outlet_adjacent_vy_after_correction": outlet_vy_after_correction,
        "outlet_adjacent_vy_after_reconstruction": outlet_vy_after_reconstruction,
        "boundary_flux_star": bflux_star,
        "boundary_flux_corrected": bflux_corrected,
        "net_boundary_flux_star": float(bflux_star["net"]),
        "net_boundary_flux_corrected": float(bflux_corrected["net"]),
        "volume_integrated_divergence_star": vol_div_star,
        "volume_integrated_divergence_corrected": vol_div_corrected,
        "regional_divergence_star": region_star,
        "regional_divergence_corrected": region_corrected,
        "imposed_inlet_flux": dict(context.imposed_inlet_flux),
        "inlet_flux_star": inlet_flux_star,
        "inlet_flux_corrected": inlet_flux_corrected,
        "inlet_flux_mismatch_star": inlet_mismatch_star,
        "inlet_flux_mismatch_corrected": inlet_mismatch_corrected,
        "inlet_zone_divergence_star": _inlet_zone_stats(region_star),
        "inlet_zone_divergence_corrected": _inlet_zone_stats(region_corrected),
        "inlet_layer_chain_star": inlet_layer_star,
        "inlet_layer_chain_corrected": inlet_layer_corrected,
        "layer_band_star": layer_band_star,
        "layer_band_corrected": layer_band_corrected,
        "interface_12_star": interface_12_star,
        "interface_12_corrected": interface_12_corrected,
        "outlet_band_star": outlet_band_star,
        "outlet_band_corrected": outlet_band_corrected,
        "first_interior_layer_flux_mismatch_star_max_abs": float(
            inlet_layer_star["first_interior_layer_flux_mismatch_max_abs"]
        ),
        "first_interior_layer_flux_mismatch_corrected_max_abs": float(
            inlet_layer_corrected["first_interior_layer_flux_mismatch_max_abs"]
        ),
        "inlet_face_flux_mismatch_star_max_abs": float(
            inlet_layer_star["inlet_face_flux_mismatch_max_abs"]
        ),
        "inlet_face_flux_mismatch_corrected_max_abs": float(
            inlet_layer_corrected["inlet_face_flux_mismatch_max_abs"]
        ),
        "interface_12_flux_max": float(
            interface_12_corrected["interface_12_flux"]["max_abs"]
        ),
        "interface_12_flux_mean": float(
            interface_12_corrected["interface_12_flux"]["mean_abs"]
        ),
        "interface_12_flux_correction_max": float(
            interface_12_corrected["interface_12_flux_correction"]["max_abs"]
        ),
        "interface_12_flux_correction_mean": float(
            interface_12_corrected["interface_12_flux_correction"]["mean_abs"]
        ),
        "interface_12_flux_mismatch_max": float(
            interface_12_corrected["interface_12_flux_mismatch"]["max_abs"]
        ),
        "interface_12_flux_mismatch_mean": float(
            interface_12_corrected["interface_12_flux_mismatch"]["mean_abs"]
        ),
        "layer2_div_star_max": float(interface_12_star["layer2_divergence"]["max_abs"]),
        "layer2_div_star_mean": float(
            interface_12_star["layer2_divergence"]["mean_abs"]
        ),
        "layer2_div_corrected_max": float(
            interface_12_corrected["layer2_divergence"]["max_abs"]
        ),
        "layer2_div_corrected_mean": float(
            interface_12_corrected["layer2_divergence"]["mean_abs"]
        ),
        "layer2_pressure_gradient_max": float(
            interface_12_corrected["layer2_pressure_gradient"]["max_abs"]
        ),
        "layer2_pressure_gradient_mean": float(
            interface_12_corrected["layer2_pressure_gradient"]["mean_abs"]
        ),
        "layer2_velocity_jump_max": float(
            interface_12_corrected["layer2_velocity_jump"]["max_abs"]
        ),
        "layer2_velocity_jump_mean": float(
            interface_12_corrected["layer2_velocity_jump"]["mean_abs"]
        ),
        "layer2_flux_response_max": float(
            interface_12_corrected["interface_12_flux_correction"]["max_abs"]
        ),
        "layer2_flux_response_mean": float(
            interface_12_corrected["interface_12_flux_correction"]["mean_abs"]
        ),
        "outlet_layer0_div_star_max": float(
            outlet_band_star["outlet_layer0_divergence"]["max_abs"]
        ),
        "outlet_layer0_div_star_mean": float(
            outlet_band_star["outlet_layer0_divergence"]["mean_abs"]
        ),
        "outlet_layer0_div_corrected_max": float(
            outlet_band_corrected["outlet_layer0_divergence"]["max_abs"]
        ),
        "outlet_layer0_div_corrected_mean": float(
            outlet_band_corrected["outlet_layer0_divergence"]["mean_abs"]
        ),
        "outlet_layer1_div_star_max": float(
            outlet_band_star["outlet_layer1_divergence"]["max_abs"]
        ),
        "outlet_layer1_div_star_mean": float(
            outlet_band_star["outlet_layer1_divergence"]["mean_abs"]
        ),
        "outlet_layer1_div_corrected_max": float(
            outlet_band_corrected["outlet_layer1_divergence"]["max_abs"]
        ),
        "outlet_layer1_div_corrected_mean": float(
            outlet_band_corrected["outlet_layer1_divergence"]["mean_abs"]
        ),
        "outlet_layer2_div_star_max": float(
            outlet_band_star["outlet_layer2_divergence"]["max_abs"]
        ),
        "outlet_layer2_div_star_mean": float(
            outlet_band_star["outlet_layer2_divergence"]["mean_abs"]
        ),
        "outlet_layer2_div_corrected_max": float(
            outlet_band_corrected["outlet_layer2_divergence"]["max_abs"]
        ),
        "outlet_layer2_div_corrected_mean": float(
            outlet_band_corrected["outlet_layer2_divergence"]["mean_abs"]
        ),
        "outlet_pressure_gradient_max": float(
            outlet_band_corrected["outlet_pressure_gradient"]["max_abs"]
        ),
        "outlet_pressure_gradient_mean": float(
            outlet_band_corrected["outlet_pressure_gradient"]["mean_abs"]
        ),
        "outlet_velocity_jump_max": float(
            outlet_band_corrected["outlet_velocity_jump"]["max_abs"]
        ),
        "outlet_velocity_jump_mean": float(
            outlet_band_corrected["outlet_velocity_jump"]["mean_abs"]
        ),
        "outlet_flux_mismatch_max": float(
            outlet_band_corrected["outlet_flux_mismatch"]["max_abs"]
        ),
        "outlet_flux_mismatch_mean": float(
            outlet_band_corrected["outlet_flux_mismatch"]["mean_abs"]
        ),
        "outlet_interface_flux_max": float(
            outlet_band_corrected["outlet_interface_flux"]["max_abs"]
        ),
        "outlet_interface_flux_mean": float(
            outlet_band_corrected["outlet_interface_flux"]["mean_abs"]
        ),
        "outlet_interface_flux_correction_max": float(
            outlet_band_corrected["outlet_interface_flux_correction"]["max_abs"]
        ),
        "outlet_interface_flux_correction_mean": float(
            outlet_band_corrected["outlet_interface_flux_correction"]["mean_abs"]
        ),
        "outlet_interface_flux_mismatch_max": float(
            outlet_band_corrected["outlet_interface_flux_mismatch"]["max_abs"]
        ),
        "outlet_interface_flux_mismatch_mean": float(
            outlet_band_corrected["outlet_interface_flux_mismatch"]["mean_abs"]
        ),
    }
    return flux_new, velocity_new, pressure, diag


def _projection_from_flux_torch(
    predicted_velocity, pressure_guess, context: MeshNativeFlowContext
):
    torch_mod = context.torch_module
    flux_star = compute_face_fluxes(predicted_velocity, context)
    div_star = divergence_from_face_fluxes(flux_star, context)
    rhs_raw = div_star / context.cfg.dt
    rhs, rhs_mean_raw = _compatibility_correct_rhs_torch(rhs_raw, torch_mod)
    pressure, iterations, poisson_res = _solve_poisson_pcg_torch(
        rhs, pressure_guess, context
    )
    flux_correction = _pressure_flux_correction_torch(pressure, context)
    flux_new = flux_star - flux_correction
    flux_new = _apply_flux_projection_closure_torch(flux_new, context)
    div_new_flux = divergence_from_face_fluxes(flux_new, context)
    velocity_reconstructed_raw = _velocity_from_face_fluxes_torch(flux_new, context)
    flux_from_reconstructed_raw = compute_face_fluxes(
        velocity_reconstructed_raw, context
    )
    div_from_reconstructed_raw = divergence_from_face_fluxes(
        flux_from_reconstructed_raw, context
    )
    flux_reconstruction_delta = flux_new - flux_from_reconstructed_raw
    div_reconstruction_delta = div_new_flux - div_from_reconstructed_raw
    velocity_new = velocity_reconstructed_raw.clone()
    outlet_adj_ids = np.flatnonzero(
        context.region_masks.get(
            "outlet_adjacent", np.zeros(context.topology.n_cells, dtype=bool)
        )
    )
    outlet_vy_after_correction = _masked_component_stats(
        _to_numpy_array(velocity_reconstructed_raw, context.backend),
        outlet_adj_ids,
        component=1,
    )
    outlet_vy_after_reconstruction = _masked_component_stats(
        _to_numpy_array(velocity_new, context.backend),
        outlet_adj_ids,
        component=1,
    )
    correction_vec = velocity_new - predicted_velocity

    flux_delta_cell_max = torch_mod.max(
        torch_mod.abs(flux_reconstruction_delta), dim=1
    ).values
    flux_delta_cell_mean = torch_mod.mean(
        torch_mod.abs(flux_reconstruction_delta), dim=1
    )
    flux_reconstruction_region_max = _regional_scalar_abs_stats(
        _to_numpy_array(flux_delta_cell_max, context.backend),
        context.region_masks,
    )
    flux_reconstruction_region_mean = _regional_scalar_abs_stats(
        _to_numpy_array(flux_delta_cell_mean, context.backend),
        context.region_masks,
    )
    div_reconstruction_region = _regional_scalar_abs_stats(
        _to_numpy_array(div_reconstruction_delta, context.backend),
        context.region_masks,
    )
    outlet_flux_stages = _outlet_flux_stage_stats(
        flux_star=_to_numpy_array(flux_star, context.backend),
        flux_corrected=_to_numpy_array(flux_new, context.backend),
        flux_reconstructed=_to_numpy_array(
            flux_from_reconstructed_raw, context.backend
        ),
        flux_final=_to_numpy_array(flux_new, context.backend),
        context=context,
    )
    outlet_near_div_chain = _outlet_near_divergence_chain(
        div_star=_to_numpy_array(div_star, context.backend),
        div_corrected_flux=_to_numpy_array(div_new_flux, context.backend),
        div_reconstructed_flux=_to_numpy_array(
            div_from_reconstructed_raw, context.backend
        ),
        div_final=_to_numpy_array(div_new_flux, context.backend),
        context=context,
    )
    star_max = float(torch_mod.max(torch_mod.abs(div_star)).item())
    new_max = float(torch_mod.max(torch_mod.abs(div_new_flux)).item())
    star_l2 = float(torch_mod.sqrt(torch_mod.mean(div_star * div_star)).item())
    new_l2 = float(torch_mod.sqrt(torch_mod.mean(div_new_flux * div_new_flux)).item())
    flux_max = float(torch_mod.max(torch_mod.abs(div_new_flux)).item())
    flux_l2 = float(torch_mod.sqrt(torch_mod.mean(div_new_flux * div_new_flux)).item())
    region_star = _regional_divergence_stats_numpy(
        _to_numpy_array(div_star, context.backend),
        context.region_masks,
    )
    region_corrected = _regional_divergence_stats_numpy(
        _to_numpy_array(div_new_flux, context.backend),
        context.region_masks,
    )
    bflux_star = _boundary_flux_stats_numpy(
        _to_numpy_array(flux_star, context.backend), context
    )
    bflux_corrected = _boundary_flux_stats_numpy(
        _to_numpy_array(flux_new, context.backend),
        context,
    )
    inlet_mismatch_star = _inlet_flux_mismatch(bflux_star, context.imposed_inlet_flux)
    inlet_mismatch_corrected = _inlet_flux_mismatch(
        bflux_corrected,
        context.imposed_inlet_flux,
    )
    inlet_flux_star = {
        "left_inlet": float(bflux_star["left_inlet"]),
        "right_inlet": float(bflux_star["right_inlet"]),
        "total_inlet": float(bflux_star["left_inlet"] + bflux_star["right_inlet"]),
    }
    inlet_flux_corrected = {
        "left_inlet": float(bflux_corrected["left_inlet"]),
        "right_inlet": float(bflux_corrected["right_inlet"]),
        "total_inlet": float(
            bflux_corrected["left_inlet"] + bflux_corrected["right_inlet"]
        ),
    }
    vol_div_star = float(
        torch_mod.sum(div_star * context.torch_data["cell_volumes"]).item()
    )
    vol_div_corrected = float(
        torch_mod.sum(div_new_flux * context.torch_data["cell_volumes"]).item()
    )
    inlet_layer_star = _inlet_layer_chain_diagnostics(
        velocity=_to_numpy_array(predicted_velocity, context.backend),
        pressure=_to_numpy_array(pressure_guess, context.backend),
        fluxes=_to_numpy_array(flux_star, context.backend),
        divergence=_to_numpy_array(div_star, context.backend),
        context=context,
    )
    inlet_layer_corrected = _inlet_layer_chain_diagnostics(
        velocity=_to_numpy_array(velocity_new, context.backend),
        pressure=_to_numpy_array(pressure, context.backend),
        fluxes=_to_numpy_array(flux_new, context.backend),
        divergence=_to_numpy_array(div_new_flux, context.backend),
        context=context,
    )
    layer_band_star = _layer_by_layer_diagnostics(
        velocity=_to_numpy_array(predicted_velocity, context.backend),
        pressure=_to_numpy_array(pressure_guess, context.backend),
        fluxes=_to_numpy_array(flux_star, context.backend),
        divergence=_to_numpy_array(div_star, context.backend),
        context=context,
    )
    layer_band_corrected = _layer_by_layer_diagnostics(
        velocity=_to_numpy_array(velocity_new, context.backend),
        pressure=_to_numpy_array(pressure, context.backend),
        fluxes=_to_numpy_array(flux_new, context.backend),
        divergence=_to_numpy_array(div_new_flux, context.backend),
        context=context,
    )
    interface_12_star = _interface_12_diagnostics(
        velocity=_to_numpy_array(predicted_velocity, context.backend),
        pressure=_to_numpy_array(pressure_guess, context.backend),
        fluxes=_to_numpy_array(flux_star, context.backend),
        divergence=_to_numpy_array(div_star, context.backend),
        context=context,
        flux_reference=None,
    )
    interface_12_corrected = _interface_12_diagnostics(
        velocity=_to_numpy_array(velocity_new, context.backend),
        pressure=_to_numpy_array(pressure, context.backend),
        fluxes=_to_numpy_array(flux_new, context.backend),
        divergence=_to_numpy_array(div_new_flux, context.backend),
        context=context,
        flux_reference=_to_numpy_array(flux_star, context.backend),
    )
    outlet_band_star = _outlet_band_diagnostics(
        velocity=_to_numpy_array(predicted_velocity, context.backend),
        pressure=_to_numpy_array(pressure_guess, context.backend),
        fluxes=_to_numpy_array(flux_star, context.backend),
        divergence=_to_numpy_array(div_star, context.backend),
        context=context,
        flux_reference=None,
    )
    outlet_band_corrected = _outlet_band_diagnostics(
        velocity=_to_numpy_array(velocity_new, context.backend),
        pressure=_to_numpy_array(pressure, context.backend),
        fluxes=_to_numpy_array(flux_new, context.backend),
        divergence=_to_numpy_array(div_new_flux, context.backend),
        context=context,
        flux_reference=_to_numpy_array(flux_star, context.backend),
    )
    diag = {
        "divergence_star_max_abs": star_max,
        "divergence_star_l2": star_l2,
        "divergence_projected_flux_max_abs": flux_max,
        "divergence_projected_flux_l2": flux_l2,
        "divergence_projected_max_abs": new_max,
        "divergence_projected_l2": new_l2,
        "projection_reduction_factor_max_abs": _safe_ratio(new_max, star_max),
        "projection_reduction_factor_l2": _safe_ratio(new_l2, star_l2),
        "poisson_iterations": int(iterations),
        "poisson_residual_max_abs": float(poisson_res),
        "pressure_solver": "pcg_diag",
        "pressure_gauge_value": float(pressure[context.gauge_index].item()),
        "max_correction_magnitude": float(
            torch_mod.max(torch_mod.linalg.norm(correction_vec, dim=1)).item()
        ),
        "poisson_rhs_mean_raw": float(rhs_mean_raw),
        "poisson_rhs_mean_corrected": float(torch_mod.mean(rhs).item()),
        "poisson_rhs_l2": float(torch_mod.sqrt(torch_mod.mean(rhs * rhs)).item()),
        "projection_chain_flux_reconstruction_mismatch_max_abs": float(
            torch_mod.max(torch_mod.abs(flux_reconstruction_delta)).item()
        ),
        "projection_chain_flux_reconstruction_mismatch_mean_abs": float(
            torch_mod.mean(torch_mod.abs(flux_reconstruction_delta)).item()
        ),
        "projection_chain_divergence_reconstruction_mismatch_max_abs": float(
            torch_mod.max(torch_mod.abs(div_reconstruction_delta)).item()
        ),
        "projection_chain_divergence_reconstruction_mismatch_mean_abs": float(
            torch_mod.mean(torch_mod.abs(div_reconstruction_delta)).item()
        ),
        "projection_chain_flux_reconstruction_mismatch_region_max": flux_reconstruction_region_max,
        "projection_chain_flux_reconstruction_mismatch_region_mean": (
            flux_reconstruction_region_mean
        ),
        "projection_chain_divergence_reconstruction_mismatch_region": div_reconstruction_region,
        "outlet_flux_stages": outlet_flux_stages,
        "outlet_near_divergence_chain": outlet_near_div_chain,
        "outlet_adjacent_vy_after_correction": outlet_vy_after_correction,
        "outlet_adjacent_vy_after_reconstruction": outlet_vy_after_reconstruction,
        "boundary_flux_star": bflux_star,
        "boundary_flux_corrected": bflux_corrected,
        "net_boundary_flux_star": float(bflux_star["net"]),
        "net_boundary_flux_corrected": float(bflux_corrected["net"]),
        "volume_integrated_divergence_star": vol_div_star,
        "volume_integrated_divergence_corrected": vol_div_corrected,
        "regional_divergence_star": region_star,
        "regional_divergence_corrected": region_corrected,
        "imposed_inlet_flux": dict(context.imposed_inlet_flux),
        "inlet_flux_star": inlet_flux_star,
        "inlet_flux_corrected": inlet_flux_corrected,
        "inlet_flux_mismatch_star": inlet_mismatch_star,
        "inlet_flux_mismatch_corrected": inlet_mismatch_corrected,
        "inlet_zone_divergence_star": _inlet_zone_stats(region_star),
        "inlet_zone_divergence_corrected": _inlet_zone_stats(region_corrected),
        "inlet_layer_chain_star": inlet_layer_star,
        "inlet_layer_chain_corrected": inlet_layer_corrected,
        "layer_band_star": layer_band_star,
        "layer_band_corrected": layer_band_corrected,
        "interface_12_star": interface_12_star,
        "interface_12_corrected": interface_12_corrected,
        "outlet_band_star": outlet_band_star,
        "outlet_band_corrected": outlet_band_corrected,
        "first_interior_layer_flux_mismatch_star_max_abs": float(
            inlet_layer_star["first_interior_layer_flux_mismatch_max_abs"]
        ),
        "first_interior_layer_flux_mismatch_corrected_max_abs": float(
            inlet_layer_corrected["first_interior_layer_flux_mismatch_max_abs"]
        ),
        "inlet_face_flux_mismatch_star_max_abs": float(
            inlet_layer_star["inlet_face_flux_mismatch_max_abs"]
        ),
        "inlet_face_flux_mismatch_corrected_max_abs": float(
            inlet_layer_corrected["inlet_face_flux_mismatch_max_abs"]
        ),
        "interface_12_flux_max": float(
            interface_12_corrected["interface_12_flux"]["max_abs"]
        ),
        "interface_12_flux_mean": float(
            interface_12_corrected["interface_12_flux"]["mean_abs"]
        ),
        "interface_12_flux_correction_max": float(
            interface_12_corrected["interface_12_flux_correction"]["max_abs"]
        ),
        "interface_12_flux_correction_mean": float(
            interface_12_corrected["interface_12_flux_correction"]["mean_abs"]
        ),
        "interface_12_flux_mismatch_max": float(
            interface_12_corrected["interface_12_flux_mismatch"]["max_abs"]
        ),
        "interface_12_flux_mismatch_mean": float(
            interface_12_corrected["interface_12_flux_mismatch"]["mean_abs"]
        ),
        "layer2_div_star_max": float(interface_12_star["layer2_divergence"]["max_abs"]),
        "layer2_div_star_mean": float(
            interface_12_star["layer2_divergence"]["mean_abs"]
        ),
        "layer2_div_corrected_max": float(
            interface_12_corrected["layer2_divergence"]["max_abs"]
        ),
        "layer2_div_corrected_mean": float(
            interface_12_corrected["layer2_divergence"]["mean_abs"]
        ),
        "layer2_pressure_gradient_max": float(
            interface_12_corrected["layer2_pressure_gradient"]["max_abs"]
        ),
        "layer2_pressure_gradient_mean": float(
            interface_12_corrected["layer2_pressure_gradient"]["mean_abs"]
        ),
        "layer2_velocity_jump_max": float(
            interface_12_corrected["layer2_velocity_jump"]["max_abs"]
        ),
        "layer2_velocity_jump_mean": float(
            interface_12_corrected["layer2_velocity_jump"]["mean_abs"]
        ),
        "layer2_flux_response_max": float(
            interface_12_corrected["interface_12_flux_correction"]["max_abs"]
        ),
        "layer2_flux_response_mean": float(
            interface_12_corrected["interface_12_flux_correction"]["mean_abs"]
        ),
        "outlet_layer0_div_star_max": float(
            outlet_band_star["outlet_layer0_divergence"]["max_abs"]
        ),
        "outlet_layer0_div_star_mean": float(
            outlet_band_star["outlet_layer0_divergence"]["mean_abs"]
        ),
        "outlet_layer0_div_corrected_max": float(
            outlet_band_corrected["outlet_layer0_divergence"]["max_abs"]
        ),
        "outlet_layer0_div_corrected_mean": float(
            outlet_band_corrected["outlet_layer0_divergence"]["mean_abs"]
        ),
        "outlet_layer1_div_star_max": float(
            outlet_band_star["outlet_layer1_divergence"]["max_abs"]
        ),
        "outlet_layer1_div_star_mean": float(
            outlet_band_star["outlet_layer1_divergence"]["mean_abs"]
        ),
        "outlet_layer1_div_corrected_max": float(
            outlet_band_corrected["outlet_layer1_divergence"]["max_abs"]
        ),
        "outlet_layer1_div_corrected_mean": float(
            outlet_band_corrected["outlet_layer1_divergence"]["mean_abs"]
        ),
        "outlet_layer2_div_star_max": float(
            outlet_band_star["outlet_layer2_divergence"]["max_abs"]
        ),
        "outlet_layer2_div_star_mean": float(
            outlet_band_star["outlet_layer2_divergence"]["mean_abs"]
        ),
        "outlet_layer2_div_corrected_max": float(
            outlet_band_corrected["outlet_layer2_divergence"]["max_abs"]
        ),
        "outlet_layer2_div_corrected_mean": float(
            outlet_band_corrected["outlet_layer2_divergence"]["mean_abs"]
        ),
        "outlet_pressure_gradient_max": float(
            outlet_band_corrected["outlet_pressure_gradient"]["max_abs"]
        ),
        "outlet_pressure_gradient_mean": float(
            outlet_band_corrected["outlet_pressure_gradient"]["mean_abs"]
        ),
        "outlet_velocity_jump_max": float(
            outlet_band_corrected["outlet_velocity_jump"]["max_abs"]
        ),
        "outlet_velocity_jump_mean": float(
            outlet_band_corrected["outlet_velocity_jump"]["mean_abs"]
        ),
        "outlet_flux_mismatch_max": float(
            outlet_band_corrected["outlet_flux_mismatch"]["max_abs"]
        ),
        "outlet_flux_mismatch_mean": float(
            outlet_band_corrected["outlet_flux_mismatch"]["mean_abs"]
        ),
        "outlet_interface_flux_max": float(
            outlet_band_corrected["outlet_interface_flux"]["max_abs"]
        ),
        "outlet_interface_flux_mean": float(
            outlet_band_corrected["outlet_interface_flux"]["mean_abs"]
        ),
        "outlet_interface_flux_correction_max": float(
            outlet_band_corrected["outlet_interface_flux_correction"]["max_abs"]
        ),
        "outlet_interface_flux_correction_mean": float(
            outlet_band_corrected["outlet_interface_flux_correction"]["mean_abs"]
        ),
        "outlet_interface_flux_mismatch_max": float(
            outlet_band_corrected["outlet_interface_flux_mismatch"]["max_abs"]
        ),
        "outlet_interface_flux_mismatch_mean": float(
            outlet_band_corrected["outlet_interface_flux_mismatch"]["mean_abs"]
        ),
    }
    return flux_new, velocity_new, pressure, diag


def _is_finite_dict(diag: dict) -> bool:
    for value in diag.values():
        if isinstance(value, (float, int)) and not np.isfinite(float(value)):
            return False
    return True


def _safeguard_stop_reason(diag: dict, cfg: MeshNativeFlowConfig) -> str | None:
    if not _is_finite_dict(diag):
        return "non-finite diagnostics detected"
    if abs(float(diag["max_velocity_magnitude"])) > cfg.blowup_abs_limit:
        return "velocity magnitude exceeded blowup_abs_limit"
    if abs(float(diag["pressure_max"])) > cfg.blowup_abs_limit:
        return "pressure_max exceeded blowup_abs_limit"
    if abs(float(diag["pressure_min"])) > cfg.blowup_abs_limit:
        return "pressure_min exceeded blowup_abs_limit"
    if abs(float(diag["poisson_residual_max_abs"])) > cfg.max_poisson_residual:
        return "poisson_residual_max_abs exceeded max_poisson_residual"
    if abs(float(diag["divergence_final_max_abs"])) > cfg.max_divergence:
        return "divergence_final_max_abs exceeded max_divergence"
    return None


def advance_mesh_native_flow_step(
    state: MeshNativeFlowState,
    context: MeshNativeFlowContext,
) -> MeshNativeFlowState:
    if context.backend.startswith("torch"):
        torch_mod = context.torch_module
        if isinstance(state.velocity, np.ndarray):
            velocity = torch_mod.tensor(
                state.velocity,
                dtype=torch_mod.float64,
                device=context.torch_data["device"],
            )
            pressure = torch_mod.tensor(
                state.pressure,
                dtype=torch_mod.float64,
                device=context.torch_data["device"],
            )
        else:
            velocity = state.velocity
            pressure = state.pressure
        velocity = _apply_velocity_bc_torch(
            velocity,
            context,
            enforce_inlets=True,
            enforce_outlet=True,
            enforce_walls=True,
        )
        if context.cfg.viscosity > 0:
            velocity_star = (
                velocity
                + context.cfg.dt
                * context.cfg.viscosity
                * _laplace_vector_torch(velocity, context)
            )
        else:
            velocity_star = velocity.clone()
        velocity_star = _apply_velocity_bc_torch(
            velocity_star,
            context,
            enforce_inlets=True,
            enforce_outlet=True,
            enforce_walls=True,
        )
        outlet_adj_ids = np.flatnonzero(
            context.region_masks.get(
                "outlet_adjacent", np.zeros(context.topology.n_cells, dtype=bool)
            )
        )
        outlet_vy_before_correction = _masked_component_stats(
            _to_numpy_array(velocity_star, context.backend),
            outlet_adj_ids,
            component=1,
        )
        corrected_flux, corrected_velocity, pressure_new, proj_diag = (
            _projection_from_flux_torch(
                predicted_velocity=velocity_star,
                pressure_guess=pressure,
                context=context,
            )
        )
        divergence_projected = proj_diag["divergence_projected_max_abs"]
        state_velocity = corrected_velocity
        if (
            context.cfg.post_projection_enforce_inlets
            or context.cfg.post_projection_enforce_outlet
            or context.cfg.post_projection_enforce_walls
        ):
            state_velocity = _apply_velocity_bc_torch(
                state_velocity,
                context,
                enforce_inlets=context.cfg.post_projection_enforce_inlets,
                enforce_outlet=context.cfg.post_projection_enforce_outlet,
                enforce_walls=context.cfg.post_projection_enforce_walls,
            )
        pressure_new[context.gauge_index] = 0.0
        fluxes_post_bc = compute_face_fluxes(state_velocity, context)
        bflux_post_bc = _boundary_flux_stats_numpy(
            _to_numpy_array(fluxes_post_bc, context.backend),
            context,
        )
        fluxes_final = corrected_flux
        divergence_final = divergence_from_face_fluxes(fluxes_final, context)
        bc = context.torch_data["boundary_cells"]
        wall_ids = bc["wall"]
        if int(wall_ids.numel()) > 0:
            wall_speed = torch_mod.linalg.norm(state_velocity[wall_ids], dim=1)
            wall_speed_max = float(torch_mod.max(wall_speed).item())
        else:
            wall_speed_max = 0.0
        div_max = float(torch_mod.max(torch_mod.abs(divergence_final)).item())
        div_l2 = float(
            torch_mod.sqrt(torch_mod.mean(divergence_final * divergence_final)).item()
        )
        region_final = _regional_divergence_stats_numpy(
            _to_numpy_array(divergence_final, context.backend),
            context.region_masks,
        )
        bflux_final = _boundary_flux_stats_numpy(
            _to_numpy_array(fluxes_final, context.backend),
            context,
        )
        vol_div_final = float(
            torch_mod.sum(divergence_final * context.torch_data["cell_volumes"]).item()
        )
        region_corrected = proj_diag.get("regional_divergence_corrected", {})
        region_post_bc_delta: dict[str, dict[str, float]] = {}
        for name, stats in region_final.items():
            corr = region_corrected.get(name, {"max_abs": 0.0, "mean_abs": 0.0})
            region_post_bc_delta[name] = {
                "max_abs_delta": float(
                    stats["max_abs"] - float(corr.get("max_abs", 0.0))
                ),
                "mean_abs_delta": float(
                    stats["mean_abs"] - float(corr.get("mean_abs", 0.0))
                ),
            }
        inlet_flux_final = {
            "left_inlet": float(bflux_final["left_inlet"]),
            "right_inlet": float(bflux_final["right_inlet"]),
            "total_inlet": float(
                bflux_final["left_inlet"] + bflux_final["right_inlet"]
            ),
        }
        inlet_mismatch_final = _inlet_flux_mismatch(
            bflux_final, context.imposed_inlet_flux
        )
        outlet_vy_final_state = _masked_component_stats(
            _to_numpy_array(state_velocity, context.backend),
            outlet_adj_ids,
            component=1,
        )
        inlet_layer_final = _inlet_layer_chain_diagnostics(
            velocity=_to_numpy_array(state_velocity, context.backend),
            pressure=_to_numpy_array(pressure_new, context.backend),
            fluxes=_to_numpy_array(fluxes_final, context.backend),
            divergence=_to_numpy_array(divergence_final, context.backend),
            context=context,
        )
        layer_band_final = _layer_by_layer_diagnostics(
            velocity=_to_numpy_array(state_velocity, context.backend),
            pressure=_to_numpy_array(pressure_new, context.backend),
            fluxes=_to_numpy_array(fluxes_final, context.backend),
            divergence=_to_numpy_array(divergence_final, context.backend),
            context=context,
        )
        interface_12_final = _interface_12_diagnostics(
            velocity=_to_numpy_array(state_velocity, context.backend),
            pressure=_to_numpy_array(pressure_new, context.backend),
            fluxes=_to_numpy_array(fluxes_final, context.backend),
            divergence=_to_numpy_array(divergence_final, context.backend),
            context=context,
            flux_reference=None,
        )
        outlet_band_final = _outlet_band_diagnostics(
            velocity=_to_numpy_array(state_velocity, context.backend),
            pressure=_to_numpy_array(pressure_new, context.backend),
            fluxes=_to_numpy_array(fluxes_final, context.backend),
            divergence=_to_numpy_array(divergence_final, context.backend),
            context=context,
            flux_reference=None,
        )
        diagnostics = {
            **proj_diag,
            "divergence_projected_max_abs_before_post_bc": float(divergence_projected),
            "divergence_final_max_abs": div_max,
            "divergence_final_l2": div_l2,
            "projection_reduction_factor_max_abs_post_bc": _safe_ratio(
                div_max,
                float(proj_diag["divergence_star_max_abs"]),
            ),
            "projection_reduction_factor_l2_post_bc": _safe_ratio(
                div_l2,
                float(proj_diag["divergence_star_l2"]),
            ),
            "post_bc_divergence_delta_max_abs": float(
                div_max - float(divergence_projected)
            ),
            "post_bc_divergence_delta_l2": float(
                div_l2 - float(proj_diag["divergence_projected_l2"])
            ),
            "boundary_flux_final": bflux_final,
            "net_boundary_flux_final": float(bflux_final["net"]),
            "volume_integrated_divergence_final": vol_div_final,
            "inlet_flux_final": inlet_flux_final,
            "inlet_flux_mismatch_final": inlet_mismatch_final,
            "outlet_flux_predictor": float(
                proj_diag.get("boundary_flux_star", {}).get("outlet", float("nan"))
            ),
            "outlet_flux_corrected": float(
                proj_diag.get("boundary_flux_corrected", {}).get("outlet", float("nan"))
            ),
            "outlet_flux_post_bc": float(bflux_post_bc.get("outlet", float("nan"))),
            "outlet_flux_reconstructed": float(
                proj_diag.get("outlet_flux_stages", {}).get(
                    "outlet_flux_reconstructed", float("nan")
                )
            ),
            "outlet_flux_stored_state": float(bflux_final.get("outlet", float("nan"))),
            "outlet_adjacent_vy_before_correction": outlet_vy_before_correction,
            "outlet_adjacent_vy_after_correction": proj_diag.get(
                "outlet_adjacent_vy_after_correction",
                {},
            ),
            "outlet_adjacent_vy_after_reconstruction": proj_diag.get(
                "outlet_adjacent_vy_after_reconstruction",
                {},
            ),
            "outlet_adjacent_vy_after_final_state": outlet_vy_final_state,
            "inlet_layer_chain_final": inlet_layer_final,
            "layer_band_final": layer_band_final,
            "interface_12_final": interface_12_final,
            "outlet_band_final": outlet_band_final,
            "first_interior_layer_flux_mismatch_final_max_abs": float(
                inlet_layer_final["first_interior_layer_flux_mismatch_max_abs"]
            ),
            "inlet_face_flux_mismatch_final_max_abs": float(
                inlet_layer_final["inlet_face_flux_mismatch_max_abs"]
            ),
            "interface_12_flux_final_max": float(
                interface_12_final["interface_12_flux"]["max_abs"]
            ),
            "interface_12_flux_final_mean": float(
                interface_12_final["interface_12_flux"]["mean_abs"]
            ),
            "interface_12_flux_mismatch_final_max": float(
                interface_12_final["interface_12_flux_mismatch"]["max_abs"]
            ),
            "interface_12_flux_mismatch_final_mean": float(
                interface_12_final["interface_12_flux_mismatch"]["mean_abs"]
            ),
            "layer2_div_final_max": float(
                interface_12_final["layer2_divergence"]["max_abs"]
            ),
            "layer2_div_final_mean": float(
                interface_12_final["layer2_divergence"]["mean_abs"]
            ),
            "outlet_layer0_div_final_max": float(
                outlet_band_final["outlet_layer0_divergence"]["max_abs"]
            ),
            "outlet_layer0_div_final_mean": float(
                outlet_band_final["outlet_layer0_divergence"]["mean_abs"]
            ),
            "outlet_layer1_div_final_max": float(
                outlet_band_final["outlet_layer1_divergence"]["max_abs"]
            ),
            "outlet_layer1_div_final_mean": float(
                outlet_band_final["outlet_layer1_divergence"]["mean_abs"]
            ),
            "outlet_layer2_div_final_max": float(
                outlet_band_final["outlet_layer2_divergence"]["max_abs"]
            ),
            "outlet_layer2_div_final_mean": float(
                outlet_band_final["outlet_layer2_divergence"]["mean_abs"]
            ),
            "outlet_pressure_gradient_final_max": float(
                outlet_band_final["outlet_pressure_gradient"]["max_abs"]
            ),
            "outlet_pressure_gradient_final_mean": float(
                outlet_band_final["outlet_pressure_gradient"]["mean_abs"]
            ),
            "outlet_velocity_jump_final_max": float(
                outlet_band_final["outlet_velocity_jump"]["max_abs"]
            ),
            "outlet_velocity_jump_final_mean": float(
                outlet_band_final["outlet_velocity_jump"]["mean_abs"]
            ),
            "outlet_flux_mismatch_final_max": float(
                outlet_band_final["outlet_flux_mismatch"]["max_abs"]
            ),
            "outlet_flux_mismatch_final_mean": float(
                outlet_band_final["outlet_flux_mismatch"]["mean_abs"]
            ),
            "outlet_interface_flux_final_max": float(
                outlet_band_final["outlet_interface_flux"]["max_abs"]
            ),
            "outlet_interface_flux_final_mean": float(
                outlet_band_final["outlet_interface_flux"]["mean_abs"]
            ),
            "outlet_interface_flux_mismatch_final_max": float(
                outlet_band_final["outlet_interface_flux_mismatch"]["max_abs"]
            ),
            "outlet_interface_flux_mismatch_final_mean": float(
                outlet_band_final["outlet_interface_flux_mismatch"]["mean_abs"]
            ),
            "flux_divergence_final_max_abs": div_max,
            "max_velocity_magnitude": float(
                torch_mod.max(torch_mod.linalg.norm(state_velocity, dim=1)).item()
            ),
            "pressure_min": float(torch_mod.min(pressure_new).item()),
            "pressure_max": float(torch_mod.max(pressure_new).item()),
            "wall_speed_max": wall_speed_max,
            "projection_warning_non_reducing_raw": bool(
                float(proj_diag["projection_reduction_factor_max_abs"]) >= 1.0
                or float(proj_diag["projection_reduction_factor_l2"]) >= 1.0
            ),
            "projection_warning_non_reducing": bool(
                _safe_ratio(div_max, float(proj_diag["divergence_star_max_abs"])) >= 1.0
                or _safe_ratio(div_l2, float(proj_diag["divergence_star_l2"])) >= 1.0
            ),
            "regional_divergence_final": region_final,
            "regional_divergence_post_bc_delta": region_post_bc_delta,
            "outlet_near_divergence_corrected_max_abs": float(
                region_corrected.get("outlet_near_zone", {}).get("max_abs", 0.0)
            ),
            "outlet_near_divergence_corrected_mean_abs": float(
                region_corrected.get("outlet_near_zone", {}).get("mean_abs", 0.0)
            ),
            "outlet_near_divergence_final_max_abs": float(
                region_final.get("outlet_near_zone", {}).get("max_abs", 0.0)
            ),
            "outlet_near_divergence_final_mean_abs": float(
                region_final.get("outlet_near_zone", {}).get("mean_abs", 0.0)
            ),
            "outlet_near_divergence_post_bc_delta_max_abs": float(
                region_post_bc_delta.get("outlet_near_zone", {}).get(
                    "max_abs_delta", 0.0
                )
            ),
            "outlet_near_divergence_post_bc_delta_mean_abs": float(
                region_post_bc_delta.get("outlet_near_zone", {}).get(
                    "mean_abs_delta", 0.0
                )
            ),
            "inlet_zone_divergence_final": _inlet_zone_stats(region_final),
        }
        history = list(state.diagnostics_history or [])
        history.append(diagnostics)
        return MeshNativeFlowState(
            mesh=state.mesh,
            velocity=state_velocity,
            pressure=pressure_new,
            face_flux=fluxes_final,
            backend=context.backend,
            step=state.step + 1,
            diagnostics_history=history,
        )

    velocity, pressure = state.to_numpy()
    velocity = _apply_velocity_bc_numpy(
        velocity,
        context,
        enforce_inlets=True,
        enforce_outlet=True,
        enforce_walls=True,
    )
    if context.cfg.viscosity > 0:
        velocity_star = (
            velocity
            + context.cfg.dt
            * context.cfg.viscosity
            * _laplace_vector_numpy(
                velocity,
                context.topology,
            )
        )
    else:
        velocity_star = velocity.copy()
    velocity_star = _apply_velocity_bc_numpy(
        velocity_star,
        context,
        enforce_inlets=True,
        enforce_outlet=True,
        enforce_walls=True,
    )
    corrected_flux, corrected_velocity, pressure_new, proj_diag = (
        _projection_from_flux_numpy(
            predicted_velocity=velocity_star,
            pressure_guess=pressure,
            context=context,
        )
    )
    divergence_projected = proj_diag["divergence_projected_max_abs"]
    state_velocity = corrected_velocity
    if (
        context.cfg.post_projection_enforce_inlets
        or context.cfg.post_projection_enforce_outlet
        or context.cfg.post_projection_enforce_walls
    ):
        state_velocity = _apply_velocity_bc_numpy(
            state_velocity,
            context,
            enforce_inlets=context.cfg.post_projection_enforce_inlets,
            enforce_outlet=context.cfg.post_projection_enforce_outlet,
            enforce_walls=context.cfg.post_projection_enforce_walls,
        )
    pressure_new[context.gauge_index] = 0.0
    fluxes_final = corrected_flux
    divergence_final = divergence_from_face_fluxes(fluxes_final, context)
    wall_ids = context.boundary_cells["wall"]
    if wall_ids.size > 0:
        wall_speed_max = float(np.max(np.linalg.norm(state_velocity[wall_ids], axis=1)))
    else:
        wall_speed_max = 0.0
    div_max = float(np.max(np.abs(divergence_final)))
    div_l2 = float(np.sqrt(np.mean(divergence_final * divergence_final)))
    region_final = _regional_divergence_stats_numpy(
        divergence_final, context.region_masks
    )
    bflux_final = _boundary_flux_stats_numpy(fluxes_final, context)
    vol_div_final = float(np.sum(divergence_final * context.cell_volumes))
    region_corrected = proj_diag.get("regional_divergence_corrected", {})
    region_post_bc_delta: dict[str, dict[str, float]] = {}
    for name, stats in region_final.items():
        corr = region_corrected.get(name, {"max_abs": 0.0, "mean_abs": 0.0})
        region_post_bc_delta[name] = {
            "max_abs_delta": float(stats["max_abs"] - float(corr.get("max_abs", 0.0))),
            "mean_abs_delta": float(
                stats["mean_abs"] - float(corr.get("mean_abs", 0.0))
            ),
        }
    inlet_flux_final = {
        "left_inlet": float(bflux_final["left_inlet"]),
        "right_inlet": float(bflux_final["right_inlet"]),
        "total_inlet": float(bflux_final["left_inlet"] + bflux_final["right_inlet"]),
    }
    inlet_mismatch_final = _inlet_flux_mismatch(bflux_final, context.imposed_inlet_flux)
    inlet_layer_final = _inlet_layer_chain_diagnostics(
        velocity=state_velocity,
        pressure=pressure_new,
        fluxes=fluxes_final,
        divergence=divergence_final,
        context=context,
    )
    layer_band_final = _layer_by_layer_diagnostics(
        velocity=state_velocity,
        pressure=pressure_new,
        fluxes=fluxes_final,
        divergence=divergence_final,
        context=context,
    )
    interface_12_final = _interface_12_diagnostics(
        velocity=state_velocity,
        pressure=pressure_new,
        fluxes=fluxes_final,
        divergence=divergence_final,
        context=context,
        flux_reference=None,
    )
    outlet_band_final = _outlet_band_diagnostics(
        velocity=state_velocity,
        pressure=pressure_new,
        fluxes=fluxes_final,
        divergence=divergence_final,
        context=context,
        flux_reference=None,
    )
    diagnostics = {
        **proj_diag,
        "divergence_projected_max_abs_before_post_bc": float(divergence_projected),
        "divergence_final_max_abs": div_max,
        "divergence_final_l2": div_l2,
        "projection_reduction_factor_max_abs_post_bc": _safe_ratio(
            div_max,
            float(proj_diag["divergence_star_max_abs"]),
        ),
        "projection_reduction_factor_l2_post_bc": _safe_ratio(
            div_l2,
            float(proj_diag["divergence_star_l2"]),
        ),
        "post_bc_divergence_delta_max_abs": float(
            div_max - float(divergence_projected)
        ),
        "post_bc_divergence_delta_l2": float(
            div_l2 - float(proj_diag["divergence_projected_l2"])
        ),
        "boundary_flux_final": bflux_final,
        "net_boundary_flux_final": float(bflux_final["net"]),
        "volume_integrated_divergence_final": vol_div_final,
        "inlet_flux_final": inlet_flux_final,
        "inlet_flux_mismatch_final": inlet_mismatch_final,
        "inlet_layer_chain_final": inlet_layer_final,
        "layer_band_final": layer_band_final,
        "interface_12_final": interface_12_final,
        "outlet_band_final": outlet_band_final,
        "first_interior_layer_flux_mismatch_final_max_abs": float(
            inlet_layer_final["first_interior_layer_flux_mismatch_max_abs"]
        ),
        "inlet_face_flux_mismatch_final_max_abs": float(
            inlet_layer_final["inlet_face_flux_mismatch_max_abs"]
        ),
        "interface_12_flux_final_max": float(
            interface_12_final["interface_12_flux"]["max_abs"]
        ),
        "interface_12_flux_final_mean": float(
            interface_12_final["interface_12_flux"]["mean_abs"]
        ),
        "interface_12_flux_mismatch_final_max": float(
            interface_12_final["interface_12_flux_mismatch"]["max_abs"]
        ),
        "interface_12_flux_mismatch_final_mean": float(
            interface_12_final["interface_12_flux_mismatch"]["mean_abs"]
        ),
        "layer2_div_final_max": float(
            interface_12_final["layer2_divergence"]["max_abs"]
        ),
        "layer2_div_final_mean": float(
            interface_12_final["layer2_divergence"]["mean_abs"]
        ),
        "outlet_layer0_div_final_max": float(
            outlet_band_final["outlet_layer0_divergence"]["max_abs"]
        ),
        "outlet_layer0_div_final_mean": float(
            outlet_band_final["outlet_layer0_divergence"]["mean_abs"]
        ),
        "outlet_layer1_div_final_max": float(
            outlet_band_final["outlet_layer1_divergence"]["max_abs"]
        ),
        "outlet_layer1_div_final_mean": float(
            outlet_band_final["outlet_layer1_divergence"]["mean_abs"]
        ),
        "outlet_layer2_div_final_max": float(
            outlet_band_final["outlet_layer2_divergence"]["max_abs"]
        ),
        "outlet_layer2_div_final_mean": float(
            outlet_band_final["outlet_layer2_divergence"]["mean_abs"]
        ),
        "outlet_pressure_gradient_final_max": float(
            outlet_band_final["outlet_pressure_gradient"]["max_abs"]
        ),
        "outlet_pressure_gradient_final_mean": float(
            outlet_band_final["outlet_pressure_gradient"]["mean_abs"]
        ),
        "outlet_velocity_jump_final_max": float(
            outlet_band_final["outlet_velocity_jump"]["max_abs"]
        ),
        "outlet_velocity_jump_final_mean": float(
            outlet_band_final["outlet_velocity_jump"]["mean_abs"]
        ),
        "outlet_flux_mismatch_final_max": float(
            outlet_band_final["outlet_flux_mismatch"]["max_abs"]
        ),
        "outlet_flux_mismatch_final_mean": float(
            outlet_band_final["outlet_flux_mismatch"]["mean_abs"]
        ),
        "outlet_interface_flux_final_max": float(
            outlet_band_final["outlet_interface_flux"]["max_abs"]
        ),
        "outlet_interface_flux_final_mean": float(
            outlet_band_final["outlet_interface_flux"]["mean_abs"]
        ),
        "outlet_interface_flux_mismatch_final_max": float(
            outlet_band_final["outlet_interface_flux_mismatch"]["max_abs"]
        ),
        "outlet_interface_flux_mismatch_final_mean": float(
            outlet_band_final["outlet_interface_flux_mismatch"]["mean_abs"]
        ),
        "flux_divergence_final_max_abs": div_max,
        "max_velocity_magnitude": float(np.max(np.linalg.norm(state_velocity, axis=1))),
        "pressure_min": float(np.min(pressure_new)),
        "pressure_max": float(np.max(pressure_new)),
        "wall_speed_max": wall_speed_max,
        "projection_warning_non_reducing_raw": bool(
            float(proj_diag["projection_reduction_factor_max_abs"]) >= 1.0
            or float(proj_diag["projection_reduction_factor_l2"]) >= 1.0
        ),
        "projection_warning_non_reducing": bool(
            _safe_ratio(div_max, float(proj_diag["divergence_star_max_abs"])) >= 1.0
            or _safe_ratio(div_l2, float(proj_diag["divergence_star_l2"])) >= 1.0
        ),
        "regional_divergence_final": region_final,
        "regional_divergence_post_bc_delta": region_post_bc_delta,
        "outlet_near_divergence_corrected_max_abs": float(
            region_corrected.get("outlet_near_zone", {}).get("max_abs", 0.0)
        ),
        "outlet_near_divergence_corrected_mean_abs": float(
            region_corrected.get("outlet_near_zone", {}).get("mean_abs", 0.0)
        ),
        "outlet_near_divergence_final_max_abs": float(
            region_final.get("outlet_near_zone", {}).get("max_abs", 0.0)
        ),
        "outlet_near_divergence_final_mean_abs": float(
            region_final.get("outlet_near_zone", {}).get("mean_abs", 0.0)
        ),
        "outlet_near_divergence_post_bc_delta_max_abs": float(
            region_post_bc_delta.get("outlet_near_zone", {}).get("max_abs_delta", 0.0)
        ),
        "outlet_near_divergence_post_bc_delta_mean_abs": float(
            region_post_bc_delta.get("outlet_near_zone", {}).get("mean_abs_delta", 0.0)
        ),
        "inlet_zone_divergence_final": _inlet_zone_stats(region_final),
    }
    history = list(state.diagnostics_history or [])
    history.append(diagnostics)
    return MeshNativeFlowState(
        mesh=state.mesh,
        velocity=state_velocity,
        pressure=pressure_new,
        face_flux=fluxes_final,
        backend="numpy",
        step=state.step + 1,
        diagnostics_history=history,
    )


def run_mesh_native_flow(
    mesh: MeshDataModel,
    geometry: MeshGeometryModel3D,
    cfg: MeshNativeFlowConfig,
) -> dict:
    """Run minimal mesh-native incompressible flow-only baseline."""
    context = build_mesh_native_flow_context(mesh, geometry, cfg)
    if cfg.warn_on_cpu and not context.backend.startswith("torch-cuda"):
        print(
            "[native-flow] WARNING: running without CUDA backend; "
            f"selected backend={context.backend}, device={context.device}, "
            f"estimated_Re={context.estimated_reynolds:.3f}"
        )
    else:
        print(
            f"[native-flow] backend={context.backend}, device={context.device}, "
            f"estimated_Re={context.estimated_reynolds:.3f}"
        )

    operator_consistency = projection_operator_consistency_diagnostics(context)
    print(
        "[native-flow] operator consistency | laplace(p)-div(grad(p)) | "
        f"max={operator_consistency['max_abs_diff']:.3e}, "
        f"mean={operator_consistency['mean_abs_diff']:.3e}"
    )

    state = initialize_mesh_native_flow_state(context)
    initial_flux = (
        state.face_flux
        if state.face_flux is not None
        else compute_face_fluxes(state.velocity, context)
    )
    initial_div = divergence_from_face_fluxes(initial_flux, context)
    if context.backend.startswith("torch"):
        initial_div_max = float(
            context.torch_module.max(context.torch_module.abs(initial_div)).item()
        )
    else:
        initial_div_max = float(np.max(np.abs(initial_div)))

    stop_reason = None
    for step in range(cfg.steps):
        state = advance_mesh_native_flow_step(state, context)
        diag = state.diagnostics_history[-1]
        diag["step"] = int(step + 1)
        reg = diag.get("regional_divergence_final", {})
        reg_max_short = (
            "reg_max[in_adj/jn/out_adj/bulk]="
            f"{reg.get('inlet_adjacent', {}).get('max_abs', float('nan')):.2e}/"
            f"{reg.get('junction_zone', {}).get('max_abs', float('nan')):.2e}/"
            f"{reg.get('outlet_adjacent', {}).get('max_abs', float('nan')):.2e}/"
            f"{reg.get('bulk_vertical_branch', {}).get('max_abs', float('nan')):.2e}"
        )
        reg_mean_short = (
            "reg_mean[in_adj/jn/out_adj/bulk]="
            f"{reg.get('inlet_adjacent', {}).get('mean_abs', float('nan')):.2e}/"
            f"{reg.get('junction_zone', {}).get('mean_abs', float('nan')):.2e}/"
            f"{reg.get('outlet_adjacent', {}).get('mean_abs', float('nan')):.2e}/"
            f"{reg.get('bulk_vertical_branch', {}).get('mean_abs', float('nan')):.2e}"
        )
        bstar = diag.get("boundary_flux_star", {})
        bcorr = diag.get("boundary_flux_corrected", {})
        bfin = diag.get("boundary_flux_final", {})
        mstar = diag.get("inlet_flux_mismatch_star", {})
        mcorr = diag.get("inlet_flux_mismatch_corrected", {})
        mfin = diag.get("inlet_flux_mismatch_final", {})
        flux_short = (
            "net_flux[star/corr/final]="
            f"{bstar.get('net', float('nan')):.2e}/"
            f"{bcorr.get('net', float('nan')):.2e}/"
            f"{bfin.get('net', float('nan')):.2e}"
        )
        inlet_mismatch_short = (
            "inlet_mis[star/corr/final]="
            f"{mstar.get('max_abs_delta', float('nan')):.2e}/"
            f"{mcorr.get('max_abs_delta', float('nan')):.2e}/"
            f"{mfin.get('max_abs_delta', float('nan')):.2e}"
        )
        layer_final = diag.get("inlet_layer_chain_final", {})
        layer_star = diag.get("inlet_layer_chain_star", {})
        layer_corr = diag.get("inlet_layer_chain_corrected", {})
        band_star = diag.get("layer_band_star", {})
        band_corr = diag.get("layer_band_corrected", {})
        band_final = diag.get("layer_band_final", {})
        in_adj_max = layer_final.get("inlet_face_adjacent_divergence", {}).get(
            "max_abs", float("nan")
        )
        first_max = layer_final.get("first_interior_layer_divergence", {}).get(
            "max_abs", float("nan")
        )
        second_max = layer_final.get("second_interior_layer_divergence", {}).get(
            "max_abs", float("nan")
        )
        jump_max = layer_final.get("inlet_to_first_layer_velocity_jump", {}).get(
            "max",
            float("nan"),
        )
        layer_flux_short = (
            "layer_flux_mis[star/corr/final]="
            f"{layer_star.get('first_interior_layer_flux_mismatch_max_abs', float('nan')):.2e}/"
            f"{layer_corr.get('first_interior_layer_flux_mismatch_max_abs', float('nan')):.2e}/"
            f"{layer_final.get('first_interior_layer_flux_mismatch_max_abs', float('nan')):.2e}"
        )
        layer_div_short = f"layer_div[in_adj/first/second]={in_adj_max:.2e}/{first_max:.2e}/{second_max:.2e}"
        layer_jump_short = f"layer_jump_max={jump_max:.2e}"
        l0s = (
            band_star.get("layer0_inlet_adjacent", {})
            .get("divergence", {})
            .get(
                "max_abs",
                float("nan"),
            )
        )
        l0c = (
            band_corr.get("layer0_inlet_adjacent", {})
            .get("divergence", {})
            .get(
                "max_abs",
                float("nan"),
            )
        )
        l0f = (
            band_final.get("layer0_inlet_adjacent", {})
            .get("divergence", {})
            .get(
                "max_abs",
                float("nan"),
            )
        )
        l1s = (
            band_star.get("layer1_first_interior", {})
            .get("divergence", {})
            .get(
                "max_abs",
                float("nan"),
            )
        )
        l1c = (
            band_corr.get("layer1_first_interior", {})
            .get("divergence", {})
            .get(
                "max_abs",
                float("nan"),
            )
        )
        l1f = (
            band_final.get("layer1_first_interior", {})
            .get("divergence", {})
            .get(
                "max_abs",
                float("nan"),
            )
        )
        l2s = (
            band_star.get("layer2_second_interior", {})
            .get("divergence", {})
            .get(
                "max_abs",
                float("nan"),
            )
        )
        l2c = (
            band_corr.get("layer2_second_interior", {})
            .get("divergence", {})
            .get(
                "max_abs",
                float("nan"),
            )
        )
        l2f = (
            band_final.get("layer2_second_interior", {})
            .get("divergence", {})
            .get(
                "max_abs",
                float("nan"),
            )
        )
        layer0_short = f"l0_div[s/c/f]={l0s:.2e}/{l0c:.2e}/{l0f:.2e}"
        layer1_short = f"l1_div[s/c/f]={l1s:.2e}/{l1c:.2e}/{l1f:.2e}"
        layer2_short = f"l2_div[s/c/f]={l2s:.2e}/{l2c:.2e}/{l2f:.2e}"
        l2_resp_short = (
            "l2_resp(max/mean)="
            f"{diag.get('layer2_flux_response_max', float('nan')):.2e}/"
            f"{diag.get('layer2_flux_response_mean', float('nan')):.2e}"
        )
        recon_flux_mis = diag.get(
            "projection_chain_flux_reconstruction_mismatch_max_abs",
            float("nan"),
        )
        recon_div_mis = diag.get(
            "projection_chain_divergence_reconstruction_mismatch_max_abs",
            float("nan"),
        )
        recon_short = (
            f"proj_recon_mis[flux/div]={recon_flux_mis:.2e}/{recon_div_mis:.2e}"
        )
        outlet_chain = diag.get("outlet_near_divergence_chain", {})
        out_chain_short = (
            "out_chain[s/cf/rf/f]="
            f"{outlet_chain.get('star', {}).get('max_abs', float('nan')):.2e}/"
            f"{outlet_chain.get('corrected_flux', {}).get('max_abs', float('nan')):.2e}/"
            f"{outlet_chain.get('reconstructed_flux', {}).get('max_abs', float('nan')):.2e}/"
            f"{outlet_chain.get('final', {}).get('max_abs', float('nan')):.2e}"
        )
        out_short = (
            "outlet_div[l0/l1/l2]="
            f"{diag.get('outlet_layer0_div_final_max', float('nan')):.2e}/"
            f"{diag.get('outlet_layer1_div_final_max', float('nan')):.2e}/"
            f"{diag.get('outlet_layer2_div_final_max', float('nan')):.2e}, "
            "outlet_pg(max/mean)="
            f"{diag.get('outlet_pressure_gradient_final_max', float('nan')):.2e}/"
            f"{diag.get('outlet_pressure_gradient_final_mean', float('nan')):.2e}, "
            "outlet_jump(max/mean)="
            f"{diag.get('outlet_velocity_jump_final_max', float('nan')):.2e}/"
            f"{diag.get('outlet_velocity_jump_final_mean', float('nan')):.2e}, "
            "outlet_mis(max/mean)="
            f"{diag.get('outlet_flux_mismatch_final_max', float('nan')):.2e}/"
            f"{diag.get('outlet_flux_mismatch_final_mean', float('nan')):.2e}"
        )
        if cfg.progress_every > 0 and ((step + 1) % cfg.progress_every == 0):
            print(
                f"[native-flow] step {step + 1}/{cfg.steps}, "
                f"max|div*|={diag['divergence_star_max_abs']:.3e}, "
                f"max|div_corr|={diag['divergence_projected_max_abs']:.3e}, "
                f"max|div_new|={diag['divergence_final_max_abs']:.3e}, "
                f"ratio_proj={diag['projection_reduction_factor_max_abs']:.3f}, "
                f"ratio_postbc={diag['projection_reduction_factor_max_abs_post_bc']:.3f}, "
                f"postbc_dmax={diag['post_bc_divergence_delta_max_abs']:.2e}, "
                f"p_res={diag['poisson_residual_max_abs']:.3e}, "
                f"rhs_mean_raw={diag['poisson_rhs_mean_raw']:.3e}, "
                f"max|u|={diag['max_velocity_magnitude']:.3e}, "
                f"{flux_short}, "
                f"{inlet_mismatch_short}, "
                f"{layer_flux_short}, "
                f"{layer_div_short}, "
                f"{layer_jump_short}, "
                f"{layer0_short}, "
                f"{layer1_short}, "
                f"{layer2_short}, "
                f"{l2_resp_short}, "
                f"{recon_short}, "
                f"{out_chain_short}, "
                f"{out_short}, "
                f"{reg_max_short}, "
                f"{reg_mean_short}"
            )
        if bool(diag.get("projection_warning_non_reducing", False)):
            print(
                f"[native-flow] WARNING: projection not reducing divergence at step {step + 1}. "
                f"ratio_proj_max={diag['projection_reduction_factor_max_abs']:.3f}, "
                f"ratio_proj_l2={diag['projection_reduction_factor_l2']:.3f}, "
                f"ratio_postbc_max={diag['projection_reduction_factor_max_abs_post_bc']:.3f}, "
                f"ratio_postbc_l2={diag['projection_reduction_factor_l2_post_bc']:.3f}, "
                f"postbc_dmax={diag['post_bc_divergence_delta_max_abs']:.3e}, "
                f"postbc_dl2={diag['post_bc_divergence_delta_l2']:.3e}"
            )
        if cfg.enable_safeguards:
            stop_reason = _safeguard_stop_reason(diag, cfg)
            if stop_reason is not None:
                print(
                    f"[native-flow] EARLY STOP at step {step + 1}: {stop_reason}. "
                    f"max|div|={diag['divergence_final_max_abs']:.3e}, "
                    f"p_res={diag['poisson_residual_max_abs']:.3e}, "
                    f"max|u|={diag['max_velocity_magnitude']:.3e}"
                )
                break

    final_diag = state.diagnostics_history[-1]
    completed_steps = len(state.diagnostics_history)
    stopped_early = completed_steps < cfg.steps
    return {
        "context": context,
        "state": state,
        "backend": context.backend,
        "device": context.device,
        "operator_consistency": operator_consistency,
        "initial_divergence_max_abs": initial_div_max,
        "final_divergence_max_abs": final_diag["divergence_final_max_abs"],
        "completed_steps": completed_steps,
        "requested_steps": cfg.steps,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "history": state.diagnostics_history,
    }
