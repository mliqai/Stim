"""Decoding helpers for per-shot DEMs produced by the DEM generators.

``decode_with_generated_dems`` decodes a batch with the DEM(s) or PyMatching
``edge_reweights`` returned by a ``dem_gen`` callable.
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Sequence
import warnings

import numpy as np
from numpy.typing import NDArray
import sinter  # type: ignore[import-untyped]
import stim  # type: ignore[import-untyped]

from stimside.dem_generators.dem_generator_marginal import (
    GeneratedDemList,
    get_item_analysis_meta,
    get_item_leakage_events,
)
from stimside.dem_generators.graphlike import (
    decompose_dem_graphlike,
    dem_has_hyperedge,
    dets_edge_key,
    edge_to_targets,
    graphlike_inst_symptom,
    inst_canonical_symptom,
    nodes_edge_key,
    prob_xor,
    sym_to_targets,
)

_MAX_PYMATCHING_WEIGHT = float((1 << 24) - 1)


def is_pymatching_decoder(decoder: object) -> bool:
    if isinstance(decoder, str):
        return "pymatching" in decoder.lower()
    cls_name = type(decoder).__name__.lower()
    mod_name = getattr(type(decoder), "__module__", "").lower()
    return "pymatching" in cls_name or "pymatching" in mod_name


def matching_supports(*params: str) -> bool:
    import pymatching  # type: ignore[import-untyped]

    sig_params = inspect.signature(pymatching.Matching.decode_batch).parameters
    return all(p in sig_params for p in params)


def _matching_supports_edge_reweights() -> bool:
    return matching_supports("edge_reweights")


def warn_if_pymatching_hyperedge(
    decoder: object, dem: stim.DetectorErrorModel
) -> None:
    if is_pymatching_decoder(decoder) and dem_has_hyperedge(dem):
        warnings.warn(
            "DetectorErrorModel contains undecomposed hyperedge error mechanisms "
            "with >2 detectors while decoding with pymatching; pass "
            "decompose_errors=True to decompose errors into matchable graphs.",
            UserWarning,
            stacklevel=3,
        )


def _apply_reweight_update_dem(
    base_dem: stim.DetectorErrorModel, update_dem: stim.DetectorErrorModel
) -> stim.DetectorErrorModel:
    """Apply a reweight-update DEM (containing updated total probabilities) onto ``base_dem``."""
    updates: dict[Any, float] = {}
    for inst in update_dem.flattened():
        if inst.type != "error":
            continue
        sym = inst_canonical_symptom(inst)
        if sym == ((), ()):
            continue
        p = float(inst.args_copy()[0])
        updates[sym] = prob_xor(updates.get(sym, 0.0), p)

    if not updates:
        return base_dem

    base_syms: dict[Any, float] = {}
    out = stim.DetectorErrorModel()
    for inst in base_dem.flattened():
        if inst.type != "error":
            out.append(inst)
            continue
        sym = inst_canonical_symptom(inst)
        if sym == ((), ()):
            continue
        p = float(inst.args_copy()[0])
        base_syms[sym] = prob_xor(base_syms.get(sym, 0.0), p)

    for sym, p_base in base_syms.items():
        p_final = updates.pop(sym, p_base)
        if p_final > 0:
            out.append("error", p_final, sym_to_targets(sym))
    for sym, p_new in updates.items():
        if p_new > 0:
            out.append("error", p_new, sym_to_targets(sym))
    return out


def _prob_to_weight(p: float) -> float:
    if p <= 0.0:
        return _MAX_PYMATCHING_WEIGHT
    if p >= 1.0:
        return 0.0
    w = math.log((1.0 - p) / p)
    if w < 0.0:
        return 0.0
    if w > _MAX_PYMATCHING_WEIGHT:
        return _MAX_PYMATCHING_WEIGHT
    return w


def _extract_graphlike_edges(
    dem: stim.DetectorErrorModel,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], tuple[int, ...]]]:
    """Return (edge -> merged_prob, edge -> first_obs) for a graphlike DEM."""
    probs: dict[tuple[int, int], float] = {}
    first_obs: dict[tuple[int, int], tuple[int, ...]] = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = float(inst.args_copy()[0])
        if p <= 0.0:
            continue
        for dets, obs in graphlike_inst_symptom(inst)[0]:
            if not (1 <= len(dets) <= 2):
                continue
            edge = dets_edge_key(dets)
            if edge not in first_obs:
                first_obs[edge] = obs
            probs[edge] = prob_xor(probs.get(edge, 0.0), p)
    return probs, first_obs


def _weight_to_prob(w: float) -> float:
    if w >= _MAX_PYMATCHING_WEIGHT:
        return 0.0
    if w <= -_MAX_PYMATCHING_WEIGHT:
        return 1.0
    if w >= 0.0:
        ew = math.exp(-w)
        return ew / (1.0 + ew)
    ew = math.exp(w)
    return 1.0 / (1.0 + ew)


def _ensure_base_dem_has_reweight_edges(
    base_dem: stim.DetectorErrorModel,
    rw_list: Sequence[NDArray[np.float64] | None],
) -> stim.DetectorErrorModel:
    eff_base = decompose_dem_graphlike(base_dem)
    existing_probs, existing_obs = _extract_graphlike_edges(eff_base)
    missing_edges: set[tuple[int, int]] = set()
    for rw in rw_list:
        if rw is None:
            continue
        for row in rw:
            edge = nodes_edge_key(int(row[0]), int(row[1]))
            if edge not in existing_probs and edge not in existing_obs:
                missing_edges.add(edge)
    if not missing_edges:
        return eff_base
    out = eff_base.copy()
    for edge in sorted(missing_edges):
        out.append("error", 1e-12, edge_to_targets(edge))
    return out


def _apply_edge_reweights_to_dem(
    eff_base: stim.DetectorErrorModel,
    base_edges: tuple[dict[tuple[int, int], float], dict[tuple[int, int], tuple[int, ...]]],
    rw: NDArray[np.float64] | None,
) -> stim.DetectorErrorModel:
    """``eff_base`` is a graphlike base DEM and ``base_edges`` its ``_extract_graphlike_edges``."""
    if rw is None or rw.shape[0] == 0:
        return eff_base
    base_probs, base_obs = base_edges
    updated_probs = dict(base_probs)
    for row in rw:
        edge = nodes_edge_key(int(row[0]), int(row[1]))
        updated_probs[edge] = _weight_to_prob(float(row[2]))
    out = stim.DetectorErrorModel()
    for inst in eff_base.flattened():
        if inst.type != "error":
            out.append(inst)
    for edge, p in updated_probs.items():
        if p > 0.0:
            out.append("error", p, edge_to_targets(edge, base_obs.get(edge, ())))
    return out


def _resolve_shot_dem(
    dem: stim.DetectorErrorModel,
    *,
    apply_reweight: bool,
    decompose: bool,
    base_dem: stim.DetectorErrorModel | None,
) -> stim.DetectorErrorModel:
    if apply_reweight and base_dem is not None:
        dem = _apply_reweight_update_dem(base_dem, dem)
    if decompose:
        imeta = get_item_analysis_meta(dem)
        if not (imeta is not None and imeta[1]):
            dem = decompose_dem_graphlike(dem, base_dem=base_dem)
    return dem


def decode_with_edge_reweights(
    decoder: Any,
    reweights: NDArray[np.float64] | Sequence[NDArray[np.float64] | None],
    bit_packed_dets: NDArray[np.uint8],
    *,
    base_dem: stim.DetectorErrorModel | None = None,
    analysis: Any = None,
) -> NDArray[np.uint8]:
    dets_2d = (
        bit_packed_dets[np.newaxis, :]
        if bit_packed_dets.ndim == 1
        else bit_packed_dets
    )
    num_shots = dets_2d.shape[0]
    if isinstance(reweights, np.ndarray):
        if reweights.ndim != 2 or reweights.shape[1] != 3:
            raise ValueError(
                f"Expected edge_reweights array of shape (num_reweights, 3), got {reweights.shape}."
            )
        rw_arr = (
            None
            if reweights.shape[0] == 0
            else np.ascontiguousarray(reweights, dtype=np.float64)
        )
        rw_list: list[NDArray[np.float64] | None] = [rw_arr] * num_shots
    else:
        if len(reweights) != num_shots:
            raise ValueError(
                f"dem_gen returned {len(reweights)} DEMs for a batch of {num_shots} shots."
            )
        rw_list = []
        seen_arrays: dict[int, NDArray[np.float64] | None] = {}
        for rw in reweights:
            if rw is None:
                rw_list.append(None)
                continue
            rid = id(rw)
            if rid not in seen_arrays:
                arr = np.asarray(rw, dtype=np.float64)
                if arr.ndim != 2 or arr.shape[1] != 3:
                    raise ValueError(
                        f"Expected edge_reweights array of shape (num_reweights, 3), got {arr.shape}."
                    )
                seen_arrays[rid] = (
                    None
                    if arr.shape[0] == 0
                    else np.ascontiguousarray(arr, dtype=np.float64)
                )
            rw_list.append(seen_arrays[rid])

    if not hasattr(decoder, "decode_batch") and (
        not is_pymatching_decoder(decoder) or not _matching_supports_edge_reweights()
    ):
        eff_base_dem = base_dem or (
            analysis.matching_base_dem if analysis is not None else None
        )
        if eff_base_dem is None:
            raise ValueError(
                "base_dem (or dems generated by MarginalLeakageDemGenerator) is required "
                "when decoding edge_reweights."
            )
        if num_shots == 0:
            return np.zeros((0, (eff_base_dem.num_observables + 7) // 8), dtype=np.uint8)
        shots_by_rw: dict[int, list[int]] = {}
        rw_by_key: dict[int, NDArray[np.float64] | None] = {}
        for shot_idx, rw_item in enumerate(rw_list):
            key = 0 if rw_item is None else id(rw_item)
            shots_by_rw.setdefault(key, []).append(shot_idx)
            rw_by_key[key] = rw_item
        out: NDArray[np.uint8] | None = None
        graphlike_base = decompose_dem_graphlike(eff_base_dem)
        base_edges = _extract_graphlike_edges(graphlike_base)
        for key, shots in shots_by_rw.items():
            dem_for_group = _apply_edge_reweights_to_dem(
                graphlike_base, base_edges, rw_by_key[key]
            )
            preds = decoder.compile_decoder_for_dem(
                dem=dem_for_group
            ).decode_shots_bit_packed(
                bit_packed_detection_event_data=dets_2d[shots]
            )
            if out is None:
                out = np.zeros((num_shots, preds.shape[1]), dtype=np.uint8)
            out[shots] = preds
        assert out is not None
        return out

    if hasattr(decoder, "decode_batch"):
        matcher = decoder
    else:
        if analysis is not None and base_dem is None:
            matcher = analysis.get_cached_matcher()
        elif base_dem is not None:
            import pymatching  # type: ignore[import-untyped]

            eff_base = _ensure_base_dem_has_reweight_edges(base_dem, rw_list)
            matcher = pymatching.Matching.from_detector_error_model(eff_base)
        else:
            raise ValueError(
                "base_dem (or dems generated by MarginalLeakageDemGenerator) is required "
                "when decoding edge_reweights."
            )

    if num_shots == 0:
        num_obs_bytes = (matcher.num_fault_ids + 7) // 8
        return np.zeros((0, num_obs_bytes), dtype=np.uint8)

    return matcher.decode_batch(
        dets_2d,
        bit_packed_shots=True,
        bit_packed_predictions=True,
        edge_reweights=rw_list,
    )


def _extract_regen_spec(
    dems: Any, single_imeta: Any
) -> tuple[bool, bool, Any, Any, Any, Any] | None:
    if isinstance(dems, GeneratedDemList) and dems.generator is not None:
        return (
            dems.decompose_errors,
            dems.reweight_only,
            dems.generator,
            dems.circuit,
            dems.records,
            dems.leakage_events,
        )
    if single_imeta is not None:
        _, orig_dec, orig_rew, gen_inst, circ_inst, rec_inst = single_imeta
        return orig_dec, orig_rew, gen_inst, circ_inst, rec_inst, get_item_leakage_events(dems)
    if not isinstance(dems, (stim.DetectorErrorModel, np.ndarray)):
        seq = dems if isinstance(dems, list) else list(dems)
        if seq:
            metas = [get_item_analysis_meta(x) for x in seq]
            m0 = metas[0]
            if m0 is not None and all(m is not None and m[3] is m0[3] and m[4] is m0[4] for m in metas):
                rec_2d = np.stack([m[5] for m in metas if m is not None], axis=0)
                evs = [ev for ev in (get_item_leakage_events(x) for x in seq) if ev is not None]
                return m0[1], m0[2], m0[3], m0[4], rec_2d, evs or None
    return None


def _decode_non_pymatching_reweights(
    decoder: Any, regen_spec: tuple[bool, bool, Any, Any, Any, Any] | None, dets_2d: NDArray[np.uint8]
) -> Any:
    if regen_spec is None or is_pymatching_decoder(decoder) or hasattr(decoder, "decode_batch"):
        return None
    _, _, gen_inst, circ_inst, rec_inst, evs = regen_spec
    dem_full = gen_inst(circ_inst, rec_inst, decompose_errors=True, reweight_only=False, leakage_events=evs)
    return decode_with_generated_dems(decoder, dem_full, dets_2d, decompose_errors=True)


def decode_with_generated_dems(
    decoder: sinter.Decoder | Any,
    dems: (
        stim.DetectorErrorModel
        | Sequence[stim.DetectorErrorModel]
        | NDArray[np.float64]
        | Sequence[NDArray[np.float64] | None]
    ),
    bit_packed_dets: NDArray[np.uint8] | None = None,
    decompose_errors: bool = False,
    reweight_only: bool = False,
    *,
    base_dem: stim.DetectorErrorModel | None = None,
) -> Any:
    """Decode a batch with the DEM(s) or edge_reweights returned by a dem_gen callable."""
    dec_flag, rew_flag = bool(decompose_errors), bool(reweight_only)
    if isinstance(decoder, str):
        decoder = sinter.BUILT_IN_DECODERS[decoder]

    single_imeta = (
        get_item_analysis_meta(dems)
        if isinstance(dems, (stim.DetectorErrorModel, np.ndarray))
        else None
    )
    regen_spec = _extract_regen_spec(dems, single_imeta)
    if regen_spec is not None:
        orig_dec, orig_rew, gen_inst, circ_inst, rec_inst, evs = regen_spec
        eff_dec, eff_rew = dec_flag or orig_dec, rew_flag or orig_rew
        if (eff_dec, eff_rew) != (orig_dec, orig_rew):
            dems = gen_inst(
                circ_inst, rec_inst, decompose_errors=eff_dec, reweight_only=eff_rew, leakage_events=evs
            )
            single_imeta = (
                get_item_analysis_meta(dems)
                if isinstance(dems, (stim.DetectorErrorModel, np.ndarray))
                else None
            )
            regen_spec = (eff_dec, eff_rew, gen_inst, circ_inst, rec_inst, evs)
        dec_flag, rew_flag = eff_dec, eff_rew
        if bit_packed_dets is None:
            return dems

    dets_2d = (
        None
        if bit_packed_dets is None
        else (
            bit_packed_dets[np.newaxis, :]
            if bit_packed_dets.ndim == 1
            else bit_packed_dets
        )
    )

    # Case 1: dems is already in PyMatching edge_reweights format (np.ndarray or Sequence[np.ndarray | None])
    if isinstance(dems, np.ndarray):
        if dets_2d is None:
            return dems
        fb = _decode_non_pymatching_reweights(decoder, regen_spec, dets_2d)
        if fb is not None:
            return fb
        return decode_with_edge_reweights(
            decoder,
            dems,
            dets_2d,
            base_dem=base_dem,
            analysis=single_imeta[0] if single_imeta is not None else None,
        )
    if not isinstance(dems, stim.DetectorErrorModel):
        dems_seq = dems if isinstance(dems, list) else list(dems)
        first_non_none = next((x for x in dems_seq if x is not None), None)
        if isinstance(first_non_none, np.ndarray) or (
            len(dems_seq) > 0 and first_non_none is None
        ):
            if dets_2d is None:
                return dems
            fb = _decode_non_pymatching_reweights(decoder, regen_spec, dets_2d)
            if fb is not None:
                return fb
            analysis = getattr(dems, "analysis", None)
            if analysis is None and first_non_none is not None:
                imeta = get_item_analysis_meta(first_non_none)
                if imeta is not None:
                    analysis = imeta[0]
            return decode_with_edge_reweights(
                decoder,
                dems_seq,  # type: ignore[arg-type]
                dets_2d,
                base_dem=base_dem,
                analysis=analysis,
            )

    # Case 2: Single stim.DetectorErrorModel
    if isinstance(dems, stim.DetectorErrorModel):
        imeta = single_imeta
        single_analysis = imeta[0] if imeta is not None else None
        is_rew_single = rew_flag or (imeta is not None and imeta[2])
        eff_base_dem = base_dem or (
            single_analysis.baseline if single_analysis is not None else None
        )

        if (
            dec_flag
            and rew_flag
            and (dets_2d is None or is_pymatching_decoder(decoder))
            and eff_base_dem is not None
        ):
            if dets_2d is None:
                res_list = decode_with_generated_dems(
                    decoder,
                    [dems],
                    None,
                    decompose_errors=True,
                    reweight_only=True,
                    base_dem=eff_base_dem,
                )
                return res_list[0]
            num_shots = dets_2d.shape[0]
            return decode_with_generated_dems(
                decoder,
                [dems] * num_shots,
                dets_2d,
                decompose_errors=True,
                reweight_only=True,
                base_dem=eff_base_dem,
            )

        dem_single = _resolve_shot_dem(
            dems,
            apply_reweight=is_rew_single,
            decompose=dec_flag,
            base_dem=eff_base_dem,
        )
        if dets_2d is None:
            return dem_single
        if not dec_flag:
            warn_if_pymatching_hyperedge(decoder, dem_single)
        return decoder.compile_decoder_for_dem(dem=dem_single).decode_shots_bit_packed(
            bit_packed_detection_event_data=dets_2d
        )

    # Case 3: Sequence of stim.DetectorErrorModel
    dems_list: list[stim.DetectorErrorModel] = list(dems)  # type: ignore[arg-type]
    if dets_2d is not None:
        num_shots = dets_2d.shape[0]
        if len(dems_list) != num_shots:
            raise ValueError(
                f"dem_gen returned {len(dems_list)} DEMs for a batch of {num_shots} shots."
            )
        if num_shots == 0:
            return np.zeros((0, 0), dtype=np.uint8)
    elif not dems_list:
        return []

    imeta0 = get_item_analysis_meta(dems_list[0])
    analysis = getattr(dems, "analysis", None) or (
        imeta0[0] if imeta0 is not None else None
    )
    is_reweight_dems = bool(getattr(dems, "reweight_only", False)) or (
        imeta0[2] if imeta0 is not None else False
    )
    apply_rew = rew_flag or is_reweight_dems
    gen_base_dem = (
        base_dem
        or getattr(dems, "base_dem", None)
        or (analysis.baseline if analysis is not None else None)
    )

    shots_by_dem: dict[int, list[int]] = {}
    for shot, dem in enumerate(dems_list):
        shots_by_dem.setdefault(id(dem), []).append(shot)

    # If both decompose_errors and reweight_only are requested on plain DEMs with PyMatching (or bit_packed_dets is None):
    if dec_flag and rew_flag and (dets_2d is None or is_pymatching_decoder(decoder)):
        unique_ids = list(shots_by_dem.keys())
        resolved_by_id = {
            did: _resolve_shot_dem(
                dems_list[shots_by_dem[did][0]],
                apply_reweight=apply_rew,
                decompose=True,
                base_dem=gen_base_dem,
            )
            for did in unique_ids
        }

        ref_dem = (
            decompose_dem_graphlike(gen_base_dem)
            if gen_base_dem is not None
            else resolved_by_id[unique_ids[0]]
        )
        base_probs, base_obs = _extract_graphlike_edges(ref_dem)
        obs_compatible = True
        edges_by_id: dict[int, dict[tuple[int, int], float]] = {}
        for did, r_dem in resolved_by_id.items():
            e_probs, e_obs = _extract_graphlike_edges(r_dem)
            edges_by_id[did] = e_probs
            for edge, obs in e_obs.items():
                if edge in base_obs and base_obs[edge] != obs:
                    obs_compatible = False
                    break
                if edge not in base_obs:
                    base_obs[edge] = obs

        if obs_compatible:
            full_base_dem = ref_dem.copy()
            for edge, obs in base_obs.items():
                if edge not in base_probs:
                    full_base_dem.append("error", 1e-12, edge_to_targets(edge, obs))
            rw_by_id: dict[int, NDArray[np.float64] | None] = {}
            empty_arr = np.empty((0, 3), dtype=np.float64)
            for did, e_probs in edges_by_id.items():
                rows: list[list[float]] = []
                all_edges = set(base_probs.keys()) | set(e_probs.keys())
                for edge in sorted(all_edges):
                    p_b = base_probs.get(edge, 0.0)
                    p_cur = e_probs.get(edge, 0.0)
                    if abs(p_cur - p_b) > 1e-15:
                        rows.append(
                            [float(edge[0]), float(edge[1]), _prob_to_weight(p_cur)]
                        )
                rw_by_id[did] = (
                    np.asarray(rows, dtype=np.float64)
                    if rows
                    else (empty_arr if dets_2d is None else None)
                )
            rw_per_shot = [rw_by_id[id(d)] for d in dems_list]
            if dets_2d is None:
                return rw_per_shot
            return decode_with_edge_reweights(
                decoder,
                rw_per_shot,
                dets_2d,
                base_dem=full_base_dem,
            )

    if dets_2d is None:
        transformed_by_id = {
            did: _resolve_shot_dem(
                dems_list[shots[0]],
                apply_reweight=apply_rew,
                decompose=dec_flag,
                base_dem=gen_base_dem,
            )
            for did, shots in shots_by_dem.items()
        }
        return [transformed_by_id[id(d)] for d in dems_list]

    out = None
    for shots in shots_by_dem.values():
        dem_item = _resolve_shot_dem(
            dems_list[shots[0]],
            apply_reweight=apply_rew,
            decompose=dec_flag,
            base_dem=gen_base_dem,
        )
        if not dec_flag:
            warn_if_pymatching_hyperedge(decoder, dem_item)
        predictions = decoder.compile_decoder_for_dem(
            dem=dem_item
        ).decode_shots_bit_packed(
            bit_packed_detection_event_data=dets_2d[shots]
        )
        if out is None:
            out = np.zeros((num_shots, predictions.shape[1]), dtype=np.uint8)
        out[shots] = predictions
    assert out is not None
    return out
