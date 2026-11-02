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


def _extract_canonical_probs(
    dem: stim.DetectorErrorModel, out: stim.DetectorErrorModel | None = None
) -> dict[Any, float]:
    probs: dict[Any, float] = {}
    for inst in dem.flattened():
        if inst.type != "error":
            if out is not None:
                out.append(inst)
            continue
        sym = inst_canonical_symptom(inst)
        if sym != ((), ()):
            probs[sym] = prob_xor(probs.get(sym, 0.0), float(inst.args_copy()[0]))
    return probs


def _preserve_dem_counts(
    out: stim.DetectorErrorModel, ref: stim.DetectorErrorModel
) -> None:
    if out.num_detectors < ref.num_detectors:
        out.append(
            "detector",
            [],
            [stim.target_relative_detector_id(ref.num_detectors - 1)],
        )
    if out.num_observables < ref.num_observables:
        out.append(
            "logical_observable",
            [],
            [stim.target_logical_observable_id(ref.num_observables - 1)],
        )


def _apply_reweight_update_dem(
    base_dem: stim.DetectorErrorModel, update_dem: stim.DetectorErrorModel
) -> stim.DetectorErrorModel:
    """Apply a reweight-update DEM (containing updated total probabilities) onto ``base_dem``."""
    updates = _extract_canonical_probs(update_dem)
    if not updates:
        return base_dem
    out = stim.DetectorErrorModel()
    base_syms = _extract_canonical_probs(base_dem, out)
    for sym, p_base in base_syms.items():
        if (p_final := updates.pop(sym, p_base)) > 0:
            out.append("error", p_final, sym_to_targets(sym))
    for sym, p_new in updates.items():
        if p_new > 0:
            out.append("error", p_new, sym_to_targets(sym))
    _preserve_dem_counts(out, base_dem)
    return out


def _extract_graphlike_edges(
    dem: stim.DetectorErrorModel,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], tuple[int, ...]]]:
    """Return (edge -> merged_prob, edge -> first_obs) for a graphlike DEM."""
    probs: dict[tuple[int, int], float] = {}
    first_obs: dict[tuple[int, int], tuple[int, ...]] = {}
    for inst in dem.flattened():
        if inst.type != "error" or (p := float(inst.args_copy()[0])) <= 0.0:
            continue
        for dets, obs in graphlike_inst_symptom(inst)[0]:
            if 1 <= len(dets) <= 2:
                edge = dets_edge_key(dets)
                first_obs.setdefault(edge, obs)
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
    return 1.0 / (1.0 + math.exp(w))


def _ensure_base_dem_has_reweight_edges(
    base_dem: stim.DetectorErrorModel,
    rw_list: Sequence[NDArray[np.float64] | None],
) -> stim.DetectorErrorModel:
    eff_base = decompose_dem_graphlike(base_dem)
    existing_probs, _ = _extract_graphlike_edges(eff_base)
    missing_edges = {
        nodes_edge_key(int(row[0]), int(row[1]))
        for rw in rw_list
        if rw is not None
        for row in rw
    } - existing_probs.keys()
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
        updated_probs[nodes_edge_key(int(row[0]), int(row[1]))] = _weight_to_prob(float(row[2]))
    out = stim.DetectorErrorModel()
    for inst in eff_base.flattened():
        if inst.type != "error":
            out.append(inst)
    for edge, p in updated_probs.items():
        if p > 0.0:
            out.append("error", p, edge_to_targets(edge, base_obs.get(edge, ())))
    _preserve_dem_counts(out, eff_base)
    return out


def _resolve_shot_dem(
    dem: stim.DetectorErrorModel,
    *,
    apply_reweight: bool,
    decompose: bool,
    base_dem: stim.DetectorErrorModel | None,
) -> stim.DetectorErrorModel:
    if apply_reweight:
        if base_dem is None:
            raise ValueError(
                "base_dem (or dems generated by MarginalLeakageDemGenerator) is required "
                "when reweight_only=True."
            )
        dem = _apply_reweight_update_dem(base_dem, dem)
    if decompose:
        imeta = get_item_analysis_meta(dem)
        if not (imeta is not None and imeta[1]):
            dem = decompose_dem_graphlike(dem, base_dem=base_dem)
    return dem


def _decode_grouped_shots(
    decoder: Any,
    dets_2d: NDArray[np.uint8],
    groups: Sequence[tuple[list[int], stim.DetectorErrorModel]],
) -> NDArray[np.uint8]:
    out: NDArray[np.uint8] | None = None
    for shots, dem in groups:
        preds = decoder.compile_decoder_for_dem(dem=dem).decode_shots_bit_packed(
            bit_packed_detection_event_data=dets_2d[shots]
        )
        if out is None:
            out = np.zeros((dets_2d.shape[0], preds.shape[1]), dtype=np.uint8)
        out[shots] = preds
    assert out is not None
    return out


def _normalize_reweight_array(arr: NDArray[Any]) -> NDArray[np.float64] | None:
    arr_f64 = np.asarray(arr, dtype=np.float64)
    if arr_f64.ndim != 2 or arr_f64.shape[1] != 3:
        raise ValueError(
            f"Expected edge_reweights array of shape (num_reweights, 3), got {arr_f64.shape}."
        )
    return None if arr_f64.shape[0] == 0 else np.ascontiguousarray(arr_f64, dtype=np.float64)


def decode_with_edge_reweights(
    decoder: Any,
    reweights: NDArray[np.float64] | Sequence[NDArray[np.float64] | None],
    bit_packed_dets: NDArray[np.uint8],
    *,
    base_dem: stim.DetectorErrorModel | None = None,
    analysis: Any = None,
) -> NDArray[np.uint8]:
    dets_2d = bit_packed_dets[np.newaxis, :] if bit_packed_dets.ndim == 1 else bit_packed_dets
    num_shots = dets_2d.shape[0]
    if isinstance(reweights, np.ndarray):
        rw_list: list[NDArray[np.float64] | None] = [_normalize_reweight_array(reweights)] * num_shots
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
            else:
                rid = id(rw)
                if rid not in seen_arrays:
                    seen_arrays[rid] = _normalize_reweight_array(rw)
                rw_list.append(seen_arrays[rid])

    if not hasattr(decoder, "decode_batch") and (
        not is_pymatching_decoder(decoder) or not _matching_supports_edge_reweights()
    ):
        eff_base_dem = (
            analysis.matching_base_dem if analysis is not None else None
        ) or base_dem
        if eff_base_dem is None:
            raise ValueError(
                "base_dem (or dems generated by MarginalLeakageDemGenerator) is required "
                "when decoding edge_reweights."
            )
        if num_shots == 0:
            return np.zeros((0, (eff_base_dem.num_observables + 7) // 8), dtype=np.uint8)
        shots_by_rw: dict[int, tuple[list[int], NDArray[np.float64] | None]] = {}
        for shot_idx, rw_item in enumerate(rw_list):
            key = 0 if rw_item is None else id(rw_item)
            if key not in shots_by_rw:
                shots_by_rw[key] = ([], rw_item)
            shots_by_rw[key][0].append(shot_idx)
        graphlike_base = decompose_dem_graphlike(eff_base_dem)
        base_edges = _extract_graphlike_edges(graphlike_base)
        return _decode_grouped_shots(
            decoder,
            dets_2d,
            [
                (shots, _apply_edge_reweights_to_dem(graphlike_base, base_edges, rw_item))
                for shots, rw_item in shots_by_rw.values()
            ],
        )

    if hasattr(decoder, "decode_batch"):
        matcher = decoder
    elif analysis is not None:
        matcher = analysis.get_cached_matcher()
    elif base_dem is not None:
        import pymatching  # type: ignore[import-untyped]

        matcher = pymatching.Matching.from_detector_error_model(
            _ensure_base_dem_has_reweight_edges(base_dem, rw_list)
        )
    else:
        raise ValueError(
            "base_dem (or dems generated by MarginalLeakageDemGenerator) is required "
            "when decoding edge_reweights."
        )

    if num_shots == 0:
        return np.zeros((0, (matcher.num_fault_ids + 7) // 8), dtype=np.uint8)
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
    """Decode a batch with the DEM(s) or ``edge_reweights`` returned by a ``dem_gen`` callable.

    Args:
        decoder: A ``sinter.Decoder``, built-in decoder name (e.g. ``"pymatching"``),
            or a ``pymatching.Matching`` instance (which only decodes ``edge_reweights``
            arrays with the PyMatching fork; raises ``TypeError`` on DEM inputs or stock
            PyMatching).
        dems: Single or per-shot ``stim.DetectorErrorModel`` or ``(k, 3)`` ``edge_reweights``
            array(s) (or ``GeneratedDemList``).
        bit_packed_dets: Shot-major bit-packed detection events, or ``None`` to return the
            transformed DEM(s) / ``edge_reweights`` without decoding.
        decompose_errors: Decompose hyperedges into graphlike errors (OR-ed with generator
            metadata; ``decompose_errors`` + ``reweight_only`` produces ``edge_reweights``
            arrays only for ``MarginalLeakageDemGenerator`` outputs, returning DEMs for
            plain DEM inputs).
        reweight_only: Apply per-shot reweight updates onto ``base_dem`` / generator baseline.
        base_dem: Fallback baseline DEM for metadata-less ``reweight_only`` or
            ``edge_reweights`` inputs (generator metadata takes precedence when present).

    Returns:
        Bit-packed observable predictions of shape ``(num_shots, ceil(num_obs / 8))``, or
        transformed DEM(s) / ``edge_reweights`` when ``bit_packed_dets is None``.
    """
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
        dec_flag, rew_flag = eff_dec, eff_rew
        if bit_packed_dets is None:
            return dems

    dets_2d = (
        None
        if bit_packed_dets is None
        else (bit_packed_dets[np.newaxis, :] if bit_packed_dets.ndim == 1 else bit_packed_dets)
    )

    if isinstance(dems, np.ndarray):
        if dets_2d is None:
            return dems
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
        is_reweight_seq = (
            isinstance(first_non_none, np.ndarray)
            or (dems_seq and first_non_none is None)
            or (bool(getattr(dems, "decompose_errors", False)) and bool(getattr(dems, "reweight_only", False)))
        )
        if is_reweight_seq:
            if dets_2d is None:
                return dems
            analysis = getattr(dems, "analysis", None)
            if analysis is None and first_non_none is not None:
                imeta = get_item_analysis_meta(first_non_none)
                analysis = imeta[0] if imeta is not None else None
            return decode_with_edge_reweights(
                decoder, dems_seq, dets_2d, base_dem=base_dem, analysis=analysis  # type: ignore[arg-type]
            )

    if dets_2d is not None and not hasattr(decoder, "compile_decoder_for_dem"):
        raise TypeError(
            "A pymatching.Matching instance only decodes edge_reweights arrays, "
            "not stim.DetectorErrorModel inputs."
        )

    if isinstance(dems, stim.DetectorErrorModel):
        single_analysis = single_imeta[0] if single_imeta is not None else None
        dem_single = _resolve_shot_dem(
            dems,
            apply_reweight=rew_flag or bool(single_imeta and single_imeta[2]),
            decompose=dec_flag,
            base_dem=(single_analysis.baseline if single_analysis is not None else None) or base_dem,
        )
        if dets_2d is None:
            return dem_single
        if not dec_flag:
            warn_if_pymatching_hyperedge(decoder, dem_single)
        return decoder.compile_decoder_for_dem(dem=dem_single).decode_shots_bit_packed(
            bit_packed_detection_event_data=dets_2d
        )

    dems_list: list[stim.DetectorErrorModel] = list(dems)  # type: ignore[arg-type]
    if dets_2d is not None:
        num_shots = dets_2d.shape[0]
        if len(dems_list) != num_shots:
            raise ValueError(
                f"dem_gen returned {len(dems_list)} DEMs for a batch of {num_shots} shots."
            )
        if num_shots == 0:
            analysis0 = getattr(dems, "analysis", None)
            eff_base = (
                getattr(dems, "base_dem", None)
                or (analysis0.baseline if analysis0 is not None else None)
                or base_dem
            )
            circ0 = getattr(dems, "circuit", None)
            n_obs = (
                eff_base.num_observables
                if eff_base is not None
                else (circ0.num_observables if circ0 is not None else 0)
            )
            return np.zeros((0, (n_obs + 7) // 8), dtype=np.uint8)
    elif not dems_list:
        return []

    imeta0 = get_item_analysis_meta(dems_list[0])
    analysis = getattr(dems, "analysis", None) or (imeta0[0] if imeta0 is not None else None)
    apply_rew = (
        rew_flag
        or bool(getattr(dems, "reweight_only", False))
        or bool(imeta0 and imeta0[2])
    )
    gen_base_dem = (
        getattr(dems, "base_dem", None)
        or (analysis.baseline if analysis is not None else None)
        or base_dem
    )

    shots_by_dem: dict[int, list[int]] = {}
    for shot, dem in enumerate(dems_list):
        shots_by_dem.setdefault(id(dem), []).append(shot)
    resolved_by_id = {
        did: _resolve_shot_dem(
            dems_list[shots[0]],
            apply_reweight=apply_rew,
            decompose=dec_flag,
            base_dem=gen_base_dem,
        )
        for did, shots in shots_by_dem.items()
    }
    if dets_2d is None:
        return [resolved_by_id[id(d)] for d in dems_list]
    if not dec_flag:
        for dem_item in resolved_by_id.values():
            warn_if_pymatching_hyperedge(decoder, dem_item)
    return _decode_grouped_shots(
        decoder,
        dets_2d,
        [(shots, resolved_by_id[did]) for did, shots in shots_by_dem.items()],
    )

