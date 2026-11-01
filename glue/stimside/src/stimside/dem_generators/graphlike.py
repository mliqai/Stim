"""Symptom representation and graph-like DEM decomposition helpers."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import stim  # type: ignore[import-untyped]

Symptom = tuple[tuple[int, ...], tuple[int, ...]]  # (detector ids, observable ids)


def is_compound_symptom(sym: Any) -> bool:
    return (
        bool(sym)
        and isinstance(sym[0], tuple)
        and bool(sym[0])
        and isinstance(sym[0][0], tuple)
    )


def symptom_components(sym: Any) -> tuple[Symptom, ...]:
    if is_compound_symptom(sym):
        return sym
    if sym == ((), ()):
        return ()
    return (sym,)


def canonical_symptom_from_components(comps: Sequence[Symptom]) -> Any:
    non_empty = [c for c in comps if c[0] or c[1]]
    if not non_empty:
        return ((), ())
    if len(non_empty) == 1:
        return non_empty[0]
    return tuple(sorted(non_empty))


def sym_to_targets(sym: Any) -> list[stim.DemTarget]:
    comps = symptom_components(sym)
    targets: list[stim.DemTarget] = []
    for idx, (dets, obs) in enumerate(comps):
        if idx > 0:
            targets.append(stim.target_separator())
        targets.extend(stim.target_relative_detector_id(d) for d in dets)
        targets.extend(stim.target_logical_observable_id(o) for o in obs)
    return targets


def dets_edge_key(dets: tuple[int, ...]) -> tuple[int, int]:
    return (dets[0], -1) if len(dets) == 1 else (dets[0], dets[1])


def nodes_edge_key(u: int, v: int) -> tuple[int, int]:
    return (max(u, v), -1) if (u < 0 or v < 0) else (min(u, v), max(u, v))


def edge_to_targets(
    edge: tuple[int, int], obs: Sequence[int] = ()
) -> list[stim.DemTarget]:
    dets = (edge[0],) if edge[1] < 0 else edge
    return sym_to_targets((dets, tuple(obs)))


def inst_symptom(inst: stim.DemInstruction) -> Symptom:
    dets: set[int] = set()
    obs: set[int] = set()
    for t in inst.targets_copy():
        if t.is_relative_detector_id():
            dets ^= {t.val}
        elif t.is_logical_observable_id():
            obs ^= {t.val}
    return tuple(sorted(dets)), tuple(sorted(obs))


def raw_inst_components(inst: stim.DemInstruction) -> list[Symptom]:
    """Extract separator-delimited (detectors, observables) components of a DEM instruction."""
    comps: list[Symptom] = []
    dets: set[int] = set()
    obs: set[int] = set()
    for t in inst.targets_copy():
        if t.is_separator():
            if dets or obs:
                comps.append((tuple(sorted(dets)), tuple(sorted(obs))))
            dets.clear()
            obs.clear()
        elif t.is_relative_detector_id():
            dets ^= {t.val}
        elif t.is_logical_observable_id():
            obs ^= {t.val}
    if dets or obs:
        comps.append((tuple(sorted(dets)), tuple(sorted(obs))))
    return comps


def inst_canonical_symptom(inst: stim.DemInstruction) -> Any:
    return canonical_symptom_from_components(raw_inst_components(inst))


def collect_known_graphlike(
    *dems: stim.DetectorErrorModel | None,
) -> dict[tuple[int, ...], Symptom]:
    """Collect known 1- and 2-detector components from DEM(s) for hyperedge decomposition."""
    known: dict[tuple[int, ...], Symptom] = {}
    for dem in dems:
        if dem is None:
            continue
        for inst in dem.flattened():
            if inst.type != "error":
                continue
            args = inst.args_copy()
            if not args or args[0] <= 0:
                continue
            for dets, obs in raw_inst_components(inst):
                if 1 <= len(dets) <= 2 and dets not in known:
                    known[dets] = (dets, obs)
    return known


def decompose_hyperedge_component(
    dets: tuple[int, ...],
    obs: tuple[int, ...],
    known_graphlike: dict[tuple[int, ...], Symptom] | None = None,
) -> list[Symptom]:
    """Decompose a >2-detector component into <=2-detector components using Stim's algorithm."""
    n = len(dets)
    if n == 0:
        return []
    if n <= 2:
        return [(dets, obs)]

    if known_graphlike:
        target_obs = set(obs)
        out: list[Symptom] = []

        def _search(start: int, used_mask: int, cur_obs: set[int]) -> bool:
            while start < n and ((used_mask >> start) & 1):
                start += 1
            if start >= n:
                return cur_obs == target_obs
            used_mask |= 1 << start
            d0 = dets[start]
            for k in range(start + 1, n + 1):
                if k < n:
                    if (used_mask >> k) & 1:
                        continue
                    key: tuple[int, ...] = (d0, dets[k])
                    next_mask = used_mask | (1 << k)
                else:
                    key = (d0,)
                    next_mask = used_mask
                match = known_graphlike.get(key)
                if match is not None:
                    if _search(start + 1, next_mask, cur_obs ^ set(match[1])):
                        out.append(match)
                        return True
            return False

        if n < 64 and _search(0, 0, set()):
            out.reverse()
            return out

        # Fallback: find a disjoint subset of known graphlike edges covering the
        # maximum number of detectors (minimizing missed detectors), then add
        # remnant edge(s) for any remaining missed detectors.
        if n <= 16:
            best_missed = n + 1
            best_peeled: list[Symptom] = []
            best_missed_dets: list[int] = []
            cur_peeled: list[Symptom] = []
            cur_missed_dets: list[int] = []

            def _search_cover(start: int, used_mask: int) -> None:
                nonlocal best_missed, best_peeled, best_missed_dets
                if len(cur_missed_dets) >= best_missed:
                    return
                while start < n and ((used_mask >> start) & 1):
                    start += 1
                if start >= n:
                    best_missed = len(cur_missed_dets)
                    best_peeled = list(cur_peeled)
                    best_missed_dets = list(cur_missed_dets)
                    return
                used_mask |= 1 << start
                d0 = dets[start]
                for k in range(start + 1, n):
                    if (used_mask >> k) & 1:
                        continue
                    match = known_graphlike.get((d0, dets[k]))
                    if match is not None:
                        cur_peeled.append(match)
                        _search_cover(start + 1, used_mask | (1 << k))
                        cur_peeled.pop()
                        if best_missed == 0:
                            return
                match_single = known_graphlike.get((d0,))
                if match_single is not None:
                    cur_peeled.append(match_single)
                    _search_cover(start + 1, used_mask)
                    cur_peeled.pop()
                    if best_missed == 0:
                        return
                if len(cur_missed_dets) + 1 < best_missed:
                    cur_missed_dets.append(d0)
                    _search_cover(start + 1, used_mask)
                    cur_missed_dets.pop()

            _search_cover(0, 0)
            peeled = best_peeled
            missed = tuple(best_missed_dets)
            rem_obs = set(obs)
            for _, o in peeled:
                rem_obs ^= set(o)
        else:
            done = [False] * n
            rem_obs = set(obs)
            peeled = []
            for k in range(n):
                if not done[k]:
                    for k2 in range(k + 1, n):
                        if not done[k2]:
                            match = known_graphlike.get((dets[k], dets[k2]))
                            if match is not None:
                                done[k] = done[k2] = True
                                peeled.append(match)
                                rem_obs ^= set(match[1])
                                break
            for k in range(n):
                if not done[k]:
                    match = known_graphlike.get((dets[k],))
                    if match is not None:
                        done[k] = True
                        peeled.append(match)
                        rem_obs ^= set(match[1])
            missed = tuple(dets[k] for k in range(n) if not done[k])

        if len(missed) <= 2:
            if missed:
                peeled.append((missed, tuple(sorted(rem_obs))))
            elif rem_obs and peeled:
                d0, o0 = peeled[0]
                peeled[0] = (d0, tuple(sorted(set(o0) ^ rem_obs)))
            return peeled
        for idx in range(0, len(missed), 2):
            peeled.append(
                (missed[idx : idx + 2], tuple(sorted(rem_obs)) if idx == 0 else ())
            )
        return peeled

    return [(dets[idx : idx + 2], obs if idx == 0 else ()) for idx in range(0, n, 2)]


def graphlike_inst_symptom(
    inst: stim.DemInstruction,
    known_graphlike: (
        dict[tuple[int, ...], Symptom]
        | Callable[[], dict[tuple[int, ...], Symptom]]
        | None
    ) = None,
) -> tuple[list[Symptom], Any]:
    """Return ``(graphlike_components, canonical_symptom)`` for an error instruction."""
    comps: list[Symptom] = []
    for d_sorted, o_sorted in raw_inst_components(inst):
        if len(d_sorted) <= 2:
            if d_sorted:
                comps.append((d_sorted, o_sorted))
        else:
            if callable(known_graphlike):
                known_graphlike = known_graphlike()
            comps.extend(
                decompose_hyperedge_component(d_sorted, o_sorted, known_graphlike)
            )
    return comps, canonical_symptom_from_components(comps)


def graphlike_components(
    inst: stim.DemInstruction,
    known_graphlike: (
        dict[tuple[int, ...], Symptom]
        | Callable[[], dict[tuple[int, ...], Symptom]]
        | None
    ) = None,
) -> list[Symptom]:
    """Extract graphlike (<=2 detector) symptom components from a DEM error instruction."""
    return graphlike_inst_symptom(inst, known_graphlike)[0]


def prob_xor(p: float, q: float) -> float:
    """Probability that exactly one of two independent mechanisms fires."""
    return p + q - 2 * p * q


def dem_has_hyperedge(dem: stim.DetectorErrorModel) -> bool:
    """Return True if any error component in ``dem`` affects >2 detectors."""
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        args = inst.args_copy()
        if not args or args[0] <= 0:
            continue
        dets: set[int] = set()
        for t in inst.targets_copy():
            if t.is_separator():
                if len(dets) > 2:
                    return True
                dets.clear()
            elif t.is_relative_detector_id():
                dets ^= {t.val}
        if len(dets) > 2:
            return True
    return False


def decompose_dem_graphlike(
    dem: stim.DetectorErrorModel,
    base_dem: stim.DetectorErrorModel | None = None,
) -> stim.DetectorErrorModel:
    """Ensure every error instruction in ``dem`` is decomposed into <=2-detector components."""
    if not dem_has_hyperedge(dem):
        return dem
    known_graphlike = collect_known_graphlike(base_dem, dem)
    out = stim.DetectorErrorModel()
    for inst in dem.flattened():
        if inst.type != "error":
            out.append(inst)
            continue
        p = float(inst.args_copy()[0])
        _, sym = graphlike_inst_symptom(inst, known_graphlike)
        if sym != ((), ()):
            out.append("error", p, sym_to_targets(sym))
    return out
