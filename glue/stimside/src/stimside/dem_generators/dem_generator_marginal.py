"""Per-shot "marginal" leakage DEM generator for the stimside samplers.

This implements the DEM construction of the marginal decoder in
surface-code-leakage-erasure (Appendix B.3 of surface_code_leakage_erasure.pdf)
for circuits written with stimside's leakage tags:

* ``G_base`` is the DEM of the circuit with every qubit assumed unleaked.
* Every leakage source L (an ``LEAKAGE_TRANSITION_1/2`` branch that takes an
  unleaked qubit to a leaked state) has an "envelope" DEM ``G_L``: the noiseless
  circuit with depolarizing channels wherever the leaked qubit makes the
  simulated trajectory differ from the unleaked one (e.g. a skipped two-qubit
  gate, a role change, the partner of a leakage transition, unleaking, or a
  measurement or gate acting on a leaked qubit after gates skipped it). When
  the leak hops to the partner of a ``LEAKAGE_TRANSITION_2`` with probability
  p < 1, the hop is followed on a separate branch of weight p (only once per
  source: a branch that already hopped does not branch again). Likewise, when
  the leak's next leaked state is random (e.g. ``LEAKAGE_TRANSITION_1:
  (p, 2-->3)``, a ``LEAKAGE_TRANSITION_2`` that changes the leaked qubit's
  state, or a certain hop into one of several states), each state other than
  the one the flow goes on in is followed on a separate branch weighted by its
  probability. Branches never branch again: at a later random transition a
  branch goes on in its own state if it can stay in it, else in its most
  likely state, so only the trajectories that leave the source's root flow at
  most once are followed exactly. A spread (a ``LEAKAGE_TRANSITION_2`` outcome
  where the leaked qubit stays leaked and its unleaked partner leaks too)
  starts the partner's own flow, with an unknown frozen state, at the next pair
  of the op: in the weighted model an extra flow of the source with weight
  P(the leak is there) * p (only a source's root flow spreads, as only it
  hops), and in the loss oracle a source of its own.
* In each shot, a raised ``MPAD[LEAKAGE_MEASUREMENT...]`` flag that is not
  explained by an earlier raised flag on the same leakage flow is an
  erasure-check event E. Its DEM is the disjoint average
  ``G_E = sum_L P(L | E) G_L`` over the sources whose flow can raise it,
  with ``P(L | E)`` proportional to the probability that L happens, is still
  there at the flag and is first detected by it. Like the paper's ideal
  erasure checks, a raised flag is always attributed to a leak.
* Events combine independently, and the shot's DEM is
  ``G_base + (G_E1 xor G_E2 xor ...)``.

Not modelled exactly (see the README):

* A leak's flows assume an unleaked partner in every ``LEAKAGE_TRANSITION_2``,
  so a key with both inputs leaked (e.g. ``(p, 2_3<->3_4)``) is never followed
  and the DEM misses what it does (unleaking included).
* A spread partner's leak is seen by the weighted model only through a flag on
  the partner, like any leak.
* Random leaked-to-leaked transitions: branches never branch again (above), a
  root flow lumps the outcomes that clear its leak, and the loss oracle mixes a
  source's branches instead of selecting them from the shot's leaked-to-leaked
  events.
* A skipped pair's depolarization is placed per instruction, not per pair:
  after the whole instruction for a two-qubit unitary that is not a controlled
  Pauli (a later pair acting on the skipped pair's partner would not spread
  it), before it for a controlled Pauli (a later pair of the leaked qubit with
  another control basis would not get its own). When another pair of the
  instruction uses a qubit of such a depolarization, the analysis is built
  again with the pairs applied one by one, and NotImplementedError is raised
  if any shot's DEM would differ (separate the pairs with TICK).

By default (``decompose_errors=False``), error mechanisms are never decomposed:
every mechanism keeps its full symptom (detectors and observables), so
hyperedges stay hyperedges. Setting ``decompose_errors=True`` decomposes the
DEM into matchable graphlike components using Stim's native decomposition APIs.
Setting ``reweight_only=True`` returns only the per-shot DEM reweight updates
(or, when both ``decompose_errors=True`` and ``reweight_only=True`` are set,
PyMatching ``edge_reweights`` arrays of shape ``(num_reweights, 3)``).
"""

from __future__ import annotations

from typing import Any, Sequence
import warnings
import weakref

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]

from stimside.dem_generators.marginal_builder import (
    Analysis,
    Builder,
    LEAKAGE_SWAP_UNSUPPORTED as LEAKAGE_SWAP_UNSUPPORTED,
)
from stimside.util.leakage_events import ShotLeakageEvents

__all__ = [
    "GeneratedDemList",
    "LEAKAGE_SWAP_UNSUPPORTED",
    "MarginalLeakageDemGenerator",
    "get_item_analysis_meta",
    "get_item_leakage_events",
    "has_leakage_swap",
]


_ITEM_ANALYSIS_META: dict[
    int,
    tuple[
        weakref.ReferenceType[Any],
        Analysis,
        bool,
        bool,
        MarginalLeakageDemGenerator,
        stim.Circuit,
        NDArray[np.bool_],
        ShotLeakageEvents | None,
    ],
] = {}


def _register_item_analysis(
    obj: Any,
    analysis: Analysis,
    decompose_errors: bool,
    reweight_only: bool,
    generator: MarginalLeakageDemGenerator,
    circuit: stim.Circuit,
    record_1d: NDArray[np.bool_],
    leakage_events_1d: ShotLeakageEvents | None = None,
) -> None:
    oid = id(obj)

    def _evict(ref: weakref.ReferenceType[Any], key: int = oid) -> None:
        cur = _ITEM_ANALYSIS_META.get(key)
        if cur is not None and cur[0] is ref:
            del _ITEM_ANALYSIS_META[key]

    _ITEM_ANALYSIS_META[oid] = (
        weakref.ref(obj, _evict),
        analysis,
        decompose_errors,
        reweight_only,
        generator,
        circuit,
        record_1d,
        leakage_events_1d,
    )


def get_item_analysis_meta(
    obj: Any,
) -> tuple[
    Analysis,
    bool,
    bool,
    MarginalLeakageDemGenerator,
    stim.Circuit,
    NDArray[np.bool_],
] | None:
    entry = _ITEM_ANALYSIS_META.get(id(obj))
    if entry is not None and entry[0]() is obj:
        return entry[1:7]
    return None


def get_item_leakage_events(obj: Any) -> ShotLeakageEvents | None:
    """The leakage events of the shot an item was generated for (loss-oracle mode), else None."""
    entry = _ITEM_ANALYSIS_META.get(id(obj))
    if entry is not None and entry[0]() is obj:
        return entry[7]
    return None


class GeneratedDemList(list):
    """A list of per-shot DEMs or edge_reweights arrays carrying circuit/baseline metadata."""

    def __init__(
        self,
        items: Sequence[Any],
        *,
        analysis: Analysis | None = None,
        generator: MarginalLeakageDemGenerator | None = None,
        circuit: stim.Circuit | None = None,
        records: NDArray[np.bool_] | None = None,
        decompose_errors: bool = False,
        reweight_only: bool = False,
        leakage_events: Sequence[ShotLeakageEvents] | None = None,
    ) -> None:
        super().__init__(items)
        self.analysis = analysis
        self.base_dem = analysis.baseline if analysis is not None else None
        self.generator = generator
        self.circuit = circuit
        self.records = records
        self.decompose_errors = decompose_errors
        self.reweight_only = reweight_only
        self.leakage_events = leakage_events

    def __getitem__(self, index: Any) -> Any:
        res = super().__getitem__(index)
        if isinstance(index, slice):
            return GeneratedDemList(
                res,
                analysis=self.analysis,
                generator=self.generator,
                circuit=self.circuit,
                records=self.records[index] if self.records is not None else None,
                decompose_errors=self.decompose_errors,
                reweight_only=self.reweight_only,
                leakage_events=(
                    self.leakage_events[index]
                    if self.leakage_events is not None
                    else None
                ),
            )
        return res


_LEAKAGE_SWAP_ORACLE_WARNING = (
    "SWAP[LEAKAGE_SWAP] is an unsupported feature of MarginalLeakageDemGenerator: the loss "
    "oracle treats it as a plain SWAP and ignores the leakage events it makes, so leakage it "
    "moves is not modelled and the oracle DEMs are approximate."
)


def has_leakage_swap(circuit: stim.Circuit) -> bool:
    """Whether ``circuit`` (REPEAT blocks included) has a ``SWAP[LEAKAGE_SWAP]``."""
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            if has_leakage_swap(op.body_copy()):
                return True
        elif op.name == "SWAP" and op.tag == "LEAKAGE_SWAP":
            return True
    return False


class MarginalLeakageDemGenerator:
    """Per-shot DEM generator implementing the marginal leakage decoder's DEMs.

    Pass an instance to the samplers as ``MarginalDecoder(dem_gen=...)`` (the
    ``dem_decoder`` of ``TablesideSampler``, ``CosetsideSampler``, or
    ``FlipsideSampler``). Called with a circuit and its
    ``(shots, measurements)`` measurement records, it returns one DEM per shot
    (a single DEM for 1D records), built from the ``MPAD[LEAKAGE_MEASUREMENT...]``
    flags of that shot.

    Args:
        unconditional_condition_on_U: must match the op handler's flag of the
            same name (``LeakageUint8`` and ``LeakageUint8Coset`` default to
            True). It decides whether untagged gates skip leaked qubits.
        decompose_errors: if True, decompose the DEM into matchable graphlike
            components (<=2 detectors per component) using Stim's native APIs.
        reweight_only: if True, produce only the DEM reweight update instead of
            the full DEM. When both ``decompose_errors=True`` and
            ``reweight_only=True``, produces PyMatching ``edge_reweights``
            arrays of shape ``(num_reweights, 3)`` with rows
            ``[node1, node2, weight]``.
        loss_oracle: if True, build each shot's DEM from the shot's true
            leakage events (``leakage_events=``, recorded by the simulator)
            instead of its leakage flags: ``G_base + (G_L1 xor G_L2 xor ...)``
            over the sources L that actually leaked in the shot, each included
            with weight 1 (heralds / ``MPAD`` flags are ignored). A shot without
            leakage gets ``G_base``. ``G_L`` is L's full envelope: the disjoint
            mixture of L's root flow and its branches (hops and leaked-state
            changes; the shot's events do not pick one), where a branch forked
            at an op into leaked state m has weight
            P(the leak is still on the root flow just before that op) * p_m
            (with the unleak / move / fork factors of the weighted model) and the root
            has weight 1 - (sum of the branch weights). A source is the
            unleaked-to-leaked transition of a ``LEAKAGE_TRANSITION_1`` target, of
            a ``LEAKAGE_TRANSITION_2`` pair whose partner was unleaked before the
            op, or of the partner of a spread (a ``LEAKAGE_TRANSITION_2`` target
            whose partner was already leaked before the op and stays leaked; its
            flow starts at the next pair of the op); the unleaked-to-leaked event
            of a hop target (whose leaked partner unleaks at the op) is not a
            source and is ignored (its effect is part of the leaked partner's
            source's envelope). Raises NotImplementedError for
            circuits with a qubit listed twice in one leakage transition, and
            ValueError for an unleaked-to-leaked event that matches no source.
            ``SWAP[LEAKAGE_SWAP]``, an unsupported feature (leakage moving between
            qubits is not modelled), is accepted only with ``loss_oracle=True``,
            with a ``UserWarning`` once per analysed circuit: it is treated as a
            plain ``SWAP`` and the unleaked-to-leaked events it makes are ignored,
            so the DEMs are approximate. Without ``loss_oracle`` it raises
            NotImplementedError.
    """

    def __init__(
        self,
        unconditional_condition_on_U: bool = True,
        decompose_errors: bool = False,
        reweight_only: bool = False,
        loss_oracle: bool = False,
    ) -> None:
        self.unconditional_condition_on_U = unconditional_condition_on_U
        self.decompose_errors = bool(decompose_errors)
        self.reweight_only = bool(reweight_only)
        self.loss_oracle = bool(loss_oracle)
        self._circuit: stim.Circuit | None = None
        self._analyses: dict[bool, Analysis] = {}

    @property
    def needs_leakage_events(self) -> bool:
        return self.loss_oracle

    def _get_analysis(
        self, circuit: stim.Circuit, decompose_errors: bool
    ) -> Analysis:
        if self._circuit is None or circuit != self._circuit:
            self._circuit = circuit.copy()
            self._analyses = {}
        if decompose_errors not in self._analyses:
            builder = Builder(circuit, self.unconditional_condition_on_U)
            analysis = builder.build(decompose_errors=decompose_errors, oracle=self.loss_oracle)
            if self.loss_oracle and builder.leakage_swap_ops and not self._analyses:
                warnings.warn(_LEAKAGE_SWAP_ORACLE_WARNING, UserWarning, stacklevel=3)
            self._analyses[decompose_errors] = analysis
        return self._analyses[decompose_errors]

    def base_dem(
        self, circuit: stim.Circuit, *, decompose_errors: bool | None = None
    ) -> stim.DetectorErrorModel:
        """Return the baseline DEM for ``circuit`` (matchable when ``decompose_errors=True``)."""
        dec = self.decompose_errors if decompose_errors is None else bool(decompose_errors)
        analysis = self._get_analysis(circuit, dec)
        return (
            analysis.matching_base_dem.copy()
            if dec
            else analysis.baseline.copy()
        )

    def __call__(
        self,
        circuit: stim.Circuit,
        records: NDArray[np.bool_],
        decompose_errors: bool | None = None,
        reweight_only: bool | None = None,
        *,
        leakage_events: Sequence[ShotLeakageEvents] | ShotLeakageEvents | None = None,
    ) -> (
        stim.DetectorErrorModel
        | list[stim.DetectorErrorModel]
        | NDArray[np.float64]
        | list[NDArray[np.float64]]
    ):
        dec = self.decompose_errors if decompose_errors is None else bool(decompose_errors)
        rew = self.reweight_only if reweight_only is None else bool(reweight_only)
        analysis = self._get_analysis(circuit, dec)

        records_arr = np.array(records, dtype=bool, copy=True)
        if (
            records_arr.ndim not in (1, 2)
            or records_arr.shape[-1] != analysis.num_measurements
        ):
            raise ValueError(
                f"Expected records of shape (shots, {analysis.num_measurements}) or "
                f"({analysis.num_measurements},), got {records_arr.shape}."
            )
        rows = records_arr.reshape(-1, records_arr.shape[-1])
        shot_events: list[ShotLeakageEvents] | None = None
        if analysis.oracle:
            if leakage_events is None:
                raise ValueError(
                    "MarginalLeakageDemGenerator(loss_oracle=True) needs the shots' "
                    "leakage_events (record them with the simulator's "
                    "record_leakage_events=True)."
                )
            shot_events = (
                [leakage_events]  # type: ignore[list-item]
                if records_arr.ndim == 1
                else list(leakage_events)  # type: ignore[arg-type]
            )
            if len(shot_events) != rows.shape[0]:
                raise ValueError(
                    f"Got leakage_events for {len(shot_events)} shots but records "
                    f"for {rows.shape[0]} shots."
                )
            num_sources = len(analysis.flag_env_offsets) - 1
            raised = np.zeros((rows.shape[0], num_sources), dtype=bool)
            for shot_idx, evs in enumerate(shot_events):
                matched, _ = _match_shot_sources(analysis, evs)
                raised[shot_idx, matched] = True
        else:
            raised = rows[:, analysis.flag_records] ^ analysis.flag_invert

        items = analysis.generate_batch(
            raised, decompose_errors=dec, reweight_only=rew
        )
        assert self._circuit is not None
        if records_arr.ndim == 1:
            single_out = items[0]
            _register_item_analysis(
                single_out,
                analysis,
                dec,
                rew,
                generator=self,
                circuit=self._circuit,
                record_1d=records_arr,
                leakage_events_1d=shot_events[0] if shot_events is not None else None,
            )
            return single_out
        seen_ids: set[int] = set()
        for shot_idx, item_obj in enumerate(items):
            oid = id(item_obj)
            if oid not in seen_ids:
                seen_ids.add(oid)
                _register_item_analysis(
                    item_obj,
                    analysis,
                    dec,
                    rew,
                    generator=self,
                    circuit=self._circuit,
                    record_1d=rows[shot_idx].copy(),
                    leakage_events_1d=(
                        shot_events[shot_idx] if shot_events is not None else None
                    ),
                )
        return GeneratedDemList(
            items,
            analysis=analysis,
            generator=self,
            circuit=self._circuit,
            records=records_arr,
            decompose_errors=dec,
            reweight_only=rew,
            leakage_events=shot_events,
        )


def _match_shot_sources(
    analysis: Analysis, events: ShotLeakageEvents
) -> tuple[list[int], int]:
    """(sorted indices of the sources that leaked in a shot, number of ignored events).

    Replays the shot's events op by op (stable-sorted by op index). An
    unleaked-to-leaked event is ignored and counted if it is at a
    ``LEAKAGE_TRANSITION_2`` op whose pair partner was leaked before the op and
    unleaks at it (a hop target), or at a ``SWAP[LEAKAGE_SWAP]`` (a leak moved
    onto the qubit, which the oracle does not model). One whose partner was
    leaked before the op and stays leaked (a spread) matches a spread source;
    every other unleaked-to-leaked event must match a source.
    """
    assert analysis.unrolled_to_flat is not None
    u2f = analysis.unrolled_to_flat
    ordered = sorted(events, key=lambda ev: ev.op_index)
    leaked: dict[int, int] = {}
    matched: set[int] = set()
    ignored = 0
    i = 0
    while i < len(ordered):
        op_index = ordered[i].op_index
        j = i
        while j < len(ordered) and ordered[j].op_index == op_index:
            j += 1
        group = ordered[i:j]
        for ev in group:
            if not (ev.old_state < 2 <= ev.new_state):
                continue
            flat_j = int(u2f[op_index]) if 0 <= op_index < len(u2f) else -1
            if flat_j in analysis.leakage_swap_ops:
                ignored += 1
                continue
            partner = analysis.lt2_partner.get((flat_j, ev.qubit))
            if partner is not None and leaked.get(partner, 0) >= 2:
                # A hop target (the partner unleaks at this op: part of the partner's source's
                # envelope) is ignored; a spread (the partner stays leaked) is a source of its own.
                if any(e.qubit == partner and e.new_state < 2 for e in group):
                    ignored += 1
                    continue
                src = analysis.spread_lookup.get((flat_j, ev.qubit, ev.new_state))
            else:
                src = analysis.source_lookup.get((flat_j, ev.qubit, ev.new_state))
            if src is None:
                raise ValueError(f"Leakage event {ev} matches no leakage source of the circuit.")
            matched.add(src)
        for ev in group:
            if ev.new_state >= 2:
                leaked[ev.qubit] = ev.new_state
            else:
                leaked.pop(ev.qubit, None)
        i = j
    return sorted(matched), ignored
