"""Simulator-level checks of the engine's block command (switch on vs off).

With `_LIVE_STORE_ENGINE` on, `_flush_magic` issues one `LiveExpansions.apply_block` engine
command per compiled block (grouping of identical shots, fan-out copies and the `max_components`
bound inside the engine). Everything observable must equal the reference list path bit for bit:
records, expansions, branch masses, RNG streams. B3's Python grouping
(`simulator_magic._group_shots_by_state`) stays the oracle: at every flush the engine's partition
must equal it.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from typing import Any

import pytest
import stim  # type: ignore[import-untyped]

import stimside._magic.phase_qfe as pq
import stimside.simulator_magic as sm
from stimside import golden_test as gt
from stimside._magic.live_store import LiveExpansions, block_table
from stimside._magic.plan import CompiledOpenBlockRecord
from stimside._magic.query import QueryBudgetExceededError
from stimside.simulator_magic import MagicsideSimulator
from stimside.simulator_magic_dedupe_test import _MAGIC_PAIRS, _state_digest

_EXPECTED = json.loads(gt.DIGESTS_PATH.read_text(encoding="utf-8"))
_ORIG_RUN = MagicsideSimulator.run
_ORIG_COMPILE = sm.compile_open_block


class _BlockStore(LiveExpansions):
    """Records every engine block command; `dedupe` and `check_partition` are class switches."""

    dedupe = True
    check_partition = True
    commands = 0
    groups = 0
    live = 0
    mismatches = 0

    def apply_block(self, table: Any, max_components: Any, dedupe: bool = True) -> tuple[int, int]:
        cls = type(self)
        cls.commands += 1
        if cls.check_partition:
            rep = self.engine.partition(cls.dedupe)
            exps = [self[s] for s in range(len(self))]
            # B3's grouping is the oracle; with dedupe off every live shot is its own group.
            want = sm._group_shots_by_state(exps) if cls.dedupe else live_singletons(exps)
            got: dict[int, list[int]] = {}
            for s, r in enumerate(rep.tolist()):
                if r >= 0:
                    got.setdefault(r, []).append(s)
            cls.mismatches += sorted(got.values()) != sorted(want)
        g, n = super().apply_block(table, max_components, cls.dedupe)
        cls.groups += g
        cls.live += n
        return g, n


def live_singletons(exps: list[Any]) -> list[list[int]]:
    return [[s] for s, e in enumerate(exps) if e.chi > 0]


def _setup(monkeypatch: pytest.MonkeyPatch, on: bool, dedupe: bool = True) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", on)
    monkeypatch.setattr(sm, "LiveExpansions", _BlockStore)
    _BlockStore.dedupe = dedupe
    _BlockStore.commands = _BlockStore.groups = _BlockStore.live = _BlockStore.mismatches = 0


def _run_case(
    monkeypatch: pytest.MonkeyPatch, case: tuple[str, str, bool, str]
) -> tuple[str, list[str]]:
    name, mode, has_leak, text = case
    finals: list[str] = []

    def run(self: MagicsideSimulator) -> Any:
        res = _ORIG_RUN(self)
        finals.append(_state_digest(self))
        return res

    monkeypatch.setattr(MagicsideSimulator, "run", run)
    mb = name == "c16_sparse_multibatch"
    digest = gt._run_case(
        stim.Circuit(text), mode, has_leak, batch=70 if mb else gt.BATCH, multibatch_packed=mb
    )
    return digest, finals


def test_every_golden_digest_is_unchanged_with_blocks_in_the_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup(monkeypatch, True)
    actual = gt._compute_all_digests()
    assert actual == _EXPECTED
    # Most golden cases take the compiled path (plain list); the live ones still send blocks.
    assert _BlockStore.commands >= 5 and _BlockStore.live > _BlockStore.groups
    assert _BlockStore.mismatches == 0  # the engine's partition == B3's Python grouping, always


@pytest.mark.parametrize("case", _MAGIC_PAIRS, ids=[f"{c[0]}-{c[1]}" for c in _MAGIC_PAIRS])
def test_block_dedupe_on_and_off_and_the_list_path_agree_on_every_magic_golden_pair(
    case: tuple[str, str, bool, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Records, final live states, branch masses and RNG states: engine dedupe on == engine
    dedupe off == the reference list path == the committed golden digest."""
    results = []
    for on, dedupe in ((False, True), (True, True), (True, False)):
        _setup(monkeypatch, on, dedupe)
        results.append(_run_case(monkeypatch, case))
        assert on or _BlockStore.commands == 0
        assert _BlockStore.mismatches == 0
    assert results[0] == results[1] == results[2]
    assert results[0][0] == _EXPECTED[case[0]][case[1]]


_LIVE_CIRCUITS = sorted(
    {
        text for _, modes, leak, text in gt.CORPUS
        if not leak and any(m.startswith("magic_") for m in modes)
    }
)


def _live_run(text: str, batch: int, seed: int) -> tuple[str, bytes]:
    circ = stim.Circuit(text)
    sim = MagicsideSimulator(circ, batch_size=batch, seed=seed)
    sim.interactive_do(circ)
    sim.finish_interactive_run()
    return _state_digest(sim), sim.get_final_measurement_records().tobytes()


@pytest.mark.parametrize("idx", range(len(_LIVE_CIRCUITS)))
def test_every_corpus_circuit_run_live_agrees_with_the_list_path_for_both_dedupe_settings(
    idx: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole corpus through the live (interactive) path, where every block is an engine
    command: final states, branch masses, RNG state and records equal the list path's."""
    text, out = _LIVE_CIRCUITS[idx], []
    for on, dedupe in ((False, True), (True, True), (True, False)):
        _setup(monkeypatch, on, dedupe)
        out.append(_live_run(text, 8, 11))
        assert _BlockStore.mismatches == 0
    assert out[0] == out[1] == out[2]


_NOISY = """R 0 1 2
H 0 1 2
I[T] 0 1
Z_ERROR(0.5) 0 1 2
H 0
I[T] 0 2
X_ERROR(0.5) 1
H 1 2
I[T] 1 2
Y_ERROR(0.3) 0 2
H 0 2
I[T] 0 1 2
M 0 1 2"""


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("batch", [3, 16, 64])
def test_noisy_circuit_with_mixed_group_sizes_agrees_with_the_list_path(
    seed: int, batch: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = []
    for on, dedupe in ((False, True), (True, True), (True, False)):
        _setup(monkeypatch, on, dedupe)
        sim = MagicsideSimulator(stim.Circuit(_NOISY), batch_size=batch, seed=seed)
        sim.run()
        out.append((_state_digest(sim), sim.get_final_measurement_records().tobytes()))
        assert _BlockStore.mismatches == 0
    assert out[0] == out[1] == out[2]
    assert _BlockStore.commands > 0


_D2 = "R 0 1 2\nH 0 1 2\nI[T] 0 1\nI[CLIFFORD_FAULT: (1.0, CZ)] 0 1\nI[CCZ] 0 1 2\nI[T] 2"


def test_fan_out_copies_are_never_shared_between_shots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Six identical shots form one group; each must own its arrays after the engine command
    (a shared copy would let an edit of one shot change the others)."""
    _setup(monkeypatch, True)
    circ = stim.Circuit(_D2)
    sim = MagicsideSimulator(circ, batch_size=6, seed=1)
    sim._do(circ)
    sim._flush_magic()
    store = sim._live_exps
    assert isinstance(store, LiveExpansions)
    before = [store[s] for s in range(6)]
    assert len({sm._state_key(e) for e in before}) == 1  # one group
    e2 = store[2]
    e2.weights = e2.weights * 2.0
    store[2] = e2  # edit one shot and write it back
    after = [store[s] for s in range(6)]
    for s in (0, 1, 3, 4, 5):
        assert sm._state_key(after[s]) == sm._state_key(before[s])
    assert sm._state_key(after[2]) != sm._state_key(before[2])


def _flush_error(monkeypatch: pytest.MonkeyPatch, on: bool, cap: int) -> tuple[str, str]:
    _setup(monkeypatch, on)
    circ = stim.Circuit("H 0 1 2\nI[T] 0\nZ_ERROR(0.5) 0 1 2\nI[T] 1\nI[CCZ] 0 1 2\nH 0 1 2\nM 0")
    sim = MagicsideSimulator(circ, batch_size=8, seed=3, max_components=cap,
                             on_budget_exceeded="raise")
    with pytest.raises(QueryBudgetExceededError) as exc:
        sim.run()
    return str(exc.value), _state_digest(sim)


@pytest.mark.parametrize("cap", [1, 2, 4])
def test_max_components_exception_and_partial_state_match_the_list_path(
    cap: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = _flush_error(monkeypatch, False, cap)
    eng = _flush_error(monkeypatch, True, cap)
    assert eng == ref and "max_components=" in eng[0]
    assert _BlockStore.commands >= 1


@pytest.mark.skipif(
    sm._LIVE_STORE_ENGINE is not True,
    reason="pins the ON default: forcing the switch OFF (to run the suite on the reference "
    "list) skips this default pin, not a regression",
)
def test_the_default_uses_the_engine_path_and_switch_off_keeps_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stimside._magic.live_store import engine_numerics_available

    monkeypatch.setattr(sm, "LiveExpansions", _BlockStore)
    _BlockStore.commands = 0
    assert sm._LIVE_STORE_ENGINE is True
    sim = MagicsideSimulator(stim.Circuit(_NOISY), batch_size=4, seed=1)
    sim.run()
    if engine_numerics_available():
        assert isinstance(sim._live_exps, LiveExpansions) and _BlockStore.commands >= 1
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", False)
    _BlockStore.commands = 0
    ref = MagicsideSimulator(stim.Circuit(_NOISY), batch_size=4, seed=1)
    ref.run()
    assert isinstance(ref._live_exps, list) and _BlockStore.commands == 0


_CAPPED = "H 0 1 2\nI[T] 0\nZ_ERROR(0.5) 0 1 2\nI[T] 1\nI[CCZ] 0 1 2\nH 0 1 2\nM 0"


@pytest.mark.parametrize("cap", [1, 2])
def test_max_components_is_not_enforced_when_the_budget_policy_is_fallback(
    cap: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`on_budget_exceeded="fallback"` never raises for the bound (the reference passes no cap):
    the engine run completes with the same state as the list path, over the same circuit that
    raises with "raise" (an engine given the cap anyway would raise here)."""
    results = []
    for on in (False, True):
        _setup(monkeypatch, on)
        sim = MagicsideSimulator(stim.Circuit(_CAPPED), batch_size=8, seed=3, max_components=cap,
                                 on_budget_exceeded="fallback")
        sim.run()
        results.append((_state_digest(sim), sim.get_final_measurement_records().tobytes()))
        assert not on or _BlockStore.commands >= 1  # the engine really ran the blocks
        assert max(e.chi for e in sim._live_exps) > cap  # the bound was exceeded, silently
    assert results[0] == results[1]
    _flush_error(monkeypatch, True, cap)  # and with "raise" the same circuit does raise


def _patched_blocks(monkeypatch: pytest.MonkeyPatch, wrap: Any) -> None:
    """Every compiled block goes through `wrap` (always applied to the ORIGINAL compiler's)."""

    def patched(records: Any, **kw: Any) -> Any:
        return tuple(
            CompiledOpenBlockRecord(wrap(r.block), r.axes)
            if isinstance(r, CompiledOpenBlockRecord) else r
            for r in _ORIG_COMPILE(records, **kw)
        )

    monkeypatch.setattr(sm, "compile_open_block", patched)


def _bad_coeff(blk: pq.CompiledPauliBlock) -> pq.CompiledPauliBlock:
    if blk.mode != "pauli_sum" or blk.coeffs is None:
        return blk
    c = blk.coeffs.copy()
    c[0] = complex("inf")
    return dataclasses.replace(blk, coeffs=c)


def _bogus_mode_block(n_qubits: int) -> pq.CompiledPauliBlock:
    mode: Any = "bogus"  # not a legal Literal: that is the point
    return pq.CompiledPauliBlock(n_qubits=n_qubits, mode=mode)


_MALFORMED = {
    "bad_mode": lambda b: _bogus_mode_block(b.n_qubits),
    "wrong_n_qubits": lambda b: dataclasses.replace(b, n_qubits=b.n_qubits + 1),
    "infinite_coeff": _bad_coeff,
    "nested_block": lambda b: pq.CompiledPauliBlock(
        n_qubits=b.n_qubits, mode="sequential", steps=(b,)),
    "nan_scalar": lambda b: dataclasses.replace(b, global_scalar=complex("nan")),
}


_RUNS_WITHOUT_ERROR = {"nested_block"}


@pytest.mark.parametrize("name", sorted(_MALFORMED))
def test_a_block_the_reference_rejects_or_nests_never_reaches_the_engine_and_matches_it(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`block_table` validates like the reference and returns None for a block the engine must
    not run, so `_flush_magic` falls back to `apply_compiled_block`, which raises what it always
    did: same exception type and text and the same state with the engine switch on and off. (A
    `block_table` that skipped the validation would hand the engine a table it must not run.)"""
    outcomes = []
    for on in (False, True):
        _setup(monkeypatch, on)
        _patched_blocks(monkeypatch, _MALFORMED[name])
        sim = MagicsideSimulator(stim.Circuit(_NOISY), batch_size=8, seed=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                sim.run()
                raised: tuple[str, str] | None = None
            except (FloatingPointError, ValueError, TypeError) as exc:  # compared below
                raised = (type(exc).__name__, str(exc))
        outcomes.append((raised, _state_digest(sim)))
        assert not on or _BlockStore.commands == 0, name  # the engine was never asked
    assert outcomes[0] == outcomes[1], name
    # (a nested block and an infinite coefficient in a non-pauli_sum are legal for the reference:
    # they run, via the per-group path, to the same state; the rest raise.)
    assert (outcomes[0][0] is None) == (name in _RUNS_WITHOUT_ERROR), (name, outcomes[0][0])


def test_block_table_declines_what_the_reference_validation_rejects() -> None:
    n = 5
    ok = pq.CompiledPauliBlock(n_qubits=n, mode="pauli_sum", is_identity=True)
    assert block_table(ok, n) is not None
    assert block_table(ok, n + 1) is None  # n_qubits mismatch
    assert block_table(dataclasses.replace(ok, global_scalar=complex("inf")), n) is None
    assert block_table(_bogus_mode_block(n), n) is None
    assert block_table(ok, n) is not None and block_table("not a block", n) is None
    nested = pq.CompiledPauliBlock(n_qubits=n, mode="sequential", steps=(ok,))
    assert block_table(nested, n) is None
