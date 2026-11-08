"""The engine measurement path inside whole simulations (WP-25 stage S5a).

`_project_substep_batch` with the shots in the C++ engine (`_LIVE_STORE_ENGINE` on) is compared with
the reference path (a plain list) on complete runs: the golden corpus of every `magic_*` mode
(leakage handlers, sync RNG, heralded noise, feedback), a fuzz of 96 random circuits (T / T_DAG /
CCZ, Cliffords, every measurement and reset kind, noise, several batch sizes, sync and not), and
an interactive replay. After each run the records, every shot's expansion, the branch masses, the
numpy generator state and every per-shot / tableau stream must be byte-identical: a missing or
extra random draw anywhere shows in a generator state. A counter shows the engine path really ran.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

import stimside._native.magic_engine as me
import stimside.simulator_magic as sm
from stimside import golden_test as gt
from stimside._magic.live_store import LiveExpansions
from stimside.simulator_magic import MagicsideSimulator

_ARRAYS = ("weights", "h_words", "ell", "A_words", "a_union", "Gamma", "pivots", "ell_odd")
_PAIRS = [
    (name, mode, has_leak, text)
    for name, modes, has_leak, text in gt.CORPUS
    for mode in modes
    if mode.startswith("magic_")
]
_EXPECTED = json.loads(gt.DIGESTS_PATH.read_text(encoding="utf-8"))


def _exp_bytes(e: Any) -> bytes:
    h = hashlib.sha256()
    for name in _ARRAYS:
        a = getattr(e, name)
        h.update(a.tobytes() + str(a.shape).encode())
    meta = (e.n_qubits, e.d, e.chi, e.log_mass.hex(), e.atol, e._is_canonical_merged)
    h.update(repr(meta).encode())
    return h.digest()


def state_digest(sim: MagicsideSimulator) -> str:
    """Every live expansion, the branch masses and every random stream, as one hash."""
    h = hashlib.sha256()
    for b in range(sim.batch_size):
        h.update(_exp_bytes(sim._live_exps[b]))
    h.update(np.asarray(sim._log_branch_masses).tobytes())
    h.update(repr(sim.np_rng.bit_generator.state).encode())
    for rng in sim._shot_np_rngs or []:
        h.update(repr(rng.bit_generator.state).encode())
    for tz, tx in zip(sim._tab_rngs or [], sim._tab_rngs_x or []):
        h.update(repr((tz.current_inverse_tableau(), tx.current_inverse_tableau())).encode())
    return h.hexdigest()


class _Counter:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.splits = self.finishes = self.active = 0
        real_s, real_f = me.MagicBatchEngine.measure_split, me.MagicBatchEngine.measure_finish
        outer = self

        def split(eng: Any, *a: Any, **k: Any) -> Any:
            res = real_s(eng, *a, **k)
            outer.splits += 1
            outer.active += res.n_active
            return res

        def finish(eng: Any, *a: Any, **k: Any) -> Any:
            outer.finishes += 1
            return real_f(eng, *a, **k)

        monkeypatch.setattr(me.MagicBatchEngine, "measure_split", split)
        monkeypatch.setattr(me.MagicBatchEngine, "measure_finish", finish)


def _run_pair(
    monkeypatch: pytest.MonkeyPatch, case: tuple[str, str, bool, str]
) -> tuple[tuple[str, list[str]], tuple[str, list[str]], _Counter]:
    """(golden digest, final state digest per run()) with the engine OFF, then ON."""
    name, mode, has_leak, text = case
    finals: list[str] = []
    real_run = MagicsideSimulator.run

    def run(self: MagicsideSimulator) -> Any:
        res = real_run(self)
        finals.append(state_digest(self))
        return res

    monkeypatch.setattr(MagicsideSimulator, "run", run)
    mb = name == "c16_sparse_multibatch"
    results = []
    counter = _Counter(monkeypatch)
    for engine in (False, True):
        monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
        finals.clear()
        digest = gt._run_case(
            stim.Circuit(text), mode, has_leak, batch=70 if mb else gt.BATCH, multibatch_packed=mb
        )
        results.append((digest, list(finals)))
        if not engine:
            assert counter.splits == 0  # the reference path never touches the engine commands
    return results[0], results[1], counter


@pytest.mark.parametrize("case", _PAIRS, ids=[f"{c[0]}-{c[1]}" for c in _PAIRS])
def test_every_magic_golden_pair_is_byte_identical_with_the_engine_measurement_path(
    case: tuple[str, str, bool, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Records (golden digest), final states, masses and all random streams: engine == reference,
    and both equal the committed digest (never re-blessed)."""
    off, on, counter = _run_pair(monkeypatch, case)
    assert off == on
    assert on[0] == _EXPECTED[case[0]][case[1]]
    assert counter.finishes <= counter.splits


def test_the_golden_corpus_really_runs_through_split_and_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards the comparisons above against being vacuous (a silent fallback to the list)."""
    totals = _Counter(monkeypatch)
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", True)
    for case in _PAIRS:
        if case[1] in ("magic_ds", "magic_sync"):
            name, mode, has_leak, text = case
            mb = name == "c16_sparse_multibatch"
            gt._run_case(
                stim.Circuit(text), mode, has_leak, batch=70 if mb else gt.BATCH,
                multibatch_packed=mb,
            )
    assert totals.splits >= 150 and totals.finishes >= 100 and totals.active >= 2000, vars(totals)


# --------------------------------------------------------------------------------------------
# A fuzz of random circuits
# --------------------------------------------------------------------------------------------

_MEAS = ("M", "MX", "MY", "MR", "MRX", "MRY", "R", "RX", "RY")
_CLIFF = ("H", "S", "S_DAG", "SQRT_X", "H_YZ")
_PRODUCTS = ("MPP", "MXX", "MYY", "MZZ")


def random_circuit(seed: int) -> str:
    """A legal Magicside circuit from `seed`: T / T_DAG / CCZ spread over Cliffords, CX / CZ,
    noise and every measurement / reset kind (a measurement only before the end, so that later
    instructions run on the projected states)."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 7)) if seed % 3 else int(rng.integers(65, 72))
    def qs(k: int) -> list[int]:
        return [int(q) for q in rng.choice(min(n, 9), size=k, replace=False)]

    allq = " ".join(map(str, range(min(n, 9))))
    lines = [f"R {allq}", f"H {allq}"]
    for _ in range(int(rng.integers(8, 16))):
        r = rng.random()
        if r < 0.28:
            lines.append(f"I[{rng.choice(['T', 'T_DAG'])}] {qs(1)[0]}")
        elif r < 0.33:
            lines.append("I[CCZ] " + " ".join(map(str, qs(3))))
        elif r < 0.45:
            lines.append(f"{rng.choice(_CLIFF)} {qs(1)[0]}")
        elif r < 0.55:
            a, b = qs(2)
            lines.append(f"{rng.choice(['CX', 'CZ'])} {a} {b}")
        elif r < 0.62:
            err = rng.choice(["X_ERROR", "Z_ERROR", "Y_ERROR"])
            lines.append(f"{err}({rng.choice([0.2, 0.5])}) {qs(1)[0]}")
        elif r < 0.80:
            m = str(rng.choice(_MEAS))
            lines.append(f"{m}{'(0.1)' if m[0] == 'M' and rng.random() < 0.3 else ''} {qs(1)[0]}")
        elif r < 0.90:
            a, b = qs(2)
            lines.append(f"{rng.choice(_PRODUCTS[1:])} {a} {b}")
        else:
            a, b = qs(2)
            lines.append(f"MPP X{a}*Z{b}")
    lines.append("M " + " ".join(map(str, range(min(n, 9)))))
    return "\n".join(lines)


def _fuzz_run(seed: int, engine: bool, sync: bool, batch: int, monkeypatch: Any) -> tuple[str, str]:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    sim = MagicsideSimulator(
        stim.Circuit(random_circuit(seed)), batch_size=batch, seed=seed, sync_tableside_rng=sync
    )
    sim.run()
    recs = hashlib.sha256(sim.get_final_measurement_records().tobytes()).hexdigest()
    if engine:
        assert isinstance(sim._live_exps, LiveExpansions)
    return recs, state_digest(sim)


@pytest.mark.parametrize("seed", range(96))
def test_96_random_circuits_run_identically_on_the_engine_and_the_reference(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync = seed % 4 == 3
    batch = (1, 2, 3, 8)[seed % 4] if seed % 2 else (5, 16)[(seed // 2) % 2]
    counter = _Counter(monkeypatch)
    off = _fuzz_run(seed, False, sync, batch, monkeypatch)
    assert counter.splits == 0
    on = _fuzz_run(seed, True, sync, batch, monkeypatch)
    assert on == off, random_circuit(seed)
    assert counter.splits >= 1  # every circuit measures at least once through the engine


def test_the_fuzz_reaches_active_shots_rotations_and_resets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = _Counter(monkeypatch)
    for seed in range(0, 96, 3):
        _fuzz_run(seed, True, False, 8, monkeypatch)
    assert c.splits >= 150 and c.active >= 500, vars(c)


def test_interactive_replay_with_measurements_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same circuit fed instruction by instruction (`interactive_do`), then finished."""
    text = random_circuit(5)
    digests = []
    for engine in (False, True):
        monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
        sim = MagicsideSimulator(stim.Circuit(text), batch_size=6, seed=11)
        for op in stim.Circuit(text):
            sim.interactive_do(op)
        sim.finish_interactive_run()
        digests.append((sim.get_final_measurement_records().tobytes(), state_digest(sim)))
    assert digests[0] == digests[1]


def test_clear_and_rerun_reuses_the_engine_and_stays_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`clear()` resets the store, which drops any pending split of the previous run."""
    out = []
    for engine in (False, True):
        monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
        sim = MagicsideSimulator(stim.Circuit(random_circuit(9)), batch_size=4, seed=2)
        sim.run()
        first = state_digest(sim)
        sim.clear()
        sim.run()
        out.append((first, state_digest(sim)))
    assert out[0] == out[1]
