"""Tests for the Python<->C++ shot-state sync of ``CosetsideSimulator`` (WP-19).

``_sync_py_to_cpp`` re-imports every active shot into the C++ engine, so it must run only when a
Python writer changed the shot state (``_py_dirty``); ``_cpp_dirty`` must be raised by every C++
kernel that mutates it. Because the guard trusts those flags, each flag-setting site is
load-bearing and is pinned here, at batch sizes 64 and 256 (a sync/import that is skipped or
re-armed only for shots/batches beyond one 64-bit word must not go unnoticed):

* T1 the number of ``engine_import_shot`` calls is O(#syncs x #active), not O(#peeks x #active),
  also after a C++->Python sync that writes nothing;
* T2 the fix does not change records (digest of a pinned-seed run, taken on the pre-fix code);
* T3 differential tests (C++ vs the ``_use_cpp=False`` path, and a seeded fuzz whose output
  digest was taken on the pre-fix always-re-import code) that fail if a dirty flag is not set;
  they include a d==0 shot injected through the property and an emptied shot dict;
* T4 pins the documented semantics: ``_py_dirty`` is cleared by each sync, so a held
  MRQFState/dict reference mutated after a peek is NOT re-pushed unless ``active_shots`` is read
  again;
* T5 exception paths: a Python writer that raises midway has already mutated some shots, so it
  raises ``_py_dirty`` BEFORE its per-shot loop; and an import interrupted midway leaves
  ``_py_dirty`` set, so the next peek re-imports everything (``engine_clear`` + all shots);
  the flag is raised before the FIRST per-shot write, also when the property is read while the
  dict is empty (T5b), and an interrupt after a mutating C++ kernel returns but before
  ``_cpp_dirty`` is set leaves the engine to be rolled back to the Python state (T6).
"""

import ctypes
import hashlib
from collections.abc import Callable
from typing import Literal, cast

import numpy as np
import pytest
import stim

from stimside._coset.mrqf_state import MRQFState
from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
from stimside.simulator_coset import CosetsideSimulator

_Pauli = Literal["X", "Y", "Z"]
_PAULIS: tuple[_Pauli, ...] = ("X", "Y", "Z")
_FUZZ_BATCH = {64: 8, 256: 128}  # fuzz batch per test batch (128 shots span two 64-bit words)
_RUN_DIGESTS = {64: "d84158c5b3af7ad7", 256: "a5ca2f3263099f01"}  # pre-fix (22237da) code
_FUZZ_DIGESTS = {64: "4ee18546550ea8e8", 256: "c3b1cfd6fa4ef0f2"}  # pre-fix (22237da) code
_CIRCUIT = stim.Circuit(
    """
    R 0 1 2 3
    H 0 1 2 3
    I[LEAKAGE_TRANSITION_1: (0.5, U-->2)] 0
    CZ[CONDITIONED_ON_PAIR: (U, U)] 0 1 2 3
    TICK
    I[LEAKAGE_TRANSITION_1: (0.2, 1-->2)] 1 2 3
    TICK
    I[LEAKAGE_TRANSITION_1: (0.2, 1-->2)] 1 2 3
    TICK
    I[LEAKAGE_TRANSITION_1: (0.2, 1-->2)] 1 2 3
    TICK
    H 1 2 3
    M 0 1 2 3
    """
)
_DET_MEASURE_CIRCUIT = stim.Circuit(
    """
    R 0 1 2 3
    H 0 1 2 3
    I[LEAKAGE_TRANSITION_1: (0.5, U-->2)] 0
    CZ[CONDITIONED_ON_PAIR: (U, U)] 0 1 2 3
    TICK
    CZ 0 1 2 3
    H 0 1 2 3
    TICK
    M 0 1 2 3
    """
)
_CX_CIRCUIT = stim.Circuit(
    str(_DET_MEASURE_CIRCUIT[:4]) + "\nTICK\nCX 0 1 1 2 2 3\nTICK\nM 0 1 2 3"
)  # overlapping CX groups: do_conditional_clifford takes its Python (non-disjoint) path
_DISJOINT_CX_CIRCUIT = stim.Circuit(
    str(_DET_MEASURE_CIRCUIT[:4]) + "\nTICK\nCX 0 1 2 3\nTICK\nM 0 1 2 3"
)  # disjoint CX groups: do_conditional_clifford takes its C++ kernel path
_N_SETUP = 4  # R, H, leakage transition, skipped CZ: leaves active (superposed) shots


@pytest.fixture(params=[64, 256], ids=["B64", "B256"])
def batch(request: pytest.FixtureRequest) -> int:
    return int(request.param)


class _CountingLib:
    """Delegates to the real ctypes library and counts selected ``engine_*`` calls.

    ``fail_import_call=n`` makes the n-th ``engine_import_shot`` raise KeyboardInterrupt (once).
    """

    def __init__(self, lib: ctypes.CDLL, fail_import_call: int | None = None) -> None:
        self._lib = lib
        self.fail_import_call = fail_import_call
        self.counts = {"engine_import_shot": 0, "engine_clear": 0}

    def __getattr__(self, name: str):
        fn = getattr(self._lib, name)
        if name not in self.counts:
            return fn

        def counted(*args):
            self.counts[name] += 1
            if name == "engine_import_shot" and self.counts[name] == self.fail_import_call:
                raise KeyboardInterrupt
            return fn(*args)

        return counted


def _install_counting_lib(sim: CosetsideSimulator, **kwargs) -> _CountingLib:
    assert sim._cpp_lib is not None
    lib = _CountingLib(sim._cpp_lib, **kwargs)
    sim._cpp_lib = cast(ctypes.CDLL, lib)
    return lib


def _make_sim(
    batch: int, use_cpp: bool = True, circuit: stim.Circuit = _CIRCUIT, seed: int = 11
) -> CosetsideSimulator:
    handler = LeakageUint8Coset(unconditional_condition_on_U=False).compile_op_handler(
        circuit=circuit, batch_size=batch
    )
    sim = CosetsideSimulator(circuit, compiled_op_handler=handler, batch_size=batch, seed=seed)
    sim._use_cpp = use_cpp
    return sim


def _setup_sim(
    batch: int, use_cpp: bool = True, circuit: stim.Circuit = _CIRCUIT
) -> CosetsideSimulator:
    sim = _make_sim(batch, use_cpp, circuit)
    for inst in list(circuit)[:_N_SETUP]:
        sim.interactive_do(inst)
    assert len(sim.active_shots) > 0  # property read: cpp->py sync (``_active_shots`` is stale)
    return sim


def _postselect(sim: CosetsideSimulator, pauli: _Pauli = "Z") -> None:
    ones = np.ones(sim.batch_size, dtype=bool)
    sim.postselect_z_batch(0, ones, np.zeros(sim.batch_size, dtype=np.uint8), pauli)


def _peek_all(sim: CosetsideSimulator) -> np.ndarray:
    qs = list(range(sim.num_qubits))
    return np.vstack([sim.peek_pauli_batch(qs, p) for p in _PAULIS])


def _shot_state(sim: CosetsideSimulator) -> dict:
    """Active-shot contents via the property (forces a C++->Python sync if C++ is newer)."""
    return {b: (st.d, st.h_words.tolist()) for b, st in sorted(sim.active_shots.items())}


def _raw_state(sim: CosetsideSimulator) -> dict:
    """Python-side shot contents WITHOUT touching the property (flags stay as they are)."""
    return {b: (st.d, st.h_words.tolist()) for b, st in sorted(sim._active_shots.items())}


def test_repeated_peeks_after_python_write_import_once_not_per_peek(batch: int) -> None:
    sim = _setup_sim(batch)
    lib = _install_counting_lib(sim)
    imports = lib.counts
    max_active = 0
    for _ in range(2):  # each postselect dirties the Python side once -> exactly one re-import
        _postselect(sim)
        assert sim._py_dirty
        max_active = max(max_active, len(sim._active_shots))
        sim.peek_pauli_batch([0], "Z")
        first = imports["engine_import_shot"]
        for _ in range(10):
            sim.peek_pauli_batch([1], "Z")
        assert imports["engine_import_shot"] == first
        assert not sim._py_dirty
    assert max_active > 0
    assert imports["engine_clear"] <= 2
    assert imports["engine_import_shot"] <= 2 * max_active  # O(#syncs x #active)


def test_cpp_to_python_sync_without_python_write_does_not_rearm_import(batch: int) -> None:
    sim = _setup_sim(batch)
    _peek_all(sim)  # engine in sync, both flags clean
    lib = _install_counting_lib(sim)
    _unmatched_reset(sim, "Z")  # C++ kernel: only the engine changes
    assert sim._cpp_dirty
    r1 = np.zeros((sim.num_qubits, batch), dtype=bool)
    r1[1] = True
    # An empty `order` syncs C++->Python (so ``_cpp_dirty`` clears) but writes no shot state.
    sim.do_unmatched_noisy_reset_z(np.zeros_like(r1), r1, "X", order=[])
    assert not sim._cpp_dirty
    assert not sim._py_dirty
    peeks = _peek_all(sim)
    assert lib.counts["engine_import_shot"] == 0  # nothing newer in Python: no re-import
    assert lib.counts["engine_clear"] == 0
    _ = sim.active_shots  # python truth: re-import, same values
    np.testing.assert_array_equal(peeks, _peek_all(sim))


def test_run_records_match_pinned_digest(batch: int) -> None:
    # Pinned on the pre-fix code: skipping redundant re-imports must not change any record.
    # (Not compared with the Python path: its random draws come from different RNG streams.)
    sim = _make_sim(batch)
    sim.run()
    records = sim.get_final_measurement_records()
    assert records.shape == (batch, 4)
    digest = hashlib.sha256(np.packbits(records, axis=None).tobytes()).hexdigest()[:16]
    assert digest == _RUN_DIGESTS[batch]


def _write_via_property(sim: CosetsideSimulator) -> None:
    st = sim.active_shots[min(sim.active_shots)]
    st.h_words[0] ^= np.uint64(1)


def _unmatched_reset(sim: CosetsideSimulator, pauli: _Pauli) -> None:
    r0 = np.zeros((sim.num_qubits, sim.batch_size), dtype=bool)
    r1 = np.zeros_like(r0)
    r1[1, ::2] = True
    r0[1, 1::2] = True
    sim.do_unmatched_noisy_reset_z(r0, r1, pauli=pauli)


@pytest.mark.parametrize(
    "writer",
    [
        pytest.param(_write_via_property, id="active_shots_property_write"),
        pytest.param(lambda s: _unmatched_reset(s, "Z"), id="unmatched_reset_Z"),
        pytest.param(lambda s: _unmatched_reset(s, "X"), id="unmatched_reset_X"),
        pytest.param(lambda s: _unmatched_reset(s, "Y"), id="unmatched_reset_Y"),
        pytest.param(_postselect, id="postselect_Z"),
        pytest.param(lambda s: _postselect(s, "X"), id="postselect_X"),
    ],
)
def test_state_writers_agree_between_cpp_and_python_paths(
    batch: int, writer: Callable[[CosetsideSimulator], None]
) -> None:
    cpp, py = _setup_sim(batch, True), _setup_sim(batch, False)
    _peek_all(cpp)  # sync once so the C++ engine is clean before the write
    for sim in (cpp, py):
        writer(sim)
    np.testing.assert_array_equal(_peek_all(cpp), _peek_all(py))  # Python -> C++ visible
    assert _shot_state(cpp) == _shot_state(py)  # C++ -> Python visible (property sync)


def test_deterministic_batched_measure_with_active_shots_agrees_with_python_path(
    batch: int,
) -> None:
    cpp = _setup_sim(batch, True, _DET_MEASURE_CIRCUIT)
    py = _setup_sim(batch, False, _DET_MEASURE_CIRCUIT)
    _peek_all(cpp)
    for sim in (cpp, py):
        for inst in list(_DET_MEASURE_CIRCUIT)[_N_SETUP:]:
            sim.interactive_do(inst)
    assert _shot_state(cpp) == _shot_state(py)
    np.testing.assert_array_equal(_peek_all(cpp), _peek_all(py))


def test_healed_shot_with_nonzero_h_injected_through_property_is_imported(batch: int) -> None:
    # A d == 0 shot is still state (its residual Pauli h): the import must not skip it. Such
    # shots never arise naturally in the circuits above, so inject them (indices past the
    # first 64-bit word too, at B=256) and let a deterministic batched measure consume them.
    states = []
    for use_cpp in (True, False):
        sim = _setup_sim(batch, use_cpp, _DET_MEASURE_CIRCUIT)
        free = [b for b in range(batch) if b not in sim.active_shots][-3:]
        assert len(free) == 3
        for b in free:
            healed = MRQFState(sim.num_qubits, sim._n_words)
            healed.h_words[0] ^= np.uint64(0b1011)
            assert healed.d == 0
            sim.active_shots[b] = healed
        _peek_all(sim)
        for inst in list(_DET_MEASURE_CIRCUIT)[_N_SETUP:]:
            sim.interactive_do(inst)
        states.append((_shot_state(sim), _peek_all(sim)))
    assert states[0][0] == states[1][0]
    np.testing.assert_array_equal(states[0][1], states[1][1])


def test_clear_drops_stale_engine_shots_for_reuse(batch: int) -> None:
    sim = _setup_sim(batch)
    _postselect(sim)
    _peek_all(sim)  # the engine now holds the active shots
    sim.clear()
    np.testing.assert_array_equal(_peek_all(sim), _peek_all(_make_sim(batch)))


def test_emptying_the_active_shots_dict_after_a_peek_agrees_with_python_path(batch: int) -> None:
    cpp, py = _setup_sim(batch, True), _setup_sim(batch, False)
    with_shots = _peek_all(cpp)  # the engine now holds the shots
    for sim in (cpp, py):
        sim.active_shots.clear()  # an empty dict is state too: the engine must lose its shots
    assert not cpp._active_shots
    after = _peek_all(cpp)
    assert not np.array_equal(with_shots, after)  # the shots did matter
    np.testing.assert_array_equal(after, _peek_all(py))


def test_shots_inserted_through_a_property_read_made_while_dict_is_empty_are_imported(
    batch: int,
) -> None:
    # The property raises ``_py_dirty`` on every read, also when the dict is EMPTY at that
    # time: shots inserted later through that reference must still reach the engine.
    states = []
    for use_cpp in (True, False):
        sim = _setup_sim(batch, use_cpp, _DET_MEASURE_CIRCUIT)
        _peek_all(sim)
        sim.active_shots.clear()
        _peek_all(sim)  # sync: the engine is emptied, both flags clean
        held = sim.active_shots  # read while empty
        for b in (3, batch - 2):
            healed = MRQFState(sim.num_qubits, sim._n_words)
            healed.h_words[0] ^= np.uint64(0b1011)
            held[b] = healed
        for inst in list(_DET_MEASURE_CIRCUIT)[_N_SETUP:]:
            sim.interactive_do(inst)
        states.append((_shot_state(sim), _peek_all(sim)))
    assert states[0][0] == states[1][0]
    np.testing.assert_array_equal(states[0][1], states[1][1])


def test_conditional_clifford_group_fn_peeking_mid_loop_agrees_with_python_path(
    batch: int,
) -> None:
    circuit = _CX_CIRCUIT
    masks = np.random.default_rng(5).random((3, batch)) < 0.5
    peeks = []
    for use_cpp in (True, False):
        sim = _setup_sim(batch, use_cpp, circuit)
        seen: list[np.ndarray] = []
        sim.interactive_do(list(circuit)[4])  # TICK
        sim.peek_pauli_batch([0], "Z")  # engine clean, so the group peeks below must re-import

        def group_fn(g: int, sim: CosetsideSimulator = sim, seen: list = seen) -> np.ndarray:
            seen.append(sim.peek_pauli_batch([0, 1, 2, 3], "X").copy())
            return masks[g]

        sim.do_conditional_clifford(list(circuit)[5], condition_group_fn=group_fn)
        peeks.append(np.stack(seen))
    np.testing.assert_array_equal(peeks[0], peeks[1])


def test_held_active_shots_reference_is_stale_until_property_is_read_again(batch: int) -> None:
    # Documented semantics (see ``CosetsideSimulator.active_shots``): each property read raises
    # ``_py_dirty`` once and each sync clears it, so writes through a reference held across a
    # peek are not pushed to the engine until the property is read again.
    sim = _setup_sim(batch)
    held = sim.active_shots  # property read raises _py_dirty
    base = _peek_all(sim)  # sync: clears _py_dirty
    for st in held.values():
        st.h_words[0] ^= np.uint64(1)  # mutate through the held reference only
    np.testing.assert_array_equal(_peek_all(sim), base)  # documented: not re-pushed
    _ = sim.active_shots  # reading the property again re-arms the sync
    assert not np.array_equal(_peek_all(sim), base)
    oracle = _setup_sim(batch, False)
    for st in oracle.active_shots.values():
        st.h_words[0] ^= np.uint64(1)
    np.testing.assert_array_equal(_peek_all(sim), _peek_all(oracle))


def test_import_interrupted_midway_is_redone_by_the_next_peek(batch: int) -> None:
    cpp, py = _setup_sim(batch, True), _setup_sim(batch, False)
    _peek_all(cpp)
    for sim in (cpp, py):
        _postselect(sim)
    assert len(cpp._active_shots) >= 3
    lib = _install_counting_lib(cpp, fail_import_call=3)
    with pytest.raises(KeyboardInterrupt):
        _peek_all(cpp)  # engine now holds a partial import
    assert lib.counts["engine_import_shot"] == 3
    assert cpp._py_dirty  # still owed: only a completed import may clear it
    np.testing.assert_array_equal(_peek_all(cpp), _peek_all(py))  # retry re-imports everything


class _InterruptingBits:
    """``desired_bits`` stand-in whose 6th read raises KeyboardInterrupt."""

    def __init__(self) -> None:
        self.reads = 0

    def __getitem__(self, _: int) -> int:
        self.reads += 1
        if self.reads > 5:
            raise KeyboardInterrupt
        return 0


def _post_short_desired_bits(sim: CosetsideSimulator, _: pytest.MonkeyPatch) -> None:
    ones = np.ones(sim.batch_size, dtype=bool)
    sim.postselect_z_batch(0, ones, np.zeros(10, dtype=np.uint8), "Z")  # IndexError at shot 10


def _post_keyboard_interrupt(sim: CosetsideSimulator, _: pytest.MonkeyPatch) -> None:
    bits = cast("np.ndarray", _InterruptingBits())
    sim.postselect_z_batch(0, np.ones(sim.batch_size, dtype=bool), bits, "Z")


def _reset_bad_order(pauli: _Pauli) -> Callable[[CosetsideSimulator, pytest.MonkeyPatch], None]:
    def scenario(sim: CosetsideSimulator, _: pytest.MonkeyPatch) -> None:
        r0 = np.zeros((sim.num_qubits, sim.batch_size), dtype=bool)
        r1 = np.zeros_like(r0)
        r1[1:3] = True
        sim.do_unmatched_noisy_reset_z(r0, r1, pauli, order=[1, 2, 99])  # IndexError at qubit 99

    return scenario


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(_post_short_desired_bits, id="postselect_short_desired_bits"),
        pytest.param(_post_keyboard_interrupt, id="postselect_keyboard_interrupt"),
        pytest.param(_reset_bad_order("X"), id="unmatched_reset_X_bad_order"),
        pytest.param(_reset_bad_order("Y"), id="unmatched_reset_Y_bad_order"),
    ],
)
def test_python_writer_raising_midway_leaves_engine_consistent(
    batch: int,
    scenario: Callable[[CosetsideSimulator, pytest.MonkeyPatch], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The writers mutate shots one by one, so an exception leaves some shots changed. The next
    # peek must still see them: the engine view (a peek taken BEFORE reading the property)
    # must equal the Python view (a peek after a fresh property read, which re-imports).
    sim = _setup_sim(batch)
    _peek_all(sim)  # synced: both flags clean
    before = _raw_state(sim)
    with pytest.raises((IndexError, KeyboardInterrupt)):
        scenario(sim, monkeypatch)
    assert _raw_state(sim) != before  # the writer really changed Python shots before raising
    engine_view = _peek_all(sim)
    _ = sim.active_shots
    np.testing.assert_array_equal(engine_view, _peek_all(sim))


def test_conditional_clifford_raising_midway_keeps_python_state_flagged_dirty(
    batch: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same exception-path rule for the non-disjoint conditional Clifford. Its half-written
    # shots (healed d == 0 shots, evicted only after the group) do not show up in a peek, so
    # the invariant is checked on the flag itself: Python state changed => ``_py_dirty``.
    sim = _setup_sim(batch, True, _CX_CIRCUIT)
    _peek_all(sim)
    sim.interactive_do(list(_CX_CIRCUIT)[_N_SETUP])  # TICK
    assert not sim._py_dirty
    before = _raw_state(sim)
    real = sim._inject_inverse_cocycle_on_shot
    calls: list[int] = []

    def flaky(*args, **kwargs) -> None:
        calls.append(1)
        if len(calls) > 3:
            raise RuntimeError("boom")
        real(*args, **kwargs)

    monkeypatch.setattr(sim, "_inject_inverse_cocycle_on_shot", flaky)
    masks = np.random.default_rng(7).random((3, batch)) < 0.3
    with pytest.raises(RuntimeError):
        sim.do_conditional_clifford(list(_CX_CIRCUIT)[_N_SETUP + 1], masks)
    assert _raw_state(sim) != before  # shots were rewritten before the raise
    assert sim._py_dirty


def _raise_after_first_call(
    sim: CosetsideSimulator, attr: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Makes ``sim.<attr>`` raise KeyboardInterrupt right after its FIRST call returned."""
    real = getattr(sim, attr)
    calls: list[int] = []

    def wrapper(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(sim, attr, wrapper)


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(lambda s: _postselect(s, "Z"), id="postselect_Z"),
        pytest.param(lambda s: _postselect(s, "X"), id="postselect_X"),
        pytest.param(lambda s: _postselect(s, "Y"), id="postselect_Y"),
        pytest.param(lambda s: _unmatched_reset(s, "X"), id="unmatched_reset_X"),
        pytest.param(lambda s: _unmatched_reset(s, "Y"), id="unmatched_reset_Y"),
    ],
)
def test_python_writer_interrupted_right_after_its_first_shot_write_keeps_engine_consistent(
    batch: int,
    scenario: Callable[[CosetsideSimulator], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The flag must be raised before the first per-shot write, not after the first loop iteration.
    sim = _setup_sim(batch)
    _peek_all(sim)  # synced: both flags clean
    before = _raw_state(sim)
    _raise_after_first_call(sim, "_project_and_retire", monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        scenario(sim)
    assert _raw_state(sim) != before  # the first shot was really written
    assert sim._py_dirty
    engine_view = _peek_all(sim)
    _ = sim.active_shots
    np.testing.assert_array_equal(engine_view, _peek_all(sim))


def test_conditional_clifford_with_a_single_failed_shot_raising_flags_python_state_dirty(
    batch: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One failed shot in the group: the flag must not depend on the group having several.
    sim = _setup_sim(batch, True, _CX_CIRCUIT)
    _peek_all(sim)
    sim.interactive_do(list(_CX_CIRCUIT)[_N_SETUP])  # TICK
    assert not sim._py_dirty
    _raise_after_first_call(sim, "_inject_inverse_cocycle_on_shot", monkeypatch)
    masks = np.ones((3, batch), dtype=bool)
    masks[0, 5] = False
    with pytest.raises(KeyboardInterrupt):
        sim.do_conditional_clifford(list(_CX_CIRCUIT)[_N_SETUP + 1], masks)
    assert sim._py_dirty


class _RaiseAfterKernelLib:
    """Delegates to the real library; the ``nth`` call of ``kernel`` raises once it has RETURNED.

    Models an asynchronous KeyboardInterrupt landing between a C++ kernel's return and the
    Python statement that records that the engine changed.
    """

    def __init__(
        self,
        lib: ctypes.CDLL,
        kernel: str,
        nth: int = 1,
        on_return: Callable[[], None] | None = None,
    ) -> None:
        self._lib, self._kernel, self._nth, self.calls = lib, kernel, nth, 0
        self._on_return = on_return

    @property
    def fired(self) -> bool:
        return self.calls >= self._nth

    def __getattr__(self, name: str):
        fn = getattr(self._lib, name)
        if name != self._kernel:
            return fn

        def call_then_raise(*args):
            result = fn(*args)
            self.calls += 1
            if self.calls == self._nth:
                if self._on_return is None:
                    raise KeyboardInterrupt
                self._on_return()
            return result

        return call_then_raise


def _cc_disjoint(sim: CosetsideSimulator) -> None:
    masks = np.random.default_rng(4).random((2, sim.batch_size)) < 0.5
    sim.do_conditional_clifford(list(_DISJOINT_CX_CIRCUIT)[_N_SETUP + 1], masks)


def _measure(circuit: stim.Circuit) -> Callable[[CosetsideSimulator], None]:
    def run(sim: CosetsideSimulator) -> None:
        for inst in list(circuit)[_N_SETUP:]:
            sim.interactive_do(inst)

    return run


_KERNEL_CASES = [
    pytest.param(
        "engine_batch_inject_general_clifford",
        _DISJOINT_CX_CIRCUIT,
        _cc_disjoint,
        1,
        id="conditional_clifford_disjoint",
    ),
    pytest.param(
        "engine_batch_unmatched_noisy_reset_z",
        _CIRCUIT,
        lambda s: _unmatched_reset(s, "Z"),
        1,
        id="unmatched_reset_Z",
    ),
    pytest.param(
        "engine_batch_mixed_substep", _CIRCUIT, _measure(_CIRCUIT), 1, id="mixed_substep_1st"
    ),
    pytest.param(  # the 1st substep's kernel already left the engine ahead of Python
        "engine_batch_mixed_substep", _CIRCUIT, _measure(_CIRCUIT), 2, id="mixed_substep_2nd"
    ),
    pytest.param(
        "engine_batch_det_meas_step",
        _DET_MEASURE_CIRCUIT,
        _measure(_DET_MEASURE_CIRCUIT),
        1,
        id="measure_deterministic_batch",
    ),
]


@pytest.mark.parametrize(("kernel", "circuit", "op", "nth"), _KERNEL_CASES)
def test_interrupt_after_a_mutating_kernel_returns_leaves_a_flag_raised(
    batch: int, kernel: str, circuit: stim.Circuit, op: Callable[[CosetsideSimulator], None],
    nth: int,
) -> None:
    # `_sync_py_to_cpp` trusts the flags, so an interrupt after the kernel changed the engine
    # must never leave both clean: the (pre-kernel) Python state must still be re-imported, or
    # the engine flagged as ahead. Either way engine view == Python view afterwards. When the
    # engine was already ahead (``nth`` > 1) it must stay flagged ahead and never be overwritten
    # by the stale Python state (``_py_dirty`` must not be armed).
    sim = _setup_sim(batch, True, circuit)
    if circuit is _DISJOINT_CX_CIRCUIT:
        sim.interactive_do(list(circuit)[_N_SETUP])  # TICK
    _peek_all(sim)  # synced: both flags clean
    assert sim._cpp_lib is not None
    lib = _RaiseAfterKernelLib(sim._cpp_lib, kernel, nth)
    sim._cpp_lib = cast(ctypes.CDLL, lib)
    with pytest.raises(KeyboardInterrupt):
        op(sim)
    assert lib.fired
    assert sim._py_dirty or sim._cpp_dirty
    if nth > 1:
        assert sim._cpp_dirty and not sim._py_dirty
    engine_view = _peek_all(sim)
    _ = sim.active_shots
    np.testing.assert_array_equal(engine_view, _peek_all(sim))


_G1 = ["H", "S", "SQRT_X", "C_XYZ", "S_DAG", "SQRT_Y", "X", "Z"]
_G2 = ["CX", "CZ", "ISWAP", "SQRT_XX", "CY", "XCZ", "SWAP"]
_NQ = 6


def _random_ops(seed: int, fb: int) -> list[tuple]:
    r = np.random.default_rng(seed)
    ops: list[tuple] = [("bare", "H", [0, 2, 4]), ("bare", "CX", [0, 1, 2, 3, 4, 5])]
    kinds = ["cond", "cond_fn", "bare", "meas", "nreset", "post", "peek", "prop", "frame"]
    probs = [0.22, 0.1, 0.08, 0.12, 0.14, 0.1, 0.12, 0.06, 0.06]
    for _ in range(int(r.integers(8, 20))):
        k = str(r.choice(kinds, p=probs))
        perm = [int(x) for x in r.permutation(_NQ)]
        if k in ("cond", "cond_fn", "bare"):
            if r.random() < 0.5:
                g, tg = str(r.choice(_G1)), perm[: int(r.integers(1, _NQ + 1))]
            else:
                g, m = str(r.choice(_G2)), int(r.integers(1, _NQ // 2 + 1))
                tg = perm[: 2 * m]
                if r.random() < 0.35:  # overlapping groups -> non-disjoint (Python) path
                    tg = [int(x) for x in r.integers(0, _NQ, size=2 * m)]
                    for i in range(0, len(tg), 2):
                        tg[i + 1] = (tg[i] + 1) % _NQ if tg[i] == tg[i + 1] else tg[i + 1]
            ng = len(tg) // (2 if g in _G2 else 1)
            mask = r.random((ng, fb)) < r.choice([0.3, 0.6, 0.9])
            ops.append((k, g, tg, mask, bool(r.random() < 0.5)))
        elif k == "meas":
            ops.append((k, str(r.choice(["M", "MR", "R", "MX"])), perm[: int(r.integers(1, 7))]))
        elif k == "nreset":
            r0, r1 = np.zeros((_NQ, fb), bool), np.zeros((_NQ, fb), bool)
            sel, one = r.random(fb) < 0.6, r.random(fb) < 0.5
            r0[perm[0]], r1[perm[0]] = sel & ~one, sel & one
            ops.append((k, r0, r1, str(r.choice(["Z", "Z", "X", "Y"]))))
        elif k == "post":
            fm, des = r.random(fb) < 0.6, (r.random(fb) < 0.5).astype(np.uint8)
            ops.append((k, perm[0], fm, des, str(r.choice(["Z", "Z", "X", "Y"]))))
        elif k == "peek":
            ops.append((k, perm[: int(r.integers(1, 7))], str(r.choice(["X", "Y", "Z"]))))
        elif k == "frame":
            ops.append((k, str(r.choice(["X", "Z"])), r.random((_NQ, fb)) < 0.2))
        else:
            ops.append((k, int(r.integers(1, 8))))
    return ops


def _apply_random_op(sim: CosetsideSimulator, o: tuple, out: list) -> None:
    tick = stim.CircuitInstruction("TICK", [])
    k = o[0]
    if k in ("bare", "meas"):
        sim._do_bare_instruction(stim.CircuitInstruction(o[1], o[2]))
        sim._do_bare_instruction(tick)
    elif k == "cond":
        sim.do_conditional_clifford(stim.CircuitInstruction(o[1], o[2]), o[3].copy())
        sim._do_bare_instruction(tick)
    elif k == "cond_fn":

        def group_fn(g, mask=o[3], peeky=o[4]):
            row = mask[g].copy()
            if peeky:  # mid-loop peek (re-entrancy) feeding back into the condition
                row ^= (sim.peek_pauli_batch([0], "Z")[0] == 0) & (g % 2 == 0)
            return row

        op = stim.CircuitInstruction(o[1], o[2])
        sim.do_conditional_clifford(op, condition_group_fn=group_fn)
        sim._do_bare_instruction(tick)
    elif k == "nreset":
        sim.do_unmatched_noisy_reset_z(o[1].copy(), o[2].copy(), o[3])
    elif k == "post":
        pk = sim.peek_pauli_batch([o[1]], o[4])[0]
        sim.postselect_z_batch(o[1], o[2] & (pk == 0), o[3], o[4])
    elif k == "peek":
        out.append(sim.peek_pauli_batch(o[1], o[2]))
    elif k == "prop":
        out.append(np.array([len(sim.active_shots)]))
        for st in list(sim.active_shots.values())[:1]:
            st.h_words[0] ^= np.uint64(o[1])
    elif k == "frame":
        sim.apply_pauli(o[1], o[2].copy())


def _run_random_ops(seed: int, fb: int) -> list[np.ndarray]:
    ops = _random_ops(seed, fb)
    circuit = stim.Circuit()
    for o in ops:
        if o[0] in ("bare", "cond", "cond_fn", "meas"):
            circuit.append(o[1], o[2])
            circuit.append("TICK")
    circuit.append("M", list(range(_NQ)))
    handler = LeakageUint8Coset().compile_op_handler(circuit=circuit, batch_size=fb)
    sim = CosetsideSimulator(circuit, compiled_op_handler=handler, batch_size=fb, seed=seed)
    out: list[np.ndarray] = []
    for rep in range(2):  # second repetition reuses the simulator after clear()
        for o in ops:
            _apply_random_op(sim, o, out)
        out += [sim.peek_pauli_batch(list(range(_NQ)), p) for p in _PAULIS]
        if rep == 0:
            sim.clear()
    return out


def test_random_op_sequences_match_pre_fix_digest(batch: int) -> None:
    # Digest of every peek/property result of 60 seeded random op sequences (cpp engine), taken
    # on the pre-fix always-re-import code: the fix must not change any observable value, and
    # each flag-setting / sync-calling site above is exercised by some sequence.
    digest = hashlib.sha256()
    for seed in range(60):
        for out in _run_random_ops(seed, _FUZZ_BATCH[batch]):
            digest.update(np.ascontiguousarray(out).tobytes())
    assert digest.hexdigest()[:16] == _FUZZ_DIGESTS[batch]


def _first_kernel_case_params() -> list:
    return [c for c in _KERNEL_CASES if c.values[3] == 1]


@pytest.mark.parametrize(("kernel", "circuit", "op", "nth"), _first_kernel_case_params())
def test_interrupt_between_the_two_flag_stores_after_a_kernel_keeps_engine_consistent(
    batch: int, kernel: str, circuit: stim.Circuit, op: Callable[[CosetsideSimulator], None],
    nth: int,
) -> None:
    # Even an interrupt right after the FIRST flag store that follows the kernel's return (in
    # ``_engine_kernel_end``) must not leave both flags clean, i.e. `_cpp_dirty` must be raised
    # before `_py_dirty` is cleared.
    assert nth == 1
    sim = _setup_sim(batch, True, circuit)
    if circuit is _DISJOINT_CX_CIRCUIT:
        sim.interactive_do(list(circuit)[_N_SETUP])  # TICK
    _peek_all(sim)
    base_cls = type(sim)
    stores: list[str] = []

    class Trapped(base_cls):  # type: ignore[valid-type, misc]
        armed = False

        def __setattr__(self, name: str, value: object) -> None:
            super().__setattr__(name, value)
            if Trapped.armed and name in ("_py_dirty", "_cpp_dirty"):
                Trapped.armed = False
                stores.append(name)
                raise KeyboardInterrupt

    def arm() -> None:
        Trapped.armed = True

    sim.__class__ = Trapped
    assert sim._cpp_lib is not None
    sim._cpp_lib = cast(ctypes.CDLL, _RaiseAfterKernelLib(sim._cpp_lib, kernel, 1, arm))
    with pytest.raises(KeyboardInterrupt):
        op(sim)
    assert stores  # the interrupt really landed on a flag store after the kernel returned
    assert sim._py_dirty or sim._cpp_dirty
    engine_view = _peek_all(sim)
    _ = sim.active_shots
    np.testing.assert_array_equal(engine_view, _peek_all(sim))


class _RaiseAtEngineClear:
    """Delegates to the real library; the first ``engine_clear`` raises instead of clearing."""

    def __init__(self, lib: ctypes.CDLL) -> None:
        self._lib, self.armed = lib, True

    def __getattr__(self, name: str):
        fn = getattr(self._lib, name)
        if name != "engine_clear":
            return fn

        def clear_or_raise(*args):
            if self.armed:
                self.armed = False
                raise KeyboardInterrupt
            return fn(*args)

        return clear_or_raise


def _interrupted_clear_peeks(batch: int, use_cpp: bool, engine_ahead: bool) -> np.ndarray:
    sim = _setup_sim(batch, use_cpp)
    _peek_all(sim)  # both flags clean, the engine holds the active shots
    if engine_ahead:
        _unmatched_reset(sim, "Z")  # C++ kernel: the engine is now ahead of Python
    assert sim._cpp_lib is not None
    lib = _RaiseAtEngineClear(sim._cpp_lib)
    sim._cpp_lib = cast(ctypes.CDLL, lib)
    with pytest.raises(KeyboardInterrupt):
        sim.clear()
    assert not lib.armed and not sim._active_shots
    if use_cpp:
        assert sim._py_dirty
    _ = sim.active_shots  # property first: with the engine ahead it must not resurrect old shots
    return _peek_all(sim)


@pytest.mark.parametrize("engine_ahead", [False, True], ids=["engine_synced", "engine_ahead"])
def test_interrupted_clear_matches_the_python_path_at_the_next_peek(
    batch: int, engine_ahead: bool
) -> None:
    # `clear()` empties the Python side before `engine_clear`; the guard trusts the flags, so an
    # interrupt in between must leave `_py_dirty` raised (the engine still holds the old shots)
    # and the next peek must see the cleared state, as the Python path does.
    cpp = _interrupted_clear_peeks(batch, True, engine_ahead)
    py = _interrupted_clear_peeks(batch, False, engine_ahead)
    np.testing.assert_array_equal(cpp, py)


def _interrupt_at_cpp_dirty_store(
    batch: int, kernel: str, circuit: stim.Circuit, op: Callable[[CosetsideSimulator], None]
) -> tuple[CosetsideSimulator, dict]:
    """Interrupt right after the kernel's ``_cpp_dirty = True`` store: both flags end up set.

    Returns the simulator and its Python-side shot state as of just before the kernel ran.
    """
    sim = _setup_sim(batch, True, circuit)
    if circuit is _DISJOINT_CX_CIRCUIT:
        sim.interactive_do(list(circuit)[_N_SETUP])  # TICK
    _peek_all(sim)
    base_cls = type(sim)
    pre_kernel: dict = {}

    class Trapped(base_cls):  # type: ignore[valid-type, misc]
        armed = False

        def __setattr__(self, name: str, value: object) -> None:
            super().__setattr__(name, value)
            if Trapped.armed and name == "_cpp_dirty":
                Trapped.armed = False
                raise KeyboardInterrupt

    def arm() -> None:  # the kernel has just returned; Python still holds the pre-kernel state
        pre_kernel.update(_raw_state(sim))
        Trapped.armed = True

    sim.__class__ = Trapped
    assert sim._cpp_lib is not None
    sim._cpp_lib = cast(ctypes.CDLL, _RaiseAfterKernelLib(sim._cpp_lib, kernel, 1, arm))
    with pytest.raises(KeyboardInterrupt):
        op(sim)
    assert pre_kernel
    assert sim._py_dirty and sim._cpp_dirty
    return sim, pre_kernel


@pytest.mark.parametrize(("kernel", "circuit", "op", "nth"), _first_kernel_case_params())
def test_both_flags_raised_by_an_interrupt_resolve_to_the_python_state_in_any_read_order(
    batch: int, kernel: str, circuit: stim.Circuit, op: Callable[[CosetsideSimulator], None],
    nth: int,
) -> None:
    # An interrupt between the two stores of `_engine_kernel_end` leaves BOTH flags set. The
    # pre-kernel Python state must then win whether a peek or the `active_shots` property is
    # read first (the property used to adopt the post-kernel engine state instead).
    assert nth == 1
    sim, pre_kernel = _interrupt_at_cpp_dirty_store(batch, kernel, circuit, op)
    peeks_first = _peek_all(sim)
    state_after_peek = _shot_state(sim)
    sim, _ = _interrupt_at_cpp_dirty_store(batch, kernel, circuit, op)
    state_first = _shot_state(sim)
    assert not sim._cpp_dirty
    np.testing.assert_array_equal(peeks_first, _peek_all(sim))
    assert state_first == state_after_peek
    assert state_first == pre_kernel


def test_property_write_while_the_engine_is_ahead_is_reimported_like_the_python_path(
    batch: int,
) -> None:
    # The engine is ahead (a C++ kernel ran) when the property is read: the read syncs C++ ->
    # Python and must STILL arm `_py_dirty`, or the in-place write below is never re-imported.
    peeks = {}
    for use_cpp, mutate in ((True, True), (False, True), (False, False)):
        sim = _setup_sim(batch, use_cpp)
        _peek_all(sim)
        _unmatched_reset(sim, "Z")
        if mutate:
            shots = sim.active_shots  # ONE read: it alone must arm `_py_dirty`
            shots[min(shots)].h_words[0] ^= np.uint64(1)
        peeks[use_cpp, mutate] = _peek_all(sim)
    assert not np.array_equal(peeks[False, True], peeks[False, False])  # the write is observable
    np.testing.assert_array_equal(peeks[True, True], peeks[False, True])


def test_interrupt_after_the_first_nonrandom_mixed_kernel_leaves_a_flag_raised(
    batch: int,
) -> None:
    # `_eval_substep_cpp` has two kernel calls (is_random 0/1) sharing one begin/end pair. The
    # other mixed-substep tests interrupt only the random branch, so the first kernel here, a
    # deterministic SQRT_X-then-CX measurement setup, must be the non-random one.
    circuit = stim.Circuit(
        str(_CIRCUIT[:_N_SETUP]) + "\nTICK\nSQRT_X 1 3 2 0\nTICK\nCX 1 0\nTICK\nM 0 1 2 3"
    )
    sim = _setup_sim(batch, True, circuit)
    _peek_all(sim)
    assert sim._cpp_lib is not None
    real, modes = sim._cpp_lib, []

    class Spy:
        def __getattr__(self, name: str):
            fn = getattr(real, name)
            if name != "engine_batch_mixed_substep":
                return fn

            def record(*args):
                modes.append(args[1])  # the is_random argument
                return fn(*args)

            return record

    lib = _RaiseAfterKernelLib(cast(ctypes.CDLL, Spy()), "engine_batch_mixed_substep", 1)
    sim._cpp_lib = cast(ctypes.CDLL, lib)
    with pytest.raises(KeyboardInterrupt):
        for inst in list(circuit)[_N_SETUP:]:
            _ = sim.active_shots
            _peek_all(sim)
            sim.interactive_do(inst)
    assert modes == [0]  # the interrupted kernel was the non-random branch
    assert sim._py_dirty or sim._cpp_dirty
    engine_view = _peek_all(sim)
    _ = sim.active_shots
    np.testing.assert_array_equal(engine_view, _peek_all(sim))
