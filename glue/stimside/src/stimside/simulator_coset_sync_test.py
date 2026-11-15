"""Single-ownership C++<->Python export and D-11/D-32 invariants of ``CosetsideSimulator``.

In C++ mode, ``BatchedMRQFEngine`` is the sole owner of active shot states and
``_cpp_dirty`` is set immediately before every validated mutating C++ kernel call.
``active_shots`` returns a read-only ``MappingProxyType`` of fresh ``MRQFState.copy()``
objects materialized at read time via ``_sync_cpp_to_py()``.
"""

import ctypes
import hashlib
from collections.abc import Callable
from types import MappingProxyType
from typing import Literal, cast

import numpy as np
import pytest
import stim

from stimside._coset.mrqf_state import MRQFState
from stimside._native._kernel_loader import i32_p, u8_p, u64_p
from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
from stimside.simulator_coset import CosetsideSimulator

_Pauli = Literal["X", "Y", "Z"]
_PAULIS: tuple[_Pauli, ...] = ("X", "Y", "Z")
_FUZZ_BATCH = {64: 8, 256: 128}  # fuzz batch per test batch (128 shots span two 64-bit words)
_RUN_DIGESTS = {64: "d84158c5b3af7ad7", 256: "a5ca2f3263099f01"}  # pre-fix (22237da) code
_FUZZ_DIGESTS = {64: "873814f8a3eded7c", 256: "eb548a1e6947e8d4"}  # B3 read-only active_shots
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

    ``fail_export_call=n`` makes the n-th ``engine_export_shot`` raise KeyboardInterrupt (once).
    """

    def __init__(self, lib: ctypes.CDLL, fail_export_call: int | None = None) -> None:
        self._lib = lib
        self.fail_export_call = fail_export_call
        self.counts = {
            "engine_import_shot": 0,
            "engine_clear": 0,
            "engine_export_shot": 0,
        }

    def __getattr__(self, name: str):
        fn = getattr(self._lib, name)
        if name not in self.counts:
            return fn

        def counted(*args):
            self.counts[name] += 1
            if name == "engine_export_shot" and self.counts[name] == self.fail_export_call:
                raise KeyboardInterrupt
            return fn(*args)

        return counted


def _seed_shot(sim: CosetsideSimulator, b: int, st: MRQFState) -> None:
    if sim._use_cpp and sim._cpp_lib is not None and sim._cpp_eng is not None:
        m = max(1, st.d)
        p_arr = np.array(st.P if st.d > 0 else [-1], dtype=np.int32)
        sim._cpp_lib.engine_import_shot(
            sim._cpp_eng,
            b,
            st.d,
            np.ascontiguousarray(st.h_words, dtype=np.uint64).ctypes.data_as(u64_p),
            np.ascontiguousarray(st.a_union, dtype=np.uint64).ctypes.data_as(u64_p),
            np.ascontiguousarray(st._A_buf[:m], dtype=np.uint64).ctypes.data_as(u64_p),
            np.ascontiguousarray(st._ell_buf[:m], dtype=np.uint8).ctypes.data_as(u8_p),
            np.ascontiguousarray(st._Gamma_buf[:m, :m], dtype=np.uint8).ctypes.data_as(u8_p),
            p_arr.ctypes.data_as(i32_p),
        )
        sim._cpp_dirty = True
    else:
        sim._active_shots[b] = st


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


def test_cpp_ops_and_peeks_never_import_from_python(batch: int) -> None:
    sim = _setup_sim(batch)
    lib = _install_counting_lib(sim)
    for _ in range(2):
        _postselect(sim)
        assert sim._cpp_dirty
        sim.peek_pauli_batch([0], "Z")
        for _ in range(5):
            sim.peek_pauli_batch([1], "Z")
        assert sim._cpp_dirty
    assert lib.counts["engine_import_shot"] == 0
    assert lib.counts["engine_clear"] == 0
    assert len(sim.active_shots) > 0
    assert not sim._cpp_dirty


def test_empty_order_reset_does_not_clear_cpp_dirty_or_import(batch: int) -> None:
    sim = _setup_sim(batch)
    _peek_all(sim)
    lib = _install_counting_lib(sim)
    _unmatched_reset(sim, "Z")
    assert sim._cpp_dirty
    r1 = np.zeros((sim.num_qubits, batch), dtype=bool)
    r1[1] = True
    sim.do_unmatched_noisy_reset_z(np.zeros_like(r1), r1, "X", order=[])
    assert sim._cpp_dirty
    peeks = _peek_all(sim)
    assert lib.counts["engine_import_shot"] == 0
    assert lib.counts["engine_clear"] == 0
    assert len(sim.active_shots) > 0
    assert not sim._cpp_dirty
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


def _unmatched_reset(sim: CosetsideSimulator, pauli: _Pauli) -> None:
    r0 = np.zeros((sim.num_qubits, sim.batch_size), dtype=bool)
    r1 = np.zeros_like(r0)
    r1[1, ::2] = True
    r0[1, 1::2] = True
    sim.do_unmatched_noisy_reset_z(r0, r1, pauli=pauli)


@pytest.mark.parametrize(
    "writer",
    [
        pytest.param(lambda s: _unmatched_reset(s, "Z"), id="unmatched_reset_Z"),
        pytest.param(lambda s: _unmatched_reset(s, "X"), id="unmatched_reset_X"),
        pytest.param(lambda s: _unmatched_reset(s, "Y"), id="unmatched_reset_Y"),
        pytest.param(_postselect, id="postselect_Z"),
        pytest.param(lambda s: _postselect(s, "X"), id="postselect_X"),
        pytest.param(lambda s: _postselect(s, "Y"), id="postselect_Y"),
    ],
)
def test_state_writers_agree_between_cpp_and_python_paths(
    batch: int, writer: Callable[[CosetsideSimulator], None]
) -> None:
    cpp, py = _setup_sim(batch, True), _setup_sim(batch, False)
    _peek_all(cpp)
    for sim in (cpp, py):
        writer(sim)
    np.testing.assert_array_equal(_peek_all(cpp), _peek_all(py))
    assert _shot_state(cpp) == _shot_state(py)


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


def test_healed_shot_with_nonzero_h_seeded_into_engine_agrees_with_python(batch: int) -> None:
    states = []
    for use_cpp in (True, False):
        sim = _setup_sim(batch, use_cpp, _DET_MEASURE_CIRCUIT)
        free = [b for b in range(batch) if b not in sim.active_shots][-3:]
        assert len(free) == 3
        for b in free:
            healed = MRQFState(sim.num_qubits, sim._n_words)
            healed.h_words[0] ^= np.uint64(0b1011)
            assert healed.d == 0
            _seed_shot(sim, b, healed)
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


def test_active_shots_is_readonly_and_copy_isolated_in_cpp_and_python_modes(batch: int) -> None:
    cpp, py = _setup_sim(batch, True), _setup_sim(batch, False)
    cpp_peeks, py_peeks = _peek_all(cpp), _peek_all(py)
    for sim in (cpp, py):
        shots = sim.active_shots
        assert isinstance(shots, MappingProxyType)
        with pytest.raises(TypeError):
            cast(dict, shots)[0] = MRQFState(sim.num_qubits, sim._n_words)
        with pytest.raises(AttributeError):
            cast(dict, shots).clear()
        for st in shots.values():
            st.h_words[0] ^= np.uint64(0b1111)
    np.testing.assert_array_equal(_peek_all(cpp), cpp_peeks)
    np.testing.assert_array_equal(_peek_all(py), py_peeks)
    assert _shot_state(cpp) == _shot_state(py)


def test_d11_sync_cpp_to_py_populates_cache_and_clears_cpp_dirty(batch: int) -> None:
    sim = _setup_sim(batch)
    _postselect(sim)
    assert sim._cpp_dirty
    sim._sync_cpp_to_py()
    assert not sim._cpp_dirty
    assert len(sim._active_shots) > 0
    sim._cpp_dirty = True
    _ = sim.active_shots
    assert not sim._cpp_dirty


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


def test_active_shots_returns_fresh_copies_isolated_from_caller_writes(batch: int) -> None:
    sim = _setup_sim(batch)
    base = _peek_all(sim)
    first = sim.active_shots
    second = sim.active_shots
    for b, st in first.items():
        assert st is not second[b] and st is not sim._active_shots[b]
        st.h_words[0] ^= np.uint64(1)
    np.testing.assert_array_equal(_peek_all(sim), base)
    oracle = _setup_sim(batch, False)
    assert _shot_state(sim) == _shot_state(oracle)


def test_export_interrupted_midway_keeps_cpp_dirty_and_retries_cleanly(batch: int) -> None:
    cpp, py = _setup_sim(batch, True), _setup_sim(batch, False)
    _peek_all(cpp)
    for sim in (cpp, py):
        _postselect(sim)
    assert cpp._cpp_dirty
    real_lib = cpp._cpp_lib
    assert real_lib is not None
    lib = _install_counting_lib(cpp, fail_export_call=2)
    with pytest.raises(KeyboardInterrupt):
        _ = cpp.active_shots
    assert lib.counts["engine_export_shot"] == 2
    assert cpp._cpp_dirty
    cpp._cpp_lib = real_lib
    assert _shot_state(cpp) == _shot_state(py)
    assert not cpp._cpp_dirty
    np.testing.assert_array_equal(_peek_all(cpp), _peek_all(py))


class _InterruptingBits:
    """``desired_bits`` stand-in whose 6th element read raises KeyboardInterrupt."""

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.reads = 0

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self, _: int) -> int:
        self.reads += 1
        if self.reads > 5:
            raise KeyboardInterrupt
        return 0


def _post_short_desired_bits(sim: CosetsideSimulator, _: pytest.MonkeyPatch) -> None:
    ones = np.ones(sim.batch_size, dtype=bool)
    sim.postselect_z_batch(0, ones, np.zeros(10, dtype=np.uint8), "Z")


def _post_keyboard_interrupt(sim: CosetsideSimulator, _: pytest.MonkeyPatch) -> None:
    bits = cast("np.ndarray", _InterruptingBits(sim.batch_size))
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
    # Upfront operand normalization rejects invalid/interrupting inputs before _cpp_dirty or shots.
    for start_dirty in (False, True):
        sim = _setup_sim(batch)
        if start_dirty:
            _postselect(sim)
        engine_before = _peek_all(sim)
        before = _raw_state(sim)
        assert sim._cpp_dirty is start_dirty
        with pytest.raises((IndexError, ValueError, KeyboardInterrupt)):
            scenario(sim, monkeypatch)
        assert sim._cpp_dirty is start_dirty
        assert _raw_state(sim) == before
        np.testing.assert_array_equal(_peek_all(sim), engine_before)


def test_conditional_clifford_python_fallback_and_cpp_dirty_flag(
    batch: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # _use_cpp=False right after construction runs Python _dcc_sequential_py without touching C++.
    py_sim = _setup_sim(batch, False, _CX_CIRCUIT)
    py_sim.interactive_do(list(_CX_CIRCUIT)[_N_SETUP])
    before = _raw_state(py_sim)
    real = py_sim._inject_inverse_cocycle_on_shot
    calls: list[int] = []

    def flaky(*args, **kwargs) -> None:
        calls.append(1)
        if len(calls) > 3:
            raise RuntimeError("boom")
        real(*args, **kwargs)

    monkeypatch.setattr(py_sim, "_inject_inverse_cocycle_on_shot", flaky)
    masks = np.random.default_rng(7).random((3, batch)) < 0.3
    with pytest.raises(RuntimeError):
        py_sim.do_conditional_clifford(list(_CX_CIRCUIT)[_N_SETUP + 1], masks)
    assert _raw_state(py_sim) != before and not py_sim._cpp_dirty

    # In C++ mode, entry shape error leaves _cpp_dirty False; mid-loop callback error keeps True.
    cpp_sim = _setup_sim(batch, True, _CX_CIRCUIT)
    _peek_all(cpp_sim)
    cpp_sim.interactive_do(list(_CX_CIRCUIT)[_N_SETUP])
    assert not cpp_sim._cpp_dirty
    with pytest.raises(ValueError):
        cpp_sim.do_conditional_clifford(list(_CX_CIRCUIT)[_N_SETUP + 1], np.zeros((1, batch), bool))
    assert not cpp_sim._cpp_dirty

    def flaky_cb(g: int) -> np.ndarray:
        if g == 1:
            raise RuntimeError("cb boom")
        return masks[g]

    with pytest.raises(RuntimeError, match="cb boom"):
        cpp_sim.do_conditional_clifford(
            list(_CX_CIRCUIT)[_N_SETUP + 1], condition_group_fn=flaky_cb
        )
    assert cpp_sim._cpp_dirty


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
def test_python_fallback_writer_interrupted_right_after_first_write(
    batch: int,
    scenario: Callable[[CosetsideSimulator], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = _setup_sim(batch, use_cpp=False)
    before = _raw_state(sim)
    _raise_after_first_call(sim, "_project_and_retire", monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        scenario(sim)
    assert _raw_state(sim) != before
    assert not sim._cpp_dirty


def test_conditional_clifford_single_failed_shot_sets_cpp_dirty(batch: int) -> None:
    sim = _setup_sim(batch, True, _CX_CIRCUIT)
    _peek_all(sim)
    sim.interactive_do(list(_CX_CIRCUIT)[_N_SETUP])
    assert not sim._cpp_dirty
    masks = np.ones((3, batch), dtype=bool)
    masks[0, 5] = False
    sim.do_conditional_clifford(list(_CX_CIRCUIT)[_N_SETUP + 1], masks)
    assert sim._cpp_dirty


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


def _cc_sequential(sim: CosetsideSimulator) -> None:
    masks = np.random.default_rng(4).random((3, sim.batch_size)) < 0.5
    sim.do_conditional_clifford(list(_CX_CIRCUIT)[_N_SETUP + 1], masks)


def _measure(circuit: stim.Circuit) -> Callable[[CosetsideSimulator], None]:
    def run(sim: CosetsideSimulator) -> None:
        for inst in list(circuit)[_N_SETUP:]:
            sim.interactive_do(inst)

    return run


_KERNEL_CASES = [
    pytest.param(
        "engine_batch_inject_general_clifford_v3",
        _DISJOINT_CX_CIRCUIT,
        _cc_disjoint,
        1,
        id="conditional_clifford_disjoint",
    ),
    pytest.param(
        "engine_inject_group_v3",
        _CX_CIRCUIT,
        _cc_sequential,
        1,
        id="conditional_clifford_sequential",
    ),
    pytest.param(
        "engine_batch_unmatched_noisy_reset_z",
        _CIRCUIT,
        lambda s: _unmatched_reset(s, "Z"),
        1,
        id="unmatched_reset_Z",
    ),
    pytest.param(
        "engine_project_pauli",
        _CIRCUIT,
        lambda s: _unmatched_reset(s, "X"),
        1,
        id="unmatched_reset_X",
    ),
    pytest.param(
        "engine_postselect_batch",
        _CIRCUIT,
        lambda s: _postselect(s, "Z"),
        1,
        id="postselect_Z",
    ),
    pytest.param(
        "engine_batch_mixed_substep", _CIRCUIT, _measure(_CIRCUIT), 1, id="mixed_substep_1st"
    ),
    pytest.param(
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
    sim = _setup_sim(batch, True, circuit)
    if circuit in (_DISJOINT_CX_CIRCUIT, _CX_CIRCUIT):
        sim.interactive_do(list(circuit)[_N_SETUP])  # TICK
    _peek_all(sim)
    assert not sim._cpp_dirty
    assert sim._cpp_lib is not None
    lib = _RaiseAfterKernelLib(sim._cpp_lib, kernel, nth)
    sim._cpp_lib = cast(ctypes.CDLL, lib)
    with pytest.raises(KeyboardInterrupt):
        op(sim)
    assert lib.fired
    assert sim._cpp_dirty
    engine_view = _peek_all(sim)
    _ = sim.active_shots
    assert not sim._cpp_dirty
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
    digest = hashlib.sha256()
    for seed in range(60):
        for out in _run_random_ops(seed, _FUZZ_BATCH[batch]):
            digest.update(np.ascontiguousarray(out).tobytes())
    assert digest.hexdigest()[:16] == _FUZZ_DIGESTS[batch]


def test_random_op_sequences_cpp_agrees_bit_exact_with_python_path(batch: int) -> None:
    fb = _FUZZ_BATCH[batch]
    for seed in range(60):
        ops = [o for o in _random_ops(seed, fb) if o[0] != "meas"]
        circuit = stim.Circuit()
        for o in ops:
            if o[0] in ("bare", "cond", "cond_fn"):
                circuit.append(o[1], o[2])
                circuit.append("TICK")
        circuit.append("M", list(range(_NQ)))
        sims = []
        for use_cpp in (True, False):
            h = LeakageUint8Coset().compile_op_handler(circuit=circuit, batch_size=fb)
            s = CosetsideSimulator(circuit, compiled_op_handler=h, batch_size=fb, seed=seed)
            s._use_cpp = use_cpp
            sims.append(s)
        for rep in range(2):
            outs = [[], []]
            for s, o_list in zip(sims, outs):
                for o in ops:
                    _apply_random_op(s, o, o_list)
                o_list += [s.peek_pauli_batch(list(range(_NQ)), p) for p in _PAULIS]
            assert len(outs[0]) == len(outs[1])
            for a, b in zip(outs[0], outs[1]):
                np.testing.assert_array_equal(a, b)
            assert _shot_state(sims[0]) == _shot_state(sims[1])
            if rep == 0:
                sims[0].clear()
                sims[1].clear()


def _first_kernel_case_params() -> list:
    return [c for c in _KERNEL_CASES if c.values[3] == 1]


@pytest.mark.parametrize(("kernel", "circuit", "op", "nth"), _first_kernel_case_params())
def test_interrupt_before_mutating_kernel_at_cpp_dirty_store_keeps_engine_consistent(
    batch: int, kernel: str, circuit: stim.Circuit, op: Callable[[CosetsideSimulator], None],
    nth: int,
) -> None:
    assert nth == 1
    sim = _setup_sim(batch, True, circuit)
    if circuit in (_DISJOINT_CX_CIRCUIT, _CX_CIRCUIT):
        sim.interactive_do(list(circuit)[_N_SETUP])
    _peek_all(sim)
    before = _shot_state(sim)
    base_cls = type(sim)

    class Trapped(base_cls):  # type: ignore[valid-type, misc]
        armed = True

        def __setattr__(self, name: str, value: object) -> None:
            super().__setattr__(name, value)
            if Trapped.armed and name == "_cpp_dirty" and value is True:
                Trapped.armed = False
                raise KeyboardInterrupt

    sim.__class__ = Trapped
    with pytest.raises(KeyboardInterrupt):
        op(sim)
    assert sim._cpp_dirty
    assert _shot_state(sim) == before
    assert not sim._cpp_dirty


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


@pytest.mark.parametrize("engine_ahead", [False, True], ids=["engine_synced", "engine_ahead"])
def test_interrupted_clear_leaves_pre_clear_state_until_retried(
    batch: int, engine_ahead: bool
) -> None:
    sim = _setup_sim(batch, True)
    _peek_all(sim)
    if engine_ahead:
        _unmatched_reset(sim, "Z")
    peeks_before = _peek_all(sim)
    assert sim._cpp_dirty is engine_ahead
    assert sim._cpp_lib is not None
    lib = _RaiseAtEngineClear(sim._cpp_lib)
    sim._cpp_lib = cast(ctypes.CDLL, lib)
    with pytest.raises(KeyboardInterrupt):
        sim.clear()
    assert not lib.armed
    assert sim._cpp_dirty is engine_ahead
    np.testing.assert_array_equal(_peek_all(sim), peeks_before)
    sim.clear()
    assert not sim._cpp_dirty and len(sim.active_shots) == 0
    np.testing.assert_array_equal(_peek_all(sim), _peek_all(_make_sim(batch)))


def test_property_write_while_the_engine_is_ahead_only_mutates_caller_copy(
    batch: int,
) -> None:
    peeks = {}
    for use_cpp, mutate in ((True, True), (False, True), (False, False)):
        sim = _setup_sim(batch, use_cpp)
        _peek_all(sim)
        _unmatched_reset(sim, "Z")
        if mutate:
            shots = sim.active_shots
            shots[min(shots)].h_words[0] ^= np.uint64(1)
        peeks[use_cpp, mutate] = _peek_all(sim)
    np.testing.assert_array_equal(peeks[False, True], peeks[False, False])
    np.testing.assert_array_equal(peeks[True, True], peeks[False, True])


def test_interrupt_after_the_first_nonrandom_mixed_kernel_leaves_a_flag_raised(
    batch: int,
) -> None:
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
    assert sim._cpp_dirty
    engine_view = _peek_all(sim)
    _ = sim.active_shots
    assert not sim._cpp_dirty
    np.testing.assert_array_equal(engine_view, _peek_all(sim))
