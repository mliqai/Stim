from __future__ import annotations

import ctypes
import dataclasses
from typing import Any, Callable, Iterable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]

from stimside._coset.mrqf_state import MRQFState
from stimside._coset.reference_compile import (
    CompiledReferenceCircuit,
    MeasStepMeta,
    StepMeta,
    UnitaryStepMeta,
)
from stimside._native._kernel_loader import i8_p, i32_p, i64_p, u8_p, u64_p
from stimside._native.coset_kernels import get_coset_kernels_lib
from stimside.op_handlers.abstract_op_handler import CompiledOpHandler
from stimside.op_handlers.seam import SyncRngs
from stimside.util.bitwords import (
    first_set_bit_in_words as _first_set_bit_in_words,
    word_dot_parity as _word_dot_parity,
    words_any_nonzero as _words_any_nonzero,
    words_any_overlap as _words_any_overlap,
)
from stimside.util.circuit import (
    ShotMajorM2DMixin,
    apply_coords_op,
    check_replay,
    stack_measurement_columns,
    unroll_circuit,
)
from stimside.util.leakage_events import LeakageEvent, LeakageEventsRecorderMixin
from stimside.util.reference_chp import (
    PackedSymplecticTables,
    ReferenceCHPTableau,
    pauli_image,
    pauli_image_batch,
)
from stimside.util.seeding import derived_seed
from stimside.util.stim_workarounds import broadcast_pauli_errors
from stimside.util.sync_rng import (
    apply_sync_meas_flips,
    build_sync_rngs,
    conditioned_targets_per_shot,
    draw_tab_rng_bit,
    exec_sync_correlated_error,
    exec_sync_heralded_noise,
    exec_sync_pauli_batch,
)
from stimside.util.unleaked_to_leaked import UnleakedToLeakedRecordsMixin

# Pauli applied to the qubit of a record-controlled 2Q gate (stim only parses these with rec[] controls).
_REC_FIRST_PAULI = {"CX": "X", "CNOT": "X", "ZCX": "X", "CY": "Y", "ZCY": "Y"}
_REC_SECOND_PAULI = {"XCZ": "X", "YCZ": "Y"}


def _own_mask(mask: Any) -> NDArray[Any]:
    """The caller's `mask` as a private, C-ordered, plain `np.ndarray` copy (dtype unchanged).

    Nobody else holds a reference to the copy, so no signal handler, thread, or `__bool__` /
    `__setstate__` hook of the caller can change its shape, dtype or size afterwards. The one
    `np.asarray` runs the caller's `__array__` once and drops an ndarray subclass (its `shape`
    and `__getitem__` never run). For a C-contiguous uint8 mask `np.ascontiguousarray` returns
    its argument itself, so without this copy the kernel would read the caller's own buffer.
    """
    return np.array(np.asarray(mask), order="C", copy=True)


def _check_mask_shape(
    name: str, shape: tuple[int, ...], rows: int, cols: int, rows_are: str
) -> None:
    """Raise ValueError unless a mask of this `shape` is (rows, cols).

    The C++ kernels read rows * cols mask bytes from the buffer they are given, so a smaller
    array is an out-of-bounds read there. Callers pass the shape of the `_own_mask` copy and hand
    that same copy on: `np.ascontiguousarray(copy, dtype=np.uint8)` keeps its shape, so the shape
    alone fixes the number of bytes the kernel reads.
    """
    if shape != (rows, cols):
        raise ValueError(
            f"{name} has shape {shape}, expected ({rows}, {cols}): one row per {rows_are} "
            "and one column per shot (batch_size)"
        )


def _check_qubit_counts(
    circuit: stim.Circuit, compiled_ref: CompiledReferenceCircuit | None, handler: Any
) -> None:
    """Raise ValueError unless `compiled_ref` and `handler` were compiled for `circuit`'s qubits.

    The engine and its kernels are sized by the circuit: a reference (packed words, tables) or a
    handler (scratch masks) of another size is read past its end. A handler without an integer
    `num_qubits` attribute is not checked.
    """
    n = circuit.num_qubits
    if compiled_ref is not None and compiled_ref.num_qubits != n:
        raise ValueError(
            f"compiled_ref was compiled for {compiled_ref.num_qubits} qubits, but the circuit "
            f"has {n}"
        )
    handler_n = getattr(handler, "num_qubits", None)
    if isinstance(handler_n, (int, np.integer)) and handler_n != n:
        raise ValueError(
            f"compiled_op_handler was compiled for {handler_n} qubits, but the circuit has {n}"
        )


def _owned_block(
    x: Any, dtype: Any, block: tuple[int, ...], what: str, exact_cols: bool = False
) -> NDArray[Any]:
    """A private C-contiguous copy of the leading `block` of array `x`, which must be `dtype`.

    `x` is copied before it is checked, so the checks and the returned block see one buffer that
    no caller can resize or rewrite. `exact_cols` also requires the last axis to equal block's.
    """
    a = np.array(x, copy=True)
    bad = a.dtype != dtype or a.ndim != len(block) or any(s < k for s, k in zip(a.shape, block))
    if bad or (exact_cols and a.shape[-1] != block[-1]):
        raise ValueError(
            f"{what} must be a {np.dtype(dtype).name} array of at least {block}, got "
            f"{a.dtype} {a.shape}"
        )
    return np.ascontiguousarray(a[tuple(slice(0, k) for k in block)])


def _owned_p_array(raw_p: Any, d: int, n_q: int, what: str) -> NDArray[np.int32]:
    if d == 0:
        return np.array([-1], dtype=np.int32)
    msg = f"{what}.P must hold d={d} distinct integer qubit indices in [0, {n_q})"
    if type(raw_p) in (list, tuple):
        vals: list[Any] = list(raw_p)
    else:
        p_raw = np.array(raw_p, copy=True)
        if p_raw.ndim != 1 or p_raw.shape != (d,):
            raise ValueError(msg)
        if p_raw.dtype.kind in "iu":
            if bool(np.any((p_raw < 0) | (p_raw >= n_q))) or len(set(p_raw.tolist())) != d:
                raise ValueError(msg)
            return p_raw.astype(np.int32)
        if p_raw.dtype != object:
            raise ValueError(msg)
        vals = p_raw.tolist()
    if len(vals) != d or not all(
        isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in vals
    ):
        raise ValueError(msg)
    ints = [int(v) for v in vals]
    if any(v < 0 or v >= n_q for v in ints) or len(set(ints)) != d:
        raise ValueError(msg)
    return np.array(ints, dtype=np.int32)


def _owned_shot_import(
    b: Any, st: Any, n_q: int, n_w: int, batch: int
) -> tuple[int, int, tuple[NDArray[Any], ...]]:
    """Check one `active_shots` entry; return its index, rank and private copies of its buffers.

    `engine_import_shot` trusts the shot index, the rank `d` and every buffer length, so an
    out-of-range key or a short buffer in the public mutable dict is an out-of-bounds access in
    C++ (R3-F7). Raises ValueError for any entry that does not fit the engine.
    """
    if isinstance(b, (bool, np.bool_)) or not isinstance(b, (int, np.integer)):
        raise TypeError(f"active_shots key {b!r} must be an integer")
    bi = int(b)
    if not 0 <= bi < batch:
        raise ValueError(f"active_shots key {b!r} is out of range for batch_size={batch}")
    raw_d = st._d
    if isinstance(raw_d, (bool, np.bool_)) or not isinstance(raw_d, (int, np.integer)):
        raise TypeError(f"active_shots[{bi}] rank d={raw_d!r} must be an integer")
    d = int(raw_d)
    if not 0 <= d <= n_q:
        raise ValueError(f"active_shots[{bi}] has rank d={d}, expected 0 <= d <= {n_q}")
    m, what = max(1, d), f"active_shots[{bi}]"
    p_i32 = _owned_p_array(st.P if d else (), d, n_q, what)
    bufs = (
        _owned_block(st.h_words, np.uint64, (n_w,), f"{what}.h_words", exact_cols=True),
        _owned_block(st.a_union, np.uint64, (n_w,), f"{what}.a_union", exact_cols=True),
        _owned_block(st._A_buf, np.uint64, (m, n_w), f"{what}._A_buf", exact_cols=True),
        _owned_block(st._ell_buf, np.uint8, (m,), f"{what}._ell_buf"),
        _owned_block(st._Gamma_buf, np.uint8, (m, m), f"{what}._Gamma_buf"),
        p_i32,
    )
    return bi, d, bufs


def _copy_export_into(st: MRQFState, d: int, out: tuple[NDArray[Any], ...]) -> None:
    h, au, a_mat, ell, gamma, p = out
    st._ensure_capacity(max(1, d))
    st.h_words[...] = h
    st.a_union[...] = au
    st._A_buf[:d] = a_mat[:d]
    st._ell_buf[:d] = ell[:d]
    st._Gamma_buf[:d, :d] = gamma[:d, :d]
    st._d = d
    st.P = [int(x) for x in p[:d]]
    st._refresh_views()


def _reusable(st: Any, n_w: int) -> bool:
    """`st` is an MRQFState whose buffers have the layout `MRQFState` itself allocates."""
    if type(st) is not MRQFState:
        return False
    cap, d = st._cap, st._d
    if (
        type(cap) is not int
        or cap < 1
        or type(st.n_words) is not int
        or st.n_words != n_w
        or type(d) is not int
        or not 0 <= d <= cap
    ):
        return False
    bufs = (st.h_words, st.a_union, st._A_buf, st._ell_buf, st._Gamma_buf)
    want = (
        (bufs[0], np.uint64, (n_w,)), (bufs[1], np.uint64, (n_w,)),
        (bufs[2], np.uint64, (cap, n_w)), (bufs[3], np.uint8, (cap,)),
        (bufs[4], np.uint8, (cap, cap)),
    )
    if not all(
        type(a) is np.ndarray and a.dtype == dt and a.shape == shp and a.flags.writeable
        for a, dt, shp in want
    ):
        return False
    return len({id(b) for b in bufs}) == 5 and all(
        not np.shares_memory(bufs[i], bufs[j]) for i in range(5) for j in range(i + 1, 5)
    )


def _install_export(
    st: Any, d: int, n_q: int, n_w: int, out: tuple[NDArray[Any], ...]
) -> MRQFState:
    """Copy one exported shot `(h, a_union, A, ell, Gamma, P)` into `st`; return the state used.

    The export writes only into fresh buffers (R3-F7: a held `active_shots` reference can swap
    short buffers into `st`). A `st` whose buffers do not have MRQFState's own layout is replaced.
    """
    if not _reusable(st, n_w):
        st = MRQFState(n_q, n_w, _init_cap=max(16, d))
    _copy_export_into(st, d, out)
    return st


def _is_mixed_two_qubit(op: stim.CircuitInstruction) -> bool:
    """A two-qubit gate with a record or sweep target (`CX rec[-1] 1`)."""
    return stim.gate_data(op.name).is_two_qubit_gate and any(
        t.qubit_value is None for t in op.targets_copy()
    )


def _count_all_quantum_groups(op: stim.CircuitInstruction) -> int:
    """Groups of a two-qubit `op` with two qubit targets: those its reference step keeps."""
    groups = op.target_groups()
    return sum(g[0].qubit_value is not None and g[1].qubit_value is not None for g in groups)


def _check_mask_shape_1d(name: str, shape: tuple[int, ...], cols: int) -> None:
    """Raise ValueError unless a 1-D mask of this `shape` has length `cols`."""
    if shape != (cols,):
        raise ValueError(
            f"{name} has shape {shape}, expected ({cols},): one entry per shot (batch_size)"
        )


@dataclasses.dataclass(slots=True)
class _DccState:
    """Mutable step/commit state for one `do_conditional_clifford` invocation."""

    orig_op: stim.CircuitInstruction
    op_exec: stim.CircuitInstruction
    mask: NDArray[Any] | None
    eff: tuple[tuple[int, ...], str, str, bool] | None
    has_group_fn: bool
    quant_indices: list[int] | None
    q_mask: NDArray[Any] | None
    rec_flips: list[tuple[str, int, NDArray[np.bool_]]]
    step_meta: StepMeta
    packed_words_before: PackedSymplecticTables
    steps_len_before: int
    cursor_after_step: int
    pending: tuple[ReferenceCHPTableau, PackedSymplecticTables] | None
    clear_at_step: int


class CosetsideSimulator(
    ShotMajorM2DMixin, UnleakedToLeakedRecordsMixin, LeakageEventsRecorderMixin
):
    """High-performance batched Clifford-error & stabilizer simulator via Moving-Reference Quadratic Forms (MRQF).

    Combines:
    1. Phase 0 Reference CHP Tableau (backed by C++ stim.Tableau),
    2. Batched C++ SIMD baseline Pauli frame tracking via stim.FlipSimulator across all B shots,
    3. Compressed polynomial-space MRQF states D_b = (h, A, ell, Gamma, P) only for active shots with d_b >= 1,
       with O(0) cost on ideal unitary gates and automatic O(1) self-healing eviction back into stim.FlipSimulator
       whenever d_b drops to 0.

    Implements the batched OpHandler seam ``op_handlers.seam.BatchedLeakageSimulator`` (as does
    ``MagicsideSimulator``).
    """

    def __init__(
        self,
        circuit: stim.Circuit,
        *,
        compiled_op_handler: CompiledOpHandler["CosetsideSimulator"],
        seed: int | None = None,
        batch_size: int = 1,
        compiled_ref: CompiledReferenceCircuit | None = None,
        sync_tableside_rng: bool | None = None,
        record_unleaked_to_leaked: bool = False,
        record_leakage_events: bool = False,
    ) -> None:
        """record_leakage_events: record every leakage-state change of each shot
        (``get_leakage_events()``, see ``stimside.util.leakage_events``)."""
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, (int, np.integer))
            or not (0 < int(batch_size) <= 0x7FFFFFFF)
        ):
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        batch_size = int(batch_size)
        _check_qubit_counts(circuit, compiled_ref, compiled_op_handler)  # before engine_create
        self.circuit = circuit
        n_q = int(circuit.num_qubits)
        self.num_qubits = n_q
        self._init_num_qubits = n_q
        self.seed = seed
        self.batch_size = batch_size
        self._init_batch_size = batch_size
        self.record_unleaked_to_leaked = bool(record_unleaked_to_leaked)
        self._unleaked_to_leaked_events: list[list[int]] = (
            [[] for _ in range(batch_size)] if self.record_unleaked_to_leaked else []
        )
        self.record_leakage_events = bool(record_leakage_events)
        self._leakage_events: list[list[LeakageEvent]] = (
            [[] for _ in range(batch_size)] if self.record_leakage_events else []
        )
        self.sync_tableside_rng = bool(sync_tableside_rng)
        self._n_words = max(1, (n_q + 63) // 64)
        self._batches_completed = 0
        self._tab_scratch_q = n_q
        self._init_cpp_and_rng_state(n_q, batch_size)
        self._flip_simulator = stim.FlipSimulator(
            batch_size=batch_size,
            disable_stabilizer_randomization=True,
            num_qubits=n_q,
            seed=self.seed,
        )
        self._compiled_ref = (
            compiled_ref if compiled_ref is not None else CompiledReferenceCircuit(circuit)
        )
        self._compiled_m2d_converter = circuit.compile_m2d_converter()
        self._compiled_op_handler = compiled_op_handler
        self.compiled_op_handler = compiled_op_handler
        self._init_runtime_buffers(n_q, batch_size)

    def _init_cpp_and_rng_state(self, n_q: int, batch_size: int) -> None:
        self._shot_np_rngs: list[np.random.Generator] | None = None
        self._tab_rngs: list[stim.TableauSimulator] | None = None
        self._tab_rngs_x: list[stim.TableauSimulator] | None = None
        self._shot_tiebreak_rng: NDArray[np.uint64]
        self._cpp_lib: ctypes.CDLL | None = None
        self._cpp_eng: int | None = None
        self._cpp_dirty: bool = False
        self._py_dirty: bool = False
        self._any_x_c = ctypes.c_int(0)
        self._any_z_c = ctypes.c_int(0)
        self._init_rngs()
        self._use_cpp = self._tab_rngs is None
        if self._use_cpp:
            self._cpp_lib = get_coset_kernels_lib()
            self._cpp_eng = self._cpp_lib.engine_create(
                n_q, batch_size, self._shot_tiebreak_rng.ctypes.data_as(u64_p)
            )
            if not self._cpp_eng:
                raise ValueError("engine_create failed for invalid num_qubits/batch_size")

    def _init_runtime_buffers(self, n_q: int, batch_size: int) -> None:
        self.qubit_coords: dict[int, list[float]] = {}
        self.qubit_tags: dict[int, str] = {}
        self.coords_shifts: list[float] = []
        self._active_shots: dict[int, MRQFState] = {}
        self._measurement_columns: list[NDArray[np.bool_]] = []
        self._scratch_mask = np.zeros((n_q, batch_size), dtype=np.bool_)
        self._scratch_x_mask = np.zeros((n_q, batch_size), dtype=np.bool_)
        self._scratch_z_mask = np.zeros((n_q, batch_size), dtype=np.bool_)
        self._cpp_m_buf: NDArray[np.bool_] = np.empty(0, dtype=np.bool_)
        self._correlated_error_occurred: NDArray[np.bool_] = np.zeros(batch_size, dtype=np.bool_)
        self._circuit_time = 0
        self._step_cursor = 0
        self._used_interactively = False
        self._unrolled_ops: list[stim.CircuitInstruction] | None = None  # set by interactive_do
        self._finished_running_circuit = False
        self._final_measurement_records: NDArray[np.bool_] | None = None
        self._detector_flips: NDArray[np.bool_] | None = None
        self._observable_flips: NDArray[np.bool_] | None = None
        self._override_packed_words: PackedSymplecticTables | None = None
        self._in_group_callback: bool = False
        self._unitary_step_ids: dict[int, tuple[UnitaryStepMeta, int, bool]] = {}
        self._clear_epoch: int = 0

    def _engine_dims(self) -> tuple[int, int]:
        """Return authoritative (num_qubits, batch_size) from C++ engine when active."""
        if self._use_cpp and self._cpp_lib is not None and self._cpp_eng is not None:
            nq = int(self._cpp_lib.engine_num_qubits(self._cpp_eng))
            bs = int(self._cpp_lib.engine_batch_size(self._cpp_eng))
            if nq >= 0 and bs > 0:
                return nq, bs
        return self._init_num_qubits, self._init_batch_size

    def _engine_n_words(self) -> int:
        if self._use_cpp and self._cpp_lib is not None and self._cpp_eng is not None:
            nw = int(self._cpp_lib.engine_n_words(self._cpp_eng))
            if nw > 0:
                return nw
        return self._n_words

    def _reject_if_in_group_callback(self, method: str) -> None:
        if self._in_group_callback:
            raise RuntimeError(f"{method} is forbidden inside condition_group_fn")

    def __del__(self) -> None:
        if getattr(self, "_cpp_eng", None) is not None and getattr(self, "_cpp_lib", None) is not None:
            try:
                self._cpp_lib.engine_destroy(self._cpp_eng)
            except Exception:
                pass
            self._cpp_eng = None

    def __copy__(self) -> CosetsideSimulator:
        raise TypeError("CosetsideSimulator does not support shallow/deep copy")

    def __deepcopy__(self, memo: dict[int, Any]) -> CosetsideSimulator:
        raise TypeError("CosetsideSimulator does not support shallow/deep copy")

    def __getstate__(self) -> None:
        raise TypeError("CosetsideSimulator owns native state and cannot be pickled or copied")

    @property
    def active_shots(self) -> dict[int, MRQFState]:
        """Live active-shot dict; each read marks it as possibly mutated (re-pushed to C++).

        Mutate through a fresh read: a reference held across a peek is not re-pushed.
        """
        if self._use_cpp:
            if self._cpp_dirty:
                self._sync_cpp_to_py()
            self._py_dirty = True
        return self._active_shots

    def _sync_cpp_to_py(self) -> None:
        if not self._cpp_dirty or self._cpp_eng is None or self._cpp_lib is None:
            return
        if self._py_dirty:
            # Both flags are only ever set together by an interrupt between the two stores of
            # `_engine_kernel_end`. The Python (pre-kernel) state wins, exactly as it does when
            # a peek re-imports it, so the outcome does not depend on which is read first.
            self._cpp_dirty = False
            return
        lib = self._cpp_lib
        n_q, n_b = self._engine_dims()
        n_w = self._engine_n_words()
        out_idx = np.empty(n_b, dtype=np.int32)
        out_d = np.empty(n_b, dtype=np.int32)
        n_act = int(
            lib.engine_get_active_shot_indices(
                self._cpp_eng,
                out_idx.ctypes.data_as(i32_p),
                out_d.ctypes.data_as(i32_p),
            )
        )
        new_active_set = set(int(out_idx[i]) for i in range(n_act))
        for old_b in list(self._active_shots.keys()):
            if old_b not in new_active_set:
                del self._active_shots[old_b]
        for i in range(n_act):
            b = int(out_idx[i])
            d = int(out_d[i])
            m = max(1, d)
            out = (
                np.empty(n_w, dtype=np.uint64),
                np.empty(n_w, dtype=np.uint64),
                np.empty((m, n_w), dtype=np.uint64),
                np.empty(m, dtype=np.uint8),
                np.empty((m, m), dtype=np.uint8),
                np.empty(m, dtype=np.int32),
            )
            lib.engine_export_shot(
                self._cpp_eng,
                b,
                *(a.ctypes.data_as(u8_p if a.dtype == np.uint8 else u64_p) for a in out[:5]),
                out[5].ctypes.data_as(i32_p),
            )
            self._active_shots[b] = _install_export(
                self._active_shots.get(b), d, n_q, n_w, out
            )
        self._cpp_dirty = False

    def _sync_py_to_cpp(self) -> None:
        # Re-imports only when Python state is newer. `_py_dirty` is raised by every Python
        # writer and cleared here, so a held MRQFState/dict reference mutated after a peek is
        # not re-pushed unless `active_shots` is read again.
        if not self._py_dirty:
            return
        lib = self._cpp_lib
        assert lib is not None and self._cpp_eng is not None
        n_q, n_b = self._engine_dims()
        n_w = self._engine_n_words()
        # Check and copy every entry first (R3-F7), so a bad entry raises before the engine is
        # cleared: the rejected call changes nothing and the C++ shots stay as they were.
        shots = [
            _owned_shot_import(b, st, n_q, n_w, n_b)
            for b, st in list(self._active_shots.items())
        ]
        lib.engine_clear(self._cpp_eng)
        for bi, d, (h, au, a_mat, ell, gamma, p) in shots:
            lib.engine_import_shot(
                self._cpp_eng,
                bi,
                d,
                h.ctypes.data_as(u64_p),
                au.ctypes.data_as(u64_p),
                a_mat.ctypes.data_as(u64_p),
                ell.ctypes.data_as(u8_p),
                gamma.ctypes.data_as(u8_p),
                p.ctypes.data_as(i32_p),
            )
        self._py_dirty = False
        self._cpp_dirty = False

    def _engine_kernel_begin(self) -> None:
        # Call right before a C++ kernel that mutates the engine. An async KeyboardInterrupt after
        # the kernel returns but before `_cpp_dirty` is set would leave both flags clean with the
        # engine ahead of Python (`_sync_py_to_cpp` trusts the flags). Arming `_py_dirty` first
        # makes such an interrupt roll the engine back to the (pre-kernel) Python state at the
        # next peek. If the engine is already ahead (`_cpp_dirty`) the flags stay as they are.
        if not self._cpp_dirty:
            self._py_dirty = True

    def _engine_kernel_end(self) -> None:
        # Call right after the kernel returns: `_cpp_dirty` is raised BEFORE `_py_dirty` is
        # cleared, so no interrupt can leave both flags clean.
        self._cpp_dirty = True
        self._py_dirty = False

    def _init_rngs(self) -> None:
        if self.sync_tableside_rng:
            (
                base_seed,
                self._shot_np_rngs,
                self._tab_rngs,
                self._tab_rngs_x,
            ) = build_sync_rngs(
                self.seed, self._batches_completed, self._init_batch_size, self._init_num_qubits
            )
            self.np_rng = self._shot_np_rngs[0]
        else:
            self._shot_np_rngs = None
            self._tab_rngs = None
            self._tab_rngs_x = None
            base_seed = (
                None
                if self.seed is None
                else derived_seed(self.seed, self._batches_completed)
            )
            self.np_rng = np.random.default_rng(seed=base_seed)

        tie_seed = (
            int(self.np_rng.integers(0, 1 << 63, dtype=np.uint64))
            if base_seed is None
            else int(base_seed)
        )
        b_arange = np.arange(self._init_batch_size, dtype=np.uint64)
        base_hash = np.uint64(
            ((tie_seed + 1) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        )
        self._shot_tiebreak_rng = np.ascontiguousarray(
            base_hash ^ (b_arange * np.uint64(0xBF58476D1CE4E5B9)),
            dtype=np.uint64,
        )
        if (
            getattr(self, "_cpp_eng", None) is not None
            and getattr(self, "_cpp_lib", None) is not None
        ):
            self._cpp_lib.engine_set_rng_states(
                self._cpp_eng, self._shot_tiebreak_rng.ctypes.data_as(u64_p)
            )

    def _draw_tab_rng_bit(self, b_idx: int) -> bool:
        if self._tab_rngs is not None:
            assert self._tab_rngs_x is not None
            return draw_tab_rng_bit(
                self._tab_rngs[b_idx], self._tab_rngs_x[b_idx], self._tab_scratch_q
            )
        z = (
            int(self._shot_tiebreak_rng[b_idx]) + 0x9E3779B97F4A7C15
        ) & 0xFFFFFFFFFFFFFFFF
        self._shot_tiebreak_rng[b_idx] = np.uint64(z)
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return bool((z ^ (z >> 31)) & 1)

    def _sample_pauli_noise_via_tab_rng(
        self,
        op_name: str,
        targets_per_shot: Sequence[Sequence[int]],
        gate_args: Sequence[float],
    ) -> None:
        assert self._tab_rngs is not None and self._tab_rngs_x is not None
        self._scratch_x_mask.fill(False)
        self._scratch_z_mask.fill(False)
        any_x, any_z = exec_sync_pauli_batch(
            self._tab_rngs,
            self._tab_rngs_x,
            op_name,
            targets_per_shot,
            gate_args,
            self._scratch_x_mask,
            self._scratch_z_mask,
        )
        self._flush_xz(any_x, any_z)

    sample_pauli_noise_sync = _sample_pauli_noise_via_tab_rng

    def sync_rngs(self) -> SyncRngs | None:
        """The per-shot sync RNGs, or None unless ``sync_tableside_rng=True``."""
        if self._tab_rngs is None:
            return None
        assert self._tab_rngs_x is not None and self._shot_np_rngs is not None
        return SyncRngs(self._tab_rngs, self._tab_rngs_x, self._shot_np_rngs)

    def measurement_column(self, index: int) -> NDArray[np.bool_]:
        """Measurement record column ``index`` (one bit per shot). Do not modify it in place."""
        return self._measurement_columns[index]

    def overwrite_measurement_column(self, index: int, values: NDArray[np.bool_]) -> None:
        """Overwrite measurement record column ``index`` in place."""
        self._measurement_columns[index][:] = values

    ############################################################################
    # Public lifecycle methods
    ############################################################################

    def run(self) -> list[np.ndarray] | None:
        """Run the full circuit across all batch_size shots.

        When `record_unleaked_to_leaked=True`, returns a list of length `batch_size`
        where element `b` is a 1D `np.ndarray` (`dtype=np.int64`) of the unrolled
        circuit operation indices (`op_idx` into `unroll_circuit(circuit)`) where an
        unleaked-to-leaked transition occurred in shot `b`.
        """
        self._reject_if_in_group_callback("run")
        if self._used_interactively or self._circuit_time != 0:
            raise ValueError(
                "CosetsideSimulator is not in a clean state. "
                "Use .clear() to prepare the simulator for a new run."
            )
        self._do(self.circuit)
        self._finished_running_circuit = True
        self._batches_completed += 1
        if self.record_unleaked_to_leaked:
            return self.get_unleaked_to_leaked_records()
        return None

    def clear(self) -> None:
        """Reset the simulator state for a new batch without recompiling the reference circuit."""
        self._reject_if_in_group_callback("clear")
        if self._use_cpp:
            # Raised first: an interrupt before `engine_clear` must re-import the cleared Python
            # state at the next peek instead of leaving a stale engine behind clean flags.
            self._py_dirty = True
        self._flip_simulator.clear()
        self._active_shots.clear()
        if self._cpp_eng is not None and self._cpp_lib is not None:
            self._cpp_lib.engine_clear(self._cpp_eng)
        self._cpp_dirty = False
        self._py_dirty = False
        self._measurement_columns.clear()
        self._correlated_error_occurred.fill(False)

        self.qubit_coords = {}
        self.qubit_tags = {}
        self.coords_shifts = []

        self._init_rngs()
        self._compiled_op_handler.clear()

        if self.record_unleaked_to_leaked:
            self._unleaked_to_leaked_events = [
                [] for _ in range(self._init_batch_size)
            ]
        if self.record_leakage_events:
            self._leakage_events = [[] for _ in range(self._init_batch_size)]

        self._circuit_time = 0
        self._step_cursor = 0
        self._clear_epoch += 1
        self._used_interactively = False
        self._finished_running_circuit = False

        self._final_measurement_records = None
        self._detector_flips = None
        self._observable_flips = None

    def get_current_circuit_time(self) -> int:
        return self._circuit_time

    def interactive_do(
        self, this: stim.Circuit | stim.CircuitInstruction | stim.CircuitRepeatBlock, /
    ) -> None:
        """Do the next operation(s): the circuit this simulator was constructed with, then any more.

        The reference tape, detector/observable converter and op handler are precomputed from that
        circuit, so `this` must continue it in order (REPEAT blocks unrolled), except that an untagged
        noisy gate (e.g. X_ERROR, M(p)) may have a different noise strength; else ValueError. Past its
        end, untagged operations other than DETECTOR and OBSERVABLE_INCLUDE may follow (their
        measurements are recorded, but get_detector_flips then raises); others, and operations
        acting on qubits >= the circuit's num_qubits, raise ValueError.

        Pass the circuit itself or consecutive parts of it, not `circuit.flattened()` (which drops
        SHIFT_COORDS, rewrites DETECTOR coordinates and can merge operations) or `circuit + more`
        (where stim can merge operations at the seam).
        """
        self._reject_if_in_group_callback("interactive_do")
        if self._finished_running_circuit:
            raise RuntimeError(
                "Cannot do more operations after finishing an interactive run."
            )
        if self._unrolled_ops is None:
            self._unrolled_ops = unroll_circuit(self.circuit)
        self._used_interactively = True
        self._do(this)

    def finish_interactive_run(self) -> None:
        self._reject_if_in_group_callback("finish_interactive_run")
        if not self._finished_running_circuit:
            self._batches_completed += 1
        self._finished_running_circuit = True

    ############################################################################
    # Properties & Lookthroughs
    ############################################################################

    @property
    def num_measurements(self) -> int:
        return self.circuit.num_measurements

    @property
    def num_detectors(self) -> int:
        return self.circuit.num_detectors

    @property
    def num_observables(self) -> int:
        return self.circuit.num_observables

    def num_qubits_in_new_circuit(self) -> int:
        return self.num_qubits

    def _current_packed_words(self) -> PackedSymplecticTables:
        override = getattr(self, "_override_packed_words", None)
        if override is not None:
            return override
        if self._step_cursor == 0:
            return self._compiled_ref._initial_packed_words
        return self._compiled_ref.steps[self._step_cursor - 1]._get_packed_words(
            self._n_words
        )

    def _replays(self, step: StepMeta, op: stim.CircuitInstruction) -> bool:
        """Whether `op` may be run against `step`, the reference step at the cursor.

        interactive_do checked the full op against the circuit; this guards the steps earlier
        interactive runs appended past its end (kept by clear()). A step only depends on the
        op's name and targets, so tags (which op handlers remove) and arguments aren't compared.
        """
        return not self._used_interactively or (
            step.op.name == op.name and op.targets_copy() == step.op.targets_copy()
        )

    def _next_step_meta(self, op: stim.CircuitInstruction) -> StepMeta:
        self._reject_if_in_group_callback("_next_step_meta")
        while self._step_cursor < len(self._compiled_ref.steps):
            cursor = self._step_cursor
            step = self._compiled_ref.steps[cursor]
            if not self._replays(step, op):
                # The live reference is at the end of the tape, not here: appending is wrong.
                raise ValueError(
                    f"interactive_do: `{op}` doesn't match `{step.op}`, operation {cursor} of "
                    "the reference the CosetsideSimulator compiled from its circuit (plus any "
                    "operations earlier interactive runs appended after it). interactive_do can "
                    "only replay that reference in order, then continue past its end."
                )
            if self._step_cursor == cursor:
                self._step_cursor = cursor + 1
                return step
        step = self._compiled_ref.compile_and_append_instruction(op)
        self._step_cursor = len(self._compiled_ref.steps)
        return step

    def do_copies_on_fused_step(
        self,
        op: stim.CircuitInstruction,
        copies: list[stim.CircuitInstruction],
        do_copy: Callable[[stim.CircuitInstruction], None],
    ) -> None:
        """Apply the copies of a stim-fused op in sequence on the op's one (no-op) reference step."""
        self._reject_if_in_group_callback("do_copies_on_fused_step")
        self._next_step_meta(op)
        step_cursor = self._step_cursor - 1
        used_interactively, self._used_interactively = self._used_interactively, False
        try:
            for sub_op in copies:
                self._step_cursor = step_cursor
                do_copy(sub_op)
        finally:
            self._used_interactively = used_interactively
        self._step_cursor = step_cursor + 1

    ############################################################################
    # Internal Instruction Dispatch
    ############################################################################

    def _do(
        self, this: stim.Circuit | stim.CircuitInstruction | stim.CircuitRepeatBlock
    ) -> None:
        if isinstance(this, stim.Circuit):
            for op in this:
                self._do(op)
        elif isinstance(this, stim.CircuitRepeatBlock):
            loop_body = this.body_copy()
            for _ in range(this.repeat_count):
                self._do(loop_body)
        elif isinstance(this, stim.CircuitInstruction):
            if self._used_interactively:
                assert self._unrolled_ops is not None
                check_replay(
                    this,
                    self._unrolled_ops,
                    self._circuit_time,
                    "CosetsideSimulator",
                    num_qubits=self._init_num_qubits,
                )
            self._do_instruction(this)
        else:
            raise NotImplementedError(f"Unsupported circuit object: {type(this)}")

    def _do_instruction(self, op: stim.CircuitInstruction) -> None:
        if apply_coords_op(op, self.qubit_coords, self.qubit_tags, self.coords_shifts):
            self._circuit_time += 1
            return

        self._compiled_op_handler.handle_op(op=op, sss=self)
        self._circuit_time += 1

    def _do_bare_instruction(self, op: stim.CircuitInstruction) -> None:
        """Execute an unconditional instruction across the reference tape and all B shots."""
        self._reject_if_in_group_callback("_do_bare_instruction")
        name = op.name
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE", "TICK", "BARRIER"):
            self._next_step_meta(op)
            return

        if name == "MPAD":
            step_meta = self._next_step_meta(op)
            raw_targets = op.targets_copy()
            for gt in raw_targets:
                bit_val = bool(gt.value) ^ bool(gt.is_inverted_result_target)
                self._measurement_columns.append(
                    np.full(self._init_batch_size, bit_val, dtype=np.bool_)
                )
            if len(raw_targets) > 0:
                self._apply_measurement_flip_noise(
                    step_meta, op, len(raw_targets)
                )
            return

        if name in ("HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"):
            self._next_step_meta(op)
            self._do_heralded_noise(op)
            return

        gd = stim.gate_data(name)
        if gd.is_reset or gd.produces_measurements:
            step_meta = self._next_step_meta(op)
            if isinstance(step_meta, MeasStepMeta):
                self._execute_meas_step(step_meta, op_override=op)
            return

        if gd.is_unitary:
            self.do_conditional_clifford(op, condition_met_mask=None)
            return

        if gd.is_noisy_gate:
            self.do_conditional_pauli_noise(op, condition_met_mask=None)

    def _do_heralded_noise(self, op: stim.CircuitInstruction) -> None:
        raw_targets = op.targets_copy()
        if not raw_targets:
            return
        qs = [
            gt.qubit_value if gt.qubit_value is not None else gt.value
            for gt in raw_targets
        ]
        self._scratch_x_mask.fill(False)
        self._scratch_z_mask.fill(False)
        if self._tab_rngs is not None:
            assert self._tab_rngs_x is not None
            any_x, any_z, cols = exec_sync_heralded_noise(
                self._tab_rngs,
                self._tab_rngs_x,
                op,
                set(qs),
                len(qs),
                self._scratch_x_mask,
                self._scratch_z_mask,
            )
            self._flush_xz(any_x, any_z)
            self._measurement_columns.extend(cols)
            return
        args = op.gate_args_copy()
        any_x = any_z = False
        is_erase = op.name == "HERALDED_ERASE"
        p_erase = args[0] if (is_erase and args) else 0.0
        ci = args[0] if len(args) > 0 else 0.0
        cx = ci + (args[1] if len(args) > 1 else 0.0)
        cy = cx + (args[2] if len(args) > 2 else 0.0)
        cz = cy + (args[3] if len(args) > 3 else 0.0)
        for gt, q in zip(raw_targets, qs):
            if is_erase:
                hk = self.np_rng.random(self._init_batch_size) < p_erase
                pi = self.np_rng.integers(0, 4, size=self._init_batch_size)
                xk = hk & ((pi == 1) | (pi == 2))
                zk = hk & ((pi == 2) | (pi == 3))
            else:
                u = self.np_rng.random(self._init_batch_size)
                hk = u < cz
                xk = (u >= ci) & (u < cy)
                zk = (u >= cx) & (u < cz)
            if np.any(xk):
                self._scratch_x_mask[q, :] ^= xk
                any_x = True
            if np.any(zk):
                self._scratch_z_mask[q, :] ^= zk
                any_z = True
            self._measurement_columns.append(
                ~hk if gt.is_inverted_result_target else hk
            )
        self._flush_xz(any_x, any_z)

    ############################################################################
    # Phase 1 & Phase 2: Unitary Gates, Inverse Clifford Injection & Pauli Noise
    ############################################################################

    def _apply_unitary_to_frames(self, op: stim.CircuitInstruction) -> None:
        """Apply an ideal unitary Clifford instruction to stim.FlipSimulator in O(1) time (O(0) on MRQF)."""
        if op.name in ("I", "II", "I_ERROR", "II_ERROR"):
            return
        self._flip_simulator.do(op)

    def _cpp_out_args(self) -> tuple:
        """Scratch X/Z mask pointers + any-X/any-Z out flags (the shared tail of the batch kernels)."""
        return (
            self._scratch_x_mask.ctypes.data_as(u8_p),
            self._scratch_z_mask.ctypes.data_as(u8_p),
            ctypes.byref(self._any_x_c),
            ctypes.byref(self._any_z_c),
        )

    def _flush_xz(self, any_x: int, any_z: int) -> None:
        """XOR the scratch X/Z masks into the frames (p=1 broadcasts are pure XORs and draw no randomness)."""
        if any_x:
            self._flip_simulator.broadcast_pauli_errors(pauli="X", mask=self._scratch_x_mask, p=1.0)
        if any_z:
            self._flip_simulator.broadcast_pauli_errors(pauli="Z", mask=self._scratch_z_mask, p=1.0)

    def _broadcast_flips(
        self, pauli: str, qubits: Sequence[int], flips: NDArray[np.bool_]
    ) -> None:
        """XOR row i of `flips` onto qubit qubits[i] of the scratch mask and apply it as a p=1 (RNG-free) Pauli."""
        if np.any(flips):
            self._scratch_mask.fill(False)
            for i, q in enumerate(qubits):
                self._scratch_mask[q, :] ^= flips[i]
            self._flip_simulator.broadcast_pauli_errors(
                pauli=pauli, mask=self._scratch_mask, p=1.0
            )

    def _broadcast_pair_flips(self, targets, x1, z1, x2, z2) -> None:
        """Pair variant: targets are flattened (q1, q2) pairs; row g of x1/z1 (x2/z2) acts on pair g's q1 (q2)."""
        for pauli, f1, f2 in (("X", x1, x2), ("Z", z1, z2)):
            flips = np.stack([f1, f2], axis=1).reshape(-1, self._init_batch_size)
            self._broadcast_flips(pauli, targets, flips)

    def _evict_healed_shots(
        self,
        evict_words_or_step: tuple[NDArray[np.uint64], NDArray[np.uint64]] | StepMeta,
        *,
        xs_base: NDArray[np.bool_] | None = None,
        zs_base: NDArray[np.bool_] | None = None,
        accum_x_mask: NDArray[np.bool_] | None = None,
        accum_z_mask: NDArray[np.bool_] | None = None,
    ) -> tuple[bool, bool]:
        """Evict any active shot with d == 0 and fold D_t(h) = B_t X(h) B_t^dagger into stim.FlipSimulator."""
        healed = [b_idx for b_idx, mrqf in self._active_shots.items() if mrqf.d == 0]
        if not healed:
            return False, False

        nz_b: list[int] = []
        nz_h: list[NDArray[np.uint64]] = []
        for b_idx in healed:
            mrqf = self._active_shots.pop(b_idx)
            if _words_any_nonzero(mrqf.h_words):
                nz_b.append(b_idx)
                nz_h.append(mrqf.h_words)

        if not nz_b:
            return False, False

        nz_b_arr = np.asarray(nz_b, dtype=np.intp)
        x2z_w, z2z_w = (
            evict_words_or_step
            if isinstance(evict_words_or_step, tuple)
            else evict_words_or_step.get_evict_words(self._n_words)
        )
        h_w_mat = np.stack(nz_h, axis=0)
        xor_x = z2z_w[:, None, 0] & h_w_mat[None, :, 0]
        xor_z = x2z_w[:, None, 0] & h_w_mat[None, :, 0]
        for wb in range(1, self._n_words):
            xor_x ^= z2z_w[:, None, wb] & h_w_mat[None, :, wb]
            xor_z ^= x2z_w[:, None, wb] & h_w_mat[None, :, wb]
        x_phys_mat = (np.bitwise_count(xor_x) & 1).astype(np.bool_)
        z_phys_mat = (np.bitwise_count(xor_z) & 1).astype(np.bool_)

        any_x = bool(np.any(x_phys_mat))
        any_z = bool(np.any(z_phys_mat))

        if any_x:
            if accum_x_mask is not None:
                accum_x_mask[:, nz_b_arr] ^= x_phys_mat
            else:
                self._scratch_mask.fill(False)
                self._scratch_mask[:, nz_b_arr] = x_phys_mat
                self._flip_simulator.broadcast_pauli_errors(
                    pauli="X", mask=self._scratch_mask, p=1.0
                )
            if xs_base is not None:
                xs_base[:, nz_b_arr] ^= x_phys_mat

        if any_z:
            if accum_z_mask is not None:
                accum_z_mask[:, nz_b_arr] ^= z_phys_mat
            else:
                self._scratch_mask.fill(False)
                self._scratch_mask[:, nz_b_arr] = z_phys_mat
                self._flip_simulator.broadcast_pauli_errors(
                    pauli="Z", mask=self._scratch_mask, p=1.0
                )
            if zs_base is not None:
                zs_base[:, nz_b_arr] ^= z_phys_mat

        return any_x, any_z

    def _inject_inverse_cocycle_on_shot(
        self,
        b_idx: int,
        f_arrs: list[NDArray],
        g_idx: int,
        n_fac: int,
        f_x_local: NDArray[np.bool_],
        f_z_local: NDArray[np.bool_],
    ) -> None:
        """Inject E = U_t^dagger on shot b_idx via P1, P2, or PCP."""
        mrqf = self._active_shots.get(b_idx)
        if mrqf is None:
            mrqf = MRQFState(self._init_num_qubits, self._n_words)
            self._active_shots[b_idx] = mrqf

        f_kinds, f_lx1, f_lz1, f_a1, f_b1, f_k1, f_lx2, f_lz2, f_a2, f_b2, f_k2 = f_arrs
        fx = f_x_local.astype(np.uint8, copy=False)
        fz = f_z_local.astype(np.uint8, copy=False)
        for f_i in range(n_fac):
            kind = int(f_kinds[g_idx, f_i])
            a1 = f_a1[g_idx, f_i]
            b1 = f_b1[g_idx, f_i]
            if kind == 0:
                mrqf.p1_apply_pauli(a1, b1)
                continue
            gamma = (
                int(np.count_nonzero((f_lx1[g_idx, f_i] & fz) ^ (f_lz1[g_idx, f_i] & fx)))
                & 1
            )
            kappa_shot = (int(f_k1[g_idx, f_i]) + (gamma << 1)) & 3
            if kind == 2:
                gamma2 = (
                    int(np.count_nonzero((f_lx2[g_idx, f_i] & fz) ^ (f_lz2[g_idx, f_i] & fx)))
                    & 1
                )
                kappa2_shot = (int(f_k2[g_idx, f_i]) + (gamma2 << 1)) & 3
                mrqf.p2_apply_controlled_pauli(
                    a1,
                    b1,
                    kappa_shot,
                    f_a2[g_idx, f_i],
                    f_b2[g_idx, f_i],
                    kappa2_shot,
                )
            else:
                mrqf.p2_apply_quarter_turn(a1, b1, kappa_shot)

    def _past_end_step_meta(
        self, op: stim.CircuitInstruction
    ) -> tuple[StepMeta, ReferenceCHPTableau, PackedSymplecticTables]:
        return self._compiled_ref.compile_instruction(op)

    def _past_end_unitary_rows(self, op: stim.CircuitInstruction) -> int | None:
        step, _, _ = self._past_end_step_meta(op)
        return len(step.target_groups) if isinstance(step, UnitaryStepMeta) else None

    def _check_op_qubit_bounds(self, op: stim.CircuitInstruction) -> None:
        n_q, _ = self._engine_dims()
        for t in op.targets_copy():
            q = t.qubit_value
            if q is not None and not (0 <= q < n_q):
                raise IndexError(f"qubit {q} out of range [0, {n_q})")

    def _dcc_expected_mask_shapes(
        self, op: stim.CircuitInstruction, shape: tuple[int, ...]
    ) -> tuple[tuple[int, ...], str, str, bool]:
        _, cols = self._engine_dims()
        if _is_mixed_two_qubit(op):
            _check_mask_shape(
                "condition_met_mask", shape, len(op.target_groups()), cols, f"{op.name} group"
            )
            return (
                (_count_all_quantum_groups(op), cols),
                "condition_met_mask restricted to the record-free groups",
                "step group",
                True,
            )
        return shape, "condition_met_mask", f"{op.name} group", False

    def _check_dcc_mask(
        self,
        op: stim.CircuitInstruction,
        shape: tuple[int, ...],
        has_group_fn: bool,
        eff: tuple[tuple[int, ...], str, str, bool] | None = None,
    ) -> None:
        """Raise ValueError if a `condition_met_mask` of this `shape` does not fit `op`."""
        steps = self._compiled_ref.steps
        _, cols = self._engine_dims()
        rows: int | None = None
        if self._step_cursor >= len(steps):
            rows = self._past_end_unitary_rows(op)
        else:
            step = steps[self._step_cursor]
            if isinstance(step, UnitaryStepMeta) and self._replays(step, op):
                rows = len(step.target_groups)
        eff_shape, name, rows_are, _ = (
            eff if eff is not None else self._dcc_expected_mask_shapes(op, shape)
        )
        if rows is not None and not has_group_fn:
            _check_mask_shape(name, eff_shape, rows, cols, rows_are)

    def _rollback_dcc_step(self, step_meta: StepMeta, steps_len_before: int) -> None:
        steps = self._compiled_ref.steps
        if len(steps) > steps_len_before and steps[-1] is step_meta:
            steps.pop()
            prev = (
                steps[-1]._get_packed_words(self._n_words)
                if steps
                else self._compiled_ref._initial_packed_words
            )
            self._compiled_ref._last_packed_snapshot = prev
            self._compiled_ref.ref_chp = ReferenceCHPTableau.from_packed(
                self._compiled_ref.num_qubits, prev
            )
            self._step_cursor = len(steps)
        elif self._step_cursor > 0 and steps[self._step_cursor - 1] is step_meta:
            self._step_cursor -= 1

    @staticmethod
    def _step_matches_op(step_meta: StepMeta, op: stim.CircuitInstruction) -> bool:
        return step_meta.op.name == op.name and op.targets_copy() == step_meta.op.targets_copy()

    def _dcc_validate_and_step(
        self,
        op: stim.CircuitInstruction,
        eff: tuple[tuple[int, ...], str, str, bool] | None,
        has_group_fn: bool,
        cursor_at_check: int,
        clear_at_check: int,
    ) -> tuple[
        StepMeta,
        PackedSymplecticTables,
        int,
        tuple[ReferenceCHPTableau, PackedSymplecticTables] | None,
    ]:
        clear_at_entry = self._clear_epoch
        if eff is not None and not has_group_fn and self._step_cursor != cursor_at_check:
            self._check_dcc_step_candidate(op, eff)
        packed_words_before = self._current_packed_words()
        steps = self._compiled_ref.steps
        steps_len_before = len(steps)
        if self._step_cursor < steps_len_before:
            step_meta = self._next_step_meta(op)
            pending: tuple[ReferenceCHPTableau, PackedSymplecticTables] | None = None
        else:
            step_meta, next_chp, next_snap = self._past_end_step_meta(op)
            pending = (next_chp, next_snap)
        if eff is not None and not has_group_fn and isinstance(step_meta, UnitaryStepMeta):
            try:
                eff_shape, name, rows_are, _ = eff
                _, cols = self._engine_dims()
                _check_mask_shape(name, eff_shape, len(step_meta.target_groups), cols, rows_are)
            except ValueError:
                if pending is None and self._clear_epoch == clear_at_entry:
                    self._rollback_dcc_step(step_meta, steps_len_before)
                raise
        if self._clear_epoch != clear_at_check:
            if pending is None and self._clear_epoch == clear_at_entry:
                self._rollback_dcc_step(step_meta, steps_len_before)
            raise ValueError(
                f"do_conditional_clifford: simulator was cleared before `{op}` completed"
            )
        if (
            pending is None
            and self._step_cursor != cursor_at_check + 1
            and self._step_cursor > 0
            and self._compiled_ref.steps[self._step_cursor - 1] is step_meta
            and getattr(self, "_override_packed_words", None) is None
        ):
            idx = self._step_cursor - 1
            packed_words_before = (
                self._compiled_ref._initial_packed_words
                if idx == 0
                else self._compiled_ref.steps[idx - 1]._get_packed_words(self._n_words)
            )
        return step_meta, packed_words_before, steps_len_before, pending

    def _dcc_sync_step(self, st: _DccState) -> None:
        while True:
            if self._clear_epoch != st.clear_at_step:
                raise ValueError(
                    "do_conditional_clifford: simulator was cleared during re-entrant call"
                )
            if st.pending is None:
                if (
                    self._step_cursor != st.cursor_after_step
                    or len(self._compiled_ref.steps) != st.steps_len_before
                ):
                    raise ValueError(
                        "do_conditional_clifford: simulator state changed during re-entrant call"
                    )
                return
            if (
                len(self._compiled_ref.steps) == st.steps_len_before
                and self._step_cursor == st.steps_len_before
            ):
                return
            st.step_meta, st.packed_words_before, st.steps_len_before, st.pending = (
                self._dcc_validate_and_step(
                    st.orig_op, st.eff, st.has_group_fn, self._step_cursor, st.clear_at_step
                )
            )
            st.cursor_after_step = self._step_cursor
            if st.pending is None and not self._step_matches_op(st.step_meta, st.orig_op):
                self._rollback_dcc_step(st.step_meta, st.steps_len_before)
                raise ValueError(
                    f"do_conditional_clifford: `{st.orig_op}` doesn't match step "
                    f"`{st.step_meta.op}` at cursor"
                )
            if st.quant_indices is not None:
                try:
                    _, _, st.rec_flips = self._dcc_collect_mixed_record(st.orig_op, st.mask, True)
                except Exception:
                    if st.pending is None:
                        self._rollback_dcc_step(st.step_meta, st.steps_len_before)
                    raise
            if isinstance(st.step_meta, UnitaryStepMeta) and st.quant_indices is not None:
                assert st.step_meta.quantum_op is not None
                st.op_exec = st.step_meta.quantum_op

    def _dcc_commit_and_flip(
        self, st: _DccState, mark_committed: Callable[[], None] | None = None
    ) -> StepMeta:
        self._dcc_sync_step(st)
        if mark_committed is not None:
            mark_committed()
        if st.pending is not None:
            next_chp, next_snap = st.pending
            self._compiled_ref.append_compiled_step(st.step_meta, next_chp, next_snap)
            self._step_cursor = len(self._compiled_ref.steps)
        for pauli_ch, q_tgt, act in st.rec_flips:
            self._broadcast_flips(pauli_ch, [q_tgt], act)
        return st.step_meta

    def _check_dcc_step_candidate(
        self, op: stim.CircuitInstruction, eff: tuple[tuple[int, ...], str, str, bool]
    ) -> None:
        steps = self._compiled_ref.steps
        _, cols = self._engine_dims()
        rows: int | None = None
        if self._step_cursor >= len(steps):
            rows = self._past_end_unitary_rows(op)
        else:
            step = steps[self._step_cursor]
            if isinstance(step, UnitaryStepMeta) and self._replays(step, op):
                rows = len(step.target_groups)
        if rows is not None:
            eff_shape, name, rows_are, _ = eff
            _check_mask_shape(name, eff_shape, rows, cols, rows_are)

    def _dcc_collect_mixed_record(
        self,
        op: stim.CircuitInstruction,
        mask: NDArray[Any] | None,
        is_mixed: bool | None = None,
    ) -> tuple[list[int] | None, NDArray[Any] | None, list[tuple[str, int, NDArray[np.bool_]]]]:
        if is_mixed is False:
            return None, mask, []
        if is_mixed is None and not _is_mixed_two_qubit(op):
            return None, mask, []
        name = op.name
        n_q, _ = self._engine_dims()
        quant_indices: list[int] = []
        rec_flips: list[tuple[str, int, NDArray[np.bool_]]] = []
        for g_idx, (t0, t1) in enumerate(op.target_groups()):
            if t0.qubit_value is not None and t1.qubit_value is not None:
                quant_indices.append(g_idx)
                continue
            rec, tq, paulis = (
                (t0, t1, _REC_FIRST_PAULI)
                if t0.is_measurement_record_target
                else (t1, t0, _REC_SECOND_PAULI)
            )
            if rec.is_measurement_record_target and tq.qubit_value is not None:
                q_tgt = tq.qubit_value
                if not (0 <= q_tgt < n_q):
                    raise IndexError(f"qubit {q_tgt} out of range [0, {n_q})")
                active = self._measurement_columns[rec.value]
                if mask is not None:
                    active = active & mask[g_idx]
                act = active[None, :]
                if np.any(act):
                    self._scratch_mask[:0] ^= act[:0]
                rec_flips.append((paulis.get(name, "Z"), q_tgt, act))
        q_mask = mask[quant_indices, :] if mask is not None else None
        return quant_indices, q_mask, rec_flips

    def _dcc_apply_mixed_record(
        self,
        op: stim.CircuitInstruction,
        mask: NDArray[Any] | None,
        is_mixed: bool | None = None,
    ) -> tuple[list[int] | None, NDArray[Any] | None]:
        quant_indices, q_mask, rec_flips = self._dcc_collect_mixed_record(op, mask, is_mixed)
        for pauli_ch, q_tgt, act in rec_flips:
            self._broadcast_flips(pauli_ch, [q_tgt], act)
        return quant_indices, q_mask

    def _own_group_row(self, raw_row: Any, g_idx: int, cols: int) -> NDArray[np.bool_]:
        cm_row = _own_mask(raw_row)
        if cm_row.shape != (cols,):
            raise ValueError(
                f"condition_group_fn({g_idx}) returned shape {cm_row.shape}, expected ({cols},)"
            )
        return cm_row.astype(np.bool_, copy=False)

    def _register_unitary_step_cpp(
        self, step_meta: UnitaryStepMeta, *, need_group_evict: bool = False
    ) -> int:
        lib, eng = self._cpp_lib, self._cpp_eng
        assert lib is not None and eng is not None
        n_q, _ = self._engine_dims()
        n_w = self._engine_n_words()
        key = id(step_meta)
        cached = self._unitary_step_ids.get(key)
        if cached is not None:
            step_id, has_ge = cached[1], cached[2]
        else:
            num_g, arity, n_fac, t_groups, f_arrs = step_meta.get_cpp_cocycle_arrays(n_w)
            x2z_w, z2z_w = step_meta.get_evict_words(n_w)
            tg_c = np.ascontiguousarray(t_groups, dtype=np.int64)
            fa_c = [
                np.ascontiguousarray(a, dtype=np.uint64 if a.dtype == np.uint64 else np.uint8)
                for a in f_arrs
            ]
            x2z_c = np.ascontiguousarray(x2z_w, dtype=np.uint64)
            z2z_c = np.ascontiguousarray(z2z_w, dtype=np.uint64)
            rc = int(
                lib.engine_register_unitary_step(
                    eng, num_g, arity, n_fac, tg_c.ctypes.data_as(i64_p),
                    *(a.ctypes.data_as(u64_p if a.dtype == np.uint64 else u8_p) for a in fa_c),
                    n_q, n_w, x2z_c.ctypes.data_as(u64_p), z2z_c.ctypes.data_as(u64_p),
                )
            )
            if rc < 0:
                raise ValueError(f"engine_register_unitary_step failed with code {rc}")
            step_id, has_ge = rc, False
            self._unitary_step_ids[key] = (step_meta, step_id, False)
        if need_group_evict and not has_ge:
            g_pws = step_meta._group_packed_words
            assert g_pws is not None
            num_g = len(step_meta.target_groups)
            gx = np.ascontiguousarray(
                np.stack([pw[1] for pw in g_pws], axis=0)
                if num_g > 0 else np.empty((0, n_q, n_w), dtype=np.uint64),
                dtype=np.uint64,
            )
            gz = np.ascontiguousarray(
                np.stack([pw[3] for pw in g_pws], axis=0)
                if num_g > 0 else np.empty((0, n_q, n_w), dtype=np.uint64),
                dtype=np.uint64,
            )
            rc = int(
                lib.engine_register_unitary_group_evict(
                    eng, step_id, num_g, n_q, n_w,
                    gx.ctypes.data_as(u64_p), gz.ctypes.data_as(u64_p),
                )
            )
            if rc < 0:
                raise ValueError(f"engine_register_unitary_group_evict failed with code {rc}")
            self._unitary_step_ids[key] = (step_meta, step_id, True)
        return step_id

    def _dcc_disjoint_cpp(
        self,
        op: stim.CircuitInstruction,
        step_meta: UnitaryStepMeta,
        mask: NDArray[Any],
        steps_len_before: int,
        can_rollback: bool,
    ) -> None:
        lib = self._cpp_lib
        assert lib is not None and self._cpp_eng is not None
        _, cols = self._engine_dims()
        n_w = self._engine_n_words()
        num_g = len(step_meta.target_groups)
        try:
            cm_c = np.ascontiguousarray(mask, dtype=np.uint8)
            _check_mask_shape("condition_met_mask", cm_c.shape, num_g, cols, "step group")
        except Exception:
            if can_rollback:
                self._rollback_dcc_step(step_meta, steps_len_before)
            raise
        num_g, arity, n_fac, t_groups, f_arrs = step_meta.get_cpp_cocycle_arrays(n_w)
        self._sync_py_to_cpp()
        self._apply_unitary_to_frames(op)
        xs_base, zs_base, _, _, _ = self._flip_simulator.to_numpy(output_xs=True, output_zs=True)
        x2z_w, z2z_w = step_meta.get_evict_words(n_w)
        self._scratch_x_mask.fill(False)
        self._scratch_z_mask.fill(False)
        xs_c = np.ascontiguousarray(xs_base, dtype=np.uint8)
        zs_c = np.ascontiguousarray(zs_base, dtype=np.uint8)
        _, cols = self._engine_dims()
        _check_mask_shape("condition_met_mask", cm_c.shape, num_g, cols, "step group")
        self._engine_kernel_begin()
        lib.engine_batch_inject_general_clifford(
            self._cpp_eng,
            num_g,
            arity,
            n_fac,
            t_groups.ctypes.data_as(i64_p),
            cm_c.ctypes.data_as(u8_p),
            xs_c.ctypes.data_as(u8_p),
            zs_c.ctypes.data_as(u8_p),
            *(a.ctypes.data_as(u64_p if a.dtype == np.uint64 else u8_p) for a in f_arrs),
            x2z_w.ctypes.data_as(u64_p),
            z2z_w.ctypes.data_as(u64_p),
            *self._cpp_out_args(),
        )
        self._engine_kernel_end()
        self._flush_xz(self._any_x_c.value, self._any_z_c.value)

    def _dcc_disjoint_cpp_v3(
        self,
        op: stim.CircuitInstruction,
        step_meta: UnitaryStepMeta,
        mask: NDArray[Any],
        steps_len_before: int,
        can_rollback: bool,
    ) -> None:
        lib, eng = self._cpp_lib, self._cpp_eng
        assert lib is not None and eng is not None
        n_q, n_b = self._engine_dims()
        num_g = len(step_meta.target_groups)
        try:
            cm_c = np.ascontiguousarray(_own_mask(mask), dtype=np.uint8)
            _check_mask_shape("condition_met_mask", cm_c.shape, num_g, n_b, "step group")
        except Exception:
            if can_rollback:
                self._rollback_dcc_step(step_meta, steps_len_before)
            raise
        self._sync_py_to_cpp()
        step_id = self._register_unitary_step_cpp(step_meta, need_group_evict=False)
        reg_g = int(lib.engine_unitary_step_num_groups(eng, step_id))
        _check_mask_shape("condition_met_mask", cm_c.shape, reg_g, n_b, "step group")
        self._apply_unitary_to_frames(op)
        xs_base, zs_base, _, _, _ = self._flip_simulator.to_numpy(output_xs=True, output_zs=True)
        self._scratch_x_mask.fill(False)
        self._scratch_z_mask.fill(False)
        self._any_x_c.value = 0
        self._any_z_c.value = 0
        xs_c = np.ascontiguousarray(xs_base, dtype=np.uint8)
        zs_c = np.ascontiguousarray(zs_base, dtype=np.uint8)
        self._engine_kernel_begin()
        rc = int(
            lib.engine_batch_inject_general_clifford_v3(
                eng, step_id, cm_c.shape[0], cm_c.shape[1], cm_c.ctypes.data_as(u8_p),
                n_q, n_b, xs_c.ctypes.data_as(u8_p), n_q, n_b, zs_c.ctypes.data_as(u8_p),
                n_q, n_b, self._scratch_x_mask.ctypes.data_as(u8_p),
                n_q, n_b, self._scratch_z_mask.ctypes.data_as(u8_p),
                ctypes.byref(self._any_x_c), ctypes.byref(self._any_z_c),
            )
        )
        self._engine_kernel_end()
        if rc < 0:
            raise ValueError(f"engine_batch_inject_general_clifford_v3 failed ({rc})")
        self._flush_xz(self._any_x_c.value, self._any_z_c.value)

    def _dcc_disjoint_py(
        self,
        op: stim.CircuitInstruction,
        step_meta: UnitaryStepMeta,
        mask: NDArray[Any],
        *,
        failed_groups: list[tuple[int, NDArray[np.intp]]] | None = None,
    ) -> None:
        if failed_groups is None:
            failed_groups = [
                (int(g), np.flatnonzero(~mask[int(g)]))
                for g in np.flatnonzero(~np.all(mask, axis=1))
            ]
        self._apply_unitary_to_frames(op)
        xs_base, zs_base, _, _, _ = self._flip_simulator.to_numpy(output_xs=True, output_zs=True)
        if failed_groups:
            _, _, n_fac, _, f_arrs = step_meta.get_cpp_cocycle_arrays(self._n_words)
            for g_idx, failed_shots in failed_groups:
                group = step_meta.target_groups[g_idx]
                group_arr = np.array(group, dtype=np.intp)
                for b_idx_np in failed_shots:
                    b_idx = int(b_idx_np)
                    f_x_local = xs_base[group_arr, b_idx]
                    f_z_local = zs_base[group_arr, b_idx]
                    self._inject_inverse_cocycle_on_shot(
                        b_idx, f_arrs, g_idx, n_fac, f_x_local, f_z_local
                    )
        if self._active_shots and any(m.d == 0 for m in self._active_shots.values()):
            self._evict_healed_shots(step_meta, xs_base=xs_base, zs_base=zs_base)

    def _eval_group_condition(
        self,
        g_idx: int,
        pw_before: PackedSymplecticTables,
        group_pws: list[PackedSymplecticTables],
        mask: NDArray[Any] | None,
        condition_group_fn: Any,
        cols: int,
    ) -> NDArray[Any]:
        if condition_group_fn is not None:
            prev_cb = self._in_group_callback
            self._override_packed_words = pw_before if g_idx == 0 else group_pws[g_idx - 1]
            self._in_group_callback = True
            try:
                raw_row = condition_group_fn(g_idx)
                self._override_packed_words = None
                return self._own_group_row(raw_row, g_idx, cols)
            finally:
                self._override_packed_words = None
                self._in_group_callback = prev_cb
        assert mask is not None
        return mask[g_idx]

    def _dcc_sequential_py(
        self,
        step_meta: UnitaryStepMeta,
        packed_words_before: PackedSymplecticTables,
        mask: NDArray[Any] | None,
        condition_group_fn: Any,
        *,
        first_row: NDArray[Any] | None = None,
        failed_by_group: list[NDArray[np.intp]] | None = None,
    ) -> None:
        if self._use_cpp and self._cpp_dirty:
            self._sync_cpp_to_py()
        _, cols = self._engine_dims()
        group_pws = step_meta.ensure_group_packed_words(packed_words_before)
        _, _, n_fac, _, f_arrs = step_meta.get_cpp_cocycle_arrays(self._n_words)
        for g_idx, group in enumerate(step_meta.target_groups):
            if failed_by_group is not None:
                failed_shots = failed_by_group[g_idx]
            else:
                cm_row = (
                    first_row
                    if g_idx == 0 and first_row is not None
                    else self._eval_group_condition(
                        g_idx, packed_words_before, group_pws, mask, condition_group_fn, cols
                    )
                )
                failed_shots = np.where(~cm_row)[0]
            self._apply_unitary_to_frames(step_meta.group_ops[g_idx])
            if len(failed_shots) == 0:
                continue
            xs_base, zs_base, _, _, _ = self._flip_simulator.to_numpy(
                output_xs=True, output_zs=True
            )
            group_arr = np.array(group, dtype=np.intp)
            if self._use_cpp:
                self._py_dirty = True
            for b_idx_np in failed_shots:
                b_idx = int(b_idx_np)
                f_x_local = xs_base[group_arr, b_idx]
                f_z_local = zs_base[group_arr, b_idx]
                self._inject_inverse_cocycle_on_shot(
                    b_idx, f_arrs, g_idx, n_fac, f_x_local, f_z_local
                )
            if self._active_shots:
                self._evict_healed_shots(
                    (group_pws[g_idx][1], group_pws[g_idx][3]),
                    xs_base=xs_base,
                    zs_base=zs_base,
                )
        if self._use_cpp:
            self._py_dirty = True

    def _dcc_sequential_cpp(
        self,
        step_meta: UnitaryStepMeta,
        packed_words_before: PackedSymplecticTables,
        mask: NDArray[Any] | None,
        condition_group_fn: Any,
    ) -> None:
        lib, eng = self._cpp_lib, self._cpp_eng
        assert lib is not None and eng is not None
        n_q, n_b = self._engine_dims()
        num_g = len(step_meta.target_groups)
        if num_g == 0:
            return
        cm_all: NDArray[np.uint8] | None = None
        if condition_group_fn is None:
            cm_all = np.ascontiguousarray(_own_mask(mask), dtype=np.uint8)
            _check_mask_shape("condition_met_mask", cm_all.shape, num_g, n_b, "step group")
        group_pws = step_meta.ensure_group_packed_words(packed_words_before)
        step_id: int | None = None
        for g_idx in range(num_g):
            cm_row = (
                cm_all[g_idx]
                if cm_all is not None
                else self._eval_group_condition(
                    g_idx, packed_words_before, group_pws, mask, condition_group_fn, n_b
                )
            )
            cm_c = np.ascontiguousarray(cm_row, dtype=np.uint8)
            self._apply_unitary_to_frames(step_meta.group_ops[g_idx])
            if bool(np.all(cm_c)):
                continue
            if self._py_dirty or step_id is None:
                self._sync_py_to_cpp()
            if step_id is None:
                step_id = self._register_unitary_step_cpp(step_meta, need_group_evict=True)
            xs_base, zs_base, _, _, _ = self._flip_simulator.to_numpy(
                output_xs=True, output_zs=True
            )
            self._scratch_x_mask.fill(False)
            self._scratch_z_mask.fill(False)
            self._any_x_c.value = 0
            self._any_z_c.value = 0
            xs_c = np.ascontiguousarray(xs_base, dtype=np.uint8)
            zs_c = np.ascontiguousarray(zs_base, dtype=np.uint8)
            self._engine_kernel_begin()
            rc = int(
                lib.engine_inject_group_v3(
                    eng, step_id, g_idx, cm_c.shape[0], cm_c.ctypes.data_as(u8_p),
                    n_q, n_b, xs_c.ctypes.data_as(u8_p), n_q, n_b, zs_c.ctypes.data_as(u8_p),
                    n_q, n_b, self._scratch_x_mask.ctypes.data_as(u8_p),
                    n_q, n_b, self._scratch_z_mask.ctypes.data_as(u8_p),
                    ctypes.byref(self._any_x_c), ctypes.byref(self._any_z_c),
                )
            )
            self._engine_kernel_end()
            if rc < 0:
                raise ValueError(f"engine_inject_group_v3 failed with code {rc}")
            self._flush_xz(self._any_x_c.value, self._any_z_c.value)

    def _dcc_run_disjoint(
        self,
        st: _DccState,
        step_meta: UnitaryStepMeta,
        cols: int,
        mark_committed: Callable[[], None],
    ) -> None:
        assert st.q_mask is not None
        use_cpp = self._use_cpp and self._cpp_lib is not None and self._cpp_eng is not None
        if use_cpp:
            cm_c = np.ascontiguousarray(st.q_mask, dtype=np.uint8)
            _check_mask_shape(
                "condition_met_mask", cm_c.shape, len(step_meta.target_groups), cols, "step group"
            )
            step_now = self._dcc_commit_and_flip(st, mark_committed)
            assert isinstance(step_now, UnitaryStepMeta)
            self._dcc_disjoint_cpp(
                st.op_exec, step_now, cm_c, st.steps_len_before, st.quant_indices is None
            )
            return
        failed_groups = [
            (int(g), np.flatnonzero(~st.q_mask[int(g)]))
            for g in np.flatnonzero(~np.all(st.q_mask, axis=1))
        ]
        step_now = self._dcc_commit_and_flip(st, mark_committed)
        assert isinstance(step_now, UnitaryStepMeta)
        self._dcc_disjoint_py(st.op_exec, step_now, st.q_mask, failed_groups=failed_groups)

    def _dcc_run_unitary(
        self,
        st: _DccState,
        all_met: bool,
        condition_group_fn: Any,
        mark_committed: Callable[[], None],
    ) -> None:
        _, cols = self._engine_dims()
        self._dcc_sync_step(st)
        op, step_meta = st.op_exec, st.step_meta
        assert isinstance(step_meta, UnitaryStepMeta)
        name = op.name
        if all_met:
            self._dcc_commit_and_flip(st, mark_committed)
            self._apply_unitary_to_frames(st.op_exec)
            return
        if name in ("I", "II", "I_ERROR", "II_ERROR"):
            if condition_group_fn is not None:
                self._in_group_callback = True
                try:
                    for g_idx in range(len(step_meta.target_groups)):
                        self._override_packed_words = st.packed_words_before
                        raw_row = condition_group_fn(g_idx)
                        self._override_packed_words = None
                        self._own_group_row(raw_row, g_idx, cols)
                    self._dcc_commit_and_flip(st, mark_committed)
                finally:
                    self._in_group_callback = False
                    self._override_packed_words = None
                return
            self._dcc_commit_and_flip(st, mark_committed)
            return
        if condition_group_fn is None and name in ("X", "Y", "Z"):
            assert st.q_mask is not None
            not_met = ~st.q_mask
            if np.any(not_met):
                self._scratch_mask[:0] ^= not_met[:0]
            step_now = self._dcc_commit_and_flip(st, mark_committed)
            assert isinstance(step_now, UnitaryStepMeta)
            self._apply_unitary_to_frames(st.op_exec)
            self._broadcast_flips(name, [g[0] for g in step_now.target_groups], not_met)
            return
        if condition_group_fn is None and step_meta.are_groups_disjoint:
            self._dcc_run_disjoint(st, step_meta, cols, mark_committed)
            return
        first_row: NDArray[Any] | None = None
        failed_by_group: list[NDArray[np.intp]] | None = None
        if condition_group_fn is not None and len(step_meta.target_groups) > 0:
            self._in_group_callback = True
            try:
                first_row = self._eval_group_condition(
                    0, st.packed_words_before, [], None, condition_group_fn, cols
                )
                step_now = self._dcc_commit_and_flip(st, mark_committed)
            finally:
                self._in_group_callback = False
        else:
            if condition_group_fn is None:
                assert st.q_mask is not None
                failed_by_group = [
                    np.where(~st.q_mask[g])[0] for g in range(len(step_meta.target_groups))
                ]
            step_now = self._dcc_commit_and_flip(st, mark_committed)
        assert isinstance(step_now, UnitaryStepMeta)
        self._dcc_sequential_py(
            step_now,
            st.packed_words_before,
            st.q_mask,
            condition_group_fn,
            first_row=first_row,
            failed_by_group=failed_by_group,
        )

    def do_conditional_clifford(
        self,
        op: stim.CircuitInstruction,
        condition_met_mask: NDArray[np.bool_] | None = None,
        *,
        condition_group_fn: Any = None,
    ) -> None:
        """Execute conditional Clifford op via Phase 2 and E = U_t^dagger where ~mask.

        A mask of the wrong shape raises ValueError atomically before anything changes.

        Callback contract for ``condition_group_fn(g_idx)``:
            Called sequentially for ``g_idx`` in ``range(len(target_groups))`` with the
            reference tableau pinned before group ``g_idx``; must return a 1-D boolean
            mask of shape ``(batch_size,)``. Allowed: read-only queries (``peek_*``,
            ``measurement_column``, ``active_shots``, ``sync_rngs``) and
            ``postselect_z_batch``. Forbidden (raises ``RuntimeError``): ``run``,
            ``clear``, ``interactive_do``, ``finish_interactive_run``,
            ``do_conditional_clifford``, ``do_conditional_pauli_noise``,
            ``do_unmatched_noisy_reset_z``, ``do_copies_on_fused_step``,
            ``_do_bare_instruction``, ``_next_step_meta``.
        """
        self._reject_if_in_group_callback("do_conditional_clifford")
        mask = _own_mask(condition_met_mask) if condition_met_mask is not None else None
        eff = self._dcc_expected_mask_shapes(op, mask.shape) if mask is not None else None
        cursor_at_check = self._step_cursor
        clear_at_check = self._clear_epoch
        if mask is not None:
            self._check_dcc_mask(op, mask.shape, condition_group_fn is not None, eff)
        step_meta, packed_words_before, steps_len_before, pending = self._dcc_validate_and_step(
            op, eff, condition_group_fn is not None, cursor_at_check, clear_at_check
        )
        clear_at_step = self._clear_epoch
        committed = [False]
        try:
            quant_indices, q_mask, rec_flips = self._dcc_collect_mixed_record(
                op, mask, eff[3] if eff is not None else None
            )
            op_exec = (
                step_meta.quantum_op
                if quant_indices is not None and isinstance(step_meta, UnitaryStepMeta)
                else op
            )
            assert op_exec is not None
            st = _DccState(
                orig_op=op,
                op_exec=op_exec,
                mask=mask,
                eff=eff,
                has_group_fn=condition_group_fn is not None,
                quant_indices=quant_indices,
                q_mask=q_mask,
                rec_flips=rec_flips,
                step_meta=step_meta,
                packed_words_before=packed_words_before,
                steps_len_before=steps_len_before,
                cursor_after_step=self._step_cursor,
                pending=pending,
                clear_at_step=clear_at_step,
            )
            if not isinstance(step_meta, UnitaryStepMeta):
                self._dcc_commit_and_flip(st, lambda: committed.__setitem__(0, True))
                return
            all_met = condition_group_fn is None and (q_mask is None or bool(np.all(q_mask)))
            self._check_op_qubit_bounds(op_exec)
            self._dcc_run_unitary(
                st, all_met, condition_group_fn, lambda: committed.__setitem__(0, True)
            )
        except Exception:
            if pending is None and not committed[0] and self._clear_epoch == clear_at_step:
                self._rollback_dcc_step(step_meta, steps_len_before)
            raise

    def apply_pauli(
        self, pauli: Literal["X", "Y", "Z"], mask: NDArray[np.bool_], p: float = 1.0
    ) -> None:
        """Inject single-qubit Pauli frames across ``mask`` (sampled with probability ``p`` via ``np_rng`` when ``p < 1``)."""
        if p <= 0.0:
            return
        if p < 1.0:
            broadcast_pauli_errors(
                self._flip_simulator, pauli=pauli, mask=mask, p=p, np_rng=self.np_rng
            )
        else:
            self._flip_simulator.broadcast_pauli_errors(pauli=pauli, mask=mask, p=1.0)

    def do_conditional_pauli_noise(
        self,
        op: stim.CircuitInstruction,
        condition_met_mask: NDArray[np.bool_] | None = None,
    ) -> None:
        """Apply a (possibly conditioned) Pauli noise channel directly into stim.FlipSimulator in O(1) time."""
        self._next_step_meta(op)
        name = op.name
        if name in ("I", "II", "I_ERROR", "II_ERROR"):
            return

        args = op.gate_args_copy()
        if name in ("E", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR"):
            p_err = float(args[0]) if len(args) > 0 else 0.0
            raw_targets = op.targets_copy()
            if self._tab_rngs is not None:
                assert self._tab_rngs_x is not None
                self._scratch_x_mask.fill(False)
                self._scratch_z_mask.fill(False)
                any_x, any_z = exec_sync_correlated_error(
                    self._tab_rngs,
                    self._tab_rngs_x,
                    name,
                    raw_targets,
                    p_err,
                    condition_met_mask,
                    self._scratch_x_mask,
                    self._scratch_z_mask,
                )
                self._flush_xz(any_x, any_z)
                return

            if p_err <= 0.0:
                if name in ("E", "CORRELATED_ERROR"):
                    self._correlated_error_occurred.fill(False)
                return
            u = self.np_rng.random(self.batch_size)
            if name in ("E", "CORRELATED_ERROR"):
                occurred = u < p_err
                self._correlated_error_occurred[:] = occurred
            else:
                occurred = (~self._correlated_error_occurred) & (u < p_err)
                self._correlated_error_occurred |= occurred
            if np.any(occurred) and len(raw_targets) > 0:
                self._scratch_x_mask.fill(False)
                self._scratch_z_mask.fill(False)
                any_x = False
                any_z = False
                for t_idx, gt in enumerate(raw_targets):
                    q = int(gt.qubit_value)  # type: ignore[arg-type]
                    hit = (
                        occurred
                        if condition_met_mask is None
                        else (occurred & condition_met_mask[t_idx])
                    )
                    if not np.any(hit):
                        continue
                    if gt.is_x_target or gt.is_y_target:
                        self._scratch_x_mask[q, :] ^= hit
                        any_x = True
                    if gt.is_z_target or gt.is_y_target:
                        self._scratch_z_mask[q, :] ^= hit
                        any_z = True
                self._flush_xz(any_x, any_z)
            return

        if len(args) == 0 or max(args) <= 0.0:
            return
        p_err = args[0]

        targets = [t.qubit_value for t in op.targets_copy()]
        if len(targets) == 0:
            return

        if self._tab_rngs is not None:
            arity = 2 if name in ("DEPOLARIZE2", "PAULI_CHANNEL_2") else 1
            targets_per_shot = conditioned_targets_per_shot(
                targets, arity, self.batch_size, condition_met_mask
            )
            self._sample_pauli_noise_via_tab_rng(name, targets_per_shot, args)
            return

        if condition_met_mask is None or np.all(condition_met_mask):
            self._flip_simulator.do(op)
            return

        if name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
            flips = self.np_rng.random((len(targets), self.batch_size)) < p_err
            flips &= condition_met_mask
            self._broadcast_flips(name[0], targets, flips)
            return

        if name in ("DEPOLARIZE1", "PAULI_CHANNEL_1"):
            u = self.np_rng.random((len(targets), self.batch_size))
            u = np.where(condition_met_mask, u, 1.0)
            if name == "DEPOLARIZE1":
                px = py = pz = p_err / 3.0
            else:
                px, py, pz = args[0], args[1], args[2]
            self._broadcast_flips("X", targets, u < (px + py))
            self._broadcast_flips("Z", targets, (u >= px) & (u < (px + py + pz)))
            return

        if name == "DEPOLARIZE2":
            u = self.np_rng.random((len(targets) // 2, self.batch_size))
            u = np.where(condition_met_mask, u, 1.0)
            bucket = np.floor(u / (p_err / 15.0)).astype(np.int32) + 1
            bucket[u >= p_err] = 0
            x1 = (bucket & 1) != 0
            z1 = (bucket & 2) != 0
            x2 = (bucket & 4) != 0
            z2 = (bucket & 8) != 0
            self._broadcast_pair_flips(targets, x1, z1, x2, z2)
            return

        if name == "PAULI_CHANNEL_2":
            u = self.np_rng.random((len(targets) // 2, self.batch_size))
            u = np.where(condition_met_mask, u, 1.0)
            cdf = np.cumsum(args)
            idx_sampled = np.searchsorted(cdf, u, side="right")
            p1_code = np.where(idx_sampled < 15, (idx_sampled + 1) // 4, 0)
            p2_code = np.where(idx_sampled < 15, (idx_sampled + 1) % 4, 0)
            x1 = (p1_code == 1) | (p1_code == 2)
            z1 = (p1_code == 2) | (p1_code == 3)
            x2 = (p2_code == 1) | (p2_code == 2)
            z2 = (p2_code == 2) | (p2_code == 3)
            self._broadcast_pair_flips(targets, x1, z1, x2, z2)
            return

    ############################################################################
    # Phase 3 & Phase 4: Projective Measurements & Mid-Circuit Resets via P2/P3
    ############################################################################

    def _flush_meas_accum(self) -> None:
        self._flush_xz(self._any_x_c.value, self._any_z_c.value)
        if self._any_x_c.value:
            self._scratch_x_mask.fill(False)
            self._any_x_c.value = 0
        if self._any_z_c.value:
            self._scratch_z_mask.fill(False)
            self._any_z_c.value = 0

    def _build_meas_seg_map(
        self,
        step_meta: MeasStepMeta,
        op_override: stim.CircuitInstruction | None,
        use_cpp: bool,
    ) -> dict[int, int]:
        substeps = step_meta.substeps
        eff_name = op_override.name if op_override is not None else step_meta.op.name
        seg_end_to_len: dict[int, int] = {}
        if (
            use_cpp
            or not step_meta.produces_measurements
            or eff_name not in ("MXX", "MYY", "MZZ", "MPP")
            or len(substeps) <= 1
        ):
            return seg_end_to_len
        cur_len = 0
        seen_qs: set[int] = set()
        for s_idx, sub in enumerate(substeps):
            if sub.meta.q < 0:
                if cur_len > 0:
                    seg_end_to_len[s_idx - 1] = cur_len
                    cur_len = 0
                    seen_qs.clear()
                seg_end_to_len[s_idx] = 1
                continue
            t_qs = {int(sub.meta.q)}
            for pre_op in sub.pre_ops:
                for gt in pre_op.targets_copy():
                    if gt.qubit_value is not None:
                        t_qs.add(int(gt.qubit_value))
            if t_qs & seen_qs:
                seg_end_to_len[s_idx - 1] = cur_len
                cur_len = 1
                seen_qs = set(t_qs)
            else:
                cur_len += 1
                seen_qs |= t_qs
        if cur_len > 0:
            seg_end_to_len[len(substeps) - 1] = cur_len
        return seg_end_to_len

    def _evict_meas_step(
        self, step_meta: MeasStepMeta, xs_cache: NDArray[Any] | None, use_cpp: bool
    ) -> None:
        if use_cpp:
            assert self._cpp_lib is not None and self._cpp_eng is not None
            x2z_w, z2z_w = step_meta.get_evict_words(self._engine_n_words())
            ev_x_c, ev_z_c = ctypes.c_int(0), ctypes.c_int(0)
            xs_ptr = xs_cache.ctypes.data_as(u8_p) if xs_cache is not None else ctypes.cast(0, u8_p)
            self._engine_kernel_begin()
            self._cpp_lib.engine_batch_evict(
                self._cpp_eng,
                x2z_w.ctypes.data_as(u64_p),
                z2z_w.ctypes.data_as(u64_p),
                xs_ptr,
                ctypes.cast(0, u8_p),
                self._scratch_x_mask.ctypes.data_as(u8_p),
                self._scratch_z_mask.ctypes.data_as(u8_p),
                ctypes.byref(ev_x_c),
                ctypes.byref(ev_z_c),
            )
            self._engine_kernel_end()
            self._any_x_c.value |= ev_x_c.value
            self._any_z_c.value |= ev_z_c.value
        elif self._active_shots:
            ev_x, ev_z = self._evict_healed_shots(
                step_meta,
                xs_base=xs_cache,
                accum_x_mask=self._scratch_x_mask,
                accum_z_mask=self._scratch_z_mask,
            )
            self._any_x_c.value |= int(ev_x)
            self._any_z_c.value |= int(ev_z)

    def _execute_meas_step(
        self,
        step_meta: MeasStepMeta,
        op_override: stim.CircuitInstruction | None = None,
    ) -> None:
        """Execute a compiled measurement or reset step across all B shots using P2 and P3."""
        substeps = step_meta.substeps
        det_batch = (
            len(substeps) > 1
            and not any(s.pre_ops or s.post_ops for s in substeps)
            and all(not s.meta.is_random and s.meta.q >= 0 for s in substeps)
            and len({s.meta.q for s in substeps}) == len(substeps)
        )
        use_cpp = bool(
            self._use_cpp and self._cpp_lib is not None and self._cpp_eng is not None
        )
        if use_cpp:
            self._sync_py_to_cpp()
        self._scratch_x_mask.fill(False)
        self._scratch_z_mask.fill(False)
        self._any_x_c.value = 0
        self._any_z_c.value = 0
        if det_batch and use_cpp:
            self._exec_det_batch_cpp(step_meta, op_override)
            return
        seg_end_to_len = self._build_meas_seg_map(step_meta, op_override, use_cpp)
        xs_cache: NDArray[Any] | None = None
        for substep_idx, substep in enumerate(substeps):
            if substep.pre_ops:
                self._flush_meas_accum()
                for pre_op in substep.pre_ops:
                    self._apply_unitary_to_frames(pre_op)
                xs_cache = None
            if xs_cache is None:
                xs_raw, _, _, _, _ = self._flip_simulator.to_numpy(output_xs=True, output_zs=False)
                xs_cache = np.ascontiguousarray(xs_raw, dtype=np.uint8) if use_cpp else xs_raw
            m_shots = (
                self._eval_substep_cpp(substep.meta, xs_cache)
                if use_cpp
                else self._eval_substep_py(substep.meta, xs_cache)
            )
            if step_meta.produces_measurements:
                self._measurement_columns.append(
                    ~m_shots if substep.invert_result else m_shots.copy()
                )
                if substep_idx in seg_end_to_len:
                    self._apply_measurement_flip_noise(
                        step_meta, op_override, seg_end_to_len[substep_idx]
                    )
            if step_meta.is_reset:
                kick_shots = m_shots ^ bool(substep.meta.m_ref)
                if np.any(kick_shots):
                    q = substep.meta.q
                    self._scratch_x_mask[q, :] ^= kick_shots
                    xs_cache[q, :] ^= kick_shots.view(np.uint8) if use_cpp else kick_shots
                    self._any_x_c.value = 1
            if substep.post_ops:
                self._flush_meas_accum()
                for post_op in substep.post_ops:
                    self._apply_unitary_to_frames(post_op)
                xs_cache = None
        self._evict_meas_step(step_meta, xs_cache, use_cpp)
        self._flush_meas_accum()
        if step_meta.produces_measurements and not seg_end_to_len:
            self._apply_measurement_flip_noise(step_meta, op_override, len(substeps))

    def _exec_det_batch_cpp(
        self, step_meta: MeasStepMeta, op_override: stim.CircuitInstruction | None
    ) -> None:
        assert self._cpp_lib is not None and self._cpp_eng is not None
        _, n_b = self._engine_dims()
        n_w = self._engine_n_words()
        num_sub = len(step_meta.substeps)
        q_arr, m_ref_arr, b_sub_words = step_meta.get_det_batch_arrays()
        needed = num_sub * n_b
        if self._cpp_m_buf.size < needed:
            self._cpp_m_buf = np.empty(needed, dtype=np.bool_)
        m_mat = self._cpp_m_buf[:needed].reshape(num_sub, n_b)
        xs_base, _, _, _, _ = self._flip_simulator.to_numpy(
            output_xs=True, output_zs=False
        )
        xs_c = np.ascontiguousarray(xs_base, dtype=np.uint8)
        x2z_w, z2z_w = step_meta.get_evict_words(n_w)
        self._engine_kernel_begin()
        self._cpp_lib.engine_batch_det_meas_step(
            self._cpp_eng,
            num_sub,
            q_arr.ctypes.data_as(i64_p),
            m_ref_arr.ctypes.data_as(u8_p),
            b_sub_words.ctypes.data_as(u64_p),
            xs_c.ctypes.data_as(u8_p),
            m_mat.ctypes.data_as(u8_p),
            x2z_w.ctypes.data_as(u64_p),
            z2z_w.ctypes.data_as(u64_p),
            *self._cpp_out_args(),
        )
        self._engine_kernel_end()
        any_x = int(self._any_x_c.value)
        any_z = int(self._any_z_c.value)
        # Epilogue of a deterministic batched step: record, reset-kick, flush and flip noise.
        if step_meta.produces_measurements:
            for k_idx, substep in enumerate(step_meta.substeps):
                row = m_mat[k_idx]
                self._measurement_columns.append(
                    ~row if substep.invert_result else row.copy()
                )
        if step_meta.is_reset:
            kick_mat = m_mat ^ m_ref_arr[:, None]
            if np.any(kick_mat):
                self._scratch_x_mask[q_arr, :] ^= kick_mat
                any_x = 1
        self._flush_xz(any_x, any_z)
        if step_meta.produces_measurements:
            self._apply_measurement_flip_noise(step_meta, op_override, num_sub)

    def _eval_substep_cpp(
        self, meta: Any, xs_c: NDArray[np.uint8]
    ) -> NDArray[np.bool_]:
        _, n_b = self._engine_dims()
        if meta.q < 0:
            return np.zeros(n_b, dtype=np.bool_)
        assert self._cpp_lib is not None and self._cpp_eng is not None
        self._engine_kernel_begin()
        if not meta.is_random:
            m_shots = np.empty(n_b, dtype=np.bool_)
            null_u64, null_i64 = ctypes.cast(0, u64_p), ctypes.cast(0, i64_p)
            self._cpp_lib.engine_batch_mixed_substep(
                self._cpp_eng, 0, meta.q, int(meta.m_ref), 0,
                meta.beta_Z_words.ctypes.data_as(u64_p),
                null_u64, null_u64, 0, 0, null_i64, 0, null_i64,
                xs_c.ctypes.data_as(u8_p), m_shots.ctypes.data_as(u8_p),
                *self._cpp_out_args(),
            )
        else:
            m_shots = self.np_rng.integers(0, 2, size=n_b, dtype=np.bool_)
            self._cpp_lib.engine_batch_mixed_substep(
                self._cpp_eng, 1, meta.q, int(meta.m_ref), meta.p,
                meta.a_words.ctypes.data_as(u64_p),
                meta.b_rot_words.ctypes.data_as(u64_p),
                meta.e_p_words.ctypes.data_as(u64_p),
                meta.kappa_rot,
                len(meta.G_x_indices), meta.G_x_indices.ctypes.data_as(i64_p),
                len(meta.G_z_indices), meta.G_z_indices.ctypes.data_as(i64_p),
                xs_c.ctypes.data_as(u8_p), m_shots.ctypes.data_as(u8_p),
                *self._cpp_out_args(),
            )
        self._engine_kernel_end()
        return m_shots

    def _eval_substep_py(
        self, meta: Any, xs_base: NDArray[np.bool_]
    ) -> NDArray[np.bool_]:
        if meta.q < 0:
            return np.zeros(self._init_batch_size, dtype=np.bool_)
        f_q_shots = xs_base[meta.q, :].copy()
        m_ref = meta.m_ref
        if not meta.is_random:
            m_shots = f_q_shots ^ bool(m_ref)
            if self._active_shots:
                b_words = meta.beta_Z_words
                for b_idx, mrqf in self._active_shots.items():
                    if mrqf.d == 0 or not _words_any_overlap(b_words, mrqf.a_union):
                        if _word_dot_parity(b_words, mrqf.h_words):
                            m_shots[b_idx] = not m_shots[b_idx]
                    else:
                        sigma = (int(f_q_shots[b_idx]) ^ int(m_ref)) & 1
                        m_val, _ = mrqf.p3_measure_diagonal(
                            b_words, sigma, draw_bit_fn=lambda b=b_idx: int(self._draw_tab_rng_bit(b))
                        )
                        m_shots[b_idx] = bool(m_val)
            return m_shots

        p = meta.p
        p_word, p_bit = p >> 6, p & 63
        p_mask = np.uint64(1 << p_bit)
        m_shots = np.zeros(self._init_batch_size, dtype=np.bool_)
        delta_shots = np.zeros(self._init_batch_size, dtype=np.bool_)
        for b_idx in range(self._init_batch_size):
            mrqf = self._active_shots.get(b_idx)
            if mrqf is not None:
                if not bool(mrqf.a_union[p_word] & p_mask):
                    m_b = self._draw_tab_rng_bit(b_idx)
                    m_shots[b_idx] = m_b
                    h_p = bool((int(mrqf.h_words[p_word]) >> p_bit) & 1)
                    delta_shots[b_idx] = m_b ^ bool(f_q_shots[b_idx]) ^ h_p ^ bool(m_ref)
                else:
                    f_q_b = int(f_q_shots[b_idx])
                    mrqf.p2_apply_quarter_turn(meta.a_words, meta.b_rot_words, meta.kappa_rot)
                    m_eff, _ = mrqf.p3_measure_diagonal(
                        meta.e_p_words, 0, draw_bit_fn=lambda b=b_idx, fq=f_q_b: int(self._draw_tab_rng_bit(b)) ^ fq
                    )
                    m_shots[b_idx] = bool(m_eff ^ f_q_b)
            else:
                m_shots[b_idx] = self._draw_tab_rng_bit(b_idx)
                delta_shots[b_idx] = m_shots[b_idx] ^ f_q_shots[b_idx] ^ bool(m_ref)

        if np.any(delta_shots):
            if len(meta.G_x_indices) > 0:
                self._scratch_x_mask[meta.G_x_indices, :] ^= delta_shots[None, :]
                xs_base[meta.G_x_indices, :] ^= delta_shots[None, :]
                self._any_x_c.value = 1
            if len(meta.G_z_indices) > 0:
                self._scratch_z_mask[meta.G_z_indices, :] ^= delta_shots[None, :]
                self._any_z_c.value = 1
        return m_shots

    def _apply_measurement_flip_noise(
        self,
        step_meta: MeasStepMeta,
        op_override: stim.CircuitInstruction | None,
        num_m: int,
    ) -> None:
        args = (
            op_override.gate_args_copy()
            if op_override is not None
            else step_meta.op.gate_args_copy()
        )
        if len(args) > 0 and args[0] > 0.0:
            p_meas_err = args[0]
            if self._tab_rngs is not None:
                assert self._tab_rngs_x is not None
                apply_sync_meas_flips(
                    self._tab_rngs,
                    self._tab_rngs_x,
                    self._measurement_columns,
                    p_meas_err,
                    num_m,
                )
            else:
                flips_2d = (
                    self.np_rng.random((num_m, self._init_batch_size)) < p_meas_err
                )
                for k_idx in range(num_m):
                    self._measurement_columns[-num_m + k_idx] ^= flips_2d[
                        k_idx
                    ]

    ############################################################################
    # Unmatched Mid-Circuit Noisy Resets (e.g. Leakage Transitions 2 -> 0 / 1)
    ############################################################################

    def _projection_words(self, aw, bw, kap):
        """(a_is_zero, sign_bool, rot) for projecting onto (aw, bw, kap); rot = (b_rot, e_p, kappa_rot) when a != 0."""
        a_is_zero = not _words_any_nonzero(aw)
        sign_bool = int(((kap - int(np.bitwise_count(aw & bw).sum())) & 3) == 2)
        if a_is_zero:
            return a_is_zero, sign_bool, None
        p = _first_set_bit_in_words(aw)
        b_rot = bw.copy()
        b_rot[p >> 6] ^= np.uint64(1 << (p & 63))
        e_p = np.zeros(self._n_words, dtype=np.uint64)
        e_p[p >> 6] = np.uint64(1 << (p & 63))
        return a_is_zero, sign_bool, (b_rot, e_p, (kap + 1) & 3)

    def _project_and_retire(self, mrqf, aw, bw, sign_bool, rot, f_q, m, x2z_w, z2z_w, xs_base, zs_base):
        """Project active MRQF state onto outcome m (P3, or P2/P3/P2 if a != 0) and retire if healed; returns kicks."""
        if rot is None:
            mrqf.p3_measure_diagonal(bw, (f_q ^ sign_bool) & 1, requested_bit=m)
        else:
            b_rot, e_p, k_rot = rot
            mrqf.p2_apply_quarter_turn(aw, b_rot, k_rot)
            mrqf.p3_measure_diagonal(e_p, 0, requested_bit=(m ^ f_q) & 1)
            mrqf.p2_apply_quarter_turn(aw, b_rot, (k_rot + 2) & 3)
        if mrqf.d == 0:
            return self._evict_healed_shots(
                (x2z_w, z2z_w),
                xs_base=xs_base,
                zs_base=zs_base,
                accum_x_mask=self._scratch_x_mask,
                accum_z_mask=self._scratch_z_mask,
            )
        return False, False

    def _reset_z_cpp(
        self,
        reset_zero_mask: NDArray[Any],
        reset_one_mask: NDArray[Any],
        z2x_w: NDArray[np.uint64],
        z2z_w: NDArray[np.uint64],
        z_kappa: NDArray[np.uint8],
        x2z_w: NDArray[np.uint64],
    ) -> None:
        assert self._cpp_lib is not None and self._cpp_eng is not None
        r0_c = np.ascontiguousarray(reset_zero_mask, dtype=np.uint8)
        r1_c = np.ascontiguousarray(reset_one_mask, dtype=np.uint8)
        n_q, n_b = self._engine_dims()
        _check_mask_shape("reset_zero_mask", r0_c.shape, n_q, n_b, "qubit")
        _check_mask_shape("reset_one_mask", r1_c.shape, n_q, n_b, "qubit")
        self._sync_py_to_cpp()
        xs_base, _, _, _, _ = self._flip_simulator.to_numpy(output_xs=True, output_zs=False)
        xs_c = np.ascontiguousarray(xs_base, dtype=np.uint8)
        self._scratch_x_mask.fill(False)
        self._scratch_z_mask.fill(False)
        self._any_x_c.value = 0
        self._any_z_c.value = 0
        self._engine_kernel_begin()
        self._cpp_lib.engine_batch_unmatched_noisy_reset_z(
            self._cpp_eng,
            r0_c.ctypes.data_as(u8_p),
            r1_c.ctypes.data_as(u8_p),
            xs_c.ctypes.data_as(u8_p),
            z2x_w.ctypes.data_as(u64_p),
            z2z_w.ctypes.data_as(u64_p),
            z_kappa.ctypes.data_as(u8_p),
            x2z_w.ctypes.data_as(u64_p),
            *self._cpp_out_args(),
        )
        self._engine_kernel_end()
        self._flush_xz(self._any_x_c.value, self._any_z_c.value)

    def _validate_order_qubits(
        self, order: Sequence[int] | None, n_q: int
    ) -> NDArray[np.int64] | None:
        if order is None:
            return None
        dedup = list(dict.fromkeys(order))
        for q in dedup:
            if isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)):
                raise TypeError(f"order entry {q!r} must be an integer")
            if not (0 <= int(q) < n_q):
                raise IndexError(f"order qubit {q} out of range [0, {n_q})")
        return np.array([int(q) for q in dedup], dtype=np.int64)

    def _project_pauli_cpp(
        self,
        reset_zero_mask: NDArray[Any],
        reset_one_mask: NDArray[Any],
        pauli: Literal["X", "Y", "Z"] = "Z",
        order: Sequence[int] | None = None,
    ) -> None:
        if pauli not in ("X", "Y", "Z"):
            raise ValueError(f"Unrecognised Pauli: {pauli}")
        lib, eng = self._cpp_lib, self._cpp_eng
        assert lib is not None and eng is not None
        n_q, n_b = self._engine_dims()
        n_w = self._engine_n_words()
        r0 = _own_mask(reset_zero_mask)
        r1 = _own_mask(reset_one_mask)
        _check_mask_shape("reset_zero_mask", r0.shape, n_q, n_b, "qubit")
        _check_mask_shape("reset_one_mask", r1.shape, n_q, n_b, "qubit")
        order_arr = self._validate_order_qubits(order, n_q)
        if not (np.any(r0) or np.any(r1)) or (order_arr is not None and len(order_arr) == 0):
            return
        r0_c = np.ascontiguousarray(r0, dtype=np.uint8)
        r1_c = np.ascontiguousarray(r1, dtype=np.uint8)
        self._sync_py_to_cpp()
        x2x_w, x2z_w, z2x_w, z2z_w, x_kappa, z_kappa = self._current_packed_words()
        xs_base, zs_base, _, _, _ = self._flip_simulator.to_numpy(
            output_xs=True, output_zs=(pauli != "Z")
        )
        xs_c = np.ascontiguousarray(xs_base, dtype=np.uint8)
        zs_c = np.ascontiguousarray(zs_base, dtype=np.uint8) if zs_base is not None else None
        self._scratch_x_mask.fill(False)
        self._scratch_z_mask.fill(False)
        self._any_x_c.value = 0
        self._any_z_c.value = 0
        p_kind = 2 if pauli == "Z" else (0 if pauli == "X" else 1)
        n_ord = int(order_arr.shape[0]) if order_arr is not None else -1
        ord_ptr = (
            order_arr.ctypes.data_as(i64_p)
            if order_arr is not None
            else ctypes.cast(0, i64_p)
        )
        zs_ptr = zs_c.ctypes.data_as(u8_p) if zs_c is not None else ctypes.cast(0, u8_p)
        self._engine_kernel_begin()
        rc = int(
            lib.engine_project_pauli(
                eng, p_kind, n_q, n_b, r0_c.ctypes.data_as(u8_p),
                n_q, n_b, r1_c.ctypes.data_as(u8_p), n_ord, ord_ptr,
                n_q, n_b, xs_c.ctypes.data_as(u8_p),
                n_q if zs_c is not None else 0, n_b if zs_c is not None else 0, zs_ptr,
                n_q, n_w, x2x_w.ctypes.data_as(u64_p), x2z_w.ctypes.data_as(u64_p),
                z2x_w.ctypes.data_as(u64_p), z2z_w.ctypes.data_as(u64_p),
                x_kappa.ctypes.data_as(u8_p), z_kappa.ctypes.data_as(u8_p),
                n_q, n_b, self._scratch_x_mask.ctypes.data_as(u8_p),
                n_q, n_b, self._scratch_z_mask.ctypes.data_as(u8_p),
                ctypes.byref(self._any_x_c), ctypes.byref(self._any_z_c),
            )
        )
        self._engine_kernel_end()
        if rc < 0:
            raise ValueError(f"engine_project_pauli failed with code {rc}")
        self._flush_xz(self._any_x_c.value, self._any_z_c.value)

    def _reset_py_qubit(
        self,
        q: int,
        shots_for_q: NDArray[np.intp],
        reset_one_mask: NDArray[Any],
        pauli: Literal["X", "Y", "Z"],
        xs_base: NDArray[np.bool_],
        zs_base: NDArray[np.bool_] | None,
        packed_words: PackedSymplecticTables,
    ) -> tuple[bool, bool]:
        x2x_w, x2z_w, z2x_w, z2z_w, x_kappa, z_kappa = packed_words
        aw, bw, kap = pauli_image(pauli, q, x2x_w, x2z_w, z2x_w, z2z_w, x_kappa, z_kappa)
        a_is_zero, sign_bool, rot = self._projection_words(aw, bw, kap)
        ref_exp = -1 if sign_bool else 1
        any_x_kick = False
        any_z_kick = False
        for b_np in shots_for_q:
            b_idx = int(b_np)
            target_state = 1 if reset_one_mask[q, b_idx] else 0
            if pauli == "Z":
                f_q = int(xs_base[q, b_idx])
            elif pauli == "X":
                assert zs_base is not None
                f_q = int(zs_base[q, b_idx])
            else:
                assert zs_base is not None
                f_q = int(xs_base[q, b_idx] ^ zs_base[q, b_idx])
            mrqf = self._active_shots.get(b_idx)
            if mrqf is not None and (mrqf.d > 0 or _words_any_nonzero(mrqf.h_words)):
                mu = mrqf.q1_pauli_expectation(aw, bw, kap)
                z_exp = -mu if f_q else mu
            else:
                z_exp = (-ref_exp if f_q else ref_exp) if a_is_zero else 0
            if z_exp in (1, -1):
                need_flip = (target_state == 1) if z_exp == 1 else (target_state == 0)
            else:
                m = int(self._draw_tab_rng_bit(b_idx))
                if mrqf is None:
                    mrqf = MRQFState(self._init_num_qubits, self._n_words)
                    self._active_shots[b_idx] = mrqf
                kx, kz = self._project_and_retire(
                    mrqf, aw, bw, sign_bool, rot, f_q, m, x2z_w, z2z_w, xs_base, zs_base
                )
                any_x_kick |= kx
                any_z_kick |= kz
                need_flip = m != target_state
            if need_flip:
                if pauli == "X":
                    self._scratch_z_mask[q, b_idx] ^= True
                    if zs_base is not None:
                        zs_base[q, b_idx] ^= True
                    any_z_kick = True
                else:
                    self._scratch_x_mask[q, b_idx] ^= True
                    xs_base[q, b_idx] ^= True
                    any_x_kick = True
        return any_x_kick, any_z_kick

    def do_unmatched_noisy_reset_z(
        self,
        reset_zero_mask: NDArray[np.bool_],
        reset_one_mask: NDArray[np.bool_],
        pauli: Literal["X", "Y", "Z"] = "Z",
        order: Sequence[int] | None = None,
    ) -> None:
        """Reset noisy qubits to +1 (reset_zero_mask) or -1 (reset_one_mask) eigenstate of
        `pauli` without altering the reference tableau.

        A mask that is not (num_qubits, batch_size) raises ValueError before anything changes.
        """
        self._reject_if_in_group_callback("do_unmatched_noisy_reset_z")
        r0 = _own_mask(reset_zero_mask)
        r1 = _own_mask(reset_one_mask)
        n_q, n_b = self._engine_dims()
        _check_mask_shape("reset_zero_mask", r0.shape, n_q, n_b, "qubit")
        _check_mask_shape("reset_one_mask", r1.shape, n_q, n_b, "qubit")
        if not (np.any(r0) or np.any(r1)):
            return
        packed_words = self._current_packed_words()
        if (
            pauli == "Z"
            and self._use_cpp
            and self._cpp_lib is not None
            and self._cpp_eng is not None
        ):
            _, x2z_w, z2x_w, z2z_w, _, z_kappa = packed_words
            self._reset_z_cpp(r0, r1, z2x_w, z2z_w, z_kappa, x2z_w)
            return
        if self._use_cpp and self._cpp_dirty:
            self._sync_cpp_to_py()
        combined_mask = r0 | r1
        affected_qubits = np.flatnonzero(np.any(combined_mask, axis=1))
        if order is not None:
            affected_qubits = np.array(list(dict.fromkeys(order)), dtype=np.intp)
        if len(affected_qubits) == 0:
            return
        xs_base, zs_base, _, _, _ = self._flip_simulator.to_numpy(
            output_xs=True, output_zs=(pauli != "Z")
        )
        self._scratch_x_mask.fill(False)
        self._scratch_z_mask.fill(False)
        any_x_kick = False
        any_z_kick = False
        any_q_projected = False
        if pauli != "Z":
            self._py_dirty = True
        for q_np in affected_qubits:
            q = int(q_np)
            shots_for_q = np.flatnonzero(combined_mask[q])
            if len(shots_for_q) == 0:
                continue
            any_q_projected = True
            kx, kz = self._reset_py_qubit(
                q, shots_for_q, r1, pauli, xs_base, zs_base, packed_words
            )
            any_x_kick |= kx
            any_z_kick |= kz
        if any_q_projected and self._active_shots:
            kx, kz = self._evict_healed_shots(
                (packed_words[1], packed_words[3]),
                xs_base=xs_base,
                zs_base=zs_base,
                accum_x_mask=self._scratch_x_mask,
                accum_z_mask=self._scratch_z_mask,
            )
            any_x_kick |= kx
            any_z_kick |= kz
        self._flush_xz(any_x_kick, any_z_kick)

    def _postselect_cpp(
        self,
        q: int,
        flip_mask: NDArray[Any],
        desired_bits: NDArray[Any],
        pauli: Literal["X", "Y", "Z"] = "Z",
    ) -> None:
        if pauli not in ("X", "Y", "Z"):
            raise ValueError(f"Unrecognised Pauli: {pauli}")
        lib, eng = self._cpp_lib, self._cpp_eng
        assert lib is not None and eng is not None
        n_q, n_b = self._engine_dims()
        n_w = self._engine_n_words()
        if isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)):
            raise TypeError(f"q must be an integer, got {q!r}")
        qi = int(q)
        if not (0 <= qi < n_q):
            raise ValueError(f"postselect qubit {qi} out of range [0, {n_q})")
        fm = _own_mask(flip_mask)
        db = _own_mask(desired_bits)
        _check_mask_shape_1d("flip_mask", fm.shape, n_b)
        _check_mask_shape_1d("desired_bits", db.shape, n_b)
        fm_c = np.ascontiguousarray(fm, dtype=np.uint8)
        db_c = np.ascontiguousarray(db, dtype=np.uint8)
        if not np.any(fm_c):
            return
        self._sync_py_to_cpp()
        x2x_w, x2z_w, z2x_w, z2z_w, x_kappa, z_kappa = self._current_packed_words()
        xs_base, zs_base, _, _, _ = self._flip_simulator.to_numpy(
            output_xs=True, output_zs=(pauli != "Z")
        )
        xs_c = np.ascontiguousarray(xs_base, dtype=np.uint8)
        zs_c = np.ascontiguousarray(zs_base, dtype=np.uint8) if zs_base is not None else None
        self._scratch_x_mask.fill(False)
        self._scratch_z_mask.fill(False)
        self._any_x_c.value = 0
        self._any_z_c.value = 0
        p_kind = 2 if pauli == "Z" else (0 if pauli == "X" else 1)
        zs_ptr = zs_c.ctypes.data_as(u8_p) if zs_c is not None else ctypes.cast(0, u8_p)
        self._engine_kernel_begin()
        rc = int(
            lib.engine_postselect_batch(
                eng, qi, p_kind, n_b, fm_c.ctypes.data_as(u8_p),
                n_b, db_c.ctypes.data_as(u8_p),
                n_q, n_b, xs_c.ctypes.data_as(u8_p),
                n_q if zs_c is not None else 0, n_b if zs_c is not None else 0, zs_ptr,
                n_q, n_w, x2x_w.ctypes.data_as(u64_p), x2z_w.ctypes.data_as(u64_p),
                z2x_w.ctypes.data_as(u64_p), z2z_w.ctypes.data_as(u64_p),
                x_kappa.ctypes.data_as(u8_p), z_kappa.ctypes.data_as(u8_p),
                n_q, n_b, self._scratch_x_mask.ctypes.data_as(u8_p),
                n_q, n_b, self._scratch_z_mask.ctypes.data_as(u8_p),
                ctypes.byref(self._any_x_c), ctypes.byref(self._any_z_c),
            )
        )
        self._engine_kernel_end()
        if rc < 0:
            raise ValueError(f"engine_postselect_batch failed with code {rc}")
        self._flush_xz(self._any_x_c.value, self._any_z_c.value)

    def postselect_z_batch(
        self,
        q: int,
        flip_mask: NDArray[np.bool_],
        desired_bits: NDArray[np.uint8],
        pauli: Literal["X", "Y", "Z"] = "Z",
    ) -> None:
        """Project Pauli_q (`pauli` in 'X', 'Y', 'Z') onto (-1)^desired_bits[b] for shots where flip_mask[b] is True."""
        shots_for_q = np.flatnonzero(flip_mask)
        if len(shots_for_q) == 0:
            return

        if self._use_cpp and self._cpp_dirty:
            self._sync_cpp_to_py()

        x2x_w, x2z_w, z2x_w, z2z_w, x_kappa, z_kappa = self._current_packed_words()
        xs_base, zs_base, _, _, _ = self._flip_simulator.to_numpy(
            output_xs=True, output_zs=(pauli != "Z")
        )
        self._scratch_x_mask.fill(False)
        self._scratch_z_mask.fill(False)
        any_x_kick = False
        any_z_kick = False

        aw, bw, kap = pauli_image(pauli, q, x2x_w, x2z_w, z2x_w, z2z_w, x_kappa, z_kappa)
        _, sign_bool, rot = self._projection_words(aw, bw, kap)

        # Raised BEFORE the per-shot writes: if one raises midway (short `desired_bits`,
        # KeyboardInterrupt) the earlier shots are already projected in Python and the engine
        # must be re-imported; `_sync_py_to_cpp` trusts this flag.
        self._py_dirty = True
        for b_np in shots_for_q:
            b_idx = int(b_np)
            m = int(desired_bits[b_idx]) & 1
            if pauli == "Z":
                f_q = int(xs_base[q, b_idx])
            elif pauli == "X":
                f_q = int(zs_base[q, b_idx])
            else:
                f_q = int(xs_base[q, b_idx] ^ zs_base[q, b_idx])
            mrqf = self._active_shots.get(b_idx)
            if mrqf is None:
                mrqf = MRQFState(self._init_num_qubits, self._n_words)
                self._active_shots[b_idx] = mrqf
            kx, kz = self._project_and_retire(
                mrqf, aw, bw, sign_bool, rot, f_q, m, x2z_w, z2z_w, xs_base, zs_base
            )
            any_x_kick |= kx
            any_z_kick |= kz

        if self._active_shots:
            kx, kz = self._evict_healed_shots(
                (x2z_w, z2z_w),
                xs_base=xs_base,
                zs_base=zs_base,
                accum_x_mask=self._scratch_x_mask,
                accum_z_mask=self._scratch_z_mask,
            )
            any_x_kick |= kx
            any_z_kick |= kz

        self._flush_xz(any_x_kick, any_z_kick)

    def record_direct_measurements(self, outcomes_per_target: NDArray[np.bool_]) -> None:
        """Append externally sampled measurement outcomes (shape: (num_targets, batch_size)), e.g. for MPAD[LEAKAGE_MEASUREMENT]."""
        for t_idx in range(outcomes_per_target.shape[0]):
            self._measurement_columns.append(
                outcomes_per_target[t_idx].astype(np.bool_)
            )

    ############################################################################
    # Observable Expectation & State Peeking via Q1 (PauliExpectation)
    ############################################################################

    def peek_pauli_born_batch(
        self,
        targets: Sequence[int] | NDArray[np.int_],
        pauli: Literal["X", "Y", "Z"] = "Z",
    ) -> NDArray[np.float64]:
        """Return exact single-qubit Born probabilities ``P(-1)`` of shape ``(len(targets), batch_size)``."""
        return (1.0 - self.peek_pauli_batch(targets, pauli).astype(np.float64)) * 0.5

    def _own_peek_targets(self, targets: Any) -> NDArray[np.int64]:
        n_q, _ = self._engine_dims()
        targets_in = np.asarray(targets)
        if targets_in.ndim != 1 or (
            targets_in.size and not np.issubdtype(targets_in.dtype, np.integer)
        ):
            raise ValueError("peek targets must be a 1-D sequence of integer qubit indices")
        targets_own = np.array(targets_in, order="C", copy=True)
        if targets_own.ndim != 1 or (
            targets_own.size and not np.issubdtype(targets_own.dtype, np.integer)
        ):
            raise ValueError("peek targets must be a 1-D sequence of integer qubit indices")
        if targets_own.size > 0 and (
            int(targets_own.min()) < 0 or int(targets_own.max()) >= n_q
        ):
            raise ValueError(f"peek target out of range [0, {n_q})")
        return np.ascontiguousarray(targets_own, dtype=np.int64)

    def _peek_pauli_batch_py(
        self,
        targets_arr: NDArray[np.int64],
        pauli: Literal["X", "Y", "Z"],
        xs_base: NDArray[np.bool_],
        zs_base: NDArray[np.bool_],
        packed_words: PackedSymplecticTables,
    ) -> NDArray[np.int8]:
        x2x_w, x2z_w, z2x_w, z2z_w, x_kappa, z_kappa = packed_words
        f_comm_mat = (
            xs_base[targets_arr, :]
            if pauli == "Z"
            else zs_base[targets_arr, :]
            if pauli == "X"
            else (xs_base[targets_arr, :] ^ zs_base[targets_arr, :])
        )
        a_words_all, b_words_all, kappas = pauli_image_batch(
            pauli, targets_arr, x2x_w, x2z_w, z2x_w, z2z_w, x_kappa, z_kappa
        )
        a_is_zero = ~np.any(a_words_all != 0, axis=1)
        signs_bool = (kappas & 3) == 2
        ref_exp = np.where(signs_bool, -1, 1).astype(np.int8)
        out = np.where(
            a_is_zero[:, None],
            np.where(f_comm_mat, -ref_exp[:, None], ref_exp[:, None]),
            np.int8(0),
        )
        if self._active_shots:
            for idx_t in range(len(targets_arr)):
                aw = a_words_all[idx_t]
                bw = b_words_all[idx_t]
                kap = int(kappas[idx_t])
                f_c_row = f_comm_mat[idx_t]
                for b_idx, mrqf in self._active_shots.items():
                    mu = mrqf.q1_pauli_expectation(aw, bw, kap)
                    out[idx_t, b_idx] = -mu if f_c_row[b_idx] else mu
        return out

    def peek_pauli_batch(
        self,
        targets: Sequence[int] | NDArray[np.int_],
        pauli: Literal["X", "Y", "Z"] = "Z",
    ) -> NDArray[np.int8]:
        """Return exact single-qubit Pauli expectations (+1, -1, or 0) of shape
        (len(targets), batch_size) via Q1.
        """
        if pauli not in ("X", "Y", "Z"):
            raise ValueError(f"Unrecognised Pauli: {pauli}")
        _, n_b = self._engine_dims()
        targets_arr = self._own_peek_targets(targets)
        n_targets = int(targets_arr.shape[0])
        if n_targets == 0:
            return np.zeros((0, n_b), dtype=np.int8)
        packed_words = self._current_packed_words()
        xs_base, zs_base, _, _, _ = self._flip_simulator.to_numpy(output_xs=True, output_zs=True)
        if self._use_cpp and self._cpp_lib is not None and self._cpp_eng is not None:
            self._sync_py_to_cpp()
            xs_c = np.ascontiguousarray(xs_base, dtype=np.uint8)
            zs_c = np.ascontiguousarray(zs_base, dtype=np.uint8)
            out = np.empty((n_targets, n_b), dtype=np.int8)
            x2x_w, x2z_w, z2x_w, z2z_w, x_kappa, z_kappa = packed_words
            p_kind = 2 if pauli == "Z" else (0 if pauli == "X" else 1)
            self._cpp_lib.engine_peek_pauli_batch(
                self._cpp_eng,
                n_targets,
                targets_arr.ctypes.data_as(i64_p),
                p_kind,
                xs_c.ctypes.data_as(u8_p),
                zs_c.ctypes.data_as(u8_p),
                x2x_w.ctypes.data_as(u64_p),
                x2z_w.ctypes.data_as(u64_p),
                z2x_w.ctypes.data_as(u64_p),
                z2z_w.ctypes.data_as(u64_p),
                x_kappa.ctypes.data_as(u8_p),
                z_kappa.ctypes.data_as(u8_p),
                out.ctypes.data_as(i8_p),
            )
            return out
        return self._peek_pauli_batch_py(targets_arr, pauli, xs_base, zs_base, packed_words)

    def peek_z(self, target: int) -> int | NDArray[np.int8]:
        res = self.peek_pauli_batch([target], "Z")[0]
        if self._init_batch_size == 1 or np.all(res == res[0]):
            return int(res[0])
        return res

    def peek_x(self, target: int) -> int | NDArray[np.int8]:
        res = self.peek_pauli_batch([target], "X")[0]
        if self._init_batch_size == 1 or np.all(res == res[0]):
            return int(res[0])
        return res

    def peek_y(self, target: int) -> int | NDArray[np.int8]:
        res = self.peek_pauli_batch([target], "Y")[0]
        if self._init_batch_size == 1 or np.all(res == res[0]):
            return int(res[0])
        return res

    def get_current_noisy_tableau_pauli_state(
        self, targets: Iterable[int], pauli: Literal["X", "Y", "Z"]
    ) -> list[int] | NDArray[np.int8]:
        targets_list = list(targets)
        batch_res = self.peek_pauli_batch(targets_list, pauli)
        if self._init_batch_size == 1:
            return [int(x) for x in batch_res[:, 0]]
        return batch_res

    ############################################################################
    # Readout Records & Detector/Observable Conversion
    ############################################################################

    def current_tableau_measurement_record(self) -> NDArray[np.bool_]:
        if len(self._measurement_columns) == 0:
            return np.zeros((0,), dtype=np.bool_)
        records = np.column_stack(self._measurement_columns)
        return records[0] if self._init_batch_size == 1 else records

    def get_final_measurement_records(self) -> NDArray[np.bool_]:
        if not self._finished_running_circuit:
            raise RuntimeError("The circuit has not been fully run yet.")
        if self._final_measurement_records is None:
            self._final_measurement_records = stack_measurement_columns(
                self._measurement_columns, self._init_batch_size
            )
        return self._final_measurement_records.copy()
