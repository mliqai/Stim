from __future__ import annotations

import ctypes
from typing import Any, Iterable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]

from stimside._coset.mrqf_state import MRQFState
from stimside._coset.reference_compile import (
    CompiledReferenceCircuit,
    InverseCliffordCocycle,
    MeasStepMeta,
    StepMeta,
    UnitaryStepMeta,
)
from stimside._native.coset_kernels import get_coset_kernels_lib
from stimside.op_handlers.abstract_op_handler import CompiledOpHandler
from stimside.util.bitwords import (
    first_set_bit_in_words as _first_set_bit_in_words,
    i8_p,
    i32_p,
    i64_p,
    u8_p,
    u64_p,
    word_dot_parity as _word_dot_parity,
    words_any_nonzero as _words_any_nonzero,
    words_any_overlap as _words_any_overlap,
)
from stimside.util.circuit import apply_coords_op
from stimside.util.known_states import _replays, _unroll_circuit
from stimside.util.leakage_events import LeakageEvent, LeakageEventsRecorderMixin
from stimside.util.reference_chp import PackedSymplecticTables, _pauli_image
from stimside.util.seeding import derived_seed
from stimside.util.unleaked_to_leaked import UnleakedToLeakedRecordsMixin

# Pauli applied to the qubit of a record-controlled 2Q gate (stim only parses these with rec[] controls).
_REC_FIRST_PAULI = {"CX": "X", "CNOT": "X", "ZCX": "X", "CY": "Y", "ZCY": "Y"}
_REC_SECOND_PAULI = {"XCZ": "X", "YCZ": "Y"}


class CosetsideSimulator(UnleakedToLeakedRecordsMixin, LeakageEventsRecorderMixin):
    """High-performance batched Clifford-error & stabilizer simulator via Moving-Reference Quadratic Forms (MRQF).

    Combines:
    1. Phase 0 Reference CHP Tableau (backed by C++ stim.Tableau),
    2. Batched C++ SIMD baseline Pauli frame tracking via stim.FlipSimulator across all B shots,
    3. Compressed polynomial-space MRQF states D_b = (h, A, ell, Gamma, P) only for active shots with d_b >= 1,
       with O(0) cost on ideal unitary gates and automatic O(1) self-healing eviction back into stim.FlipSimulator
       whenever d_b drops to 0.
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
        self.circuit = circuit
        self.num_qubits = circuit.num_qubits
        self.seed = seed
        self.batch_size = batch_size
        self.record_unleaked_to_leaked = bool(record_unleaked_to_leaked)
        self._unleaked_to_leaked_events: list[list[int]] = (
            [[] for _ in range(batch_size)]
            if self.record_unleaked_to_leaked
            else []
        )
        self.record_leakage_events = bool(record_leakage_events)
        self._leakage_events: list[list[LeakageEvent]] = (
            [[] for _ in range(batch_size)] if self.record_leakage_events else []
        )
        self.sync_tableside_rng = bool(sync_tableside_rng)
        self._n_words = max(1, (self.num_qubits + 63) // 64)
        self._batches_completed = 0
        self._tab_scratch_q = self.num_qubits

        self._shot_np_rngs: list[np.random.Generator] | None = None
        self._tab_rngs: list[stim.TableauSimulator] | None = None
        self._tab_rngs_x: list[stim.TableauSimulator] | None = None
        self._shot_tiebreak_rng: NDArray[np.uint64] = np.zeros(
            self.batch_size, dtype=np.uint64
        )
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
                self.num_qubits,
                self.batch_size,
                self._shot_tiebreak_rng.ctypes.data_as(u64_p),
            )

        self._flip_simulator = stim.FlipSimulator(
            batch_size=self.batch_size,
            disable_stabilizer_randomization=True,
            num_qubits=self.num_qubits,
            seed=self.seed,
        )

        self._compiled_ref = (
            compiled_ref
            if compiled_ref is not None
            else CompiledReferenceCircuit(circuit)
        )
        self._compiled_m2d_converter = circuit.compile_m2d_converter()
        self._compiled_op_handler = compiled_op_handler
        self.compiled_op_handler = compiled_op_handler

        self.qubit_coords: dict[int, list[float]] = {}
        self.qubit_tags: dict[int, str] = {}
        self.coords_shifts: list[float] = []

        self._active_shots: dict[int, MRQFState] = {}
        self._measurement_columns: list[NDArray[np.bool_]] = []
        self._scratch_mask = np.zeros(
            (self.num_qubits, self.batch_size), dtype=np.bool_
        )
        self._scratch_x_mask = np.zeros(
            (self.num_qubits, self.batch_size), dtype=np.bool_
        )
        self._scratch_z_mask = np.zeros(
            (self.num_qubits, self.batch_size), dtype=np.bool_
        )
        self._cpp_m_buf: NDArray[np.bool_] = np.empty(0, dtype=np.bool_)
        self._correlated_error_occurred: NDArray[np.bool_] = np.zeros(
            self.batch_size, dtype=np.bool_
        )

        self._circuit_time = 0
        self._step_cursor = 0
        self._used_interactively = False
        self._unrolled_ops: list[stim.CircuitInstruction] | None = None  # set by interactive_do
        self._finished_running_circuit = False

        self._final_measurement_records: NDArray[np.bool_] | None = None
        self._detector_flips: NDArray[np.bool_] | None = None
        self._observable_flips: NDArray[np.bool_] | None = None

    def __del__(self) -> None:
        if getattr(self, "_cpp_eng", None) is not None and getattr(self, "_cpp_lib", None) is not None:
            try:
                self._cpp_lib.engine_destroy(self._cpp_eng)
            except Exception:
                pass
            self._cpp_eng = None

    @property
    def active_shots(self) -> dict[int, MRQFState]:
        if self._use_cpp:
            if self._cpp_dirty:
                self._sync_cpp_to_py()
            self._py_dirty = True
        return self._active_shots

    @active_shots.setter
    def active_shots(self, val: dict[int, MRQFState]) -> None:
        self._active_shots = val
        if self._use_cpp:
            self._cpp_dirty = False
            self._py_dirty = True

    def _sync_cpp_to_py(self) -> None:
        if not self._cpp_dirty or self._cpp_eng is None or self._cpp_lib is None:
            return
        lib = self._cpp_lib
        out_idx = np.empty(self.batch_size, dtype=np.int32)
        out_d = np.empty(self.batch_size, dtype=np.int32)
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
            st = self._active_shots.get(b)
            if st is None:
                st = MRQFState(
                    self.num_qubits, self._n_words, _init_cap=max(16, d)
                )
                self._active_shots[b] = st
            else:
                st._ensure_capacity(max(1, d))
            st._d = d
            out_P = np.empty(max(1, d), dtype=np.int32)
            out_Gamma = np.empty((max(1, d), max(1, d)), dtype=np.uint8)
            lib.engine_export_shot(
                self._cpp_eng,
                b,
                st.h_words.ctypes.data_as(u64_p),
                st.a_union.ctypes.data_as(u64_p),
                st._A_buf.ctypes.data_as(u64_p),
                st._ell_buf.ctypes.data_as(u8_p),
                out_Gamma.ctypes.data_as(u8_p),
                out_P.ctypes.data_as(i32_p),
            )
            if d > 0:
                st._Gamma_buf[:d, :d] = out_Gamma[:d, :d]
                st.P = [int(x) for x in out_P[:d]]
            else:
                st.P = []
            st._refresh_views()
        self._cpp_dirty = False

    def _sync_py_to_cpp(self, *, keep_py_dirty: bool = False) -> None:
        if self._cpp_eng is None or self._cpp_lib is None:
            return
        if not self._py_dirty and (self._cpp_dirty or len(self._active_shots) == 0):
            return
        lib = self._cpp_lib
        lib.engine_clear(self._cpp_eng)
        for b, st in self._active_shots.items():
            d = st._d
            in_A = np.ascontiguousarray(st._A_buf[: max(1, d)], dtype=np.uint64)
            in_ell = np.ascontiguousarray(st._ell_buf[: max(1, d)], dtype=np.uint8)
            in_Gamma = np.ascontiguousarray(
                st._Gamma_buf[: max(1, d), : max(1, d)], dtype=np.uint8
            )
            in_P = np.ascontiguousarray(
                st.P if d > 0 else [-1], dtype=np.int32
            )
            lib.engine_import_shot(
                self._cpp_eng,
                int(b),
                int(d),
                st.h_words.ctypes.data_as(u64_p),
                st.a_union.ctypes.data_as(u64_p),
                in_A.ctypes.data_as(u64_p),
                in_ell.ctypes.data_as(u8_p),
                in_Gamma.ctypes.data_as(u8_p),
                in_P.ctypes.data_as(i32_p),
            )
        self._py_dirty = keep_py_dirty
        self._cpp_dirty = False

    def _init_rngs(self) -> None:
        if self.sync_tableside_rng:
            if self.seed is None:
                base_seed = int(np.random.default_rng().integers(0, 2**63))
                shot_seeds = [derived_seed(base_seed, b) for b in range(self.batch_size)]
            else:
                # Shot n of the run (counting across batches) is seeded as in TablesideSimulator.
                n0 = self._batches_completed * self.batch_size
                shot_seeds = [derived_seed(self.seed, n0 + b) for b in range(self.batch_size)]
                base_seed = shot_seeds[0]
            self._shot_np_rngs = [
                np.random.default_rng(seed=shot_seeds[b])
                for b in range(self.batch_size)
            ]
            self.np_rng = self._shot_np_rngs[0]
            self._tab_rngs = []
            self._tab_rngs_x = []
            all_data_q = list(range(self.num_qubits))
            for b in range(self.batch_size):
                ts_z = stim.TableauSimulator(seed=shot_seeds[b])
                ts_z.set_num_qubits(self.num_qubits + 1)
                self._tab_rngs.append(ts_z)

                ts_x = stim.TableauSimulator(seed=shot_seeds[b])
                ts_x.set_num_qubits(self.num_qubits + 1)
                if all_data_q:
                    ts_x.h(*all_data_q)
                self._tab_rngs_x.append(ts_x)
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
        b_arange = np.arange(self.batch_size, dtype=np.uint64)
        base_hash = np.uint64(
            ((tie_seed + 1) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        )
        self._shot_tiebreak_rng = np.ascontiguousarray(
            base_hash ^ (b_arange * np.uint64(0xBF58476D1CE4E5B9)),
            dtype=np.uint64,
        )
        if getattr(self, "_cpp_eng", None) is not None and getattr(self, "_cpp_lib", None) is not None:
            self._cpp_lib.engine_set_rng_states(
                self._cpp_eng, self._shot_tiebreak_rng.ctypes.data_as(u64_p)
            )

    def _draw_tab_rng_bit(self, b_idx: int) -> bool:
        if self._tab_rngs is not None:
            assert self._tab_rngs_x is not None
            tr_z = self._tab_rngs[b_idx]
            tr_x = self._tab_rngs_x[b_idx]
            sq = self._tab_scratch_q
            tr_z.h(sq)
            bit = bool(tr_z.measure(sq))
            if bit:
                tr_z.x(sq)
            tr_x.h(sq)
            bit_x = bool(tr_x.measure(sq))
            if bit_x:
                tr_x.x(sq)
            return bit
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
        any_x = False
        any_z = False
        for b_idx in range(self.batch_size):
            t_b = targets_per_shot[b_idx]
            if len(t_b) == 0:
                continue
            inst_b = stim.CircuitInstruction(op_name, t_b, gate_args)
            tr_z = self._tab_rngs[b_idx]
            tr_x = self._tab_rngs_x[b_idx]
            tr_z.do(inst_b)
            tr_x.do(inst_b)
            for q in set(t_b):
                if tr_z.peek_z(q) == -1:
                    self._scratch_x_mask[q, b_idx] ^= True
                    tr_z.x(q)
                    any_x = True
                if tr_x.peek_x(q) == -1:
                    self._scratch_z_mask[q, b_idx] ^= True
                    tr_x.z(q)
                    any_z = True
        self._flush_xz(any_x, any_z)

    ############################################################################
    # Public lifecycle methods
    ############################################################################

    def run(self) -> list[np.ndarray] | None:
        """Run the full circuit across all batch_size shots.

        When `record_unleaked_to_leaked=True`, returns a list of length `batch_size`
        where element `b` is a 1D `np.ndarray` (`dtype=np.int64`) of the unrolled
        circuit operation indices (`op_idx` into `_unroll_circuit(circuit)`) where an
        unleaked-to-leaked transition occurred in shot `b`.
        """
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
                [] for _ in range(self.batch_size)
            ]
        if self.record_leakage_events:
            self._leakage_events = [[] for _ in range(self.batch_size)]

        self._circuit_time = 0
        self._step_cursor = 0
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
        if self._finished_running_circuit:
            raise RuntimeError(
                "Cannot do more operations after finishing an interactive run."
            )
        if self._unrolled_ops is None:
            self._unrolled_ops = _unroll_circuit(self.circuit)
        self._used_interactively = True
        self._do(this)

    def finish_interactive_run(self) -> None:
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

    def _next_step_meta(self, op: stim.CircuitInstruction) -> StepMeta:
        if self._step_cursor < len(self._compiled_ref.steps):
            step = self._compiled_ref.steps[self._step_cursor]
            # interactive_do checked the full op against the circuit; this guards the steps earlier
            # interactive runs appended past its end (kept by clear()). A step only depends on the
            # op's name and targets, so tags (which op handlers remove) and arguments aren't compared.
            if not self._used_interactively or (
                step.op.name == op.name
                and op.targets_copy() == step.op.targets_copy()
            ):
                self._step_cursor += 1
                return step
            # The live reference is at the end of the tape, not at this step: appending here is wrong.
            raise ValueError(
                f"interactive_do: `{op}` doesn't match `{step.op}`, operation {self._step_cursor} of "
                "the reference the CosetsideSimulator compiled from its circuit (plus any operations "
                "earlier interactive runs appended after it). interactive_do can only replay that "
                "reference in order, then continue past its end."
            )
        step = self._compiled_ref.compile_and_append_instruction(op)
        self._step_cursor = len(self._compiled_ref.steps)
        return step

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
                self._check_interactive_op(this)
            self._do_instruction(this)
        else:
            raise NotImplementedError(f"Unsupported circuit object: {type(this)}")

    def _check_interactive_op(self, this: stim.CircuitInstruction) -> None:
        """Raise ValueError unless interactive_do may do `this` now (see interactive_do)."""
        assert self._unrolled_ops is not None
        t, n = self._circuit_time, len(self._unrolled_ops)
        if t < n and not _replays(this, self._unrolled_ops[t]):
            raise ValueError(
                f"interactive_do: operation {t} (REPEAT blocks unrolled) `{this}` doesn't match the "
                "circuit the CosetsideSimulator was constructed with, which has "
                f"`{self._unrolled_ops[t]}` there. Its reference tape, detector/observable converter "
                "and op handler are precomputed from that circuit, so interactive_do must replay it "
                "in order before continuing past its end. Pass the circuit itself or its parts, "
                "not `circuit.flattened()` or `circuit + more`."
            )
        if t >= n and (this.tag or this.name in ("DETECTOR", "OBSERVABLE_INCLUDE")):
            raise ValueError(
                f"interactive_do: operation {t} (REPEAT blocks unrolled) `{this}` is past the end "
                f"of the circuit the CosetsideSimulator was constructed with ({n} operations), "
                "where tagged operations, DETECTOR and OBSERVABLE_INCLUDE aren't supported: the op "
                "handler and detector/observable converter only know that circuit."
            )
        if t >= n and this.name not in ("MPAD", "QUBIT_COORDS"):  # bit values; an annotation
            for target in this.targets_copy():
                q = target.qubit_value  # None for rec, sweep and combiner targets
                if q is not None and q >= self.num_qubits:
                    raise ValueError(
                        f"interactive_do: operation {t} (REPEAT blocks unrolled) `{this}` targets "
                        f"qubit {q}, which the CosetsideSimulator wasn't constructed for (its "
                        f"circuit has num_qubits={self.num_qubits}). Construct it with a circuit "
                        "that uses all qubits."
                    )

    def _do_instruction(self, op: stim.CircuitInstruction) -> None:
        if apply_coords_op(op, self.qubit_coords, self.qubit_tags, self.coords_shifts):
            self._circuit_time += 1
            return

        self._compiled_op_handler.handle_op(op=op, sss=self)
        self._circuit_time += 1

    def _do_bare_instruction(self, op: stim.Circuit | stim.CircuitInstruction) -> None:
        """Execute an unconditional instruction across the reference tape and all B shots."""
        if isinstance(op, stim.Circuit):
            for sub_op in op:
                self._do_bare_instruction(sub_op)
            return

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
                    np.full(self.batch_size, bit_val, dtype=np.bool_)
                )
            if len(raw_targets) > 0:
                self._apply_measurement_flip_noise(
                    step_meta, op, len(raw_targets)
                )
            return

        if name in ("HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"):
            self._next_step_meta(op)
            raw_targets = op.targets_copy()
            ts_targets = [
                gt.qubit_value if gt.qubit_value is not None else gt.value
                for gt in raw_targets
            ]
            inv_flags = [
                bool(gt.is_inverted_result_target) for gt in raw_targets
            ]
            num_m = len(ts_targets)
            if num_m == 0:
                return
            if self._tab_rngs is not None:
                assert self._tab_rngs_x is not None
                herald_cols = [
                    np.zeros(self.batch_size, dtype=np.bool_)
                    for _ in range(num_m)
                ]
                self._scratch_x_mask.fill(False)
                self._scratch_z_mask.fill(False)
                any_x = False
                any_z = False
                for b_idx in range(self.batch_size):
                    tr_z = self._tab_rngs[b_idx]
                    tr_x = self._tab_rngs_x[b_idx]
                    tr_z.do(op)
                    tr_x.do(op)
                    rec = tr_z.current_measurement_record()
                    for k_i in range(num_m):
                        herald_cols[k_i][b_idx] = bool(rec[-num_m + k_i])
                    for q in set(ts_targets):
                        if tr_z.peek_z(q) == -1:
                            self._scratch_x_mask[q, b_idx] ^= True
                            tr_z.x(q)
                            any_x = True
                        if tr_x.peek_x(q) == -1:
                            self._scratch_z_mask[q, b_idx] ^= True
                            tr_x.z(q)
                            any_z = True
                self._flush_xz(any_x, any_z)
                for col in herald_cols:
                    self._measurement_columns.append(col)
            else:
                args = op.gate_args_copy()
                self._scratch_x_mask.fill(False)
                self._scratch_z_mask.fill(False)
                any_x = False
                any_z = False
                if name == "HERALDED_ERASE":
                    p_erase = args[0] if args else 0.0
                    for q, inv in zip(ts_targets, inv_flags):
                        herald_k = self.np_rng.random(self.batch_size) < p_erase
                        pauli_idx = self.np_rng.integers(
                            0, 4, size=self.batch_size
                        )
                        x_k = herald_k & ((pauli_idx == 1) | (pauli_idx == 2))
                        z_k = herald_k & ((pauli_idx == 2) | (pauli_idx == 3))
                        if np.any(x_k):
                            self._scratch_x_mask[q, :] ^= x_k
                            any_x = True
                        if np.any(z_k):
                            self._scratch_z_mask[q, :] ^= z_k
                            any_z = True
                        self._measurement_columns.append(
                            ~herald_k if inv else herald_k
                        )
                else:
                    pi = args[0] if len(args) > 0 else 0.0
                    px = args[1] if len(args) > 1 else 0.0
                    py = args[2] if len(args) > 2 else 0.0
                    pz = args[3] if len(args) > 3 else 0.0
                    c_i = pi
                    c_x = c_i + px
                    c_y = c_x + py
                    c_z = c_y + pz
                    for q, inv in zip(ts_targets, inv_flags):
                        u = self.np_rng.random(self.batch_size)
                        herald_k = u < c_z
                        is_x = (u >= c_i) & (u < c_x)
                        is_y = (u >= c_x) & (u < c_y)
                        is_z = (u >= c_y) & (u < c_z)
                        x_k = is_x | is_y
                        z_k = is_y | is_z
                        if np.any(x_k):
                            self._scratch_x_mask[q, :] ^= x_k
                            any_x = True
                        if np.any(z_k):
                            self._scratch_z_mask[q, :] ^= z_k
                            any_z = True
                        self._measurement_columns.append(
                            ~herald_k if inv else herald_k
                        )
                self._flush_xz(any_x, any_z)
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
            return

        self._next_step_meta(op)

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
            flips = np.stack([f1, f2], axis=1).reshape(-1, self.batch_size)
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
        cocycle: InverseCliffordCocycle,
        f_x_local: NDArray[np.bool_],
        f_z_local: NDArray[np.bool_],
    ) -> None:
        """Inject E = U_t^dagger on shot b_idx via P1, P2, or PCP."""
        mrqf = self._active_shots.get(b_idx)
        if mrqf is None:
            mrqf = MRQFState(self.num_qubits, self._n_words)
            self._active_shots[b_idx] = mrqf

        for factor in cocycle.factors:
            if factor.kind == "P1":
                mrqf.p1_apply_pauli(factor.a_words, factor.b_words)
                continue
            gamma = (
                int(
                    np.count_nonzero(
                        (factor.local_x & f_z_local)
                        ^ (factor.local_z & f_x_local)
                    )
                )
                & 1
            )
            kappa_shot = (factor.kappa + (gamma << 1)) & 3
            if factor.kind == "PCP":
                assert (
                    factor.local_x2 is not None
                    and factor.local_z2 is not None
                    and factor.a2_words is not None
                    and factor.b2_words is not None
                )
                gamma2 = (
                    int(
                        np.count_nonzero(
                            (factor.local_x2 & f_z_local)
                            ^ (factor.local_z2 & f_x_local)
                        )
                    )
                    & 1
                )
                kappa2_shot = (factor.kappa2 + (gamma2 << 1)) & 3
                mrqf.p2_apply_controlled_pauli(
                    factor.a_words,
                    factor.b_words,
                    kappa_shot,
                    factor.a2_words,
                    factor.b2_words,
                    kappa2_shot,
                )
            else:
                mrqf.p2_apply_quarter_turn(
                    factor.a_words, factor.b_words, kappa_shot
                )

    def do_conditional_clifford(
        self,
        op: stim.CircuitInstruction,
        condition_met_mask: NDArray[np.bool_] | None = None,
        *,
        condition_group_fn: Any = None,
    ) -> None:
        """Execute a conditional Clifford op by running Phase 2 and injecting E = U_t^dagger where ~condition_met_mask."""
        packed_words_before = self._current_packed_words()
        step_meta = self._next_step_meta(op)
        name = op.name
        gd = stim.gate_data(name)
        if gd.is_two_qubit_gate:
            raw_t = op.targets_copy()
            if any(t.qubit_value is None for t in raw_t):
                target_groups = op.target_groups()
                quant_indices: list[int] = []
                for g_idx, (t0, t1) in enumerate(target_groups):
                    if t0.qubit_value is not None and t1.qubit_value is not None:
                        quant_indices.append(g_idx)
                        continue
                    rec, tq, paulis = (
                        (t0, t1, _REC_FIRST_PAULI)
                        if t0.is_measurement_record_target
                        else (t1, t0, _REC_SECOND_PAULI)
                    )
                    if rec.is_measurement_record_target and tq.qubit_value is not None:
                        active = self._measurement_columns[rec.value]
                        if condition_met_mask is not None:
                            active = active & condition_met_mask[g_idx]
                        pauli = paulis.get(name, "Z")
                        self._broadcast_flips(pauli, [tq.qubit_value], active[None, :])
                if not isinstance(step_meta, UnitaryStepMeta):
                    return
                assert step_meta.quantum_op is not None
                op = step_meta.quantum_op
                if condition_met_mask is not None:
                    condition_met_mask = condition_met_mask[quant_indices, :]

        if not isinstance(step_meta, UnitaryStepMeta):
            return

        if condition_group_fn is None and (
            condition_met_mask is None or np.all(condition_met_mask)
        ):
            self._apply_unitary_to_frames(op)
            return

        if name in ("I", "II", "I_ERROR", "II_ERROR"):
            if condition_group_fn is not None:
                self._override_packed_words = packed_words_before
                try:
                    for g_idx in range(len(step_meta.target_groups)):
                        condition_group_fn(g_idx)
                finally:
                    self._override_packed_words = None
            return

        if condition_group_fn is None and name in ("X", "Y", "Z"):
            self._apply_unitary_to_frames(op)
            qubits = [group[0] for group in step_meta.target_groups]
            self._broadcast_flips(name, qubits, ~condition_met_mask)
            return

        if condition_group_fn is None and step_meta.are_groups_disjoint:
            if self._use_cpp and self._cpp_lib is not None and self._cpp_eng is not None:
                self._sync_py_to_cpp()
                self._apply_unitary_to_frames(op)
                lib = self._cpp_lib
                if name in ("CZ", "ZCZ"):
                    if step_meta._cpp_cz_qs is None:
                        step_meta._cpp_cz_qs = (
                            np.array(
                                [g[0] for g in step_meta.target_groups], dtype=np.int64
                            ),
                            np.array(
                                [g[1] for g in step_meta.target_groups], dtype=np.int64
                            ),
                        )
                    q0_arr, q1_arr = step_meta._cpp_cz_qs
                    xs_base, _, _, _, _ = self._flip_simulator.to_numpy(
                        output_xs=True, output_zs=False
                    )
                    _, x2z_w, z2x_w, z2z_w, _, z_kappa = step_meta._get_packed_words(
                        self._n_words
                    )
                    self._scratch_x_mask.fill(False)
                    self._scratch_z_mask.fill(False)
                    cm_c = np.ascontiguousarray(condition_met_mask, dtype=np.uint8)
                    xs_c = np.ascontiguousarray(xs_base, dtype=np.uint8)
                    lib.engine_batch_inject_cz(
                        self._cpp_eng,
                        len(step_meta.target_groups),
                        q0_arr.ctypes.data_as(i64_p),
                        q1_arr.ctypes.data_as(i64_p),
                        cm_c.ctypes.data_as(u8_p),
                        xs_c.ctypes.data_as(u8_p),
                        z2x_w.ctypes.data_as(u64_p),
                        z2z_w.ctypes.data_as(u64_p),
                        z_kappa.ctypes.data_as(u8_p),
                        x2z_w.ctypes.data_as(u64_p),
                        *self._cpp_out_args(),
                    )
                    self._cpp_dirty = True
                    self._flush_xz(self._any_x_c.value, self._any_z_c.value)
                    return

                if step_meta._cpp_cocycle_arrays is None:
                    num_g = len(step_meta.target_groups)
                    arity = len(step_meta.target_groups[0])
                    cocycles = [
                        step_meta.get_inverse_cocycle(
                            g, self.num_qubits, self._n_words
                        )
                        for g in range(num_g)
                    ]
                    n_fac = len(cocycles[0].factors)
                    t_groups = np.array(step_meta.target_groups, dtype=np.int64)
                    g_f = (num_g, n_fac)
                    shapes = [g_f, (*g_f, arity), (*g_f, self._n_words)]
                    # C-call order: kinds, then (lx, lz, a, b, kappa) for factor halves 1 and 2.
                    f_arrs = [
                        np.zeros(shapes[s], dtype=np.uint64 if s == 2 else np.uint8)
                        for s in (0, 1, 1, 2, 2, 0, 1, 1, 2, 2, 0)
                    ]
                    f_kinds, f_lx1, f_lz1, f_a1, f_b1, f_k1 = f_arrs[:6]
                    f_lx2, f_lz2, f_a2, f_b2, f_k2 = f_arrs[6:]
                    for g_i, coc in enumerate(cocycles):
                        for f_i, fac in enumerate(coc.factors):
                            f_kinds[g_i, f_i] = (
                                0
                                if fac.kind == "P1"
                                else (1 if fac.kind == "P2" else 2)
                            )
                            f_lx1[g_i, f_i] = fac.local_x
                            f_lz1[g_i, f_i] = fac.local_z
                            f_a1[g_i, f_i] = fac.a_words
                            f_b1[g_i, f_i] = fac.b_words
                            f_k1[g_i, f_i] = fac.kappa
                            if fac.kind == "PCP":
                                f_lx2[g_i, f_i] = fac.local_x2
                                f_lz2[g_i, f_i] = fac.local_z2
                                f_a2[g_i, f_i] = fac.a2_words
                                f_b2[g_i, f_i] = fac.b2_words
                                f_k2[g_i, f_i] = fac.kappa2
                    step_meta._cpp_cocycle_arrays = (
                        num_g, arity, n_fac, t_groups, f_arrs
                    )
                num_g, arity, n_fac, t_groups, f_arrs = step_meta._cpp_cocycle_arrays
                xs_base, zs_base, _, _, _ = self._flip_simulator.to_numpy(
                    output_xs=True, output_zs=True
                )
                x2z_w, z2z_w = step_meta.get_evict_words(self._n_words)
                self._scratch_x_mask.fill(False)
                self._scratch_z_mask.fill(False)
                cm_c = np.ascontiguousarray(condition_met_mask, dtype=np.uint8)
                xs_c = np.ascontiguousarray(xs_base, dtype=np.uint8)
                zs_c = np.ascontiguousarray(zs_base, dtype=np.uint8)
                lib.engine_batch_inject_general_clifford(
                    self._cpp_eng,
                    num_g,
                    arity,
                    n_fac,
                    t_groups.ctypes.data_as(i64_p),
                    cm_c.ctypes.data_as(u8_p),
                    xs_c.ctypes.data_as(u8_p),
                    zs_c.ctypes.data_as(u8_p),
                    *(
                        a.ctypes.data_as(u64_p if a.dtype == np.uint64 else u8_p)
                        for a in f_arrs
                    ),
                    x2z_w.ctypes.data_as(u64_p),
                    z2z_w.ctypes.data_as(u64_p),
                    *self._cpp_out_args(),
                )
                self._cpp_dirty = True
                self._flush_xz(self._any_x_c.value, self._any_z_c.value)
                return

            self._apply_unitary_to_frames(op)
            if name in ("CZ", "ZCZ"):
                xs_base, _, _, _, _ = self._flip_simulator.to_numpy(
                    output_xs=True, output_zs=False
                )
                _, _, z2x_w, z2z_w, _, z_kappa = step_meta._get_packed_words(
                    self._n_words
                )
                failed_g_indices = np.flatnonzero(
                    ~np.all(condition_met_mask, axis=1)
                )
                for g_idx_np in failed_g_indices:
                    g_idx = int(g_idx_np)
                    q0, q1 = step_meta.target_groups[g_idx]
                    aw1, bw1, kap1_base = (
                        z2x_w[q0],
                        z2z_w[q0],
                        int(z_kappa[q0]),
                    )
                    aw2, bw2, kap2_base = (
                        z2x_w[q1],
                        z2z_w[q1],
                        int(z_kappa[q1]),
                    )
                    xs_q0 = xs_base[q0]
                    xs_q1 = xs_base[q1]
                    for b_idx_np in np.flatnonzero(~condition_met_mask[g_idx]):
                        b_idx = int(b_idx_np)
                        mrqf = self._active_shots.get(b_idx)
                        if mrqf is None:
                            mrqf = MRQFState(self.num_qubits, self._n_words)
                            self._active_shots[b_idx] = mrqf
                        kap1 = (kap1_base + (int(xs_q0[b_idx]) << 1)) & 3
                        kap2 = (kap2_base + (int(xs_q1[b_idx]) << 1)) & 3
                        mrqf.p2_apply_controlled_pauli(
                            aw1, bw1, kap1, aw2, bw2, kap2
                        )
                if self._active_shots and any(
                    m.d == 0 for m in self._active_shots.values()
                ):
                    self._evict_healed_shots(
                        step_meta,
                        xs_base=xs_base,
                    )
                return

            xs_base, zs_base, _, _, _ = self._flip_simulator.to_numpy(
                output_xs=True, output_zs=True
            )
            failed_g_indices = np.flatnonzero(~np.all(condition_met_mask, axis=1))
            for g_idx_np in failed_g_indices:
                g_idx = int(g_idx_np)
                group = step_meta.target_groups[g_idx]
                failed_shots = np.flatnonzero(~condition_met_mask[g_idx])
                cocycle = step_meta.get_inverse_cocycle(
                    g_idx, self.num_qubits, self._n_words
                )
                group_arr = np.array(group, dtype=np.intp)
                for b_idx_np in failed_shots:
                    b_idx = int(b_idx_np)
                    f_x_local = xs_base[group_arr, b_idx]
                    f_z_local = zs_base[group_arr, b_idx]
                    self._inject_inverse_cocycle_on_shot(
                        b_idx, cocycle, f_x_local, f_z_local
                    )
            if self._active_shots and any(
                m.d == 0 for m in self._active_shots.values()
            ):
                self._evict_healed_shots(
                    step_meta,
                    xs_base=xs_base,
                    zs_base=zs_base,
                )
            return

        if self._use_cpp and self._cpp_dirty:
            self._sync_cpp_to_py()
        group_pws = step_meta.ensure_group_packed_words(packed_words_before)
        for g_idx, group in enumerate(step_meta.target_groups):
            if condition_group_fn is not None:
                self._override_packed_words = (
                    packed_words_before if g_idx == 0 else group_pws[g_idx - 1]
                )
                try:
                    cm_row = condition_group_fn(g_idx)
                finally:
                    self._override_packed_words = None
            else:
                cm_row = condition_met_mask[g_idx]
            self._apply_unitary_to_frames(step_meta.group_ops[g_idx])
            failed_shots = np.where(~cm_row)[0]
            if len(failed_shots) == 0:
                continue
            cocycle = step_meta.get_inverse_cocycle(
                g_idx, self.num_qubits, self._n_words
            )
            xs_base, zs_base, _, _, _ = self._flip_simulator.to_numpy(
                output_xs=True, output_zs=True
            )
            group_arr = np.array(group, dtype=np.intp)
            for b_idx_np in failed_shots:
                b_idx = int(b_idx_np)
                f_x_local = xs_base[group_arr, b_idx]
                f_z_local = zs_base[group_arr, b_idx]
                self._inject_inverse_cocycle_on_shot(
                    b_idx, cocycle, f_x_local, f_z_local
                )
            if self._active_shots:
                self._evict_healed_shots(
                    (group_pws[g_idx][1], group_pws[g_idx][3]),
                    xs_base=xs_base,
                    zs_base=zs_base,
                )
            if self._use_cpp:
                self._py_dirty = True
        if self._use_cpp:
            self._py_dirty = True

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
                any_x = False
                any_z = False
                for b_idx in range(self.batch_size):
                    t_b = (
                        raw_targets
                        if condition_met_mask is None
                        else [
                            raw_targets[i]
                            for i in range(len(raw_targets))
                            if condition_met_mask[i, b_idx]
                        ]
                    )
                    inst_b = stim.CircuitInstruction(name, t_b, [p_err])
                    tr_z = self._tab_rngs[b_idx]
                    tr_x = self._tab_rngs_x[b_idx]
                    tr_z.do(inst_b)
                    tr_x.do(inst_b)
                    for q in {
                        gt.qubit_value
                        for gt in t_b
                        if gt.qubit_value is not None
                    }:
                        if tr_z.peek_z(q) == -1:
                            self._scratch_x_mask[q, b_idx] ^= True
                            tr_z.x(q)
                            any_x = True
                        if tr_x.peek_x(q) == -1:
                            self._scratch_z_mask[q, b_idx] ^= True
                            tr_x.z(q)
                            any_z = True
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
                    q = gt.qubit_value
                    if q is None:
                        continue
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
            is_two_qubit = name in ("DEPOLARIZE2", "PAULI_CHANNEL_2")
            if condition_met_mask is None:
                targets_per_shot: list[list[int]] = [
                    targets for _ in range(self.batch_size)
                ]
            elif not is_two_qubit:
                targets_per_shot = [
                    [
                        targets[i]
                        for i in range(len(targets))
                        if condition_met_mask[i, b]
                    ]
                    for b in range(self.batch_size)
                ]
            else:
                pairs = [
                    (targets[i], targets[i + 1])
                    for i in range(0, len(targets), 2)
                ]
                targets_per_shot = []
                for b in range(self.batch_size):
                    flat_b: list[int] = []
                    for g_idx, (q1, q2) in enumerate(pairs):
                        if condition_met_mask[g_idx, b]:
                            flat_b.append(q1)
                            flat_b.append(q2)
                    targets_per_shot.append(flat_b)
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
            and all(not s.meta.is_random for s in substeps)
            and len({s.meta.q for s in substeps}) == len(substeps)
        )

        if self._use_cpp and self._cpp_lib is not None and self._cpp_eng is not None:
            self._sync_py_to_cpp()
            lib = self._cpp_lib

            if det_batch:
                num_sub = len(substeps)
                q_arr, m_ref_arr, b_sub_words = step_meta.get_det_batch_arrays()
                needed = num_sub * self.batch_size
                if self._cpp_m_buf.size < needed:
                    self._cpp_m_buf = np.empty(needed, dtype=np.bool_)
                m_mat = self._cpp_m_buf[:needed].reshape(num_sub, self.batch_size)

                xs_base, _, _, _, _ = self._flip_simulator.to_numpy(
                    output_xs=True, output_zs=False
                )
                xs_c = np.ascontiguousarray(xs_base, dtype=np.uint8)
                x2z_w, z2z_w = step_meta.get_evict_words(self._n_words)
                self._scratch_x_mask.fill(False)
                self._scratch_z_mask.fill(False)

                lib.engine_batch_det_meas_step(
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
                self._cpp_dirty = True
                any_x_accum = bool(self._any_x_c.value)
                any_z_accum = bool(self._any_z_c.value)
                self._finish_det_batch(
                    step_meta, op_override, m_mat, any_x_accum, any_z_accum
                )
                return

            xs_c: NDArray[np.uint8] | None = None
            self._scratch_x_mask.fill(False)
            self._scratch_z_mask.fill(False)
            self._any_x_c.value = 0
            self._any_z_c.value = 0

            def _flush_cpp_accum() -> None:
                self._flush_xz(self._any_x_c.value, self._any_z_c.value)
                if self._any_x_c.value:
                    self._scratch_x_mask.fill(False)
                    self._any_x_c.value = 0
                if self._any_z_c.value:
                    self._scratch_z_mask.fill(False)
                    self._any_z_c.value = 0

            null_u64 = ctypes.cast(0, u64_p)
            null_i64 = ctypes.cast(0, i64_p)

            for substep in substeps:
                if substep.pre_ops:
                    _flush_cpp_accum()
                    for pre_op in substep.pre_ops:
                        self._apply_unitary_to_frames(pre_op)
                    xs_c = None
                if xs_c is None:
                    xs_base_tmp, _, _, _, _ = self._flip_simulator.to_numpy(
                        output_xs=True, output_zs=False
                    )
                    xs_c = np.ascontiguousarray(xs_base_tmp, dtype=np.uint8)

                meta = substep.meta
                q = meta.q
                m_ref = int(meta.m_ref)
                if not meta.is_random:
                    m_shots = np.empty(self.batch_size, dtype=np.bool_)
                    lib.engine_batch_mixed_substep(
                        self._cpp_eng,
                        0,
                        q,
                        m_ref,
                        0,
                        meta.beta_Z_words.ctypes.data_as(u64_p),
                        null_u64,
                        null_u64,
                        0,
                        0,
                        null_i64,
                        0,
                        null_i64,
                        xs_c.ctypes.data_as(u8_p),
                        m_shots.ctypes.data_as(u8_p),
                        *self._cpp_out_args(),
                    )
                else:
                    m_shots = self.np_rng.integers(
                        0, 2, size=self.batch_size, dtype=np.bool_
                    )
                    lib.engine_batch_mixed_substep(
                        self._cpp_eng,
                        1,
                        q,
                        m_ref,
                        meta.p,
                        meta.a_words.ctypes.data_as(u64_p),
                        meta.b_rot_words.ctypes.data_as(u64_p),
                        meta.e_p_words.ctypes.data_as(u64_p),
                        meta.kappa_rot,
                        len(meta.G_x_indices),
                        meta.G_x_indices.ctypes.data_as(i64_p),
                        len(meta.G_z_indices),
                        meta.G_z_indices.ctypes.data_as(i64_p),
                        xs_c.ctypes.data_as(u8_p),
                        m_shots.ctypes.data_as(u8_p),
                        *self._cpp_out_args(),
                    )
                self._cpp_dirty = True

                if step_meta.produces_measurements:
                    self._measurement_columns.append(
                        ~m_shots if substep.invert_result else m_shots.copy()
                    )

                if step_meta.is_reset:
                    kick_shots = m_shots ^ bool(m_ref)
                    if np.any(kick_shots):
                        self._scratch_x_mask[q, :] ^= kick_shots
                        xs_c[q, :] ^= kick_shots.view(np.uint8)
                        self._any_x_c.value = 1

                if substep.post_ops:
                    _flush_cpp_accum()
                    for post_op in substep.post_ops:
                        self._apply_unitary_to_frames(post_op)
                    xs_c = None

            x2z_w, z2z_w = step_meta.get_evict_words(self._n_words)
            ev_x_c = ctypes.c_int(0)
            ev_z_c = ctypes.c_int(0)
            lib.engine_batch_evict(
                self._cpp_eng,
                x2z_w.ctypes.data_as(u64_p),
                z2z_w.ctypes.data_as(u64_p),
                xs_c.ctypes.data_as(u8_p)
                if xs_c is not None
                else ctypes.cast(0, u8_p),
                ctypes.cast(0, u8_p),
                self._scratch_x_mask.ctypes.data_as(u8_p),
                self._scratch_z_mask.ctypes.data_as(u8_p),
                ctypes.byref(ev_x_c),
                ctypes.byref(ev_z_c),
            )
            if ev_x_c.value:
                self._any_x_c.value = 1
            if ev_z_c.value:
                self._any_z_c.value = 1
            _flush_cpp_accum()
            if step_meta.produces_measurements:
                self._apply_measurement_flip_noise(
                    step_meta, op_override, len(substeps)
                )
            return

        xs_base: NDArray[np.bool_] | None = None
        self._scratch_x_mask.fill(False)
        self._scratch_z_mask.fill(False)
        any_x_accum = False
        any_z_accum = False

        def _flush_accum() -> None:
            nonlocal any_x_accum, any_z_accum
            self._flush_xz(any_x_accum, any_z_accum)
            if any_x_accum:
                self._scratch_x_mask.fill(False)
            if any_z_accum:
                self._scratch_z_mask.fill(False)
            any_x_accum = any_z_accum = False

        if det_batch:
            q_arr, m_ref_arr, b_sub_words = step_meta.get_det_batch_arrays()
            xs_base, _, _, _, _ = self._flip_simulator.to_numpy(
                output_xs=True, output_zs=False
            )
            m_mat = xs_base[q_arr, :] ^ m_ref_arr[:, None]

            if self._active_shots:
                active_indices = list(self._active_shots.keys())
                active_mrqfs = [self._active_shots[b] for b in active_indices]
                h_mat = np.stack([m.h_words for m in active_mrqfs], axis=0)
                au_mat = np.stack([m.a_union for m in active_mrqfs], axis=0)
                xor_bh = b_sub_words[:, None, 0] & h_mat[None, :, 0]
                ov_words = b_sub_words[:, None, 0] & au_mat[None, :, 0]
                for w in range(1, self._n_words):
                    xor_bh ^= b_sub_words[:, None, w] & h_mat[None, :, w]
                    ov_words |= b_sub_words[:, None, w] & au_mat[None, :, w]
                bh_mat = (np.bitwise_count(xor_bh) & 1).astype(np.bool_)
                m_mat[:, active_indices] ^= bh_mat
                overlap_mat = ov_words != 0
                if np.any(overlap_mat):
                    for a_idx_np in np.flatnonzero(np.any(overlap_mat, axis=0)):
                        a_idx = int(a_idx_np)
                        b_idx = active_indices[a_idx]
                        mrqf = active_mrqfs[a_idx]
                        if mrqf.d == 0:
                            continue
                        h_init = mrqf.h_words.copy()
                        for k_np in np.flatnonzero(overlap_mat[:, a_idx]):
                            k_idx = int(k_np)
                            b_k = b_sub_words[k_idx]
                            if mrqf.d == 0 or not _words_any_overlap(
                                b_k, mrqf.a_union
                            ):
                                if _word_dot_parity(
                                    b_k, mrqf.h_words ^ h_init
                                ):
                                    m_mat[k_idx, b_idx] = not m_mat[
                                        k_idx, b_idx
                                    ]
                                continue
                            q_k = int(q_arr[k_idx])
                            m_ref_k = int(m_ref_arr[k_idx])
                            f_q = int(xs_base[q_k, b_idx])
                            sigma = (f_q ^ m_ref_k) & 1
                            m_val, _ = mrqf.p3_measure_diagonal(
                                b_k,
                                sigma,
                                draw_bit_fn=lambda b=b_idx: int(
                                    self._draw_tab_rng_bit(b)
                                ),
                            )
                            m_mat[k_idx, b_idx] = bool(m_val)

                ev_x, ev_z = self._evict_healed_shots(
                    step_meta,
                    xs_base=xs_base,
                    accum_x_mask=self._scratch_x_mask,
                    accum_z_mask=self._scratch_z_mask,
                )
                any_x_accum |= ev_x
                any_z_accum |= ev_z

            self._finish_det_batch(
                step_meta, op_override, m_mat, any_x_accum, any_z_accum
            )
            return

        assert self._tab_rngs is not None
        eff_op_name = (
            op_override.name if op_override is not None else step_meta.op.name
        )
        seg_end_to_len: dict[int, int] = {}
        if (
            step_meta.produces_measurements
            and eff_op_name in ("MXX", "MYY", "MZZ", "MPP")
            and len(substeps) > 1
        ):
            cur_len = 0
            seen_qs: set[int] = set()
            for s_idx, sub in enumerate(substeps):
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

        for substep_idx, substep in enumerate(substeps):
            if substep.pre_ops:
                _flush_accum()
                for pre_op in substep.pre_ops:
                    self._apply_unitary_to_frames(pre_op)
                xs_base = None

            meta = substep.meta
            q = meta.q
            m_ref = meta.m_ref
            if xs_base is None:
                xs_base, _, _, _, _ = self._flip_simulator.to_numpy(
                    output_xs=True, output_zs=False
                )
            f_q_shots = xs_base[q, :].copy()

            if not meta.is_random:
                m_shots = f_q_shots ^ bool(m_ref)
                if self._active_shots:
                    b_words = meta.beta_Z_words
                    for b_idx, mrqf in self._active_shots.items():
                        if mrqf.d == 0 or not _words_any_overlap(
                            b_words, mrqf.a_union
                        ):
                            if _word_dot_parity(b_words, mrqf.h_words):
                                m_shots[b_idx] = not m_shots[b_idx]
                        else:
                            sigma = (int(f_q_shots[b_idx]) ^ int(m_ref)) & 1
                            m_val, _ = mrqf.p3_measure_diagonal(
                                b_words,
                                sigma,
                                draw_bit_fn=lambda b=b_idx: int(
                                    self._draw_tab_rng_bit(b)
                                ),
                            )
                            m_shots[b_idx] = bool(m_val)
            else:
                p = meta.p
                p_word = p >> 6
                p_bit = p & 63
                p_mask = np.uint64(1 << p_bit)
                m_shots = np.zeros(self.batch_size, dtype=np.bool_)
                delta_shots = np.zeros(self.batch_size, dtype=np.bool_)
                for b_idx in range(self.batch_size):
                    mrqf = self._active_shots.get(b_idx)
                    if mrqf is not None:
                        if not bool(mrqf.a_union[p_word] & p_mask):
                            m_b = self._draw_tab_rng_bit(b_idx)
                            m_shots[b_idx] = m_b
                            h_p = bool(
                                (int(mrqf.h_words[p_word]) >> p_bit) & 1
                            )
                            delta_shots[b_idx] = (
                                m_b
                                ^ bool(f_q_shots[b_idx])
                                ^ h_p
                                ^ bool(m_ref)
                            )
                        else:
                            f_q_b = int(f_q_shots[b_idx])
                            mrqf.p2_apply_quarter_turn(
                                meta.a_words,
                                meta.b_rot_words,
                                meta.kappa_rot,
                            )
                            m_eff, _ = mrqf.p3_measure_diagonal(
                                meta.e_p_words,
                                0,
                                draw_bit_fn=lambda b=b_idx, fq=f_q_b: int(
                                    self._draw_tab_rng_bit(b)
                                )
                                ^ fq,
                            )
                            m_shots[b_idx] = bool(m_eff ^ f_q_b)
                            delta_shots[b_idx] = False
                    else:
                        m_shots[b_idx] = self._draw_tab_rng_bit(b_idx)
                        delta_shots[b_idx] = (
                            m_shots[b_idx] ^ f_q_shots[b_idx] ^ bool(m_ref)
                        )

                if np.any(delta_shots):
                    if len(meta.G_x_indices) > 0:
                        self._scratch_x_mask[meta.G_x_indices, :] ^= (
                            delta_shots[None, :]
                        )
                        xs_base[meta.G_x_indices, :] ^= delta_shots[None, :]
                        any_x_accum = True
                    if len(meta.G_z_indices) > 0:
                        self._scratch_z_mask[meta.G_z_indices, :] ^= (
                            delta_shots[None, :]
                        )
                        any_z_accum = True

            if step_meta.produces_measurements:
                recorded_shots = (
                    ~m_shots if substep.invert_result else m_shots.copy()
                )
                self._measurement_columns.append(recorded_shots)
                if substep_idx in seg_end_to_len:
                    self._apply_measurement_flip_noise(
                        step_meta, op_override, seg_end_to_len[substep_idx]
                    )

            if step_meta.is_reset:
                kick_shots = m_shots ^ bool(m_ref)
                if np.any(kick_shots):
                    self._scratch_x_mask[q, :] ^= kick_shots
                    xs_base[q, :] ^= kick_shots
                    any_x_accum = True

            if substep.post_ops:
                _flush_accum()
                for post_op in substep.post_ops:
                    self._apply_unitary_to_frames(post_op)
                xs_base = None

        if self._active_shots:
            ev_x, ev_z = self._evict_healed_shots(
                step_meta,
                xs_base=xs_base,
                accum_x_mask=self._scratch_x_mask,
                accum_z_mask=self._scratch_z_mask,
            )
            any_x_accum |= ev_x
            any_z_accum |= ev_z

        _flush_accum()

        if step_meta.produces_measurements and not seg_end_to_len:
            self._apply_measurement_flip_noise(
                step_meta, op_override, len(step_meta.substeps)
            )

    def _finish_det_batch(
        self,
        step_meta: MeasStepMeta,
        op_override: stim.CircuitInstruction | None,
        m_mat: NDArray[np.bool_],
        any_x: int,
        any_z: int,
    ) -> None:
        """Record, reset-kick, flush and flip-noise epilogue of a deterministic batched step (C++ and Python)."""
        q_arr, m_ref_arr, _ = step_meta.get_det_batch_arrays()
        if step_meta.produces_measurements:
            for k_idx, substep in enumerate(step_meta.substeps):
                row = m_mat[k_idx]
                self._measurement_columns.append(~row if substep.invert_result else row.copy())
        if step_meta.is_reset:
            kick_mat = m_mat ^ m_ref_arr[:, None]
            if np.any(kick_mat):
                self._scratch_x_mask[q_arr, :] ^= kick_mat
                any_x = True
        self._flush_xz(any_x, any_z)
        if step_meta.produces_measurements:
            self._apply_measurement_flip_noise(step_meta, op_override, len(step_meta.substeps))

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
                mpad_op = stim.CircuitInstruction(
                    "MPAD", [0] * num_m, [p_meas_err]
                )
                for b_idx in range(self.batch_size):
                    tr = self._tab_rngs[b_idx]
                    tr.do(mpad_op)
                    self._tab_rngs_x[b_idx].do(mpad_op)
                    rec = tr.current_measurement_record()
                    flips = rec[-num_m:]
                    for k_idx, flipped in enumerate(flips):
                        if flipped:
                            self._measurement_columns[-num_m + k_idx][
                                b_idx
                            ] = not self._measurement_columns[-num_m + k_idx][
                                b_idx
                            ]
            else:
                flips_2d = (
                    self.np_rng.random((num_m, self.batch_size)) < p_meas_err
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

    def _project_and_retire(self, b_idx, mrqf, aw, bw, sign_bool, rot, f_q, m, x2z_w, z2z_w, xs_base, zs_base):
        """Project shot b_idx onto outcome m (P3, or P2/P3/P2 if a != 0) and retire it if healed; returns kicks."""
        if rot is None:
            mrqf.p3_measure_diagonal(bw, (f_q ^ sign_bool) & 1, requested_bit=m)
        else:
            b_rot, e_p, k_rot = rot
            mrqf.p2_apply_quarter_turn(aw, b_rot, k_rot)
            mrqf.p3_measure_diagonal(e_p, 0, requested_bit=(m ^ f_q) & 1)
            mrqf.p2_apply_quarter_turn(aw, b_rot, (k_rot + 2) & 3)
        any_x = any_z = False
        if mrqf.d == 0:
            self._active_shots.pop(b_idx, None)
            if _words_any_nonzero(mrqf.h_words):
                hw = mrqf.h_words
                for qq in range(self.num_qubits):
                    if _word_dot_parity(z2z_w[qq], hw):
                        self._scratch_x_mask[qq, b_idx] ^= True
                        xs_base[qq, b_idx] ^= True
                        any_x = True
                    if _word_dot_parity(x2z_w[qq], hw):
                        self._scratch_z_mask[qq, b_idx] ^= True
                        if zs_base is not None:
                            zs_base[qq, b_idx] ^= True
                        any_z = True
        return any_x, any_z

    def do_unmatched_noisy_reset_z(
        self,
        reset_zero_mask: NDArray[np.bool_],
        reset_one_mask: NDArray[np.bool_],
        pauli: Literal["X", "Y", "Z"] = "Z",
        order: Sequence[int] | None = None,
    ) -> None:
        """Reset noisy qubits to +1 (reset_zero_mask) or -1 (reset_one_mask) eigenstate of `pauli` without altering the reference tableau."""
        if not (np.any(reset_zero_mask) or np.any(reset_one_mask)):
            return

        x2x_w, x2z_w, z2x_w, z2z_w, x_kappa, z_kappa = self._current_packed_words()

        if (
            pauli == "Z"
            and self._use_cpp
            and self._cpp_lib is not None
            and self._cpp_eng is not None
        ):
            self._sync_py_to_cpp()
            xs_base, _, _, _, _ = self._flip_simulator.to_numpy(
                output_xs=True, output_zs=False
            )
            xs_c = np.ascontiguousarray(xs_base, dtype=np.uint8)
            r0_c = np.ascontiguousarray(reset_zero_mask, dtype=np.uint8)
            r1_c = np.ascontiguousarray(reset_one_mask, dtype=np.uint8)
            self._scratch_x_mask.fill(False)
            self._scratch_z_mask.fill(False)
            self._any_x_c.value = 0
            self._any_z_c.value = 0

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
            self._cpp_dirty = True
            self._flush_xz(self._any_x_c.value, self._any_z_c.value)
            return

        if self._use_cpp and self._cpp_dirty:
            self._sync_cpp_to_py()

        combined_mask = reset_zero_mask | reset_one_mask
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

        for q_np in affected_qubits:
            q = int(q_np)
            shots_for_q = np.flatnonzero(combined_mask[q])
            if len(shots_for_q) == 0:
                continue

            aw, bw, kap = _pauli_image(pauli, q, x2x_w, x2z_w, z2x_w, z2z_w, x_kappa, z_kappa)
            a_is_zero, sign_bool, rot = self._projection_words(aw, bw, kap)
            ref_exp = -1 if sign_bool else 1

            for b_np in shots_for_q:
                b_idx = int(b_np)
                target_state = 1 if reset_one_mask[q, b_idx] else 0
                if pauli == "Z":
                    f_q = int(xs_base[q, b_idx])
                elif pauli == "X":
                    f_q = int(zs_base[q, b_idx])
                else:
                    f_q = int(xs_base[q, b_idx] ^ zs_base[q, b_idx])
                mrqf = self._active_shots.get(b_idx)
                if mrqf is not None and (
                    mrqf.d > 0 or _words_any_nonzero(mrqf.h_words)
                ):
                    mu = mrqf.q1_pauli_expectation(aw, bw, kap)
                    z_exp = -mu if f_q else mu
                else:
                    z_exp = (-ref_exp if f_q else ref_exp) if a_is_zero else 0

                need_flip = False
                if z_exp == 1:
                    need_flip = target_state == 1
                elif z_exp == -1:
                    need_flip = target_state == 0
                else:
                    m = int(self._draw_tab_rng_bit(b_idx))
                    if mrqf is None:
                        mrqf = MRQFState(self.num_qubits, self._n_words)
                        self._active_shots[b_idx] = mrqf
                    kx, kz = self._project_and_retire(
                        b_idx, mrqf, aw, bw, sign_bool, rot, f_q, m, x2z_w, z2z_w, xs_base, zs_base
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

        if pauli != "Z":
            self._py_dirty = True
        self._flush_xz(any_x_kick, any_z_kick)

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

        aw, bw, kap = _pauli_image(pauli, q, x2x_w, x2z_w, z2x_w, z2z_w, x_kappa, z_kappa)
        _, sign_bool, rot = self._projection_words(aw, bw, kap)

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
                mrqf = MRQFState(self.num_qubits, self._n_words)
                self._active_shots[b_idx] = mrqf
            kx, kz = self._project_and_retire(
                b_idx, mrqf, aw, bw, sign_bool, rot, f_q, m, x2z_w, z2z_w, xs_base, zs_base
            )
            any_x_kick |= kx
            any_z_kick |= kz

        self._py_dirty = True
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

    def peek_pauli_batch(
        self, targets: Sequence[int] | NDArray[np.int_], pauli: Literal["X", "Y", "Z"]
    ) -> NDArray[np.int8]:
        """Return exact single-qubit Pauli expectations (+1, -1, or 0) of shape (len(targets), batch_size) via Q1."""
        targets_arr = np.ascontiguousarray(targets, dtype=np.int64)
        if len(targets_arr) == 0:
            return np.zeros((0, self.batch_size), dtype=np.int8)

        x2x_w, x2z_w, z2x_w, z2z_w, x_kappa, z_kappa = (
            self._current_packed_words()
        )
        xs_base, zs_base, _, _, _ = self._flip_simulator.to_numpy(
            output_xs=True, output_zs=True
        )

        if self._use_cpp and self._cpp_lib is not None and self._cpp_eng is not None:
            self._sync_py_to_cpp(keep_py_dirty=self._py_dirty)
            xs_c = np.ascontiguousarray(xs_base, dtype=np.uint8)
            zs_c = np.ascontiguousarray(zs_base, dtype=np.uint8)
            out = np.empty((len(targets_arr), self.batch_size), dtype=np.int8)
            p_kind = 2 if pauli == "Z" else (0 if pauli == "X" else 1)
            self._cpp_lib.engine_peek_pauli_batch(
                self._cpp_eng,
                len(targets_arr),
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

        if pauli == "Z":
            f_comm_mat = xs_base[targets_arr, :]
            a_words_all = z2x_w[targets_arr]
            b_words_all = z2z_w[targets_arr]
            kappas = z_kappa[targets_arr]
        elif pauli == "X":
            f_comm_mat = zs_base[targets_arr, :]
            a_words_all = x2x_w[targets_arr]
            b_words_all = x2z_w[targets_arr]
            kappas = x_kappa[targets_arr]
        elif pauli == "Y":
            f_comm_mat = xs_base[targets_arr, :] ^ zs_base[targets_arr, :]
            a_words_all = x2x_w[targets_arr] ^ z2x_w[targets_arr]
            b_words_all = x2z_w[targets_arr] ^ z2z_w[targets_arr]
            dp = (
                np.bitwise_count(
                    x2z_w[targets_arr] & z2x_w[targets_arr]
                ).sum(axis=1, dtype=np.int32)
                & 1
            )
            kappas = (
                (
                    1
                    + x_kappa[targets_arr].astype(np.int32)
                    + z_kappa[targets_arr].astype(np.int32)
                    + (dp << 1)
                )
                & 3
            ).astype(np.uint8)
        else:
            raise ValueError(f"Unrecognised Pauli: {pauli}")

        a_is_zero = ~np.any(a_words_all != 0, axis=1)
        signs_bool = (kappas & 3) == 2
        ref_exp = np.where(signs_bool, -1, 1).astype(np.int8)
        out = np.where(
            a_is_zero[:, None],
            np.where(f_comm_mat, -ref_exp[:, None], ref_exp[:, None]),
            np.int8(0),
        )

        if self._active_shots:
            num_t = len(targets_arr)
            for idx_t in range(num_t):
                aw = a_words_all[idx_t]
                bw = b_words_all[idx_t]
                kap = int(kappas[idx_t])
                f_c_row = f_comm_mat[idx_t]
                for b_idx, mrqf in self._active_shots.items():
                    mu = mrqf.q1_pauli_expectation(aw, bw, kap)
                    out[idx_t, b_idx] = -mu if f_c_row[b_idx] else mu

        return out

    def peek_z(self, target: int) -> int | NDArray[np.int8]:
        res = self.peek_pauli_batch([target], "Z")[0]
        if self.batch_size == 1 or np.all(res == res[0]):
            return int(res[0])
        return res

    def peek_x(self, target: int) -> int | NDArray[np.int8]:
        res = self.peek_pauli_batch([target], "X")[0]
        if self.batch_size == 1 or np.all(res == res[0]):
            return int(res[0])
        return res

    def peek_y(self, target: int) -> int | NDArray[np.int8]:
        res = self.peek_pauli_batch([target], "Y")[0]
        if self.batch_size == 1 or np.all(res == res[0]):
            return int(res[0])
        return res

    def get_current_noisy_tableau_pauli_state(
        self, targets: Iterable[int], pauli: Literal["X", "Y", "Z"]
    ) -> list[int] | NDArray[np.int8]:
        targets_list = list(targets)
        batch_res = self.peek_pauli_batch(targets_list, pauli)
        if self.batch_size == 1:
            return [int(x) for x in batch_res[:, 0]]
        return batch_res

    ############################################################################
    # Readout Records & Detector/Observable Conversion
    ############################################################################

    def current_tableau_measurement_record(self) -> NDArray[np.bool_]:
        if len(self._measurement_columns) == 0:
            return np.zeros((0,), dtype=np.bool_)
        records = np.column_stack(self._measurement_columns)
        return records[0] if self.batch_size == 1 else records

    def get_final_measurement_records(self) -> NDArray[np.bool_]:
        if not self._finished_running_circuit:
            raise RuntimeError("The circuit has not been fully run yet.")
        if self._final_measurement_records is None:
            if len(self._measurement_columns) == 0:
                self._final_measurement_records = np.zeros(
                    (self.batch_size, 0), dtype=np.bool_
                )
            else:
                self._final_measurement_records = np.column_stack(
                    self._measurement_columns
                )
        return self._final_measurement_records

    def _convert_measurements_to_detector_flips(self) -> None:
        records = self.get_final_measurement_records()
        self._detector_flips, self._observable_flips = (
            self._compiled_m2d_converter.convert(
                measurements=records,
                separate_observables=True,
            )
        )

    def get_detector_flips(self, append_observables: bool = False) -> NDArray[np.bool_]:
        if self._detector_flips is None:
            self._convert_measurements_to_detector_flips()
        assert self._detector_flips is not None
        if append_observables and self._observable_flips is not None:
            return np.hstack([self._detector_flips, self._observable_flips])
        return self._detector_flips

    def get_observable_flips(self) -> NDArray[np.bool_]:
        if self._observable_flips is None:
            self._convert_measurements_to_detector_flips()
        assert self._observable_flips is not None
        return self._observable_flips
