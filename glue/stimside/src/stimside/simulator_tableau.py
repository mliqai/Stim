from typing import Literal, Iterable

import numpy as np
import stim  # type: ignore[import-untyped]
from numpy.typing import NDArray

from stimside._tableau.runner import (
    STANDARD_LEAKAGE_PARAM_TYPES,
    CppTablesideRunner,
    do_sync_bare_inst,
    get_precompiled_circuit,
    run_tableside_sync_rng,
)
from stimside.op_handlers.abstract_op_handler import CompiledOpHandler
from stimside.op_handlers.leakage_handlers.leakage_parameters import (
    LeakageConditioningParams,
    LeakageMeasurementParams,
    LeakageTransition1Params,
    LeakageTransition2Params,
)
from stimside.util.circuit import ShotMajorM2DMixin, apply_coords_op, check_replay, unroll_circuit
from stimside.util.leakage_events import LeakageEvent, LeakageEventsRecorderMixin
from stimside.util.seeding import derived_seed
from stimside.util.unleaked_to_leaked import UnleakedToLeakedRecordsMixin


class TablesideSimulator(
    ShotMajorM2DMixin, UnleakedToLeakedRecordsMixin, LeakageEventsRecorderMixin
):
    """A convenient correlated error simulator based on stim.TableauSimulator.

    Inherits the advantages and disadvantages of a tableau simulator:
     - Handles all stabilizer operations
     - much slower than a flip simulator (O(N^2) cost, rather than O(1))

    The TablesideSimulator has three major additions over the stim.TableauSimulator:
    1. Extra methods to help use a TableauSimulator:

    2. Additional information regarding the circuit being simulated, particularly:
        - The qubit coordinates (including the effects of SHIFT_COORDS)
            This lets the simulator infer where qubits are, allowing behaviour (like errors)
            to depend on location. (See CompiledOpHandler for a way of adding such behaviour)

    3. CompiledOpHandler, which let the simulator track a state and implement complex behaviours:
        Lets the simulator hold configurable state, and provides an easy way of registering
        repeatable behaviours that depend on the capabilities provides above.

        We take a CompiledOpHandler rather than a OpHandler that we then compile to try
        to simplify the class and to encourage users not to do any slow work inside the simulator.
        You are of course free to implement a CompiledOpHandler by hand rather than by making a
        OpHandler, especially when you're using the simulator outside `sinter.collect`

        We take a CompiledOpHandler at all in order to help guide the dividing line between the
        simulator itself and the custom behaviour the user wants. Two rules of thumb are:
         - everything that is hardware or noise model dependent is in the CompiledOpHandler
    """

    @property
    def _tableau_simulator(self) -> stim.TableauSimulator:
        if self._tableau_simulator_inst is None:
            self._tableau_simulator_inst = stim.TableauSimulator(seed=self._tab_seed)
        return self._tableau_simulator_inst

    @property
    def _compiled_m2d_converter(self) -> stim.CompiledMeasurementsToDetectionEventsConverter:
        if self._compiled_m2d_converter_inst is None:
            if self._precompiled_circuit is not None:
                self._compiled_m2d_converter_inst = self._precompiled_circuit.m2d_converter
            else:
                self._compiled_m2d_converter_inst = self.circuit.compile_m2d_converter()
        return self._compiled_m2d_converter_inst  # type: ignore[return-value]

    def __init__(
        self,
        circuit: stim.Circuit,
        *,
        compiled_op_handler: CompiledOpHandler["TablesideSimulator"],
        seed: int | None = None,
        batch_size: int = 1,
        running_tableau: bool = True,
        construct_reference_circuit: bool = False,
        sync_tableside_rng: bool | None = None,
        record_unleaked_to_leaked: bool = False,
        record_leakage_events: bool = False,
    ) -> None:
        """record_leakage_events: record every leakage-state change of each shot
        (``get_leakage_events()``, see ``stimside.util.leakage_events``). With
        ``batch_size > 1`` (and no ``sync_tableside_rng``) all shots of a batch share
        one leakage trajectory, so they get the same events."""

        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or int(batch_size) <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        batch_size = int(batch_size)

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
        self._shot_counter = 0

        self.qubit_coords: dict[int, list[float]] = {}
        self.qubit_tags: dict[int, str] = {}
        self.coords_shifts: list[float] = []

        self._sync_tableside_rng = bool(sync_tableside_rng)

        shot_seed = derived_seed(seed, 0) if seed is not None else None
        self.np_rng = np.random.default_rng(seed=shot_seed)

        self._tab_seed = shot_seed if shot_seed is not None else int(self.np_rng.integers(0, 2**63))
        self._tableau_simulator_inst: stim.TableauSimulator | None = None
        self._new_circuit = stim.Circuit()
        self._construct_reference_circuit = construct_reference_circuit
        if self._construct_reference_circuit:
            self._new_reference_circuit = stim.Circuit()

        self._compiled_op_handler = compiled_op_handler
        self._precompiled_circuit = None
        from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import (
            CompiledLeakageUint8,
        )

        if type(compiled_op_handler) is CompiledLeakageUint8:
            self._precompiled_circuit = get_precompiled_circuit(
                circuit,
                bool(compiled_op_handler.unconditional_condition_on_U),
                compiled_op_handler.ops_to_params,
            )

        self._compiled_m2d_converter_inst: object | None = None
        if self._sync_tableside_rng:
            # Sync lockstep needs the live tableau: the deferred replay skips the X/Y split.
            eff_rt = True
        elif self._precompiled_circuit is not None:
            eff_rt = bool(running_tableau and self._precompiled_circuit.requires_tableau)
        else:
            eff_rt = bool(running_tableau)

        self._initial_running_tableau = eff_rt
        self._running_tableau = eff_rt
        if eff_rt:
            self._tableau_simulator_inst = stim.TableauSimulator(seed=self._tab_seed)
        self._finished_running_circuit = False

        self._cpp_runner = None
        if not self._sync_tableside_rng and self._precompiled_circuit is not None:
            self._cpp_runner = CppTablesideRunner(
                self._precompiled_circuit, seed=self._tab_seed
            )
            if hasattr(self._compiled_op_handler, "state"):
                np.copyto(self._cpp_runner.state, self._compiled_op_handler.state)
                self._compiled_op_handler.state = self._cpp_runner.state

        self._circuit_time = 0
        self._used_interactively = False
        self._unrolled_ops: list[stim.CircuitInstruction] | None = None  # set by interactive_do

        self._final_measurement_records: NDArray[np.bool_] | None = None
        self._detector_flips: NDArray[np.bool_] | None = None
        self._observable_flips: NDArray[np.bool_] | None = None
        self._asymmetric_readout_postproc: list[tuple[int, float, float, bool]] = []

    ############################################################################
    # Public methods without running interactively
    ############################################################################

    @staticmethod
    def _ops_collapse_z(ops_to_params: dict[object, object]) -> bool:
        for p in ops_to_params.values():
            if isinstance(p, LeakageConditioningParams) and any(
                0 in g or 1 in g for g in p.args
            ):
                return True
            if isinstance(p, LeakageTransition1Params) and (
                0 in p.args_by_input_state or 1 in p.args_by_input_state
            ):
                return True
            if isinstance(p, LeakageTransition2Params) and any(
                0 in k or 1 in k for k in p.args_by_input_state
            ):
                return True
        return False

    def _run_single_trajectory(self):
        pre = self._precompiled_circuit
        coh = self._compiled_op_handler
        can_use_fast = (
            pre is not None
            and coh.ops_to_params == pre.parsed_ops  # type: ignore[attr-defined]
            and all(
                isinstance(v, STANDARD_LEAKAGE_PARAM_TYPES)
                for v in coh.ops_to_params.values()  # type: ignore[attr-defined]
            )
        )

        if can_use_fast:
            assert pre is not None
            if self._sync_tableside_rng:
                run_tableside_sync_rng(self, pre)
            else:
                assert self._cpp_runner is not None
                self._cpp_runner.run(self)
        else:
            self._do(self.circuit)

    def _collapses_z(self) -> bool:
        pre = self._precompiled_circuit
        coh = self._compiled_op_handler
        if pre is not None and coh.ops_to_params == pre.parsed_ops:  # type: ignore[attr-defined]
            return pre.collapses_z
        if not hasattr(coh, "ops_to_params"):
            return False
        ops_to_params = coh.ops_to_params  # type: ignore[attr-defined]
        if self._ops_collapse_z(ops_to_params):
            return True
        found_asym = False
        for op in unroll_circuit(self.circuit):
            if op.name in ("DETECTOR", "TICK", "OBSERVABLE_INCLUDE"):
                continue
            if found_asym and any(t.is_measurement_record_target for t in op.targets_copy()):
                return True
            p_step = ops_to_params.get(op)
            if isinstance(p_step, LeakageMeasurementParams):
                is_mp = p_step.targets is not None
                has_01 = (0 in p_step.prob_for_input_state) or (1 in p_step.prob_for_input_state)
                if (is_mp and (has_01 or not np.isclose(p_step.p0, p_step.p1))) or (
                    not is_mp and not np.isclose(p_step.p0, 1.0 - p_step.p1)
                ):
                    found_asym = True
        return False

    def run(self):
        """run the simulator."""
        if self._used_interactively or self._circuit_time != 0:
            raise ValueError(
                "TablesideSimulator is not designed for interactive use. "
                "Use .clear() to prepare the simulator for a new run. "
            )

        if (self._sync_tableside_rng or self._collapses_z()) and self.batch_size > 1:
            orig_batch = self.batch_size
            meas_list = []
            saved_events = [[] for _ in range(orig_batch)] if self.record_unleaked_to_leaked else []
            saved_leakage_events: list[list[LeakageEvent]] = [[] for _ in range(orig_batch)]
            self.batch_size = 1
            try:
                for b in range(orig_batch):
                    if b > 0:
                        self.clear()
                    self._run_single_trajectory()
                    self._shot_counter += 1
                    rec = self.current_tableau_measurement_record()
                    meas_list.append(rec.copy())
                    if self.record_unleaked_to_leaked:
                        saved_events[b] = list(self._unleaked_to_leaked_events[0])
                    if self.record_leakage_events:
                        saved_leakage_events[b] = list(self._leakage_events[0])
            finally:
                self.batch_size = orig_batch
            self._finished_running_circuit = True
            self._final_measurement_records = np.vstack(meas_list)
            if self.record_leakage_events:
                self._leakage_events = saved_leakage_events
            if self.record_unleaked_to_leaked:
                self._unleaked_to_leaked_events = saved_events
                return self.get_unleaked_to_leaked_records()
            return None

        self._run_single_trajectory()

        self._shot_counter += 1
        self._finished_running_circuit = True

        if self.batch_size > 1:
            self._final_measurement_records = None
            self._share_shot0_leakage_events()
        else:
            if self._running_tableau:
                rec = self.current_tableau_measurement_record()
                self._final_measurement_records = np.array([rec])
            else:
                self._final_measurement_records = None

        if self.record_unleaked_to_leaked:
            return self.get_unleaked_to_leaked_records()
        return None

    def clear(self):
        """clear the simulator so it can be reused.

        This is performance critical, as it's going to be used by the sampler before every batch.
        Avoid just recreating the simulator, we shouldn't have to redo any configuration steps here
        """
        self._running_tableau = self._initial_running_tableau
        if self.seed is not None:
            shot_seed = derived_seed(self.seed, self._shot_counter)
            self.np_rng = np.random.default_rng(seed=shot_seed)
            self._tab_seed = shot_seed
            self._tableau_simulator_inst = (
                stim.TableauSimulator(seed=shot_seed) if self._running_tableau else None
            )
            if self._cpp_runner is not None:
                self._cpp_runner.clear(shot_seed)
        else:
            self._tab_seed = int(self.np_rng.integers(0, 2**63))
            self._tableau_simulator_inst = (
                stim.TableauSimulator(seed=self._tab_seed)
                if self._running_tableau
                else None
            )
            if self._cpp_runner is not None:
                self._cpp_runner.clear(self._tab_seed)

        self._new_circuit.clear()
        if self._construct_reference_circuit:
            self._new_reference_circuit = stim.Circuit()
        self._final_measurement_records = None
        self._detector_flips = None
        self._observable_flips = None
        self._asymmetric_readout_postproc.clear()
        self._finished_running_circuit = False

        self.qubit_coords = {}
        self.qubit_tags = {}
        self.coords_shifts = []

        self._compiled_op_handler.clear()
        if self._cpp_runner is not None and hasattr(self._compiled_op_handler, "state"):
            self._compiled_op_handler.state = self._cpp_runner.state

        if self.record_unleaked_to_leaked:
            self._unleaked_to_leaked_events = [
                [] for _ in range(self.batch_size)
            ]
        if self.record_leakage_events:
            self._leakage_events = [[] for _ in range(self.batch_size)]

        self._circuit_time = 0
        self._used_interactively = False

    def get_current_circuit_time(self) -> int:
        return self._circuit_time

    ############################################################################
    # Public interactive methods to run the simulation
    ############################################################################
    def interactive_do(
        self, this: stim.Circuit | stim.CircuitInstruction | stim.CircuitRepeatBlock, /
    ):
        """Do the next operation(s): the circuit this simulator was constructed with, then any more.

        The detector/observable converter and op handler are precomputed from that circuit, so
        `this` must continue it in order (REPEAT blocks unrolled), except that an untagged noisy gate
        (e.g. X_ERROR, M(p)) may have a different noise strength; else ValueError. Past its end,
        untagged operations other than DETECTOR and OBSERVABLE_INCLUDE may follow (their
        measurements are recorded, but get_detector_flips may then raise); others raise ValueError.

        Pass the circuit itself or consecutive parts of it, not `circuit.flattened()` (which drops
        SHIFT_COORDS, rewrites DETECTOR coordinates and can merge operations) or `circuit + more`
        (where stim can merge operations at the seam).
        """
        if self._finished_running_circuit:
            raise RuntimeError(
                "Cannot do more operations after finishing an interactive run."
            )
        if self._unrolled_ops is None:
            self._unrolled_ops = unroll_circuit(self.circuit)
        if not self._used_interactively:
            if self._sync_tableside_rng and self.batch_size > 1:
                raise ValueError(
                    "interactive_do with sync_tableside_rng=True requires batch_size=1."
                )
            self._used_interactively = True
            self._running_tableau = True
        self._do(this)

    def finish_interactive_run(self):
        """Finish an interactive run of the simulator."""
        if not self._finished_running_circuit:
            self._shot_counter += 1
        self._finished_running_circuit = True
        if self.batch_size > 1:
            self._share_shot0_leakage_events()

    def _share_shot0_leakage_events(self) -> None:
        # All shots of the batch share the one simulated leakage trajectory.
        if self.record_leakage_events:
            ev0 = self._leakage_events[0]
            self._leakage_events = [list(ev0) for _ in range(self.batch_size)]
        if (
            self.record_unleaked_to_leaked
            and len(self._unleaked_to_leaked_events) == self.batch_size
            and len(self._unleaked_to_leaked_events[0]) > 0
            and len(self._unleaked_to_leaked_events[1]) == 0
        ):
            ev0 = list(self._unleaked_to_leaked_events[0])
            for b in range(1, self.batch_size):
                self._unleaked_to_leaked_events[b] = list(ev0)

    ############################################################################
    # private methods to run the simulation
    ############################################################################

    def _do(
        self, this: stim.Circuit | stim.CircuitInstruction | stim.CircuitRepeatBlock
    ):
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
                check_replay(this, self._unrolled_ops, self._circuit_time, "TablesideSimulator")
            self._do_instruction(this)

    def _do_instruction(self, op: stim.CircuitInstruction):
        """for each instruction, update the state of the simulator."""

        # first handle simulator specific stuff
        if apply_coords_op(op, self.qubit_coords, self.qubit_tags, self.coords_shifts):
            self._new_circuit.append(op)
            if self._construct_reference_circuit:
                self._new_reference_circuit.append(op)
            self._circuit_time += 1
            return

        self._compiled_op_handler.handle_op(op=op, sss=self)
        self._circuit_time += 1

    def _do_bare_instruction(
        self,
        op: stim.Circuit | stim.CircuitInstruction,
        *,
        include_in_reference: bool = True,
    ):
        """Apply a stabilizer / error instruction to `_new_circuit` (and TableauSimulator if running)."""
        if self.batch_size > 1 and self._asymmetric_readout_postproc:
            insts = op if isinstance(op, stim.Circuit) else (op,)
            for inst in insts:
                if inst.name not in ("DETECTOR", "OBSERVABLE_INCLUDE") and any(
                    t.is_measurement_record_target for t in inst.targets_copy()
                ):
                    raise ValueError(
                        "interactive_do with batch_size > 1 cannot apply classical rec[-k] feedback "
                        "after an asymmetric 0/1 readout; use run() or batch_size=1."
                    )
        inc_ref = include_in_reference and self._construct_reference_circuit
        if isinstance(op, stim.Circuit):
            self._new_circuit += op
            if inc_ref:
                self._new_reference_circuit += op
            if self._running_tableau:
                self._tableau_simulator.do_circuit(op)
        else:
            self._new_circuit.append(op)
            if inc_ref:
                self._new_reference_circuit.append(op)
            if self._running_tableau:
                if self._sync_tableside_rng:
                    do_sync_bare_inst(self._tableau_simulator, op)
                else:
                    self._tableau_simulator.do(op)

    def _do_bare_only_on_tableau(self, op: stim.Circuit | stim.CircuitInstruction):
        self._do_bare_instruction(op, include_in_reference=False)

    def _append_to_new_reference_circuit(
        self, op: stim.Circuit | stim.CircuitInstruction
    ):
        """
        Append to the new reference circuit
        """
        if self._construct_reference_circuit:
            self._new_reference_circuit.append(op)

    def _run_tableau(self):
        """
        Run the tableau if not already running
        """
        if self._running_tableau:
            return
        else:
            self._tableau_simulator.do_circuit(self._new_circuit)
            self._running_tableau = True
    ############################################################################
    # known states methods:
    ############################################################################

    def get_current_noisy_tableau_pauli_state(
        self, targets: Iterable[int], pauli: Literal["X", "Y", "Z"]
    ) -> list[int]:
        """returns true if all targets are in a known state of the given pauli."""
        match pauli:
            case "X":
                peek_pauli = self.peek_x
            case "Y":
                peek_pauli = self.peek_y
            case "Z":
                peek_pauli = self.peek_z
            case _:
                raise ValueError("Unrecognised Pauli")

        pauli_states = []
        for target in targets:
            pauli_states.append(peek_pauli(target))
        return pauli_states
    
    ############################################################################
    # properties
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

    ############################################################################
    # tableau simulator lookthrough methods
    ############################################################################
    def num_qubits_in_new_circuit(self) -> int:
        return self._new_circuit.num_qubits
    
    def peek_z(self, target: int) -> int:
        self._run_tableau()
        return self._tableau_simulator.peek_z(target)

    def peek_x(self, target: int) -> int:
        self._run_tableau()
        return self._tableau_simulator.peek_x(target)

    def peek_y(self, target: int) -> int:
        self._run_tableau()
        return self._tableau_simulator.peek_y(target)

    def peek_bloch(self, target: int) -> stim.PauliString:
        self._run_tableau()
        return self._tableau_simulator.peek_bloch(target)

    def peek_observable_expectation(
        self,
        observable: stim.PauliString,
    ) -> int:
        self._run_tableau()
        return self._tableau_simulator.peek_observable_expectation(observable)

    def current_tableau_measurement_record(self) -> NDArray[np.bool_]:
        if self.batch_size > 1:
            print( "Warning when calling current_tableau_measurement_record: "
                "The tableau simulator is not directly called if batch_size > 1."
            )
        self._run_tableau()
        return np.array(self._tableau_simulator.current_measurement_record(), dtype=np.bool_)

    def get_final_measurement_records(self) -> NDArray[np.bool_]:
        if self._finished_running_circuit is False:
            raise RuntimeError("The circuit has not been fully run yet.")
        if self._final_measurement_records is None:
            if self._running_tableau and self.batch_size == 1:
                self._final_measurement_records = np.array([self.current_tableau_measurement_record()])
            else:
                self._final_measurement_records = self._new_circuit.compile_sampler(
                        seed=int(self.np_rng.integers(0, 2**63))
                    ).sample(shots=self.batch_size, bit_packed=False)
                for meas_idx, p0, p1, is_inverted in self._asymmetric_readout_postproc:
                    col = self._final_measurement_records[:, meas_idx]
                    raw_bit = ~col if is_inverted else col
                    prob_flip = np.where(raw_bit, 1.0 - p1, p0)
                    flips = self.np_rng.random(self.batch_size) < prob_flip
                    self._final_measurement_records[:, meas_idx] ^= flips
        return self._final_measurement_records.copy()
