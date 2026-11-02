"""Non-Clifford circuit front-end and `DIRECT` / `SELECTOR` trajectory compilers (`M1`: `F6`, `F9`, `F10`).

Re-export barrel over `_magic.{gates,channels,plan,statevector,trajectory}`.
"""

from __future__ import annotations

from stimside._magic.channels import (
    CompiledCliffordFactorList,
    CompiledOpenBlockRecord,
    CompiledRecord,
    CorrelatedNoiseRecord,
    DirectFaultSlotRecord,
    FeedbackPauliRecord,
    FrameEvent,
    HeraldedNoiseRecord,
    LeakageBarrierRecord,
    LeftResetXEvent,
    LeftSppEvent,
    LeftUnitaryEvent,
    MagicPhaseTerm,
    MagicTermRecord,
    MPadRecord,
    MeasurementRecord,
    PauliNoiseRecord,
    ResetRecord,
    RightRotEvent,
    _validate_and_emit_magic_op,
    apply_clifford_factors,
    compile_pauli_product,
)
from stimside._magic.gates import (
    CONTROLLED_PAULI_2Q_AXES,
    CliffordFaultChannel,
    CompiledPauliAxis,
    DECOMP_1Q,
    DECOMP_2Q,
    ExactCliffordDecomposition,
    FaultBranch,
    decompose_clifford_exact,
    eval_decomposition_unitary_le,
    extract_circuit_faults,
    parse_clifford_fault_instruction,
    parse_clifford_fault_tag,
)
from stimside._magic.phase_qfe import OMEGA, OMEGA_POWERS
from stimside._magic.plan import (
    DirectPlan,
    SelectorPlan,
    TrajectoryRunResult,
    compile_coordinates,
    compile_direct,
    compile_open_block,
    compile_selector,
)
from stimside._magic.statevector import (
    extract_data_expansion,
    reconstruct_statevector,
)
from stimside._magic.trajectory import (
    fault_selector_bits,
    run_direct,
    run_selector,
    sample_selector_activations,
)

_ = _validate_and_emit_magic_op

__all__ = [
    "OMEGA", "OMEGA_POWERS", "DECOMP_1Q", "DECOMP_2Q", "CONTROLLED_PAULI_2Q_AXES",
    "CompiledPauliAxis", "FaultBranch", "CliffordFaultChannel", "ExactCliffordDecomposition",
    "CompiledCliffordFactorList", "MagicPhaseTerm", "MagicTermRecord", "CompiledOpenBlockRecord",
    "DirectFaultSlotRecord", "PauliNoiseRecord", "HeraldedNoiseRecord", "CorrelatedNoiseRecord",
    "MeasurementRecord", "ResetRecord", "MPadRecord", "FeedbackPauliRecord", "LeakageBarrierRecord",
    "CompiledRecord", "LeftUnitaryEvent", "LeftSppEvent", "LeftResetXEvent", "RightRotEvent",
    "FrameEvent", "DirectPlan", "SelectorPlan", "TrajectoryRunResult",
    "decompose_clifford_exact", "eval_decomposition_unitary_le", "apply_clifford_factors",
    "parse_clifford_fault_tag", "parse_clifford_fault_instruction",
    "extract_circuit_faults", "compile_pauli_product",
    "compile_coordinates", "compile_open_block", "compile_direct", "run_direct",
    "compile_selector", "fault_selector_bits", "sample_selector_activations", "run_selector",
    "extract_data_expansion", "reconstruct_statevector",
]
