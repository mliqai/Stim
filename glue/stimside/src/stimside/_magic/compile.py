"""Non-Clifford circuit front-end and `DIRECT` / `SELECTOR` trajectory compilers (`M1`: `F6`, `F9`, `F10`).

Provides exact `omega = exp(i*pi/4)` Clifford decompositions, `CLIFFORD_FAULT` & magic-tag parsing,
moving-reference lowering (`C1`/`C1.5`), commuting phase-block compilation (`C2`), and `DIRECT`/`SELECTOR` runners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal, Mapping, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]

from stimside.op_handlers.leakage_handlers.common_parsing import split_arguments
from stimside.op_handlers.leakage_handlers.leakage_parameters import LeakageParams
from stimside.op_handlers.leakage_handlers.tag_registry import check_tag_qubits, parse_leakage_tag_magic
from stimside.util.bitwords import (
    I_POWERS_ARR as _PHASE4_ARR,
    INV_SQRT2 as _INV_SQRT2,
    OMEGA,
    OMEGA_ARR as _OMEGA_ARR,
    OMEGA_POWERS,
    first_set_bit_in_words,
    get_bit_in_words,
    n_words_for_qubits,
    pauli_multiply,
    tail_mask,
    unit_bit_words,
    validate_hermitian_pauli,
    word_dot_parity,
    word_dot_parity_batch,
    words_any_nonzero,
)
import stimside._magic.phase_qfe as _pq
from stimside._magic.phase_qfe import (
    DEFAULT_ATOL, K1_CANCEL_REL_TOL, CompiledPauliBlock, CompiledPauliBlockStep,
    ImpossiblePostselectionError, ShotExpansion,
    _apply_local_unitary_c128, apply_compiled_block, apply_pauli, apply_quarter_turn,
    apply_reference_tableau_to_statevector, merge_frames, project_diagonal, validate_atol,
)
from stimside.util.reference_chp import DetMeasMeta, RandMeasMeta, ReferenceCHPTableau

__all__ = [
    "OMEGA", "OMEGA_POWERS", "DECOMP_1Q", "DECOMP_2Q", "CONTROLLED_PAULI_2Q_AXES",
    "CompiledPauliAxis", "FaultBranch", "CliffordFaultChannel", "ExactCliffordDecomposition",
    "CompiledCliffordFactorList", "MagicPhaseTerm", "MagicTermRecord", "CompiledOpenBlockRecord",
    "DirectFaultSlotRecord", "PauliNoiseRecord", "HeraldedNoiseRecord", "CorrelatedNoiseRecord",
    "MeasurementRecord", "ResetRecord", "MPadRecord", "FeedbackPauliRecord", "LeakageBarrierRecord",
    "CompiledRecord", "LeftUnitaryEvent", "LeftSppEvent", "LeftResetXEvent", "RightRotEvent",
    "FrameEvent", "CoordinateCompilation", "DirectPlan", "SelectorPlan", "TrajectoryRunResult",
    "clone_reference_chp", "decompose_clifford_exact", "eval_decomposition_unitary_le",
    "parse_clifford_fault_tag", "parse_clifford_fault_instruction",
    "extract_circuit_faults", "compile_pauli_product", "apply_spp_to_reference_chp",
    "compile_coordinates", "compile_open_block", "compile_direct", "run_direct",
    "compile_selector", "sample_selector_activations", "run_selector",
    "extract_data_expansion", "reconstruct_statevector",
]

_PAU_ABK: dict[str, tuple[int, int, int]] = {"I": (0, 0, 0), "X": (1, 0, 0), "Z": (0, 1, 0), "Y": (1, 1, 1)}

# Verified exact F = OMEGA_POWERS[v] * R(r_specs[-1]) ... R(r_specs[0]) * P(p0_spec) over all 46 canonical 1Q/2Q Stim gates
DECOMP_1Q: dict[str, tuple[int, str, tuple[str, ...]]] = {
    "I": (0, "+I", ()), "X": (0, "+X", ()), "Y": (0, "+Y", ()), "Z": (0, "+Z", ()),
    "H": (0, "+X", ("+Y",)), "H_NXZ": (0, "+X", ("-Y",)),
    "H_XY": (0, "+X", ("-Z",)), "H_NXY": (0, "+X", ("+Z",)),
    "H_YZ": (0, "+Z", ("+X",)), "H_NYZ": (0, "+Y", ("+X",)),
    "S": (1, "+I", ("-Z",)), "S_DAG": (7, "+I", ("+Z",)),
    "SQRT_X": (1, "+I", ("-X",)), "SQRT_X_DAG": (7, "+I", ("+X",)),
    "SQRT_Y": (1, "+I", ("-Y",)), "SQRT_Y_DAG": (7, "+I", ("+Y",)),
    "C_XYZ": (0, "+I", ("-X", "-Z")), "C_ZYX": (0, "+I", ("+Z", "+X")),
    "C_NXYZ": (0, "+I", ("+Y", "-X")), "C_NZYX": (0, "+I", ("+Z", "-X")),
    "C_XNYZ": (0, "+I", ("+X", "+Z")), "C_XYNZ": (0, "+I", ("+X", "-Z")),
    "C_ZNYX": (0, "+I", ("+Y", "-Z")), "C_ZYNX": (0, "+I", ("+X", "-Y")),
}
DECOMP_2Q: dict[str, tuple[int, str, tuple[str, ...]]] = {
    "II": (0, "+II", ()),
    "SQRT_XX": (1, "+II", ("-XX",)), "SQRT_XX_DAG": (7, "+II", ("+XX",)),
    "SQRT_YY": (1, "+II", ("-YY",)), "SQRT_YY_DAG": (7, "+II", ("+YY",)),
    "SQRT_ZZ": (1, "+II", ("-ZZ",)), "SQRT_ZZ_DAG": (7, "+II", ("+ZZ",)),
    "CZ": (1, "+II", ("+ZZ", "-ZI", "-IZ")),
    "CX": (1, "+II", ("+ZX", "-ZI", "-IX")),
    "CY": (1, "+II", ("+ZY", "-ZI", "-IY")),
    "XCX": (1, "+II", ("+XX", "-XI", "-IX")), "XCY": (1, "+II", ("+XY", "-XI", "-IY")),
    "XCZ": (1, "+II", ("+XZ", "-XI", "-IZ")), "YCX": (1, "+II", ("+YX", "-YI", "-IX")),
    "YCY": (1, "+II", ("+YY", "-YI", "-IY")), "YCZ": (1, "+II", ("+YZ", "-YI", "-IZ")),
    "SWAP": (1, "+II", ("-ZZ", "-YY", "-XX")),
    "ISWAP": (0, "+II", ("+XX", "+YY")), "ISWAP_DAG": (0, "+II", ("-XX", "-YY")),
    "CZSWAP": (0, "+IZ", ("-XY", "+YX")),
    "CXSWAP": (2, "+XY", ("-IY", "-XY", "-YX", "+YI")), "SWAPCX": (2, "+YX", ("-YI", "-YX", "-XY", "+IY")),
}
CONTROLLED_PAULI_2Q_AXES: dict[str, tuple[str, str]] = {
    "CZ": ("Z", "Z"), "CX": ("Z", "X"), "CY": ("Z", "Y"),
    "XCX": ("X", "X"), "XCY": ("X", "Y"), "XCZ": ("X", "Z"),
    "YCX": ("Y", "X"), "YCY": ("Y", "Y"), "YCZ": ("Y", "Z"),
}


def _coerce_strict_int(val: object, label: str, *, allow_bool: bool = False) -> int:
    if isinstance(val, (bool, np.bool_)):
        if allow_bool:
            return int(bool(val))
        raise ValueError(f"Boolean value {val!r} is not valid for {label}")
    if not isinstance(val, (int, np.integer)):
        raise ValueError(f"Expected integer for {label}, got {val!r}")
    return int(val)


def _canonical_gate_data(gate_name: str) -> stim.GateData:
    if not isinstance(gate_name, str) or not gate_name.strip():
        raise ValueError(f"Invalid gate name: {gate_name!r}")
    try:
        return stim.gate_data(gate_name.strip())
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(f"Unknown Stim gate {gate_name!r}") from exc


def _canonical_clifford_gate_name_and_arity(gate_name: str) -> tuple[str, int]:
    gd = _canonical_gate_data(gate_name)
    if not gd.is_unitary or not (gd.is_single_qubit_gate or gd.is_two_qubit_gate):
        raise ValueError(f"Gate {gate_name!r} is not a 1Q or 2Q unitary Clifford gate")
    return gd.name, (1 if gd.is_single_qubit_gate else 2)


def _plain_qubits(targets: Sequence[stim.GateTarget], label: str) -> list[int]:
    out: list[int] = []
    for t in targets:
        if not t.is_qubit_target or t.is_inverted_result_target:
            raise ValueError(f"{label} targets must be plain non-inverted qubits, got {t!r}")
        out.append(int(t.qubit_value))
    return out


# ---------------------------------------------------------------------------
# Core Compiler Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledPauliAxis:
    """Hermitian N-qubit Pauli `P = i**kappa X(a_words) Z(b_words)`."""
    a_words: NDArray[np.uint64]
    b_words: NDArray[np.uint64]
    kappa: int

    def as_tuple(self) -> tuple[NDArray[np.uint64], NDArray[np.uint64], int]:
        return (self.a_words, self.b_words, self.kappa)


@dataclass(frozen=True)
class FaultBranch:
    """One non-identity branch of a stochastic Clifford fault channel."""
    prob: float
    gate_name: str
    targets: tuple[int, ...]

    def __post_init__(self) -> None:
        p = float(self.prob)
        if not math.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"Invalid FaultBranch probability: {self.prob!r}")
        canon_name, arity = _canonical_clifford_gate_name_and_arity(self.gate_name)
        t_tup = tuple(_coerce_strict_int(q, "FaultBranch target") for q in self.targets)
        if len(t_tup) != arity or any(q < 0 for q in t_tup) or len(set(t_tup)) != arity:
            raise ValueError(f"Invalid targets {t_tup} for {arity}Q gate {canon_name!r}")
        object.__setattr__(self, "prob", p)
        object.__setattr__(self, "gate_name", canon_name)
        object.__setattr__(self, "targets", t_tup)


@dataclass(frozen=True)
class CliffordFaultChannel:
    """Categorical Clifford fault slot firing before flattened instruction `step_index` (`0..len(ops)`)."""
    step_index: int
    branches: tuple[FaultBranch, ...]
    slot_id: int | None = None

    def __post_init__(self) -> None:
        step = _coerce_strict_int(self.step_index, "CliffordFaultChannel step_index")
        if step < 0:
            raise ValueError(f"CliffordFaultChannel step_index must be >= 0, got {step}")
        object.__setattr__(self, "step_index", step)
        if self.slot_id is not None:
            sid = _coerce_strict_int(self.slot_id, "CliffordFaultChannel slot_id")
            if sid < 0:
                raise ValueError(f"CliffordFaultChannel slot_id must be >= 0, got {sid}")
            object.__setattr__(self, "slot_id", sid)
        b_tup = tuple(self.branches)
        if not b_tup or any(not isinstance(b, FaultBranch) for b in b_tup):
            raise ValueError(f"CliffordFaultChannel requires non-empty FaultBranch tuple, got {b_tup!r}")
        total_p = sum(b.prob for b in b_tup)
        if not math.isfinite(total_p) or total_p > 1.0 + 1e-12:
            raise ValueError(f"CliffordFaultChannel branch probabilities sum to {total_p!r} > 1.0")
        object.__setattr__(self, "branches", b_tup)


@dataclass(frozen=True)
class ExactCliffordDecomposition:
    """Exact unitary decomposition `F = omega**v * R(r_specs[-1]) ... R(r_specs[0]) * P(p0_spec)`."""
    num_qubits: int
    v: int
    p0_spec: str
    r_specs: tuple[str, ...]
    targets: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "v", int(self.v) & 7)
        if not self.targets:
            object.__setattr__(self, "targets", tuple(range(self.num_qubits)))

    @property
    def global_scalar(self) -> complex:
        return OMEGA_POWERS[self.v]


# ---------------------------------------------------------------------------
# Exact Clifford Decomposition & Symplectic Defect Helpers
# ---------------------------------------------------------------------------


def _spec_to_abk_small(spec: str) -> tuple[int, int, int]:
    a_m, b_m, kap = 0, 0, (2 if spec[0] == "-" else 0)
    for i, ch in enumerate(spec[1:]):
        if ch not in _PAU_ABK:
            raise ValueError(f"Invalid Pauli char {ch!r} in {spec!r}")
        xa, zb, dk = _PAU_ABK[ch]
        a_m |= xa << i
        b_m |= zb << i
        kap = (kap + dk) & 3
    return a_m, b_m, kap


def _abk_small_to_spec(a_m: int, b_m: int, kap: int, k: int) -> str:
    exp4 = (kap - (a_m & b_m).bit_count()) & 3
    if exp4 not in (0, 2):
        raise ValueError("Non-Hermitian Pauli encountered in decomposition")
    body = "".join("IZXY"[(((a_m >> i) & 1) << 1) | ((b_m >> i) & 1)] for i in range(k))
    return ("-" if exp4 == 2 else "+") + body


def _spec_terms(spec: str, targets: Sequence[int]) -> tuple[bool, list[tuple[int, str]]]:
    return spec[0] == "-", [(int(targets[i]), ch) for i, ch in enumerate(spec[1:]) if ch != "I"]


def _dense_pauli_from_spec_le(spec: str) -> NDArray[np.complex128]:
    dim = 1 << (len(spec) - 1)
    a_m, b_m, kap = _spec_to_abk_small(spec)
    cols = np.arange(dim, dtype=np.int64)
    mat = np.zeros((dim, dim), dtype=np.complex128)
    mat[cols ^ a_m, cols] = _PHASE4_ARR[(kap + ((np.bitwise_count(cols & b_m) & 1) << 1)) & 3]
    return mat


def _validate_decomp_targets(targets: Sequence[int] | None, k: int, label: object) -> tuple[int, ...]:
    t_tup = tuple(range(k)) if targets is None else tuple(int(q) for q in targets)
    if len(t_tup) != k or len(set(t_tup)) != k or any(q < 0 for q in t_tup):
        raise ValueError(f"Invalid targets={t_tup} for {k}-qubit Clifford {label!r}")
    return t_tup


def eval_decomposition_unitary_le(
    decomp: ExactCliffordDecomposition,
    *,
    num_qubits: int | None = None,
    targets: Sequence[int] | None = None,
) -> NDArray[np.complex128]:
    """Evaluate `decomp` as an exact `complex128` little-endian unitary matrix."""
    k = decomp.num_qubits
    eye_k = np.eye(1 << k, dtype=np.complex128)
    U_local = _dense_pauli_from_spec_le(decomp.p0_spec)
    for r_spec in decomp.r_specs:
        U_local = (_INV_SQRT2 * (eye_k + 1.0j * _dense_pauli_from_spec_le(r_spec))) @ U_local
    U_local = OMEGA_POWERS[decomp.v] * U_local

    t_seq = decomp.targets if targets is None else tuple(int(q) for q in targets)
    n_tot = k if num_qubits is None else int(num_qubits)
    if n_tot == k and tuple(t_seq) == tuple(range(k)):
        return U_local
    if len(t_seq) != k or len(set(t_seq)) != k or any(q < 0 or q >= n_tot for q in t_seq):
        raise ValueError(f"Invalid embedding targets={t_seq} for k={k}, num_qubits={n_tot}")
    dim_tot = 1 << n_tot
    out = np.zeros((dim_tot, dim_tot), dtype=np.complex128)
    for col in range(dim_tot):
        psi = np.zeros(dim_tot, dtype=np.complex128)
        psi[col] = 1.0 + 0.0j
        out[:, col] = _apply_local_unitary_c128(psi, U_local, t_seq, n_tot, "little")
    return out


def decompose_clifford_exact(
    gate_or_tableau: str | stim.Tableau,
    targets: Sequence[int] | None = None,
    *,
    inverse: bool = False,
) -> ExactCliffordDecomposition:
    """Decompose a named 1Q/2Q Stim Clifford or `stim.Tableau` with exact `omega**v` phase."""
    if isinstance(gate_or_tableau, str):
        canon_name, k = _canonical_clifford_gate_name_and_arity(gate_or_tableau)
        v, p0_spec, r_specs = (DECOMP_1Q if k == 1 else DECOMP_2Q)[canon_name]
        t_tup = _validate_decomp_targets(targets, k, gate_or_tableau)
        if not inverse:
            return ExactCliffordDecomposition(k, v, p0_spec, r_specs, t_tup)
        inv_r: list[str] = []
        for r_sp in reversed(r_specs):
            comm = sum(c1 != "I" and c2 != "I" and c1 != c2 for c1, c2 in zip(p0_spec[1:], r_sp[1:])) % 2 == 0
            new_neg = (r_sp[0] != "-") if comm else (r_sp[0] == "-")
            inv_r.append(("-" if new_neg else "+") + r_sp[1:])
        return ExactCliffordDecomposition(k, (-v) & 7, p0_spec, tuple(inv_r), t_tup)

    if not isinstance(gate_or_tableau, stim.Tableau):
        raise TypeError(f"Expected gate name str or stim.Tableau, got {type(gate_or_tableau)!r}")

    tab = gate_or_tableau.inverse() if inverse else gate_or_tableau
    k = len(tab)
    t_tup = _validate_decomp_targets(targets, k, "Tableau")
    if k == 0:
        return ExactCliffordDecomposition(0, 0, "+", (), ())

    v_acc, p_a, p_b, p_kap = 0, 0, 0, 0
    r_abk: list[tuple[int, int, int]] = []

    def _embed(spec: str, loc_qs: Sequence[int]) -> tuple[int, int, int]:
        la, lb, lk = _spec_to_abk_small(spec)
        ga = sum(1 << q_l for idx_l, q_l in enumerate(loc_qs) if (la >> idx_l) & 1)
        gb = sum(1 << q_l for idx_l, q_l in enumerate(loc_qs) if (lb >> idx_l) & 1)
        return ga, gb, lk

    for inst in tab.to_circuit(method="elimination"):
        gname = inst.name
        inst_qs = [t.value for t in inst.targets_copy()]
        arity = 1 if gname in DECOMP_1Q else 2
        sub_dec = decompose_clifford_exact(gname)
        for off in range(0, len(inst_qs), arity):
            loc_qs = inst_qs[off : off + arity]
            v_acc = (v_acc + sub_dec.v) & 7
            sa, sb, sk = _embed(sub_dec.p0_spec, loc_qs)
            r_abk = [
                (ra, rb, (rk + ((((sa & rb).bit_count() ^ (sb & ra).bit_count()) & 1) << 1)) & 3)
                for ra, rb, rk in r_abk
            ]
            r_abk.extend(_embed(r_sp, loc_qs) for r_sp in sub_dec.r_specs)
            p_kap = (sk + p_kap + (((sb & p_a).bit_count() & 1) << 1)) & 3
            p_a ^= sa
            p_b ^= sb

    p0_spec = _abk_small_to_spec(p_a, p_b, p_kap, k)
    r_specs_out = tuple(_abk_small_to_spec(ra, rb, rk, k) for ra, rb, rk in r_abk)
    dec = ExactCliffordDecomposition(k, v_acc, p0_spec, r_specs_out, t_tup)
    if k <= 6:
        U_tab = tab.to_unitary_matrix(endian="little").astype(np.complex128)
        U_dec = eval_decomposition_unitary_le(dec)
        r_i, c_i = divmod(int(np.argmax(np.abs(U_dec))), U_dec.shape[1])
        ratio = U_tab[r_i, c_i] / U_dec[r_i, c_i]
        shift_v = int(round(math.atan2(ratio.imag, ratio.real) / (math.pi / 4.0))) & 7
        dec = ExactCliffordDecomposition(k, (v_acc + shift_v) & 7, p0_spec, r_specs_out, t_tup)
    return dec


# ---------------------------------------------------------------------------
# CLIFFORD_FAULT & Magic Tag Parsing
# ---------------------------------------------------------------------------


def parse_clifford_fault_tag(tag: str) -> tuple[tuple[float, str], ...]:
    """Parse `CLIFFORD_FAULT: (p1, GATE1) (p2, GATE2) ...` into `((p, canon_gate), ...)`."""
    s = tag.strip()
    if not s.startswith("CLIFFORD_FAULT:"):
        raise ValueError(f"Malformed CLIFFORD_FAULT tag (expected 'CLIFFORD_FAULT: ...'): {tag!r}")
    body = s[len("CLIFFORD_FAULT:") :].strip()
    if not body or not (body.startswith("(") and body.endswith(")")):
        raise ValueError(f"CLIFFORD_FAULT body must consist of '(prob, GATE)' groups, got {tag!r}")
    depth = 0
    for ch in body:
        if ch == "(":
            depth += 1
            if depth > 1:
                raise ValueError(f"Nested '(' in CLIFFORD_FAULT tag: {tag!r}")
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced ')' in CLIFFORD_FAULT tag: {tag!r}")
        elif depth == 0 and ch not in " \t\n\r,;":
            raise ValueError(f"Unexpected character {ch!r} between groups in CLIFFORD_FAULT tag: {tag!r}")
    if depth != 0:
        raise ValueError(f"Unclosed '(' in CLIFFORD_FAULT tag: {tag!r}")

    raw_groups = split_arguments(body)
    if not raw_groups:
        raise ValueError(f"Empty CLIFFORD_FAULT tag: {tag!r}")

    parsed: list[tuple[float, str]] = []
    expected_arity: int | None = None
    total_prob = 0.0
    for grp in raw_groups:
        if len(grp) != 2:
            raise ValueError(f"Each CLIFFORD_FAULT tuple must be (prob, GATE), got {grp!r} in {tag!r}")
        p_str, g_str = grp
        try:
            prob = float(p_str)
        except ValueError as exc:
            raise ValueError(f"Invalid probability {p_str!r} in CLIFFORD_FAULT tag {tag!r}") from exc
        if not math.isfinite(prob) or prob < 0.0 or prob > 1.0:
            raise ValueError(f"CLIFFORD_FAULT probability must be finite in [0, 1], got {prob!r}")
        canon_gate, arity = _canonical_clifford_gate_name_and_arity(g_str)
        if expected_arity is None:
            expected_arity = arity
        elif arity != expected_arity:
            raise ValueError(f"Mixed gate arities ({expected_arity} vs {arity}) in CLIFFORD_FAULT tag {tag!r}")
        total_prob += prob
        parsed.append((prob, canon_gate))

    if total_prob > 1.0 + 1e-12:
        raise ValueError(f"CLIFFORD_FAULT probabilities sum to {total_prob!r} > 1.0 in {tag!r}")
    return tuple(parsed)


def parse_clifford_fault_instruction(
    op: stim.CircuitInstruction, step_index: int
) -> list[CliffordFaultChannel]:
    """Validate and expand an `I[CLIFFORD_FAULT: ...]` or `II[CLIFFORD_FAULT: ...]` instruction."""
    canon_carrier = _canonical_gate_data(op.name).name
    if canon_carrier not in ("I", "II") or op.gate_args_copy():
        raise ValueError(f"Invalid CLIFFORD_FAULT carrier/args on {op!r}")
    parsed_branches = parse_clifford_fault_tag(op.tag)
    _, arity = _canonical_clifford_gate_name_and_arity(parsed_branches[0][1])
    if canon_carrier == "II" and arity != 2:
        raise ValueError(f"Carrier 'II' requires 2-qubit CLIFFORD_FAULT gates, got {parsed_branches}")
    qs = _plain_qubits(op.targets_copy(), "CLIFFORD_FAULT")
    if len(qs) == 0 or (len(qs) % arity) != 0:
        raise ValueError(f"CLIFFORD_FAULT with {arity}Q gates requires a positive multiple of {arity} targets")
    channels: list[CliffordFaultChannel] = []
    for off in range(0, len(qs), arity):
        grp = tuple(qs[off : off + arity])
        if len(set(grp)) != arity:
            raise ValueError(f"Duplicate qubit in {arity}Q CLIFFORD_FAULT target group {grp}")
        channels.append(
            CliffordFaultChannel(step_index, tuple(FaultBranch(p, g, grp) for p, g in parsed_branches))
        )
    return channels


def extract_circuit_faults(
    circuit: stim.Circuit,
) -> tuple[list[stim.CircuitInstruction], tuple[CliffordFaultChannel, ...]]:
    """Flatten `REPEAT` blocks and extract inline `CLIFFORD_FAULT` tagged instructions."""
    flat_ops: list[stim.CircuitInstruction] = []
    inline_faults: list[CliffordFaultChannel] = []

    def _walk(blk: stim.Circuit) -> None:
        for item in blk:
            if isinstance(item, stim.CircuitRepeatBlock):
                body = item.body_copy()
                for _ in range(item.repeat_count):
                    _walk(body)
            elif item.tag.startswith("CLIFFORD_FAULT"):
                inline_faults.extend(parse_clifford_fault_instruction(item, len(flat_ops)))
                flat_ops.append(stim.CircuitInstruction("I", [], []))
            else:
                flat_ops.append(item)

    _walk(circuit)
    return flat_ops, tuple(inline_faults)


def _merge_and_validate_fault_channels(
    num_steps: int,
    inline_faults: Sequence[CliffordFaultChannel],
    external_faults: Sequence[CliffordFaultChannel] | None,
) -> tuple[
    dict[int, list[tuple[int, CliffordFaultChannel]]],
    dict[int, list[tuple[int, CliffordFaultChannel]]],
    tuple[CliffordFaultChannel, ...],
]:
    """Validate, chronologically sort `(step_index, inline_rank, reg_order)`, and assign `slot_id`s."""
    raw_entries: list[tuple[int, int, int, CliffordFaultChannel]] = []
    used_slot_ids: set[int] = set()
    reg_counter = 0

    def _register(channels: Sequence[CliffordFaultChannel], rank: int, max_step: int, label: str) -> None:
        nonlocal reg_counter
        for ch in channels:
            if not isinstance(ch, CliffordFaultChannel):
                raise ValueError(f"Expected CliffordFaultChannel, got {type(ch)!r}")
            if ch.step_index < 0 or ch.step_index > max_step:
                raise ValueError(f"{label} step_index={ch.step_index} out of range 0..{max_step}")
            if ch.slot_id is not None:
                if ch.slot_id in used_slot_ids:
                    raise ValueError(f"Duplicate CliffordFaultChannel slot_id={ch.slot_id}")
                used_slot_ids.add(ch.slot_id)
            raw_entries.append((ch.step_index, rank, reg_counter, ch))
            reg_counter += 1

    if external_faults is not None:
        _register(external_faults, 0, num_steps, "CliffordFaultChannel")
    if inline_faults:
        _register(inline_faults, 1, num_steps - 1, "Inline CLIFFORD_FAULT")

    raw_entries.sort(key=lambda item: (item[0], item[1], item[2]))
    prog_by_step: dict[int, list[tuple[int, CliffordFaultChannel]]] = {}
    inline_by_step: dict[int, list[tuple[int, CliffordFaultChannel]]] = {}
    all_channels: list[CliffordFaultChannel] = []
    next_auto_id = 0

    for slot_idx, (step_idx, rank, _, ch) in enumerate(raw_entries):
        if ch.slot_id is not None:
            sid = ch.slot_id
        else:
            while next_auto_id in used_slot_ids:
                next_auto_id += 1
            sid = next_auto_id
            used_slot_ids.add(sid)
            next_auto_id += 1
        norm_ch = CliffordFaultChannel(step_idx, ch.branches, sid)
        all_channels.append(norm_ch)
        (prog_by_step if rank == 0 else inline_by_step).setdefault(step_idx, []).append((slot_idx, norm_ch))

    return prog_by_step, inline_by_step, tuple(all_channels)


# ---------------------------------------------------------------------------
# Record Hierarchy & Frame Events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledCliffordFactorList:
    v: int
    p0: CompiledPauliAxis | None
    r_axes: tuple[CompiledPauliAxis, ...]


@dataclass(frozen=True)
class MagicPhaseTerm:
    axes: tuple[CompiledPauliAxis, ...]
    kind: Literal["T", "CCZ", "Z8_DIAG"]
    epsilon: int = 1


@dataclass(frozen=True)
class MagicTermRecord:
    term: MagicPhaseTerm


@dataclass(frozen=True)
class CompiledOpenBlockRecord:
    block: CompiledPauliBlock
    axes: tuple[CompiledPauliAxis, ...]


@dataclass(frozen=True)
class DirectFaultSlotRecord:
    slot_index: int
    channel: CliffordFaultChannel
    branch_factors: tuple[CompiledCliffordFactorList, ...]
    probs: NDArray[np.float64]


@dataclass(frozen=True)
class PauliNoiseRecord:
    kind: str
    probs: tuple[float, ...]
    branch_axes: tuple[CompiledPauliAxis, ...]
    noise_index: int


@dataclass(frozen=True)
class HeraldedNoiseRecord:
    kind: str
    meas_index: int
    probs: tuple[float, ...]
    branch_axes: tuple[CompiledPauliAxis | None, ...]
    noise_index: int


@dataclass(frozen=True)
class CorrelatedNoiseRecord:
    is_else: bool
    prob: float
    axis: CompiledPauliAxis
    noise_index: int


@dataclass(frozen=True)
class MeasurementRecord:
    meas_index: int
    flip_prob: float
    invert_result: bool
    is_random: bool
    obs_axis: CompiledPauliAxis
    rot_axis: CompiledPauliAxis | None


@dataclass(frozen=True)
class ResetRecord:
    meas_index: int | None
    flip_prob: float
    invert_result: bool
    is_random: bool
    obs_axis: CompiledPauliAxis
    rot_axis: CompiledPauliAxis | None
    reset_flip_ref: bool
    reset_kick_axis: CompiledPauliAxis


@dataclass(frozen=True)
class MPadRecord:
    meas_index: int
    pad_bit: bool
    flip_prob: float = 0.0


@dataclass(frozen=True)
class FeedbackPauliRecord:
    source_kind: Literal["rec", "sweep"]
    source_index: int
    current_meas_count: int
    axis: CompiledPauliAxis


@dataclass(frozen=True)
class LeakageBarrierRecord:
    instruction: stim.CircuitInstruction
    step_index: int
    tag_prefix: str
    meas_index_start: int = 0
    num_measurements: int = 0
    params: LeakageParams | None = None


CompiledRecord: TypeAlias = (
    MagicTermRecord | CompiledOpenBlockRecord | DirectFaultSlotRecord
    | PauliNoiseRecord | HeraldedNoiseRecord | CorrelatedNoiseRecord
    | MeasurementRecord | ResetRecord | MPadRecord
    | FeedbackPauliRecord | LeakageBarrierRecord
)


@dataclass(frozen=True)
class LeftUnitaryEvent:
    gate_name: str
    targets: tuple[int, ...]


@dataclass(frozen=True)
class LeftSppEvent:
    terms: tuple[tuple[int, str], ...]
    negate: bool
    dag: bool


@dataclass(frozen=True)
class LeftResetXEvent:
    qubit: int


@dataclass(frozen=True)
class RightRotEvent:
    rot_axis: CompiledPauliAxis


FrameEvent: TypeAlias = LeftUnitaryEvent | LeftSppEvent | LeftResetXEvent | RightRotEvent


@dataclass
class CoordinateCompilation:
    num_qubits: int
    num_measurements: int
    raw_records: tuple[CompiledRecord, ...]
    final_tableau: ReferenceCHPTableau
    frame_events: tuple[FrameEvent, ...]
    fault_channels: tuple[CliffordFaultChannel, ...]
    noise_channel_max_choices: tuple[int, ...] = ()
    num_resets: int = 0


# ---------------------------------------------------------------------------
# ReferenceCHPTableau Helpers, C1 & C1.5
# ---------------------------------------------------------------------------


def clone_reference_chp(ref_chp: ReferenceCHPTableau) -> ReferenceCHPTableau:
    """Deep-copy `ReferenceCHPTableau`."""
    return ref_chp.copy()


def compile_pauli_product(
    ref_chp: ReferenceCHPTableau,
    terms: Sequence[tuple[int, str]],
    *,
    negate: bool = False,
) -> CompiledPauliAxis:
    """Pull back a physical Pauli product `(-1)^negate prod (q, P)` through `ref_chp`."""
    n_w = ref_chp.n_words
    a_acc = np.zeros(n_w, dtype=np.uint64)
    b_acc = np.zeros(n_w, dtype=np.uint64)
    kap_acc = 2 if negate else 0
    for q, pchar in terms:
        if pchar == "I":
            continue
        if q < 0 or q >= ref_chp.num_qubits:
            raise ValueError(f"Qubit {q} out of range for num_qubits={ref_chp.num_qubits}")
        aw, bw, kw = ref_chp.pauli_image(pchar, q)
        a_acc, b_acc, kap_acc = pauli_multiply(a_acc, b_acc, kap_acc, aw, bw, kw)
    kap_acc = validate_hermitian_pauli(a_acc, b_acc, kap_acc)
    return CompiledPauliAxis(np.ascontiguousarray(a_acc), np.ascontiguousarray(b_acc), kap_acc)


def _compile_clifford_factors(
    ref_chp: ReferenceCHPTableau,
    gate_name: str,
    targets: Sequence[int],
) -> CompiledCliffordFactorList:
    dec = decompose_clifford_exact(gate_name, targets)
    p0_neg, p0_terms = _spec_terms(dec.p0_spec, targets)
    p0_axis = None if (not p0_terms and not p0_neg) else compile_pauli_product(ref_chp, p0_terms, negate=p0_neg)
    r_axes = tuple(
        compile_pauli_product(ref_chp, r_terms, negate=r_neg)
        for r_neg, r_terms in (_spec_terms(r_sp, targets) for r_sp in dec.r_specs)
    )
    return CompiledCliffordFactorList(v=dec.v, p0=p0_axis, r_axes=r_axes)


def _split_pauli_product_targets(
    op: stim.CircuitInstruction,
) -> list[tuple[list[tuple[int, str]], bool]]:
    """Split combiner-separated targets of `MPP`/`SPP`/`SPP_DAG`/`CORRELATED_ERROR`."""
    targs = op.targets_copy()
    if not targs:
        return []
    groups: list[tuple[list[tuple[int, str]], bool]] = []
    cur_terms: list[tuple[int, str]] = []
    cur_inv = False
    expect_pauli = True
    for t in targs:
        if t.is_combiner:
            if expect_pauli:
                raise ValueError(f"Invalid combiner placement in {op!r}")
            expect_pauli = True
            continue
        if not expect_pauli and cur_terms:
            groups.append((cur_terms, cur_inv))
            cur_terms, cur_inv = [], False
        pch = "X" if t.is_x_target else ("Y" if t.is_y_target else ("Z" if t.is_z_target else ""))
        if not pch:
            raise ValueError(f"Expected Pauli target in {op.name!r}, got {t!r}")
        cur_terms.append((int(t.qubit_value), pch))
        if t.is_inverted_result_target:
            cur_inv = not cur_inv
        expect_pauli = False
    if expect_pauli:
        raise ValueError(f"Trailing combiner in {op!r}")
    if cur_terms:
        groups.append((cur_terms, cur_inv))
    return groups


def _reduce_pauli_product_targets(
    raw_targets: Sequence[tuple[int, str]],
) -> list[tuple[int, str]]:
    """Reduce `(qubit, pauli)` targets modulo global phase into unique unsigned Hermitian Paulis."""
    phys_map: dict[int, tuple[int, int]] = {}
    for q_raw, pch_raw in raw_targets:
        if pch_raw not in _PAU_ABK:
            raise ValueError(f"Expected Pauli character in ('I','X','Y','Z'), got {pch_raw!r}")
        q_c = int(q_raw)
        xc, zc = phys_map.get(q_c, (0, 0))
        dx, dz, _ = _PAU_ABK[pch_raw]
        phys_map[q_c] = (xc ^ dx, zc ^ dz)
    return [(q_c, "IZXY"[(xc << 1) | zc]) for q_c, (xc, zc) in phys_map.items() if (xc or zc)]


def apply_spp_to_reference_chp(
    ref_chp: ReferenceCHPTableau,
    terms: Sequence[tuple[int, str]],
    *,
    negate: bool = False,
    dag: bool = False,
) -> None:
    """Update `ref_chp` in `O(|supp| * n_words)` under `SPP` / `SPP_DAG`."""
    ref_chp.do_spp(terms, negate=negate, dag=dag)


def _apply_right_rot_to_ref_chp(
    ref_chp: ReferenceCHPTableau, rot_axis: CompiledPauliAxis
) -> None:
    """Apply `B_t <- B_t * R(-rot_axis)` in-place via vectorized rank-1 transvection."""
    ref_chp.apply_right_rot(rot_axis.a_words, rot_axis.b_words, (rot_axis.kappa + 1) & 3)


def _compile_pauli_measurement_c15(
    ref_chp: ReferenceCHPTableau,
    terms: Sequence[tuple[int, str]],
    *,
    invert_result: bool,
    flip_prob: float,
    meas_index: int,
    frame_events: list[FrameEvent] | None = None,
) -> MeasurementRecord:
    bar_M = compile_pauli_product(ref_chp, terms, negate=False)
    if not words_any_nonzero(bar_M.a_words):
        return MeasurementRecord(meas_index, flip_prob, invert_result, False, bar_M, None)
    e_p = unit_bit_words(first_set_bit_in_words(bar_M.a_words), ref_chp.n_words)
    rot_axis = CompiledPauliAxis(bar_M.a_words.copy(), np.ascontiguousarray(bar_M.b_words ^ e_p), (bar_M.kappa + 1) & 3)
    _apply_right_rot_to_ref_chp(ref_chp, rot_axis)
    if frame_events is not None:
        frame_events.append(RightRotEvent(rot_axis=rot_axis))
    return MeasurementRecord(meas_index, flip_prob, invert_result, True, bar_M, rot_axis)


def _compile_reset_c15(
    ref_chp: ReferenceCHPTableau,
    q: int,
    basis: Literal["Z", "X", "Y"],
    *,
    meas_index: int | None,
    invert_result: bool,
    flip_prob: float,
    frame_events: list[FrameEvent] | None = None,
) -> ResetRecord:
    conj_gate = "H" if basis == "X" else ("H_YZ" if basis == "Y" else None)
    if conj_gate is not None:
        ref_chp.do_unitary(conj_gate, [q])
        if frame_events is not None:
            frame_events.append(LeftUnitaryEvent(conj_gate, (q,)))

    obs_axis = compile_pauli_product(ref_chp, [(q, "Z")], negate=False)
    meta = ref_chp.measure_z(q, is_reset=True, m_ref_override=0)
    if isinstance(meta, RandMeasMeta):
        rot_axis: CompiledPauliAxis | None = CompiledPauliAxis(
            np.ascontiguousarray(meta.a_words.copy()),
            np.ascontiguousarray(meta.b_rot_words.copy()),
            int(meta.kappa_rot) & 3,
        )
        if frame_events is not None:
            frame_events.append(RightRotEvent(rot_axis=rot_axis))
    else:
        assert isinstance(meta, DetMeasMeta)
        rot_axis = None
    reset_flip_ref = bool(meta.reset_flip_ref)
    if reset_flip_ref and frame_events is not None:
        frame_events.append(LeftResetXEvent(qubit=q))

    kick_axis = compile_pauli_product(ref_chp, [(q, "X")], negate=False)
    if conj_gate is not None:
        ref_chp.do_unitary(conj_gate, [q])
        if frame_events is not None:
            frame_events.append(LeftUnitaryEvent(conj_gate, (q,)))

    return ResetRecord(
        meas_index, flip_prob, invert_result, rot_axis is not None,
        obs_axis, rot_axis, reset_flip_ref, kick_axis,
    )


_ANNOTATION_GATES: frozenset[str] = frozenset(
    {"TICK", "QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"}
)
_FEEDBACK_2Q_GATES: dict[str, tuple[str, str]] = {
    "CX": ("Z", "X"), "CY": ("Z", "Y"), "CZ": ("Z", "Z"), "XCZ": ("X", "Z"), "YCZ": ("Y", "Z"),
}
_PAULI_2Q_ORDER: tuple[str, ...] = tuple(
    p0 + p1 for p0 in "IXYZ" for p1 in "IXYZ" if not (p0 == "I" and p1 == "I")
)
_CAT_NOISE_META: dict[str, tuple[int, int]] = {
    "X_ERROR": (1, 1), "Y_ERROR": (1, 1), "Z_ERROR": (1, 1),
    "DEPOLARIZE1": (1, 3), "PAULI_CHANNEL_1": (1, 3),
    "DEPOLARIZE2": (2, 15), "PAULI_CHANNEL_2": (2, 15),
}


def _validate_and_emit_magic_op(
    op: stim.CircuitInstruction,
    ref_chp: ReferenceCHPTableau,
    out_records: list[CompiledRecord],
) -> None:
    tag = op.tag
    if tag not in ("T", "T_DAG", "CCZ") or op.name != "I" or op.gate_args_copy():
        raise ValueError(f"Invalid magic carrier/tag/args on {op!r}")
    qs = _plain_qubits(op.targets_copy(), f"Magic gate I[{tag}]")
    if tag in ("T", "T_DAG"):
        eps = 1 if tag == "T" else -1
        for q in qs:
            out_records.append(MagicTermRecord(MagicPhaseTerm((compile_pauli_product(ref_chp, [(q, "Z")]),), "T", eps)))
    else:
        if (len(qs) % 3) != 0:
            raise ValueError(f"I[CCZ] requires a multiple of 3 targets, got {len(qs)}")
        for off in range(0, len(qs), 3):
            triple = (qs[off], qs[off + 1], qs[off + 2])
            if len(set(triple)) != 3:
                raise ValueError(f"I[CCZ] targets in each triple must be distinct, got {triple}")
            axes = tuple(compile_pauli_product(ref_chp, [(q, "Z")]) for q in triple)
            out_records.append(MagicTermRecord(MagicPhaseTerm(axes=axes, kind="CCZ")))


def _infer_total_data_qubits(
    circuit: stim.Circuit, all_channels: Sequence[CliffordFaultChannel]
) -> int:
    return max(
        [max(1, int(circuit.num_qubits))]
        + [q + 1 for ch in all_channels for b in ch.branches for q in b.targets]
    )


def _instruction_num_measurements(op: stim.CircuitInstruction, canon_name: str) -> int:
    if not _canonical_gate_data(canon_name).produces_measurements:
        return 0
    if canon_name == "MPP":
        return len(_split_pauli_product_targets(op))
    n_t = len(op.targets_copy())
    return (n_t // 2) if canon_name in ("MXX", "MYY", "MZZ") else n_t


def _validate_and_emit_non_magic_tagged_op(
    op: stim.CircuitInstruction,
    step_index: int,
    num_qubits: int,
    next_meas_idx: int,
    out_records: list[CompiledRecord],
) -> int:
    """Validate a registered leakage tag (emitting `LeakageBarrierRecord`) or reject unknown tags."""
    tag = op.tag
    if not (tag.startswith("LEAKAGE") or tag.startswith("CONDITIONED_ON")):
        raise ValueError(f"Unrecognized instruction tag {tag!r} on {op!r}")
    params, _ = parse_leakage_tag_magic(op, syntax_only=True)
    tag_prefix = tag.split(":", 1)[0].strip()
    if params is not None:
        check_tag_qubits({op: params}, num_qubits)
        tag_targets = getattr(params, "targets", None)
        if tag_targets and tag_prefix in ("LEAKAGE_MEASUREMENT", "CONDITIONED_ON_OTHER"):
            raw_targs = op.targets_copy()
            if len(raw_targs) == 0 or (len(raw_targs) % len(tag_targets)) != 0:
                raise ValueError(
                    f"{op.name}[{tag_prefix}] target count {len(raw_targs)} not a positive multiple "
                    f"of tag target count {len(tag_targets)} in {op!r}"
                )

    canon_name = _canonical_gate_data(op.name).name
    n_meas = _instruction_num_measurements(op, canon_name)
    out_records.append(
        LeakageBarrierRecord(op, step_index, tag_prefix, next_meas_idx, n_meas, params)
    )
    return next_meas_idx + n_meas


def _slice_axis_to_qubits(ax: CompiledPauliAxis, n_qubits: int) -> CompiledPauliAxis:
    n_w, tm = n_words_for_qubits(n_qubits), tail_mask(n_qubits)
    a_d = np.ascontiguousarray(ax.a_words[:n_w].copy(), dtype=np.uint64)
    b_d = np.ascontiguousarray(ax.b_words[:n_w].copy(), dtype=np.uint64)
    a_d[-1] &= tm
    b_d[-1] &= tm
    return CompiledPauliAxis(a_words=a_d, b_words=b_d, kappa=ax.kappa)


def _restrict_tableau_to_data(
    ref_sel: ReferenceCHPTableau, n_data: int, sel_bits: NDArray[np.bool_] | None = None
) -> ReferenceCHPTableau:
    """Slice `ref_sel` onto `0 .. n_data - 1` conditioned on `|sel_bits>` (`|0>` if `None`)."""
    data_tab = ReferenceCHPTableau(n_data)
    n_w_data, tm_d = data_tab.n_words, tail_mask(n_data)
    for dst, src in (
        (data_tab.x2x, ref_sel.x2x), (data_tab.x2z, ref_sel.x2z),
        (data_tab.z2x, ref_sel.z2x), (data_tab.z2z, ref_sel.z2z),
    ):
        dst[:] = src[:n_data, :n_w_data]
        dst[:, -1] &= tm_d
    if sel_bits is not None and np.any(sel_bits):
        sel_mask = np.zeros(ref_sel.n_words, dtype=np.uint64)
        for j in np.flatnonzero(sel_bits):
            q_s = n_data + int(j)
            sel_mask[q_s >> 6] |= np.uint64(1) << np.uint64(q_s & 63)
        px_par = word_dot_parity_batch(ref_sel.x2z[:n_data], sel_mask)
        pz_par = word_dot_parity_batch(ref_sel.z2z[:n_data], sel_mask)
        data_tab.x_kap[:] = (ref_sel.x_kap[:n_data] + (px_par << np.uint8(1))) & np.uint8(3)
        data_tab.z_kap[:] = (ref_sel.z_kap[:n_data] + (pz_par << np.uint8(1))) & np.uint8(3)
    else:
        data_tab.x_kap[:] = ref_sel.x_kap[:n_data]
        data_tab.z_kap[:] = ref_sel.z_kap[:n_data]
    return data_tab


def _count_gadget_pauli_selectors(flat_ops: Sequence[stim.CircuitInstruction]) -> int:
    total = 0
    for op in flat_ops:
        if op.tag != "":
            continue
        cname = _canonical_gate_data(op.name).name
        if cname in _CAT_NOISE_META:
            arity, k_br = _CAT_NOISE_META[cname]
            total += k_br * (len(op.targets_copy()) // arity)
        elif cname in ("E", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR"):
            total += 1
    return total


def _emit_selector_pauli_branch(
    s_qubit: int,
    pauli_terms: Sequence[tuple[int, str]],
    n_data: int,
    ref_sel: ReferenceCHPTableau,
    frame_events: list[FrameEvent],
    gauge_steps: list[tuple[int, CompiledPauliAxis]],
) -> None:
    """Synthesize `C_s(P_data)` on `ref_sel` and record data-qubit gauge step."""
    for q_t, pch in pauli_terms:
        if pch != "I":
            gauge_steps.append((s_qubit, _slice_axis_to_qubits(compile_pauli_product(ref_sel, [(q_t, pch)]), n_data)))
            c_gate = "C" + pch
            ref_sel.do_unitary(c_gate, [s_qubit, q_t])
            frame_events.append(LeftUnitaryEvent(c_gate, (s_qubit, q_t)))


def _emit_selector_clifford_branch(
    s_qubit: int,
    branch: FaultBranch,
    n_data: int,
    ref_sel: ReferenceCHPTableau,
    raw_records: list[CompiledRecord],
    frame_events: list[FrameEvent],
    gauge_steps: list[tuple[int, CompiledPauliAxis]],
) -> None:
    """Synthesize `C_s(F)` (`B1`), using 1-gate `CCZ` shortcut for all 9 2Q controlled Paulis."""
    g, qs = branch.gate_name, branch.targets
    if g in CONTROLLED_PAULI_2Q_AXES:
        ax0_ch, ax1_ch = CONTROLLED_PAULI_2Q_AXES[g]
        axes = (
            compile_pauli_product(ref_sel, [(s_qubit, "Z")]),
            compile_pauli_product(ref_sel, [(qs[0], ax0_ch)]),
            compile_pauli_product(ref_sel, [(qs[1], ax1_ch)]),
        )
        raw_records.append(MagicTermRecord(MagicPhaseTerm(axes=axes, kind="CCZ")))
        return

    dec = decompose_clifford_exact(g, qs)
    p0_neg, p0_terms = _spec_terms(dec.p0_spec, qs)
    v_eff = (dec.v + (4 if p0_neg else 0)) & 7
    _emit_selector_pauli_branch(s_qubit, p0_terms, n_data, ref_sel, frame_events, gauge_steps)

    for r_sp in dec.r_specs:
        neg_r, r_terms = _spec_terms(r_sp, qs)
        bar_P = compile_pauli_product(ref_sel, r_terms, negate=neg_r)
        bar_ZsP = compile_pauli_product(ref_sel, [(s_qubit, "Z")] + r_terms, negate=neg_r)
        raw_records.append(MagicTermRecord(MagicPhaseTerm(axes=(bar_ZsP,), kind="T", epsilon=1)))
        raw_records.append(MagicTermRecord(MagicPhaseTerm(axes=(bar_P,), kind="T", epsilon=-1)))

    if v_eff != 0:
        bar_Zs = compile_pauli_product(ref_sel, [(s_qubit, "Z")])
        kind: Literal["T", "Z8_DIAG"] = "T" if v_eff in (1, 7) else "Z8_DIAG"
        eps = (1 if v_eff == 1 else -1) if kind == "T" else v_eff
        raw_records.append(MagicTermRecord(MagicPhaseTerm(axes=(bar_Zs,), kind=kind, epsilon=eps)))


def _compile_circuit_pass(
    circuit: stim.Circuit,
    fault_channels: Sequence[CliffordFaultChannel] | None = None,
    *,
    selector_mode: bool = False,
    gadgetize_pauli_noise: bool = False,
) -> dict[str, Any]:
    """Shared single-pass lowering for `compile_coordinates`, `compile_direct`, and `compile_selector`."""
    flat_ops, inline_faults = extract_circuit_faults(circuit)
    prog_by_step, inline_by_step, all_channels = _merge_and_validate_fault_channels(
        len(flat_ops), inline_faults, fault_channels
    )
    n_data = _infer_total_data_qubits(circuit, all_channels)
    gad_noise = selector_mode and gadgetize_pauli_noise
    n_fault_sel = sum(len(ch.branches) for ch in all_channels) if selector_mode else 0
    n_sel = n_fault_sel + (_count_gadget_pauli_selectors(flat_ops) if gad_noise else 0)
    n_total = n_data + n_sel

    ref_chp = ReferenceCHPTableau(n_total)
    raw_records: list[CompiledRecord] = []
    frame_events: list[FrameEvent] = []
    gauge_steps: list[tuple[int, CompiledPauliAxis]] = []
    fault_sel_groups: list[tuple[int, ...]] = []
    noise_sel_groups: list[tuple[str, tuple[float, ...], tuple[int, ...]]] = []
    noise_sel_ch_indices: list[int] = []
    noise_max_choices: list[int] = []
    next_sel_q = n_data
    next_meas_idx = 0
    next_noise_idx = 0
    num_resets = 0

    def _emit_fault_slots(entries: Sequence[tuple[int, CliffordFaultChannel]]) -> None:
        nonlocal next_sel_q
        for slot_idx, ch in entries:
            if selector_mode:
                s_grp = tuple(range(next_sel_q, next_sel_q + len(ch.branches)))
                next_sel_q += len(ch.branches)
                fault_sel_groups.append(s_grp)
                for s_q, br in zip(s_grp, ch.branches):
                    _emit_selector_clifford_branch(
                        s_q, br, n_data, ref_chp, raw_records, frame_events, gauge_steps
                    )
            else:
                b_factors = tuple(_compile_clifford_factors(ref_chp, b.gate_name, b.targets) for b in ch.branches)
                raw_records.append(
                    DirectFaultSlotRecord(slot_idx, ch, b_factors, np.array([b.prob for b in ch.branches], dtype=np.float64))
                )

    def _emit_cat_pauli_noise(
        kind: str, probs_tup: tuple[float, ...], branch_term_groups: Sequence[Sequence[tuple[int, str]]]
    ) -> None:
        nonlocal next_sel_q, next_noise_idx
        k_br = len(probs_tup)
        if gad_noise:
            s_grp = tuple(range(next_sel_q, next_sel_q + k_br))
            next_sel_q += k_br
            noise_sel_groups.append(("CAT", probs_tup, s_grp))
            noise_sel_ch_indices.append(next_noise_idx)
            for s_q, b_terms in zip(s_grp, branch_term_groups):
                _emit_selector_pauli_branch(s_q, b_terms, n_data, ref_chp, frame_events, gauge_steps)
        else:
            b_axes = tuple(compile_pauli_product(ref_chp, b_terms) for b_terms in branch_term_groups)
            raw_records.append(PauliNoiseRecord(kind, probs_tup, b_axes, next_noise_idx))
        noise_max_choices.append(k_br)
        next_noise_idx += 1

    def _emit_measurements(groups: Sequence[tuple[Sequence[tuple[int, str]], bool]], p_flip: float) -> None:
        nonlocal next_meas_idx
        for grp_terms, grp_inv in groups:
            m_rec = _compile_pauli_measurement_c15(
                ref_chp, grp_terms, invert_result=grp_inv, flip_prob=p_flip,
                meas_index=next_meas_idx, frame_events=frame_events,
            )
            raw_records.append(m_rec)
            if selector_mode and m_rec.rot_axis is not None:
                gauge_steps.append((-1, _slice_axis_to_qubits(m_rec.rot_axis, n_data)))
            next_meas_idx += 1

    for step_idx, op in enumerate(flat_ops):
        if step_idx in prog_by_step:
            _emit_fault_slots(prog_by_step[step_idx])

        if op.tag != "":
            if op.tag in ("T", "T_DAG", "CCZ"):
                _validate_and_emit_magic_op(op, ref_chp, raw_records)
            else:
                next_meas_idx = _validate_and_emit_non_magic_tagged_op(
                    op, step_idx, n_data, next_meas_idx, raw_records
                )
        else:
            name = op.name
            canon_name = _canonical_gate_data(name).name
            args = op.gate_args_copy()
            targs = op.targets_copy()

            if canon_name in _ANNOTATION_GATES or canon_name in ("I_ERROR", "II_ERROR"):
                pass
            elif canon_name in ("SPP", "SPP_DAG"):
                dag = canon_name == "SPP_DAG"
                for grp_terms, grp_inv in _split_pauli_product_targets(op):
                    ref_chp.do_spp(grp_terms, negate=grp_inv, dag=dag)
                    frame_events.append(LeftSppEvent(tuple(grp_terms), grp_inv, dag))
            elif canon_name in _FEEDBACK_2Q_GATES and any(
                t.is_measurement_record_target or t.is_sweep_bit_target for t in targs
            ):
                ax0, ax1 = _FEEDBACK_2Q_GATES[canon_name]
                for off in range(0, len(targs), 2):
                    t0, t1 = targs[off], targs[off + 1]
                    q0_plain = t0.is_qubit_target and not t0.is_inverted_result_target
                    q1_plain = t1.is_qubit_target and not t1.is_inverted_result_target
                    if q0_plain and q1_plain:
                        q_pair = (int(t0.qubit_value), int(t1.qubit_value))
                        ref_chp.do_unitary(canon_name, list(q_pair))
                        frame_events.append(LeftUnitaryEvent(canon_name, q_pair))
                        continue
                    if (t0.is_measurement_record_target or t0.is_sweep_bit_target) and q1_plain and ax0 == "Z":
                        t_src, q_dst, p_dst = t0, int(t1.qubit_value), ax1
                    elif (t1.is_measurement_record_target or t1.is_sweep_bit_target) and q0_plain and ax1 == "Z":
                        t_src, q_dst, p_dst = t1, int(t0.qubit_value), ax0
                    else:
                        raise ValueError(f"Unsupported classical feedback target pair {(t0, t1)} on {name!r}")
                    src_kind: Literal["rec", "sweep"] = "rec" if t_src.is_measurement_record_target else "sweep"
                    raw_records.append(
                        FeedbackPauliRecord(
                            src_kind, int(t_src.value), next_meas_idx,
                            compile_pauli_product(ref_chp, [(q_dst, p_dst)]),
                        )
                    )
            elif canon_name in DECOMP_1Q or canon_name in DECOMP_2Q:
                arity = 1 if canon_name in DECOMP_1Q else 2
                qs_u = _plain_qubits(targs, name)
                if canon_name not in ("I", "II"):
                    for off in range(0, len(qs_u), arity):
                        grp_u = tuple(qs_u[off : off + arity])
                        ref_chp.do_unitary(canon_name, list(grp_u))
                        frame_events.append(LeftUnitaryEvent(canon_name, grp_u))
            elif canon_name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
                probs_1, pch = (float(args[0]),), canon_name[0]
                for t in targs:
                    _emit_cat_pauli_noise(canon_name, probs_1, ([(int(t.qubit_value), pch)],))
            elif canon_name in ("DEPOLARIZE1", "PAULI_CHANNEL_1"):
                probs_3 = (
                    (float(args[0]) / 3.0,) * 3 if canon_name == "DEPOLARIZE1"
                    else (float(args[0]), float(args[1]), float(args[2]))
                )
                for t in targs:
                    q = int(t.qubit_value)
                    _emit_cat_pauli_noise(canon_name, probs_3, tuple([(q, pch)] for pch in "XYZ"))
            elif canon_name in ("DEPOLARIZE2", "PAULI_CHANNEL_2"):
                probs_15 = (
                    (float(args[0]) / 15.0,) * 15 if canon_name == "DEPOLARIZE2"
                    else tuple(float(x) for x in args[:15])
                )
                for off in range(0, len(targs), 2):
                    q0, q1 = int(targs[off].qubit_value), int(targs[off + 1].qubit_value)
                    _emit_cat_pauli_noise(
                        canon_name, probs_15, tuple([(q0, pair[0]), (q1, pair[1])] for pair in _PAULI_2Q_ORDER)
                    )
            elif canon_name in ("HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"):
                probs_h = (
                    (float(args[0]) * 0.25,) * 4 if canon_name == "HERALDED_ERASE"
                    else (float(args[0]), float(args[1]), float(args[2]), float(args[3]))
                )
                for t in targs:
                    q = int(t.qubit_value)
                    b_axes: tuple[CompiledPauliAxis | None, ...] = (
                        None,
                        compile_pauli_product(ref_chp, [(q, "X")]),
                        compile_pauli_product(ref_chp, [(q, "Y")]),
                        compile_pauli_product(ref_chp, [(q, "Z")]),
                    )
                    raw_records.append(HeraldedNoiseRecord(canon_name, next_meas_idx, probs_h, b_axes, next_noise_idx))
                    noise_max_choices.append(4)
                    next_meas_idx += 1
                    next_noise_idx += 1
            elif canon_name in ("E", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR"):
                p, is_else = float(args[0]), (canon_name == "ELSE_CORRELATED_ERROR")
                simp_terms = _reduce_pauli_product_targets(
                    [tm for g_terms, _ in _split_pauli_product_targets(op) for tm in g_terms]
                )
                if gad_noise:
                    s_q = next_sel_q
                    next_sel_q += 1
                    noise_sel_groups.append(("ELSE_CORR" if is_else else "CORR", (p,), (s_q,)))
                    noise_sel_ch_indices.append(next_noise_idx)
                    _emit_selector_pauli_branch(s_q, simp_terms, n_data, ref_chp, frame_events, gauge_steps)
                else:
                    raw_records.append(
                        CorrelatedNoiseRecord(is_else, p, compile_pauli_product(ref_chp, simp_terms), next_noise_idx)
                    )
                noise_max_choices.append(1)
                next_noise_idx += 1
            elif canon_name in ("M", "MZ", "MX", "MY", "MXX", "MYY", "MZZ", "MPP"):
                p_flip = float(args[0]) if args else 0.0
                if canon_name == "MPP":
                    groups_m: Sequence[tuple[Sequence[tuple[int, str]], bool]] = _split_pauli_product_targets(op)
                elif canon_name in ("MXX", "MYY", "MZZ"):
                    pch = canon_name[1]
                    groups_m = [
                        (
                            [(int(targs[off].qubit_value), pch), (int(targs[off + 1].qubit_value), pch)],
                            bool(targs[off].is_inverted_result_target) ^ bool(targs[off + 1].is_inverted_result_target),
                        )
                        for off in range(0, len(targs), 2)
                    ]
                else:
                    pch = "Z" if canon_name in ("M", "MZ") else canon_name[1]
                    groups_m = [([(int(t.qubit_value), pch)], bool(t.is_inverted_result_target)) for t in targs]
                _emit_measurements(groups_m, p_flip)
            elif canon_name == "MPAD":
                p_flip = float(args[0]) if args else 0.0
                if not math.isfinite(p_flip) or p_flip < 0.0 or p_flip > 1.0:
                    raise ValueError(f"Invalid MPAD flip probability {p_flip!r}")
                for t in targs:
                    raw_records.append(MPadRecord(next_meas_idx, bool(t.value) ^ bool(t.is_inverted_result_target), p_flip))
                    next_meas_idx += 1
            elif canon_name in ("R", "RZ", "RX", "RY", "MR", "MRZ", "MRX", "MRY"):
                is_mr = canon_name.startswith("MR")
                p_flip = float(args[0]) if (is_mr and args) else 0.0
                basis_ch: Literal["Z", "X", "Y"] = (
                    "X" if canon_name.endswith("X") else ("Y" if canon_name.endswith("Y") else "Z")
                )
                for t in targs:
                    m_idx = next_meas_idx if is_mr else None
                    if is_mr:
                        next_meas_idx += 1
                    r_rec = _compile_reset_c15(
                        ref_chp, int(t.qubit_value), basis_ch, meas_index=m_idx,
                        invert_result=bool(t.is_inverted_result_target) if is_mr else False,
                        flip_prob=p_flip, frame_events=frame_events,
                    )
                    raw_records.append(r_rec)
                    num_resets += 1
                    if selector_mode and r_rec.rot_axis is not None:
                        gauge_steps.append((-1, _slice_axis_to_qubits(r_rec.rot_axis, n_data)))
            else:
                raise ValueError(f"Unsupported Stim instruction {op!r}")

        if step_idx in inline_by_step:
            _emit_fault_slots(inline_by_step[step_idx])

    if len(flat_ops) in prog_by_step:
        _emit_fault_slots(prog_by_step[len(flat_ops)])

    frame_tup = tuple(frame_events)
    out: dict[str, Any] = {
        "n_data": n_data, "n_sel": n_sel, "n_total": n_total, "num_measurements": next_meas_idx,
        "raw_records": tuple(raw_records), "final_tableau": ref_chp, "frame_events": frame_tup,
        "fault_channels": all_channels, "fault_sel_groups": tuple(fault_sel_groups),
        "noise_sel_groups": tuple(noise_sel_groups), "noise_sel_ch_indices": tuple(noise_sel_ch_indices),
        "total_noise_channels": next_noise_idx, "noise_max_choices": tuple(noise_max_choices),
        "num_resets": num_resets, "gauge_steps": tuple(gauge_steps),
    }
    if selector_mode:
        data_events = tuple(
            RightRotEvent(_slice_axis_to_qubits(ev.rot_axis, n_data)) if isinstance(ev, RightRotEvent) else ev
            for ev in frame_tup
            if not (isinstance(ev, LeftUnitaryEvent) and any(q >= n_data for q in ev.targets))
        )
        data_tab = _restrict_tableau_to_data(ref_chp, n_data)
        out["data_frame_events"] = data_events
        out["data_final_tableau"] = data_tab
    return out


def compile_coordinates(
    circuit: stim.Circuit,
    fault_channels: Sequence[CliffordFaultChannel] | None = None,
) -> CoordinateCompilation:
    """`C1` + `C1.5`: Compile circuit into moving-reference records and `ReferenceCHPTableau`."""
    res = _compile_circuit_pass(circuit, fault_channels=fault_channels, selector_mode=False)
    return CoordinateCompilation(
        res["n_data"], res["num_measurements"], res["raw_records"], res["final_tableau"],
        res["frame_events"], res["fault_channels"], res["noise_max_choices"], res["num_resets"],
    )


# ---------------------------------------------------------------------------
# C2: Commuting Diagonal Phase Open-Block Compiler (`compile_open_block`)
# ---------------------------------------------------------------------------

_DEG1_SEQ: dict[int, tuple[int, tuple[tuple[Literal["P1", "P2", "N1"], int], ...]]] = {
    1: (0, (("N1", 1),)),
    2: (1, (("P2", -1),)),
    3: (0, (("P1", 1), ("N1", -1))),
    4: (0, (("P1", 1),)),
    5: (0, (("P1", 1), ("N1", 1))),
    6: (7, (("P2", 1),)),
    7: (0, (("N1", -1),)),
}


def _axes_commute(ax1: CompiledPauliAxis, ax2: CompiledPauliAxis) -> bool:
    return word_dot_parity(ax1.a_words, ax2.b_words) == word_dot_parity(ax2.a_words, ax1.b_words)


def _canonical_unsigned_axis(ax: CompiledPauliAxis) -> tuple[CompiledPauliAxis, bool]:
    ab_par = word_dot_parity(ax.a_words, ax.b_words)
    diff = (ax.kappa - ab_par) & 3
    if diff == 0:
        return ax, False
    if diff == 2:
        return CompiledPauliAxis(ax.a_words, ax.b_words, ab_par), True
    raise ValueError(f"Non-Hermitian CompiledPauliAxis: {ax}")


def _monomial_to_sequential_steps(
    axes: Sequence[CompiledPauliAxis],
    mask: int,
    coeff_mod8: int,
) -> tuple[int, list[CompiledPauliBlockStep]]:
    """Convert one simplified Z_8 monomial `(mask, coeff_mod8)` into exact sequential steps."""
    bits = [u for u in range(len(axes)) if (mask >> u) & 1]
    deg, c = len(bits), (coeff_mod8 & 7)
    if c == 0 or deg == 0:
        return 0, []
    if deg == 1:
        ax = axes[bits[0]].as_tuple()
        dv, spec = _DEG1_SEQ[c]
        return dv, [CompiledPauliBlockStep(kind=kind, paulis=(ax,), sign=sgn) for kind, sgn in spec]
    if deg == 2 and c == 4:
        p0, p1 = axes[bits[0]], axes[bits[1]]
        p01 = pauli_multiply(p0.a_words, p0.b_words, p0.kappa, p1.a_words, p1.b_words, p1.kappa)
        return 1, [
            CompiledPauliBlockStep(kind="P2", paulis=(p1.as_tuple(),), sign=-1),
            CompiledPauliBlockStep(kind="P2", paulis=(p0.as_tuple(),), sign=-1),
            CompiledPauliBlockStep(kind="P2", paulis=(p01,), sign=1),
        ]
    if deg == 3 and c == 4:
        return 0, [CompiledPauliBlockStep(kind="N2", paulis=tuple(axes[u].as_tuple() for u in bits))]
    raise ValueError(f"Unsupported sequential monomial degree={deg}, coeff_mod8={c}")


def _walsh_expand_subblock(
    n_qubits: int,
    axes: Sequence[CompiledPauliAxis],
    poly_items: Sequence[tuple[int, int]],
    global_v_mod8: int,
) -> CompiledPauliBlock:
    """Compute exact 2^r Walsh-Hadamard Pauli sum for `r` commuting axes."""
    r = len(axes)
    n_w = n_words_for_qubits(n_qubits)
    if r == 0:
        return CompiledPauliBlock(
            n_qubits=n_qubits, mode="pauli_sum", global_scalar=OMEGA_POWERS[global_v_mod8 & 7],
            is_unitary=True, is_identity=True,
        )
    dim = 1 << r
    s_grid = np.arange(dim, dtype=np.uint64)
    exps = np.full(dim, global_v_mod8 & 7, dtype=np.uint8)
    for m_mask, c_val in poly_items:
        m_u64 = np.uint64(m_mask)
        active = (s_grid & m_u64) == m_u64
        exps[active] = (exps[active] + np.uint8(c_val & 7)) & np.uint8(7)

    vals = _OMEGA_ARR[exps].copy()
    step = 1
    while step < dim:
        vals_2d = vals.reshape(-1, 2 * step)
        u, v = vals_2d[:, :step].copy(), vals_2d[:, step:].copy()
        vals_2d[:, :step] = u + v
        vals_2d[:, step:] = u - v
        step <<= 1
    vals *= math.ldexp(1.0, -r)

    nz_masks = np.flatnonzero(np.abs(vals) > 1e-15)
    prod_a = np.zeros((dim, n_w), dtype=np.uint64)
    prod_b = np.zeros((dim, n_w), dtype=np.uint64)
    prod_k = np.zeros(dim, dtype=np.uint8)
    for bit_idx, ax in enumerate(axes):
        bit = 1 << bit_idx
        for prev in range(bit):
            a_p, b_p, k_p = pauli_multiply(
                prod_a[prev], prod_b[prev], int(prod_k[prev]), ax.a_words, ax.b_words, ax.kappa
            )
            prod_a[prev | bit], prod_b[prev | bit], prod_k[prev | bit] = a_p, b_p, np.uint8(k_p)

    return CompiledPauliBlock(
        n_qubits=n_qubits, mode="pauli_sum",
        coeffs=np.ascontiguousarray(vals[nz_masks], dtype=np.complex128),
        a_words=np.ascontiguousarray(prod_a[nz_masks], dtype=np.uint64),
        b_words=np.ascontiguousarray(prod_b[nz_masks], dtype=np.uint64),
        kappas=np.ascontiguousarray(prod_k[nz_masks], dtype=np.uint8),
        global_scalar=1.0 + 0.0j, is_unitary=True, is_identity=False,
    )


def compile_open_block(
    records: Sequence[CompiledRecord] | CoordinateCompilation,
    *,
    n_qubits: int | None = None,
    budget: int = 64,
    c2_max_rank: int = 12,
    c2_max_poly_terms: int = 64,
) -> tuple[CompiledRecord, ...]:
    """`C2`: Fuse consecutive pairwise-commuting `MagicTermRecord`s, never crossing barriers."""
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")

    if isinstance(records, CoordinateCompilation):
        if n_qubits is None:
            n_qubits = records.num_qubits
        rec_seq: Sequence[CompiledRecord] = records.raw_records
    else:
        rec_seq = records
    if n_qubits is None:
        raise ValueError("compile_open_block requires explicit n_qubits when records is not CoordinateCompilation")
    n_q = int(n_qubits)

    out_records: list[CompiledRecord] = []
    cur_axes: list[CompiledPauliAxis] = []
    cur_axis_map: dict[tuple[bytes, bytes], int] = {}
    cur_poly: dict[int, int] = {}
    cur_v_mod8: int = 0
    has_terms: bool = False

    def _reset_cur() -> None:
        nonlocal cur_v_mod8, has_terms
        cur_axes.clear()
        cur_axis_map.clear()
        cur_poly.clear()
        has_terms = False
        cur_v_mod8 = 0

    def _flush() -> None:
        if not has_terms and cur_v_mod8 == 0 and not cur_poly:
            return
        surviving = {m: (c & 7) for m, c in cur_poly.items() if (c & 7) != 0 and m != 0}
        v0 = cur_v_mod8 & 7
        if not surviving and v0 == 0:
            _reset_cur()
            return

        used_mask = 0
        for m in surviving:
            used_mask |= m
        old_to_new: dict[int, int] = {}
        pruned_axes: list[CompiledPauliAxis] = []
        for old_u, ax in enumerate(cur_axes):
            if (used_mask >> old_u) & 1:
                old_to_new[old_u] = len(pruned_axes)
                pruned_axes.append(ax)

        sorted_items = sorted(
            (sum(1 << new_u for old_u, new_u in old_to_new.items() if (m >> old_u) & 1), c)
            for m, c in surviving.items()
        )
        axes_tup = tuple(pruned_axes)
        r = len(axes_tup)
        block_obj: CompiledPauliBlock | None = None
        if r == 0:
            block_obj = CompiledPauliBlock(
                n_qubits=n_q, mode="pauli_sum", global_scalar=OMEGA_POWERS[v0],
                is_unitary=True, is_identity=True,
            )
        elif budget >= 2 and r <= c2_max_rank and (1 << r) <= max(4096, budget * 16):
            cand = _walsh_expand_subblock(n_q, axes_tup, sorted_items, v0)
            assert cand.coeffs is not None
            if len(cand.coeffs) <= budget:
                block_obj = cand

        if block_obj is None:
            v_seq = v0
            seq_steps: list[CompiledPauliBlockStep | CompiledPauliBlock] = []
            for m_item, c_item in sorted_items:
                dv, st_list = _monomial_to_sequential_steps(axes_tup, m_item, c_item)
                v_seq = (v_seq + dv) & 7
                seq_steps.extend(st_list)
            block_obj = CompiledPauliBlock(
                n_qubits=n_q, mode="sequential", steps=tuple(seq_steps),
                global_scalar=OMEGA_POWERS[v_seq], is_unitary=True,
                is_identity=(len(seq_steps) == 0),
            )

        out_records.append(CompiledOpenBlockRecord(block_obj, axes_tup))
        _reset_cur()

    for rec in rec_seq:
        if not isinstance(rec, MagicTermRecord):
            _flush()
            out_records.append(rec)
            continue

        term = rec.term
        norm_infos = [_canonical_unsigned_axis(ax) for ax in term.axes]
        for i in range(len(norm_infos)):
            for j in range(i + 1, len(norm_infos)):
                if not _axes_commute(norm_infos[i][0], norm_infos[j][0]):
                    raise ValueError("Non-commuting axes inside single MagicPhaseTerm")

        new_unique_keys: set[tuple[bytes, bytes]] = set()
        commutes_with_all = True
        for c_ax, _ in norm_infos:
            k_bytes = (c_ax.a_words.tobytes(), c_ax.b_words.tobytes())
            if k_bytes not in cur_axis_map:
                new_unique_keys.add(k_bytes)
                if any(not _axes_commute(c_ax, ex_ax) for ex_ax in cur_axes):
                    commutes_with_all = False
                    break

        if (
            not commutes_with_all
            or (len(cur_axes) + len(new_unique_keys)) > max(3, c2_max_rank)
            or (len(cur_poly) + 8) > max(8, c2_max_poly_terms)
        ):
            _flush()

        u_indices: list[int] = []
        neg_flags: list[bool] = []
        for c_ax, neg_f in norm_infos:
            k_bytes = (c_ax.a_words.tobytes(), c_ax.b_words.tobytes())
            u_idx = cur_axis_map.get(k_bytes)
            if u_idx is None:
                u_idx = len(cur_axes)
                cur_axes.append(c_ax)
                cur_axis_map[k_bytes] = u_idx
            u_indices.append(u_idx)
            neg_flags.append(neg_f)

        has_terms = True
        if term.kind in ("T", "Z8_DIAG"):
            m0, c_val = 1 << u_indices[0], int(term.epsilon) & 7
            if neg_flags[0]:
                cur_v_mod8 = (cur_v_mod8 + c_val) & 7
                cur_poly[m0] = (cur_poly.get(m0, 0) - c_val) & 7
            else:
                cur_poly[m0] = (cur_poly.get(m0, 0) + c_val) & 7
        elif term.kind == "CCZ":
            for sub in range(8):
                if any(not ((sub >> j) & 1) and not neg_flags[j] for j in range(3)):
                    continue
                m_acc = 0
                for j in range(3):
                    if (sub >> j) & 1:
                        m_acc |= 1 << u_indices[j]
                if m_acc == 0:
                    cur_v_mod8 = (cur_v_mod8 + 4) & 7
                else:
                    cur_poly[m_acc] = (cur_poly.get(m_acc, 0) + 4) & 7
        else:
            raise ValueError(f"Unknown MagicPhaseTerm kind: {term.kind!r}")

    _flush()
    return tuple(out_records)


# ---------------------------------------------------------------------------
# Plans & Trajectory Results (`DIRECT` and `SELECTOR`)
# ---------------------------------------------------------------------------


@dataclass
class DirectPlan:
    num_qubits: int
    num_measurements: int
    records: tuple[CompiledRecord, ...]
    final_tableau: ReferenceCHPTableau
    frame_events: tuple[FrameEvent, ...]
    fault_channels: tuple[CliffordFaultChannel, ...]
    raw_records: tuple[CompiledRecord, ...] = ()
    noise_channel_max_choices: tuple[int, ...] = ()
    num_resets: int = 0


@dataclass
class SelectorPlan:
    num_data_qubits: int
    num_selector_qubits: int
    num_total_qubits: int
    num_measurements: int
    records: tuple[CompiledRecord, ...]
    final_tableau: ReferenceCHPTableau
    data_final_tableau: ReferenceCHPTableau
    frame_events: tuple[FrameEvent, ...]
    data_frame_events: tuple[FrameEvent, ...]
    fault_channels: tuple[CliffordFaultChannel, ...]
    fault_selector_groups: tuple[tuple[int, ...], ...]
    noise_selector_groups: tuple[tuple[str, tuple[float, ...], tuple[int, ...]], ...] = ()
    gadgetize_pauli_noise: bool = False
    raw_records: tuple[CompiledRecord, ...] = ()
    noise_selector_channel_indices: tuple[int, ...] = ()
    total_noise_channels: int = 0
    noise_channel_max_choices: tuple[int, ...] = ()
    num_resets: int = 0
    _gauge_steps: tuple[tuple[int, CompiledPauliAxis], ...] = ()


@dataclass
class TrajectoryRunResult:
    expansion: ShotExpansion
    final_tableau: ReferenceCHPTableau
    measurements: NDArray[np.bool_]
    branch_mass: float
    log_branch_mass: float
    fault_choices: tuple[int, ...]
    pauli_noise_choices: tuple[int, ...] = ()
    reset_choices: tuple[int, ...] = ()
    quantum_outcomes: dict[int, int] = field(default_factory=dict)
    readout_flips: dict[int, bool] = field(default_factory=dict)
    born_masses: tuple[tuple[int | None, float, float, float], ...] = ()
    selector_bits: NDArray[np.bool_] | None = None
    peak_components: int = 1
    peak_dim: int = 0
    frame_events: tuple[FrameEvent, ...] | None = None


# ---------------------------------------------------------------------------
# Compilers, Strict Trace/Outcome Validators & Runners (`DIRECT` and `SELECTOR`)
# ---------------------------------------------------------------------------


def compile_direct(
    circuit: stim.Circuit,
    fault_channels: Sequence[CliffordFaultChannel] | None = None,
    *,
    c2_budget: int = 64,
    c2_max_rank: int = 12,
    c2_max_poly_terms: int = 64,
) -> DirectPlan:
    """`A1`: Compile `circuit` for zero-branching `DIRECT` trajectory execution."""
    coord = compile_coordinates(circuit, fault_channels=fault_channels)
    records = compile_open_block(
        coord.raw_records, n_qubits=coord.num_qubits,
        budget=c2_budget, c2_max_rank=c2_max_rank, c2_max_poly_terms=c2_max_poly_terms,
    )
    return DirectPlan(
        coord.num_qubits, coord.num_measurements, records, coord.final_tableau,
        coord.frame_events, coord.fault_channels, coord.raw_records,
        coord.noise_channel_max_choices, coord.num_resets,
    )


def compile_selector(
    circuit: stim.Circuit,
    fault_channels: Sequence[CliffordFaultChannel] | None = None,
    *,
    gadgetize_pauli_noise: bool = False,
    c2_budget: int = 64,
    c2_max_rank: int = 12,
    c2_max_poly_terms: int = 64,
) -> SelectorPlan:
    """`B1`: Compile `circuit` into private-selector-ancilla template (`SELECTOR` mode)."""
    res = _compile_circuit_pass(
        circuit, fault_channels=fault_channels, selector_mode=True, gadgetize_pauli_noise=gadgetize_pauli_noise
    )
    records = compile_open_block(
        res["raw_records"], n_qubits=res["n_total"],
        budget=c2_budget, c2_max_rank=c2_max_rank, c2_max_poly_terms=c2_max_poly_terms,
    )
    return SelectorPlan(
        res["n_data"], res["n_sel"], res["n_total"], res["num_measurements"], records,
        res["final_tableau"], res["data_final_tableau"], res["frame_events"], res["data_frame_events"],
        res["fault_channels"], res["fault_sel_groups"], res["noise_sel_groups"], gadgetize_pauli_noise,
        res["raw_records"], res["noise_sel_ch_indices"], res["total_noise_channels"],
        res["noise_max_choices"], res["num_resets"], res["gauge_steps"],
    )


def _sample_categorical_branch(probs: Sequence[float] | NDArray[np.float64], rng: np.random.Generator) -> int:
    """Return `0` with prob `1 - sum(probs)`, or `1..K` for branch `0..K-1`."""
    u = float(rng.random())
    acc = 0.0
    for idx, p in enumerate(probs):
        acc += float(p)
        if u < acc:
            return idx + 1
    return 0


def _normalize_choice_map(
    trace: object,
    max_choices: Sequence[int],
    name: str,
    *,
    key_to_slot: Mapping[int, int] | None = None,
    allow_none_zero: bool = False,
    allow_bool_values: bool = False,
    mapping_only: bool = False,
) -> dict[int, int] | None:
    """Validate integer keys/indices and choice bounds, returning slot-indexed `dict[int, int]`."""
    if trace is None:
        return None
    if isinstance(trace, Mapping):
        slot_lookup = key_to_slot if key_to_slot is not None else {i: i for i in range(len(max_choices))}
        out: dict[int, int] = {}
        for k, v in trace.items():
            ik = _coerce_strict_int(k, f"{name} key", allow_bool=False)
            if ik not in slot_lookup:
                raise ValueError(f"Out-of-range or unknown key {k!r} in {name}")
            slot = slot_lookup[ik]
            iv = 0 if (v is None and allow_none_zero) else _coerce_strict_int(v, f"{name}[{ik}]", allow_bool=allow_bool_values)
            if iv < 0 or iv > max_choices[slot]:
                raise ValueError(f"{name}[{ik}]={iv} out of range [0, {max_choices[slot]}]")
            out[slot] = iv
        return out
    if mapping_only or isinstance(trace, (str, bytes)) or not isinstance(trace, (Sequence, np.ndarray)):
        raise ValueError(f"Invalid {name} container type: {type(trace)!r}")
    if isinstance(trace, np.ndarray) and trace.ndim != 1:
        raise ValueError(f"{name} array must be 1D, got shape {trace.shape}")
    if len(trace) < len(max_choices):
        raise ValueError(f"{name} length {len(trace)} shorter than required {len(max_choices)}")
    out_seq: dict[int, int] = {}
    for idx, max_c in enumerate(max_choices):
        iv = _coerce_strict_int(trace[idx], f"{name}[{idx}]", allow_bool=allow_bool_values)
        if iv < 0 or iv > max_c:
            raise ValueError(f"{name}[{idx}]={iv} out of range [0, {max_c}]")
        out_seq[idx] = iv
    return out_seq


def _prepare_run_maps(
    plan: DirectPlan | SelectorPlan,
    *,
    fault_trace: object = None,
    pauli_noise_trace: object = None,
    reset_outcomes: object = None,
    requested_outcomes: object = None,
    requested_quantum_outcomes: object = None,
) -> tuple[
    dict[int, int] | None,
    dict[int, int] | None,
    dict[int, int],
    dict[int, int],
    dict[int, int],
]:
    fault_max = [len(ch.branches) for ch in plan.fault_channels]
    fault_id_to_slot = {
        (ch.slot_id if ch.slot_id is not None else idx): idx for idx, ch in enumerate(plan.fault_channels)
    }
    meas_bounds = (1,) * plan.num_measurements
    f_map = _normalize_choice_map(
        fault_trace, fault_max, "fault_trace", key_to_slot=fault_id_to_slot, allow_none_zero=True
    )
    n_map = _normalize_choice_map(pauli_noise_trace, plan.noise_channel_max_choices, "pauli_noise_trace")
    r_map = _normalize_choice_map(reset_outcomes, (1,) * plan.num_resets, "reset_outcomes", allow_bool_values=True)
    c_map = _normalize_choice_map(
        requested_outcomes, meas_bounds, "requested_outcomes", allow_bool_values=True, mapping_only=True
    )
    q_map = _normalize_choice_map(
        requested_quantum_outcomes, meas_bounds, "requested_quantum_outcomes",
        allow_bool_values=True, mapping_only=True,
    )
    return f_map, n_map, r_map or {}, c_map or {}, q_map or {}


def _measure_with_readout_and_postselection(
    exp: ShotExpansion,
    obs_axis: CompiledPauliAxis,
    *,
    flip_prob: float,
    invert_result: bool,
    rng: np.random.Generator,
    req_q: int | None,
    fixed_classical: int | None,
    zero_mass: bool,
    atol: float,
) -> tuple[ShotExpansion, int, bool, bool, tuple[float, float, float], bool]:
    """Unified exact Bayesian readout + postselection collapse for `M*`, `MPP`, `R*`, and `MR*`."""
    iota = 1 if invert_result else 0
    p = float(flip_prob)
    if not (math.isfinite(p) and 0.0 <= p <= 1.0):
        raise ValueError(f"Invalid measurement flip probability {flip_prob!r}")

    if zero_mass:
        exp.log_mass = -math.inf
        q_out = (
            req_q if req_q is not None
            else ((fixed_classical ^ iota ^ (1 if p == 1.0 else 0)) if fixed_classical is not None else 0)
        )
        rep_bit = bool(fixed_classical) if fixed_classical is not None else bool(q_out ^ iota)
        return exp, q_out, rep_bit, bool(int(rep_bit) ^ q_out ^ iota), (0.0, 0.0, 0.0), True

    if fixed_classical is None:
        m_res = _pq.measure_expansion(
            exp, obs_axis, rng=rng, requested_bit=req_q, p_flip=p, invert_result=invert_result,
            renormalize=True, keep_rotated_frame=True, atol=atol,
        )
        return (
            m_res.post_state, m_res.outcome, bool(m_res.reported_outcome), bool(m_res.readout_flipped),
            (m_res.z0, m_res.z1, m_res.input_norm_sq), False,
        )

    tol = validate_atol(atol)
    c = int(fixed_classical) & 1
    z_in, in_log_mass = exp.norm_sq(), exp.log_mass
    if not (math.isfinite(z_in) and z_in >= 0.0):
        raise FloatingPointError(f"Invalid pre-measurement norm squared z={z_in!r}")
    if exp.chi == 0 or z_in <= 0.0:
        if req_q is not None:
            raise ImpossiblePostselectionError("Cannot postselect quantum outcome on zero-norm state")
        exp.log_mass = -math.inf
        return exp, 0, bool(c), bool(c ^ iota), (0.0, 0.0, 0.0), True

    if words_any_nonzero(obs_axis.a_words):
        e_p_w = unit_bit_words(first_set_bit_in_words(obs_axis.a_words), exp.n_words)
        work_exp = _pq.apply_quarter_turn(
            exp, obs_axis.a_words, np.ascontiguousarray(obs_axis.b_words ^ e_p_w),
            (obs_axis.kappa + 1) & 3, sign=1, check_unitary=True, atol=tol,
        )
        b_diag, sig_diag = e_p_w, 0
    else:
        work_exp, b_diag, sig_diag = exp, obs_axis.b_words, (obs_axis.kappa >> 1) & 1

    exp0, z0_raw = _pq.project_diagonal(work_exp, b_diag, sig_diag, 0)
    exp1, z1_raw = _pq.project_diagonal(work_exp, b_diag, sig_diag, 1)

    scale_z = max(1.0, z_in)
    if not (math.isfinite(z0_raw) and math.isfinite(z1_raw)):
        raise FloatingPointError(f"Non-finite projected Born masses: z0={z0_raw!r}, z1={z1_raw!r}")
    if z0_raw < -1e-12 * scale_z or z1_raw < -1e-12 * scale_z:
        raise FloatingPointError(f"Significantly negative projected Born mass: z0={z0_raw!r}, z1={z1_raw!r}")
    z0, z1 = max(0.0, float(z0_raw)), max(0.0, float(z1_raw))
    if abs((z0 + z1) - z_in) > tol * scale_z:
        raise FloatingPointError(
            f"M2 Born mass conservation violated: z0={z0:.16e}, z1={z1:.16e}, z={z_in:.16e}, tol={tol:.3e}"
        )

    zero_floor = K1_CANCEL_REL_TOL * z_in
    if req_q is not None:
        req_z, req_exp = (z0, exp0) if req_q == 0 else (z1, exp1)
        if req_z <= zero_floor or req_exp.chi == 0:
            raise ImpossiblePostselectionError(
                f"Requested quantum outcome {req_q} has zero Born probability (z_{req_q}={req_z:.3e})"
            )

    m0 = 0.0 if ((req_q is not None and req_q != 0) or z0 <= zero_floor or exp0.chi == 0) else z0 * ((1.0 - p) if c == iota else p)
    m1 = 0.0 if ((req_q is not None and req_q != 1) or z1 <= zero_floor or exp1.chi == 0) else z1 * ((1.0 - p) if c == (1 ^ iota) else p)
    m_step = m0 + m1
    if m_step <= 1e-15 * scale_z:
        q_dead = req_q if req_q is not None else (c ^ iota ^ (1 if p == 1.0 else 0))
        exp.log_mass = -math.inf
        return exp, q_dead, bool(c), bool(c ^ q_dead ^ iota), (0.0, 0.0, 0.0), True

    q_out = 0 if m1 <= 0.0 else (1 if m0 <= 0.0 else int(float(rng.random()) < min(1.0, max(0.0, m1 / m_step))))
    chosen_z, chosen_exp = (z0, exp0) if q_out == 0 else (z1, exp1)
    chosen_exp.weights = np.ascontiguousarray(chosen_exp.weights * (1.0 / math.sqrt(chosen_z)))
    chosen_exp.log_mass = in_log_mass + math.log(m_step)
    return chosen_exp, q_out, bool(c), bool(c ^ q_out ^ iota), (z0, z1, z_in), False


def _execute_compiled_records(
    exp: ShotExpansion,
    records: Sequence[CompiledRecord],
    num_measurements: int,
    final_tableau: ReferenceCHPTableau,
    *,
    rng: np.random.Generator,
    fault_map: Mapping[int, int] | None,
    noise_map: Mapping[int, int] | None,
    reset_map: Mapping[int, int],
    req_c_map: Mapping[int, int],
    req_q_map: Mapping[int, int],
    sweep_bits: Sequence[int | bool] | NDArray[np.bool_] | None,
    max_components: int | None,
    atol: float,
    preset_fault_choices: tuple[int, ...] | None = None,
    preset_noise_map: Mapping[int, int] | None = None,
    total_noise_channels: int | None = None,
    selector_bits: NDArray[np.bool_] | None = None,
    frame_events: tuple[FrameEvent, ...] | None = None,
) -> TrajectoryRunResult:
    atol = validate_atol(atol)
    meas_record = np.zeros(num_measurements, dtype=np.bool_)
    chosen_faults: list[int] = []
    chosen_noise: list[int] = []
    recorded_noise_map: dict[int, int] = {}
    chosen_resets: list[int] = []
    quantum_outcomes: dict[int, int] = {}
    readout_flips: dict[int, bool] = {}
    born_masses: list[tuple[int | None, float, float, float]] = []
    reset_cursor = 0
    correlated_occurred, zero_mass = False, False
    peak_chi, peak_d = max(1, exp.chi), exp.d

    for rec in records:
        if isinstance(rec, CompiledOpenBlockRecord):
            if not zero_mass:
                exp = apply_compiled_block(exp, rec.block, atol=atol)
                peak_chi, peak_d = max(peak_chi, exp.chi), max(peak_d, exp.d)
                if max_components is not None and exp.chi > max_components:
                    raise RuntimeError(f"Trajectory exceeded max_components={max_components} (got chi={exp.chi})")

        elif isinstance(rec, DirectFaultSlotRecord):
            c_idx = fault_map.get(rec.slot_index, 0) if fault_map is not None else _sample_categorical_branch(rec.probs, rng)
            chosen_faults.append(c_idx)
            if c_idx > 0 and not zero_mass:
                fac = rec.branch_factors[c_idx - 1]
                if fac.p0 is not None:
                    exp = apply_pauli(exp, fac.p0)
                for r_ax in fac.r_axes:
                    exp = apply_quarter_turn(exp, r_ax, sign=1, atol=atol)
                if fac.v != 0 and exp.chi > 0:
                    exp.weights = np.ascontiguousarray(exp.weights * OMEGA_POWERS[fac.v])
                peak_chi, peak_d = max(peak_chi, exp.chi), max(peak_d, exp.d)

        elif isinstance(rec, PauliNoiseRecord):
            ch_idx = rec.noise_index
            c_n = noise_map.get(ch_idx, 0) if noise_map is not None else _sample_categorical_branch(rec.probs, rng)
            chosen_noise.append(c_n)
            recorded_noise_map[ch_idx] = c_n
            if c_n > 0 and not zero_mass:
                exp = apply_pauli(exp, rec.branch_axes[c_n - 1])

        elif isinstance(rec, HeraldedNoiseRecord):
            ch_idx = rec.noise_index
            m_idx = rec.meas_index
            p_herald = min(1.0, max(0.0, float(sum(rec.probs))))
            req_h = req_c_map.get(m_idx)

            if noise_map is not None:
                c_h = noise_map.get(ch_idx, 0)
                true_h = rep_h = c_h > 0
                if req_h is not None and rep_h != bool(req_h):
                    zero_mass, rep_h, exp.log_mass = True, bool(req_h), -math.inf
            elif req_h is not None:
                req_rep = bool(req_h)
                if not req_rep:
                    c_h, true_h = 0, False
                    p_req = max(0.0, 1.0 - p_herald)
                    if p_req <= 1e-15:
                        zero_mass, exp.log_mass = True, -math.inf
                    else:
                        exp.log_mass += math.log(p_req)
                elif p_herald <= 1e-15:
                    c_h, true_h, zero_mass, exp.log_mass = 0, False, True, -math.inf
                else:
                    true_h = True
                    exp.log_mass += math.log(p_herald)
                    u_cond = float(rng.random()) * p_herald
                    acc_h, c_h, last_pos = 0.0, 0, 1
                    for idx_p, pr_h in enumerate(rec.probs):
                        if pr_h > 0.0:
                            last_pos = idx_p + 1
                        acc_h += float(pr_h)
                        if u_cond < acc_h:
                            c_h = idx_p + 1
                            break
                    if c_h == 0:
                        c_h = last_pos
                rep_h = req_rep
            else:
                c_h = _sample_categorical_branch(rec.probs, rng)
                true_h = rep_h = c_h > 0

            chosen_noise.append(c_h)
            recorded_noise_map[ch_idx] = c_h
            meas_record[m_idx] = rep_h
            quantum_outcomes[m_idx] = int(true_h)
            readout_flips[m_idx] = False
            if c_h > 0 and rec.branch_axes[c_h - 1] is not None and not zero_mass:
                exp = apply_pauli(exp, rec.branch_axes[c_h - 1])

        elif isinstance(rec, CorrelatedNoiseRecord):
            ch_idx = rec.noise_index
            if not rec.is_else:
                correlated_occurred = False
            if correlated_occurred:
                c_corr = 0
            else:
                c_corr = (
                    noise_map.get(ch_idx, 0) if noise_map is not None
                    else int(float(rng.random()) < rec.prob)
                )
                if c_corr > 0:
                    correlated_occurred = True
                    if not zero_mass:
                        exp = apply_pauli(exp, rec.axis)
            chosen_noise.append(c_corr)
            recorded_noise_map[ch_idx] = c_corr

        elif isinstance(rec, MeasurementRecord):
            m_idx = rec.meas_index
            exp, q_out, rep_bit, flipped, born_tup, zero_mass = _measure_with_readout_and_postselection(
                exp, rec.obs_axis, flip_prob=rec.flip_prob, invert_result=rec.invert_result, rng=rng,
                req_q=req_q_map.get(m_idx), fixed_classical=req_c_map.get(m_idx),
                zero_mass=zero_mass, atol=atol,
            )
            meas_record[m_idx], quantum_outcomes[m_idx], readout_flips[m_idx] = rep_bit, q_out, flipped
            born_masses.append((m_idx, *born_tup))
            if not zero_mass:
                peak_chi, peak_d = max(peak_chi, exp.chi), max(peak_d, exp.d)

        elif isinstance(rec, ResetRecord):
            req_r, m_idx = reset_map.get(reset_cursor), rec.meas_index
            req_mq = req_q_map.get(m_idx) if m_idx is not None else None
            if req_r is not None and req_mq is not None and req_r != req_mq and not zero_mass:
                raise ImpossiblePostselectionError(
                    f"Conflicting reset_outcomes[{reset_cursor}]={req_r} and "
                    f"requested_quantum_outcomes[{m_idx}]={req_mq}"
                )
            exp, q_out, rep_bit, flipped, born_tup, zero_mass = _measure_with_readout_and_postselection(
                exp, rec.obs_axis, flip_prob=0.0 if m_idx is None else rec.flip_prob,
                invert_result=rec.invert_result, rng=rng,
                req_q=req_r if req_r is not None else req_mq,
                fixed_classical=req_c_map.get(m_idx) if m_idx is not None else None,
                zero_mass=zero_mass, atol=atol,
            )
            chosen_resets.append(q_out)
            reset_cursor += 1
            if not zero_mass and (bool(q_out) != rec.reset_flip_ref):
                exp = apply_pauli(exp, rec.reset_kick_axis)
            if m_idx is not None:
                meas_record[m_idx], quantum_outcomes[m_idx], readout_flips[m_idx] = rep_bit, q_out, flipped
            born_masses.append((m_idx, *born_tup))
            if not zero_mass:
                peak_chi, peak_d = max(peak_chi, exp.chi), max(peak_d, exp.d)

        elif isinstance(rec, MPadRecord):
            m_idx, pad_b, p_flip = rec.meas_index, bool(rec.pad_bit), float(rec.flip_prob)
            p1_pad = (1.0 - p_flip) if pad_b else p_flip
            req_pad = req_c_map.get(m_idx)
            if req_pad is not None:
                rep_pad = bool(req_pad)
                flipped_pad = rep_pad ^ pad_b
                p_req = p1_pad if rep_pad else (1.0 - p1_pad)
                if p_req <= 1e-15:
                    zero_mass, exp.log_mass = True, -math.inf
                else:
                    exp.log_mass += math.log(p_req)
            else:
                flipped_pad = bool(float(rng.random()) < p_flip) if p_flip > 0.0 else False
                rep_pad = pad_b ^ flipped_pad
            meas_record[m_idx], quantum_outcomes[m_idx], readout_flips[m_idx] = rep_pad, int(pad_b), flipped_pad

        elif isinstance(rec, FeedbackPauliRecord):
            if rec.source_kind == "rec":
                idx = rec.current_meas_count + rec.source_index
                if idx < 0 or idx >= rec.current_meas_count:
                    raise ValueError(
                        f"Feedback rec[{rec.source_index}] out of bounds at meas_count={rec.current_meas_count}"
                    )
                cond = bool(meas_record[idx])
            else:
                cond = bool(sweep_bits[rec.source_index]) if (sweep_bits is not None and 0 <= rec.source_index < len(sweep_bits)) else False
            if cond and not zero_mass:
                exp = apply_pauli(exp, rec.axis)

        elif isinstance(rec, LeakageBarrierRecord):
            raise ValueError(
                f"Cannot execute leakage-tagged instruction {rec.instruction!r} in standalone "
                "trajectory runner; execute via MagicsideSimulator with LeakageUint8Magic."
            )
        else:
            raise ValueError(f"Unexpected record during trajectory run: {rec!r}")

    if zero_mass:
        exp = ShotExpansion.from_components(
            [], n_qubits=exp.n_qubits, merge=False, atol=atol
        )
        exp.log_mass = -math.inf

    if preset_noise_map is not None:
        merged_noise = {**preset_noise_map, **recorded_noise_map}
        n_ch = total_noise_channels if total_noise_channels is not None else (max(merged_noise) + 1 if merged_noise else 0)
        final_noise_choices = tuple(int(merged_noise.get(i, 0)) for i in range(n_ch))
    else:
        final_noise_choices = tuple(chosen_noise)

    return TrajectoryRunResult(
        expansion=exp,
        final_tableau=final_tableau,
        measurements=meas_record,
        branch_mass=exp.branch_mass(),
        log_branch_mass=exp.log_branch_mass(),
        fault_choices=preset_fault_choices if preset_fault_choices is not None else tuple(chosen_faults),
        pauli_noise_choices=final_noise_choices,
        reset_choices=tuple(chosen_resets),
        quantum_outcomes=quantum_outcomes,
        readout_flips=readout_flips,
        born_masses=tuple(born_masses),
        selector_bits=selector_bits,
        peak_components=peak_chi,
        peak_dim=peak_d,
        frame_events=frame_events,
    )


def run_direct(
    plan: DirectPlan,
    *,
    rng: np.random.Generator | int | None = None,
    fault_trace: Sequence[int] | Mapping[int, int] | None = None,
    pauli_noise_trace: Sequence[int] | Mapping[int, int] | None = None,
    reset_outcomes: Sequence[int] | Mapping[int, int] | None = None,
    requested_outcomes: Mapping[int, int | bool] | None = None,
    requested_quantum_outcomes: Mapping[int, int | bool] | None = None,
    sweep_bits: Sequence[int | bool] | NDArray[np.bool_] | None = None,
    max_components: int | None = None,
    atol: float = DEFAULT_ATOL,
) -> TrajectoryRunResult:
    """`A2`: Execute a `DirectPlan` trajectory."""
    f_map, n_map, r_map, c_map, q_map = _prepare_run_maps(
        plan, fault_trace=fault_trace, pauli_noise_trace=pauli_noise_trace,
        reset_outcomes=reset_outcomes, requested_outcomes=requested_outcomes,
        requested_quantum_outcomes=requested_quantum_outcomes,
    )
    rng_obj = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    exp = ShotExpansion.zero(plan.num_qubits, atol=atol)
    return _execute_compiled_records(
        exp, plan.records, plan.num_measurements, plan.final_tableau,
        rng=rng_obj, fault_map=f_map, noise_map=n_map, reset_map=r_map,
        req_c_map=c_map, req_q_map=q_map, sweep_bits=sweep_bits,
        max_components=max_components, atol=atol,
        frame_events=plan.frame_events,
    )


def _sample_selector_activations_from_maps(
    plan: SelectorPlan,
    rng_obj: np.random.Generator,
    fault_map: Mapping[int, int] | None,
    noise_map: Mapping[int, int] | None,
) -> tuple[NDArray[np.bool_], tuple[int, ...], dict[int, int]]:
    sel_bits = np.zeros(plan.num_selector_qubits, dtype=np.bool_)
    fault_choices: list[int] = []
    for slot_idx, (ch, s_grp) in enumerate(zip(plan.fault_channels, plan.fault_selector_groups)):
        c_val = (
            fault_map.get(slot_idx, 0) if fault_map is not None
            else _sample_categorical_branch([b.prob for b in ch.branches], rng_obj)
        )
        fault_choices.append(c_val)
        if c_val > 0:
            sel_bits[s_grp[c_val - 1] - plan.num_data_qubits] = True

    noise_choice_map: dict[int, int] = {}
    if plan.gadgetize_pauli_noise and plan.noise_selector_groups:
        corr_active = False
        for (g_kind, probs_g, s_grp), ch_idx in zip(plan.noise_selector_groups, plan.noise_selector_channel_indices):
            if g_kind == "CAT":
                c_n = noise_map.get(ch_idx, 0) if noise_map is not None else _sample_categorical_branch(probs_g, rng_obj)
                noise_choice_map[ch_idx] = c_n
                if c_n > 0:
                    sel_bits[s_grp[c_n - 1] - plan.num_data_qubits] = True
            else:
                if g_kind == "CORR":
                    corr_active = False
                if corr_active:
                    c_c = 0
                else:
                    c_c = (
                        noise_map.get(ch_idx, 0) if noise_map is not None
                        else int(float(rng_obj.random()) < probs_g[0])
                    )
                    if c_c > 0:
                        corr_active = True
                        sel_bits[s_grp[0] - plan.num_data_qubits] = True
                noise_choice_map[ch_idx] = c_c

    return sel_bits, tuple(fault_choices), noise_choice_map


def sample_selector_activations(
    plan: SelectorPlan,
    rng: np.random.Generator | int | None = None,
    *,
    fault_trace: Sequence[int] | Mapping[int, int] | None = None,
    pauli_noise_trace: Sequence[int] | Mapping[int, int] | None = None,
) -> NDArray[np.bool_]:
    """Sample joint one-hot `selector_bits` (`bool[num_selector_qubits]`) for `plan` at `t=0`."""
    if not isinstance(plan, SelectorPlan):
        raise TypeError(f"Expected SelectorPlan, got {type(plan)!r}")
    f_map, n_map, _, _, _ = _prepare_run_maps(plan, fault_trace=fault_trace, pauli_noise_trace=pauli_noise_trace)
    rng_obj = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    return _sample_selector_activations_from_maps(plan, rng_obj, f_map, n_map)[0]


def run_selector(
    plan: SelectorPlan,
    *,
    rng: np.random.Generator | int | None = None,
    fault_trace: Sequence[int] | Mapping[int, int] | None = None,
    pauli_noise_trace: Sequence[int] | Mapping[int, int] | None = None,
    reset_outcomes: Sequence[int] | Mapping[int, int] | None = None,
    requested_outcomes: Mapping[int, int | bool] | None = None,
    requested_quantum_outcomes: Mapping[int, int | bool] | None = None,
    sweep_bits: Sequence[int | bool] | NDArray[np.bool_] | None = None,
    max_components: int | None = None,
    atol: float = DEFAULT_ATOL,
    extract_data: bool = False,
) -> TrajectoryRunResult:
    """`B2`: Execute `SelectorPlan` via joint one-hot selector `X_s` activations at `t=0`."""
    f_map, n_map, r_map, c_map, q_map = _prepare_run_maps(
        plan, fault_trace=fault_trace, pauli_noise_trace=pauli_noise_trace,
        reset_outcomes=reset_outcomes, requested_outcomes=requested_outcomes,
        requested_quantum_outcomes=requested_quantum_outcomes,
    )
    rng_obj = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    exp = ShotExpansion.zero(plan.num_total_qubits, atol=atol)
    sel_bits, fault_choices, gad_noise_map = _sample_selector_activations_from_maps(plan, rng_obj, f_map, n_map)
    n_w_tot = n_words_for_qubits(plan.num_total_qubits)
    zero_w = np.zeros(n_w_tot, dtype=np.uint64)
    for j in np.flatnonzero(sel_bits):
        s_active = plan.num_data_qubits + int(j)
        exp = apply_pauli(exp, CompiledPauliAxis(unit_bit_words(s_active, n_w_tot), zero_w, 0))

    res = _execute_compiled_records(
        exp, plan.records, plan.num_measurements, plan.final_tableau,
        rng=rng_obj, fault_map=None, noise_map=n_map, reset_map=r_map,
        req_c_map=c_map, req_q_map=q_map, sweep_bits=sweep_bits,
        max_components=max_components, atol=atol,
        preset_fault_choices=fault_choices,
        preset_noise_map=gad_noise_map if plan.gadgetize_pauli_noise else None,
        total_noise_channels=plan.total_noise_channels if plan.gadgetize_pauli_noise else None,
        selector_bits=sel_bits,
        frame_events=plan.data_frame_events if extract_data else plan.frame_events,
    )
    if extract_data:
        res.final_tableau, res.expansion = extract_data_expansion(
            plan, res.expansion, sel_bits
        )
    return res


def _slice_expansion_to_data_qubits(
    expansion: ShotExpansion,
    n_data: int,
    sel_arr: NDArray[np.bool_],
) -> ShotExpansion:
    """Project/verify selector bits and slice `ShotExpansion` to `0 .. n_data - 1`."""
    if not isinstance(expansion, ShotExpansion):
        raise TypeError(f"Expected ShotExpansion, got {type(expansion)!r}")
    n_total, n_sel = expansion.n_qubits, len(sel_arr)
    if n_data + n_sel != n_total:
        raise ValueError(f"n_data ({n_data}) + len(selector_bits) ({n_sel}) != expansion.n_qubits ({n_total})")
    work, n_w_tot = expansion, n_words_for_qubits(n_total)
    for j in range(n_sel):
        s_q, want_bit = n_data + j, int(sel_arr[j])
        has_x = work.d > 0 and any(get_bit_in_words(work.A_words[r], s_q) != 0 for r in range(work.d))
        mismatch_h = work.chi > 0 and any(get_bit_in_words(work.h_words[c], s_q) != want_bit for c in range(work.chi))
        if has_x or mismatch_h:
            proj_exp, _ = project_diagonal(
                work, unit_bit_words(s_q, n_w_tot), 0, want_bit
            )
            proj_exp.log_mass = work.log_mass
            work = proj_exp

    n_w_data, tm_data = n_words_for_qubits(n_data), tail_mask(n_data)
    A_d = np.ascontiguousarray(work.A_words[:, :n_w_data].copy(), dtype=np.uint64)
    if work.d > 0:
        A_d[:, -1] &= tm_data
    h_d = np.ascontiguousarray(work.h_words[:, :n_w_data].copy(), dtype=np.uint64)
    if work.chi > 0:
        h_d[:, -1] &= tm_data
    data_exp = work.copy()
    data_exp.n_qubits, data_exp.A_words, data_exp.h_words = n_data, A_d, h_d
    data_exp._refresh_a_union()
    return merge_frames(data_exp)


def extract_data_expansion(
    plan_or_tableau: SelectorPlan | ReferenceCHPTableau,
    expansion: ShotExpansion,
    n_data_or_bits: int | Sequence[int | bool] | NDArray[Any],
    selector_bits: Sequence[int | bool] | NDArray[Any] | None = None,
) -> tuple[ReferenceCHPTableau, ShotExpansion]:
    """`B2`: Matrix-free `O(n_words * chi)` extraction of `(data_tableau, data_expansion)`."""
    plan_meta: SelectorPlan | None = None
    if isinstance(plan_or_tableau, SelectorPlan):
        plan_meta = plan_or_tableau
        raw_bits = n_data_or_bits if selector_bits is None else selector_bits
        n_data = plan_meta.num_data_qubits
    elif isinstance(plan_or_tableau, ReferenceCHPTableau):
        if selector_bits is None or not isinstance(n_data_or_bits, (int, np.integer)):
            raise ValueError(
                "When passing ReferenceCHPTableau, signature is "
                "extract_data_expansion(ref_tableau, expansion, n_data_qubits, selector_bits)"
            )
        n_data, raw_bits = int(n_data_or_bits), selector_bits
    else:
        raise TypeError(f"Expected SelectorPlan or ReferenceCHPTableau, got {type(plan_or_tableau)!r}")

    sel_arr = np.asarray(raw_bits, dtype=np.bool_).ravel()
    data_exp = _slice_expansion_to_data_qubits(expansion, n_data, sel_arr)

    if plan_meta is not None:
        n_w_data = n_words_for_qubits(n_data)
        w_a, w_b, w_kap = np.zeros(n_w_data, dtype=np.uint64), np.zeros(n_w_data, dtype=np.uint64), 0
        for s_q, ax_d in plan_meta._gauge_steps:
            if s_q >= 0:
                if bool(sel_arr[s_q - n_data]):
                    w_kap = (ax_d.kappa + w_kap + (word_dot_parity(ax_d.b_words, w_a) << 1)) & 3
                    w_a ^= ax_d.a_words
                    w_b ^= ax_d.b_words
            elif get_bit_in_words(w_a, first_set_bit_in_words(ax_d.a_words)) == 1:
                f2 = word_dot_parity(w_a, ax_d.b_words) ^ word_dot_parity(w_b, ax_d.a_words)
                fac_kap = (ax_d.kappa + (3 if f2 else 1)) & 3
                w_kap = (w_kap + fac_kap + (word_dot_parity(w_b, ax_d.a_words) << 1)) & 3
                w_a ^= ax_d.a_words
                w_b ^= ax_d.b_words

        odd_ph = ((w_kap - word_dot_parity(w_a, w_b)) & 1) != 0
        kap_herm = ((w_kap - 1) & 3) if odd_ph else w_kap
        if words_any_nonzero(w_a) or words_any_nonzero(w_b) or kap_herm != 0:
            data_exp = apply_pauli(data_exp, CompiledPauliAxis(w_a, w_b, kap_herm))
            data_exp = merge_frames(data_exp)
        if odd_ph and data_exp.chi > 0:
            data_exp.weights = np.ascontiguousarray(data_exp.weights * 1.0j)
        return plan_meta.data_final_tableau.copy(), data_exp

    assert isinstance(plan_or_tableau, ReferenceCHPTableau)
    return _restrict_tableau_to_data(plan_or_tableau, n_data, sel_arr), data_exp


# ---------------------------------------------------------------------------
# Exact Dense Statevector Reconstruction (`reconstruct_statevector`)
# ---------------------------------------------------------------------------

_UNITARY_1Q_2Q_CACHE: dict[str, NDArray[np.complex128]] = {}


def _cached_gate_unitary_le(gate_name: str) -> NDArray[np.complex128]:
    u = _UNITARY_1Q_2Q_CACHE.get(gate_name)
    if u is None:
        u = eval_decomposition_unitary_le(decompose_clifford_exact(gate_name))
        _UNITARY_1Q_2Q_CACHE[gate_name] = u
    return u


def _apply_pauli_to_dense_le(
    psi: NDArray[np.complex128],
    a_words: NDArray[np.uint64],
    b_words: NDArray[np.uint64],
    kappa: int,
) -> NDArray[np.complex128]:
    x_idx = np.arange(psi.shape[0], dtype=np.uint64) ^ np.uint64(int(a_words[0]))
    par = np.bitwise_count(x_idx & np.uint64(int(b_words[0]))) & np.uint64(1)
    p_idx = (np.uint64(kappa & 3) + (par << np.uint64(1))) & np.uint64(3)
    return np.ascontiguousarray(_PHASE4_ARR[p_idx] * psi[x_idx.astype(np.intp)])


def reconstruct_statevector(
    first: ShotExpansion | ReferenceCHPTableau | TrajectoryRunResult,
    second: ShotExpansion | ReferenceCHPTableau | None = None,
    *,
    frame_events: Sequence[FrameEvent] | None = None,
    ref_tableau: ReferenceCHPTableau | None = None,
    plan: DirectPlan | SelectorPlan | None = None,
) -> NDArray[np.complex128]:
    """Reconstruct exact `complex128` little-endian physical statevector (`< 1e-15` global phase)."""
    if isinstance(first, TrajectoryRunResult):
        expansion = first.expansion
        if ref_tableau is None:
            ref_tableau = first.final_tableau
        if frame_events is None:
            frame_events = first.frame_events
    elif isinstance(first, ReferenceCHPTableau):
        if not isinstance(second, ShotExpansion):
            raise TypeError("Expected ShotExpansion as second argument when first is ReferenceCHPTableau")
        ref_tableau, expansion = first, second
    elif isinstance(first, ShotExpansion):
        expansion = first
        if isinstance(second, ReferenceCHPTableau) and ref_tableau is None:
            ref_tableau = second
    else:
        raise TypeError(f"Unsupported argument type for reconstruct_statevector: {type(first)!r}")

    n = expansion.n_qubits
    if n > 26:
        raise ValueError(f"reconstruct_statevector requires n_qubits <= 26, got {n}")
    events = frame_events
    if events is None and plan is not None:
        if isinstance(plan, DirectPlan):
            events = plan.frame_events
        elif isinstance(plan, SelectorPlan):
            events = plan.data_frame_events if n == plan.num_data_qubits else plan.frame_events

    if events is None:
        psi_ref = expansion.to_statevector(endian="little")
        return psi_ref if ref_tableau is None else apply_reference_tableau_to_statevector(psi_ref, ref_tableau, endian="little")

    psi = expansion.to_statevector(endian="little")
    for r_ev in reversed([ev for ev in events if isinstance(ev, RightRotEvent)]):
        p_psi = _apply_pauli_to_dense_le(psi, r_ev.rot_axis.a_words, r_ev.rot_axis.b_words, r_ev.rot_axis.kappa)
        psi = np.ascontiguousarray((psi - 1.0j * p_psi) * _INV_SQRT2)

    n_w = n_words_for_qubits(n)
    zero_w = np.zeros(n_w, dtype=np.uint64)
    for l_ev in (ev for ev in events if not isinstance(ev, RightRotEvent)):
        if isinstance(l_ev, LeftUnitaryEvent):
            psi = _apply_local_unitary_c128(psi, _cached_gate_unitary_le(l_ev.gate_name), l_ev.targets, n, "little")
        elif isinstance(l_ev, LeftResetXEvent):
            psi = _apply_local_unitary_c128(psi, _cached_gate_unitary_le("X"), (l_ev.qubit,), n, "little")
        elif isinstance(l_ev, LeftSppEvent):
            a_w, b_w, kap = zero_w.copy(), zero_w.copy(), (2 if l_ev.negate else 0)
            for q_s, pch_s in l_ev.terms:
                if pch_s != "I":
                    qa = unit_bit_words(q_s, n_w) if pch_s in ("X", "Y") else zero_w
                    qb = unit_bit_words(q_s, n_w) if pch_s in ("Z", "Y") else zero_w
                    a_w, b_w, kap = pauli_multiply(a_w, b_w, kap, qa, qb, 1 if pch_s == "Y" else 0)
            p_psi = _apply_pauli_to_dense_le(psi, a_w, b_w, kap)
            phase, sign_j = (OMEGA_POWERS[7], 1.0j) if l_ev.dag else (OMEGA_POWERS[1], -1.0j)
            psi = np.ascontiguousarray(phase * _INV_SQRT2 * (psi + sign_j * p_psi))
        else:
            raise ValueError(f"Unexpected Left event {l_ev!r}")

    return psi
