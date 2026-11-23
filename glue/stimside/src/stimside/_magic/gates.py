"""Exact 1Q/2Q Clifford decompositions and `CLIFFORD_FAULT` parsing (`stimside._magic.gates`)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]

from stimside._magic.phase_qfe import (
    I_POWERS_ARR as _PHASE4_ARR,
    INV_SQRT2 as _INV_SQRT2,
    OMEGA_POWERS,
)
from stimside.op_handlers.leakage_handlers.common_parsing import split_arguments
from stimside.util.circuit_ir import CircuitIR, build_ir

_IDENTITY_OP = stim.CircuitInstruction("I", [], [])

PAU_ABK: dict[str, tuple[int, int, int]] = {"I": (0, 0, 0), "X": (1, 0, 0), "Z": (0, 1, 0), "Y": (1, 1, 1)}

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


def coerce_strict_int(val: object, label: str, *, allow_bool: bool = False) -> int:
    if isinstance(val, (bool, np.bool_)):
        if allow_bool:
            return int(bool(val))
        raise ValueError(f"Boolean value {val!r} is not valid for {label}")
    if not isinstance(val, (int, np.integer)):
        raise ValueError(f"Expected integer for {label}, got {val!r}")
    return int(val)


def canonical_gate_data(gate_name: str) -> stim.GateData:
    if not isinstance(gate_name, str) or not gate_name.strip():
        raise ValueError(f"Invalid gate name: {gate_name!r}")
    try:
        return stim.gate_data(gate_name.strip())
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(f"Unknown Stim gate {gate_name!r}") from exc


def canonical_clifford_gate_name_and_arity(gate_name: str) -> tuple[str, int]:
    gd = canonical_gate_data(gate_name)
    if not gd.is_unitary or not (gd.is_single_qubit_gate or gd.is_two_qubit_gate):
        raise ValueError(f"Gate {gate_name!r} is not a 1Q or 2Q unitary Clifford gate")
    return gd.name, (1 if gd.is_single_qubit_gate else 2)


def plain_qubits(targets: Sequence[stim.GateTarget], label: str) -> list[int]:
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
        canon_name, arity = canonical_clifford_gate_name_and_arity(self.gate_name)
        t_tup = tuple(coerce_strict_int(q, "FaultBranch target") for q in self.targets)
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
        step = coerce_strict_int(self.step_index, "CliffordFaultChannel step_index")
        if step < 0:
            raise ValueError(f"CliffordFaultChannel step_index must be >= 0, got {step}")
        object.__setattr__(self, "step_index", step)
        if self.slot_id is not None:
            sid = coerce_strict_int(self.slot_id, "CliffordFaultChannel slot_id")
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


# ---------------------------------------------------------------------------
# Exact Clifford Decomposition & Symplectic Defect Helpers
# ---------------------------------------------------------------------------


def _spec_to_abk_small(spec: str) -> tuple[int, int, int]:
    a_m, b_m, kap = 0, 0, (2 if spec[0] == "-" else 0)
    for i, ch in enumerate(spec[1:]):
        if ch not in PAU_ABK:
            raise ValueError(f"Invalid Pauli char {ch!r} in {spec!r}")
        xa, zb, dk = PAU_ABK[ch]
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


def spec_terms(spec: str, targets: Sequence[int]) -> tuple[bool, list[tuple[int, str]]]:
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
) -> NDArray[np.complex128]:
    """Evaluate `decomp` as an exact `complex128` little-endian unitary matrix."""
    k = decomp.num_qubits
    eye_k = np.eye(1 << k, dtype=np.complex128)
    U_local = _dense_pauli_from_spec_le(decomp.p0_spec)
    for r_spec in decomp.r_specs:
        U_local = (_INV_SQRT2 * (eye_k + 1.0j * _dense_pauli_from_spec_le(r_spec))) @ U_local
    return OMEGA_POWERS[decomp.v] * U_local


def decompose_clifford_exact(
    gate_or_tableau: str | stim.Tableau,
    targets: Sequence[int] | None = None,
    *,
    inverse: bool = False,
) -> ExactCliffordDecomposition:
    """Decompose a named 1Q/2Q Stim Clifford or `stim.Tableau` with exact `omega**v` phase."""
    if isinstance(gate_or_tableau, str):
        canon_name, k = canonical_clifford_gate_name_and_arity(gate_or_tableau)
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
        canon_gate, arity = canonical_clifford_gate_name_and_arity(g_str)
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
    canon_carrier = canonical_gate_data(op.name).name
    if canon_carrier not in ("I", "II") or op.gate_args_copy():
        raise ValueError(f"Invalid CLIFFORD_FAULT carrier/args on {op!r}")
    parsed_branches = parse_clifford_fault_tag(op.tag)
    _, arity = canonical_clifford_gate_name_and_arity(parsed_branches[0][1])
    if canon_carrier == "II" and arity != 2:
        raise ValueError(f"Carrier 'II' requires 2-qubit CLIFFORD_FAULT gates, got {parsed_branches}")
    qs = plain_qubits(op.targets_copy(), "CLIFFORD_FAULT")
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


def extract_inline_faults_from_ir(ir: CircuitIR) -> tuple[CliffordFaultChannel, ...]:
    """Extract inline ``CLIFFORD_FAULT`` channels from a pre-built ``CircuitIR``."""
    inline_faults: list[CliffordFaultChannel] = []
    for ins in ir.instrs:
        if ins.tag.startswith("CLIFFORD_FAULT"):
            inline_faults.extend(
                parse_clifford_fault_instruction(ins.op, ins.unrolled_index)
            )
    return tuple(inline_faults)


def validate_circuit_ir(circuit: stim.Circuit, ir: CircuitIR) -> CircuitIR:
    """Validate that a supplied ``CircuitIR`` matches ``circuit``."""
    if not isinstance(ir, CircuitIR):
        raise TypeError(f"Expected CircuitIR or None, got {type(ir)!r}")
    got = (ir.num_qubits, ir.num_measurements, len(ir.instrs))
    exp = (circuit.num_qubits, circuit.num_measurements, len(circuit))
    if got[:2] != exp[:2] or (not ir.has_repeat and not ir.from_op_list and got[2] != exp[2]):
        raise ValueError(f"CircuitIR does not match circuit: got {got}, expected {exp}")
    return ir


def extract_circuit_faults(
    circuit: stim.Circuit,
    *,
    ir: CircuitIR | None = None,
) -> tuple[list[stim.CircuitInstruction], tuple[CliffordFaultChannel, ...]]:
    """Unroll ``circuit`` via ``CircuitIR`` and extract inline ``CLIFFORD_FAULT`` ops."""
    cir = build_ir(circuit) if ir is None else validate_circuit_ir(circuit, ir)
    inline_faults = extract_inline_faults_from_ir(cir)
    fault_steps = {ch.step_index for ch in inline_faults}
    flat_ops = [
        _IDENTITY_OP if ins.unrolled_index in fault_steps else ins.op
        for ins in cir.instrs
    ]
    return flat_ops, inline_faults


def merge_and_validate_fault_channels(
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
