"""Compiled record/event types, moving-reference lowering (`C1`/`C1.5`), and channel sampling (`stimside._magic.channels`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]

from stimside._magic.gates import (
    CliffordFaultChannel,
    CompiledPauliAxis,
    PAU_ABK,
    coerce_strict_int,
    decompose_clifford_exact,
    spec_terms,
)
from stimside._magic.phase_qfe import (
    DEFAULT_ATOL,
    OMEGA_POWERS,
    CompiledPauliBlock,
    ShotExpansion,
    apply_pauli,
    apply_quarter_turn,
)
from stimside.util.bitwords import (
    first_set_bit_in_words,
    pauli_multiply,
    tail_mask,
    unit_bit_words,
    validate_hermitian_pauli,
    word_dot_parity_batch,
    words_any_nonzero,
)
from stimside.util.reference_chp import DetMeasMeta, RandMeasMeta, ReferenceCHPTableau


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
    obs_axis: CompiledPauliAxis
    rot_axis: CompiledPauliAxis | None


@dataclass(frozen=True)
class ResetRecord:
    meas_index: int | None
    flip_prob: float
    invert_result: bool
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
    tag_prefix: str
    num_measurements: int = 0


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


# ---------------------------------------------------------------------------
# ReferenceCHPTableau Helpers, C1 & C1.5
# ---------------------------------------------------------------------------


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


def compile_clifford_factors(
    ref_chp: ReferenceCHPTableau,
    gate_name: str,
    targets: Sequence[int],
) -> CompiledCliffordFactorList:
    dec = decompose_clifford_exact(gate_name, targets)
    p0_neg, p0_terms = spec_terms(dec.p0_spec, targets)
    p0_axis = None if (not p0_terms and not p0_neg) else compile_pauli_product(ref_chp, p0_terms, negate=p0_neg)
    r_axes = tuple(
        compile_pauli_product(ref_chp, r_terms, negate=r_neg)
        for r_neg, r_terms in (spec_terms(r_sp, targets) for r_sp in dec.r_specs)
    )
    return CompiledCliffordFactorList(v=dec.v, p0=p0_axis, r_axes=r_axes)


def apply_clifford_factors(
    exp: ShotExpansion,
    fac: CompiledCliffordFactorList,
    *,
    inverse: bool = False,
    check_unitary: bool = False,
    atol: float = DEFAULT_ATOL,
    n_words: int | None = None,
    on_step: Callable[[ShotExpansion], None] | None = None,
) -> ShotExpansion:
    """Apply `CompiledCliffordFactorList` (or its inverse) to `exp`."""
    if exp.chi == 0:
        return exp

    def _unpack(ax: CompiledPauliAxis) -> tuple[Any, ...]:
        if n_words is None or ax.a_words.shape[0] == n_words:
            return (ax,)
        pad = (0, n_words - ax.a_words.shape[0])
        return (
            np.ascontiguousarray(np.pad(ax.a_words, pad)),
            np.ascontiguousarray(np.pad(ax.b_words, pad)),
            int(ax.kappa) & 3,
        )

    if not inverse and fac.p0 is not None:
        exp = apply_pauli(exp, *_unpack(fac.p0))
    for r_ax in (reversed(fac.r_axes) if inverse else fac.r_axes):
        exp = apply_quarter_turn(exp, *_unpack(r_ax), sign=-1 if inverse else 1, check_unitary=check_unitary, atol=atol)
        if on_step is not None:
            on_step(exp)
    if inverse and fac.p0 is not None:
        exp = apply_pauli(exp, *_unpack(fac.p0))
    if fac.v and exp.chi > 0:
        exp.weights = np.ascontiguousarray(exp.weights * OMEGA_POWERS[(-fac.v if inverse else fac.v) & 7])
    return exp


def split_pauli_product_targets(
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


def reduce_pauli_product_targets(
    raw_targets: Sequence[tuple[int, str]],
) -> list[tuple[int, str]]:
    """Reduce `(qubit, pauli)` targets modulo global phase into unique unsigned Hermitian Paulis."""
    phys_map: dict[int, tuple[int, int]] = {}
    for q_raw, pch_raw in raw_targets:
        if pch_raw not in PAU_ABK:
            raise ValueError(f"Expected Pauli character in ('I','X','Y','Z'), got {pch_raw!r}")
        q_c = int(q_raw)
        xc, zc = phys_map.get(q_c, (0, 0))
        dx, dz, _ = PAU_ABK[pch_raw]
        phys_map[q_c] = (xc ^ dx, zc ^ dz)
    return [(q_c, "IZXY"[(xc << 1) | zc]) for q_c, (xc, zc) in phys_map.items() if (xc or zc)]


def compile_pauli_measurement_c15(
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
        return MeasurementRecord(meas_index, flip_prob, invert_result, bar_M, None)
    e_p = unit_bit_words(first_set_bit_in_words(bar_M.a_words), ref_chp.n_words)
    rot_axis = CompiledPauliAxis(bar_M.a_words.copy(), np.ascontiguousarray(bar_M.b_words ^ e_p), (bar_M.kappa + 1) & 3)
    ref_chp.apply_right_rot(rot_axis.a_words, rot_axis.b_words, (rot_axis.kappa + 1) & 3)
    if frame_events is not None:
        frame_events.append(RightRotEvent(rot_axis=rot_axis))
    return MeasurementRecord(meas_index, flip_prob, invert_result, bar_M, rot_axis)


def compile_reset_c15(
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
        meas_index, flip_prob, invert_result,
        obs_axis, rot_axis, reset_flip_ref, kick_axis,
    )


FEEDBACK_2Q_GATES: dict[str, tuple[str, str]] = {
    "CX": ("Z", "X"), "CY": ("Z", "Y"), "CZ": ("Z", "Z"), "XCZ": ("X", "Z"), "YCZ": ("Y", "Z"),
}
PAULI_2Q_ORDER: tuple[str, ...] = tuple(
    p0 + p1 for p0 in "IXYZ" for p1 in "IXYZ" if not (p0 == "I" and p1 == "I")
)


def emit_magic_terms(
    tag: str,
    qs: Sequence[int],
    ref_chp: ReferenceCHPTableau,
    out_records: list[Any],
) -> None:
    if tag in ("T", "T_DAG"):
        eps = 1 if tag == "T" else -1
        for q in qs:
            out_records.append(MagicTermRecord(MagicPhaseTerm((compile_pauli_product(ref_chp, [(q, "Z")]),), "T", eps)))
    else:
        for off in range(0, len(qs), 3):
            axes = tuple(compile_pauli_product(ref_chp, [(q, "Z")]) for q in qs[off : off + 3])
            out_records.append(MagicTermRecord(MagicPhaseTerm(axes=axes, kind="CCZ")))


def infer_total_data_qubits(
    circuit: stim.Circuit, all_channels: Sequence[CliffordFaultChannel]
) -> int:
    return max(
        [max(1, int(circuit.num_qubits))]
        + [q + 1 for ch in all_channels for b in ch.branches for q in b.targets]
    )


def restrict_tableau_to_data(
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


_REAL_KEY_MAP: dict[str, str] = {
    "faults": "faults", "measurements": "measurements", "quantum_measurements": "quantum_measurements",
    "paulis": "paulis", "pauli_noise": "paulis", "resets": "resets", "sweep": "sweep_bits", "sweep_bits": "sweep_bits",
}
_PAULI_1Q_STR: dict[str, int] = {"I": 0, "X": 1, "Y": 2, "Z": 3}
_PAULI_4_STR: dict[str, int] = {"I": 0, "X": 2, "Y": 3, "Z": 4}
_PAULI_2Q_STR: dict[str, int] = {a + b: i for i, (a, b) in enumerate((p, q) for p in "IXYZ" for q in "IXYZ")}


def split_realization_sections(raw: Mapping[Any, Any]) -> dict[str, Any]:
    if not raw:
        return {}
    if not any(isinstance(k, str) for k in raw):
        return {"faults": raw}
    sec: dict[str, Any] = {}
    for k, v in raw.items():
        ck = _REAL_KEY_MAP.get(k) if isinstance(k, str) else None
        if ck is None or ck in sec:
            raise ValueError(f"Invalid or duplicate realization key {k!r}")
        if v is not None:
            sec[ck] = v
    return sec


def normalize_pauli_slot_map(
    raw_p: Any, max_choices: Sequence[int], noise_kinds: Mapping[int, str]
) -> Any:
    if not isinstance(raw_p, Mapping):
        return raw_p
    out: dict[int, Any] = {}
    for k, v in raw_p.items():
        ik = coerce_strict_int(k, "paulis key")
        if ik < 0 or ik >= len(max_choices):
            raise ValueError(f"Pauli slot {ik} out of range")
        mc, kind = max_choices[ik], noise_kinds.get(ik, "")
        if isinstance(v, str):
            s = v.strip().upper()
            tbl = (
                ({"I": 0, kind[:1]: 1} if kind in ("X_ERROR", "Y_ERROR", "Z_ERROR") else {"I": 0})
                if mc == 1
                else {3: _PAULI_1Q_STR, 4: _PAULI_4_STR, 15: _PAULI_2Q_STR}.get(mc, {})
            )
            if s not in tbl:
                raise ValueError(f"Invalid Pauli label {v!r} at slot {ik}")
            out[ik] = tbl[s]
        else:
            out[ik] = int(bool(v)) if (mc == 1 and isinstance(v, (bool, np.bool_))) else v
    return out


def sample_categorical_branch(probs: Sequence[float] | NDArray[np.float64], rng: np.random.Generator) -> int:
    """Return `0` with prob `1 - sum(probs)`, or `1..K` for branch `0..K-1`."""
    u = float(rng.random())
    acc = 0.0
    for idx, p in enumerate(probs):
        acc += float(p)
        if u < acc:
            return idx + 1
    return 0


def sample_heralded_pauli_branch(
    probs: Sequence[float] | NDArray[np.float64],
    req_h: int | bool | None,
    rng: np.random.Generator,
    *,
    presampled: bool = False,
    skip_single_rng: bool = False,
) -> tuple[int, float]:
    """Sample `(choice_idx, branch_prob)` for a heralded Pauli channel (`req_h` in `{None, 0, 1}`)."""
    if req_h is None:
        return sample_categorical_branch(probs, rng), 1.0
    p_h = min(1.0, max(0.0, float(sum(probs))))
    if not req_h:
        p_no = max(0.0, 1.0 - p_h)
        if p_no <= 1e-15:
            return (1 if presampled else 0), 0.0
        return 0, p_no
    if p_h <= 1e-15:
        return 0, 0.0
    pos = [(i + 1, float(p)) for i, p in enumerate(probs) if p > 0.0]
    if skip_single_rng and len(pos) == 1:
        return pos[0][0], p_h
    u = float(rng.random()) * p_h
    acc = 0.0
    for idx_1, pr in pos[:-1]:
        acc += pr
        if u < acc:
            return idx_1, p_h
    return pos[-1][0], p_h


def normalize_choice_map(
    trace: object,
    max_choices: Sequence[int],
    name: str,
    *,
    key_to_slot: Mapping[int, int] | None = None,
    allow_none_zero: bool = False,
    allow_bool_values: bool = False,
    allow_binary_bool: bool = False,
    strict_length: bool = False,
    mapping_only: bool = False,
) -> dict[int, int] | None:
    """Validate integer keys/indices and choice bounds, returning slot-indexed `dict[int, int]`."""
    if trace is None:
        return None
    if isinstance(trace, Mapping):
        slot_lookup = key_to_slot if key_to_slot is not None else {i: i for i in range(len(max_choices))}
        out: dict[int, int] = {}
        for k, v in trace.items():
            ik = coerce_strict_int(k, f"{name} key", allow_bool=False)
            if ik not in slot_lookup:
                raise ValueError(f"Out-of-range or unknown key {k!r} in {name}")
            slot = slot_lookup[ik]
            allow_b = allow_bool_values or (allow_binary_bool and max_choices[slot] == 1)
            iv = 0 if (v is None and allow_none_zero) else coerce_strict_int(v, f"{name}[{ik}]", allow_bool=allow_b)
            if iv < 0 or iv > max_choices[slot]:
                raise ValueError(f"{name}[{ik}]={iv} out of range [0, {max_choices[slot]}]")
            out[slot] = iv
        return out
    if mapping_only or isinstance(trace, (str, bytes)) or not isinstance(trace, (Sequence, np.ndarray)):
        raise ValueError(f"Invalid {name} container type: {type(trace)!r}")
    if isinstance(trace, np.ndarray) and trace.ndim != 1:
        raise ValueError(f"{name} array must be 1D, got shape {trace.shape}")
    if (len(trace) != len(max_choices)) if strict_length else (len(trace) < len(max_choices)):
        raise ValueError(f"{name} length {len(trace)} invalid for {len(max_choices)} slots")
    out_seq: dict[int, int] = {}
    for idx, max_c in enumerate(max_choices):
        allow_b = allow_bool_values or (allow_binary_bool and max_c == 1)
        iv = coerce_strict_int(trace[idx], f"{name}[{idx}]", allow_bool=allow_b)
        if iv < 0 or iv > max_c:
            raise ValueError(f"{name}[{idx}]={iv} out of range [0, {max_c}]")
        out_seq[idx] = iv
    return out_seq
