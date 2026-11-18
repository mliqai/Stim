"""Tag-agnostic circuit intermediate representation (IR) for WP-27.

Implements the RFC section 3 design rules:
- Imports only stdlib, ``stim``, and sibling ``util`` helpers; carries no tag
  literals, no native-ABI names, and no engine-specific constants.
- Never inspects or parses instruction tags: ``op.tag`` is copied verbatim into
  ``IRInstr`` as a plain string.
- Uses one fixed unrolled-index definition (``unroll_circuit``) alongside
  explicit ``flat_index`` and ``flat_offset`` mappings into ``flattened()``.
- Derived passes (``step_map``, ``tableau_step_layout``, ``split_copies``,
  ``unrolled_to_flat``) are pure functions over ``CircuitIR``, not IR fields.
- Per-consumer lazy tag parsing lives outside this module (see ``ir_tags.py``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

import stim

from stimside.util.circuit import unroll_circuit
from stimside.util.stim_workarounds import split_fused_instruction

_ANNOTATION = frozenset(
    {"DETECTOR", "TICK", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"}
)
_HERALDED = frozenset({"HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"})
_CORRELATED = frozenset({"E", "ELSE_CORRELATED_ERROR"})
_PAULI_TERMS = frozenset({"SPP", "SPP_DAG"})
_NON_FUSING = frozenset(
    {
        "DETECTOR",
        "E",
        "ELSE_CORRELATED_ERROR",
        "OBSERVABLE_INCLUDE",
        "QUBIT_COORDS",
        "SHIFT_COORDS",
        "TICK",
    }
)
_SHIFTED_ARGS_GATES = frozenset({"DETECTOR", "QUBIT_COORDS"})


class Category(str, Enum):
    """Coarse instruction category used across consumers."""

    ANNOTATION = "annotation"
    MPAD = "mpad"
    HERALD = "herald"
    MEASURE_RESET = "measure_reset"
    MEASURE = "measure"
    RESET = "reset"
    UNITARY = "unitary"
    NOISE = "noise"
    OTHER = "other"


class UntaggedFilter(str, Enum):
    """Untagged-instruction filter class (rule R1) when ``uc=True``."""

    NONE = "none"
    ONE_QUBIT = "one_qubit"
    TWO_QUBIT = "two_qubit"
    CORRELATED = "correlated"
    PAULI_TERMS = "pauli_terms"


def classify_category(name: str) -> Category:
    """Classify a Stim gate/instruction name into its coarse ``Category``."""
    gd = stim.gate_data(name)
    name = gd.name
    if name in _ANNOTATION:
        return Category.ANNOTATION
    if name == "MPAD":
        return Category.MPAD
    if name in _HERALDED:
        return Category.HERALD
    if gd.produces_measurements and gd.is_reset:
        return Category.MEASURE_RESET
    if gd.produces_measurements:
        return Category.MEASURE
    if gd.is_reset:
        return Category.RESET
    if gd.is_unitary:
        return Category.UNITARY
    if gd.is_noisy_gate:
        return Category.NOISE
    return Category.OTHER


def classify_untagged(name: str, *, uc: bool) -> UntaggedFilter:
    """Rule R1 for an op not claimed by a tag (the caller decides 'claimed')."""
    if not uc:
        return UntaggedFilter.NONE
    gd = stim.gate_data(name)
    name = gd.name
    if gd.produces_measurements or not (gd.is_noisy_gate or gd.is_unitary):
        return UntaggedFilter.NONE
    if name in _CORRELATED:
        return UntaggedFilter.CORRELATED
    if name in _PAULI_TERMS:
        return UntaggedFilter.PAULI_TERMS
    if gd.is_single_qubit_gate:
        return UntaggedFilter.ONE_QUBIT
    if gd.is_two_qubit_gate:
        return UntaggedFilter.TWO_QUBIT
    return UntaggedFilter.NONE


@dataclass(frozen=True)
class IRInstr:
    """Single unrolled instruction record in ``CircuitIR``."""

    unrolled_index: int
    op: stim.CircuitInstruction
    tag: str
    category: Category
    untagged_filter: UntaggedFilter
    flat_index: int  # -1 for SHIFT_COORDS or an op-list source
    flat_offset: int  # first target of this op inside its (possibly fused) flat op
    num_measurements: int
    first_measurement: int


@dataclass(frozen=True)
class CircuitIR:
    """Immutable, hashable circuit IR built from a circuit or op sequence."""

    instrs: tuple[IRInstr, ...]
    flat_ops: tuple[stim.CircuitInstruction, ...]  # () for an op-list source
    num_qubits: int
    num_measurements: int
    has_repeat: bool
    uc: bool
    from_op_list: bool


def _tag_token(op: stim.CircuitInstruction) -> str:
    """Return Stim's injective escaped-tag header token without reading ``.tag``."""
    text = str(op)
    n = len(op.name)
    if len(text) > n and text[n] == "[":
        return text[n + 1 : text.index("]", n + 1)]
    return ""


def _op_matches_flat(
    op: stim.CircuitInstruction,
    op_targets: list[stim.GateTarget],
    flat_op: stim.CircuitInstruction,
    flat_targets: list[stim.GateTarget],
    used: int,
    flat_args: list[float],
    flat_tag: str,
) -> bool:
    """Return True iff unrolled ``op`` matches ``flat_op`` at target offset ``used``."""
    if op.name != flat_op.name:
        return False
    if op.name in _NON_FUSING and op_targets != flat_targets:
        return False
    if op_targets != flat_targets[used : used + len(op_targets)]:
        return False
    if op.name not in _SHIFTED_ARGS_GATES and op.gate_args_copy() != flat_args:
        return False
    return _tag_token(op) == flat_tag


def _try_fuse_zero(
    op: stim.CircuitInstruction,
    cur: stim.CircuitInstruction,
    meta_ready: bool,
    flat_args: list[float],
    flat_tag: str,
) -> tuple[bool, bool, list[float], str]:
    """Return ``(fused, meta_ready, flat_args, flat_tag)`` for a zero-target op."""
    if op.name in _NON_FUSING or op.name != cur.name:
        return False, meta_ready, flat_args, flat_tag
    if not meta_ready:
        flat_args = cur.gate_args_copy()
        flat_tag = _tag_token(cur)
        meta_ready = True
    fused = op.gate_args_copy() == flat_args and _tag_token(op) == flat_tag
    return fused, meta_ready, flat_args, flat_tag


def _align_flat_ops(
    ops: Sequence[stim.CircuitInstruction],
    flat_ops: tuple[stim.CircuitInstruction, ...],
) -> list[tuple[int, int]]:
    """Map unrolled ops to ``(flat_index, flat_offset)`` with full validation."""
    where: list[tuple[int, int]] = []
    f = -1
    flat_targets: list[stim.GateTarget] = []
    flat_len = 0
    flat_args: list[float] = []
    flat_tag = ""
    meta_ready = False
    used = 0
    for u_idx, op in enumerate(ops):
        if op.name == "SHIFT_COORDS":
            where.append((-1, 0))
            continue
        op_targets = op.targets_copy()
        n_tgt = len(op_targets)
        if used == flat_len:
            fused_zero = False
            if n_tgt == 0 and f >= 0:
                fused_zero, meta_ready, flat_args, flat_tag = _try_fuse_zero(
                    op, flat_ops[f], meta_ready, flat_args, flat_tag
                )
            if not fused_zero:
                f += 1
                if f >= len(flat_ops):
                    raise ValueError(
                        f"circuit.flattened() has {len(flat_ops)} ops, "
                        "too few for the unrolled circuit."
                    )
                cur = flat_ops[f]
                used = 0
                if op == cur:
                    flat_targets = op_targets
                    flat_len = n_tgt
                    meta_ready = False
                else:
                    flat_targets = cur.targets_copy()
                    flat_len = len(flat_targets)
                    flat_args = cur.gate_args_copy()
                    flat_tag = _tag_token(cur)
                    meta_ready = True
                    if not _op_matches_flat(
                        op, op_targets, cur, flat_targets, 0, flat_args, flat_tag
                    ):
                        raise ValueError(
                            f"Unrolled op {u_idx} ({op}) does not match "
                            f"flattened op {f} ({cur})."
                        )
        elif not _op_matches_flat(
            op, op_targets, flat_ops[f], flat_targets, used, flat_args, flat_tag
        ):
            raise ValueError(
                f"Unrolled op {u_idx} ({op}) does not match "
                f"flattened op {f} ({flat_ops[f]})."
            )
        where.append((f, used))
        used += n_tgt
    if f != len(flat_ops) - 1 or used != flat_len:
        raise ValueError(
            f"circuit.flattened() has {len(flat_ops)} ops, "
            "more than the unrolled circuit maps to."
        )
    return where


def build_ir(
    src: stim.Circuit | Sequence[stim.CircuitInstruction],
    *,
    uc: bool = False,
) -> CircuitIR:
    """Build a ``CircuitIR`` from a ``stim.Circuit`` or an explicit op list."""
    if isinstance(src, stim.Circuit):
        from_list = False
        ops = unroll_circuit(src)
        flat_ops: tuple[stim.CircuitInstruction, ...] = tuple(src.flattened())
        has_repeat = any(isinstance(i, stim.CircuitRepeatBlock) for i in src)
        num_qubits, num_meas = src.num_qubits, src.num_measurements
    else:
        from_list = True
        ops = list(src)
        if any(isinstance(o, stim.CircuitRepeatBlock) for o in ops):
            raise TypeError(
                "build_ir op sequence cannot contain stim.CircuitRepeatBlock; "
                "pass a stim.Circuit or unroll_circuit() output instead."
            )
        flat_ops = ()
        has_repeat = False
        qs = [
            t.qubit_value
            for o in ops
            for t in o.targets_copy()
            if t.qubit_value is not None
        ]
        num_qubits = 1 + max(qs) if qs else 0
        num_meas = sum(o.num_measurements for o in ops)
    where = [(-1, 0)] * len(ops) if from_list else _align_flat_ops(ops, flat_ops)
    instrs: list[IRInstr] = []
    first = 0
    for i, op in enumerate(ops):
        k = op.num_measurements
        instrs.append(
            IRInstr(
                i,
                op,
                op.tag,
                classify_category(op.name),
                classify_untagged(op.name, uc=uc),
                where[i][0],
                where[i][1],
                k,
                first,
            )
        )
        first += k
    return CircuitIR(
        tuple(instrs), flat_ops, num_qubits, num_meas, has_repeat, uc, from_list
    )


# ---- derived passes (functions of the IR, not IR fields) ---------------------


@dataclass(frozen=True)
class StepMap:
    """Bidirectional mapping between unrolled indices and non-skipped steps."""

    skip: frozenset[str]
    step_to_unrolled: tuple[int, ...]
    unrolled_to_step: tuple[int, ...]  # -1 for skipped ops


def step_map(ir: CircuitIR, skip: frozenset[str]) -> StepMap:
    """Derive a ``StepMap`` skipping instructions whose name is in ``skip``."""
    s2u: list[int] = []
    u2s: list[int] = []
    for i in ir.instrs:
        if i.op.name in skip:
            u2s.append(-1)
        else:
            u2s.append(len(s2u))
            s2u.append(i.unrolled_index)
    return StepMap(skip, tuple(s2u), tuple(u2s))


@dataclass(frozen=True)
class TableauStepLayout:
    """Tableau engine step maps derived from a StepMap plus per-step bare-op bit."""

    step_to_unrolled: tuple[int, ...]
    step_to_quant: tuple[int, ...]  # len num_steps + 1
    step_to_full: tuple[int, ...]  # len num_steps + 1


def tableau_step_layout(
    ir: CircuitIR, sm: StepMap, has_bare_op: Sequence[bool]
) -> TableauStepLayout:
    """Compute ``step_to_quant`` and ``step_to_full`` from ``sm`` and ``has_bare_op``."""
    n = len(sm.step_to_unrolled)
    if len(has_bare_op) < n:
        raise ValueError(
            f"has_bare_op length {len(has_bare_op)} is shorter than step count {n}."
        )
    quant = [0]
    for b in has_bare_op[:n]:
        quant.append(quant[-1] + (1 if b else 0))
    full = [
        (sm.step_to_unrolled[s] if s < n else len(ir.instrs)) - s + quant[s]
        for s in range(n + 1)
    ]
    return TableauStepLayout(sm.step_to_unrolled, tuple(quant), tuple(full))


@dataclass(frozen=True)
class CopyMap:
    """Flattened-level copy-split map produced by ``split_copies``."""

    ops: tuple[stim.CircuitInstruction, ...]
    flat_to_op: tuple[int, ...]
    copy_len: tuple[int, ...]  # per flat op; 0 = not split


def split_copies(
    ir: CircuitIR, copy_len_of: Callable[[stim.CircuitInstruction], int]
) -> CopyMap:
    """Split fused flat ops using a consumer-injected ``copy_len_of`` rule.

    Records each raw value returned by ``copy_len_of`` in ``CopyMap.copy_len``
    (values ``<= 0`` leave the op unsplit via ``split_fused_instruction``).
    For ``from_op_list=True`` IRs, ``ir.flat_ops`` is ``()`` so this returns an
    empty ``CopyMap((), (), ())``.
    """
    ops: list[stim.CircuitInstruction] = []
    flat_to_op: list[int] = []
    copy_len: list[int] = []
    for op in ir.flat_ops:
        n = copy_len_of(op)
        flat_to_op.append(len(ops))
        copy_len.append(n)
        ops.extend(split_fused_instruction(op, n))
    return CopyMap(tuple(ops), tuple(flat_to_op), tuple(copy_len))


def unrolled_to_flat(ir: CircuitIR) -> list[int]:
    """Return ``flat_index`` for every unrolled instruction in ``ir``."""
    return [i.flat_index for i in ir.instrs]
