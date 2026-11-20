"""Tag-agnostic circuit intermediate representation (IR).

Design and behaviour:
- Imports only stdlib, ``stim``, and sibling ``util`` helpers; carries no tag
  literals, native-ABI names, or engine-specific constants.
- Never inspects or parses instruction tags: ``op.tag`` is copied verbatim into
  ``IRInstr`` as a plain string. Per-consumer lazy tag parsing lives outside
  this module (see ``ir_tags.py``).
- Unrolled instructions follow ``unroll_circuit`` order. ``flat_ops``,
  ``flat_index``, and ``flat_offset`` are computed lazily on first request from
  a snapshot of the input circuit; the first alignment runs in O(N) time and
  raises ``ValueError`` if ``flattened()`` disagrees with unrolling.
- ``IRInstr`` and ``CircuitIR`` are immutable (frozen) and use identity
  equality and hashing.
- Derived passes (``step_map``, ``tableau_step_layout``, ``split_copies``,
  ``unrolled_to_flat``) are pure functions over ``CircuitIR``, not IR fields.
"""

from __future__ import annotations

import gc
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Self

import stim

from stimside.util.circuit import unroll_circuit
from stimside.util.stim_workarounds import split_fused_instruction

PauliBasis = Literal["X", "Y", "Z"]
UnitaryArity = Literal[1, 2]

_ANNOTATION = frozenset(
    {"DETECTOR", "TICK", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"}
)
_HERALDED = frozenset({"HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"})
_CORRELATED = frozenset({"E", "ELSE_CORRELATED_ERROR"})
_PAULI_TERMS = frozenset({"SPP", "SPP_DAG"})
_PRODUCT_MEAS = frozenset({"MPP", "MXX", "MYY", "MZZ"})
_BASIS_BY_CANONICAL: dict[str, PauliBasis] = {
    "MX": "X",
    "MRX": "X",
    "RX": "X",
    "MXX": "X",
    "MY": "Y",
    "MRY": "Y",
    "RY": "Y",
    "MYY": "Y",
    "M": "Z",
    "MR": "Z",
    "R": "Z",
    "MZZ": "Z",
}
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
    """Untagged-instruction filter class when ``uc=True``."""

    NONE = "none"
    ONE_QUBIT = "one_qubit"
    TWO_QUBIT = "two_qubit"
    CORRELATED = "correlated"
    PAULI_TERMS = "pauli_terms"


_GateMeta = tuple[
    Category,
    UntaggedFilter,
    PauliBasis | None,
    bool,
    UnitaryArity | None,
    bool,
    bool,
]


def _classify_category_from_gd(gd: Any) -> Category:
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


def _classify_untagged_from_gd(gd: Any) -> UntaggedFilter:
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


def _classify_arity_from_gd(gd: Any) -> UnitaryArity | None:
    if not gd.is_unitary:
        return None
    if gd.is_single_qubit_gate:
        return 1
    if gd.is_two_qubit_gate:
        return 2
    return None


def _meta_from_gd(gd: Any) -> _GateMeta:
    canon = gd.name
    cat = _classify_category_from_gd(gd)
    uf_uc = _classify_untagged_from_gd(gd)
    basis = _BASIS_BY_CANONICAL.get(canon)
    is_pm = canon in _PRODUCT_MEAS
    arity = _classify_arity_from_gd(gd)
    can_cc = bool(gd.is_two_qubit_gate and gd.takes_measurement_record_targets)
    prod_m = bool(gd.produces_measurements)
    return (cat, uf_uc, basis, is_pm, arity, can_cc, prod_m)


def _build_gate_meta_tables() -> tuple[dict[str, _GateMeta], dict[str, _GateMeta]]:
    meta_uc: dict[str, _GateMeta] = {}
    meta_no_uc: dict[str, _GateMeta] = {}
    none_uf = UntaggedFilter.NONE
    for gd in stim.gate_data().values():
        entry_uc = _meta_from_gd(gd)
        entry_no_uc: _GateMeta = (
            entry_uc[0],
            none_uf,
            entry_uc[2],
            entry_uc[3],
            entry_uc[4],
            entry_uc[5],
            entry_uc[6],
        )
        for alias in gd.aliases:
            meta_uc[alias] = entry_uc
            meta_no_uc[alias] = entry_no_uc
        meta_uc[gd.name] = entry_uc
        meta_no_uc[gd.name] = entry_no_uc
    return meta_uc, meta_no_uc


_GATE_META, _GATE_META_NO_UC = _build_gate_meta_tables()


def _gate_meta(name: str, *, uc: bool = True) -> _GateMeta:
    table = _GATE_META if uc else _GATE_META_NO_UC
    meta = table.get(name)
    if meta is not None:
        return meta
    meta = table.get(name.upper())
    if meta is not None:
        return meta
    entry = _meta_from_gd(stim.gate_data(name))
    if not uc:
        return (
            entry[0],
            UntaggedFilter.NONE,
            entry[2],
            entry[3],
            entry[4],
            entry[5],
            entry[6],
        )
    return entry


def classify_category(name: str) -> Category:
    """Classify a Stim gate or instruction name into its coarse ``Category``."""
    meta = _GATE_META.get(name)
    if meta is not None:
        return meta[0]
    meta = _GATE_META.get(name.upper())
    if meta is not None:
        return meta[0]
    return _classify_category_from_gd(stim.gate_data(name))


def classify_untagged(name: str, *, uc: bool) -> UntaggedFilter:
    """Classify an untagged gate when conditioning on the unleaked subspace."""
    if not uc:
        return UntaggedFilter.NONE
    meta = _GATE_META.get(name)
    if meta is not None:
        return meta[1]
    meta = _GATE_META.get(name.upper())
    if meta is not None:
        return meta[1]
    return _classify_untagged_from_gd(stim.gate_data(name))


def classify_basis(name: str) -> PauliBasis | None:
    """Return the Pauli basis (``'X'``, ``'Y'``, ``'Z'``) for basis gates, else ``None``."""
    meta = _GATE_META.get(name)
    if meta is not None:
        return meta[2]
    meta = _GATE_META.get(name.upper())
    if meta is not None:
        return meta[2]
    return _BASIS_BY_CANONICAL.get(stim.gate_data(name).name)


def classify_is_product_measurement(name: str) -> bool:
    """Return True iff ``name`` is a multi-qubit product measurement (MPP/MXX/MYY/MZZ)."""
    meta = _GATE_META.get(name)
    if meta is not None:
        return meta[3]
    meta = _GATE_META.get(name.upper())
    if meta is not None:
        return meta[3]
    return stim.gate_data(name).name in _PRODUCT_MEAS


def classify_arity(name: str) -> UnitaryArity | None:
    """Return ``1`` or ``2`` for fixed-arity unitary gates, else ``None``."""
    meta = _GATE_META.get(name)
    if meta is not None:
        return meta[4]
    meta = _GATE_META.get(name.upper())
    if meta is not None:
        return meta[4]
    return _classify_arity_from_gd(stim.gate_data(name))


def op_has_classical_control(op: stim.CircuitInstruction) -> bool:
    """Return True iff ``op`` is a 2-qubit controlled-Pauli op with a rec/sweep target."""
    name = op.name
    meta = _GATE_META.get(name)
    if meta is None:
        meta = _GATE_META.get(name.upper())
    if meta is not None:
        can_cc = meta[5]
    else:
        gd = stim.gate_data(name)
        can_cc = bool(gd.is_two_qubit_gate and gd.takes_measurement_record_targets)
    if not can_cc:
        return False
    return any(t.qubit_value is None for t in op.targets_copy())


class _FlatAlignment:
    """Lazy cache for ``flattened()`` ops and unrolled-to-flat target alignment."""

    __slots__ = ("_circuit", "_flat_ops", "_where")

    def __init__(self, circuit: stim.Circuit) -> None:
        self._circuit: stim.Circuit | None = circuit
        self._flat_ops: tuple[stim.CircuitInstruction, ...] | None = None
        self._where: list[tuple[int, int]] | None = None

    def get_flat_ops(self) -> tuple[stim.CircuitInstruction, ...]:
        circ = self._circuit
        flat_ops = self._flat_ops
        if flat_ops is None:
            assert circ is not None
            flat_ops = tuple(circ.flattened())
            self._flat_ops = flat_ops
        return flat_ops

    def get_where(
        self,
        ops: Sequence[stim.CircuitInstruction] | None = None,
    ) -> list[tuple[int, int]]:
        circ = self._circuit
        where = self._where
        if where is None:
            flat_ops = self.get_flat_ops()
            if ops is None:
                assert circ is not None
                ops = unroll_circuit(circ)
            where = _align_flat_ops(ops, flat_ops)
            self._where = where
            self._circuit = None
        return where


def _ensure_flat_alignment(box: list[Any]) -> _FlatAlignment:
    circ: stim.Circuit | None = box[0]
    align: _FlatAlignment | None = box[1]
    if align is None:
        assert circ is not None
        align = _FlatAlignment(circ)
        box[1] = align
        box[0] = None
    return align


_TUPLE_NEW = tuple.__new__


class _IRInstrBase(tuple[Any, ...]):
    """Tuple-backed storage with C-level itemgetter accessors for ``IRInstr``."""

    __slots__ = ()
    __hash__ = object.__hash__  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        return self is other

    def __ne__(self, other: object) -> bool:
        return self is not other

    def __copy__(self) -> Self:
        return self

    def __reduce_ex__(self, protocol: Any) -> tuple[Any, ...]:
        return (type(self), tuple(self))

    def _reject_order(self, other: object, symbol: str) -> Any:
        if isinstance(other, tuple):
            raise TypeError(
                f"'{symbol}' not supported between instances of "
                f"'{type(self).__name__}' and '{type(other).__name__}'"
            )
        return NotImplemented

    def __lt__(self, other: object) -> Any:
        return self._reject_order(other, "<")

    def __le__(self, other: object) -> Any:
        return self._reject_order(other, "<=")

    def __gt__(self, other: object) -> Any:
        return self._reject_order(other, ">")

    def __ge__(self, other: object) -> Any:
        return self._reject_order(other, ">=")

    unrolled_index = property(operator.itemgetter(0))
    op = property(operator.itemgetter(1))
    tag = property(operator.itemgetter(2))
    category = property(operator.itemgetter(3))
    untagged_filter = property(operator.itemgetter(4))
    num_measurements = property(operator.itemgetter(5))
    first_measurement = property(operator.itemgetter(6))
    basis = property(operator.itemgetter(7))
    is_product_measurement = property(operator.itemgetter(8))
    arity = property(operator.itemgetter(9))
    has_classical_control = property(operator.itemgetter(10))
    _flat = property(operator.itemgetter(11))


@dataclass(frozen=True, eq=False, init=False, repr=False)
class IRInstr(_IRInstrBase):
    """Single unrolled instruction record in ``CircuitIR``.

    Instances use immutable tuple-backed storage with identity equality and
    hashing; ``flat_index`` and ``flat_offset`` are resolved lazily on first
    access.
    """

    __slots__ = ()

    unrolled_index: int  # type: ignore[assignment]
    op: stim.CircuitInstruction  # type: ignore[assignment]
    tag: str  # type: ignore[assignment]
    category: Category  # type: ignore[assignment]
    untagged_filter: UntaggedFilter  # type: ignore[assignment]
    num_measurements: int  # type: ignore[assignment]
    first_measurement: int  # type: ignore[assignment]
    basis: PauliBasis | None  # type: ignore[assignment]
    is_product_measurement: bool  # type: ignore[assignment]
    arity: UnitaryArity | None  # type: ignore[assignment]
    has_classical_control: bool  # type: ignore[assignment]
    _flat: list[Any] | None  # type: ignore[assignment]

    def __new__(
        cls,
        unrolled_index: int,
        op: stim.CircuitInstruction,
        tag: str,
        category: Category,
        untagged_filter: UntaggedFilter,
        num_measurements: int,
        first_measurement: int,
        basis: PauliBasis | None = None,
        is_product_measurement: bool = False,
        arity: UnitaryArity | None = None,
        has_classical_control: bool = False,
        _flat: list[Any] | None = None,
    ) -> Self:
        return _TUPLE_NEW(
            cls,
            (
                unrolled_index,
                op,
                tag,
                category,
                untagged_filter,
                num_measurements,
                first_measurement,
                basis,
                is_product_measurement,
                arity,
                has_classical_control,
                _flat,
            ),
        )

    def __repr__(self) -> str:
        return (
            f"IRInstr(unrolled_index={self.unrolled_index!r}, op={self.op!r}, "
            f"tag={self[2]!r}, category={self.category!r}, "
            f"untagged_filter={self.untagged_filter!r}, "
            f"num_measurements={self.num_measurements!r}, "
            f"first_measurement={self.first_measurement!r}, basis={self.basis!r}, "
            f"is_product_measurement={self.is_product_measurement!r}, "
            f"arity={self.arity!r}, has_classical_control={self.has_classical_control!r})"
        )

    @property
    def flat_index(self) -> int:
        """Index in ``flattened()``, or -1 for ``SHIFT_COORDS`` / op lists."""
        if self._flat is None:
            return -1
        return _ensure_flat_alignment(self._flat).get_where()[self.unrolled_index][0]

    @property
    def flat_offset(self) -> int:
        """First target index inside the corresponding flattened instruction."""
        if self._flat is None:
            return 0
        return _ensure_flat_alignment(self._flat).get_where()[self.unrolled_index][1]


@dataclass(frozen=True, eq=False)
class CircuitIR:
    """Immutable circuit IR built from a ``stim.Circuit`` or an op sequence.

    Uses identity equality and hashing. ``flat_ops`` is computed lazily on
    first access from a snapshot of the input circuit.
    """

    instrs: tuple[IRInstr, ...]
    num_qubits: int
    num_measurements: int
    has_repeat: bool
    uc: bool
    from_op_list: bool
    _flat: list[Any] | None = field(default=None, repr=False, compare=False)

    @property
    def flat_ops(self) -> tuple[stim.CircuitInstruction, ...]:
        """Flattened instruction tuple (``()`` for an op-list source)."""
        if self._flat is None:
            return ()
        return _ensure_flat_alignment(self._flat).get_flat_ops()


def _op_matches_flat(
    op: stim.CircuitInstruction,
    op_targets: list[stim.GateTarget],
    flat_op: stim.CircuitInstruction,
    flat_targets: list[stim.GateTarget],
    used: int,
    flat_args: list[float],
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
    return op.tag == flat_op.tag


def _try_fuse_zero(
    op: stim.CircuitInstruction,
    cur: stim.CircuitInstruction,
    meta_ready: bool,
    flat_args: list[float],
) -> tuple[bool, bool, list[float]]:
    """Return ``(fused, meta_ready, flat_args)`` for a zero-target op."""
    if op.name in _NON_FUSING or op.name != cur.name:
        return False, meta_ready, flat_args
    if not meta_ready:
        flat_args = cur.gate_args_copy()
        meta_ready = True
    fused = op.tag == cur.tag and op.gate_args_copy() == flat_args
    return fused, meta_ready, flat_args


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
                fused_zero, meta_ready, flat_args = _try_fuse_zero(
                    op, flat_ops[f], meta_ready, flat_args
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
                    meta_ready = True
                    if not _op_matches_flat(
                        op, op_targets, cur, flat_targets, 0, flat_args
                    ):
                        raise ValueError(
                            f"Unrolled op {u_idx} ({op}) does not match "
                            f"flattened op {f} ({cur})."
                        )
        elif not _op_matches_flat(
            op, op_targets, flat_ops[f], flat_targets, used, flat_args
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


def _has_cc_cached(
    op: stim.CircuitInstruction,
    cc_memo: dict[stim.CircuitInstruction, bool],
) -> bool:
    # Only for a gate that can take classical control (elsewhere a combiner has no qubit value
    # either). A rec[...] or sweep[...] target always shows in the instruction's text. Most
    # instructions have none, and this test costs less than hashing the instruction for the memo.
    # A tag with the same characters only sends the instruction on to the exact check.
    text = str(op)
    if "rec[" not in text and "sweep[" not in text:
        return False
    cached = cc_memo.get(op)
    if cached is not None:
        return cached
    res = any(t.qubit_value is None for t in op.targets_copy())
    if len(cc_memo) < 4096:
        cc_memo[op] = res
    return res


def _append_repeat_body(
    body: list[stim.CircuitInstruction],
    repeat_count: int,
    uc: bool,
    flat_box: list[Any],
    cc_memo: dict[stim.CircuitInstruction, bool],
    instrs: list[IRInstr],
    idx: int,
    first: int,
) -> tuple[int, int]:
    append = instrs.append
    gate_meta_get = (_GATE_META if uc else _GATE_META_NO_UC).get
    body_meta: list[
        tuple[
            stim.CircuitInstruction,
            Category,
            UntaggedFilter,
            int,
            PauliBasis | None,
            bool,
            UnitaryArity | None,
            bool,
        ]
    ] = []
    for op in body:
        meta = gate_meta_get(op.name)
        if meta is None:
            meta = _gate_meta(op.name, uc=uc)
        cat, uf, basis, is_pm, arity, can_cc, prod_m = meta
        k = op.num_measurements if prod_m else 0
        has_cc = _has_cc_cached(op, cc_memo) if can_cc else False
        body_meta.append((op, cat, uf, k, basis, is_pm, arity, has_cc))
    for _ in range(repeat_count):
        for op, cat, uf, k, basis, is_pm, arity, has_cc in body_meta:
            append(
                IRInstr(
                    idx,
                    op,
                    op.tag,
                    cat,
                    uf,
                    k,
                    first,
                    basis,
                    is_pm,
                    arity,
                    has_cc,
                    flat_box,
                )
            )
            idx += 1
            first += k
    return idx, first


def _build_circuit_instrs(
    src: stim.Circuit,
    uc: bool,
    flat_box: list[Any],
) -> tuple[tuple[IRInstr, ...], bool]:
    repeat_cls = stim.CircuitRepeatBlock
    instrs: list[IRInstr] = []
    append = instrs.append
    gate_meta_get = (_GATE_META if uc else _GATE_META_NO_UC).get
    cc_memo: dict[stim.CircuitInstruction, bool] = {}
    idx = 0
    first = 0
    has_repeat = False
    was_gc = len(src) >= 1000 and gc.isenabled()
    if was_gc:
        gc.disable()
    try:
        for item in src:
            if isinstance(item, repeat_cls):
                has_repeat = True
                body = unroll_circuit(item.body_copy())
                if not body:
                    continue
                idx, first = _append_repeat_body(
                    body, item.repeat_count, uc, flat_box, cc_memo, instrs, idx, first
                )
                continue
            meta = gate_meta_get(item.name)
            if meta is None:
                meta = _gate_meta(item.name, uc=uc)
            cat, uf, basis, is_pm, arity, can_cc, prod_m = meta
            k = item.num_measurements if prod_m else 0
            has_cc = _has_cc_cached(item, cc_memo) if can_cc else False
            append(
                IRInstr(
                    idx,
                    item,
                    item.tag,
                    cat,
                    uf,
                    k,
                    first,
                    basis,
                    is_pm,
                    arity,
                    has_cc,
                    flat_box,
                )
            )
            idx += 1
            first += k
    finally:
        if was_gc:
            gc.enable()
    return tuple(instrs), has_repeat


def _build_oplist_instrs(
    ops: list[stim.CircuitInstruction],
    uc: bool,
) -> tuple[tuple[IRInstr, ...], int, int]:
    instrs: list[IRInstr] = []
    append = instrs.append
    gate_meta_get = (_GATE_META if uc else _GATE_META_NO_UC).get
    op_memo: dict[
        stim.CircuitInstruction,
        tuple[
            Category,
            UntaggedFilter,
            int,
            int,
            PauliBasis | None,
            bool,
            UnitaryArity | None,
            bool,
        ],
    ] = {}
    max_q = -1
    first = 0
    for i, op in enumerate(ops):
        if isinstance(op, stim.CircuitRepeatBlock):
            raise TypeError(
                "build_ir op sequence cannot contain stim.CircuitRepeatBlock; "
                "pass a stim.Circuit or unroll_circuit() output instead."
            )
        entry = op_memo.get(op)
        if entry is None:
            meta = gate_meta_get(op.name)
            if meta is None:
                meta = _gate_meta(op.name, uc=uc)
            cat, uf, basis, is_pm, arity, can_cc, prod_m = meta
            omq = -1
            has_non_qubit = False
            for t in op.targets_copy():
                qv = t.qubit_value
                if qv is None:
                    has_non_qubit = True
                elif qv > omq:
                    omq = qv
            nm = op.num_measurements if prod_m else 0
            has_cc = can_cc and has_non_qubit
            entry = (cat, uf, omq, nm, basis, is_pm, arity, has_cc)
            if len(op_memo) < 4096:
                op_memo[op] = entry
        else:
            cat, uf, omq, nm, basis, is_pm, arity, has_cc = entry
        max_q = max(max_q, omq)
        append(
            IRInstr(
                i,
                op,
                op.tag,
                cat,
                uf,
                nm,
                first,
                basis,
                is_pm,
                arity,
                has_cc,
            )
        )
        first += nm
    num_qubits = 1 + max_q if max_q >= 0 else 0
    return tuple(instrs), num_qubits, first


def build_ir(
    src: stim.Circuit | Sequence[stim.CircuitInstruction],
    *,
    uc: bool = False,
) -> CircuitIR:
    """Build a ``CircuitIR`` from a ``stim.Circuit`` or an explicit op list."""
    if isinstance(src, stim.Circuit):
        flat_box: list[Any] = [src.copy(), None]
        instrs, has_repeat = _build_circuit_instrs(src, uc, flat_box)
        return CircuitIR(
            instrs,
            src.num_qubits,
            src.num_measurements,
            has_repeat,
            uc,
            False,
            flat_box,
        )
    ops = list(src)
    instrs, num_qubits, num_meas = _build_oplist_instrs(ops, uc)
    return CircuitIR(instrs, num_qubits, num_meas, False, uc, True)


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
    if ir._flat is None:
        return [-1] * len(ir.instrs)
    align = _ensure_flat_alignment(ir._flat)
    if align._where is None:
        where = align.get_where([i.op for i in ir.instrs])
    else:
        where = align.get_where()
    return [f for f, _off in where]
