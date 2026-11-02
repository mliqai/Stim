"""Moving-reference stabilizer circuit compiler for `CosetsideSimulator`."""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]

from stimside.util.bitwords import pauli_multiply
from stimside.util.reference_chp import (
    DetMeasMeta,
    PackedSymplecticTables,
    RandMeasMeta,
    ReferenceCHPTableau,
    get_cached_gate_tableau,
    pauli_image,
    pauli_image_batch,
)

RawFactor = (
    tuple[Literal["P1", "P2"], tuple[int, ...], tuple[int, ...], int]
    | tuple[
        Literal["PCP"],
        tuple[int, ...],
        tuple[int, ...],
        int,
        tuple[int, ...],
        tuple[int, ...],
        int,
    ]
)

_GATE_MRQF_FACTORS_CACHE: dict[str, tuple[RawFactor, ...]] = {}

# Elementary local decompositions for E = U_gate^dagger in application order (rightmost factor first).
# Entries are either:
# - ("P1" | "P2", local_x_tuple, local_z_tuple, sign_pm1)
# - ("PCP", local_x1, local_z1, sign1, local_x2, local_z2, sign2) for (I + P1 + P2 - P1 P2) / 2
_EXPLICIT_INVERSE_GATE_FACTORS: dict[str, tuple[RawFactor, ...]] = {
    "I": (),
    "II": (),
    "X": (("P1", (1,), (0,), 1),),
    "Y": (("P1", (1,), (1,), 1),),
    "Z": (("P1", (0,), (1,), 1),),
    # S^dagger = R(+Z); S = R(-Z)
    "S": (("P2", (0,), (1,), 1),),
    "S_DAG": (("P2", (0,), (1,), -1),),
    # H^dagger = H = R(Y) X (apply X first, then R(Y))
    "H": (("P1", (1,), (0,), 1), ("P2", (1,), (1,), 1)),
    "H_XZ": (("P1", (1,), (0,), 1), ("P2", (1,), (1,), 1)),
    # H_XY^dagger = H_XY = (X + Y)/sqrt(2) = R(-Z) X
    "H_XY": (("P1", (1,), (0,), 1), ("P2", (0,), (1,), -1)),
    # H_YZ^dagger = H_YZ = (Y + Z)/sqrt(2) = R(+X) Z
    "H_YZ": (("P1", (0,), (1,), 1), ("P2", (1,), (0,), 1)),
    # SQRT_X^dagger = R(+X); SQRT_X = R(-X)
    "SQRT_X": (("P2", (1,), (0,), 1),),
    "SQRT_X_DAG": (("P2", (1,), (0,), -1),),
    # SQRT_Y^dagger = R(+Y); SQRT_Y = R(-Y)
    "SQRT_Y": (("P2", (1,), (1,), 1),),
    "SQRT_Y_DAG": (("P2", (1,), (1,), -1),),
    # Period-3 single-qubit rotations: C_XYZ^dagger = R(X) R(Z), C_ZYX^dagger = R(-Z) R(-X)
    "C_XYZ": (("P2", (0,), (1,), 1), ("P2", (1,), (0,), 1)),
    "C_ZYX": (("P2", (1,), (0,), -1), ("P2", (0,), (1,), -1)),
    # Two-qubit quarter-turn rotations: SQRT_PP^dagger = R(+P_0 P_1)
    "SQRT_XX": (("P2", (1, 1), (0, 0), 1),),
    "SQRT_XX_DAG": (("P2", (1, 1), (0, 0), -1),),
    "SQRT_YY": (("P2", (1, 1), (1, 1), 1),),
    "SQRT_YY_DAG": (("P2", (1, 1), (1, 1), -1),),
    "SQRT_ZZ": (("P2", (0, 0), (1, 1), 1),),
    "SQRT_ZZ_DAG": (("P2", (0, 0), (1, 1), -1),),
    # Controlled-Pauli gates via direct 2-column rank-2 injection: U^dagger = U = (I + P_1 + P_2 - P_1 P_2) / 2
    "CZ": (("PCP", (0, 0), (1, 0), 1, (0, 0), (0, 1), 1),),
    "ZCZ": (("PCP", (0, 0), (1, 0), 1, (0, 0), (0, 1), 1),),
    "CX": (("PCP", (0, 0), (1, 0), 1, (0, 1), (0, 0), 1),),
    "CNOT": (("PCP", (0, 0), (1, 0), 1, (0, 1), (0, 0), 1),),
    "ZCX": (("PCP", (0, 0), (1, 0), 1, (0, 1), (0, 0), 1),),
    "CY": (("PCP", (0, 0), (1, 0), 1, (0, 1), (0, 1), 1),),
    "ZCY": (("PCP", (0, 0), (1, 0), 1, (0, 1), (0, 1), 1),),
    "XCX": (("PCP", (1, 0), (0, 0), 1, (0, 1), (0, 0), 1),),
    "XCY": (("PCP", (1, 0), (0, 0), 1, (0, 1), (0, 1), 1),),
    "XCZ": (("PCP", (1, 0), (0, 0), 1, (0, 0), (0, 1), 1),),
    "YCX": (("PCP", (1, 0), (1, 0), 1, (0, 1), (0, 0), 1),),
    "YCY": (("PCP", (1, 0), (1, 0), 1, (0, 1), (0, 1), 1),),
    "YCZ": (("PCP", (1, 0), (1, 0), 1, (0, 0), (0, 1), 1),),
    # SWAP^dagger = SWAP = R(-X_0 X_1) R(-Y_0 Y_1) R(-Z_0 Z_1)
    "SWAP": (
        ("P2", (0, 0), (1, 1), -1),
        ("P2", (1, 1), (1, 1), -1),
        ("P2", (1, 1), (0, 0), -1),
    ),
    # ISWAP^dagger = R(-Y_0 Y_1) R(-X_0 X_1); ISWAP_DAG^dagger = R(+Y_0 Y_1) R(+X_0 X_1)
    "ISWAP": (
        ("P2", (1, 1), (0, 0), -1),
        ("P2", (1, 1), (1, 1), -1),
    ),
    "ISWAP_DAG": (
        ("P2", (1, 1), (0, 0), 1),
        ("P2", (1, 1), (1, 1), 1),
    ),
}

# Forward elementary decompositions used when decomposing general inverse tableaus via to_circuit().
_FORWARD_ELEMENTARY_FACTORS: dict[str, tuple[RawFactor, ...]] = {
    "X": (("P1", (1,), (0,), 1),),
    "Y": (("P1", (1,), (1,), 1),),
    "Z": (("P1", (0,), (1,), 1),),
    "H": (("P1", (1,), (0,), 1), ("P2", (1,), (1,), 1)),
    "S": (("P2", (0,), (1,), -1),),
    "S_DAG": (("P2", (0,), (1,), 1),),
    "CX": _EXPLICIT_INVERSE_GATE_FACTORS["CX"],
    "CZ": _EXPLICIT_INVERSE_GATE_FACTORS["CZ"],
}


def get_inverse_clifford_mrqf_factors(
    gate_name: str,
) -> tuple[RawFactor, ...]:
    """Return the ordered tuple of MRQF primitive factors for E = U_gate^dagger."""
    cached = _GATE_MRQF_FACTORS_CACHE.get(gate_name)
    if cached is not None:
        return cached
    if gate_name in _EXPLICIT_INVERSE_GATE_FACTORS:
        res = _EXPLICIT_INVERSE_GATE_FACTORS[gate_name]
        _GATE_MRQF_FACTORS_CACHE[gate_name] = res
        return res

    _, inv_tab = get_cached_gate_tableau(gate_name)
    arity = len(inv_tab)
    circ = inv_tab.to_circuit(method="elimination")
    factors: list[RawFactor] = []
    for inst in circ:
        sub_factors = _FORWARD_ELEMENTARY_FACTORS[inst.name]
        targets = [t.value for t in inst.targets_copy()]
        step = len(sub_factors[0][1])
        for i in range(0, len(targets), step):
            grp = targets[i : i + step]
            for f in sub_factors:
                if f[0] == "PCP":
                    _, lx1, lz1, sgn1, lx2, lz2, sgn2 = f
                    fx1, fz1, fx2, fz2 = ([0] * arity for _ in range(4))
                    for r_local, q_local in enumerate(grp):
                        fx1[q_local], fz1[q_local] = lx1[r_local], lz1[r_local]
                        fx2[q_local], fz2[q_local] = lx2[r_local], lz2[r_local]
                    factors.append(("PCP", tuple(fx1), tuple(fz1), sgn1, tuple(fx2), tuple(fz2), sgn2))
                else:
                    kind, lx, lz, sgn = f
                    fx, fz = [0] * arity, [0] * arity
                    for r_local, q_local in enumerate(grp):
                        fx[q_local], fz[q_local] = lx[r_local], lz[r_local]
                    factors.append((kind, tuple(fx), tuple(fz), sgn))
    res = tuple(factors)
    _GATE_MRQF_FACTORS_CACHE[gate_name] = res
    return res


@dataclasses.dataclass
class CompiledFaultFactor:
    """Single compiled P1 (Pauli), P2 (quarter-turn), or PCP (controlled-Pauli) primitive in reference coordinates B_t."""

    kind: Literal["P1", "P2", "PCP"]
    local_x: NDArray[np.bool_]  # shape (k,)
    local_z: NDArray[np.bool_]  # shape (k,)
    a_words: NDArray[np.uint64]  # shape (n_words,)
    b_words: NDArray[np.uint64]  # shape (n_words,)
    kappa: int  # in {0, 1, 2, 3}
    local_x2: NDArray[np.bool_] | None = None
    local_z2: NDArray[np.bool_] | None = None
    a2_words: NDArray[np.uint64] | None = None
    b2_words: NDArray[np.uint64] | None = None
    kappa2: int = 0


@dataclasses.dataclass
class InverseCliffordCocycle:
    """Compiled MRQF decomposition of E = U_t^dagger at step t+1 on target group K."""

    targets: tuple[int, ...]
    factors: tuple[CompiledFaultFactor, ...]


class _StepWordsMixin:
    _packed_words: PackedSymplecticTables | None

    def _get_packed_words(self, n_words: int) -> PackedSymplecticTables:
        assert self._packed_words is not None
        return self._packed_words

    def get_evict_words(
        self, n_words: int
    ) -> tuple[NDArray[np.uint64], NDArray[np.uint64]]:
        pw = self._get_packed_words(n_words)
        return pw[1], pw[3]


@dataclasses.dataclass
class UnitaryStepMeta(_StepWordsMixin):
    kind: Literal["unitary"]
    op: stim.CircuitInstruction
    gate_name: str
    num_qubits: int
    target_groups: list[tuple[int, ...]]
    quantum_op: stim.CircuitInstruction | None = None
    _group_ops: list[stim.CircuitInstruction] | None = None
    _group_packed_words: list[PackedSymplecticTables] | None = None
    _cocycle_cache: dict[int, InverseCliffordCocycle] = dataclasses.field(default_factory=dict)
    are_groups_disjoint: bool = True
    _packed_words: PackedSymplecticTables | None = None
    _cpp_cocycle_arrays: (
        tuple[int, int, int, NDArray[np.int64], list[NDArray]] | None
    ) = dataclasses.field(default=None, init=False, repr=False, compare=False)

    @property
    def group_ops(self) -> list[stim.CircuitInstruction]:
        if self._group_ops is None:
            self._group_ops = [
                stim.CircuitInstruction(self.gate_name, list(grp))
                for grp in self.target_groups
            ]
        return self._group_ops

    def ensure_group_packed_words(
        self, packed_words_before: PackedSymplecticTables
    ) -> list[PackedSymplecticTables]:
        if self._group_packed_words is None:
            tmp_chp = ReferenceCHPTableau.from_packed(self.num_qubits, packed_words_before)
            group_pws: list[PackedSymplecticTables] = []
            for grp in self.target_groups:
                tmp_chp.do_unitary(self.gate_name, grp)
                group_pws.append(tmp_chp.snapshot_packed())
            self._group_packed_words = group_pws
            self._cocycle_cache.clear()
            self._cpp_cocycle_arrays = None
        return self._group_packed_words

    def get_inverse_cocycle(
        self, g_idx: int, num_qubits: int, n_words: int
    ) -> InverseCliffordCocycle:
        """Lazily compute and cache the compiled MRQF fault factors for E = U_t^dagger on group g_idx."""
        cached = self._cocycle_cache.get(g_idx)
        if cached is not None:
            return cached

        target_group = self.target_groups[g_idx]
        raw_factors = get_inverse_clifford_mrqf_factors(self.gate_name)
        packed_tables = (
            self._group_packed_words[g_idx]
            if self._group_packed_words is not None
            else self._get_packed_words(n_words)
        )

        single_q_cache: dict[
            tuple[int, int, int],
            tuple[NDArray[np.uint64], NDArray[np.uint64], int],
        ] = {}

        def _compile_pauli_tuple(
            lx: tuple[int, ...], lz: tuple[int, ...], sgn: int
        ) -> tuple[NDArray[np.uint64], NDArray[np.uint64], int]:
            active_terms: list[
                tuple[NDArray[np.uint64], NDArray[np.uint64], int]
            ] = []
            for r in range(len(target_group)):
                bx, bz = lx[r], lz[r]
                if bx == 0 and bz == 0:
                    continue
                key = (r, bx, bz)
                term = single_q_cache.get(key)
                if term is None:
                    pauli = {(1, 0): "X", (0, 1): "Z"}.get((bx, bz), "Y")
                    term = pauli_image(pauli, target_group[r], *packed_tables)
                    single_q_cache[key] = term
                active_terms.append(term)

            if len(active_terms) == 1:
                a_w, b_w, kap = active_terms[0]
                a_w, b_w = a_w.copy(), b_w.copy()
            else:
                a_w, b_w, kap = pauli_multiply(*active_terms[0], *active_terms[1])

            if sgn == -1:
                kap = (kap + 2) & 3
            return a_w, b_w, kap

        compiled_factors: list[CompiledFaultFactor] = []
        for f in raw_factors:
            if f[0] == "PCP":
                _, lx1, lz1, sgn1, lx2, lz2, sgn2 = f
                aw1, bw1, kap1 = _compile_pauli_tuple(lx1, lz1, sgn1)
                aw2, bw2, kap2 = _compile_pauli_tuple(lx2, lz2, sgn2)
                compiled_factors.append(
                    CompiledFaultFactor(
                        kind="PCP",
                        local_x=np.asarray(lx1, dtype=np.bool_),
                        local_z=np.asarray(lz1, dtype=np.bool_),
                        a_words=aw1,
                        b_words=bw1,
                        kappa=kap1,
                        local_x2=np.asarray(lx2, dtype=np.bool_),
                        local_z2=np.asarray(lz2, dtype=np.bool_),
                        a2_words=aw2,
                        b2_words=bw2,
                        kappa2=kap2,
                    )
                )
            else:
                kind, lx, lz, sgn = f
                a_w, b_w, kap = _compile_pauli_tuple(lx, lz, sgn)
                compiled_factors.append(
                    CompiledFaultFactor(
                        kind=kind,
                        local_x=np.asarray(lx, dtype=np.bool_),
                        local_z=np.asarray(lz, dtype=np.bool_),
                        a_words=a_w,
                        b_words=b_w,
                        kappa=kap,
                    )
                )

        cocycle = InverseCliffordCocycle(
            targets=target_group,
            factors=tuple(compiled_factors),
        )
        self._cocycle_cache[g_idx] = cocycle
        return cocycle

    def get_cpp_cocycle_arrays(
        self, n_words: int
    ) -> tuple[int, int, int, NDArray[np.int64], list[NDArray]]:
        """Vectorized compilation of C++ cocycle arrays across all disjoint target groups."""
        if self._cpp_cocycle_arrays is not None:
            return self._cpp_cocycle_arrays
        num_g = len(self.target_groups)
        arity = 2 if stim.gate_data(self.gate_name).is_two_qubit_gate else 1
        raw_factors = get_inverse_clifford_mrqf_factors(self.gate_name)
        n_fac = len(raw_factors)
        t_groups = (
            np.array(self.target_groups, dtype=np.int64)
            if num_g > 0
            else np.empty((0, arity), dtype=np.int64)
        )
        packed_tables = self._get_packed_words(n_words)
        slot_cache: dict[
            tuple[int, int, int],
            tuple[NDArray[np.uint64], NDArray[np.uint64], NDArray[np.uint8]],
        ] = {}

        def _compile_batch_tuple(
            lx: tuple[int, ...], lz: tuple[int, ...], sgn: int
        ) -> tuple[NDArray[np.uint64], NDArray[np.uint64], NDArray[np.uint8]]:
            terms = []
            for r, (bx, bz) in enumerate(zip(lx, lz)):
                if bx == 0 and bz == 0:
                    continue
                key = (r, bx, bz)
                term = slot_cache.get(key)
                if term is None:
                    pauli = "X" if (bx, bz) == (1, 0) else ("Z" if (bx, bz) == (0, 1) else "Y")
                    term = pauli_image_batch(pauli, t_groups[:, r], *packed_tables)
                    slot_cache[key] = term
                terms.append(term)
            if len(terms) == 1:
                a_w, b_w, kap = terms[0]
                return a_w, b_w, (((kap + 2) & 3).astype(np.uint8) if sgn == -1 else kap)
            a1, b1, k1 = terms[0]
            a2, b2, k2 = terms[1]
            dp = (
                (np.bitwise_count(b1[:, 0] & a2[:, 0]).astype(np.int32) & 1)
                if n_words == 1
                else (np.bitwise_count(b1 & a2).sum(axis=1, dtype=np.int32) & 1)
            )
            add_s = 2 if sgn == -1 else 0
            kap = ((k1.astype(np.int32) + k2.astype(np.int32) + (dp << 1) + add_s) & 3).astype(np.uint8)
            return a1 ^ a2, b1 ^ b2, kap

        g_f = (num_g, n_fac)
        shapes = [g_f, (*g_f, arity), (*g_f, n_words)]
        f_arrs = [
            np.zeros(shapes[s], dtype=np.uint64 if s == 2 else np.uint8)
            for s in (0, 1, 1, 2, 2, 0, 1, 1, 2, 2, 0)
        ]
        f_kinds, f_lx1, f_lz1, f_a1, f_b1, f_k1 = f_arrs[:6]
        f_lx2, f_lz2, f_a2, f_b2, f_k2 = f_arrs[6:]
        for f_i, f in enumerate(raw_factors):
            if f[0] == "PCP":
                _, lx1, lz1, sgn1, lx2, lz2, sgn2 = f
                aw1, bw1, kap1 = _compile_batch_tuple(lx1, lz1, sgn1)
                aw2, bw2, kap2 = _compile_batch_tuple(lx2, lz2, sgn2)
                f_kinds[:, f_i] = 2
                f_lx1[:, f_i] = lx1
                f_lz1[:, f_i] = lz1
                f_a1[:, f_i] = aw1
                f_b1[:, f_i] = bw1
                f_k1[:, f_i] = kap1
                f_lx2[:, f_i] = lx2
                f_lz2[:, f_i] = lz2
                f_a2[:, f_i] = aw2
                f_b2[:, f_i] = bw2
                f_k2[:, f_i] = kap2
            else:
                kind, lx, lz, sgn = f
                aw, bw, kap = _compile_batch_tuple(lx, lz, sgn)
                f_kinds[:, f_i] = 0 if kind == "P1" else 1
                f_lx1[:, f_i] = lx
                f_lz1[:, f_i] = lz
                f_a1[:, f_i] = aw
                f_b1[:, f_i] = bw
                f_k1[:, f_i] = kap
        self._cpp_cocycle_arrays = (num_g, arity, n_fac, t_groups, f_arrs)
        return self._cpp_cocycle_arrays



@dataclasses.dataclass
class MeasSubStepMeta:
    """Single measurement or reset substep within a MeasStepMeta instruction."""

    q: int
    invert_result: bool
    pre_ops: list[stim.CircuitInstruction]
    post_ops: list[stim.CircuitInstruction]
    meta: DetMeasMeta | RandMeasMeta


@dataclasses.dataclass
class MeasStepMeta(_StepWordsMixin):
    """Compiled step descriptor for a measurement or reset instruction."""

    kind: Literal["meas"]
    op: stim.CircuitInstruction
    gate_name: str
    produces_measurements: bool
    is_reset: bool
    substeps: list[MeasSubStepMeta]
    _packed_words: PackedSymplecticTables | None = None
    num_qubits: int = 0
    _det_batch_cache: (
        tuple[NDArray[np.int64], NDArray[np.bool_], NDArray[np.uint64]] | None
    ) = None

    def get_det_batch_arrays(
        self,
    ) -> tuple[NDArray[np.int64], NDArray[np.bool_], NDArray[np.uint64]]:
        if self._det_batch_cache is None:
            num_sub = len(self.substeps)
            q_arr = np.fromiter(
                (s.meta.q for s in self.substeps), dtype=np.int64, count=num_sub
            )
            m_ref_arr = np.fromiter(
                (bool(s.meta.m_ref) for s in self.substeps),
                dtype=np.bool_,
                count=num_sub,
            )
            b_sub_words = np.ascontiguousarray(
                np.vstack(
                    [s.meta.beta_Z_words for s in self.substeps]  # type: ignore[union-attr]
                ),
                dtype=np.uint64,
            )
            self._det_batch_cache = (q_arr, m_ref_arr, b_sub_words)
        return self._det_batch_cache


@dataclasses.dataclass
class NoopStepMeta(_StepWordsMixin):
    """Compiled step descriptor for noise/annotation instructions that leave the reference state invariant."""

    kind: Literal["noop"]
    op: stim.CircuitInstruction
    _packed_words: PackedSymplecticTables | None = None
    num_qubits: int = 0


StepMeta = UnitaryStepMeta | MeasStepMeta | NoopStepMeta


class CompiledReferenceCircuit:
    """Precompiles a stim.Circuit into a Phase 0 reference step tape without stim.Tableau conjugation."""

    def __init__(self, circuit: stim.Circuit) -> None:
        self.circuit = circuit
        self.num_qubits = circuit.num_qubits
        self.n_words = max(1, (self.num_qubits + 63) // 64)
        self.ref_chp = ReferenceCHPTableau(self.num_qubits)
        self._last_packed_snapshot = self.ref_chp.snapshot_packed()
        self._initial_packed_words = self._last_packed_snapshot
        self.steps: list[StepMeta] = []
        self._compile_block(circuit)

    def _get_packed_snapshot(self) -> PackedSymplecticTables:
        if self.ref_chp._inv_dirty:
            self._last_packed_snapshot = self.ref_chp.snapshot_packed()
            self.ref_chp._inv_dirty = False
        return self._last_packed_snapshot

    def _compile_block(
        self,
        block: stim.Circuit | stim.CircuitRepeatBlock | stim.CircuitInstruction,
    ) -> None:
        if isinstance(block, stim.Circuit):
            for op in block:
                self._compile_block(op)
        elif isinstance(block, stim.CircuitRepeatBlock):
            body = block.body_copy()
            for _ in range(block.repeat_count):
                self._compile_block(body)
        elif isinstance(block, stim.CircuitInstruction):
            if block.name in ("QUBIT_COORDS", "SHIFT_COORDS"):
                return
            self.compile_and_append_instruction(block)

    def _append_noop(self, op: stim.CircuitInstruction) -> NoopStepMeta:
        step = NoopStepMeta(
            kind="noop",
            op=op,
            _packed_words=self._last_packed_snapshot,
            num_qubits=self.num_qubits,
        )
        self.steps.append(step)
        return step

    def compile_and_append_instruction(
        self, op: stim.CircuitInstruction
    ) -> StepMeta:
        """Compile a single non-coordinate instruction and advance the live ReferenceCHPTableau."""
        name = op.name
        gd = stim.gate_data(name)

        if gd.is_reset or gd.produces_measurements:
            if name in ("MPAD", "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"):
                return self._append_noop(op)

            substeps: list[MeasSubStepMeta] = []
            raw_targets = op.targets_copy()

            def _add_substep(
                q_sub: int,
                invert_res: bool,
                pre_ops: list[stim.CircuitInstruction],
                post_ops: list[stim.CircuitInstruction],
                is_reset: bool = False,
            ) -> None:
                for p_op in pre_ops:
                    self.ref_chp.do_unitary(
                        p_op.name, [t.qubit_value for t in p_op.targets_copy()]
                    )
                meta = self.ref_chp.measure_z(
                    q_sub, is_reset=is_reset, m_ref_override=0
                )
                for p_op in post_ops:
                    self.ref_chp.do_unitary(
                        p_op.name, [t.qubit_value for t in p_op.targets_copy()]
                    )
                substeps.append(
                    MeasSubStepMeta(
                        q=q_sub,
                        invert_result=invert_res,
                        pre_ops=pre_ops,
                        post_ops=post_ops,
                        meta=meta,
                    )
                )

            if name == "MPP":
                idx = 0
                product_terms: list[list[stim.GateTarget]] = []
                while idx < len(raw_targets):
                    term = [raw_targets[idx]]
                    idx += 1
                    while idx < len(raw_targets) and raw_targets[idx].is_combiner:
                        term.append(raw_targets[idx + 1])
                        idx += 2
                    product_terms.append(term)

                for term in product_terms:
                    invert_res = False
                    pre_ops: list[stim.CircuitInstruction] = []
                    post_ops: list[stim.CircuitInstruction] = []
                    term_qubits: list[int] = []
                    for gt in term:
                        q_r = gt.qubit_value
                        term_qubits.append(q_r)
                        if gt.is_inverted_result_target:
                            invert_res = not invert_res
                        if gt.is_x_target or gt.is_y_target:
                            b_op = stim.CircuitInstruction("H" if gt.is_x_target else "H_YZ", [q_r])
                            pre_ops.append(b_op)
                            post_ops.insert(0, b_op)
                    q0 = term_qubits[0]
                    cx_ops = [stim.CircuitInstruction("CX", [q_r, q0]) for q_r in term_qubits[1:]]
                    pre_ops.extend(cx_ops)
                    post_ops = list(reversed(cx_ops)) + post_ops
                    _add_substep(q0, invert_res, pre_ops, post_ops)
            elif name in ("MXX", "MYY", "MZZ"):
                basis_pair = name[-1]
                for i in range(0, len(raw_targets), 2):
                    gt0, gt1 = raw_targets[i], raw_targets[i + 1]
                    q0, q1 = gt0.qubit_value, gt1.qubit_value
                    invert_res = bool(gt0.is_inverted_result_target ^ gt1.is_inverted_result_target)
                    pre_ops = []
                    post_ops = []
                    if basis_pair in ("X", "Y"):
                        b_op = stim.CircuitInstruction("H" if basis_pair == "X" else "H_YZ", [q0, q1])
                        pre_ops.append(b_op)
                        post_ops.append(b_op)
                    cx_op = stim.CircuitInstruction("CX", [q1, q0])
                    pre_ops.append(cx_op)
                    post_ops.insert(0, cx_op)
                    _add_substep(q0, invert_res, pre_ops, post_ops)
            else:
                basis_gate = "H" if name.endswith("X") else ("H_YZ" if name.endswith("Y") else None)
                for gt in raw_targets:
                    q = gt.qubit_value
                    ops = [stim.CircuitInstruction(basis_gate, [q])] if basis_gate else []
                    _add_substep(q, bool(gt.is_inverted_result_target), list(ops), list(ops), is_reset=gd.is_reset)

            step = MeasStepMeta(
                kind="meas",
                op=op,
                gate_name=name,
                produces_measurements=gd.produces_measurements,
                is_reset=gd.is_reset,
                substeps=substeps,
                _packed_words=self._get_packed_snapshot(),
                num_qubits=self.num_qubits,
            )
            self.steps.append(step)
            return step

        if gd.is_unitary:
            if name in ("SPP", "SPP_DAG"):
                raise NotImplementedError(
                    f"CosetsideSimulator does not support {name} (Pauli product rotations). Rewrite them "
                    "with single- and two-qubit Clifford gates, or use TablesideSimulator or FlipsideSimulator."
                )
            raw_t = op.targets_copy()
            if gd.is_two_qubit_gate and any(
                t.qubit_value is None for t in raw_t
            ):
                quantum_targets: list[int] = []
                for i in range(0, len(raw_t), 2):
                    q0, q1 = raw_t[i].qubit_value, raw_t[i + 1].qubit_value
                    if q0 is not None and q1 is not None:
                        quantum_targets.extend([q0, q1])
                if not quantum_targets:
                    return self._append_noop(op)
                targets = quantum_targets
                quantum_op = stim.CircuitInstruction(
                    name, quantum_targets, op.gate_args_copy()
                )
            else:
                targets = [t.qubit_value for t in raw_t]
                quantum_op = op
            arity = 2 if gd.is_two_qubit_gate else 1
            groups = [
                tuple(targets[i : i + arity])
                for i in range(0, len(targets), arity)
            ]
            is_disjoint = len(set(targets)) == len(targets)
            if is_disjoint:
                self.ref_chp.do_unitary(name, targets)
                packed_after = self._get_packed_snapshot()
                group_packed_words: list[PackedSymplecticTables] | None = None
            else:
                group_packed_words = []
                for grp in groups:
                    self.ref_chp.do_unitary(name, grp)
                    group_packed_words.append(self.ref_chp.snapshot_packed())
                self._last_packed_snapshot = group_packed_words[-1]
                self.ref_chp._inv_dirty = False
                packed_after = self._last_packed_snapshot
            step = UnitaryStepMeta(
                kind="unitary",
                op=op,
                gate_name=name,
                num_qubits=self.num_qubits,
                target_groups=groups,
                quantum_op=quantum_op,
                _group_packed_words=group_packed_words,
                _packed_words=packed_after,
                are_groups_disjoint=is_disjoint,
            )
            self.steps.append(step)
            return step

        return self._append_noop(op)
