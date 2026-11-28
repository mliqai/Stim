"""Thin Python batch adapter over the native C++ stabilizer-branch store (`_native.branch_kernels`)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import threading
from typing import Any, Callable, Literal, Sequence
import weakref

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]

from stimside._magic.channels import compile_clifford_factors, compile_pauli_product
from stimside._magic.phase_qfe import (
    ImpossiblePostselectionError,
    PhaseQFEComponent,
    apply_reference_tableau_to_statevector,
)
from stimside._magic.query import QueryBudgetExceededError
from stimside._magic.statevector import _apply_pauli_to_dense_le
from stimside._native import branch_kernels as bk
from stimside.util.bitwords import n_words_for_qubits, pauli_multiply, words_any_nonzero
from stimside.util.reference_chp import ReferenceCHPTableau

_DEFAULT_STORE_MAX_NODES: int = bk._BRANCH_STORE_MAX_NODES  # 300_000
_BRANCH_STORE_MAX_NODES: int = _DEFAULT_STORE_MAX_NODES

_CCZ_MASKS: tuple[int, ...] = (1, 2, 4, 3, 5, 6, 7)
_CCZ_EXPS: NDArray[np.uint8] = np.array([1, 1, 1, 7, 7, 7, 1], dtype=np.uint8)
_T_EXPS: NDArray[np.uint8] = np.array([1], dtype=np.uint8)
_TDAG_EXPS: NDArray[np.uint8] = np.array([7], dtype=np.uint8)
_RESET_KICK: dict[str, Literal["X", "Z"]] = {"Z": "X", "Y": "X", "X": "Z"}

_CUMULATIVE_SCALAR_KEYS: tuple[str, ...] = (
    "store_clears",
    "resolve_calls",
    "splits",
    "subcache_hits",
    "subcache_misses",
    "meas_definite",
    "meas_absorbed",
    "meas_forked",
)
_FAULT_KEYS: tuple[str, ...] = ("pauli_faults", "skipped_gates", "skipped_cczs")


def effective_store_max_nodes(sim_cap: int | None = None) -> int:
    """Resolve active store node cap, honoring monkeypatches on simulator, backend, or native."""
    if sim_cap is not None and int(sim_cap) != _DEFAULT_STORE_MAX_NODES:
        return int(sim_cap)
    if int(_BRANCH_STORE_MAX_NODES) != _DEFAULT_STORE_MAX_NODES:
        return int(_BRANCH_STORE_MAX_NODES)
    if int(bk._BRANCH_STORE_MAX_NODES) != _DEFAULT_STORE_MAX_NODES:
        return int(bk._BRANCH_STORE_MAX_NODES)
    return _DEFAULT_STORE_MAX_NODES if sim_cap is None else int(sim_cap)


def empty_retired_counters() -> dict[str, Any]:
    """Zero-initialized cumulative counter accumulator for retired shared stores."""
    out: dict[str, Any] = {k: 0 for k in _CUMULATIVE_SCALAR_KEYS}
    out["transitions_new"] = {name: 0 for name in bk.MBR_KIND_NAMES}
    out["transitions_hit"] = {name: 0 for name in bk.MBR_KIND_NAMES}
    return out


def copy_retired_counters(src: dict[str, Any]) -> dict[str, Any]:
    out = {k: int(src[k]) for k in _CUMULATIVE_SCALAR_KEYS}
    out["transitions_new"] = dict(src["transitions_new"])
    out["transitions_hit"] = dict(src["transitions_hit"])
    return out


def accumulate_retired_counters(dst: dict[str, Any], snap: dict[str, Any], *, extra_clears: int = 0) -> None:
    for k in _CUMULATIVE_SCALAR_KEYS:
        dst[k] = int(dst[k]) + int(snap[k])
    dst["store_clears"] = int(dst["store_clears"]) + int(extra_clears)
    for dkey in ("transitions_new", "transitions_hit"):
        for name in bk.MBR_KIND_NAMES:
            dst[dkey][name] = int(dst[dkey][name]) + int(snap[dkey][name])


def empty_faults_realized() -> dict[str, int]:
    return {k: 0 for k in _FAULT_KEYS}


def _anti_batch(
    fx: NDArray[np.uint64],
    fz: NDArray[np.uint64],
    px: NDArray[np.uint64],
    pz: NDArray[np.uint64],
) -> NDArray[np.uint8]:
    """Return `popcount((fx & pz) ^ (fz & px)) & 1` as `(M,) uint8` for arbitrary `nw >= 1`."""
    nw = fx.shape[1]
    acc = (fx[:, 0] & pz[0]) ^ (fz[:, 0] & px[0])
    for w in range(1, nw):
        acc ^= (fx[:, w] & pz[w]) ^ (fz[:, w] & px[w])
    return (np.bitwise_count(acc) & np.uint8(1)).astype(np.uint8, copy=False)


def _safe_log(p: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.full(p.shape, -math.inf, dtype=np.float64)
    pos = p > 0.0
    out[pos] = np.log(p[pos])
    return out


class _StoreOwnerToken:
    """Weakly-referenced ownership token held by each `MagicsideSimulator`."""


_FORK_POISON_MSG = "BranchStore was forked while busy in another thread"


class _PoisonedNativeStore:
    """Sentinel replacing `native` in a forked child when another thread was inside C++ at fork."""

    def __getattr__(self, _name: str) -> Any:
        raise RuntimeError(_FORK_POISON_MSG)

    def close(self) -> None:
        return None


_FORK_STORES: weakref.WeakSet[_SharedBranchStore] = weakref.WeakSet()


def _reinit_branch_stores_in_child() -> None:
    for store in list(_FORK_STORES):
        store._reinit_after_fork()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reinit_branch_stores_in_child)


class _SharedBranchStore:
    """Thread-safe shared wrapper over `bk.BranchStore` with hash-consed frames and event cache."""

    def __init__(self, n_qubits: int, max_components: int = 0) -> None:
        self.lock = threading.RLock()
        self._busy: bool = False
        self._busy_tid: int = 0
        self._fork_poisoned: bool = False
        self.n_qubits = int(n_qubits)
        self.nw = n_words_for_qubits(self.n_qubits)
        self.max_components = int(max_components)
        self.native = bk.BranchStore(self.n_qubits, max_components=self.max_components)
        self._owners: weakref.WeakSet[_StoreOwnerToken] = weakref.WeakSet()
        self.frame_trie: dict[tuple[int, Any], int] = {}
        self._next_frame_id: int = 1
        self.event_ids: dict[Any, int] = {}
        self.event_data: dict[int, Any] = {}
        self._next_event_id: int = 1
        self.node_peak_cache: dict[int, int] = {0: 1}
        _FORK_STORES.add(self)

    def _reinit_after_fork(self) -> None:
        """Run in a forked child: renew `lock` and poison stores mid-native-call in another thread."""
        self.lock = threading.RLock()
        if self._busy and self._busy_tid != threading.get_ident():
            self._fork_poisoned = True
            self._busy = False
            self._busy_tid = 0
            if isinstance(self.native, bk.BranchStore):
                # Drop C handle without calling mbr_store_destroy on potentially torn memory.
                self.native._handle = 0
            self.native = _PoisonedNativeStore()  # type: ignore[assignment]
        elif not self._busy:
            self._busy_tid = 0

    def _check_not_poisoned(self) -> None:
        if self._fork_poisoned:
            raise RuntimeError(_FORK_POISON_MSG)

    def register_owner(self, token: _StoreOwnerToken) -> None:
        with self.lock:
            self._owners.add(token)

    def owner_count(self) -> int:
        with self.lock:
            return len(self._owners)

    def set_max_components(self, max_components: int) -> None:
        mc = int(max_components)
        with self.lock:
            self._check_not_poisoned()
            if mc != self.max_components:
                prev_busy, prev_tid = self._busy, self._busy_tid
                self._busy, self._busy_tid = True, threading.get_ident()
                try:
                    self.native.set_max_components(mc)
                    self.max_components = mc
                finally:
                    self._busy, self._busy_tid = prev_busy, prev_tid

    def clear_inplace(self) -> None:
        """Reset trie and Python caches in-place (caller must verify sole ownership at batch start)."""
        with self.lock:
            self._check_not_poisoned()
            prev_busy, prev_tid = self._busy, self._busy_tid
            self._busy, self._busy_tid = True, threading.get_ident()
            try:
                self.native.clear()
                self.frame_trie.clear()
                self._next_frame_id = 1
                self.event_ids.clear()
                self.event_data.clear()
                self._next_event_id = 1
                self.node_peak_cache.clear()
                self.node_peak_cache[0] = 1
            finally:
                self._busy, self._busy_tid = prev_busy, prev_tid

    def advance_frame(self, frame_id: int, step_key: Any) -> int:
        edge = (frame_id, step_key)
        with self.lock:
            self._check_not_poisoned()
            nxt = self.frame_trie.get(edge)
            if nxt is None:
                nxt = self._next_frame_id
                self._next_frame_id += 1
                self.frame_trie[edge] = nxt
            return nxt

    def get_or_compile_event(self, event_key: Any, compile_fn: Callable[[], Any]) -> tuple[int, Any]:
        with self.lock:
            self._check_not_poisoned()
            eid = self.event_ids.get(event_key)
            if eid is not None:
                return eid, self.event_data[eid]
            payload = compile_fn()
            eid = self._next_event_id
            self._next_event_id += 1
            self.event_data[eid] = payload
            self.event_ids[event_key] = eid
            return eid, payload

    def node_peak_unlocked(self, nid: int) -> int:
        self._check_not_poisoned()
        if nid < 0:
            return 1
        pk = self.node_peak_cache.get(nid)
        if pk is None:
            prev_busy, prev_tid = self._busy, self._busy_tid
            self._busy, self._busy_tid = True, threading.get_ident()
            try:
                nb, _, tpk = self.native.node_info(nid)
            finally:
                self._busy, self._busy_tid = prev_busy, prev_tid
            pk = min(32767, max(nb, tpk))
            self.node_peak_cache[nid] = pk
        return pk

    def step_rotations_unlocked(
        self,
        nid: int,
        eid: int,
        kind: int,
        sign_bits: int,
        ax_x: NDArray[np.uint64],
        ax_z: NDArray[np.uint64],
        ax_k: NDArray[np.uint8],
        ax_e: NDArray[np.uint8],
    ) -> tuple[int, int]:
        self._check_not_poisoned()
        prev_busy, prev_tid = self._busy, self._busy_tid
        self._busy, self._busy_tid = True, threading.get_ident()
        try:
            cid = self.native.step_rotations(nid, eid, kind, sign_bits, ax_x, ax_z, ax_k, ax_e)
        except bk.BranchBudgetError as exc:
            raise QueryBudgetExceededError(str(exc)) from exc
        finally:
            self._busy, self._busy_tid = prev_busy, prev_tid
        return cid, self.node_peak_unlocked(cid)

    def step_qturns_unlocked(
        self,
        nid: int,
        eid: int,
        kind: int,
        sign_bits: int,
        qt_x: NDArray[np.uint64],
        qt_z: NDArray[np.uint64],
        qt_k: NDArray[np.uint8],
    ) -> tuple[int, int]:
        self._check_not_poisoned()
        prev_busy, prev_tid = self._busy, self._busy_tid
        self._busy, self._busy_tid = True, threading.get_ident()
        try:
            cid = self.native.step_qturns(nid, eid, kind, sign_bits, qt_x, qt_z, qt_k)
        except bk.BranchBudgetError as exc:
            raise QueryBudgetExceededError(str(exc)) from exc
        finally:
            self._busy, self._busy_tid = prev_busy, prev_tid
        return cid, self.node_peak_unlocked(cid)

    def step_meas_unlocked(
        self,
        nid: int,
        eid: int,
        kind: int,
        mx: NDArray[np.uint64],
        mz: NDArray[np.uint64],
        mk: int,
    ) -> tuple[float, int, int, int, NDArray[np.uint64], NDArray[np.uint64], int, int]:
        self._check_not_poisoned()
        prev_busy, prev_tid = self._busy, self._busy_tid
        self._busy, self._busy_tid = True, threading.get_ident()
        try:
            p0, c0, c1, mt, abs_x, abs_z = self.native.step_meas(nid, eid, kind, mx, mz, mk)
        except bk.BranchBudgetError as exc:
            raise QueryBudgetExceededError(str(exc)) from exc
        finally:
            self._busy, self._busy_tid = prev_busy, prev_tid
        pk0 = self.node_peak_unlocked(c0)
        pk1 = pk0 if c1 == c0 else self.node_peak_unlocked(c1)
        return p0, c0, c1, mt, abs_x, abs_z, pk0, pk1

    def step_peek_unlocked(
        self,
        nid: int,
        eid: int,
        mx: NDArray[np.uint64],
        mz: NDArray[np.uint64],
        mk: int,
    ) -> float:
        self._check_not_poisoned()
        prev_busy, prev_tid = self._busy, self._busy_tid
        self._busy, self._busy_tid = True, threading.get_ident()
        try:
            return self.native.step_peek(nid, eid, mx, mz, mk)
        except bk.BranchBudgetError as exc:
            raise QueryBudgetExceededError(str(exc)) from exc
        finally:
            self._busy, self._busy_tid = prev_busy, prev_tid

    def node_dense(self, nid: int, *, endian: Literal["little", "big"] = "little") -> NDArray[np.complex128]:
        with self.lock:
            self._check_not_poisoned()
            prev_busy, prev_tid = self._busy, self._busy_tid
            self._busy, self._busy_tid = True, threading.get_ident()
            try:
                return self.native.node_dense(nid, endian=endian)
            finally:
                self._busy, self._busy_tid = prev_busy, prev_tid

    def counters(self) -> dict[str, Any]:
        with self.lock:
            self._check_not_poisoned()
            prev_busy, prev_tid = self._busy, self._busy_tid
            self._busy, self._busy_tid = True, threading.get_ident()
            try:
                return self.native.counters()
            finally:
                self._busy, self._busy_tid = prev_busy, prev_tid


@dataclass(frozen=True)
class BranchStateSnapshot:
    """Leak-proof snapshot returned by `MagicsideSimulator._get_branch_state(shot)`."""

    node_id: int
    n_branches: int
    n_pending: int
    transient_peak: int
    peak: int
    log_mass: float
    frame_x: NDArray[np.uint64]
    frame_z: NDArray[np.uint64]
    tableau: stim.Tableau
    branches: tuple[tuple[complex, PhaseQFEComponent], ...]
    pending: tuple[NDArray[np.uint64], NDArray[np.uint64], NDArray[np.uint8], NDArray[np.uint8]]


class BranchDebugTrace:
    """Opt-in per-batch trajectory trace aligned 1-to-1 with `_magic/trajectory.py:run_direct`."""

    def __init__(self, batch_size: int) -> None:
        self.batch_size = int(batch_size)
        self.fault_choices: list[tuple[int, NDArray[np.int32]]] = []
        self.noise_choices: list[NDArray[np.int32]] = []
        self.reset_outcomes: list[NDArray[np.uint8]] = []
        self.meas_events: list[tuple[int | None, bool, NDArray[np.uint8], NDArray[np.float64]]] = []
        self.meas_cursor: int = 0
        self.reported_records: NDArray[np.bool_] | None = None

    def record_fault(self, slot_idx: int, choices: NDArray[np.int32]) -> None:
        self.fault_choices.append((int(slot_idx), np.asarray(choices, dtype=np.int32).copy()))

    def record_noise(self, choices: NDArray[np.int32]) -> None:
        self.noise_choices.append(np.asarray(choices, dtype=np.int32).copy())

    def record_noise_zeros(self, count: int) -> None:
        if count <= 0:
            return
        z = np.zeros(self.batch_size, dtype=np.int32)
        for _ in range(count):
            self.noise_choices.append(z)

    def record_meas_or_reset(
        self,
        meas_index: int | None,
        is_reset: bool,
        true_outcome: NDArray[np.bool_],
        p_chosen: NDArray[np.float64],
    ) -> None:
        q_u8 = np.asarray(true_outcome, dtype=np.uint8).copy()
        self.meas_events.append((meas_index, bool(is_reset), q_u8, np.asarray(p_chosen, dtype=np.float64).copy()))
        if is_reset:
            self.reset_outcomes.append(q_u8)

    def advance_meas_cursor(self, count: int) -> None:
        self.meas_cursor += int(count)

    def finalize_reported(self, records: NDArray[np.bool_]) -> None:
        self.reported_records = records.copy()

    def for_shot(self, shot: int) -> dict[str, Any]:
        s = int(shot)
        fault_trace = [int(arr[s]) for _, arr in self.fault_choices]
        pauli_noise_trace = [int(arr[s]) for arr in self.noise_choices]
        reset_outcomes = [int(arr[s]) for arr in self.reset_outcomes]
        quantum_outcomes: dict[int, int] = {}
        step_probs: list[tuple[int | None, int, float]] = []
        for m_idx, _is_reset, q_arr, p_arr in self.meas_events:
            q_val = int(q_arr[s])
            step_probs.append((m_idx, q_val, float(p_arr[s])))
            if m_idx is not None:
                quantum_outcomes[m_idx] = q_val
        reported_outcomes: dict[int, int] = {}
        if self.reported_records is not None:
            row = self.reported_records[s]
            for j in range(row.shape[0]):
                reported_outcomes[j] = int(row[j])
        return {
            "fault_trace": fault_trace,
            "pauli_noise_trace": pauli_noise_trace,
            "reset_outcomes": reset_outcomes,
            "requested_quantum_outcomes": quantum_outcomes,
            "requested_outcomes": reported_outcomes,
            "step_probs": step_probs,
        }


class BranchBatch:
    """Per-batch stabilizer-branch walk backed by `_SharedBranchStore`."""

    def __init__(
        self,
        n_qubits: int,
        batch_size: int,
        store: _SharedBranchStore,
        *,
        record_trace: bool = False,
    ) -> None:
        self.n_qubits = int(n_qubits)
        self.nw = store.nw
        self.batch_size = int(batch_size)
        self.store = store
        self.U = ReferenceCHPTableau(self.n_qubits)
        self.frame_id: int = 0
        self.fx = np.zeros((self.batch_size, self.nw), dtype=np.uint64)
        self.fz = np.zeros((self.batch_size, self.nw), dtype=np.uint64)
        self.node = np.zeros(self.batch_size, dtype=np.int64)
        self.log_mass = np.zeros(self.batch_size, dtype=np.float64)
        self.peak = np.ones(self.batch_size, dtype=np.int16)
        self.faults_realized: dict[str, int] = empty_faults_realized()
        self.trace: BranchDebugTrace | None = BranchDebugTrace(self.batch_size) if record_trace else None

    def clone_for_copy(self) -> BranchBatch:
        """Deep-copy all per-batch state while sharing `self.store`."""
        new = self.__class__.__new__(self.__class__)
        new.n_qubits = self.n_qubits
        new.nw = self.nw
        new.batch_size = self.batch_size
        new.store = self.store
        new.U = self.U.copy()
        new.frame_id = self.frame_id
        new.fx = self.fx.copy()
        new.fz = self.fz.copy()
        new.node = self.node.copy()
        new.log_mass = self.log_mass.copy()
        new.peak = self.peak.copy()
        new.faults_realized = dict(self.faults_realized)
        new.trace = None
        return new

    # ------------------------------------------------------------------
    # Internal packed transition dispatch
    # ------------------------------------------------------------------

    def _step_packed(
        self,
        sel: NDArray[np.bool_] | None,
        sb: NDArray[np.int64],
        shift: int,
        step_fn: Callable[[int, int], tuple[int, int]],
    ) -> None:
        sub_node = self.node if sel is None else self.node[sel]
        if sub_node.size == 0:
            return
        n0 = int(sub_node[0])
        s0 = int(sb[0])
        if sub_node.size == 1 or (bool(np.all(sub_node == n0)) and bool(np.all(sb == s0))):
            with self.store.lock:
                cid, cpk = step_fn(n0, s0)
            cpk16 = np.int16(cpk)
            if sel is None:
                self.node.fill(cid)
                np.maximum(self.peak, cpk16, out=self.peak)
            else:
                self.node[sel] = cid
                self.peak[sel] = np.maximum(self.peak[sel], cpk16)
            return

        keys = (sub_node << np.int64(shift)) | sb
        uk, inv = np.unique(keys, return_inverse=True)
        bit_mask = (1 << shift) - 1
        new_nids = np.empty(uk.shape[0], dtype=np.int64)
        new_pks = np.empty(uk.shape[0], dtype=np.int16)
        with self.store.lock:
            for i in range(uk.shape[0]):
                kv = int(uk[i])
                cid, cpk = step_fn(kv >> shift, kv & bit_mask)
                new_nids[i] = cid
                new_pks[i] = cpk
        if sel is None:
            self.node[:] = new_nids[inv]
            np.maximum(self.peak, new_pks[inv], out=self.peak)
        else:
            self.node[sel] = new_nids[inv]
            self.peak[sel] = np.maximum(self.peak[sel], new_pks[inv])

    def _compile_qturn_event(
        self,
        canon: str,
        grp: tuple[int, ...],
        *,
        inverse: bool,
    ) -> tuple[
        NDArray[np.uint64] | None,
        NDArray[np.uint64] | None,
        NDArray[np.uint64],
        NDArray[np.uint64],
        NDArray[np.uint8],
    ]:
        fac = compile_clifford_factors(self.U, canon, grp)
        if fac.p0 is not None and (words_any_nonzero(fac.p0.a_words) or words_any_nonzero(fac.p0.b_words)):
            p0_x: NDArray[np.uint64] | None = fac.p0.a_words
            p0_z: NDArray[np.uint64] | None = fac.p0.b_words
        else:
            p0_x = None
            p0_z = None
        seq = tuple(reversed(fac.r_axes)) if inverse else fac.r_axes
        k = len(seq)
        base_x = np.empty((k, self.nw), dtype=np.uint64)
        base_z = np.empty((k, self.nw), dtype=np.uint64)
        base_k = np.empty(k, dtype=np.uint8)
        flip = 2 if inverse else 0
        for j, ax in enumerate(seq):
            base_x[j] = ax.a_words
            base_z[j] = ax.b_words
            base_k[j] = (int(ax.kappa) + flip) & 3
        return p0_x, p0_z, base_x, base_z, base_k

    # ------------------------------------------------------------------
    # Unitary Cliffords, SPP, Skips, Clifford Faults
    # ------------------------------------------------------------------

    def apply_unitary_group(
        self,
        canon: str,
        grp: tuple[int, ...],
        mask: NDArray[np.bool_] | None = None,
    ) -> None:
        pre_fid = self.frame_id
        skip_mask: NDArray[np.bool_] | None = None
        if mask is not None:
            m_bool = np.asarray(mask, dtype=np.bool_)
            if not bool(np.all(m_bool)):
                skip_mask = ~m_bool

        if skip_mask is not None:
            n_skip = int(np.count_nonzero(skip_mask))
            self.faults_realized["skipped_gates"] += n_skip
            eid, payload = self.store.get_or_compile_event(
                (pre_fid, "SKIP", canon, grp),
                lambda: self._compile_qturn_event(canon, grp, inverse=True),
            )
            p0_x, p0_z, base_x, base_z, base_k = payload
            k = base_k.shape[0]
            if k > 0:
                fx_s = self.fx[skip_mask]
                fz_s = self.fz[skip_mask]
                sb = np.zeros(n_skip, dtype=np.int64)
                for j in range(k):
                    sb |= _anti_batch(fx_s, fz_s, base_x[j], base_z[j]).astype(np.int64) << np.int64(j)
                def _step(nid: int, s_val: int) -> tuple[int, int]:
                    fk = np.empty(k, dtype=np.uint8)
                    for j in range(k):
                        fk[j] = (int(base_k[j]) + (2 if (s_val >> j) & 1 else 0)) & 3
                    return self.store.step_qturns_unlocked(
                        nid, eid, bk.MBR_KIND_SKIP, s_val, base_x, base_z, fk
                    )
                self._step_packed(skip_mask, sb, shift=k, step_fn=_step)
            if p0_x is not None and p0_z is not None:
                self.fx[skip_mask] ^= p0_x
                self.fz[skip_mask] ^= p0_z

        self.U.do_unitary(canon, grp)
        self.frame_id = self.store.advance_frame(pre_fid, ("U", canon, grp))

    def apply_spp_group(
        self,
        terms: Sequence[tuple[int, str]],
        negate: bool,
        dag: bool,
        mask: NDArray[np.bool_] | None = None,
    ) -> None:
        terms_t = tuple((int(q), p) for q, p in terms)
        neg_b = bool(negate)
        dag_b = bool(dag)
        pre_fid = self.frame_id
        skip_mask: NDArray[np.bool_] | None = None
        if mask is not None:
            m_bool = np.asarray(mask, dtype=np.bool_)
            if not bool(np.all(m_bool)):
                skip_mask = ~m_bool

        if skip_mask is not None:
            n_skip = int(np.count_nonzero(skip_mask))
            self.faults_realized["skipped_gates"] += n_skip
            def _compile_skip_spp() -> tuple[NDArray[np.uint64], NDArray[np.uint64], NDArray[np.uint8]] | None:
                ax = compile_pauli_product(self.U, terms_t, negate=neg_b ^ dag_b)
                if not (words_any_nonzero(ax.a_words) or words_any_nonzero(ax.b_words)):
                    return None
                return (
                    np.ascontiguousarray(ax.a_words.reshape(1, self.nw)),
                    np.ascontiguousarray(ax.b_words.reshape(1, self.nw)),
                    np.array([int(ax.kappa) & 3], dtype=np.uint8),
                )
            eid, payload = self.store.get_or_compile_event(
                (pre_fid, "SKIP_SPP", terms_t, neg_b, dag_b),
                _compile_skip_spp,
            )
            if payload is not None:
                base_x, base_z, base_k = payload
                sb = _anti_batch(self.fx[skip_mask], self.fz[skip_mask], base_x[0], base_z[0]).astype(np.int64)
                k0 = int(base_k[0])
                def _step(nid: int, s_val: int) -> tuple[int, int]:
                    fk = np.array([(k0 + (2 if s_val else 0)) & 3], dtype=np.uint8)
                    return self.store.step_qturns_unlocked(
                        nid, eid, bk.MBR_KIND_SKIP, s_val, base_x, base_z, fk
                    )
                self._step_packed(skip_mask, sb, shift=1, step_fn=_step)

        self.U.do_spp(terms_t, negate=neg_b, dag=dag_b)
        self.frame_id = self.store.advance_frame(pre_fid, ("SPP", terms_t, neg_b, dag_b))

    def apply_clifford_fault(
        self,
        gate_name: str,
        targets: Sequence[int],
        mask: NDArray[np.bool_],
    ) -> None:
        m_bool = np.asarray(mask, dtype=np.bool_)
        n_act = int(np.count_nonzero(m_bool))
        if n_act == 0:
            return
        grp = tuple(int(q) for q in targets)
        eid, payload = self.store.get_or_compile_event(
            (self.frame_id, "CFAULT", gate_name, grp),
            lambda: self._compile_qturn_event(gate_name, grp, inverse=False),
        )
        p0_x, p0_z, base_x, base_z, base_k = payload
        # CFAULT contract: apply F ^= p0 BEFORE computing quarter-turn sign_bits.
        if p0_x is not None and p0_z is not None:
            self.fx[m_bool] ^= p0_x
            self.fz[m_bool] ^= p0_z
        k = base_k.shape[0]
        if k == 0:
            return
        fx_s = self.fx[m_bool]
        fz_s = self.fz[m_bool]
        sb = np.zeros(n_act, dtype=np.int64)
        for j in range(k):
            sb |= _anti_batch(fx_s, fz_s, base_x[j], base_z[j]).astype(np.int64) << np.int64(j)
        def _step(nid: int, s_val: int) -> tuple[int, int]:
            fk = np.empty(k, dtype=np.uint8)
            for j in range(k):
                fk[j] = (int(base_k[j]) + (2 if (s_val >> j) & 1 else 0)) & 3
            return self.store.step_qturns_unlocked(
                nid, eid, bk.MBR_KIND_CFAULT, s_val, base_x, base_z, fk
            )
        self._step_packed(m_bool, sb, shift=k, step_fn=_step)

    # ------------------------------------------------------------------
    # Pauli Frame Updates (Faults & Feedback)
    # ------------------------------------------------------------------

    def apply_pauli_1q(self, q: int, pauli: Literal["X", "Y", "Z"], mask: NDArray[np.bool_], *, is_fault: bool = False) -> None:
        m_bool = np.asarray(mask, dtype=np.bool_)
        n_act = int(np.count_nonzero(m_bool))
        if n_act == 0:
            return
        if is_fault:
            self.faults_realized["pauli_faults"] += n_act
        if pauli == "X":
            self.fx[m_bool] ^= self.U.x2x[q]
            self.fz[m_bool] ^= self.U.x2z[q]
        elif pauli == "Z":
            self.fx[m_bool] ^= self.U.z2x[q]
            self.fz[m_bool] ^= self.U.z2z[q]
        else:
            self.fx[m_bool] ^= self.U.x2x[q] ^ self.U.z2x[q]
            self.fz[m_bool] ^= self.U.x2z[q] ^ self.U.z2z[q]

    def apply_pauli_mask(self, pauli: Literal["X", "Y", "Z"], mask: NDArray[np.bool_], *, is_fault: bool = True) -> None:
        m_s = np.asarray(mask[: self.n_qubits], dtype=np.bool_)
        affected = np.flatnonzero(np.any(m_s, axis=1))
        if affected.size == 0:
            return
        for q_np in affected:
            self.apply_pauli_1q(int(q_np), pauli, m_s[int(q_np)], is_fault=is_fault)

    def apply_pauli_masks_xz(self, xm: NDArray[np.bool_], zm: NDArray[np.bool_], *, is_fault: bool = True) -> None:
        xm_s = np.asarray(xm[: self.n_qubits], dtype=np.bool_)
        zm_s = np.asarray(zm[: self.n_qubits], dtype=np.bool_)
        if is_fault:
            n_f = int(np.count_nonzero(xm_s | zm_s))
            if n_f == 0:
                return
            self.faults_realized["pauli_faults"] += n_f
        for q_np in np.flatnonzero(np.any(xm_s, axis=1)):
            q = int(q_np)
            sel = xm_s[q]
            self.fx[sel] ^= self.U.x2x[q]
            self.fz[sel] ^= self.U.x2z[q]
        for q_np in np.flatnonzero(np.any(zm_s, axis=1)):
            q = int(q_np)
            sel = zm_s[q]
            self.fx[sel] ^= self.U.z2x[q]
            self.fz[sel] ^= self.U.z2z[q]

    # ------------------------------------------------------------------
    # Magic Gates: T, T_DAG, CCZ
    # ------------------------------------------------------------------

    def apply_magic(
        self,
        tag: str,
        targets: Sequence[int],
        mask: NDArray[np.bool_] | None = None,
    ) -> None:
        if tag in ("T", "T_DAG"):
            exps = _T_EXPS if tag == "T" else _TDAG_EXPS
            m_arr = None if mask is None else np.asarray(mask, dtype=np.bool_)
            for g_idx, q_raw in enumerate(targets):
                q = int(q_raw)
                sel: NDArray[np.bool_] | None = None
                if m_arr is not None:
                    row = m_arr if m_arr.ndim == 1 else m_arr[g_idx]
                    n_act = int(np.count_nonzero(row))
                    if n_act == 0:
                        continue
                    if n_act < row.size:
                        sel = row
                def _compile_t(qq: int = q) -> tuple[NDArray[np.uint64], NDArray[np.uint64], int]:
                    ax = compile_pauli_product(self.U, [(qq, "Z")])
                    return (
                        np.ascontiguousarray(ax.a_words.reshape(1, self.nw)),
                        np.ascontiguousarray(ax.b_words.reshape(1, self.nw)),
                        int(ax.kappa) & 3,
                    )
                eid, (ax_x, ax_z, k0) = self.store.get_or_compile_event((self.frame_id, tag, q), _compile_t)
                fx_s = self.fx if sel is None else self.fx[sel]
                fz_s = self.fz if sel is None else self.fz[sel]
                sb = _anti_batch(fx_s, fz_s, ax_x[0], ax_z[0]).astype(np.int64)
                def _step(nid: int, s_val: int, _eid: int = eid, _ax_x: NDArray[np.uint64] = ax_x, _ax_z: NDArray[np.uint64] = ax_z, _k0: int = k0) -> tuple[int, int]:
                    fk = np.array([(_k0 + (2 if s_val else 0)) & 3], dtype=np.uint8)
                    return self.store.step_rotations_unlocked(
                        nid, _eid, bk.MBR_KIND_T, s_val, _ax_x, _ax_z, fk, exps
                    )
                self._step_packed(sel, sb, shift=1, step_fn=_step)
            return

        if tag != "CCZ":
            raise ValueError(f"Unsupported magic tag {tag!r}")

        m_arr = None if mask is None else np.asarray(mask, dtype=np.bool_)
        for g_idx, off in enumerate(range(0, len(targets), 3)):
            a, b, c = int(targets[off]), int(targets[off + 1]), int(targets[off + 2])
            sel = None
            if m_arr is not None:
                row = m_arr if m_arr.ndim == 1 else m_arr[g_idx]
                n_act = int(np.count_nonzero(row))
                n_skip = int(row.size - n_act)
                self.faults_realized["skipped_cczs"] += n_skip
                if n_act == 0:
                    continue
                if n_skip > 0:
                    sel = row
            def _compile_ccz(qa: int = a, qb: int = b, qc: int = c) -> tuple[
                NDArray[np.uint64], NDArray[np.uint64], NDArray[np.uint64], NDArray[np.uint64], NDArray[np.uint8]
            ]:
                za = compile_pauli_product(self.U, [(qa, "Z")])
                zb = compile_pauli_product(self.U, [(qb, "Z")])
                zc = compile_pauli_product(self.U, [(qc, "Z")])
                gen_x = np.ascontiguousarray(np.stack((za.a_words, zb.a_words, zc.a_words), axis=0))
                gen_z = np.ascontiguousarray(np.stack((za.b_words, zb.b_words, zc.b_words), axis=0))
                base_x = np.empty((7, self.nw), dtype=np.uint64)
                base_z = np.empty((7, self.nw), dtype=np.uint64)
                base_k = np.empty(7, dtype=np.uint8)
                base_x[0], base_z[0], base_k[0] = za.a_words, za.b_words, int(za.kappa) & 3
                base_x[1], base_z[1], base_k[1] = zb.a_words, zb.b_words, int(zb.kappa) & 3
                base_x[2], base_z[2], base_k[2] = zc.a_words, zc.b_words, int(zc.kappa) & 3
                ab_x, ab_z, ab_k = pauli_multiply(za.a_words, za.b_words, za.kappa, zb.a_words, zb.b_words, zb.kappa)
                ac_x, ac_z, ac_k = pauli_multiply(za.a_words, za.b_words, za.kappa, zc.a_words, zc.b_words, zc.kappa)
                bc_x, bc_z, bc_k = pauli_multiply(zb.a_words, zb.b_words, zb.kappa, zc.a_words, zc.b_words, zc.kappa)
                abc_x, abc_z, abc_k = pauli_multiply(ab_x, ab_z, ab_k, zc.a_words, zc.b_words, zc.kappa)
                base_x[3], base_z[3], base_k[3] = ab_x, ab_z, ab_k & 3
                base_x[4], base_z[4], base_k[4] = ac_x, ac_z, ac_k & 3
                base_x[5], base_z[5], base_k[5] = bc_x, bc_z, bc_k & 3
                base_x[6], base_z[6], base_k[6] = abc_x, abc_z, abc_k & 3
                return gen_x, gen_z, base_x, base_z, base_k

            eid, (gen_x, gen_z, base_x, base_z, base_k) = self.store.get_or_compile_event(
                (self.frame_id, "CCZ", a, b, c), _compile_ccz
            )
            fx_s = self.fx if sel is None else self.fx[sel]
            fz_s = self.fz if sel is None else self.fz[sel]
            sb = (
                _anti_batch(fx_s, fz_s, gen_x[0], gen_z[0]).astype(np.int64)
                | (_anti_batch(fx_s, fz_s, gen_x[1], gen_z[1]).astype(np.int64) << np.int64(1))
                | (_anti_batch(fx_s, fz_s, gen_x[2], gen_z[2]).astype(np.int64) << np.int64(2))
            )
            def _step_ccz(
                nid: int,
                s_val: int,
                _eid: int = eid,
                _bx: NDArray[np.uint64] = base_x,
                _bz: NDArray[np.uint64] = base_z,
                _bk: NDArray[np.uint8] = base_k,
            ) -> tuple[int, int]:
                fk = np.empty(7, dtype=np.uint8)
                for i in range(7):
                    fk[i] = (int(_bk[i]) + (2 if ((s_val & _CCZ_MASKS[i]).bit_count() & 1) else 0)) & 3
                return self.store.step_rotations_unlocked(
                    nid, _eid, bk.MBR_KIND_CCZ, s_val, _bx, _bz, fk, _CCZ_EXPS
                )
            self._step_packed(sel, sb, shift=3, step_fn=_step_ccz)

    # ------------------------------------------------------------------
    # Measurements, Resets, Peek, Postselect, Unmatched Noisy Reset
    # ------------------------------------------------------------------

    def _query_meas_nodes(
        self,
        nodes: NDArray[np.int64],
        eid: int,
        kind: int,
        mx: NDArray[np.uint64],
        mz: NDArray[np.uint64],
        mk: int,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.uint8],
        NDArray[np.uint64],
        NDArray[np.uint64],
        NDArray[np.int16],
        NDArray[np.int16],
        NDArray[np.intp],
    ]:
        n0 = int(nodes[0])
        if nodes.size == 1 or bool(np.all(nodes == n0)):
            with self.store.lock:
                p0, c0, c1, mt, ax, az, pk0, pk1 = self.store.step_meas_unlocked(n0, eid, kind, mx, mz, mk)
            return (
                np.array([p0], dtype=np.float64),
                np.array([c0], dtype=np.int64),
                np.array([c1], dtype=np.int64),
                np.array([mt], dtype=np.uint8),
                ax.reshape(1, self.nw),
                az.reshape(1, self.nw),
                np.array([pk0], dtype=np.int16),
                np.array([pk1], dtype=np.int16),
                np.zeros(nodes.shape[0], dtype=np.intp),
            )
        un, inv = np.unique(nodes, return_inverse=True)
        nu = un.shape[0]
        p0s = np.empty(nu, dtype=np.float64)
        c0s = np.empty(nu, dtype=np.int64)
        c1s = np.empty(nu, dtype=np.int64)
        mts = np.empty(nu, dtype=np.uint8)
        abs_x = np.zeros((nu, self.nw), dtype=np.uint64)
        abs_z = np.zeros((nu, self.nw), dtype=np.uint64)
        pk0s = np.empty(nu, dtype=np.int16)
        pk1s = np.empty(nu, dtype=np.int16)
        with self.store.lock:
            for i in range(nu):
                p0, c0, c1, mt, ax, az, pk0, pk1 = self.store.step_meas_unlocked(
                    int(un[i]), eid, kind, mx, mz, mk
                )
                p0s[i] = p0
                c0s[i] = c0
                c1s[i] = c1
                mts[i] = mt
                abs_x[i] = ax
                abs_z[i] = az
                pk0s[i] = pk0
                pk1s[i] = pk1
        return p0s, c0s, c1s, mts, abs_x, abs_z, pk0s, pk1s, inv

    def measure_or_reset(
        self,
        terms: Sequence[tuple[int, str]],
        rng: np.random.Generator,
        *,
        is_reset: bool = False,
        reset_qubit: int | None = None,
        reset_basis: Literal["X", "Y", "Z"] | None = None,
        negate: bool = False,
        meas_index: int | None = None,
    ) -> NDArray[np.bool_]:
        # F5 determinism: ALWAYS draw exact 1 uniform per shot, even when deterministic.
        u = rng.random(self.batch_size)
        if is_reset:
            assert reset_qubit is not None and reset_basis is not None
            q = int(reset_qubit)
            kick_p = _RESET_KICK[reset_basis]
            def _compile_reset() -> tuple[NDArray[np.uint64], NDArray[np.uint64], int, NDArray[np.uint64], NDArray[np.uint64]]:
                obs = compile_pauli_product(self.U, [(q, reset_basis)])
                kick = compile_pauli_product(self.U, [(q, kick_p)])
                return obs.a_words, obs.b_words, int(obs.kappa) & 3, kick.a_words, kick.b_words
            eid, (mx, mz, mk, kx, kz) = self.store.get_or_compile_event(
                (self.frame_id, "RESET", q, reset_basis), _compile_reset
            )
        else:
            terms_t = tuple((int(q), p) for q, p in terms)
            neg_b = bool(negate)
            def _compile_meas() -> tuple[NDArray[np.uint64], NDArray[np.uint64], int]:
                obs = compile_pauli_product(self.U, terms_t, negate=neg_b)
                return obs.a_words, obs.b_words, int(obs.kappa) & 3
            eid, (mx, mz, mk) = self.store.get_or_compile_event(
                (self.frame_id, "MEAS", terms_t, neg_b), _compile_meas
            )
            kx = kz = None

        p0s, c0s, c1s, mts, abs_x, abs_z, pk0s, pk1s, inv = self._query_meas_nodes(
            self.node, eid, bk.MBR_KIND_MEAS, mx, mz, mk
        )
        p0_arr = p0s[inv]
        c0_arr = c0s[inv]
        c1_arr = c1s[inv]
        inner_m = np.where(c0_arr < 0, True, np.where(c1_arr < 0, False, u >= p0_arr))

        # Caller contract #1: compute pre-kick s_bit BEFORE mutating F.
        s_bit = _anti_batch(self.fx, self.fz, mx, mz).astype(np.bool_)
        true_m = inner_m ^ s_bit

        # Apply absorption frame kick for ABSORBED & inner_m == 1.
        for i in range(mts.shape[0]):
            if mts[i] == bk.MBR_MEAS_ABSORBED:
                sel = (inv == i) & inner_m
                if np.any(sel):
                    self.fx[sel] ^= abs_x[i]
                    self.fz[sel] ^= abs_z[i]

        self.node[:] = np.where(inner_m, c1_arr, c0_arr)
        np.maximum(self.peak, np.where(inner_m, pk1s[inv], pk0s[inv]), out=self.peak)
        p_chosen = np.where(inner_m, 1.0 - p0_arr, p0_arr)
        self.log_mass += _safe_log(p_chosen)

        if is_reset and kx is not None and kz is not None:
            if np.any(true_m):
                self.fx[true_m] ^= kx
                self.fz[true_m] ^= kz

        if self.trace is not None:
            self.trace.record_meas_or_reset(meas_index, is_reset, true_m, p_chosen)
        return true_m

    def peek_pauli_born_batch(
        self,
        targets: Sequence[int] | NDArray[np.int_],
        pauli: Literal["X", "Y", "Z"] = "Z",
    ) -> NDArray[np.float64]:
        t_list = [int(q) for q in targets]
        out = np.empty((len(t_list), self.batch_size), dtype=np.float64)
        if not t_list:
            return out
        n0 = int(self.node[0])
        all_same = self.node.size == 1 or bool(np.all(self.node == n0))
        if not all_same:
            un, inv = np.unique(self.node, return_inverse=True)
        else:
            un = inv = None
        for row_i, q in enumerate(t_list):
            def _compile_peek(qq: int = q) -> tuple[NDArray[np.uint64], NDArray[np.uint64], int]:
                obs = compile_pauli_product(self.U, [(qq, pauli)])
                return obs.a_words, obs.b_words, int(obs.kappa) & 3
            eid, (mx, mz, mk) = self.store.get_or_compile_event(
                (self.frame_id, "PEEK", q, pauli), _compile_peek
            )
            if all_same:
                with self.store.lock:
                    p0 = self.store.step_peek_unlocked(n0, eid, mx, mz, mk)
                p0_arr = np.full(self.batch_size, p0, dtype=np.float64)
            else:
                assert un is not None and inv is not None
                p0s = np.empty(un.shape[0], dtype=np.float64)
                with self.store.lock:
                    for i in range(un.shape[0]):
                        p0s[i] = self.store.step_peek_unlocked(int(un[i]), eid, mx, mz, mk)
                p0_arr = p0s[inv]
            s_bit = _anti_batch(self.fx, self.fz, mx, mz).astype(np.bool_)
            # Caller contract #4: p1 = p0_inner if s_bit else (1.0 - p0_inner), clamped to [0, 1].
            out[row_i] = np.clip(np.where(s_bit, p0_arr, 1.0 - p0_arr), 0.0, 1.0)
        return out

    def postselect_pauli_batch(
        self,
        q: int,
        flip_mask: NDArray[np.bool_],
        desired_bits: Any,
        pauli: Literal["X", "Y", "Z"] = "Z",
    ) -> None:
        shots = np.flatnonzero(flip_mask)
        if shots.size == 0:
            return
        q_int = int(q)
        if type(desired_bits) is np.ndarray and desired_bits.dtype.kind in "biu" and desired_bits.ndim == 1:
            desired = (desired_bits[shots] & desired_bits.dtype.type(1)).astype(np.bool_)
        else:
            desired = np.array([int(desired_bits[int(b)]) & 1 for b in shots], dtype=np.bool_)

        def _compile_post() -> tuple[NDArray[np.uint64], NDArray[np.uint64], int]:
            obs = compile_pauli_product(self.U, [(q_int, pauli)])
            return obs.a_words, obs.b_words, int(obs.kappa) & 3
        eid, (mx, mz, mk) = self.store.get_or_compile_event(
            (self.frame_id, "POST", q_int, pauli), _compile_post
        )

        # Compute pre-kick s_bit and validate ALL active shots before any mutation.
        s_bit = _anti_batch(self.fx[shots], self.fz[shots], mx, mz).astype(np.bool_)
        inner_m = desired ^ s_bit

        p0s, c0s, c1s, mts, abs_x, abs_z, pk0s, pk1s, inv = self._query_meas_nodes(
            self.node[shots], eid, bk.MBR_KIND_POST, mx, mz, mk
        )
        p0_arr = p0s[inv]
        child = np.where(inner_m, c1s[inv], c0s[inv])
        p_chosen = np.where(inner_m, 1.0 - p0_arr, p0_arr)
        bad = (child < 0) | (p_chosen <= 1e-13)
        if np.any(bad):
            b0 = int(shots[int(np.flatnonzero(bad)[0])])
            raise ImpossiblePostselectionError(
                f"Impossible postselection on qubit {q_int} ({pauli}) for shot {b0}"
            )

        # Commit mutations atomically after validation succeeds.
        for i in range(mts.shape[0]):
            if mts[i] == bk.MBR_MEAS_ABSORBED:
                local_sel = (inv == i) & inner_m
                if np.any(local_sel):
                    sh = shots[local_sel]
                    self.fx[sh] ^= abs_x[i]
                    self.fz[sh] ^= abs_z[i]
        self.node[shots] = child
        self.peak[shots] = np.maximum(self.peak[shots], np.where(inner_m, pk1s[inv], pk0s[inv]))
        self.log_mass[shots] += _safe_log(p_chosen)

    def unmatched_noisy_reset(
        self,
        reset_zero_mask: NDArray[np.bool_],
        reset_one_mask: NDArray[np.bool_],
        rng: np.random.Generator,
        pauli: Literal["X", "Y", "Z"] = "Z",
        order: Sequence[int] | None = None,
    ) -> None:
        z_arr = np.asarray(reset_zero_mask, dtype=np.bool_)
        o_arr = np.asarray(reset_one_mask, dtype=np.bool_)
        comb = z_arr | o_arr
        if not np.any(comb):
            return
        one_b = np.broadcast_to(o_arr, comb.shape)
        qs = (
            list(dict.fromkeys(int(x) for x in order))
            if order is not None
            else [int(x) for x in np.flatnonzero(np.any(comb, axis=1))]
        )
        kick_p = _RESET_KICK[pauli]
        for q in qs:
            shots = np.flatnonzero(comb[q])
            if shots.size == 0:
                continue
            # F5 determinism: draw exact 1 uniform per shot of batch.
            u_all = rng.random(self.batch_size)
            u = u_all[shots]
            def _compile_ureset(qq: int = q) -> tuple[
                NDArray[np.uint64], NDArray[np.uint64], int, NDArray[np.uint64], NDArray[np.uint64]
            ]:
                obs = compile_pauli_product(self.U, [(qq, pauli)])
                kick = compile_pauli_product(self.U, [(qq, kick_p)])
                return obs.a_words, obs.b_words, int(obs.kappa) & 3, kick.a_words, kick.b_words
            eid, (mx, mz, mk, kx, kz) = self.store.get_or_compile_event(
                (self.frame_id, "URESET", q, pauli), _compile_ureset
            )
            p0s, c0s, c1s, mts, abs_x, abs_z, pk0s, pk1s, inv = self._query_meas_nodes(
                self.node[shots], eid, bk.MBR_KIND_URESET, mx, mz, mk
            )
            p0_arr = p0s[inv]
            c0_arr = c0s[inv]
            c1_arr = c1s[inv]
            inner_m = np.where(c0_arr < 0, True, np.where(c1_arr < 0, False, u >= p0_arr))
            s_bit = _anti_batch(self.fx[shots], self.fz[shots], mx, mz).astype(np.bool_)
            true_m = inner_m ^ s_bit

            for i in range(mts.shape[0]):
                if mts[i] == bk.MBR_MEAS_ABSORBED:
                    local_sel = (inv == i) & inner_m
                    if np.any(local_sel):
                        sh = shots[local_sel]
                        self.fx[sh] ^= abs_x[i]
                        self.fz[sh] ^= abs_z[i]

            self.node[shots] = np.where(inner_m, c1_arr, c0_arr)
            self.peak[shots] = np.maximum(self.peak[shots], np.where(inner_m, pk1s[inv], pk0s[inv]))
            # Trace-preserving channel: log_mass stays UNCHANGED.
            want_one = one_b[q, shots]
            kick_shots = shots[true_m != want_one]
            if kick_shots.size > 0:
                self.fx[kick_shots] ^= kx
                self.fz[kick_shots] ^= kz

    # ------------------------------------------------------------------
    # State Inspection & Getters
    # ------------------------------------------------------------------

    def state_vector(
        self,
        shot: int,
        *,
        endian: Literal["little", "big"] = "little",
        normalize: bool = True,
    ) -> NDArray[np.complex128]:
        if self.n_qubits > 26:
            raise ValueError(f"state_vector requires num_qubits <= 26, got {self.n_qubits}")
        s = int(shot)
        nid = int(self.node[s])
        psi = self.store.node_dense(nid, endian="little")
        fx_s = self.fx[s]
        fz_s = self.fz[s]
        if int(fx_s[0]) != 0 or int(fz_s[0]) != 0:
            k_f = int(fx_s[0] & fz_s[0]).bit_count() & 3
            psi = _apply_pauli_to_dense_le(psi, fx_s, fz_s, k_f)
        psi = apply_reference_tableau_to_statevector(psi, self.U, endian="little")
        if not normalize:
            psi = np.ascontiguousarray(psi * math.exp(0.5 * float(self.log_mass[s])))
        if endian == "big" and self.n_qubits > 1:
            psi = np.ascontiguousarray(
                psi.reshape((2,) * self.n_qubits).transpose(tuple(range(self.n_qubits - 1, -1, -1))).ravel()
            )
        return psi

    def branch_mass(self, shot: int) -> float:
        return math.exp(float(self.log_mass[int(shot)]))

    def branch_state(self, shot: int) -> BranchStateSnapshot:
        s = int(shot)
        nid = int(self.node[s])
        with self.store.lock:
            self.store._check_not_poisoned()
            prev_busy, prev_tid = self.store._busy, self.store._busy_tid
            self.store._busy, self.store._busy_tid = True, threading.get_ident()
            try:
                nb, npd, tpk = self.store.native.node_info(nid)
                raw_branches = self.store.native.node_branches(nid)
                try:
                    converted = tuple(
                        (w, st.to_phase_qfe(PhaseQFEComponent)) for w, st in raw_branches
                    )
                finally:
                    for _, st in raw_branches:
                        st.close()
                pending = self.store.native.node_pending(nid)
            finally:
                self.store._busy, self.store._busy_tid = prev_busy, prev_tid
        return BranchStateSnapshot(
            node_id=nid,
            n_branches=nb,
            n_pending=npd,
            transient_peak=tpk,
            peak=int(self.peak[s]),
            log_mass=float(self.log_mass[s]),
            frame_x=self.fx[s].copy(),
            frame_z=self.fz[s].copy(),
            tableau=self.U.to_stim_tableau(),
            branches=converted,
            pending=pending,
        )

    def peak_per_shot(self) -> NDArray[np.int16]:
        return self.peak.copy()
