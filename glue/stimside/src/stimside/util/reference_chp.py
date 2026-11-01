from __future__ import annotations

import dataclasses
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]

from stimside.util.bitwords import (
    first_set_bit_in_words,
    pack_bit_vector,
    pauli_multiply,
    unit_bit_words,
    unpack_bit_vector,
    validate_hermitian_pauli,
    word_dot_parity,
    words_any_nonzero,
)

_GATE_TABLEAU_CACHE: dict[str, tuple[stim.Tableau, stim.Tableau]] = {}
_PAU_XZ: dict[str, tuple[int, int]] = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}


def get_cached_gate_tableau(gate_name: str) -> tuple[stim.Tableau, stim.Tableau]:
    """Return cached (forward_tableau, inverse_tableau) for a named Stim gate."""
    cached = _GATE_TABLEAU_CACHE.get(gate_name)
    if cached is not None:
        return cached
    fwd = stim.Tableau.from_named_gate(gate_name)
    inv = fwd.inverse()
    _GATE_TABLEAU_CACHE[gate_name] = (fwd, inv)
    return fwd, inv


def pauli_image(pauli, q, x2x_w, x2z_w, z2x_w, z2z_w, x_kappa, z_kappa):
    """Reference-frame image (a_words, b_words, kappa) of Pauli_q for pauli in 'X', 'Y', 'Z'."""
    if pauli == "Z":
        return z2x_w[q], z2z_w[q], int(z_kappa[q])
    if pauli == "X":
        return x2x_w[q], x2z_w[q], int(x_kappa[q])
    dp = word_dot_parity(x2z_w[q], z2x_w[q])
    return (
        x2x_w[q] ^ z2x_w[q],
        x2z_w[q] ^ z2z_w[q],
        (1 + int(x_kappa[q]) + int(z_kappa[q]) + (dp << 1)) & 3,
    )


_pauli_image = pauli_image


def pauli_image_batch(pauli, targets_arr, x2x_w, x2z_w, z2x_w, z2z_w, x_kappa, z_kappa):
    """Batched reference-frame image (a_words, b_words, kappas) for targets_arr."""
    if pauli == "Z":
        return z2x_w[targets_arr], z2z_w[targets_arr], z_kappa[targets_arr]
    if pauli == "X":
        return x2x_w[targets_arr], x2z_w[targets_arr], x_kappa[targets_arr]
    dp = np.bitwise_count(x2z_w[targets_arr] & z2x_w[targets_arr]).sum(axis=1, dtype=np.int32) & 1
    kappas = (
        (1 + x_kappa[targets_arr].astype(np.int32) + z_kappa[targets_arr].astype(np.int32) + (dp << 1)) & 3
    ).astype(np.uint8)
    return x2x_w[targets_arr] ^ z2x_w[targets_arr], x2z_w[targets_arr] ^ z2z_w[targets_arr], kappas


def _mul_rows(
    ax: NDArray[np.uint64],
    az: NDArray[np.uint64],
    ak: NDArray[np.uint8],
    idx: NDArray[np.intp],
    bx: NDArray[np.uint64],
    bz: NDArray[np.uint64],
    bk: NDArray[np.uint8],
    phase: int = 0,
) -> None:
    """Right-multiply selected rows: P_idx <- i^phase * P_idx * P_src."""
    dp = np.bitwise_count(az[idx] & bx).sum(axis=1) & 1
    ax[idx] ^= bx
    az[idx] ^= bz
    ak[idx] = (phase + ak[idx] + bk + (dp << 1)) & 3


@dataclasses.dataclass
class DetMeasMeta:
    """Metadata for a deterministic Z_q measurement (a = 0) in the noiseless reference circuit."""

    is_random: Literal[False]
    q: int
    m_ref: int
    beta_Z_words: NDArray[np.uint64]
    reset_flip_ref: bool = False


@dataclasses.dataclass
class RandMeasMeta:
    """Metadata for a random Z_q measurement (a != 0) in the noiseless reference circuit."""

    is_random: Literal[True]
    q: int
    m_ref: int
    p: int
    a_words: NDArray[np.uint64]
    b_rot_words: NDArray[np.uint64]
    e_p_words: NDArray[np.uint64]
    kappa_rot: int
    G_x_indices: NDArray[np.int64]
    G_z_indices: NDArray[np.int64]
    reset_flip_ref: bool = False


PackedSymplecticTables = tuple[
    NDArray[np.uint64],
    NDArray[np.uint64],
    NDArray[np.uint64],
    NDArray[np.uint64],
    NDArray[np.uint8],
    NDArray[np.uint8],
]


class ReferenceCHPTableau:
    """Symplectic stabilizer-destabilizer reference engine operating directly on packed uint64 coordinate arrays without tableau conjugation or inversion."""

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits
        self.n_words = max(1, (num_qubits + 63) // 64)
        self.x2x = np.zeros((num_qubits, self.n_words), dtype=np.uint64)
        self.x2z = np.zeros((num_qubits, self.n_words), dtype=np.uint64)
        self.z2x = np.zeros((num_qubits, self.n_words), dtype=np.uint64)
        self.z2z = np.zeros((num_qubits, self.n_words), dtype=np.uint64)
        for q in range(num_qubits):
            self.x2x[q, q >> 6] = np.uint64(1 << (q & 63))
            self.z2z[q, q >> 6] = np.uint64(1 << (q & 63))
        self.x_kap = np.zeros(num_qubits, dtype=np.uint8)
        self.z_kap = np.zeros(num_qubits, dtype=np.uint8)
        self._inv_dirty = False

    @classmethod
    def from_packed(cls, num_qubits: int, packed: PackedSymplecticTables) -> ReferenceCHPTableau:
        obj = cls.__new__(cls)
        obj.num_qubits = num_qubits
        obj.n_words = max(1, (num_qubits + 63) // 64)
        obj.x2x, obj.x2z, obj.z2x, obj.z2z, obj.x_kap, obj.z_kap = (a.copy() for a in packed)
        obj._inv_dirty = False
        return obj

    def copy(self) -> ReferenceCHPTableau:
        return self.from_packed(
            self.num_qubits, (self.x2x, self.x2z, self.z2x, self.z2z, self.x_kap, self.z_kap)
        )

    def snapshot_packed(self) -> PackedSymplecticTables:
        return (
            self.x2x.copy(),
            self.x2z.copy(),
            self.z2x.copy(),
            self.z2z.copy(),
            self.x_kap.copy(),
            self.z_kap.copy(),
        )

    def pauli_image(self, pauli: str, q: int) -> tuple[NDArray[np.uint64], NDArray[np.uint64], int]:
        return pauli_image(pauli, q, self.x2x, self.x2z, self.z2x, self.z2z, self.x_kap, self.z_kap)

    def apply_right_rot(
        self, rot_a: NDArray[np.uint64], rot_b: NDArray[np.uint64], fac_kap: int
    ) -> None:
        """Left-multiply anticommuting rows Q_r <- (i^fac_kap P(rot_a, rot_b)) Q_r."""
        fk = np.uint8(fac_kap & 3)
        for ax, az, ak in ((self.x2x, self.x2z, self.x_kap), (self.z2x, self.z2z, self.z_kap)):
            idx = np.flatnonzero((np.bitwise_count((ax & rot_b) ^ (az & rot_a)).sum(axis=1) & 1) != 0)
            if len(idx):
                dp = (np.bitwise_count(ax[idx] & rot_b).sum(axis=1) & 1).astype(np.uint8)
                ak[idx] = (fk + ak[idx] + (dp << np.uint8(1))) & np.uint8(3)
                ax[idx] ^= rot_a
                az[idx] ^= rot_b
        self._inv_dirty = True

    def do_unitary(self, gate_name: str, targets: Sequence[int]) -> None:
        """Update C_t <- U C_t directly on packed uint64 coordinate arrays without stim.Tableau."""
        if gate_name in ("I", "II", "I_ERROR", "II_ERROR") or not targets:
            return
        if len(set(targets)) != len(targets):
            arity = (
                2
                if gate_name in ("CZ", "ZCZ", "CX", "CNOT", "ZCX", "SWAP")
                else len(get_cached_gate_tableau(gate_name)[1])
            )
            if len(targets) > arity:
                for i in range(0, len(targets), arity):
                    self.do_unitary(gate_name, targets[i : i + arity])
                return
        self._inv_dirty = True
        q_arr = np.asarray(targets, dtype=np.intp)

        if gate_name in ("H", "H_XZ"):
            for a, b in ((self.x2x, self.z2x), (self.x2z, self.z2z), (self.x_kap, self.z_kap)):
                a[q_arr], b[q_arr] = b[q_arr].copy(), a[q_arr].copy()
            return

        if gate_name in ("CZ", "ZCZ"):
            q0, q1 = q_arr[0::2], q_arr[1::2]
            _mul_rows(self.x2x, self.x2z, self.x_kap, q0, self.z2x[q1], self.z2z[q1], self.z_kap[q1])
            _mul_rows(self.x2x, self.x2z, self.x_kap, q1, self.z2x[q0], self.z2z[q0], self.z_kap[q0])
            return

        if gate_name in ("CX", "CNOT", "ZCX"):
            c, t = q_arr[0::2], q_arr[1::2]
            _mul_rows(self.x2x, self.x2z, self.x_kap, c, self.x2x[t], self.x2z[t], self.x_kap[t])
            _mul_rows(self.z2x, self.z2z, self.z_kap, t, self.z2x[c], self.z2z[c], self.z_kap[c])
            return

        if gate_name == "X":
            self.z_kap[q_arr] = (self.z_kap[q_arr] + 2) & 3
            return
        if gate_name == "Z":
            self.x_kap[q_arr] = (self.x_kap[q_arr] + 2) & 3
            return
        if gate_name == "Y":
            self.x_kap[q_arr] = (self.x_kap[q_arr] + 2) & 3
            self.z_kap[q_arr] = (self.z_kap[q_arr] + 2) & 3
            return

        if gate_name in ("S", "S_DAG"):
            _mul_rows(
                self.x2x, self.x2z, self.x_kap, q_arr,
                self.z2x[q_arr], self.z2z[q_arr], self.z_kap[q_arr],
                phase=3 if gate_name == "S" else 1,
            )
            return

        if gate_name == "SWAP":
            q0, q1 = q_arr[0::2], q_arr[1::2]
            for arr in (self.x2x, self.x2z, self.z2x, self.z2z, self.x_kap, self.z_kap):
                arr[q0], arr[q1] = arr[q1].copy(), arr[q0].copy()
            return

        _, inv_gate = get_cached_gate_tableau(gate_name)
        arity = len(inv_gate)
        for i in range(0, len(targets), arity):
            grp = targets[i : i + arity]
            old_x = [(self.x2x[q].copy(), self.x2z[q].copy(), int(self.x_kap[q])) for q in grp]
            old_z = [(self.z2x[q].copy(), self.z2z[q].copy(), int(self.z_kap[q])) for q in grp]
            for r_loc, q_phys in enumerate(grp):
                for is_z_out, p_str in ((False, inv_gate.x_output(r_loc)), (True, inv_gate.z_output(r_loc))):
                    xs_l, zs_l = p_str.to_numpy()
                    aw_acc = np.zeros(self.n_words, dtype=np.uint64)
                    bw_acc = np.zeros(self.n_words, dtype=np.uint64)
                    kap_acc = 2 if p_str.sign == -1 else 0
                    for k_loc in range(arity):
                        bx, bz = int(xs_l[k_loc]), int(zs_l[k_loc])
                        if bx == 0 and bz == 0:
                            continue
                        if bx == 1 and bz == 0:
                            aw_t, bw_t, kap_t = old_x[k_loc]
                        elif bx == 0 and bz == 1:
                            aw_t, bw_t, kap_t = old_z[k_loc]
                        else:
                            ax_w, bx_w, kx = old_x[k_loc]
                            az_w, bz_w, kz = old_z[k_loc]
                            aw_t, bw_t, kap_t = pauli_multiply(ax_w, bx_w, kx, az_w, bz_w, kz)
                            kap_t = (kap_t + 1) & 3
                        aw_acc, bw_acc, kap_acc = pauli_multiply(aw_acc, bw_acc, kap_acc, aw_t, bw_t, kap_t)
                    if is_z_out:
                        self.z2x[q_phys], self.z2z[q_phys], self.z_kap[q_phys] = aw_acc, bw_acc, kap_acc
                    else:
                        self.x2x[q_phys], self.x2z[q_phys], self.x_kap[q_phys] = aw_acc, bw_acc, kap_acc

    def do_spp(
        self, terms: Sequence[tuple[int, str]], *, negate: bool = False, dag: bool = False
    ) -> None:
        """Update C_t in O(|supp| * n_words) under SPP / SPP_DAG."""
        bar_a = np.zeros(self.n_words, dtype=np.uint64)
        bar_b = np.zeros(self.n_words, dtype=np.uint64)
        bar_kap = 2 if negate else 0
        phys_xz: dict[int, tuple[int, int]] = {}
        for q, pch in terms:
            dx, dz = _PAU_XZ[pch]
            xp, zp = phys_xz.get(q, (0, 0))
            phys_xz[q] = (xp ^ dx, zp ^ dz)
            if pch == "I":
                continue
            if q < 0 or q >= self.num_qubits:
                raise ValueError(f"Qubit {q} out of range for num_qubits={self.num_qubits}")
            aw, bw, kw = self.pauli_image(pch, q)
            bar_a, bar_b, bar_kap = pauli_multiply(bar_a, bar_b, bar_kap, aw, bw, kw)
        bar_kap = validate_hermitian_pauli(bar_a, bar_b, bar_kap)
        kap_fac = (bar_kap + (3 if dag else 1)) & 3
        for q, (xp, zp) in phys_xz.items():
            for flag, ax, az, ak in (
                (zp, self.x2x, self.x2z, self.x_kap),
                (xp, self.z2x, self.z2z, self.z_kap),
            ):
                if flag:
                    dot = word_dot_parity(bar_b, ax[q])
                    ak[q] = np.uint8((kap_fac + int(ak[q]) + (dot << 1)) & 3)
                    ax[q] ^= bar_a
                    az[q] ^= bar_b
        self._inv_dirty = True

    def measure_z(
        self, q: int, *, is_reset: bool = False, m_ref_override: int = 0
    ) -> DetMeasMeta | RandMeasMeta:
        """Execute a reference Z_q measurement/reset via direct rank-1 symplectic transvection."""
        aw = self.z2x[q]
        if not words_any_nonzero(aw):
            m_ref = int(self.z_kap[q] >> 1) & 1
            beta_Z_words = self.z2z[q].copy()
            reset_flip_ref = False
            if is_reset and m_ref == 1:
                self.z_kap[q] = (self.z_kap[q] + 2) & 3
                self._inv_dirty = True
                reset_flip_ref = True
            return DetMeasMeta(
                is_random=False,
                q=q,
                m_ref=m_ref,
                beta_Z_words=beta_Z_words,
                reset_flip_ref=reset_flip_ref,
            )

        p = first_set_bit_in_words(aw)
        p_w, p_b = p >> 6, p & 63
        p_mask = np.uint64(1 << p_b)

        G_x_indices = np.ascontiguousarray(
            np.flatnonzero((self.z2x[:, p_w] & p_mask) != 0), dtype=np.int64
        )
        G_z_indices = np.ascontiguousarray(
            np.flatnonzero((self.x2x[:, p_w] & p_mask) != 0), dtype=np.int64
        )

        a_words = aw.copy()
        b_rot_words = self.z2z[q].copy()
        b_rot_words[p_w] ^= p_mask
        e_p_words = unit_bit_words(p, self.n_words)

        m_ref = 0 if is_reset else int(m_ref_override)
        kap = int(self.z_kap[q])
        kappa_rot = (kap + 1) & 3
        ab_rot_dot = word_dot_parity(a_words, b_rot_words)
        kap_iprot = ((m_ref << 1) - kap + (ab_rot_dot << 1)) & 3

        self.apply_right_rot(a_words, b_rot_words, kap_iprot)

        reset_flip_ref = False
        if is_reset and m_ref == 1:
            self.z_kap[q] = (self.z_kap[q] + 2) & 3
            reset_flip_ref = True

        return RandMeasMeta(
            is_random=True,
            q=q,
            m_ref=m_ref,
            p=p,
            a_words=a_words,
            b_rot_words=b_rot_words,
            e_p_words=e_p_words,
            kappa_rot=kappa_rot,
            G_x_indices=G_x_indices,
            G_z_indices=G_z_indices,
            reset_flip_ref=reset_flip_ref,
        )

    def to_stim_tableau(self) -> stim.Tableau:
        """Convert inverse symplectic tables into forward `stim.Tableau`."""
        n = self.num_qubits
        xs: list[stim.PauliString] = []
        zs: list[stim.PauliString] = []
        pauli_chars = "IZXY"
        for arr_x, arr_z, arr_k, out_list in (
            (self.x2x, self.x2z, self.x_kap, xs),
            (self.z2x, self.z2z, self.z_kap, zs),
        ):
            for q in range(n):
                aw, bw = arr_x[q], arr_z[q]
                exp4 = (int(arr_k[q]) - int(np.bitwise_count(aw & bw).sum())) & 3
                bx, bz = unpack_bit_vector(aw, n), unpack_bit_vector(bw, n)
                chars = "".join(pauli_chars[(int(bx[i]) << 1) | int(bz[i])] for i in range(n))
                out_list.append(stim.PauliString(("-" if exp4 == 2 else "+") + chars))
        return stim.Tableau.from_conjugated_generators(xs=xs, zs=zs).inverse()

    @classmethod
    def from_stim_tableau(cls, tab: stim.Tableau) -> ReferenceCHPTableau:
        """Convert forward `stim.Tableau` into inverse-convention `ReferenceCHPTableau`."""
        n = len(tab)
        out = cls(n)
        if n == 0:
            return out
        inv = tab.inverse()
        for q in range(n):
            for ax, az, ak, ps in (
                (out.x2x, out.x2z, out.x_kap, inv.x_output(q)),
                (out.z2x, out.z2z, out.z_kap, inv.z_output(q)),
            ):
                xs, zs = ps.to_numpy()
                ax[q], az[q] = pack_bit_vector(xs, n), pack_bit_vector(zs, n)
                ak[q] = ((2 if ps.sign == -1 else 0) + int(np.count_nonzero(xs & zs))) & 3
        return out
