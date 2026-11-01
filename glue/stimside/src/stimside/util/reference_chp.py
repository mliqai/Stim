from __future__ import annotations

import dataclasses
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]


_GATE_TABLEAU_CACHE: dict[str, tuple[stim.Tableau, stim.Tableau]] = {}


def get_cached_gate_tableau(gate_name: str) -> tuple[stim.Tableau, stim.Tableau]:
    """Return cached (forward_tableau, inverse_tableau) for a named Stim gate."""
    cached = _GATE_TABLEAU_CACHE.get(gate_name)
    if cached is not None:
        return cached
    fwd = stim.Tableau.from_named_gate(gate_name)
    inv = fwd.inverse()
    _GATE_TABLEAU_CACHE[gate_name] = (fwd, inv)
    return fwd, inv


def _pauli_image(pauli, q, x2x_w, x2z_w, z2x_w, z2z_w, x_kappa, z_kappa):
    """Reference-frame image (a_words, b_words, kappa) of Pauli_q for pauli in 'X', 'Y', 'Z'."""
    if pauli == "Z":
        return z2x_w[q], z2z_w[q], int(z_kappa[q])
    if pauli == "X":
        return x2x_w[q], x2z_w[q], int(x_kappa[q])
    dp = int(np.bitwise_count(x2z_w[q] & z2x_w[q]).sum()) & 1
    return (x2x_w[q] ^ z2x_w[q], x2z_w[q] ^ z2z_w[q],
            (1 + int(x_kappa[q]) + int(z_kappa[q]) + (dp << 1)) & 3)


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

    def snapshot_packed(self) -> PackedSymplecticTables:
        return (
            self.x2x.copy(),
            self.x2z.copy(),
            self.z2x.copy(),
            self.z2z.copy(),
            self.x_kap.copy(),
            self.z_kap.copy(),
        )

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
            self.x2x[q_arr], self.z2x[q_arr] = (
                self.z2x[q_arr].copy(),
                self.x2x[q_arr].copy(),
            )
            self.x2z[q_arr], self.z2z[q_arr] = (
                self.z2z[q_arr].copy(),
                self.x2z[q_arr].copy(),
            )
            self.x_kap[q_arr], self.z_kap[q_arr] = (
                self.z_kap[q_arr].copy(),
                self.x_kap[q_arr].copy(),
            )
            return

        if gate_name in ("CZ", "ZCZ"):
            q0 = q_arr[0::2]
            q1 = q_arr[1::2]
            dp0 = np.bitwise_count(self.x2z[q0] & self.z2x[q1]).sum(axis=1) & 1
            dp1 = np.bitwise_count(self.x2z[q1] & self.z2x[q0]).sum(axis=1) & 1
            self.x2x[q0] ^= self.z2x[q1]
            self.x2z[q0] ^= self.z2z[q1]
            self.x_kap[q0] = (
                self.x_kap[q0] + self.z_kap[q1] + (dp0 << 1)
            ) & 3
            self.x2x[q1] ^= self.z2x[q0]
            self.x2z[q1] ^= self.z2z[q0]
            self.x_kap[q1] = (
                self.x_kap[q1] + self.z_kap[q0] + (dp1 << 1)
            ) & 3
            return

        if gate_name in ("CX", "CNOT", "ZCX"):
            c = q_arr[0::2]
            t = q_arr[1::2]
            dpx = np.bitwise_count(self.x2z[c] & self.x2x[t]).sum(axis=1) & 1
            dpz = np.bitwise_count(self.z2z[t] & self.z2x[c]).sum(axis=1) & 1
            self.x2x[c] ^= self.x2x[t]
            self.x2z[c] ^= self.x2z[t]
            self.x_kap[c] = (self.x_kap[c] + self.x_kap[t] + (dpx << 1)) & 3
            self.z2x[t] ^= self.z2x[c]
            self.z2z[t] ^= self.z2z[c]
            self.z_kap[t] = (self.z_kap[t] + self.z_kap[c] + (dpz << 1)) & 3
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
            phase_shift = 3 if gate_name == "S" else 1
            dp = (
                np.bitwise_count(self.x2z[q_arr] & self.z2x[q_arr]).sum(axis=1)
                & 1
            )
            self.x2x[q_arr] ^= self.z2x[q_arr]
            self.x2z[q_arr] ^= self.z2z[q_arr]
            self.x_kap[q_arr] = (
                phase_shift
                + self.x_kap[q_arr]
                + self.z_kap[q_arr]
                + (dp << 1)
            ) & 3
            return

        if gate_name == "SWAP":
            q0 = q_arr[0::2]
            q1 = q_arr[1::2]
            self.x2x[q0], self.x2x[q1] = (
                self.x2x[q1].copy(),
                self.x2x[q0].copy(),
            )
            self.x2z[q0], self.x2z[q1] = (
                self.x2z[q1].copy(),
                self.x2z[q0].copy(),
            )
            self.z2x[q0], self.z2x[q1] = (
                self.z2x[q1].copy(),
                self.z2x[q0].copy(),
            )
            self.z2z[q0], self.z2z[q1] = (
                self.z2z[q1].copy(),
                self.z2z[q0].copy(),
            )
            self.x_kap[q0], self.x_kap[q1] = (
                self.x_kap[q1].copy(),
                self.x_kap[q0].copy(),
            )
            self.z_kap[q0], self.z_kap[q1] = (
                self.z_kap[q1].copy(),
                self.z_kap[q0].copy(),
            )
            return

        # General local fallback for any 1- or 2-qubit named gate via its 1x1 or 2x2 inverse tableau
        _, inv_gate = get_cached_gate_tableau(gate_name)
        arity = len(inv_gate)
        for i in range(0, len(targets), arity):
            grp = targets[i : i + arity]
            old_x = [
                (self.x2x[q].copy(), self.x2z[q].copy(), int(self.x_kap[q]))
                for q in grp
            ]
            old_z = [
                (self.z2x[q].copy(), self.z2z[q].copy(), int(self.z_kap[q]))
                for q in grp
            ]
            for r_loc, q_phys in enumerate(grp):
                for is_z_out, p_str in (
                    (False, inv_gate.x_output(r_loc)),
                    (True, inv_gate.z_output(r_loc)),
                ):
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
                            ax, bx_w, kx = old_x[k_loc]
                            az, bz_w, kz = old_z[k_loc]
                            dp = int(np.bitwise_count(bx_w & az).sum()) & 1
                            aw_t = ax ^ az
                            bw_t = bx_w ^ bz_w
                            kap_t = (1 + kx + kz + (dp << 1)) & 3
                        dp_acc = int(np.bitwise_count(bw_acc & aw_t).sum()) & 1
                        aw_acc ^= aw_t
                        bw_acc ^= bw_t
                        kap_acc = (kap_acc + kap_t + (dp_acc << 1)) & 3
                    if is_z_out:
                        self.z2x[q_phys] = aw_acc
                        self.z2z[q_phys] = bw_acc
                        self.z_kap[q_phys] = kap_acc
                    else:
                        self.x2x[q_phys] = aw_acc
                        self.x2z[q_phys] = bw_acc
                        self.x_kap[q_phys] = kap_acc

    def measure_z(
        self, q: int, *, is_reset: bool = False, m_ref_override: int = 0
    ) -> DetMeasMeta | RandMeasMeta:
        """Execute a reference Z_q measurement/reset via direct rank-1 symplectic transvection."""
        aw = self.z2x[q]
        if not np.any(aw):
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

        nz_w = int(np.flatnonzero(aw)[0])
        low_val = int(aw[nz_w])
        p = (nz_w << 6) + (low_val & -low_val).bit_length() - 1
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
        e_p_words = np.zeros(self.n_words, dtype=np.uint64)
        e_p_words[p_w] = p_mask

        m_ref = 0 if is_reset else int(m_ref_override)
        kap = int(self.z_kap[q])
        kappa_rot = (kap + 1) & 3
        ab_rot_dot = int(np.bitwise_count(a_words & b_rot_words).sum()) & 1
        kap_iprot = ((m_ref << 1) - kap + (ab_rot_dot << 1)) & 3

        # Step A1.5: Direct rank-1 transvection Q_r -> (i P_rot) Q_r on anticommuting rows
        gx = (
            np.bitwise_count(
                (self.x2x & b_rot_words) ^ (self.x2z & a_words)
            ).sum(axis=1)
            & 1
        ) != 0
        ix = np.flatnonzero(gx)
        if len(ix):
            dpx = np.bitwise_count(self.x2x[ix] & b_rot_words).sum(axis=1) & 1
            self.x_kap[ix] = (kap_iprot + self.x_kap[ix] + (dpx << 1)) & 3
            self.x2x[ix] ^= a_words
            self.x2z[ix] ^= b_rot_words

        gz = (
            np.bitwise_count(
                (self.z2x & b_rot_words) ^ (self.z2z & a_words)
            ).sum(axis=1)
            & 1
        ) != 0
        iz = np.flatnonzero(gz)
        if len(iz):
            dpz = np.bitwise_count(self.z2x[iz] & b_rot_words).sum(axis=1) & 1
            self.z_kap[iz] = (kap_iprot + self.z_kap[iz] + (dpz << 1)) & 3
            self.z2x[iz] ^= a_words
            self.z2z[iz] ^= b_rot_words

        self._inv_dirty = True

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
