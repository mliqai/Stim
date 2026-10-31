"""Seeded golden-digest guard across all stimside simulators and modes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

from stimside.op_handlers.abstract_op_handler import _TrivialOpHandler
from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
from stimside.op_handlers.leakage_handlers.leakage_uint8_flip import LeakageUint8 as LeakageUint8Flip
from stimside.op_handlers.leakage_handlers.leakage_uint8_magic import LeakageUint8Magic
from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import LeakageUint8 as LeakageUint8Tableau
from stimside.simulator_coset import CosetsideSimulator
from stimside.simulator_flip import FlipsideSimulator
from stimside.simulator_magic import MagicsideSimulator
from stimside.simulator_tableau import TablesideSimulator

DIGESTS_PATH = Path(__file__).with_name("golden_digests.json")
SEED = 20261031
BATCH = 8

_ALL_STD = (
    "flip",
    "tab_nosync",
    "tab_sync",
    "coset_nosync_cpp",
    "coset_sync",
    "magic_ds",
    "magic_dq",
    "magic_ss",
    "magic_sq",
    "magic_sync",
)
_LEAK_NO_FLIP_TAGS = (
    "tab_nosync",
    "tab_sync",
    "coset_nosync_cpp",
    "coset_sync",
    "magic_ds",
    "magic_sync",
)
_LEAK_SHARED_TAGS = ("flip",) + _LEAK_NO_FLIP_TAGS
_MAGIC_ALL = ("magic_ds", "magic_dq", "magic_ss", "magic_sq")

CORPUS: tuple[tuple[str, tuple[str, ...], bool, str], ...] = (
    (
        "c01_clifford_noise_feedback",
        _ALL_STD,
        False,
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        R 0 1 2
        RX 3
        RY 4
        H 0 1
        CX 0 1 1 2
        DEPOLARIZE1(0.08) 0 1 2 3 4
        DEPOLARIZE2(0.06) 0 1 2 3
        PAULI_CHANNEL_1(0.02, 0.03, 0.04) 0 2
        PAULI_CHANNEL_2(0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01) 1 3
        REPEAT 2 {
            SHIFT_COORDS(0, 1)
            CX 0 2 3 4
            X_ERROR(0.05) 0 3
            Z_ERROR(0.05) 1 4
            MR(0.03) 2
            CX rec[-1] 1
            DETECTOR(0, 0) rec[-1]
        }
        MX 3
        MY 4
        M(0.02) 0 1
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-3] rec[-4]
        """,
    ),
    (
        "c02_mpp_mxx_heralded_corr",
        _ALL_STD,
        False,
        """
        R 0 1 2 3
        H 0 1 2
        CX 0 1 1 2
        HERALDED_ERASE(0.15) 0 !1
        HERALDED_PAULI_CHANNEL_1(0.02, 0.03, 0.04, 0.05) 2 !3
        CORRELATED_ERROR(0.2) X0 Z1
        ELSE_CORRELATED_ERROR(0.25) Y1 X2
        ELSE_CORRELATED_ERROR(0.3) Z3
        MXX(0.04) 0 1
        MYY 1 2
        MZZ 2 3
        MPP X0*Z1 Y1*Z2 !Z3
        M 0 1 2 3
        DETECTOR rec[-1] rec[-2] rec[-5]
        OBSERVABLE_INCLUDE(0) rec[-3] rec[-4] rec[-8]
        """,
    ),
    (
        "c02b_spp_clifford",
        ("flip", "tab_nosync", "tab_sync", "magic_ds", "magic_dq", "magic_ss", "magic_sq", "magic_sync"),
        False,
        """
        R 0 1 2
        H 0 1 2
        SPP X0*Z1 !Z2
        DEPOLARIZE1(0.05) 0 1 2
        SPP_DAG Y1*Z2
        H 0 1 2
        M 0 1 2
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-3]
        """,
    ),
    (
        "c03_leakage_transitions_swap_meas",
        _LEAK_SHARED_TAGS,
        True,
        """
        R 0 1 2 3
        X 1
        H 2
        REPEAT 2 {
            I[LEAKAGE_TRANSITION_1: (0.25, U-->2) (0.3, U-->3)] 0 1 2
            II[LEAKAGE_TRANSITION_2: (0.2, U_2-->2_2) (0.25, U_3-->D_D) (0.2, 2_U-->U_D)] 0 1 2 3
            SWAP[LEAKAGE_SWAP] 0 2 1 3
            I[LEAKAGE_TRANSITION_1: (0.2, 2-->U) (0.15, 3-->U) (0.15, 2<->3)] 0 1 2 3
            MPAD[LEAKAGE_MEASUREMENT: (0.85, 2) (0.9, 3): 0 1 2 3] 0 0 0 0
            MR[LEAKAGE_PROJECTION_Z: (0.9, 2) (0.8, 3)] 0 1
            DETECTOR rec[-1] rec[-3]
        }
        MX[LEAKAGE_PROJECTION_Z: (0.85, 2)] 2
        M[LEAKAGE_PROJECTION_Z: (0.9, 2) (0.95, 3)] 0 1 3
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-3] rec[-4]
        """,
    ),
    (
        "c04_leakage_conditioning_and_repeated_targets",
        _LEAK_NO_FLIP_TAGS,
        True,
        """
        R 0 1 2 3
        X 1
        H 0 3
        I[LEAKAGE_TRANSITION_1: (0.4, 0-->2) (0.45, 1-->3)] 0 1
        II[LEAKAGE_TRANSITION_2: (0.2, 0_2-->2_2) (0.2, 2_U-->0_1) (0.15, 3_U-->X_Y)] 0 2 1 3
        H[CONDITIONED_ON_SELF: U] 0 2
        S[CONDITIONED_ON_SELF: 1] 0 3
        X[CONDITIONED_ON_SELF: 0 1] 1 3
        H[CONDITIONED_ON_OTHER: 0 : 1] 2
        I_ERROR[CONDITIONED_ON_SELF: U](0.25) 0 0 2
        X_ERROR[CONDITIONED_ON_SELF: 2 3](0.35) 0 0 1
        Z_ERROR[CONDITIONED_ON_OTHER: 2 3 : 0 1](0.4) 2 3 2 2
        CX[CONDITIONED_ON_PAIR: (U, U)] 0 2 1 3
        DEPOLARIZE2[CONDITIONED_ON_PAIR: (2, U) (3, U)](0.3) 0 2 1 3
        I[LEAKAGE_TRANSITION_1: (0.2, 2-->0) (0.2, 3-->1)] 0 1
        MPAD[LEAKAGE_MEASUREMENT: (0.1, 0) (0.2, 1) (0.9, 2) (0.9, 3): 0 1] 0 0
        DETECTOR rec[-1]
        M[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.9, 1) (0.85, 2) (0.95, 3)] 0 1 2 3
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-3] rec[-4]
        """,
    ),
    (
        "c04b_leakage_detector_tab_coset",
        ("tab_nosync", "tab_sync", "coset_nosync_cpp", "coset_sync"),
        True,
        """
        R 0 1
        I[LEAKAGE_TRANSITION_1: (0.4, U-->2)] 0 1
        MPAD[LEAKAGE_MEASUREMENT: (0.9, 2): 0 1] 0 0
        DETECTOR[LEAKAGE_DETECTOR: 0] rec[-2]
        M 0 1
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]
        """,
    ),
    (
        "c05_flip_only_tags",
        ("flip",),
        True,
        """
        REPEAT 2 {
            R[LEAKAGE_TRANSITION_Z: (0.2, 0-->2)] 0 1 2
            X 1
            I[LEAKAGE_TRANSITION_Z: (0.25, 0-->2) (0.3, 1-->2) (0.2, 2-->0)] 0 1 2
            II[LEAKAGE_CONTROLLED_ERROR: (0.35, 2-->Z) (0.3, 2-->X)] 0 1 1 2
            MPAD[LEAKAGE_MEASUREMENT: (0.9, 2): 0 1 2] 0 0 0
            MR[LEAKAGE_PROJECTION_Z: (0.95, 2)] 2
            DETECTOR rec[-1] rec[-2]
        }
        M 0 1
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]
        """,
    ),
    (
        "c06_magic_t_ccz_clifford",
        _MAGIC_ALL,
        False,
        """
        R 0 1 2
        RX 3
        H 0 1 2
        REPEAT 2 {
            I[T] 0 1
            I[CCZ] 0 1 2
            CX 2 3
            I[T_DAG] 2 3
            SPP X0*Z1
            DEPOLARIZE1(0.04) 0 1 2 3
            MR(0.02) 3
            CZ rec[-1] 0
            DETECTOR rec[-1]
        }
        H 0 1 2
        M(0.02) 0 1 2
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-3]
        """,
    ),
    (
        "c07_magic_inline_faults",
        _MAGIC_ALL,
        False,
        """
        R 0 1 2
        H 0 1 2
        I[CLIFFORD_FAULT: (0.2, H) (0.15, S)] 0
        I[T] 0 1
        I[CLIFFORD_FAULT: (0.25, CZ)] 0 1
        I[CCZ] 0 1 2
        HERALDED_ERASE(0.1) 2 !1
        CORRELATED_ERROR(0.15) X0 Z2
        ELSE_CORRELATED_ERROR(0.2) Y1
        H 0 1 2
        M 0 1 2
        DETECTOR rec[-1] rec[-4]
        OBSERVABLE_INCLUDE(0) rec[-2] rec[-3]
        """,
    ),
    (
        "c08_magic_with_leakage",
        ("magic_ds", "magic_sync"),
        True,
        """
        R 0 1 2
        H 0 1 2
        I[T] 0 1 2
        I[LEAKAGE_TRANSITION_1: (0.3, 0-->2) (0.35, 1-->2)] 0 1
        CX[CONDITIONED_ON_PAIR: (U, U)] 0 1 1 2
        I[T_DAG] 1 2
        I[CCZ] 0 1 2
        SWAP[LEAKAGE_SWAP] 0 2
        MPAD[LEAKAGE_MEASUREMENT: (0.15, U) (0.9, 2): 0 1] 0 0
        MPAD[LEAKAGE_MEASUREMENT: (0.15, U) (0.9, 2): 0 1] 0 0
        H[CONDITIONED_ON_SELF: U] 0 1 2
        M[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.9, 1) (0.95, 2)] 0 1 2
        DETECTOR rec[-1] rec[-4]
        OBSERVABLE_INCLUDE(0) rec[-2] rec[-3]
        """,
    ),
    (
        "c09_multiword_70q_leakage",
        _LEAK_SHARED_TAGS,
        True,
        """
        R 0 1 63 64 65 69
        H 0 63 65
        CX 0 64 63 69
        I[LEAKAGE_TRANSITION_1: (0.3, U-->2)] 0 1 64 69
        II[LEAKAGE_TRANSITION_2: (0.25, U_2-->2_D)] 63 64 65 69
        DEPOLARIZE2(0.05) 0 65 63 69
        MPP X0*X64 Z63*Z69
        MPAD[LEAKAGE_MEASUREMENT: (0.9, 2): 0 1 64 69] 0 0 0 0
        M[LEAKAGE_PROJECTION_Z: (0.9, 2)] 0 1 63 64 65 69
        DETECTOR rec[-1] rec[-7]
        OBSERVABLE_INCLUDE(0) rec[-2] rec[-5] rec[-11]
        """,
    ),
    (
        "c10_magic_multiword_68q",
        _MAGIC_ALL,
        False,
        """
        R 0 63 67
        H 0 63 67
        I[T] 0 67
        I[CLIFFORD_FAULT: (0.2, CX)] 0 63
        I[CCZ] 0 63 67
        I[T_DAG] 63
        DEPOLARIZE1(0.03) 0 63 67
        H 0 63 67
        M 0 63 67
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-3]
        """,
    ),
)


def _hash_payload(
    recs: np.ndarray,
    dets: np.ndarray,
    obs: np.ndarray,
    u2l: list[np.ndarray] | None,
    lev: list[Any] | None,
) -> str:
    h = hashlib.sha256()
    for tag, arr in (("R", recs), ("D", dets), ("O", obs)):
        b = np.ascontiguousarray(arr, dtype=np.bool_)
        h.update(f"{tag}:{b.shape}:".encode())
        h.update(b.tobytes())
    if u2l is not None:
        h.update(b"U2L:" + repr([[int(x) for x in row] for row in u2l]).encode())
    if lev is not None:
        h.update(
            b"LEV:"
            + repr(
                [
                    [(int(e.op_index), int(e.qubit), int(e.old_state), int(e.new_state)) for e in shot]
                    for shot in lev
                ]
            ).encode()
        )
    return h.hexdigest()[:32]


def _run_case(circuit: stim.Circuit, mode: str, has_leakage: bool) -> str:
    rec_flag = bool(has_leakage)
    if mode == "flip":
        op_h = LeakageUint8Flip() if has_leakage else _TrivialOpHandler()
        sim = FlipsideSimulator(
            circuit,
            compiled_op_handler=op_h.compile_op_handler(circuit=circuit, batch_size=BATCH),
            batch_size=BATCH,
            seed=SEED,
            record_unleaked_to_leaked=rec_flag,
        )
        sim.run()
        u2l = sim.get_unleaked_to_leaked_records() if rec_flag else None
        return _hash_payload(
            sim.get_final_measurement_records(),
            sim.get_detector_flips().T,
            sim.get_observable_flips().T,
            u2l,
            None,
        )

    if mode in ("tab_nosync", "tab_sync"):
        op_h = LeakageUint8Tableau()
        if mode == "tab_sync":
            sim_t = TablesideSimulator(
                circuit,
                compiled_op_handler=op_h.compile_op_handler(circuit=circuit, batch_size=BATCH),
                batch_size=BATCH,
                seed=SEED,
                sync_tableside_rng=True,
                record_unleaked_to_leaked=rec_flag,
                record_leakage_events=rec_flag,
            )
            sim_t.run()
            return _hash_payload(
                sim_t.get_final_measurement_records(),
                sim_t.get_detector_flips(),
                sim_t.get_observable_flips(),
                sim_t.get_unleaked_to_leaked_records() if rec_flag else None,
                sim_t.get_leakage_events() if rec_flag else None,
            )
        sim_t = TablesideSimulator(
            circuit,
            compiled_op_handler=op_h.compile_op_handler(circuit=circuit, batch_size=1),
            batch_size=1,
            seed=SEED,
            record_unleaked_to_leaked=rec_flag,
            record_leakage_events=rec_flag,
        )
        assert sim_t._cpp_runner is not None
        recs, dets, obs, u2l_all, lev_all = [], [], [], [], []
        for s in range(BATCH):
            if s > 0:
                sim_t.clear()
            sim_t.run()
            recs.append(sim_t.get_final_measurement_records()[0].copy())
            dets.append(sim_t.get_detector_flips()[0].copy())
            obs.append(sim_t.get_observable_flips()[0].copy())
            if rec_flag:
                u2l_all.append(sim_t.get_unleaked_to_leaked_records()[0])
                lev_all.append(sim_t.get_leakage_events()[0])
        return _hash_payload(
            np.vstack(recs),
            np.vstack(dets),
            np.vstack(obs),
            u2l_all if rec_flag else None,
            lev_all if rec_flag else None,
        )

    if mode.startswith("coset_"):
        op_h = LeakageUint8Coset() if has_leakage else _TrivialOpHandler()
        sim_c = CosetsideSimulator(
            circuit,
            compiled_op_handler=op_h.compile_op_handler(circuit=circuit, batch_size=BATCH),
            batch_size=BATCH,
            seed=SEED,
            sync_tableside_rng=(mode == "coset_sync"),
            record_unleaked_to_leaked=rec_flag,
            record_leakage_events=rec_flag,
        )
        sim_c.run()
        return _hash_payload(
            sim_c.get_final_measurement_records(),
            sim_c.get_detector_flips(),
            sim_c.get_observable_flips(),
            sim_c.get_unleaked_to_leaked_records() if rec_flag else None,
            sim_c.get_leakage_events() if rec_flag else None,
        )

    m_cfg = {
        "magic_ds": ("DIRECT", "STREAM", False),
        "magic_dq": ("DIRECT", "QUERY", False),
        "magic_ss": ("SELECTOR", "STREAM", False),
        "magic_sq": ("SELECTOR", "QUERY", False),
        "magic_sync": ("DIRECT", "STREAM", True),
    }[mode]
    op_h = LeakageUint8Magic() if has_leakage else _TrivialOpHandler()
    sim_m = MagicsideSimulator(
        circuit,
        compiled_op_handler=op_h.compile_op_handler(circuit=circuit, batch_size=BATCH),
        batch_size=BATCH,
        seed=SEED,
        fault_mode=m_cfg[0],
        backend=m_cfg[1],
        sync_tableside_rng=m_cfg[2],
        record_unleaked_to_leaked=rec_flag,
        record_leakage_events=rec_flag,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sim_m.run()
    return _hash_payload(
        sim_m.get_final_measurement_records(),
        sim_m.get_detector_flips(),
        sim_m.get_observable_flips(),
        sim_m.get_unleaked_to_leaked_records() if rec_flag else None,
        sim_m.get_leakage_events() if rec_flag else None,
    )


def _compute_all_digests() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for name, modes, has_leak, text in CORPUS:
        circ = stim.Circuit(text)
        out[name] = {m: _run_case(circ, m, has_leak) for m in modes}
    return out


def test_golden_digests() -> None:
    actual = _compute_all_digests()
    if os.environ.get("STIMSIDE_UPDATE_GOLDENS") == "1":
        DIGESTS_PATH.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    if not DIGESTS_PATH.is_file():
        pytest.fail(f"Missing {DIGESTS_PATH}; generate with STIMSIDE_UPDATE_GOLDENS=1")
    expected = json.loads(DIGESTS_PATH.read_text(encoding="utf-8"))
    assert actual == expected
