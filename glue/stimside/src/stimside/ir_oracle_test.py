"""Byte-for-byte structural oracle test suite for WP-27 (RFC §6 & §7.3).

Self-contained pytest suite (zero external ``e2e_tests/`` dependencies) pinning
exact structural SHA-256 digests for 24 circuit walkers across 152 circuits
(3,648 total entries: 18 golden ``CORPUS`` + 4 ``s01-s04`` + 5 ``x05-x09`` +
10 ``y10-y15`` + 6 ``y16-y21`` error-order circuits + 14 ``p01-p14`` witness
circuits + 3 ``z01-z03`` zero-target circuits + 7 ``m01-m07`` magic selector
and feedback circuits + 5 ``t01-t05`` tableau precompiler circuits + 80
``g_<NAME>`` circuits covering every non-``REPEAT`` Stim instruction name).

Per D-46 and RFC Q9, ``FROZEN_DIGESTS`` was generated once from the Stage C
start tree (FINAL ``CP-B`` + C1 ``circuit_ir.py`` / ``ir_tags.py``) across
multiple independent Python 3.13 / 3.11 / ``-BO`` runs and is frozen post-gate.
Never re-bless these digests without an explicit owner exception.
"""

from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import re
from collections.abc import Callable, Sequence
from typing import Any
from unittest import mock

import numpy as np
import pytest
import stim

import stimside.dem_generators.marginal_builder as mb
from stimside import golden_test
from stimside._coset.reference_compile import CompiledReferenceCircuit
from stimside._flip.sync_reference import compute_sync_reference
from stimside._magic.gates import (
    CliffordFaultChannel,
    FaultBranch,
    extract_circuit_faults,
)
from stimside._magic.plan import _compile_circuit_pass
from stimside._tableau.runner import PrecompiledTablesideCircuit, collapses_z
from stimside.dem_generators.dem_generator_marginal import has_leakage_swap
from stimside.op_handlers.leakage_handlers.leakage_tag_parsing_tableau import (
    parse_leakage_in_circuit,
)
from stimside.op_handlers.leakage_handlers.leakage_uint8_flip import LeakageUint8Flip
from stimside.op_handlers.leakage_handlers.leakage_uint8_magic import (
    _parse_leakage_in_circuit_magic,
    _strip_circuit_for_noiseless_ref,
)
from stimside.util.circuit import unroll_circuit
from stimside.util.known_states import compute_known_states_uint8
from stimside.util.leakage_events import unrolled_to_flattened_indices

_HEX_ADDR_RE = re.compile(r"0x[0-9a-fA-F]{6,}")

_CONTRACT_EXC: tuple[type[Exception], ...] = (
    ValueError,
    NotImplementedError,
    TypeError,
    RuntimeError,
    LookupError,
    AttributeError,
    ArithmeticError,
    AssertionError,
)

_TAG2 = "MPAD[LEAKAGE_MEASUREMENT: (0, 0) (1, 1) : 1 2]"
_PROJ = "LEAKAGE_PROJECTION_Z: (0.05, 0) (0.9, 1)"

_SK_COSET = frozenset({"QUBIT_COORDS", "SHIFT_COORDS"})
_SK_TAB = frozenset(
    {"DETECTOR", "TICK", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS"}
)

_SUPPLEMENT_S: tuple[tuple[str, str], ...] = (
    (
        "s01_rec_feedback_after_asym_readout",
        (
            "R 0 1 2\n"
            "M[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.9, 1) (0.85, 2)] 0\n"
            "CX rec[-1] 1\n"
            "M 1 2\n"
            "DETECTOR rec[-1] rec[-3]\n"
            "DETECTOR rec[-2]\n"
        ),
    ),
    (
        "s02_untagged_1q_2q_with_leak_tag",
        (
            "R 0 1\n"
            "I[LEAKAGE_TRANSITION_1: (0.3, U-->2)] 0\n"
            "H 0 1\n"
            "CX 0 1\n"
            "CX 0 1\n"
            "H 0 1\n"
            "M 0 1\n"
            "DETECTOR rec[-1] rec[-2]\n"
        ),
    ),
    (
        "s04_untagged_1q_on_leaked_qubit",
        (
            "R 0 1 2\n"
            "I_ERROR[LEAKAGE_TRANSITION_1: (0.35, U-->2)] 0\n"
            "H 0\n"
            "H 0\n"
            "X_ERROR(0.08) 0 1\n"
            "CX 0 2 1 2\n"
            "MR 2\n"
            "DETECTOR rec[-1]\n"
            "M 0 1\n"
            "DETECTOR rec[-1] rec[-2]\n"
        ),
    ),
    (
        "s03_repeat_fused_boundary",
        (
            "R 0 1\n"
            "REPEAT 3 {\n"
            "  I[LEAKAGE_TRANSITION_1: (0.1, U-->2)] 0\n"
            "  X_ERROR(0.1) 0\n"
            "  X_ERROR(0.1) 0\n"
            "  H 1\n"
            "  H 1\n"
            "  M 0\n"
            "  DETECTOR rec[-1]\n"
            "}\n"
            "X_ERROR(0.1) 1\n"
            "REPEAT 2 {\n"
            "  X_ERROR(0.1) 1\n"
            "}\n"
            "M 1\n"
            "DETECTOR rec[-1]\n"
        ),
    ),
)

_SUPPLEMENT_X: tuple[tuple[str, str], ...] = (
    (
        "x05_nested_repeat_boundary_fusion",
        (
            "R 0\n"
            "REPEAT 2 {\n"
            "  REPEAT 2 {\n"
            "    X_ERROR(0.1) 0\n"
            "  }\n"
            "  X_ERROR(0.1) 0\n"
            "  M 0\n"
            "  DETECTOR rec[-1]\n"
            "}"
        ),
    ),
    (
        "x06_shift_coords_detector_in_repeat",
        (
            "QUBIT_COORDS(0,0) 0\n"
            "QUBIT_COORDS(1,0) 1\n"
            "R 0 1\n"
            "REPEAT 3 {\n"
            "  CX 0 1\n"
            "  M 1\n"
            "  SHIFT_COORDS(0,1)\n"
            "  DETECTOR(1,0) rec[-1]\n"
            "}"
        ),
    ),
    (
        "x07_sweep_feedback",
        "R 0 1\nH 0\nCX sweep[0] 1\nM 1\nDETECTOR rec[-1]\nM 0",
    ),
    (
        "x08_tag_copy_fused_across_repeat",
        f"R 0 1 2\nREPEAT 2 {{\n  {_TAG2} 0 0\n}}\nM 0",
    ),
    (
        "x09_tag_copy_nonmultiple_fused",
        f"R 0 1 2\nREPEAT 2 {{\n  {_TAG2} 0 0 0\n}}\nM 0",
    ),
)

_SUPPLEMENT_Y_REV2: tuple[tuple[str, str], ...] = (
    (
        "y10_lt1_proj_fusion_peephole",
        (
            "R 0 1 2 3\n"
            "I[LEAKAGE_TRANSITION_1: (0.1, 1-->2)] 0 1 2 3\n"
            "M[LEAKAGE_PROJECTION_Z: (1.0, 2)] 0 1 2 3\n"
            "DETECTOR rec[-1]"
        ),
    ),
    (
        "y11_rx_x_known_state_row",
        (
            "RX 0\n"
            "I_ERROR[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0\n"
            "X 0\n"
            "MX 0\n"
            "DETECTOR rec[-1]\n"
            "MPAD[LEAKAGE_MEASUREMENT: (1.0, 2): 0] 0"
        ),
    ),
    (
        "y11b_two_qubit_prep_then_leak",
        (
            "R 0 1\n"
            "H 1\n"
            "CX 1 0\n"
            "I_ERROR[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0\n"
            "M 0 1\n"
            "DETECTOR rec[-1] rec[-2]\n"
            "MPAD[LEAKAGE_MEASUREMENT: (1.0, 2): 0] 0"
        ),
    ),
    (
        "y11c_x_then_leak",
        (
            "R 0\n"
            "X 0\n"
            "I_ERROR[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0\n"
            "M 0\n"
            "DETECTOR rec[-1]\n"
            "MPAD[LEAKAGE_MEASUREMENT: (1.0, 2): 0] 0"
        ),
    ),
    (
        "y12_tick_between_steps",
        "R 0 1\nH 0\nTICK\nCX 0 1\nTICK\nM 0 1\nDETECTOR rec[-1] rec[-2]",
    ),
    (
        "y13_split_copies_then_other_op",
        (
            f"R 0 1 2\nH 0\n{_TAG2} 0 0\nH 1\n"
            "X_ERROR(0.1) 2\nM 0 1 2\nDETECTOR rec[-1]"
        ),
    ),
    (
        "y13b_split_copies_leak_source",
        (
            "R 0 1 2\n"
            "I_ERROR[LEAKAGE_TRANSITION_1: (0.2, U-->2)] 0 1\n"
            f"{_TAG2} 0 0 0 0\n"
            "M 0 1 2\n"
            "DETECTOR rec[-1] rec[-2]"
        ),
    ),
    (
        "y14_memo_hides_second_parse",
        f"R 0 1\nM[{_PROJ}] 0\nM[{_PROJ}](0.1) 1\nDETECTOR rec[-1]",
    ),
    (
        "y14b_memo_same_name_tag_diff_args",
        f"R 0 1\nM[{_PROJ}](0.1) 1\nM[{_PROJ}] 0\nDETECTOR rec[-1]",
    ),
    (
        "y15_nonmultiple_copy_alone",
        f"R 0 1 2\n{_TAG2} 0 0 0\nM 0",
    ),
)

_SUPPLEMENT_Y_REV3: tuple[tuple[str, str], ...] = (
    (
        "y16_spp_then_badtag",
        f"R 0 1\nSPP X0*Z1\nM[{_PROJ}](0.1) 0\nM 0 1\nDETECTOR rec[-1]",
    ),
    (
        "y17_badtag_then_spp",
        f"R 0 1\nM[{_PROJ}](0.1) 0\nSPP X0*Z1\nM 0 1\nDETECTOR rec[-1]",
    ),
    (
        "y18_qubitrange_then_badtag",
        (
            "R 0 1\n"
            "MPAD[LEAKAGE_MEASUREMENT: (0, 0) (1, 1) : 9] 0\n"
            f"M[{_PROJ}](0.1) 0\n"
            "M 0 1\n"
            "DETECTOR rec[-1]"
        ),
    ),
    (
        "y19_unsupported_then_badtag",
        (
            "R 0 1\n"
            "I_ERROR[LEAKAGE_TRANSITION_1: (0.1, 1-->2)] 0\n"
            f"M[{_PROJ}](0.1) 0\n"
            "M 0 1\n"
            "DETECTOR rec[-1]"
        ),
    ),
    (
        "y20_spp_carrying_a_badtag",
        f"R 0 1\nSPP[{_PROJ}] X0*Z1\nM 0 1\nDETECTOR rec[-1]",
    ),
    (
        "y21_state01_tag_and_qubit_out_of_range",
        (
            "R 0 1 2\n"
            "H[CONDITIONED_ON_OTHER: 1 : 9] 2\n"
            "M 0 1\n"
            "DETECTOR rec[-1]"
        ),
    ),
)

_SUPPLEMENT_WITNESSES: tuple[tuple[str, str], ...] = (
    ("p01_inv_m", "R 0\nM !0"),
    ("p02_inv_mzz", "R 0 1\nMZZ 0 !1"),
    ("p03_mpad1", "MPAD 1\nM 0"),
    ("p04_mpp_rep", "R 0 1\nMPP X0*Z0*Y0"),
    ("p05_mpp_neg", "RX 0\nMPP Z0*X0*Z0"),
    ("p06_fb_mpad", "R 0\nMPAD 1\nCX rec[-1] 0\nM 0"),
    ("p07_fb_qrec", "R 0 1\nX 0\nM 0\nXCZ 1 rec[-1]\nM 1"),
    ("p08_fb_cz_qrec", "R 0 1\nX 0\nM 0\nH 1\nCZ 1 rec[-1]\nH 1\nM 1"),
    ("p09_fb_mixed", "R 0 1 2\nX 0\nM 0\nCX rec[-1] 1 0 2\nM 1 2"),
    ("p10_mpad_z_super", f"R 0 1 2\nH 1\n{_TAG2} 0 0\nM 1 2"),
    (
        "p11_lt_reset",
        "R 0\nX 0\nI[LEAKAGE_TRANSITION_1: (0.1, 2-->1)] 0\nM 0",
    ),
    ("p12_bwd_mpp", "R 0 1 2\nX 2\nM 2\nMPP Z0*Z1\nM 0"),
    ("p13_rx_order", "R 0 1 2\nH 0\nCX 0 2\nRX 1 0\nMX 2"),
    ("p14_xcz_sweep_target", "R 0 1\nH 0\nXCZ sweep[0] 1\nM 0 1"),
)

_SUPPLEMENT_Z: tuple[tuple[str, str], ...] = (
    (
        "z01_zero_target_shift_fuse",
        (
            "R 0 1\n"
            "SHIFT_COORDS(1)\n"
            "R\n"
            "H 0\n"
            "SHIFT_COORDS(1)\n"
            "H\n"
            "X_ERROR(0.1) 0\n"
            "SHIFT_COORDS(1)\n"
            "X_ERROR(0.2)\n"
            "H[a] 1\n"
            "SHIFT_COORDS(1)\n"
            "H[b]\n"
            "H 1\n"
            "SHIFT_COORDS(1)\n"
            "X\n"
            "M 0 1\n"
            "DETECTOR rec[-1]\n"
        ),
    ),
    (
        "z02_zero_target_repeat_fuse",
        (
            "R 0 1\n"
            "I_ERROR[LEAKAGE_TRANSITION_1: (0.1, U-->2)] 0\n"
            "REPEAT 2 {\n"
            "  H\n"
            "}\n"
            "H 0\n"
            "M 0 1\n"
            "DETECTOR rec[-1]\n"
        ),
    ),
    (
        "z03_zero_target_nonfusing_pairs",
        (
            "R 0\n"
            "TICK\n"
            "SHIFT_COORDS(1)\n"
            "TICK\n"
            "DETECTOR\n"
            "SHIFT_COORDS(1)\n"
            "DETECTOR\n"
            "OBSERVABLE_INCLUDE(0)\n"
            "SHIFT_COORDS(1)\n"
            "OBSERVABLE_INCLUDE(0)\n"
            "QUBIT_COORDS(0)\n"
            "SHIFT_COORDS(1)\n"
            "QUBIT_COORDS(0)\n"
            "E(0.1)\n"
            "SHIFT_COORDS(1)\n"
            "E(0.1)\n"
            "ELSE_CORRELATED_ERROR(0.1)\n"
            "SHIFT_COORDS(1)\n"
            "ELSE_CORRELATED_ERROR(0.1)\n"
            "M 0\n"
        ),
    ),
)

_SUPPLEMENT_MAGIC: tuple[tuple[str, str], ...] = (
    (
        "m01_magic_multibranch_selector",
        (
            "R 0 1 2\n"
            "H 0 2\n"
            "I[CLIFFORD_FAULT: (0.1, H) (0.1, S_DAG) (0.05, X)] 0 1\n"
            "II[CLIFFORD_FAULT: (0.1, CX) (0.05, SWAP) (0.05, CXSWAP)] 0 1\n"
            "I[T] 0\n"
            "I[T_DAG] 1\n"
            "I[CCZ] 0 1 2\n"
            "M 0 1 2\n"
        ),
    ),
    (
        "m02_magic_gauge_steps_and_data_tab",
        (
            "R 0 1 2\n"
            "H 0 1 2\n"
            "SPP X0*Z1\n"
            "SPP_DAG Y1*Z2\n"
            "I[CLIFFORD_FAULT: (0.2, H) (0.1, Z)] 0\n"
            "X_ERROR(0.01) 0\n"
            "Y_ERROR(0.01) 1\n"
            "Z_ERROR(0.01) 2\n"
            "DEPOLARIZE1(0.03) 0\n"
            "PAULI_CHANNEL_1(0.01, 0.02, 0.03) 1\n"
            "DEPOLARIZE2(0.05) 0 1\n"
            "PAULI_CHANNEL_2(0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, "
            "0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015) 1 2\n"
            "E(0.05) X0*Z1\n"
            "ELSE_CORRELATED_ERROR(0.05) Y1*Z2\n"
            "M 0\n"
            "MX 1\n"
            "MY 2\n"
            "H 0 1 2\n"
            "MXX 0 1\n"
            "MPP X0*Y1*Z2\n"
            "H 0 1 2\n"
            "R 0\n"
            "RX 1\n"
            "RY 2\n"
            "H 0 1 2\n"
            "MR(0.02) !0\n"
            "MRX(0.03) 1\n"
            "MRY(0.04) !2\n"
        ),
    ),
    (
        "m03_magic_feedback_all_forms",
        (
            "R 0 1 2 3\n"
            "H 0\n"
            "M 0\n"
            "CX rec[-1] 1 0 2 sweep[0] 3\n"
            "CY rec[-1] 2 sweep[1] 1 2 3\n"
            "CZ rec[-1] 1 2 rec[-1] sweep[0] 3 0 sweep[1] 0 1\n"
            "XCZ 1 rec[-1] 2 sweep[0] 0 3\n"
            "YCZ 1 rec[-1] 2 sweep[1] 1 3\n"
            "M 0 1 2 3\n"
        ),
    ),
    (
        "m04_magic_feedback_invalid_cx_order",
        (
            "R 0 1\n"
            "M 0\n"
            "CX 1 rec[-1]\n"
            "M 1\n"
        ),
    ),
    (
        "m05_magic_nested_repeat_inline_faults",
        (
            "QUBIT_COORDS(0, 0) 0\n"
            "R 0 1 2\n"
            "REPEAT 2 {\n"
            "  H 0\n"
            "  REPEAT 2 {\n"
            "    I[CLIFFORD_FAULT: (0.1, S) (0.05, H)] 0\n"
            "    CX 0 1\n"
            "  }\n"
            "  II[CLIFFORD_FAULT: (0.1, CZ)] 0 1\n"
            "  I[T] 0\n"
            "  HERALDED_ERASE(0.04) 1\n"
            "  HERALDED_PAULI_CHANNEL_1(0.01, 0.01, 0.01, 0.01) 2\n"
            "  TICK\n"
            "  SHIFT_COORDS(0, 1)\n"
            "  M 1\n"
            "  DETECTOR(0, 0) rec[-1]\n"
            "}\n"
            "M 0\n"
            "OBSERVABLE_INCLUDE(0) rec[-1]\n"
        ),
    ),
    (
        "m06_magic_pair_meas_inv_and_mpad",
        (
            "R 0 1 2\n"
            "I 0\n"
            "II 0 1\n"
            "I_ERROR(0.01) 0\n"
            "II_ERROR(0.01) 0 1\n"
            "H 0 1 2\n"
            "MXX(0.02) !0 1 0 !1 !0 !1\n"
            "MYY(0.03) !1 2\n"
            "MZZ(0.04) 0 !2\n"
            "MPP(0.05) !X0*Y1 Z1*X2\n"
            "MPAD(0.05) 0 1\n"
        ),
    ),
    (
        "m07_magic_selector_multiword_66q",
        (
            "R 0 65\n"
            "H 0 65\n"
            "I[CLIFFORD_FAULT: (0.15, H) (0.1, S)] 65\n"
            "II[CLIFFORD_FAULT: (0.1, CX) (0.05, SWAP)] 0 65\n"
            "DEPOLARIZE1(0.03) 65\n"
            "E(0.05) X0*Z65\n"
            "M 0 65\n"
            "H 65\n"
            "R 65\n"
        ),
    ),
)

_SUPPLEMENT_TABLEAU: tuple[tuple[str, str], ...] = (
    (
        "t01_tab_cond_u_kind8",
        (
            "R 0 1 2\n"
            "I[LEAKAGE_TRANSITION_1: (0.2, U-->2)] 0\n"
            "H[CONDITIONED_ON_OTHER: U : 0] 1\n"
            "PAULI_CHANNEL_1[CONDITIONED_ON_SELF: U](0.01, 0.02, 0.03) 0\n"
            "M 0 1 2\n"
        ),
    ),
    (
        "t02_tab_lt1_dup_and_cow_fusion",
        (
            "R 0 1\n"
            "I[LEAKAGE_TRANSITION_1: (0.2, U-->2)] 1 0\n"
            "H 1\n"
            "I[LEAKAGE_TRANSITION_1: (0.2, U-->2)] 0 0\n"
            "I[LEAKAGE_TRANSITION_1: (0.1, 1-->2)] 0 1\n"
            "M[LEAKAGE_PROJECTION_Z: (1.0, 2)] 1 0\n"
            "I[LEAKAGE_TRANSITION_1: (0.1, 1-->2)] 0 1\n"
            "H 0\n"
            "I[LEAKAGE_TRANSITION_1: (0.1, 1-->2)] 0 0\n"
            "M[LEAKAGE_PROJECTION_Z: (1.0, 2)] 0\n"
            "M 0 1\n"
        ),
    ),
    (
        "t03_tab_meas_inv_mz_dup",
        (
            "R 0 1\n"
            "I[LEAKAGE_TRANSITION_1: (0.1, U-->2)] 0 1\n"
            "M[LEAKAGE_PROJECTION_Z: (1.0, 2)] !0 1\n"
            "MZ[LEAKAGE_PROJECTION_Z: (1.0, 2)] 0\n"
            "M[LEAKAGE_PROJECTION_Z: (1.0, 2)] 0 0\n"
        ),
    ),
    (
        "t04_tab_coords_tick_only",
        (
            "QUBIT_COORDS(1, 2) 0\n"
            "SHIFT_COORDS(0, 1)\n"
            "TICK\n"
        ),
    ),
    (
        "t05_tab_lt2_01_only",
        (
            "R 0 1\n"
            "II[LEAKAGE_TRANSITION_2: (0.1, 1_U-->2_U)] 0 1\n"
            "M 0 1\n"
        ),
    ),
)

_GATE_BODY: dict[str, str] = {
    "MPP": "MPP X0*X1",
    "SPP": "SPP X0*Z1",
    "SPP_DAG": "SPP_DAG X0*Z1",
    "E": "E(0.1) X0 Z1",
    "ELSE_CORRELATED_ERROR": "E(0.1) X0\nELSE_CORRELATED_ERROR(0.1) Z1",
    "DETECTOR": "M 0\nDETECTOR rec[-1]",
    "OBSERVABLE_INCLUDE": "M 0\nOBSERVABLE_INCLUDE(0) rec[-1]",
    "QUBIT_COORDS": "QUBIT_COORDS(0, 0) 0",
    "SHIFT_COORDS": "SHIFT_COORDS(1)",
    "TICK": "TICK",
    "MPAD": "MPAD 0",
    "REPEAT": "REPEAT 2 {\nH 0\n}",
}


def _gate_text(name: str) -> str:
    if name in _GATE_BODY:
        return _GATE_BODY[name]
    gd = stim.gate_data(name)
    k = gd.num_parens_arguments_range.start
    args = "(" + ", ".join(["0.01"] * k) + ")" if k else ""
    return f"{name}{args} 0 1" if gd.is_two_qubit_gate else f"{name}{args} 0"


def _gate_circuit(name: str) -> stim.Circuit:
    pre = "H 2" if name == "S" else "S 2"
    return stim.Circuit(f"R 0 1 2\n{pre}\n" + _gate_text(name))


def build_corpus(
    extra_circuits: Sequence[tuple[str, stim.Circuit]] = (),
) -> list[tuple[str, stim.Circuit]]:
    """Build the self-contained 152-circuit corpus (no external file access)."""
    out: list[tuple[str, stim.Circuit]] = [
        (name, stim.Circuit(text)) for name, _tags, _leak, text in golden_test.CORPUS
    ]
    for group in (
        _SUPPLEMENT_S,
        _SUPPLEMENT_X,
        _SUPPLEMENT_Y_REV2,
        _SUPPLEMENT_Y_REV3,
        _SUPPLEMENT_WITNESSES,
        _SUPPLEMENT_Z,
        _SUPPLEMENT_MAGIC,
        _SUPPLEMENT_TABLEAU,
    ):
        for name, text in group:
            out.append((name, stim.Circuit(text)))
    for gname in sorted(stim.gate_data()):
        if gname != "REPEAT":
            out.append((f"g_{gname}", _gate_circuit(gname)))
    out.extend(extra_circuits)
    return out


def canon(x: Any) -> bytes:
    """Canonical deterministic bytes for nested Python/NumPy/ctypes/dataclass values."""
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return repr(x.item()).encode()
    if x is None or isinstance(x, (int, float, str)):
        return repr(x).encode()
    if isinstance(x, np.ndarray):
        return f"nd{x.dtype}{x.shape}".encode() + np.ascontiguousarray(x).tobytes()
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(
        x, (stim.CircuitInstruction, stim.Circuit, stim.DetectorErrorModel)
    ):
        return str(x).encode()
    if isinstance(x, (ctypes.Structure, ctypes.Array)):
        return type(x).__name__.encode() + bytes(x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        fields_bytes = b",".join(
            f.name.encode() + b"=" + canon(getattr(x, f.name))
            for f in dataclasses.fields(x)
            if not f.name.startswith("_")
        )
        return type(x).__name__.encode() + b"{" + fields_bytes + b"}"
    if isinstance(x, dict):
        items = sorted((canon(k), canon(v)) for k, v in x.items())
        return b"d{" + b",".join(k + b":" + v for k, v in items) + b"}"
    if isinstance(x, (set, frozenset)):
        return b"s{" + b",".join(sorted(canon(v) for v in x)) + b"}"
    if isinstance(x, (list, tuple)):
        return b"[" + b",".join(canon(v) for v in x) + b"]"
    r = repr(x)
    if " at 0x" in r or _HEX_ADDR_RE.search(r):
        raise TypeError(
            f"canon(): unsupported type {type(x).__name__} (repr has an address)"
        )
    return (type(x).__name__ + r).encode()


# ---- 24 walkers (w1..w12c) -------------------------------------------------


def w1_tableau_precompile(c: stim.Circuit) -> list[Any]:
    p = PrecompiledTablesideCircuit(c.copy(), True)
    return [
        list(p._c_steps) if p._c_steps is not None else None,
        p._raw_t_np,
        p._qubit_t_np,
        p._masks_np,
        list(p._c_branches) if p._c_branches is not None else None,
        p.step_to_unrolled_idx,
        p.step_to_full_idx,
        p.step_to_quant_idx,
        p.collapses_z,
        p.requires_tableau,
        p.unrolled_num_ops,
        p.step_has_bare_op,
        [str(x) for x in p.ref_insts],
        [str(x) for x in p.full_bare_insts],
        [str(x) for x in p.orig_ops],
        [(s[0], s[1], s[3], s[4], s[5]) for s in p.steps_py],
        p.final_qubit_coords,
        p.final_qubit_tags,
        p.final_coords_shifts,
    ]


def w1b_tableau_precompile_ucu_false(c: stim.Circuit) -> list[Any]:
    p = PrecompiledTablesideCircuit(c.copy(), False)
    return [
        list(p._c_steps) if p._c_steps is not None else None,
        p.step_to_unrolled_idx,
        p.collapses_z,
        [str(x) for x in p.ref_insts],
    ]


def _pack_tableau_precompile(p: PrecompiledTablesideCircuit) -> list[Any]:
    return [
        p.num_qubits,
        p.num_words,
        p.num_steps,
        p.unconditional_condition_on_U,
        p.requires_tableau,
        p.collapses_z,
        p._c_steps is not None,
        p._c_raw_t is not None,
        p._c_qubit_t is not None,
        p._c_masks is not None,
        p._c_branches is not None,
        [str(x) for x in p.quantum_bare_insts],
        [(s[0], s[1], s[2], s[3], s[4], s[5]) for s in p.steps_py],
        sorted((str(k), repr(v)) for k, v in p.parsed_ops.items()),
        list(p._c_steps) if p._c_steps is not None else None,
        p._raw_t_np,
        p._qubit_t_np,
        p._masks_np,
        list(p._c_branches) if p._c_branches is not None else None,
    ]


def w1c_tableau_precompile_full(c: stim.Circuit) -> list[list[Any]]:
    p_auto = PrecompiledTablesideCircuit(c.copy(), True)
    p_sup = PrecompiledTablesideCircuit(
        c.copy(), True, parsed_ops=parse_leakage_in_circuit(c)
    )
    return [_pack_tableau_precompile(p_auto), _pack_tableau_precompile(p_sup)]


def w2_coset_reference(c: stim.Circuit) -> list[list[Any]]:
    r = CompiledReferenceCircuit(c)
    rows: list[list[Any]] = []
    for st in r.steps:
        st_any: Any = st
        pw = st_any._get_packed_words(r.n_words)
        row: list[Any] = [st_any.kind, str(st_any.op), pw]
        if st_any.kind == "unitary":
            row += [
                st_any.gate_name,
                st_any.target_groups,
                st_any.are_groups_disjoint,
                str(st_any.quantum_op),
                st_any._group_packed_words,
            ]
        elif st_any.kind == "meas":
            row += [
                st_any.gate_name,
                st_any.produces_measurements,
                st_any.is_reset,
                [
                    (
                        s.q,
                        s.invert_result,
                        [str(o) for o in s.pre_ops],
                        [str(o) for o in s.post_ops],
                        s.meta,
                    )
                    for s in st_any.substeps
                ],
            ]
        rows.append(row)
    return rows


def w3_magic_pass(c: stim.Circuit) -> dict[str, Any]:
    r = _compile_circuit_pass(c)
    d: dict[str, Any] = {f.name: getattr(r, f.name) for f in dataclasses.fields(r)}
    d["final_tableau"] = r.final_tableau.snapshot_packed()
    d["data_final_tableau"] = (
        None if r.data_final_tableau is None else r.data_final_tableau.snapshot_packed()
    )
    return d


def w3b_extract_faults(c: stim.Circuit) -> Any:
    return extract_circuit_faults(c)


def w3c_magic_pass_external_fault_slots(c: stim.Circuit) -> dict[str, Any]:
    flat_ops, _inline = extract_circuit_faults(c)
    n = len(flat_ops)
    chans = [
        CliffordFaultChannel(
            step_index=s,
            branches=(FaultBranch(0.25, "X", (0,)),),
        )
        for s in sorted({0, n // 2, n})
    ]
    r = _compile_circuit_pass(c, chans)
    d: dict[str, Any] = {f.name: getattr(r, f.name) for f in dataclasses.fields(r)}
    d["final_tableau"] = r.final_tableau.snapshot_packed()
    d["data_final_tableau"] = (
        None if r.data_final_tableau is None else r.data_final_tableau.snapshot_packed()
    )
    return d


def _pack_magic_pass(r: Any) -> dict[str, Any]:
    d: dict[str, Any] = {f.name: getattr(r, f.name) for f in dataclasses.fields(r)}
    d["final_tableau"] = r.final_tableau.snapshot_packed()
    d["data_final_tableau"] = (
        None if r.data_final_tableau is None else r.data_final_tableau.snapshot_packed()
    )
    return d


def w3d_magic_pass_selector(c: stim.Circuit) -> list[dict[str, Any]]:
    return [
        _pack_magic_pass(
            _compile_circuit_pass(c, selector_mode=True, gadgetize_pauli_noise=False)
        ),
        _pack_magic_pass(
            _compile_circuit_pass(c, selector_mode=True, gadgetize_pauli_noise=True)
        ),
    ]


def w4_marginal_builder(c: stim.Circuit) -> dict[str, Any]:
    a = mb.Builder(c, True).build()
    return {
        f.name: getattr(a, f.name)
        for f in dataclasses.fields(a)
        if not f.name.startswith("_")
    }


def w4b_marginal_builder_oracle(c: stim.Circuit) -> dict[str, Any]:
    a = mb.Builder(c, True).build(oracle=True)
    return {
        f.name: getattr(a, f.name)
        for f in dataclasses.fields(a)
        if not f.name.startswith("_")
    }


def w4c_marginal_builder_abi(c: stim.Circuit) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    real = mb.trace_flows_cpp

    def _spy(**kw: Any) -> Any:
        captured.append(dict(kw))
        return real(**kw)

    with mock.patch.object(mb, "trace_flows_cpp", side_effect=_spy):
        mb.Builder(c, True).build(oracle=True)
    return captured


def w4d_marginal_builder_uc_false(c: stim.Circuit) -> dict[str, Any]:
    a = mb.Builder(c, False).build(oracle=True)
    return {
        f.name: getattr(a, f.name)
        for f in dataclasses.fields(a)
        if not f.name.startswith("_")
    }


def w5_sync_reference(c: stim.Circuit) -> Any:
    h = LeakageUint8Flip().compile_op_handler(circuit=c, batch_size=4)
    return compute_sync_reference(c.num_qubits, unroll_circuit(c), h)


def w5b_sync_reference_nohandler(c: stim.Circuit) -> Any:
    return compute_sync_reference(c.num_qubits, unroll_circuit(c), None)


def w6_known_states(c: stim.Circuit) -> Any:
    return compute_known_states_uint8(c)


def w7_strip(c: stim.Circuit) -> list[Any]:
    out, flag = _strip_circuit_for_noiseless_ref(c)
    return [str(out), flag]


def w8_parse_magic(c: stim.Circuit) -> list[Any]:
    a, b, m = _parse_leakage_in_circuit_magic(c, c.num_qubits)
    return [
        sorted((str(k), repr(v)) for k, v in a.items()),
        sorted(str(x) for x in b),
        sorted((str(k), v) for k, v in m.items()),
    ]


def w8b_parse_tableau(c: stim.Circuit) -> list[tuple[str, str]]:
    return sorted((str(k), repr(v)) for k, v in parse_leakage_in_circuit(c).items())


def w9_collapses_z(c: stim.Circuit) -> bool:
    return collapses_z(unroll_circuit(c), parse_leakage_in_circuit(c))


def w10_unroll(c: stim.Circuit) -> list[str]:
    return [str(op) for op in unroll_circuit(c)]


def w11_u2f_and_swap(c: stim.Circuit) -> list[Any]:
    return [unrolled_to_flattened_indices(c), has_leakage_swap(c)]


def _copy_len_fixture(op: stim.CircuitInstruction) -> int:
    if op.name == "MPAD" and op.tag.startswith("LEAKAGE_MEASUREMENT"):
        return len(op.tag.rsplit(":", 1)[1].split())
    return 0


def w12_ir_fields(c: stim.Circuit) -> list[Any]:
    from stimside.util import circuit_ir

    out: list[Any] = []
    for uc in (True, False):
        ir = circuit_ir.build_ir(c, uc=uc)
        out.append(
            [
                [
                    i.unrolled_index,
                    str(i.op),
                    i.tag,
                    i.category.value,
                    i.untagged_filter.value,
                    i.flat_index,
                    i.flat_offset,
                    i.num_measurements,
                    i.first_measurement,
                ]
                for i in ir.instrs
            ]
        )
        out.append([str(o) for o in ir.flat_ops])
        out.append(
            [ir.num_qubits, ir.num_measurements, ir.has_repeat, ir.uc, ir.from_op_list]
        )
    il = circuit_ir.build_ir([i.op for i in circuit_ir.build_ir(c).instrs])
    out.append(
        [
            il.num_qubits,
            il.num_measurements,
            il.has_repeat,
            il.uc,
            il.from_op_list,
            [i.flat_index for i in il.instrs],
        ]
    )
    ir = circuit_ir.build_ir(c, uc=True)
    for skip in (_SK_COSET, _SK_TAB):
        sm = circuit_ir.step_map(ir, skip)
        out.append([list(sm.step_to_unrolled), list(sm.unrolled_to_step)])
    cm = circuit_ir.split_copies(ir, _copy_len_fixture)
    out.append([[str(o) for o in cm.ops], list(cm.flat_to_op), list(cm.copy_len)])
    return out


def w12b_tableau_layout(c: stim.Circuit) -> list[list[int]]:
    from stimside.util import circuit_ir

    p = PrecompiledTablesideCircuit(c.copy(), True)
    ir = circuit_ir.build_ir(c, uc=True)
    lay = circuit_ir.tableau_step_layout(
        ir, circuit_ir.step_map(ir, _SK_TAB), p.step_has_bare_op
    )
    return [
        list(lay.step_to_unrolled),
        list(lay.step_to_quant),
        list(lay.step_to_full),
        list(p.step_to_unrolled_idx),
        list(p.step_to_quant_idx),
        list(p.step_to_full_idx),
    ]


def w12c_ir_shared_fields(c: stim.Circuit) -> list[Any]:
    from stimside.util import circuit_ir

    ir = circuit_ir.build_ir(c, uc=False)
    rows_circ = [
        [i.basis, i.is_product_measurement, i.arity, i.has_classical_control]
        for i in ir.instrs
    ]
    il = circuit_ir.build_ir([i.op for i in ir.instrs], uc=False)
    rows_list = [
        [i.basis, i.is_product_measurement, i.arity, i.has_classical_control]
        for i in il.instrs
    ]
    return [rows_circ, rows_list]


WALKERS: dict[str, Callable[[stim.Circuit], Any]] = {
    "w1_tableau_precompile": w1_tableau_precompile,
    "w1b_tableau_precompile_ucu_false": w1b_tableau_precompile_ucu_false,
    "w1c_tableau_precompile_full": w1c_tableau_precompile_full,
    "w2_coset_reference": w2_coset_reference,
    "w3_magic_pass": w3_magic_pass,
    "w3b_extract_faults": w3b_extract_faults,
    "w3c_magic_pass_external_fault_slots": w3c_magic_pass_external_fault_slots,
    "w3d_magic_pass_selector": w3d_magic_pass_selector,
    "w4_marginal_builder": w4_marginal_builder,
    "w4b_marginal_builder_oracle": w4b_marginal_builder_oracle,
    "w4c_marginal_builder_abi": w4c_marginal_builder_abi,
    "w4d_marginal_builder_uc_false": w4d_marginal_builder_uc_false,
    "w5_sync_reference": w5_sync_reference,
    "w5b_sync_reference_nohandler": w5b_sync_reference_nohandler,
    "w6_known_states": w6_known_states,
    "w7_strip": w7_strip,
    "w8_parse_magic": w8_parse_magic,
    "w8b_parse_tableau": w8b_parse_tableau,
    "w9_collapses_z": w9_collapses_z,
    "w10_unroll": w10_unroll,
    "w11_u2f_and_swap": w11_u2f_and_swap,
    "w12_ir_fields": w12_ir_fields,
    "w12b_tableau_layout": w12b_tableau_layout,
    "w12c_ir_shared_fields": w12c_ir_shared_fields,
}


def run_walker_entry(
    fn: Callable[[stim.Circuit], Any], circ: stim.Circuit
) -> tuple[str, int, str | None]:
    """Run one walker on ``circ`` and return ``(sha24, nbytes, raises_msg)``.

    ``canon(val)`` executes OUTSIDE the contract-exception handler so a
    memory-address error from ``canon()`` always propagates loudly instead of
    being hashed as a contract ``RAISES:TypeError`` entry.
    """
    try:
        val = fn(circ.copy())
    except _CONTRACT_EXC as exc:
        msg = f"RAISES:{type(exc).__name__}:{str(exc)[:120]}"
        return hashlib.sha256(msg.encode()).hexdigest()[:24], -1, msg
    raw = canon(val)
    return hashlib.sha256(raw).hexdigest()[:24], len(raw), None


def compute_oracle_digests(
    circuits: Sequence[tuple[str, stim.Circuit]] | None = None,
    walkers: dict[str, Callable[[stim.Circuit], Any]] | None = None,
) -> dict[str, dict[str, tuple[str, int, str | None]]]:
    """Compute canonical digests for all ``(circuit, walker)`` pairs."""
    use_circuits = build_corpus() if circuits is None else list(circuits)
    use_walkers = WALKERS if walkers is None else walkers
    out: dict[str, dict[str, tuple[str, int, str | None]]] = {}
    for cname, circ in use_circuits:
        out[cname] = {
            wname: run_walker_entry(fn, circ) for wname, fn in use_walkers.items()
        }
    return out


# ---- Frozen structural oracle digests (generated once from CP-B + C1) --------
FROZEN_DIGESTS: dict[str, dict[str, tuple[str, int]]] = {
    "c01_clifford_noise_feedback": {
        "w10_unroll": ("915248b6f11eec8d845d4843", 659),
        "w11_u2f_and_swap": ("59b022ea6ead9dd2dfd1791c", 260),
        "w12_ir_fields": ("361c8eee6be395307c5df17a", 5869),
        "w12b_tableau_layout": ("3388ff92b86b67354defb965", 369),
        "w12c_ir_shared_fields": ("96baac25d2ea296b7f499365", 1389),
        "w1_tableau_precompile": ("c4cc7c2975f71b6e1e4ca0f9", 4630),
        "w1b_tableau_precompile_ucu_false": ("c168f1dd2d6db30e3557d940", 2005),
        "w1c_tableau_precompile_full": ("0b0f43b42086a66a0d17448e", 6363),
        "w2_coset_reference": ("86c69cdcaffc9c3ede10154b", 10121),
        "w3_magic_pass": ("8c51abfd87b6f51108d8f363", 13221),
        "w3b_extract_faults": ("a3b00f9f61c04e708ddae25b", 604),
        "w3c_magic_pass_external_fault_slots": ("625e2fc67d8d5cefa92f4109", 14170),
        "w3d_magic_pass_selector": ("2f2cbacaac99f8d4a7dc63ab", 41950),
        "w4_marginal_builder": ("34a74d17e610e927ee9f08cb", -1),
        "w4b_marginal_builder_oracle": ("34a74d17e610e927ee9f08cb", -1),
        "w4c_marginal_builder_abi": ("34a74d17e610e927ee9f08cb", -1),
        "w4d_marginal_builder_uc_false": ("34a74d17e610e927ee9f08cb", -1),
        "w5_sync_reference": ("8272aad10727c3fcd7119a02", 26),
        "w5b_sync_reference_nohandler": ("8272aad10727c3fcd7119a02", 26),
        "w6_known_states": ("a8f4a7fe98d274eb650525da", 169),
        "w7_strip": ("72df8745cfe71311a5d1c9a4", 295),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "c02_mpp_mxx_heralded_corr": {
        "w10_unroll": ("71e43302a5ac92a8ce52584a", 356),
        "w11_u2f_and_swap": ("d80fb4daa52f568876007456", 140),
        "w12_ir_fields": ("c7c7f81b9f85dc7dd045f01c", 3129),
        "w12b_tableau_layout": ("70ca30a506cf125c48a197d5", 199),
        "w12c_ir_shared_fields": ("dd6660df97236793d122c67f", 695),
        "w1_tableau_precompile": ("55d4782f0e259d8cc2cd4a3e", 2625),
        "w1b_tableau_precompile_ucu_false": ("3cadfe2ec7b442be3aa1fa56", 1151),
        "w1c_tableau_precompile_full": ("b7719faaee2f1aa5d6ab406c", 3645),
        "w2_coset_reference": ("9c8eaae12055ab6417e040b9", 6917),
        "w3_magic_pass": ("c5f1af692218489918155ecc", 6620),
        "w3b_extract_faults": ("c200aabfe7bb8edf3fbb72f6", 331),
        "w3c_magic_pass_external_fault_slots": ("c67ed4eced27ce30c2c75a22", 7568),
        "w3d_magic_pass_selector": ("d3af92328c9b70da2512581a", 16520),
        "w4_marginal_builder": ("0d6711ac3168edfa22f4d85f", -1),
        "w4b_marginal_builder_oracle": ("0d6711ac3168edfa22f4d85f", -1),
        "w4c_marginal_builder_abi": ("0d6711ac3168edfa22f4d85f", -1),
        "w4d_marginal_builder_uc_false": ("0d6711ac3168edfa22f4d85f", -1),
        "w5_sync_reference": ("3f8d0b3d5d722229345edd08", 35),
        "w5b_sync_reference_nohandler": ("3f8d0b3d5d722229345edd08", 35),
        "w6_known_states": ("704a3f5c7223c60bce020e28", 78),
        "w7_strip": ("5e787d4f71cb7b5f76fa1018", 194),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "c02b_spp_clifford": {
        "w10_unroll": ("2fcaa5c54f8446d90c2ee5fe", 166),
        "w11_u2f_and_swap": ("161335d276595c75dd16615c", 91),
        "w12_ir_fields": ("47d94bbd9d718fda34ef070c", 1668),
        "w12b_tableau_layout": ("cbd6478f3333fbc249ec27cb", 105),
        "w12c_ir_shared_fields": ("e48f495aa532f34f0867f461", 421),
        "w1_tableau_precompile": ("815b663da8f23e4e6e3c707c", 1426),
        "w1b_tableau_precompile_ucu_false": ("2b98ae4c15826f86b92b7f1e", 598),
        "w1c_tableau_precompile_full": ("59d3a6800b92e677a0d29f8b", 2113),
        "w2_coset_reference": ("4ee56c0992997d6fec7e3b1d", -1),
        "w3_magic_pass": ("91117ad92b7333d1c04706ce", 3717),
        "w3b_extract_faults": ("e5f4125bea6e736549695bda", 153),
        "w3c_magic_pass_external_fault_slots": ("b5b70fd6cc61cae7cd9d2b28", 4664),
        "w3d_magic_pass_selector": ("c25115538b562adc5769f175", 9688),
        "w4_marginal_builder": ("df7bc772e9482d1c9a4c3088", -1),
        "w4b_marginal_builder_oracle": ("df7bc772e9482d1c9a4c3088", -1),
        "w4c_marginal_builder_abi": ("df7bc772e9482d1c9a4c3088", -1),
        "w4d_marginal_builder_uc_false": ("34a74d17e610e927ee9f08cb", -1),
        "w5_sync_reference": ("a8158a5a93d9bcf5dc2f66b8", 23),
        "w5b_sync_reference_nohandler": ("a8158a5a93d9bcf5dc2f66b8", 23),
        "w6_known_states": ("6ee345e9d624e4bbcfd18e9d", 44),
        "w7_strip": ("0f73a059d6a9fde3afa26c07", 139),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "c03_leakage_transitions_swap_meas": {
        "w10_unroll": ("682e84017de3fd58af91c74d", 982),
        "w11_u2f_and_swap": ("5cc9155636e167c166ac8a7a", 187),
        "w12_ir_fields": ("45546e70a9442a83bff096b9", 8118),
        "w12b_tableau_layout": ("cd08abc5a56648f7c9f62e76", 251),
        "w12c_ir_shared_fields": ("a2237657b76d492ef35bec9f", 943),
        "w1_tableau_precompile": ("bd2cc9292c16738078bb9c5d", 8457),
        "w1b_tableau_precompile_ucu_false": ("25821dbb9a855890043e6d6e", 1389),
        "w1c_tableau_precompile_full": ("4d0d9bf538d6962fc8a0f83d", 16973),
        "w2_coset_reference": ("8d316fdca8fce36ea0231146", 8362),
        "w3_magic_pass": ("27677c6b6baa1e33fe9bf0df", 3807),
        "w3b_extract_faults": ("2e44c168844873e921b01600", 945),
        "w3c_magic_pass_external_fault_slots": ("41dc0fb724c523e9c871cf81", 4756),
        "w3d_magic_pass_selector": ("f0fb6284648169158980aebe", 8225),
        "w4_marginal_builder": ("96c4de5fab71823dbd6f6692", -1),
        "w4b_marginal_builder_oracle": ("0d6711ac3168edfa22f4d85f", -1),
        "w4c_marginal_builder_abi": ("0d6711ac3168edfa22f4d85f", -1),
        "w4d_marginal_builder_uc_false": ("0d6711ac3168edfa22f4d85f", -1),
        "w5_sync_reference": ("5a77f94fc22fd25e33e358b3", 37),
        "w5b_sync_reference_nohandler": ("5a77f94fc22fd25e33e358b3", 37),
        "w6_known_states": ("6ee47b31f26f83935a902de1", 102),
        "w7_strip": ("1e9172f76a8495ead68ad7de", 210),
        "w8_parse_magic": ("53563fa7241495450bf76050", 1512),
        "w8b_parse_tableau": ("4ded64452dea616efd0bef3b", 1504),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "c04_leakage_conditioning_and_repeated_targets": {
        "w10_unroll": ("337102fb6916cfd1f7c980f6", 852),
        "w11_u2f_and_swap": ("953b1b91e13e0e49e162c296", 180),
        "w12_ir_fields": ("7f90554dfd1a70c8d754f403", 7123),
        "w12b_tableau_layout": ("1fa1d8ed5d5583547e76b7c4", 263),
        "w12c_ir_shared_fields": ("cc62a4a1a0e5814bc3e25c4f", 901),
        "w1_tableau_precompile": ("ae1d4fdd07ef5d29291eb380", 6554),
        "w1b_tableau_precompile_ucu_false": ("66dc7b5b5938d35f219cfd55", 1374),
        "w1c_tableau_precompile_full": ("a4e9c74ccda0ff5b9ab97063", 15315),
        "w2_coset_reference": ("dde03d18b3676baa29f73c9e", 7213),
        "w3_magic_pass": ("8527ae031209b11d40e26f68", 3771),
        "w3b_extract_faults": ("4c5412aa7b8d5dc7eb48efd3", 817),
        "w3c_magic_pass_external_fault_slots": ("3828313899b2f7d0447a53aa", 4720),
        "w3d_magic_pass_selector": ("52fcd68d4ed99a4a8d11a52c", 8241),
        "w4_marginal_builder": ("c2bce55950daa71f152a2c47", -1),
        "w4b_marginal_builder_oracle": ("c2bce55950daa71f152a2c47", -1),
        "w4c_marginal_builder_abi": ("c2bce55950daa71f152a2c47", -1),
        "w4d_marginal_builder_uc_false": ("c2bce55950daa71f152a2c47", -1),
        "w5_sync_reference": ("1c0c71bdf88e4b4a36609578", -1),
        "w5b_sync_reference_nohandler": ("8272aad10727c3fcd7119a02", 26),
        "w6_known_states": ("130e74ca04ec14828ee7aaa7", 98),
        "w7_strip": ("42df89e8813d970019ab2f7d", 248),
        "w8_parse_magic": ("8561aa43f73a9e0638dd8328", 2432),
        "w8b_parse_tableau": ("fe9d40c50b9610e138edd067", 2424),
        "w9_collapses_z": ("3cbc87c7681f34db4617feaa", 4),
    },
    "c04b_leakage_detector_tab_coset": {
        "w10_unroll": ("b4939e76f1c2037e425cbeed", 188),
        "w11_u2f_and_swap": ("54d80463c222de2ad3c941c3", 67),
        "w12_ir_fields": ("5d6c52675e9f6b8c487ecf2e", 1689),
        "w12b_tableau_layout": ("04c41f92e55b53b31d0e8150", 69),
        "w12c_ir_shared_fields": ("72bfa31f7c5952f156d92efc", 283),
        "w1_tableau_precompile": ("a9fa421f491fcc54d622b76e", 1390),
        "w1b_tableau_precompile_ucu_false": ("b3fe3acb036a130109b772dd", 396),
        "w1c_tableau_precompile_full": ("b524cdf07c4f55de66bdb073", 2675),
        "w2_coset_reference": ("9314dbc5349ce6be974b04ea", 1719),
        "w3_magic_pass": ("7aad614b2ae2185163fbada5", 1794),
        "w3b_extract_faults": ("a1fce60676605636e543bf81", 181),
        "w3c_magic_pass_external_fault_slots": ("f9ac5757b9748be4a9056324", 2741),
        "w3d_magic_pass_selector": ("0b168e51b44c434d26dd1e2e", 3889),
        "w4_marginal_builder": ("f9e1417d7db2bbf9f1abdb90", 828),
        "w4b_marginal_builder_oracle": ("7ab67e05692eb32a56f3f4d3", 901),
        "w4c_marginal_builder_abi": ("58b6be2df88d1366a3dd9284", 939),
        "w4d_marginal_builder_uc_false": ("7ab67e05692eb32a56f3f4d3", 901),
        "w5_sync_reference": ("8d979219d0b268de1a3146fb", -1),
        "w5b_sync_reference_nohandler": ("c62c18fd5a3df2a67e7d6165", 24),
        "w6_known_states": ("d59e40883864288028fe5777", 27),
        "w7_strip": ("e7f4d31933516921f871db28", 110),
        "w8_parse_magic": ("a58141cd5761f0ea2c038f68", 309),
        "w8b_parse_tableau": ("cd81b6f5699443a2f7826b72", 301),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "c05_flip_only_tags": {
        "w10_unroll": ("6461343721e2de0ab08be386", 669),
        "w11_u2f_and_swap": ("8ade461928752d128c9d9107", 148),
        "w12_ir_fields": ("5285c41a36ea47a9d5af6f12", 5662),
        "w12b_tableau_layout": ("22bc6ce67e52e6f3d9e78a65", -1),
        "w12c_ir_shared_fields": ("fa63ff6376f1b893ef7f2c8c", 727),
        "w1_tableau_precompile": ("22bc6ce67e52e6f3d9e78a65", -1),
        "w1b_tableau_precompile_ucu_false": ("22bc6ce67e52e6f3d9e78a65", -1),
        "w1c_tableau_precompile_full": ("22bc6ce67e52e6f3d9e78a65", -1),
        "w2_coset_reference": ("b839da3ab7e84dfe628175eb", 6237),
        "w3_magic_pass": ("0db67d3abaf0549894cb5ab9", 2359),
        "w3b_extract_faults": ("baacb3edb63024128d2becdc", 642),
        "w3c_magic_pass_external_fault_slots": ("6f0fb51f6356a57fd421b3cd", 3307),
        "w3d_magic_pass_selector": ("0e63f24b7151bae6b28c7f11", 5261),
        "w4_marginal_builder": ("22bc6ce67e52e6f3d9e78a65", -1),
        "w4b_marginal_builder_oracle": ("22bc6ce67e52e6f3d9e78a65", -1),
        "w4c_marginal_builder_abi": ("22bc6ce67e52e6f3d9e78a65", -1),
        "w4d_marginal_builder_uc_false": ("22bc6ce67e52e6f3d9e78a65", -1),
        "w5_sync_reference": ("4b515a9af4270d81d8e7ca23", 31),
        "w5b_sync_reference_nohandler": ("4b515a9af4270d81d8e7ca23", 31),
        "w6_known_states": ("bf106d224af7b38353f8929f", 65),
        "w7_strip": ("fd9d010db0192df52e4c4998", 134),
        "w8_parse_magic": ("22bc6ce67e52e6f3d9e78a65", -1),
        "w8b_parse_tableau": ("22bc6ce67e52e6f3d9e78a65", -1),
        "w9_collapses_z": ("22bc6ce67e52e6f3d9e78a65", -1),
    },
    "c06_magic_t_ccz_clifford": {
        "w10_unroll": ("e28fcae4d19845884950d0bd", 395),
        "w11_u2f_and_swap": ("8e814bb05bcbf5127a7b3645", 220),
        "w12_ir_fields": ("4c6e171f0b7d32317d7ea5b4", 4319),
        "w12b_tableau_layout": ("2808944805010f2692e41650", 343),
        "w12c_ir_shared_fields": ("89bafa359fdd4972fa917608", 1119),
        "w1_tableau_precompile": ("322c8926ffe80573049f891c", 3682),
        "w1b_tableau_precompile_ucu_false": ("d434b503f4565ce3a62f71a5", 1678),
        "w1c_tableau_precompile_full": ("59cbbbdde5207cee3583e128", 5423),
        "w2_coset_reference": ("4ee56c0992997d6fec7e3b1d", -1),
        "w3_magic_pass": ("afafc9309477d2f21e17288b", 9462),
        "w3b_extract_faults": ("854ead33237a2a1b67400130", 350),
        "w3c_magic_pass_external_fault_slots": ("3c748ea9621ee586e3e00500", 10411),
        "w3d_magic_pass_selector": ("ceda84e571947c175863d5a6", 23761),
        "w4_marginal_builder": ("df7bc772e9482d1c9a4c3088", -1),
        "w4b_marginal_builder_oracle": ("df7bc772e9482d1c9a4c3088", -1),
        "w4c_marginal_builder_abi": ("df7bc772e9482d1c9a4c3088", -1),
        "w4d_marginal_builder_uc_false": ("75379da034fcf5ec7fbc96ca", -1),
        "w5_sync_reference": ("7bbfcf949690955cc005a79e", 25),
        "w5b_sync_reference_nohandler": ("7bbfcf949690955cc005a79e", 25),
        "w6_known_states": ("952711bd066792c5ba9f8f8a", 118),
        "w7_strip": ("b385091adc97cbacb3d12a5c", 257),
        "w8_parse_magic": ("512e30f5127db727699c5d67", 76),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "c07_magic_inline_faults": {
        "w10_unroll": ("d534dc6a5246d311e9a3551f", 287),
        "w11_u2f_and_swap": ("773f43b023a63432c0332e68", 124),
        "w12_ir_fields": ("ed86279e9834944260a8b852", 2747),
        "w12b_tableau_layout": ("73df1e89d8fc8165b4dc768d", 163),
        "w12c_ir_shared_fields": ("b3dfec7940e00d935a11ff1d", 589),
        "w1_tableau_precompile": ("8b0c0f9891ca1a292a7fb3b2", 2234),
        "w1b_tableau_precompile_ucu_false": ("a55a733f1d9ad768aafc476b", 960),
        "w1c_tableau_precompile_full": ("bc1d10c3f297a8c2433a48ac", 3177),
        "w2_coset_reference": ("347057a959dfa711ed345e87", 3848),
        "w3_magic_pass": ("fe6e6f5ebd928e473f6670d7", 4937),
        "w3b_extract_faults": ("89c6678f0423df996ba2a0ef", 462),
        "w3c_magic_pass_external_fault_slots": ("89413caef4311bbcc46dbecf", 5886),
        "w3d_magic_pass_selector": ("24995aaf56c15244f23877ca", 11975),
        "w4_marginal_builder": ("0cd2d6289f1c259c4f0a672a", 924),
        "w4b_marginal_builder_oracle": ("ab641492fcb39e1d3dc2edc1", 1035),
        "w4c_marginal_builder_abi": ("9a3acb67dcfa2766319593d3", 1315),
        "w4d_marginal_builder_uc_false": ("ab641492fcb39e1d3dc2edc1", 1035),
        "w5_sync_reference": ("ee7d2c70eadb9c41d829f1a6", 25),
        "w5b_sync_reference_nohandler": ("ee7d2c70eadb9c41d829f1a6", 25),
        "w6_known_states": ("f611eb6d08e440cd8c61be44", 56),
        "w7_strip": ("dddeb1c57ddaaebd41622615", 143),
        "w8_parse_magic": ("53c7bf54fe3dfdf37e983684", 53),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "c08_magic_with_leakage": {
        "w10_unroll": ("ebd6113911400d0c93c6ab13", 413),
        "w11_u2f_and_swap": ("ffbdb03a8f0d26259d8e2827", 123),
        "w12_ir_fields": ("648033efc4e6ac76743f86cd", 3729),
        "w12b_tableau_layout": ("69e361486d7b6037eeb9bc42", -1),
        "w12c_ir_shared_fields": ("9dbdec7c4142071108150337", 577),
        "w1_tableau_precompile": ("69e361486d7b6037eeb9bc42", -1),
        "w1b_tableau_precompile_ucu_false": ("69e361486d7b6037eeb9bc42", -1),
        "w1c_tableau_precompile_full": ("69e361486d7b6037eeb9bc42", -1),
        "w2_coset_reference": ("40089122fb6b691bf0910cae", 4504),
        "w3_magic_pass": ("61ff314c16437e3a947b5522", 3372),
        "w3b_extract_faults": ("3b95db6db736fd7cac38a61b", 392),
        "w3c_magic_pass_external_fault_slots": ("fb81815629290a1382c43f9f", 4320),
        "w3d_magic_pass_selector": ("209abd8c80ee7000a27d4362", 7375),
        "w4_marginal_builder": ("c2bce55950daa71f152a2c47", -1),
        "w4b_marginal_builder_oracle": ("c2bce55950daa71f152a2c47", -1),
        "w4c_marginal_builder_abi": ("c2bce55950daa71f152a2c47", -1),
        "w4d_marginal_builder_uc_false": ("c2bce55950daa71f152a2c47", -1),
        "w5_sync_reference": ("254541de4284cb0417be5501", -1),
        "w5b_sync_reference_nohandler": ("bc39bdc8d70acf45561af869", 27),
        "w6_known_states": ("6fa2f769bb31de7491488138", 56),
        "w7_strip": ("1c9e03002b2e92c1fca26894", 185),
        "w8_parse_magic": ("85911638e4c9e68ffd8884fe", 1080),
        "w8b_parse_tableau": ("69e361486d7b6037eeb9bc42", -1),
        "w9_collapses_z": ("69e361486d7b6037eeb9bc42", -1),
    },
    "c09_multiword_70q_leakage": {
        "w10_unroll": ("cd5f2318170de333109e3693", 394),
        "w11_u2f_and_swap": ("0efd8d2ab76b1dfa58c17d83", 108),
        "w12_ir_fields": ("71405a7b1b9c9fcb94d7cd0e", 3259),
        "w12b_tableau_layout": ("d7725cc663daa9b67976e3ec", 129),
        "w12c_ir_shared_fields": ("0ca5f908e39597880582254e", 503),
        "w1_tableau_precompile": ("fa0a364a9c284fe1bd0a5fb1", 3310),
        "w1b_tableau_precompile_ucu_false": ("cc948f8db4f39911972b62e6", 806),
        "w1c_tableau_precompile_full": ("1189ee39fa7dfd141871a27c", 6531),
        "w2_coset_reference": ("4f48ea8c9eb0f7ae71e7ac1f", 54876),
        "w3_magic_pass": ("d9f4cb278b577125ef3ec784", 11997),
        "w3b_extract_faults": ("6068689e3d6f1fcd747f38ac", 377),
        "w3c_magic_pass_external_fault_slots": ("3e2084a59bf76f462a3a2fe9", 12993),
        "w3d_magic_pass_selector": ("664b9b3b60d7d42a790b77db", 40226),
        "w4_marginal_builder": ("0d6711ac3168edfa22f4d85f", -1),
        "w4b_marginal_builder_oracle": ("0d6711ac3168edfa22f4d85f", -1),
        "w4c_marginal_builder_abi": ("0d6711ac3168edfa22f4d85f", -1),
        "w4d_marginal_builder_uc_false": ("0d6711ac3168edfa22f4d85f", -1),
        "w5_sync_reference": ("df024d09f811d9a18cea5931", 33),
        "w5b_sync_reference_nohandler": ("df024d09f811d9a18cea5931", 33),
        "w6_known_states": ("830aac93f38d4381e6e3e4ec", 855),
        "w7_strip": ("45960e02e9b06255853dd229", 180),
        "w8_parse_magic": ("646c564cbf8894282a0c7067", 670),
        "w8b_parse_tableau": ("be5e77216f3e8a5be97cbd73", 662),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "c10_magic_multiword_68q": {
        "w10_unroll": ("bebfefbc2a18633fce79f6ee", 223),
        "w11_u2f_and_swap": ("0efd8d2ab76b1dfa58c17d83", 108),
        "w12_ir_fields": ("b170e145e8f180b831e7ad95", 2195),
        "w12b_tableau_layout": ("7519359a7edd3ad28a16bf97", 131),
        "w12c_ir_shared_fields": ("6b3fac828da898784eea99b2", 493),
        "w1_tableau_precompile": ("5bfd651ba5278ee6a778c937", 1870),
        "w1b_tableau_precompile_ucu_false": ("4a7f9dd34b048ea2f022c074", 775),
        "w1c_tableau_precompile_full": ("921a7142994c2755b7a82676", 2739),
        "w2_coset_reference": ("1a194f6621809d35806cdc93", 51746),
        "w3_magic_pass": ("a6c741916e16bc165f25d873", 9302),
        "w3b_extract_faults": ("e954c81bf8731f0761021471", 284),
        "w3c_magic_pass_external_fault_slots": ("eafadbc74d9ffcec10e193a7", 10299),
        "w3d_magic_pass_selector": ("89d2be43fa139655c26edb4d", 29228),
        "w4_marginal_builder": ("fa85daf0d2ee52d2da9fdd75", 930),
        "w4b_marginal_builder_oracle": ("8a8dd65977065efe3c41be0f", 1025),
        "w4c_marginal_builder_abi": ("09f76018604484a658a6e2c5", 1473),
        "w4d_marginal_builder_uc_false": ("8a8dd65977065efe3c41be0f", 1025),
        "w5_sync_reference": ("1a2324eb321c7b0f6420d098", 23),
        "w5b_sync_reference_nohandler": ("1a2324eb321c7b0f6420d098", 23),
        "w6_known_states": ("5e49ee76b6813e3429e2e5ac", 831),
        "w7_strip": ("ca81cccb893c36eb62a7dcbe", 157),
        "w8_parse_magic": ("eba6d295cc9d2fb45ec93ddf", 80),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "c11_high_prec_gate_arg": {
        "w10_unroll": ("85e85293ce9f752ed8956831", 220),
        "w11_u2f_and_swap": ("e6bdd9a5db6e6fac616a0e8c", 75),
        "w12_ir_fields": ("6c336dbeac520186dca41e3a", 1892),
        "w12b_tableau_layout": ("a0a58a42e5e86988131f85b5", 81),
        "w12c_ir_shared_fields": ("8e98a77e6c6aa7e6e7fc1cd0", 331),
        "w1_tableau_precompile": ("fe63cac9e86cf98c71a96503", 1679),
        "w1b_tableau_precompile_ucu_false": ("fc6a4747b853b4ebef65d89a", 455),
        "w1c_tableau_precompile_full": ("e1dcecc85971eb12e4e5c0be", 3231),
        "w2_coset_reference": ("f75ee84dfbd349ca6ceca421", 2377),
        "w3_magic_pass": ("6ecc18655dacb8f8e7421c3f", 2959),
        "w3b_extract_faults": ("c1ce60b806c4737c05ec7bad", 211),
        "w3c_magic_pass_external_fault_slots": ("29edf0b794644997817e8243", 3906),
        "w3d_magic_pass_selector": ("1a27ee22532372720b58366f", 6796),
        "w4_marginal_builder": ("e095c54583b885c89280d92a", 924),
        "w4b_marginal_builder_oracle": ("e594cff497544bc43106e675", 1106),
        "w4c_marginal_builder_abi": ("f8202d8ef4ac5ad35bfe3b37", 980),
        "w4d_marginal_builder_uc_false": ("e594cff497544bc43106e675", 1106),
        "w5_sync_reference": ("1a2324eb321c7b0f6420d098", 23),
        "w5b_sync_reference_nohandler": ("1a2324eb321c7b0f6420d098", 23),
        "w6_known_states": ("e669fa1290b4c2f929a0e473", 37),
        "w7_strip": ("f7a16126930014f17eafc094", 91),
        "w8_parse_magic": ("d1b987b8f92e9a7bea708b9d", 301),
        "w8b_parse_tableau": ("4d367f1098c0cb97e714dea8", 293),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "c12_overlapping_lt2_pauli_xor": {
        "w10_unroll": ("dbd17f092de7857b0860338a", 180),
        "w11_u2f_and_swap": ("e6bdd9a5db6e6fac616a0e8c", 75),
        "w12_ir_fields": ("283a85ae40a2506e8a6fe717", 1684),
        "w12b_tableau_layout": ("3e95b9e57ddce8d2ffc5c4cd", 81),
        "w12c_ir_shared_fields": ("b25b1e2457d653fa9bc7c298", 319),
        "w1_tableau_precompile": ("60e1770f13827edd38192798", 1515),
        "w1b_tableau_precompile_ucu_false": ("dc479e428383137c1d5121d3", 420),
        "w1c_tableau_precompile_full": ("2bead6fdaf741c225bbc9439", 2883),
        "w2_coset_reference": ("c1c01fe388f61e23bf3328ed", 2825),
        "w3_magic_pass": ("58102b0f7a61e72263a97c40", 2334),
        "w3b_extract_faults": ("ecd11858899318cd9f4f53ac", 171),
        "w3c_magic_pass_external_fault_slots": ("9835f42d558b5b6c136382b2", 3281),
        "w3d_magic_pass_selector": ("dd2891304e5271bbca59d6d1", 5563),
        "w4_marginal_builder": ("62e726039a0286b92e904043", 924),
        "w4b_marginal_builder_oracle": ("5e4b0494ad20b1437d063252", -1),
        "w4c_marginal_builder_abi": ("5e4b0494ad20b1437d063252", -1),
        "w4d_marginal_builder_uc_false": ("5e4b0494ad20b1437d063252", -1),
        "w5_sync_reference": ("1a2324eb321c7b0f6420d098", 23),
        "w5b_sync_reference_nohandler": ("1a2324eb321c7b0f6420d098", 23),
        "w6_known_states": ("d7958111bb13546ce5fe298a", 37),
        "w7_strip": ("bccbac20b81dd2711850b663", 106),
        "w8_parse_magic": ("fb6cba7bcaf77efeca905ae2", 251),
        "w8b_parse_tableau": ("3b69f89d7e68847ae87e603c", 243),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "c13_leaked_untagged_spp": {
        "w10_unroll": ("90ac5fd7988682c2ef68be95", 191),
        "w11_u2f_and_swap": ("e6bdd9a5db6e6fac616a0e8c", 75),
        "w12_ir_fields": ("98475ef59e202d30ab32491e", 1757),
        "w12b_tableau_layout": ("a0a58a42e5e86988131f85b5", 81),
        "w12c_ir_shared_fields": ("8e98a77e6c6aa7e6e7fc1cd0", 331),
        "w1_tableau_precompile": ("353fe2310a7b7abaaf895918", 1585),
        "w1b_tableau_precompile_ucu_false": ("c5c76406a8a3fc9469fa1465", 429),
        "w1c_tableau_precompile_full": ("2c35def65ac80409bd75c498", 3071),
        "w2_coset_reference": ("4ee56c0992997d6fec7e3b1d", -1),
        "w3_magic_pass": ("bd7a1fb117b9e8631d621061", 1765),
        "w3b_extract_faults": ("c44a65dc4c3e76af929854e8", 182),
        "w3c_magic_pass_external_fault_slots": ("6a23e5a7998d177dabe51ff2", 2712),
        "w3d_magic_pass_selector": ("de2a83af736b031e040f17e9", 4137),
        "w4_marginal_builder": ("df7bc772e9482d1c9a4c3088", -1),
        "w4b_marginal_builder_oracle": ("df7bc772e9482d1c9a4c3088", -1),
        "w4c_marginal_builder_abi": ("df7bc772e9482d1c9a4c3088", -1),
        "w4d_marginal_builder_uc_false": ("34a74d17e610e927ee9f08cb", -1),
        "w5_sync_reference": ("a8158a5a93d9bcf5dc2f66b8", 23),
        "w5b_sync_reference_nohandler": ("a8158a5a93d9bcf5dc2f66b8", 23),
        "w6_known_states": ("55bf210754d4426ac2b84fd2", 37),
        "w7_strip": ("a1ce005c6751a99283003bde", 118),
        "w8_parse_magic": ("06c2a367c298f5befc9bd124", 296),
        "w8b_parse_tableau": ("486e391ad392e1769ba828d9", 288),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "c14_marginal_dem_skip_gate": {
        "w10_unroll": ("a5218d91e8a06193a5d2bdf3", 333),
        "w11_u2f_and_swap": ("773f43b023a63432c0332e68", 124),
        "w12_ir_fields": ("25b5942a8c0a8f50c45858cb", 3050),
        "w12b_tableau_layout": ("1dd05926551c2034d65a6f23", 131),
        "w12c_ir_shared_fields": ("96f674ff4c1664f3c00f3986", 611),
        "w1_tableau_precompile": ("0edbe697f58664041dff097b", 2634),
        "w1b_tableau_precompile_ucu_false": ("ffaeaef3abbae58e34d3c991", 786),
        "w1c_tableau_precompile_full": ("19d81acc176f34b528936727", 4927),
        "w2_coset_reference": ("33faa702f6bb3e50a977d2aa", 4567),
        "w3_magic_pass": ("9159aec1261e2773901e341a", 3112),
        "w3b_extract_faults": ("1a52688c41b2ee7aeb82700c", 312),
        "w3c_magic_pass_external_fault_slots": ("4ac33f1ada04633cfa286327", 4060),
        "w3d_magic_pass_selector": ("afb796e3797b6a32f1ed7a08", 7028),
        "w4_marginal_builder": ("969ca0f3c5fb5c431af4ae50", 1041),
        "w4b_marginal_builder_oracle": ("b51335210296c3cf8885f55e", 1153),
        "w4c_marginal_builder_abi": ("dfd0be2926da609e3f7b286a", 1330),
        "w4d_marginal_builder_uc_false": ("34aff9fbb378d632058e075a", 1098),
        "w5_sync_reference": ("8272aad10727c3fcd7119a02", 26),
        "w5b_sync_reference_nohandler": ("8272aad10727c3fcd7119a02", 26),
        "w6_known_states": ("94c6fca26fb657b4fff6a963", 56),
        "w7_strip": ("8ee3bcbb397c26e891f22a16", 175),
        "w8_parse_magic": ("b8dede5ded5a84a245f08bb9", 465),
        "w8b_parse_tableau": ("740c5b6afeb72b8872d8912f", 457),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "c15_marginal_dem_pauli_leak": {
        "w10_unroll": ("098f92f6bb2f97785dceee8e", 262),
        "w11_u2f_and_swap": ("9d68bdc8f7b1315d12a06312", 100),
        "w12_ir_fields": ("747c3c413d69b4499272bda4", 2388),
        "w12b_tableau_layout": ("1a060162504076cb51d864cf", 105),
        "w12c_ir_shared_fields": ("9f31b955fa80fc030b050d13", 473),
        "w1_tableau_precompile": ("ef51dd8aaa39f05ffe8543ed", 2151),
        "w1b_tableau_precompile_ucu_false": ("7cd6d704433df15487949c34", 622),
        "w1c_tableau_precompile_full": ("b1fd194ea3a37b6689610afc", 4103),
        "w2_coset_reference": ("bc60eeb7534d9ea03fab8a35", 3374),
        "w3_magic_pass": ("64f93bf161be3234ba5b16cf", 2456),
        "w3b_extract_faults": ("0bdcef737058f09725f81a6c", 247),
        "w3c_magic_pass_external_fault_slots": ("dcf87789c3124074ca302f6a", 3404),
        "w3d_magic_pass_selector": ("e134bfd09324e9b4bcec0dc0", 5496),
        "w4_marginal_builder": ("c9d5466cd6ee119b115aa256", 1059),
        "w4b_marginal_builder_oracle": ("43b11b2cce901491778f0889", 1180),
        "w4c_marginal_builder_abi": ("2e27994e548f97f1bdd3d8a0", 1150),
        "w4d_marginal_builder_uc_false": ("9a1e0787fb30695e68d126a1", 1065),
        "w5_sync_reference": ("ee7d2c70eadb9c41d829f1a6", 25),
        "w5b_sync_reference_nohandler": ("ee7d2c70eadb9c41d829f1a6", 25),
        "w6_known_states": ("620defb490f21f77179cf8c6", 47),
        "w7_strip": ("91b940e9e4b5eb5ad9d0ba22", 134),
        "w8_parse_magic": ("2cf3101f2c8fc7bf123f2921", 403),
        "w8b_parse_tableau": ("d20fee125866484022284cde", 395),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "c16_sparse_multibatch": {
        "w10_unroll": ("bca7129e9ea136c0a7e83306", 805),
        "w11_u2f_and_swap": ("773f43b023a63432c0332e68", 124),
        "w12_ir_fields": ("543db921dc04f8ef3c2a60f9", 5779),
        "w12b_tableau_layout": ("5d2b7d2e1561b653e5a8870e", 157),
        "w12c_ir_shared_fields": ("8413c331b7e7f25cd476b6d0", 591),
        "w1_tableau_precompile": ("7cb808af13a391c12451e8ea", 6205),
        "w1b_tableau_precompile_ucu_false": ("9fe695af8cc02528394be6db", 1070),
        "w1c_tableau_precompile_full": ("5a7bb2181f8356fee97af656", 12599),
        "w2_coset_reference": ("bb50f715b48dd56bf9cab991", 18390),
        "w3_magic_pass": ("91054de6da5ed68d0d968765", 16946),
        "w3b_extract_faults": ("7b92a5208c352f8654cc275f", 784),
        "w3c_magic_pass_external_fault_slots": ("7a807c4968d8d86374e5c61a", 17894),
        "w3d_magic_pass_selector": ("da51065a144fc4f7c006f258", 44725),
        "w4_marginal_builder": ("c2bce55950daa71f152a2c47", -1),
        "w4b_marginal_builder_oracle": ("c2bce55950daa71f152a2c47", -1),
        "w4c_marginal_builder_abi": ("c2bce55950daa71f152a2c47", -1),
        "w4d_marginal_builder_uc_false": ("c2bce55950daa71f152a2c47", -1),
        "w5_sync_reference": ("d316fbe74b5446d835c4dde0", 248),
        "w5b_sync_reference_nohandler": ("9a0e4db8b1a4b4e2c5ffae5d", 45),
        "w6_known_states": ("d6f8efd21c4dfa3a49801fc2", 295),
        "w7_strip": ("87ab8cf3866a435707673eef", 276),
        "w8_parse_magic": ("8417af7b3ed50aa7d852c721", 1381),
        "w8b_parse_tableau": ("3549c461ab496819479ba314", 1373),
        "w9_collapses_z": ("3cbc87c7681f34db4617feaa", 4),
    },
    "g_CX": {
        "w10_unroll": ("0b94765cfa1e90e8a5b65c9e", 26),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("334b0438754402ea1ce5135d", 469),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("9c98151b16127b74c6571e54", 518),
        "w1b_tableau_precompile_ucu_false": ("1964af6657f01b4e8b80d3b4", 218),
        "w1c_tableau_precompile_full": ("0c47a4dc8c0fb50ea22fb059", 913),
        "w2_coset_reference": ("59fc0983aa646699a88bf0b1", 1034),
        "w3_magic_pass": ("7d720ca2333d4f0019c6f625", 1482),
        "w3b_extract_faults": ("a0f622d952b4ae891e45e6e8", 25),
        "w3c_magic_pass_external_fault_slots": ("281e10704c94ed3e7eab25c3", 2429),
        "w3d_magic_pass_selector": ("556130fc60f064a5b8768822", 3513),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("e0ac573e6516e94307bc9796", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("fd5272c3e9dd0515d35dd39e", 30),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_CXSWAP": {
        "w10_unroll": ("ec8ee1812a7fb0852c54ef44", 30),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("f4c9063ecddbd0cf885b1fd2", 489),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("9c0d73a9e734ceef3bb201e1", 534),
        "w1b_tableau_precompile_ucu_false": ("6d8a9e97b976125d7e84a800", 222),
        "w1c_tableau_precompile_full": ("50e9c3eaf75a4b762474fd3c", 929),
        "w2_coset_reference": ("307806c79fb91b57389ddd81", 1046),
        "w3_magic_pass": ("46a4e64761c54b51a1ae149d", 1486),
        "w3b_extract_faults": ("e5dca1e86c93a9ac474077ba", 29),
        "w3c_magic_pass_external_fault_slots": ("2799798b46dbb1b54ed97c27", 2433),
        "w3d_magic_pass_selector": ("baa65eb37e6c8a2b21385f46", 3529),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("acac37ab7d0de78b974eef4f", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("bf6e91084137d5755efbbea0", 34),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_CY": {
        "w10_unroll": ("59390f728c1d63d529b76506", 26),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("5582956bbf9f2e51b50ac0f8", 469),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("5d6d2c661b3475b628b9fcdb", 518),
        "w1b_tableau_precompile_ucu_false": ("f71ac9ace1104de83abae252", 218),
        "w1c_tableau_precompile_full": ("6e14a71d95df1efb1d7fbe4d", 913),
        "w2_coset_reference": ("31da35fe4f87b0b91a08cae1", 1034),
        "w3_magic_pass": ("1491ed7901236719bc1c02f5", 1482),
        "w3b_extract_faults": ("ee8fd5c9a7e12486c381956f", 25),
        "w3c_magic_pass_external_fault_slots": ("ed2cd490624e6ff81ca86835", 2429),
        "w3d_magic_pass_selector": ("63445d14914e5c1bfe3d2f6b", 3513),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("bf66a433bcc6e6f66b7d2259", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("1b460ca0b627d5d10c10e68b", 30),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_CZ": {
        "w10_unroll": ("988b9cbc6036c0bfc9cb6424", 26),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("21abf97375f2cf08e04cf6a6", 469),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("20c602e63859aeb74750386b", 518),
        "w1b_tableau_precompile_ucu_false": ("9be702a600446e93be4deb03", 218),
        "w1c_tableau_precompile_full": ("812ddafffea406e9db35db9d", 913),
        "w2_coset_reference": ("82886938f789e3e09fb7d0d5", 1034),
        "w3_magic_pass": ("0051de997d90b81ccd180d4c", 1482),
        "w3b_extract_faults": ("f0bbc7099b800fa1d2f7818e", 25),
        "w3c_magic_pass_external_fault_slots": ("d84412618e66d4e33f98982e", 2429),
        "w3d_magic_pass_selector": ("91a4ef9c038e43711921c0be", 3513),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("44750bcddfb46adcbf5d9e1c", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("4462d6721b5c28a7ee834c8e", 30),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_CZSWAP": {
        "w10_unroll": ("b834138b73820c86a55cadd1", 30),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("2d463fd9555bf3d71d47543f", 489),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("f9105b1f7e7dd3ba7294fbcb", 534),
        "w1b_tableau_precompile_ucu_false": ("68a481a423d76587c8f56014", 222),
        "w1c_tableau_precompile_full": ("5928abe41a788eb1a5c1083f", 929),
        "w2_coset_reference": ("f8e66c848b13803081f065ba", 1046),
        "w3_magic_pass": ("2e666e8435200ca92b403611", 1486),
        "w3b_extract_faults": ("f26f8bc44d4d254ff100676e", 29),
        "w3c_magic_pass_external_fault_slots": ("b4f5692e22186688dc2a96b5", 2433),
        "w3d_magic_pass_selector": ("d15dfe1723c3337b9f4faa62", 3529),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("acac37ab7d0de78b974eef4f", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("a3df38316da17a9ffcc2a224", 34),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_C_NXYZ": {
        "w10_unroll": ("717f66ae04e2508e35602b72", 28),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("367bef18abb472a1dc9e8089", 479),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("26ead1d1dc93144d146aa7d9", 516),
        "w1b_tableau_precompile_ucu_false": ("edba024740eccb5c3bb39882", 220),
        "w1c_tableau_precompile_full": ("10a8ccb3afe508f1d7dcca26", 901),
        "w2_coset_reference": ("64863accd1acd5d788214914", 1040),
        "w3_magic_pass": ("3dddfcb8b970f74d875dd31c", 1484),
        "w3b_extract_faults": ("eed61e6cfc21a43decc004f5", 27),
        "w3c_magic_pass_external_fault_slots": ("eb45ff43e6b127a8e7392cd5", 2431),
        "w3d_magic_pass_selector": ("76da4d04366929c28218b170", 3521),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("c444975e2e6a0e9d4ad184be", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("acc94fd1206822d79af544bc", 25),
        "w7_strip": ("d5f6cece2589fc704a66f967", 32),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_C_NZYX": {
        "w10_unroll": ("441cebaf92e45bbc4f3a5654", 28),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("f02977b83aab3729b5398ba9", 479),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("0af6b74f2e527d7ad18ed946", 516),
        "w1b_tableau_precompile_ucu_false": ("7b7d4579225875a8fdb98795", 220),
        "w1c_tableau_precompile_full": ("d6d8bb7c565405c95424fc08", 901),
        "w2_coset_reference": ("547ce8e3cf560530003147ce", 1040),
        "w3_magic_pass": ("dcbfe014743e0fb7c02d126a", 1484),
        "w3b_extract_faults": ("c2a9839b49dc838de78fb0d9", 27),
        "w3c_magic_pass_external_fault_slots": ("fe61be4ae9a4d280d71f3d9f", 2431),
        "w3d_magic_pass_selector": ("ef73b6f10c2ef9888c1b4ac6", 3521),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("6e58eab7c27ef1894dbf3502", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("260963edcf06d931555b21b4", 25),
        "w7_strip": ("7d32dbb68533806b2cc6ef5d", 32),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_C_XNYZ": {
        "w10_unroll": ("06caf1d09308728840386529", 28),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("494a6bbe177ce420a9caf75a", 479),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("1c1c03ccff32ca9e5bbe1b2c", 516),
        "w1b_tableau_precompile_ucu_false": ("9370c79aa1fbfea865dd4fc3", 220),
        "w1c_tableau_precompile_full": ("bd8c4b66d5c3799f3a546ff7", 901),
        "w2_coset_reference": ("2e5175adfe99cd67ac4a2362", 1040),
        "w3_magic_pass": ("dc2b4b9132b15ee9d27c7350", 1484),
        "w3b_extract_faults": ("d67c331ab271c59667306538", 27),
        "w3c_magic_pass_external_fault_slots": ("58988726e90d01d554a40481", 2431),
        "w3d_magic_pass_selector": ("79dd7ebeff99f744cf419abf", 3521),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("9723da511a883bc8c0e6d4e8", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("a70f5667f4e83b9d9ef0d64f", 25),
        "w7_strip": ("bf41d2d35a5fa3e3ef066f1c", 32),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_C_XYNZ": {
        "w10_unroll": ("1167b8702e5a809b0c122997", 28),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("f6d00c6467a6f7acdf9e627c", 479),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("2f1bb1a66af173f61c8a6ad7", 516),
        "w1b_tableau_precompile_ucu_false": ("a6ba00189da0891df43b682b", 220),
        "w1c_tableau_precompile_full": ("73d8f891fb1a0e11baeb54fc", 901),
        "w2_coset_reference": ("0ea870094c4dc49b4c05148d", 1040),
        "w3_magic_pass": ("e707c7b06c0549e69bcfb7a0", 1484),
        "w3b_extract_faults": ("5f1a8f40d512f0afdabf697c", 27),
        "w3c_magic_pass_external_fault_slots": ("d14a5522a9d31be1693943f9", 2431),
        "w3d_magic_pass_selector": ("e5107034c9bb5224f87f5f62", 3521),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("8898640aa739e7623e8f35dc", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("acc94fd1206822d79af544bc", 25),
        "w7_strip": ("0040dd9ae11798f69ba5f466", 32),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_C_XYZ": {
        "w10_unroll": ("7832bd434ebc45132c79cd76", 27),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("0e84819769fe8d0b3d63f6ac", 474),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("db2d95c2fe5c943a5f3b48a9", 512),
        "w1b_tableau_precompile_ucu_false": ("976804bb98120c52c9ba57b7", 219),
        "w1c_tableau_precompile_full": ("e672d08efd3548df8b96ca0f", 897),
        "w2_coset_reference": ("49bf18be276bb7c3aa8decb9", 1037),
        "w3_magic_pass": ("420fc327b5dd1ae023b7d214", 1483),
        "w3b_extract_faults": ("125a76608564bb9fd1e34707", 26),
        "w3c_magic_pass_external_fault_slots": ("7c9382270ec76555b25769b6", 2430),
        "w3d_magic_pass_selector": ("ed2bd45dae1925a867fc5c07", 3517),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("86ecc88da9f9b1f2048e321c", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("a70f5667f4e83b9d9ef0d64f", 25),
        "w7_strip": ("a670036d9c6a3ea5ca3a5044", 31),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_C_ZNYX": {
        "w10_unroll": ("76ed332124640efddda925fb", 28),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("91f73d702538517c961c8836", 479),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("53e07a6e871f058e1784117b", 516),
        "w1b_tableau_precompile_ucu_false": ("4835d04c136481bd3d938064", 220),
        "w1c_tableau_precompile_full": ("6559c004c7804955abbe124c", 901),
        "w2_coset_reference": ("0643e6df738db8fc57a23338", 1040),
        "w3_magic_pass": ("491f14454e109711d14ade11", 1484),
        "w3b_extract_faults": ("ed0e4970675105c896cb2ab2", 27),
        "w3c_magic_pass_external_fault_slots": ("4c35f37050b67d04ebed10aa", 2431),
        "w3d_magic_pass_selector": ("0735bbea28b2594583159db3", 3521),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("036e6e6645adb19563e20ebe", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("260963edcf06d931555b21b4", 25),
        "w7_strip": ("8591460dc40f3b850d532023", 32),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_C_ZYNX": {
        "w10_unroll": ("8e6e4cc2ae16a49d1fc955d6", 28),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("e4832e677c88c9177e081457", 479),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("993321c6e9e8a79cd74001c5", 516),
        "w1b_tableau_precompile_ucu_false": ("3577dc94d90473ecc1bd2658", 220),
        "w1c_tableau_precompile_full": ("4c6cbcb500495ad43b8f2012", 901),
        "w2_coset_reference": ("05c9c4074871c8b675ad645c", 1040),
        "w3_magic_pass": ("7137f545c76d5bed151ba36b", 1484),
        "w3b_extract_faults": ("7fbc669328de1e23b71c92c3", 27),
        "w3c_magic_pass_external_fault_slots": ("9198b60770ceb1f93a2b66b7", 2431),
        "w3d_magic_pass_selector": ("29500f69155d833e16f0fec4", 3521),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("bdc7ae07ce508ba19c45697b", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("497f162cf1b8512de38d3b59", 25),
        "w7_strip": ("771510982e4f27294c3b9083", 32),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_C_ZYX": {
        "w10_unroll": ("bc79110330459e4724deb038", 27),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("73a62db6ff46fda781069561", 474),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("451f08f11c5b52d800faff40", 512),
        "w1b_tableau_precompile_ucu_false": ("47de1b5a558f502e4f8165f2", 219),
        "w1c_tableau_precompile_full": ("bf0b257f3d09499a81203edf", 897),
        "w2_coset_reference": ("b26bd7a2e0f6c3cfab81e97d", 1037),
        "w3_magic_pass": ("26378f3e2a4f6e19ff9a2e30", 1483),
        "w3b_extract_faults": ("381ae4a1e1d301108b12f92e", 26),
        "w3c_magic_pass_external_fault_slots": ("4bcdded142f277c639eec1d9", 2430),
        "w3d_magic_pass_selector": ("e051c18a06a0618adcb69cab", 3517),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("a24ec3ab63cdfbaa838ac4e9", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("497f162cf1b8512de38d3b59", 25),
        "w7_strip": ("3bdf0e6ea375c2a980199aea", 31),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_DEPOLARIZE1": {
        "w10_unroll": ("0186cf87a52f6624113f7282", 39),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("28f86e6878ec796656412856", 530),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("4b9b0a9706be621aa71c23ed", 141),
        "w1_tableau_precompile": ("25d9c7977baf6a286c7e3a34", 554),
        "w1b_tableau_precompile_ucu_false": ("3e36435d9171847ba466d2f4", 231),
        "w1c_tableau_precompile_full": ("7719401c84df530ada00717c", 941),
        "w2_coset_reference": ("e33487359cb2fd9b25e70cf7", 1012),
        "w3_magic_pass": ("4974a3db119787c2f9369c42", 1830),
        "w3b_extract_faults": ("0588804eaae6d0a0be9aec6a", 38),
        "w3c_magic_pass_external_fault_slots": ("ffd42e1a5588092c8ab61b15", 2777),
        "w3d_magic_pass_selector": ("86d83d0046e47ffeaf26d918", 4314),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("47f5f54aa4ee03a293b15c51", 671),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("499d779e6da186b2b15eb9e8", 22),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_DEPOLARIZE2": {
        "w10_unroll": ("4b49e203810606b0ca28e05a", 41),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("76b42fc1324e9bd0c877f530", 540),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("4b9b0a9706be621aa71c23ed", 141),
        "w1_tableau_precompile": ("1bcfe10012d042439d5bb8d6", 572),
        "w1b_tableau_precompile_ucu_false": ("d96e9b41db9b3739b88171c7", 233),
        "w1c_tableau_precompile_full": ("7169e5186fc108a4271aa975", 969),
        "w2_coset_reference": ("cf0772abc77a1794919ef6c4", 1014),
        "w3_magic_pass": ("a730796489804996fbcf291e", 3115),
        "w3b_extract_faults": ("8ef9926beb66de6315c72a1c", 40),
        "w3c_magic_pass_external_fault_slots": ("599c218a0b0c0a97f55ff909", 4062),
        "w3d_magic_pass_selector": ("36b94aa1516acf6b495c2be1", 9196),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("b261c24b99dc21ce120f1761", 675),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("499d779e6da186b2b15eb9e8", 22),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_DETECTOR": {
        "w10_unroll": ("44816dc63c8dfe92720b052d", 42),
        "w11_u2f_and_swap": ("f082a425bca6de5a5839a87a", 51),
        "w12_ir_fields": ("2512b9ba47aa1bb866f8e2ed", 628),
        "w12b_tableau_layout": ("93f03fc1521db48a8b10a28c", 57),
        "w12c_ir_shared_fields": ("e746dc09a533057f277245a8", 187),
        "w1_tableau_precompile": ("6688bfbb1f4d5db619874f55", 514),
        "w1b_tableau_precompile_ucu_false": ("511775aba9a2f47073550568", 234),
        "w1c_tableau_precompile_full": ("a3a881e8d79e88c3169c3853", 841),
        "w2_coset_reference": ("2168e72fc960359e935a5272", 1341),
        "w3_magic_pass": ("fa616b502b4e33c5d4ba804d", 1609),
        "w3b_extract_faults": ("70ab2a426cd263abd93a2e61", 39),
        "w3c_magic_pass_external_fault_slots": ("559075a2536ce3332c546b18", 2556),
        "w3d_magic_pass_selector": ("8fb29d9a207dd6efa6694623", 3673),
        "w4_marginal_builder": ("0657f49c138ca3bf9148ecdd", 750),
        "w4b_marginal_builder_oracle": ("9253512089bafa36289b760b", 788),
        "w4c_marginal_builder_abi": ("9459994749028f98e4a5ec19", 719),
        "w4d_marginal_builder_uc_false": ("9253512089bafa36289b760b", 788),
        "w5_sync_reference": ("dc061b1593c3268e0dcd2823", 21),
        "w5b_sync_reference_nohandler": ("dc061b1593c3268e0dcd2823", 21),
        "w6_known_states": ("23db902f07fadcf04a8bdc3b", 28),
        "w7_strip": ("cba01ad8775984d7443bd2fc", 45),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_E": {
        "w10_unroll": ("711e629966243068a0506f06", 32),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("64afb4465f6bac6be5e5042a", 496),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("4b9b0a9706be621aa71c23ed", 141),
        "w1_tableau_precompile": ("449939ccbfd047eb08960ac1", 535),
        "w1b_tableau_precompile_ucu_false": ("24dd046a583d5d4b7d9d82de", 224),
        "w1c_tableau_precompile_full": ("1c14e879c35d57b155f0f4f9", 929),
        "w2_coset_reference": ("df951609c43c92583125740e", 1005),
        "w3_magic_pass": ("2891f0e3987a0b802477f7b4", 1586),
        "w3b_extract_faults": ("e1f8d79f4e6377da38b610e7", 31),
        "w3c_magic_pass_external_fault_slots": ("9813b8b733d00b3e7068c831", 2533),
        "w3d_magic_pass_selector": ("21fdf06028e7f703e1da4974", 3801),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("b261c24b99dc21ce120f1761", 675),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("499d779e6da186b2b15eb9e8", 22),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_ELSE_CORRELATED_ERROR": {
        "w10_unroll": ("cf9953b3bd67303b3fd2f4f9", 61),
        "w11_u2f_and_swap": ("f082a425bca6de5a5839a87a", 51),
        "w12_ir_fields": ("151bd9612e3fd926a5008b18", 722),
        "w12b_tableau_layout": ("b5bd5d46bdc906f8d7eaa5ae", 69),
        "w12c_ir_shared_fields": ("84d6a022f38615115d3e0d03", 189),
        "w1_tableau_precompile": ("50db058331e0975b5e52411e", 747),
        "w1b_tableau_precompile_ucu_false": ("6077229e18518132e5e5d351", 313),
        "w1c_tableau_precompile_full": ("db4aad8a64834d888af219c4", 1227),
        "w2_coset_reference": ("580bd2940b01061cd87b9954", 1231),
        "w3_magic_pass": ("70dbbf8e1e9fb65a6b993e07", 1737),
        "w3b_extract_faults": ("c83dcf28efc64a4187da2434", 58),
        "w3c_magic_pass_external_fault_slots": ("3968ce3de1bc395e3a2e8d22", 2684),
        "w3d_magic_pass_selector": ("0138d4e0797ff4fa76dc6df9", 4014),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("9d118cce661ad6f635f882f5", 766),
        "w4c_marginal_builder_abi": ("bf2288160712bb68935c45bd", 723),
        "w4d_marginal_builder_uc_false": ("9d118cce661ad6f635f882f5", 766),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("23db902f07fadcf04a8bdc3b", 28),
        "w7_strip": ("499d779e6da186b2b15eb9e8", 22),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_H": {
        "w10_unroll": ("b1514b8fcf2c5bbc83d9aae5", 23),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("179ce76eb96ddb16c75877ca", 454),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("794fa0bb39d817fe7a20b1db", 496),
        "w1b_tableau_precompile_ucu_false": ("fcb544174cc97620f27b66bb", 215),
        "w1c_tableau_precompile_full": ("1c66c7eefaeac575f81588ba", 881),
        "w2_coset_reference": ("4f57ee23e81ec4c2676ae91e", 1025),
        "w3_magic_pass": ("8d6d816d21f041c361c7d6ab", 1479),
        "w3b_extract_faults": ("01334e2d3ad1466c9b423427", 22),
        "w3c_magic_pass_external_fault_slots": ("bf407ac3f92a0c194f62aed1", 2426),
        "w3d_magic_pass_selector": ("717a58f0dbb85d8a5df9d170", 3501),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("3afa652e413816aedaf20585", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("a70f5667f4e83b9d9ef0d64f", 25),
        "w7_strip": ("7b5a56c7b2485dfd8bbf201c", 27),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_HERALDED_ERASE": {
        "w10_unroll": ("448602ba974adf3f43044218", 42),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("b0e22fb4570960ec3a64cd27", 542),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("4b9b0a9706be621aa71c23ed", 141),
        "w1_tableau_precompile": ("2a4e6b90f9b3ad8ffc52b258", 546),
        "w1b_tableau_precompile_ucu_false": ("fff4d021a5993022da5b5e06", 234),
        "w1c_tableau_precompile_full": ("6653beb381a541cbf521d620", 905),
        "w2_coset_reference": ("fc19ea259c3810316527e4f0", 1015),
        "w3_magic_pass": ("fbabad6c98fe66f53ac0f2b4", 1816),
        "w3b_extract_faults": ("eaabf004ac3fe011a1c0c909", 41),
        "w3c_magic_pass_external_fault_slots": ("241318f2d9f29d216979902e", 2763),
        "w3d_magic_pass_selector": ("8372a0c6b63ce599662cb092", 4087),
        "w4_marginal_builder": ("5e011bfacca442c9f7f2b1f3", 728),
        "w4b_marginal_builder_oracle": ("b8068e7e8656c4e94c7f6061", 758),
        "w4c_marginal_builder_abi": ("47f5f54aa4ee03a293b15c51", 671),
        "w4d_marginal_builder_uc_false": ("b8068e7e8656c4e94c7f6061", 758),
        "w5_sync_reference": ("dc061b1593c3268e0dcd2823", 21),
        "w5b_sync_reference_nohandler": ("dc061b1593c3268e0dcd2823", 21),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("a58cb536222b088e3afe7aea", 30),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_HERALDED_PAULI_CHANNEL_1": {
        "w10_unroll": ("69be758699565afee9a22b39", 70),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("7874dfbcb34c140e51c487da", 682),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("4b9b0a9706be621aa71c23ed", 141),
        "w1_tableau_precompile": ("014d15900996b0c73086f62e", 640),
        "w1b_tableau_precompile_ucu_false": ("1df8f3155b5edf481e0c4000", 262),
        "w1c_tableau_precompile_full": ("adf4055a6c02d93bf666aba2", 981),
        "w2_coset_reference": ("d3d1be62a5a41d6306c59041", 1043),
        "w3_magic_pass": ("dba30327839b478c51e62b3a", 1818),
        "w3b_extract_faults": ("cde86f9c53dea857bcc5ee2a", 69),
        "w3c_magic_pass_external_fault_slots": ("1377269143369478d638972f", 2765),
        "w3d_magic_pass_selector": ("4ec2f8e33a264707533f3258", 4091),
        "w4_marginal_builder": ("5e011bfacca442c9f7f2b1f3", 728),
        "w4b_marginal_builder_oracle": ("b8068e7e8656c4e94c7f6061", 758),
        "w4c_marginal_builder_abi": ("47f5f54aa4ee03a293b15c51", 671),
        "w4d_marginal_builder_uc_false": ("b8068e7e8656c4e94c7f6061", 758),
        "w5_sync_reference": ("dc061b1593c3268e0dcd2823", 21),
        "w5b_sync_reference_nohandler": ("dc061b1593c3268e0dcd2823", 21),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("a58cb536222b088e3afe7aea", 30),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_H_NXY": {
        "w10_unroll": ("7d19656436e9542dd77b22ef", 27),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("b820d26a8f248412f6a75b25", 474),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("2676065a5122f637448ffe95", 512),
        "w1b_tableau_precompile_ucu_false": ("5ba73ef189c1339fca5426b7", 219),
        "w1c_tableau_precompile_full": ("67ac3ca5ece323827d25705f", 897),
        "w2_coset_reference": ("7e0f04690f669a923652308c", 1037),
        "w3_magic_pass": ("604c8e5461bdf3de32868506", 1483),
        "w3b_extract_faults": ("33761cd8f51c3b357721ae23", 26),
        "w3c_magic_pass_external_fault_slots": ("0058a4ae2f472694aaa825cb", 2430),
        "w3d_magic_pass_selector": ("8a6321446796159f8b9b76dd", 3517),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("38b290fef6cb247e631597e7", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("ce18155754c8b46c3ada63b5", 25),
        "w7_strip": ("664830c1366c7260e49dbf22", 31),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_H_NXZ": {
        "w10_unroll": ("9f229d8f93fefa5ab9294840", 27),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("0e0bcd46e13ea403906dfba1", 474),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("4f965af0574fcafa4a305dcc", 512),
        "w1b_tableau_precompile_ucu_false": ("db867521adc997d5a92d9538", 219),
        "w1c_tableau_precompile_full": ("39ef00387a831a8c3850c618", 897),
        "w2_coset_reference": ("662c6335c1a388fc7d5ec29c", 1037),
        "w3_magic_pass": ("20bf752775318f02a8f5be1c", 1483),
        "w3b_extract_faults": ("03b2361732ec6be516aa699f", 26),
        "w3c_magic_pass_external_fault_slots": ("de97bdb881d2591e59b00991", 2430),
        "w3d_magic_pass_selector": ("536ec96c6dbe9dcf827f18e6", 3517),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("404d3cf88eaf6638d839b0fd", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("acc94fd1206822d79af544bc", 25),
        "w7_strip": ("2fbe5541ff6f47e07301713d", 31),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_H_NYZ": {
        "w10_unroll": ("47662e51f2710a72b58f43f0", 27),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("2b96d0573bd442f5af203101", 474),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("a2fe9bd3a6cac25510c41b64", 512),
        "w1b_tableau_precompile_ucu_false": ("67774a79b3335bda52fedca4", 219),
        "w1c_tableau_precompile_full": ("1be4e4c1c087659d91cb13b2", 897),
        "w2_coset_reference": ("7c37d5cf770cae00831f3c10", 1037),
        "w3_magic_pass": ("82f8e988bae0b553ac2ce7f6", 1483),
        "w3b_extract_faults": ("75b1b85de9d4cdb6625dd34c", 26),
        "w3c_magic_pass_external_fault_slots": ("0d8ae1dacf1cfad5b8d63163", 2430),
        "w3d_magic_pass_selector": ("3fae4b13cba7964aa8729f19", 3517),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("04d98ab04a333ecc9b70f10b", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("260963edcf06d931555b21b4", 25),
        "w7_strip": ("f1156ba05715620e6ea4e381", 31),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_H_XY": {
        "w10_unroll": ("c1c9b93f31f9529bd8d6aa6c", 26),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("3cdf3bd810204df462f842c8", 469),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("31347417fe44a93bc720c36b", 508),
        "w1b_tableau_precompile_ucu_false": ("e2899d5f555f06f9665f23a5", 218),
        "w1c_tableau_precompile_full": ("56a22addcca50045e04b6bb6", 893),
        "w2_coset_reference": ("148f6e0b05d3a2231bb5122e", 1034),
        "w3_magic_pass": ("d0af494a4e42cbaae21f32e5", 1482),
        "w3b_extract_faults": ("f9e07e62719f8dfef5dd7275", 25),
        "w3c_magic_pass_external_fault_slots": ("10912ac9a6dfb20a8d3d9503", 2429),
        "w3d_magic_pass_selector": ("ee4a17c31c65a92968f4b071", 3513),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("0643ec15a6b76f247eb04b04", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("ce18155754c8b46c3ada63b5", 25),
        "w7_strip": ("d617b8f33fe2fd00b6849c54", 30),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_H_YZ": {
        "w10_unroll": ("9a967c2eaed6e9aaa1f3577b", 26),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("35a59109bfaec59135b52ef1", 469),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("4f37b76ac7dc3b798bb93dc8", 508),
        "w1b_tableau_precompile_ucu_false": ("dcc83e59b52c82020d8b36fd", 218),
        "w1c_tableau_precompile_full": ("83c948852f211aaec7b3087d", 893),
        "w2_coset_reference": ("d8f39b6b86a8422e83853e35", 1034),
        "w3_magic_pass": ("d660d7333fed2845eaf1766a", 1482),
        "w3b_extract_faults": ("2eef26b717cb31b4c85265e1", 25),
        "w3c_magic_pass_external_fault_slots": ("09bdd9cf4c8048777ad8f9b5", 2429),
        "w3d_magic_pass_selector": ("7576708023469fbddb7a352a", 3513),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("29cd37ff95cd91bc9e7636dc", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("497f162cf1b8512de38d3b59", 25),
        "w7_strip": ("0e93ffa2a7f833be1f790415", 30),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_I": {
        "w10_unroll": ("86676cde455d2241f0c06c27", 23),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("564daf4e1df91990044b067e", 454),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("4d0c37f298578173909da389", 496),
        "w1b_tableau_precompile_ucu_false": ("e24a054a34cfc1d6d049a537", 215),
        "w1c_tableau_precompile_full": ("136e52fba1600eed7a23566e", 881),
        "w2_coset_reference": ("b873a2d513b80f09cb398428", 1025),
        "w3_magic_pass": ("d68ddf2f0d36896ab958a674", 1435),
        "w3b_extract_faults": ("73aa24a02465185fbb88acc3", 22),
        "w3c_magic_pass_external_fault_slots": ("93bae0aecededde0dd3c8948", 2382),
        "w3d_magic_pass_selector": ("ee4d390821731568d2d6e1cf", 3325),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("6094d6ec537d92c9b207a138", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("67c8e5021366fd1741dcd6a2", 27),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_II": {
        "w10_unroll": ("52c57f155a7b5ce239f3cfdc", 26),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("fc8275665a018573ba6ff04d", 469),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("be006e6cb551c6fd328f56e5", 518),
        "w1b_tableau_precompile_ucu_false": ("92fc75277f47db69f1319c24", 218),
        "w1c_tableau_precompile_full": ("92e2aa942419de6c7584068c", 913),
        "w2_coset_reference": ("02b6ecc7febac634e6ae7be6", 1034),
        "w3_magic_pass": ("d68ddf2f0d36896ab958a674", 1435),
        "w3b_extract_faults": ("04fae1ae2cc8aa37ed4fe4f4", 25),
        "w3c_magic_pass_external_fault_slots": ("93bae0aecededde0dd3c8948", 2382),
        "w3d_magic_pass_selector": ("ee4d390821731568d2d6e1cf", 3325),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("acac37ab7d0de78b974eef4f", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("5eb19d13682b6df10c4d9fca", 30),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_II_ERROR": {
        "w10_unroll": ("8a21c18ad1659447f1a27432", 32),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("50038672120871ee2e822895", 495),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("4b9b0a9706be621aa71c23ed", 141),
        "w1_tableau_precompile": ("7d424631cb964ee091be9cf4", 542),
        "w1b_tableau_precompile_ucu_false": ("dbab21b42e7d5298dc85a167", 224),
        "w1c_tableau_precompile_full": ("a27d902a3757c63a165f7532", 937),
        "w2_coset_reference": ("c76b0236a04e6630210a50cd", 1005),
        "w3_magic_pass": ("d68ddf2f0d36896ab958a674", 1435),
        "w3b_extract_faults": ("4fe110c7b61734e2a0228e79", 31),
        "w3c_magic_pass_external_fault_slots": ("93bae0aecededde0dd3c8948", 2382),
        "w3d_magic_pass_selector": ("ee4d390821731568d2d6e1cf", 3325),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("b261c24b99dc21ce120f1761", 675),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("499d779e6da186b2b15eb9e8", 22),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_ISWAP": {
        "w10_unroll": ("d29bd53947ca57745f92a8fe", 29),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("15a17d205a53a36a5c025420", 484),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("171ef4c70c02412fbe0545f3", 530),
        "w1b_tableau_precompile_ucu_false": ("0feb59f14c623f6ce2ffc89a", 221),
        "w1c_tableau_precompile_full": ("3e9fc7029e64bd39901a09d8", 925),
        "w2_coset_reference": ("e19022e2155206362ad0b933", 1043),
        "w3_magic_pass": ("5539ab962b67792796f2281a", 1485),
        "w3b_extract_faults": ("5f248ab33705393a6f358dd3", 28),
        "w3c_magic_pass_external_fault_slots": ("aa90a3b2eb161c42ffba0e06", 2432),
        "w3d_magic_pass_selector": ("bb50d2610652ad24c278d7b2", 3525),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("acac37ab7d0de78b974eef4f", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("9c5b7af95f6512adb34c9930", 33),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_ISWAP_DAG": {
        "w10_unroll": ("5b3438448a55e95d1fee633c", 33),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("45491e56a99411e2a064fcb8", 504),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("d1b3a20a3ac50322db5e5fc7", 546),
        "w1b_tableau_precompile_ucu_false": ("4c0280a07331e0eaf6d35fb8", 225),
        "w1c_tableau_precompile_full": ("eeaad5cd65c8f7365e521aeb", 941),
        "w2_coset_reference": ("c99876bc8eb7986cdf518949", 1055),
        "w3_magic_pass": ("15e21b24ef5c94fe7484c305", 1489),
        "w3b_extract_faults": ("24db3161d7115dca41c4153c", 32),
        "w3c_magic_pass_external_fault_slots": ("e4dd6ef82a416a49cf848a02", 2436),
        "w3d_magic_pass_selector": ("48b949e8ec0399a028f8fa6c", 3541),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("acac37ab7d0de78b974eef4f", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("0770f3e36a10f11b27a59869", 37),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_I_ERROR": {
        "w10_unroll": ("3daedc5ff5662a20d857fa41", 29),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("03ef129ae70847b5aeb74890", 480),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("4b9b0a9706be621aa71c23ed", 141),
        "w1_tableau_precompile": ("627d0944ebc2b2651898b214", 520),
        "w1b_tableau_precompile_ucu_false": ("7bbc6fc69f998c475640c506", 221),
        "w1c_tableau_precompile_full": ("6d3790878d2a5e0a4132740b", 905),
        "w2_coset_reference": ("9920d15a9a03b285bf3a7609", 1002),
        "w3_magic_pass": ("d68ddf2f0d36896ab958a674", 1435),
        "w3b_extract_faults": ("82931bc2855c008cd6f7bf11", 28),
        "w3c_magic_pass_external_fault_slots": ("93bae0aecededde0dd3c8948", 2382),
        "w3d_magic_pass_selector": ("ee4d390821731568d2d6e1cf", 3325),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("47f5f54aa4ee03a293b15c51", 671),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("499d779e6da186b2b15eb9e8", 22),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_M": {
        "w10_unroll": ("c3ce526ccabbe98bdf41dfc3", 23),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("99345e793f76e98c0fe1a415", 449),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("b1c032899073a7b9b95d0f9b", 139),
        "w1_tableau_precompile": ("cf805242254c16c0d7f2ea8e", 476),
        "w1b_tableau_precompile_ucu_false": ("d0d3f63c16d4f368b4ef13bb", 215),
        "w1c_tableau_precompile_full": ("a3a881e8d79e88c3169c3853", 841),
        "w2_coset_reference": ("4e18fc289097a848144d142b", 1125),
        "w3_magic_pass": ("fa616b502b4e33c5d4ba804d", 1609),
        "w3b_extract_faults": ("75823ebe477718f4a5da3f9d", 22),
        "w3c_magic_pass_external_fault_slots": ("1ee8c3b7f34fdf2877040eb8", 2556),
        "w3d_magic_pass_selector": ("8fb29d9a207dd6efa6694623", 3673),
        "w4_marginal_builder": ("5e011bfacca442c9f7f2b1f3", 728),
        "w4b_marginal_builder_oracle": ("b8068e7e8656c4e94c7f6061", 758),
        "w4c_marginal_builder_abi": ("47bd05ef7f5e094c3a22a76b", 671),
        "w4d_marginal_builder_uc_false": ("b8068e7e8656c4e94c7f6061", 758),
        "w5_sync_reference": ("dc061b1593c3268e0dcd2823", 21),
        "w5b_sync_reference_nohandler": ("dc061b1593c3268e0dcd2823", 21),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("71fec6447a0b499a6d913317", 27),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_MPAD": {
        "w10_unroll": ("39397f41b63a823033574b60", 26),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("ab76d56445f6378d1feb9efa", 458),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("4b9b0a9706be621aa71c23ed", 141),
        "w1_tableau_precompile": ("ff50f060e025bd3c4f7569f3", 488),
        "w1b_tableau_precompile_ucu_false": ("154b08e09b82aac973abf575", 218),
        "w1c_tableau_precompile_full": ("d0ab31cd423bfc8f30bfa7c5", 853),
        "w2_coset_reference": ("8bd057d9a6ad6d5fecd6ea18", 999),
        "w3_magic_pass": ("c86f7a057b8ba11d3d2f6a5b", 1488),
        "w3b_extract_faults": ("cb7afc99168ea178b4ad22c7", 25),
        "w3c_magic_pass_external_fault_slots": ("53c659f8f6b5a08945df73b6", 2435),
        "w3d_magic_pass_selector": ("1ffdec71d3f826cca009c11e", 3431),
        "w4_marginal_builder": ("5e011bfacca442c9f7f2b1f3", 728),
        "w4b_marginal_builder_oracle": ("b8068e7e8656c4e94c7f6061", 758),
        "w4c_marginal_builder_abi": ("da76bccfd15409906f2fa9ee", 667),
        "w4d_marginal_builder_uc_false": ("b8068e7e8656c4e94c7f6061", 758),
        "w5_sync_reference": ("dc061b1593c3268e0dcd2823", 21),
        "w5b_sync_reference_nohandler": ("dc061b1593c3268e0dcd2823", 21),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("a58cb536222b088e3afe7aea", 30),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_MPP": {
        "w10_unroll": ("47319da0aeab4199c50c633d", 29),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("148c06d99aa4fe297ac66916", 479),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("24db0a282c297bb90ed1f4c3", 139),
        "w1_tableau_precompile": ("075e522cc0c69bd9863e6d6d", 496),
        "w1b_tableau_precompile_ucu_false": ("7de40e62d8ec187fbbfcf774", 221),
        "w1c_tableau_precompile_full": ("e3e16e69f1695d665aadf7ff", 857),
        "w2_coset_reference": ("dffb8ccee36c44e31628c53e", 1304),
        "w3_magic_pass": ("11f50318e020142f9ec9d3f9", 1798),
        "w3b_extract_faults": ("9a4c560e9c54e98eb1cb9911", 28),
        "w3c_magic_pass_external_fault_slots": ("da60c7c38575412e3379c5f9", 2745),
        "w3d_magic_pass_selector": ("632576b1c90b889df7a30927", 4447),
        "w4_marginal_builder": ("5e011bfacca442c9f7f2b1f3", 728),
        "w4b_marginal_builder_oracle": ("b8068e7e8656c4e94c7f6061", 758),
        "w4c_marginal_builder_abi": ("08147da0b5d94c459da5d76c", 675),
        "w4d_marginal_builder_uc_false": ("b8068e7e8656c4e94c7f6061", 758),
        "w5_sync_reference": ("dc061b1593c3268e0dcd2823", 21),
        "w5b_sync_reference_nohandler": ("dc061b1593c3268e0dcd2823", 21),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("8d2e59622320cfb90a724b2b", 33),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_MR": {
        "w10_unroll": ("83a4591ae5da36e541473761", 24),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("3a9691c99fa0e8f22d140d74", 466),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("b1c032899073a7b9b95d0f9b", 139),
        "w1_tableau_precompile": ("74db025328fcb36bdb1f253f", 480),
        "w1b_tableau_precompile_ucu_false": ("4e7d3af02925558ac9d9a8df", 216),
        "w1c_tableau_precompile_full": ("256d8a027cd8a4acbe0aa9d4", 845),
        "w2_coset_reference": ("5680b44dcf8eddf7a34ffbc7", 1126),
        "w3_magic_pass": ("31aff11147dc619c2b043c5e", 1725),
        "w3b_extract_faults": ("334316d9f6e444a6a23ffbc7", 23),
        "w3c_magic_pass_external_fault_slots": ("a0b1c241f8dfb05c3c1a2d23", 2672),
        "w3d_magic_pass_selector": ("2b9e7c0216d35a070fc8f2cc", 3905),
        "w4_marginal_builder": ("5e011bfacca442c9f7f2b1f3", 728),
        "w4b_marginal_builder_oracle": ("b8068e7e8656c4e94c7f6061", 758),
        "w4c_marginal_builder_abi": ("678e594b325996c8dc6c038b", 671),
        "w4d_marginal_builder_uc_false": ("b8068e7e8656c4e94c7f6061", 758),
        "w5_sync_reference": ("dc061b1593c3268e0dcd2823", 21),
        "w5b_sync_reference_nohandler": ("dc061b1593c3268e0dcd2823", 21),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("863f68b3124fb05fc6fb71f5", 28),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_MRX": {
        "w10_unroll": ("1b86706f5103a6f077cc9ba3", 25),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("38a1deab831730ad27214e23", 471),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("2da32a3e54a316ce0ebd85ee", 139),
        "w1_tableau_precompile": ("dfd186cdd604c5aab713f27e", 484),
        "w1b_tableau_precompile_ucu_false": ("53c4144444d05ac37e0e7064", 217),
        "w1c_tableau_precompile_full": ("6efca276e060bc49bfac0b2a", 849),
        "w2_coset_reference": ("ac2053c1f28feeb2d7a2c27d", 1269),
        "w3_magic_pass": ("93d1e1ed12e0eca17b7fc574", 2002),
        "w3b_extract_faults": ("e3dcefcac508b53500cfcedc", 24),
        "w3c_magic_pass_external_fault_slots": ("4d9e0bcf06ba10dc59cf8aea", 2949),
        "w3d_magic_pass_selector": ("46df88fddbbd8a637d26d8a8", 5031),
        "w4_marginal_builder": ("5e011bfacca442c9f7f2b1f3", 728),
        "w4b_marginal_builder_oracle": ("b8068e7e8656c4e94c7f6061", 758),
        "w4c_marginal_builder_abi": ("10a426c821b0a3e169d55f2f", 671),
        "w4d_marginal_builder_uc_false": ("b8068e7e8656c4e94c7f6061", 758),
        "w5_sync_reference": ("dc061b1593c3268e0dcd2823", 21),
        "w5b_sync_reference_nohandler": ("dc061b1593c3268e0dcd2823", 21),
        "w6_known_states": ("a70f5667f4e83b9d9ef0d64f", 25),
        "w7_strip": ("d0febc1923f568a5ab9d4c5c", 29),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_MRY": {
        "w10_unroll": ("3141262ac1ebb7ef8dff8303", 25),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("749bdb67b4e16d60c67e3f96", 471),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("9f6184b09ff9f9e587b3e53f", 139),
        "w1_tableau_precompile": ("fe0d455808672137ead6d4b4", 484),
        "w1b_tableau_precompile_ucu_false": ("cfa2ae38b3dd964e04dedfce", 217),
        "w1c_tableau_precompile_full": ("81f67d7d1f2ad637c037c639", 849),
        "w2_coset_reference": ("56822ebf8876dc279018e265", 1283),
        "w3_magic_pass": ("99f81a9522c4d78690f110bc", 2008),
        "w3b_extract_faults": ("eb1a97a6f41bf8fec791a92d", 24),
        "w3c_magic_pass_external_fault_slots": ("d7efa9ca2aad38b48984defb", 2955),
        "w3d_magic_pass_selector": ("bcd445f5b44d9c18275d68ec", 5055),
        "w4_marginal_builder": ("5e011bfacca442c9f7f2b1f3", 728),
        "w4b_marginal_builder_oracle": ("b8068e7e8656c4e94c7f6061", 758),
        "w4c_marginal_builder_abi": ("7a3fa3061c74ed36fb658112", 671),
        "w4d_marginal_builder_uc_false": ("b8068e7e8656c4e94c7f6061", 758),
        "w5_sync_reference": ("dc061b1593c3268e0dcd2823", 21),
        "w5b_sync_reference_nohandler": ("dc061b1593c3268e0dcd2823", 21),
        "w6_known_states": ("497f162cf1b8512de38d3b59", 25),
        "w7_strip": ("70124b11b6186bcdc74f360d", 29),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_MX": {
        "w10_unroll": ("6b812b10587b78ed56204097", 24),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("847d377d43587bf13a8a67f7", 454),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("2da32a3e54a316ce0ebd85ee", 139),
        "w1_tableau_precompile": ("05a0ecc2d50a81cac7a42ebe", 480),
        "w1b_tableau_precompile_ucu_false": ("a2d65a4976c439febca6e735", 216),
        "w1c_tableau_precompile_full": ("e9f35277905482ac2bdcef8f", 845),
        "w2_coset_reference": ("0c566aecfa22aec121737a10", 1268),
        "w3_magic_pass": ("b75c3f5b316f15a65e674b9e", 1798),
        "w3b_extract_faults": ("da7eeb33d6fcc5fa676a24c4", 23),
        "w3c_magic_pass_external_fault_slots": ("027bfa5175b69be28cb9087b", 2745),
        "w3d_magic_pass_selector": ("bb32891ec983ea20c69e3085", 4447),
        "w4_marginal_builder": ("5e011bfacca442c9f7f2b1f3", 728),
        "w4b_marginal_builder_oracle": ("b8068e7e8656c4e94c7f6061", 758),
        "w4c_marginal_builder_abi": ("47bd05ef7f5e094c3a22a76b", 671),
        "w4d_marginal_builder_uc_false": ("b8068e7e8656c4e94c7f6061", 758),
        "w5_sync_reference": ("dc061b1593c3268e0dcd2823", 21),
        "w5b_sync_reference_nohandler": ("dc061b1593c3268e0dcd2823", 21),
        "w6_known_states": ("9bd9bdf166b75c25c2e35d16", 25),
        "w7_strip": ("81e687ac9036f145e6c58f1b", 28),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_MXX": {
        "w10_unroll": ("0d587af29b2cb7bb16d14854", 27),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("70373ab48398af8c41ad6e4c", 469),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("072f97f1cf6253245e80816e", 137),
        "w1_tableau_precompile": ("96dd5ab0d7865cd184b336ef", 490),
        "w1b_tableau_precompile_ucu_false": ("7ffbfc9941915a8726b8900f", 219),
        "w1c_tableau_precompile_full": ("4616240b9c282f0fb9a4547c", 853),
        "w2_coset_reference": ("590737ae05e8fbadd517b5f7", 1294),
        "w3_magic_pass": ("11f50318e020142f9ec9d3f9", 1798),
        "w3b_extract_faults": ("58aff00c602f910c4cddfbc0", 26),
        "w3c_magic_pass_external_fault_slots": ("da60c7c38575412e3379c5f9", 2745),
        "w3d_magic_pass_selector": ("632576b1c90b889df7a30927", 4447),
        "w4_marginal_builder": ("5e011bfacca442c9f7f2b1f3", 728),
        "w4b_marginal_builder_oracle": ("b8068e7e8656c4e94c7f6061", 758),
        "w4c_marginal_builder_abi": ("08147da0b5d94c459da5d76c", 675),
        "w4d_marginal_builder_uc_false": ("b8068e7e8656c4e94c7f6061", 758),
        "w5_sync_reference": ("dc061b1593c3268e0dcd2823", 21),
        "w5b_sync_reference_nohandler": ("dc061b1593c3268e0dcd2823", 21),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("8e33b4d5621d7e77a841c5ce", 31),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_MY": {
        "w10_unroll": ("cc31df13ba3d0452458842a3", 24),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("56c12a97cc723caa09d895df", 454),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("9f6184b09ff9f9e587b3e53f", 139),
        "w1_tableau_precompile": ("a4c72048c61679a3eb36aade", 480),
        "w1b_tableau_precompile_ucu_false": ("8e5d9eea1b82f6c6b26a8245", 216),
        "w1c_tableau_precompile_full": ("d842acd990a793912ecea8c4", 845),
        "w2_coset_reference": ("af48e0121da863f1e3313039", 1282),
        "w3_magic_pass": ("95252e0b42bcea029e895af1", 1798),
        "w3b_extract_faults": ("17632d9c96789492e333a898", 23),
        "w3c_magic_pass_external_fault_slots": ("b80009ae023c255f1a92230e", 2745),
        "w3d_magic_pass_selector": ("cf80e45a511cca5805659684", 4447),
        "w4_marginal_builder": ("5e011bfacca442c9f7f2b1f3", 728),
        "w4b_marginal_builder_oracle": ("b8068e7e8656c4e94c7f6061", 758),
        "w4c_marginal_builder_abi": ("47bd05ef7f5e094c3a22a76b", 671),
        "w4d_marginal_builder_uc_false": ("b8068e7e8656c4e94c7f6061", 758),
        "w5_sync_reference": ("dc061b1593c3268e0dcd2823", 21),
        "w5b_sync_reference_nohandler": ("dc061b1593c3268e0dcd2823", 21),
        "w6_known_states": ("9bd9bdf166b75c25c2e35d16", 25),
        "w7_strip": ("a156eb1fa005cabb1f0b5330", 28),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_MYY": {
        "w10_unroll": ("b713a53232b39548bf0a2400", 27),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("c4064ca01d791fe314e463b7", 469),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("0ea999c15e7a81bab78be320", 137),
        "w1_tableau_precompile": ("3e16a4841f9cabec42ddd8a8", 490),
        "w1b_tableau_precompile_ucu_false": ("b0994222253c43e51df7e798", 219),
        "w1c_tableau_precompile_full": ("9989aa4d9c0aacb61d6b49a4", 853),
        "w2_coset_reference": ("89a11c161716739b9edb0d8e", 1316),
        "w3_magic_pass": ("f3559ddd51f827a98e06ed0a", 1798),
        "w3b_extract_faults": ("16d35ea78106c02c3d7e2f9b", 26),
        "w3c_magic_pass_external_fault_slots": ("06a178a2c8e245e4fb505d3c", 2745),
        "w3d_magic_pass_selector": ("3d1d9048e6dff3877f27608a", 4447),
        "w4_marginal_builder": ("5e011bfacca442c9f7f2b1f3", 728),
        "w4b_marginal_builder_oracle": ("b8068e7e8656c4e94c7f6061", 758),
        "w4c_marginal_builder_abi": ("08147da0b5d94c459da5d76c", 675),
        "w4d_marginal_builder_uc_false": ("b8068e7e8656c4e94c7f6061", 758),
        "w5_sync_reference": ("dc061b1593c3268e0dcd2823", 21),
        "w5b_sync_reference_nohandler": ("dc061b1593c3268e0dcd2823", 21),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("5a1ae5eb14d701f729719c13", 31),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_MZZ": {
        "w10_unroll": ("c9c49dea8c829ca8ac2e16e6", 27),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("5ae9c3e9b09a95f7b210ce21", 469),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("19c5259f693e2bb2ddfba3d3", 137),
        "w1_tableau_precompile": ("035dfc79ef54c1504fa540d2", 490),
        "w1b_tableau_precompile_ucu_false": ("cf0d3d782d80453938a5199f", 219),
        "w1c_tableau_precompile_full": ("71131ad600209b54da0b185c", 853),
        "w2_coset_reference": ("224e1573ec8178a37d6a08ea", 1147),
        "w3_magic_pass": ("0a7a6f6b4c450550f5f75b4e", 1609),
        "w3b_extract_faults": ("e708d13e829535b940b69b9e", 26),
        "w3c_magic_pass_external_fault_slots": ("d77762c2565da87ff5340897", 2556),
        "w3d_magic_pass_selector": ("ec9f75a1eff306e65c8d3680", 3673),
        "w4_marginal_builder": ("5e011bfacca442c9f7f2b1f3", 728),
        "w4b_marginal_builder_oracle": ("b8068e7e8656c4e94c7f6061", 758),
        "w4c_marginal_builder_abi": ("08147da0b5d94c459da5d76c", 675),
        "w4d_marginal_builder_uc_false": ("b8068e7e8656c4e94c7f6061", 758),
        "w5_sync_reference": ("dc061b1593c3268e0dcd2823", 21),
        "w5b_sync_reference_nohandler": ("dc061b1593c3268e0dcd2823", 21),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("db8fdef8c63cb0e8308054ff", 31),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_OBSERVABLE_INCLUDE": {
        "w10_unroll": ("844326a5cac98be57b342b5c", 55),
        "w11_u2f_and_swap": ("f082a425bca6de5a5839a87a", 51),
        "w12_ir_fields": ("4e2985c4b538996c4173afed", 693),
        "w12b_tableau_layout": ("93f03fc1521db48a8b10a28c", 57),
        "w12c_ir_shared_fields": ("e746dc09a533057f277245a8", 187),
        "w1_tableau_precompile": ("e6b8579e2aa2aedca17289eb", 540),
        "w1b_tableau_precompile_ucu_false": ("e46461f2cfd92ae093b8373d", 247),
        "w1c_tableau_precompile_full": ("a3a881e8d79e88c3169c3853", 841),
        "w2_coset_reference": ("b8a81c35d3799ca6def6c185", 1354),
        "w3_magic_pass": ("fa616b502b4e33c5d4ba804d", 1609),
        "w3b_extract_faults": ("3fee3ffcb5e342548e2483af", 52),
        "w3c_magic_pass_external_fault_slots": ("559075a2536ce3332c546b18", 2556),
        "w3d_magic_pass_selector": ("8fb29d9a207dd6efa6694623", 3673),
        "w4_marginal_builder": ("dcf7e9d5cbd0abd47875f77d", 770),
        "w4b_marginal_builder_oracle": ("fd8e0cadc7f12182c510d118", 808),
        "w4c_marginal_builder_abi": ("9459994749028f98e4a5ec19", 719),
        "w4d_marginal_builder_uc_false": ("fd8e0cadc7f12182c510d118", 808),
        "w5_sync_reference": ("dc061b1593c3268e0dcd2823", 21),
        "w5b_sync_reference_nohandler": ("dc061b1593c3268e0dcd2823", 21),
        "w6_known_states": ("23db902f07fadcf04a8bdc3b", 28),
        "w7_strip": ("e7a241586b783b75260df15a", 58),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_PAULI_CHANNEL_1": {
        "w10_unroll": ("1b611cd9853bb0a8ac55fe73", 55),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("06a8f6a818acfadb5b51611d", 610),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("4b9b0a9706be621aa71c23ed", 141),
        "w1_tableau_precompile": ("53cf4fb49dd1908ebaca98da", 606),
        "w1b_tableau_precompile_ucu_false": ("b73d2e3b21088fcfb7f526eb", 247),
        "w1c_tableau_precompile_full": ("e43437014f6e7a19e29445b6", 1001),
        "w2_coset_reference": ("fecb3fca5590fedf8c7d91d0", 1028),
        "w3_magic_pass": ("960681fa829004e4943c223d", 1783),
        "w3b_extract_faults": ("28e6338033d56bb71a5aa3c3", 54),
        "w3c_magic_pass_external_fault_slots": ("91806ebaa76ecc2c58824cec", 2730),
        "w3d_magic_pass_selector": ("209507d14b4cde98d1327aa3", 4216),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("47f5f54aa4ee03a293b15c51", 671),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("499d779e6da186b2b15eb9e8", 22),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_PAULI_CHANNEL_2": {
        "w10_unroll": ("a9726062b15f950ebac50009", 129),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("89cabf52c3504b809887c4b3", 980),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("4b9b0a9706be621aa71c23ed", 141),
        "w1_tableau_precompile": ("cfb21ecd7859b73478254efa", 840),
        "w1b_tableau_precompile_ucu_false": ("426983092b08e60562957c6a", 321),
        "w1c_tableau_precompile_full": ("f348a0eb7c9083c5428d5988", 1293),
        "w2_coset_reference": ("0bd05f555c4ff1f25b47093b", 1102),
        "w3_magic_pass": ("767d40582b79b42054a4ce3a", 2864),
        "w3b_extract_faults": ("ecd97200debf0c5ea0635855", 128),
        "w3c_magic_pass_external_fault_slots": ("05a3e0371ee417d122c4facd", 3811),
        "w3d_magic_pass_selector": ("ddeb1e074607eb1147ea733f", 8690),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("b261c24b99dc21ce120f1761", 675),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("499d779e6da186b2b15eb9e8", 22),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_QUBIT_COORDS": {
        "w10_unroll": ("accb400065efa190ac8a9964", 40),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("a7e61a5360e38e3498a6926c", 538),
        "w12b_tableau_layout": ("d8ef921c63b4cff643496a53", 45),
        "w12c_ir_shared_fields": ("4b9b0a9706be621aa71c23ed", 141),
        "w1_tableau_precompile": ("7dd73a7a2ae43bb3d8a281d3", 422),
        "w1b_tableau_precompile_ucu_false": ("d33f6b97718dc92b220b858a", 172),
        "w1c_tableau_precompile_full": ("671cfc1ac2915ce517d66a15", 651),
        "w2_coset_reference": ("cd212aea05a4d97ecb945d48", 793),
        "w3_magic_pass": ("d68ddf2f0d36896ab958a674", 1435),
        "w3b_extract_faults": ("2d2b62b34f744198d5ba9146", 39),
        "w3c_magic_pass_external_fault_slots": ("93bae0aecededde0dd3c8948", 2382),
        "w3d_magic_pass_selector": ("ee4d390821731568d2d6e1cf", 3325),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("e6d61a7ca16083d30c1796ee", 667),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("740fb0d5a902de812c1ef9a2", 44),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_R": {
        "w10_unroll": ("a85d0156ffd0ac909be6e9a0", 23),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("576fe5f7c52a587a01b9bbb2", 445),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("b1c032899073a7b9b95d0f9b", 139),
        "w1_tableau_precompile": ("aa3703f22e632c6049a5a4d9", 476),
        "w1b_tableau_precompile_ucu_false": ("e717f89d2376dfe9a0c05a8d", 215),
        "w1c_tableau_precompile_full": ("c447ab222a73f906240ae1f5", 841),
        "w2_coset_reference": ("c4402695e43953947481b0f0", 1125),
        "w3_magic_pass": ("21168fb003db695cf1567593", 1728),
        "w3b_extract_faults": ("a9df2f2c30b0827edc3fdec7", 22),
        "w3c_magic_pass_external_fault_slots": ("c7b52369d7dcd55d38a5f002", 2675),
        "w3d_magic_pass_selector": ("ea58b092cfc08ce7bf75140b", 3911),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("676d7347f038a4c13dba009a", 671),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("6e3a1359453ed9c637f5bab2", 27),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_RX": {
        "w10_unroll": ("d72a6ebbc2d7f5c43d7ac6b3", 24),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("772b0c06978ae9df0940d6db", 450),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("2da32a3e54a316ce0ebd85ee", 139),
        "w1_tableau_precompile": ("858371309cd40cffb31e02fa", 480),
        "w1b_tableau_precompile_ucu_false": ("36d3664eabbb91649ac5ac9c", 216),
        "w1c_tableau_precompile_full": ("88e9ace2140b27ec346207ec", 845),
        "w2_coset_reference": ("c104be15e14878c2d37372fe", 1268),
        "w3_magic_pass": ("f790c6fc78cb7219e68d8fd7", 2005),
        "w3b_extract_faults": ("eed3fa99c0b3ed26dd222539", 23),
        "w3c_magic_pass_external_fault_slots": ("74aa683f3ca9b7d29b72ac05", 2952),
        "w3d_magic_pass_selector": ("b03630c954aa191c7e846f6a", 5037),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("883e319e88d389f1f52beae3", 671),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("a70f5667f4e83b9d9ef0d64f", 25),
        "w7_strip": ("3b752a1e3be669be540cb43b", 28),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_RY": {
        "w10_unroll": ("35cbf1fde177505d8ddee7de", 24),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("14584444d3b8040d1e9d53f7", 450),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("9f6184b09ff9f9e587b3e53f", 139),
        "w1_tableau_precompile": ("9d9bec34d467ba868de69a9e", 480),
        "w1b_tableau_precompile_ucu_false": ("25414aaaa264fd49b0fb96e4", 216),
        "w1c_tableau_precompile_full": ("5b2189f63ce82a23afbf0eb6", 845),
        "w2_coset_reference": ("63d731aeb6fa9e4e963191ca", 1282),
        "w3_magic_pass": ("64d864a6f6c0647b16db328e", 2011),
        "w3b_extract_faults": ("1a65c210fefe239efcdc0218", 23),
        "w3c_magic_pass_external_fault_slots": ("36e617ee9b800e196e8241c0", 2958),
        "w3d_magic_pass_selector": ("4773eb2ef1add69df90fb74c", 5061),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("d24460a9a294b2138fd735ca", 671),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("497f162cf1b8512de38d3b59", 25),
        "w7_strip": ("8bd98443ec863317d3af51c2", 28),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_S": {
        "w10_unroll": ("96a0dc6f5df5aa7701912c04", 23),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("7ef27f3e91a5bfcb76e19aeb", 454),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("5612f9c27404afbbf649c195", 496),
        "w1b_tableau_precompile_ucu_false": ("2615515967fbfedd679a4159", 215),
        "w1c_tableau_precompile_full": ("6a405ef818e90e2f8d74fb92", 881),
        "w2_coset_reference": ("a5b0d184969247351fa606ae", 1025),
        "w3_magic_pass": ("f6d00985212f98db3b4e7370", 1479),
        "w3b_extract_faults": ("199b2184460a5faf60101e77", 22),
        "w3c_magic_pass_external_fault_slots": ("c67e6195194dc902976ea456", 2426),
        "w3d_magic_pass_selector": ("1914a72bee2ca058ee8a0cb8", 3501),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("7e70cc60186ded04c5d6546b", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("25b5a10b255a9b2d0e88454b", 25),
        "w7_strip": ("192f3eadef72aa8a3a976d45", 27),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_SHIFT_COORDS": {
        "w10_unroll": ("6427d2b29aae14d7b83a628c", 35),
        "w11_u2f_and_swap": ("cab25a8860c44136c7051c4b", 43),
        "w12_ir_fields": ("450b65a0ac909d54f4be37ab", 457),
        "w12b_tableau_layout": ("d8ef921c63b4cff643496a53", 45),
        "w12c_ir_shared_fields": ("4b9b0a9706be621aa71c23ed", 141),
        "w1_tableau_precompile": ("6190ae9e7b050d9fc2e83901", 400),
        "w1b_tableau_precompile_ucu_false": ("f9abaf48369a65e0f5eb9c22", 167),
        "w1c_tableau_precompile_full": ("671cfc1ac2915ce517d66a15", 651),
        "w2_coset_reference": ("cd212aea05a4d97ecb945d48", 793),
        "w3_magic_pass": ("d68ddf2f0d36896ab958a674", 1435),
        "w3b_extract_faults": ("f8b71f7236c3616d73ad4703", 34),
        "w3c_magic_pass_external_fault_slots": ("93bae0aecededde0dd3c8948", 2382),
        "w3d_magic_pass_selector": ("ee4d390821731568d2d6e1cf", 3325),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("5f08c0cc2ac2569b40e95b97", 758),
        "w4c_marginal_builder_abi": ("dfbc65818c56e1c4907feaff", 619),
        "w4d_marginal_builder_uc_false": ("5f08c0cc2ac2569b40e95b97", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("a274d62148b420d0f3b40db7", 39),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_SPP": {
        "w10_unroll": ("c2018c26e539bc646e057115", 29),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("c68605ed3c8d9016eaa31633", 486),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("4b9b0a9706be621aa71c23ed", 141),
        "w1_tableau_precompile": ("65ce15028368d8a681d86596", 540),
        "w1b_tableau_precompile_ucu_false": ("7addaa4708be586dd922b45d", 221),
        "w1c_tableau_precompile_full": ("67893c2cf3e37bad04681182", 945),
        "w2_coset_reference": ("4ee56c0992997d6fec7e3b1d", -1),
        "w3_magic_pass": ("391f9d235712956b9f535b8c", 1496),
        "w3b_extract_faults": ("73f73aa58fcaca33dd63accd", 28),
        "w3c_magic_pass_external_fault_slots": ("69d986d35ff09344ed906d6f", 2443),
        "w3d_magic_pass_selector": ("e3a8fda1074d57cc1eea488c", 3569),
        "w4_marginal_builder": ("df7bc772e9482d1c9a4c3088", -1),
        "w4b_marginal_builder_oracle": ("df7bc772e9482d1c9a4c3088", -1),
        "w4c_marginal_builder_abi": ("df7bc772e9482d1c9a4c3088", -1),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("f5a5efa48ef416c344ad3dba", 33),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_SPP_DAG": {
        "w10_unroll": ("27b1d35a47aa2ae21527efbf", 33),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("c786abd10e4c54e2f123e76a", 506),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("4b9b0a9706be621aa71c23ed", 141),
        "w1_tableau_precompile": ("704bef68d0e3a77d298e4479", 556),
        "w1b_tableau_precompile_ucu_false": ("81795b92692de19ad5b8ed20", 225),
        "w1c_tableau_precompile_full": ("956ed453c6a3bdf9885fa7ad", 961),
        "w2_coset_reference": ("71326d5565a3b567b748d90b", -1),
        "w3_magic_pass": ("64763b458df658de32191ad7", 1495),
        "w3b_extract_faults": ("6e079f5a632bdc7e54dc8f1b", 32),
        "w3c_magic_pass_external_fault_slots": ("d1fa74885f020a31482612a6", 2442),
        "w3d_magic_pass_selector": ("da8d23971e696e52eeff4b33", 3565),
        "w4_marginal_builder": ("fc640ba20247e7fe7d999a8a", -1),
        "w4b_marginal_builder_oracle": ("fc640ba20247e7fe7d999a8a", -1),
        "w4c_marginal_builder_abi": ("fc640ba20247e7fe7d999a8a", -1),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("e2362945f90f1ca96daaee1c", 37),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_SQRT_X": {
        "w10_unroll": ("9a6a14a448939ebd7f7f7153", 28),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("d183f383f4e7109f2a548807", 479),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("f695b8a4c3d1035c42c62157", 516),
        "w1b_tableau_precompile_ucu_false": ("e73eef4c684d30fd45078f3e", 220),
        "w1c_tableau_precompile_full": ("49eda88db8b93d154f385398", 901),
        "w2_coset_reference": ("2ac10d88f4038833f1bc5171", 1040),
        "w3_magic_pass": ("a9efeda18c21cdc1e41ff070", 1484),
        "w3b_extract_faults": ("db69295b4233a2efbfcdc6ee", 27),
        "w3c_magic_pass_external_fault_slots": ("3443c582c1f22ea9375b0888", 2431),
        "w3d_magic_pass_selector": ("c86be08235f36cd9d1bf936b", 3521),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("66dbb4dd768517bc144dc11d", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("260963edcf06d931555b21b4", 25),
        "w7_strip": ("ab20444451cdf08572f78814", 32),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_SQRT_XX": {
        "w10_unroll": ("9f81d2ce54ad4846b552aa12", 31),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("6432a28d2aec5568a9008f4d", 494),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("909af96307890106662ba7bc", 538),
        "w1b_tableau_precompile_ucu_false": ("9dc972675f07983001994862", 223),
        "w1c_tableau_precompile_full": ("c4935610ee8130ad250d3ae4", 933),
        "w2_coset_reference": ("e3fab58a7ebbfef6cdc5afb3", 1049),
        "w3_magic_pass": ("52e51f4b970ec0c96dcf2b6a", 1487),
        "w3b_extract_faults": ("6ef764705e411f74fdfbfc54", 30),
        "w3c_magic_pass_external_fault_slots": ("ada6547fa90c8096f01ae37c", 2434),
        "w3d_magic_pass_selector": ("91f9b9dcefcc05ccd33e0567", 3533),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("acac37ab7d0de78b974eef4f", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("c2e373e156b3b952293a9419", 35),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_SQRT_XX_DAG": {
        "w10_unroll": ("dcaead1c6352986f94bc95ec", 35),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("8a01b0acc2f3974986f7b706", 514),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("6b348d5fb36db6c50ef79b07", 554),
        "w1b_tableau_precompile_ucu_false": ("d26f33eb97d38cf545114117", 227),
        "w1c_tableau_precompile_full": ("8e356081e88d6283624ad951", 949),
        "w2_coset_reference": ("90fbe6b69567088e4b8621ad", 1061),
        "w3_magic_pass": ("2b41b7dcdea703198a6c8651", 1491),
        "w3b_extract_faults": ("417af8e5fcdbbfa756391458", 34),
        "w3c_magic_pass_external_fault_slots": ("13207a5d89066be04cfd5389", 2438),
        "w3d_magic_pass_selector": ("9aa2681094a8f7d0d537471c", 3549),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("acac37ab7d0de78b974eef4f", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("65d40bd476c2de914fdb2320", 39),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_SQRT_X_DAG": {
        "w10_unroll": ("795170ec4ef721c2af597870", 32),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("14cb53b6f2510d658634270e", 499),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("82177463a8937faba96e90c0", 532),
        "w1b_tableau_precompile_ucu_false": ("45caa77e83474ead82ecb9ba", 224),
        "w1c_tableau_precompile_full": ("22be4d7d0aae88c21d98efc2", 917),
        "w2_coset_reference": ("97a720f56e256c0463a637a6", 1052),
        "w3_magic_pass": ("add12e4444ded3ae92312f1b", 1488),
        "w3b_extract_faults": ("8b669481613f1eef0df353bd", 31),
        "w3c_magic_pass_external_fault_slots": ("ca1899eedcbc30ae3023127f", 2435),
        "w3d_magic_pass_selector": ("2e39ceafb16832226962ee27", 3537),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("5f6bd7fb2b199b69bbed21ba", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("497f162cf1b8512de38d3b59", 25),
        "w7_strip": ("793c8e78c0179f14cf7414c6", 36),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_SQRT_Y": {
        "w10_unroll": ("12246e0a8715e9274ef5bdca", 28),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("fe3c4996b0689e7bb4c7a178", 479),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("19ec10124c5e8e00b06233ab", 516),
        "w1b_tableau_precompile_ucu_false": ("530cf9361a48edd8da5a389d", 220),
        "w1c_tableau_precompile_full": ("579a0ab63d9abb73d75cacf6", 901),
        "w2_coset_reference": ("82f86b41c8910c5fafbcf1b5", 1040),
        "w3_magic_pass": ("f3664fc153d0bacc0c3fc46d", 1484),
        "w3b_extract_faults": ("acaab262a54ee57d73587229", 27),
        "w3c_magic_pass_external_fault_slots": ("c10941a380d7462a3ff26888", 2431),
        "w3d_magic_pass_selector": ("e4e4ccabd378cb9205b95b7f", 3521),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("4213faf7ea9851ff73ea416a", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("a70f5667f4e83b9d9ef0d64f", 25),
        "w7_strip": ("e45f04360add6c4a4cf6fa18", 32),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_SQRT_YY": {
        "w10_unroll": ("ccc59eba2381e3482805db66", 31),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("e27a9e73ab697ec01fd95f16", 494),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("b9da6c74c9d47561310bcff7", 538),
        "w1b_tableau_precompile_ucu_false": ("c56eb7e5cb17142e7c71b3bb", 223),
        "w1c_tableau_precompile_full": ("371a325c865bfa53a2893d5d", 933),
        "w2_coset_reference": ("3f898fb9d7f7fbb3bce4c272", 1049),
        "w3_magic_pass": ("7fba29733e27c081963d4a71", 1487),
        "w3b_extract_faults": ("bbbf8f67d0c88c05ecc062fb", 30),
        "w3c_magic_pass_external_fault_slots": ("a7e489075b80a2dab833bb89", 2434),
        "w3d_magic_pass_selector": ("27b95311c7678b438c694269", 3533),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("acac37ab7d0de78b974eef4f", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("ab75a14f024feb247d22bf92", 35),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_SQRT_YY_DAG": {
        "w10_unroll": ("f83f96ba3f98fc600803bdc4", 35),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("fc6848ece708c9e1de00a484", 514),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("605a7d39fe6c7f71ff4c4374", 554),
        "w1b_tableau_precompile_ucu_false": ("6f418a5dcc74c85b51733494", 227),
        "w1c_tableau_precompile_full": ("0cce9a8d4399b59578e0ac37", 949),
        "w2_coset_reference": ("369804468e4f11295180126a", 1061),
        "w3_magic_pass": ("1c0e3d6cf2a26f4bfb2dbd54", 1491),
        "w3b_extract_faults": ("9d9796ed4fef9e9756db179b", 34),
        "w3c_magic_pass_external_fault_slots": ("89f475eb7023d481fd25c594", 2438),
        "w3d_magic_pass_selector": ("1f01064cd5410f10b7a1f307", 3549),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("acac37ab7d0de78b974eef4f", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("f27e921fe691d080de9590d1", 39),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_SQRT_Y_DAG": {
        "w10_unroll": ("d71fd6e826f15aa45c6a9e0d", 32),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("4cd522767c5a416aeb78af4e", 499),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("7a31e76b27e27faac4f2ac92", 532),
        "w1b_tableau_precompile_ucu_false": ("6ee73c69217135251d7401ba", 224),
        "w1c_tableau_precompile_full": ("2888ba0d14de5f457c8ba8be", 917),
        "w2_coset_reference": ("7a0d3f2939f8d1ab257d8dce", 1052),
        "w3_magic_pass": ("1ad982b1fadc613e52149ed9", 1488),
        "w3b_extract_faults": ("275ec6e256240a6714cb2a2b", 31),
        "w3c_magic_pass_external_fault_slots": ("c99e976530cef68610f0b0f6", 2435),
        "w3d_magic_pass_selector": ("383c72a8d4ad6fe4799fb0e1", 3537),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("37a40f8117dd81a76f88c486", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("acc94fd1206822d79af544bc", 25),
        "w7_strip": ("ae920c06d24218af4709b88e", 36),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_SQRT_ZZ": {
        "w10_unroll": ("aaeba3a4d09f2a98dba51f1d", 31),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("2082dff4040ee9ac68bba6d4", 494),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("a494a069fa89a5684b9165e8", 538),
        "w1b_tableau_precompile_ucu_false": ("0d8b5691c508e463ec4b3097", 223),
        "w1c_tableau_precompile_full": ("0906440570c1f4b83a4b11ef", 933),
        "w2_coset_reference": ("c824846c5ccc2ccb5dd8fa03", 1049),
        "w3_magic_pass": ("756ae8d991b1c1df66db0a15", 1487),
        "w3b_extract_faults": ("3ad54d784df3687b47729862", 30),
        "w3c_magic_pass_external_fault_slots": ("ae43bf3860a9235f297843ee", 2434),
        "w3d_magic_pass_selector": ("327b28094773347ef34ff350", 3533),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("acac37ab7d0de78b974eef4f", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("7125aea72283c1b71b308d1e", 35),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_SQRT_ZZ_DAG": {
        "w10_unroll": ("efb5882db4a61770d0e0fa54", 35),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("25ab565012e8e11d6f138459", 514),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("d85e8f390d50e92a6cf9002f", 554),
        "w1b_tableau_precompile_ucu_false": ("c42f750b63882c26cab7fbd4", 227),
        "w1c_tableau_precompile_full": ("c556236605bd0a2dc3f3f9b8", 949),
        "w2_coset_reference": ("36d93756560f55ba1a3f756a", 1061),
        "w3_magic_pass": ("3dfa0fdab04f6c2aaffc121a", 1491),
        "w3b_extract_faults": ("8f81833383d0491c558f6b4a", 34),
        "w3c_magic_pass_external_fault_slots": ("ef97a0222a0718f95c5f4345", 2438),
        "w3d_magic_pass_selector": ("ddcd5ac740ab2f6553ab83fa", 3549),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("acac37ab7d0de78b974eef4f", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("c5aaf831d71738d85ecc12b9", 39),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_SWAP": {
        "w10_unroll": ("14bcf102c2fdf8ea6d9fe72f", 28),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("9ca07f37ba69c015ff0a5643", 479),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("7da4cded179edcb9fe369e63", 526),
        "w1b_tableau_precompile_ucu_false": ("68b036299d82653d71317130", 220),
        "w1c_tableau_precompile_full": ("9840a827f0be7d8a79291bf8", 921),
        "w2_coset_reference": ("2317ec98dc67462205503168", 1040),
        "w3_magic_pass": ("ad1df41c1badf3ad92b6387f", 1484),
        "w3b_extract_faults": ("614719f15f2d95783b784502", 27),
        "w3c_magic_pass_external_fault_slots": ("a228fbcbee1974d86bcf01ff", 2431),
        "w3d_magic_pass_selector": ("b726ea85fb21b20ad40cdf85", 3521),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("acac37ab7d0de78b974eef4f", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("7065da02d3eed8b1240281e0", 32),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_SWAPCX": {
        "w10_unroll": ("1465cddd0984623e74f5bdfe", 30),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("1efb3db72b3d91b0c68c5f95", 489),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("b1f14a7c46751bc6a435c280", 534),
        "w1b_tableau_precompile_ucu_false": ("55b273bb5cf2a0df852d13e1", 222),
        "w1c_tableau_precompile_full": ("a2eb6f9ce721365068cfe025", 929),
        "w2_coset_reference": ("6c5f5f5d8e41f46a783c05b8", 1046),
        "w3_magic_pass": ("694f174f68444a39d317058b", 1486),
        "w3b_extract_faults": ("8c42a5e4e2092ce00c32385f", 29),
        "w3c_magic_pass_external_fault_slots": ("82d66da93296d847c71bdd89", 2433),
        "w3d_magic_pass_selector": ("b2643083eb5c7d4b9ec1d342", 3529),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("acac37ab7d0de78b974eef4f", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("7851b8444c95f3e0bcd9df96", 34),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_S_DAG": {
        "w10_unroll": ("58b63c07f09ddf592c043c85", 27),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("a061531ec4301c1700f255d4", 474),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("e6e6299640d3df854df4575e", 512),
        "w1b_tableau_precompile_ucu_false": ("eeb5f25dbbc66a1a2a762ce3", 219),
        "w1c_tableau_precompile_full": ("9a01aa560b34e6a33281f862", 897),
        "w2_coset_reference": ("a5c7bbcd8df21bcbdbd7c4a9", 1037),
        "w3_magic_pass": ("53fb5b7e46ad8fd7424e913e", 1483),
        "w3b_extract_faults": ("0bb77096a22a082e0dac5101", 26),
        "w3c_magic_pass_external_fault_slots": ("9e1c8c76c62e94a64cd237e6", 2430),
        "w3d_magic_pass_selector": ("aa20a0aad2c1554fbbe7da30", 3517),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("33b53bc0e0d7cae4b4d90664", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("0790e30ab02157859d139809", 31),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_TICK": {
        "w10_unroll": ("afc47bd130d7bc3c17b56e19", 24),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("a0e7109fab1802836e318232", 459),
        "w12b_tableau_layout": ("d8ef921c63b4cff643496a53", 45),
        "w12c_ir_shared_fields": ("4b9b0a9706be621aa71c23ed", 141),
        "w1_tableau_precompile": ("875882c96435b8cf56e627ae", 375),
        "w1b_tableau_precompile_ucu_false": ("af4d77b48474569096c55af2", 156),
        "w1c_tableau_precompile_full": ("671cfc1ac2915ce517d66a15", 651),
        "w2_coset_reference": ("893ee8c971f3c229c42d7c48", 997),
        "w3_magic_pass": ("d68ddf2f0d36896ab958a674", 1435),
        "w3b_extract_faults": ("4dc7026e8f03a2cc4154e849", 23),
        "w3c_magic_pass_external_fault_slots": ("93bae0aecededde0dd3c8948", 2382),
        "w3d_magic_pass_selector": ("ee4d390821731568d2d6e1cf", 3325),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("e6d61a7ca16083d30c1796ee", 667),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("3e64539b7794fd6a7964fbcc", 28),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_X": {
        "w10_unroll": ("926a8eddab240350732a0ffe", 23),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("f08c7d2ae66a3bb47125a8b0", 454),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("7c77f693f46cc8be118aadf3", 496),
        "w1b_tableau_precompile_ucu_false": ("50cfb64b6d7b5e10ff519bf5", 215),
        "w1c_tableau_precompile_full": ("0a758b8f6e9e2f4258270397", 881),
        "w2_coset_reference": ("dfe71d456dbb6d9b29b518ef", 1025),
        "w3_magic_pass": ("bc855b44f7af622e392a6ff3", 1479),
        "w3b_extract_faults": ("a50a3e8da45e5e0d447462ba", 22),
        "w3c_magic_pass_external_fault_slots": ("b22355b2d6f192df75da5d34", 2426),
        "w3d_magic_pass_selector": ("b242c95527c3f9a1f1e76ccb", 3501),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("cdec12da6fd0d38cb5188ab9", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("ce18155754c8b46c3ada63b5", 25),
        "w7_strip": ("e4dcc1b6b9adae90a22c1b11", 27),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_XCX": {
        "w10_unroll": ("4f7a496a383d02209e55a96e", 27),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("a35602bf6ecbbdaa52d44c35", 474),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("e8b6cbc2e66d1d80ca42b341", 522),
        "w1b_tableau_precompile_ucu_false": ("fa39f8388cbef810f8f17f53", 219),
        "w1c_tableau_precompile_full": ("1c6db75676e41f286a809093", 917),
        "w2_coset_reference": ("3eacfe7a54002a5b63175536", 1037),
        "w3_magic_pass": ("dbbf512abbf23e2c3a4c45e6", 1483),
        "w3b_extract_faults": ("60ca36091bc6f31ec5e78e87", 26),
        "w3c_magic_pass_external_fault_slots": ("471193aba2e7298f5321c307", 2430),
        "w3d_magic_pass_selector": ("d257d96928068ddcd7d184c5", 3517),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("6fbc5c5ea2af16631f340993", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("4670419eda61fe4a6c21a360", 31),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_XCY": {
        "w10_unroll": ("96fae38928e0423a27f7309e", 27),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("1fe19004cc0c01d5eb79ad39", 474),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("c08783ca15b6658cd8c03085", 522),
        "w1b_tableau_precompile_ucu_false": ("1fd97146a10e6d873c740e41", 219),
        "w1c_tableau_precompile_full": ("48523953d593c1bf530eb62a", 917),
        "w2_coset_reference": ("802e2a79274191463cc2c905", 1037),
        "w3_magic_pass": ("b162261f3161544d74148b48", 1483),
        "w3b_extract_faults": ("0655593113484080c3d1f491", 26),
        "w3c_magic_pass_external_fault_slots": ("761e535d7bb58b76269412b2", 2430),
        "w3d_magic_pass_selector": ("9dc45293aa64aaa4b6635907", 3517),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("e031fad1b94972e8425dd436", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("4cb60c9d6fdbb4d3c25abf85", 31),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_XCZ": {
        "w10_unroll": ("caf907ec236eddf20760fb61", 27),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("63ac167a1fcb728719d0f613", 474),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("9fffa276c073b2cc7f149b71", 522),
        "w1b_tableau_precompile_ucu_false": ("e02f7f156f7d2011d833c1e6", 219),
        "w1c_tableau_precompile_full": ("52bb15808defe3b472883c6b", 917),
        "w2_coset_reference": ("17a62008b45f5301d4aabe75", 1037),
        "w3_magic_pass": ("c621b1f321bfcc2e1167cba0", 1483),
        "w3b_extract_faults": ("dc1fd741ccea72fec2ebf893", 26),
        "w3c_magic_pass_external_fault_slots": ("73b0678191c5f4540bef758f", 2430),
        "w3d_magic_pass_selector": ("6174bb8397bbadf26104a205", 3517),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("36bc1a9f405c1dc129254710", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("67505d0c98af89b41f6d0d95", 31),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_X_ERROR": {
        "w10_unroll": ("5be58f3cf7f5e1d2657e28c6", 35),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("7706905e04004998f448ec1e", 510),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("4b9b0a9706be621aa71c23ed", 141),
        "w1_tableau_precompile": ("5047147445373b607fa61ea1", 538),
        "w1b_tableau_precompile_ucu_false": ("2ec0d1e578b217d230e2c43f", 227),
        "w1c_tableau_precompile_full": ("4a3abf8b3e3c2e415ff4a43c", 925),
        "w2_coset_reference": ("497e4752c1bcbdcf355d7685", 1008),
        "w3_magic_pass": ("5263a78d7530a58549b78615", 1595),
        "w3b_extract_faults": ("cb4ccc49d7c98e14c2e9d134", 34),
        "w3c_magic_pass_external_fault_slots": ("d29cc19a792bd10e84ad9d42", 2542),
        "w3d_magic_pass_selector": ("467c6534d80a5b588114f56f", 3674),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("47f5f54aa4ee03a293b15c51", 671),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("499d779e6da186b2b15eb9e8", 22),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_Y": {
        "w10_unroll": ("0cb2b3dc6deb44d5b45b8903", 23),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("8789a91f993d94bde3ff7c29", 454),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("c2a5dc7307996558e4b2417b", 496),
        "w1b_tableau_precompile_ucu_false": ("a2109f29d7d9f34611c8c2a8", 215),
        "w1c_tableau_precompile_full": ("292125afacce9db114a51aa2", 881),
        "w2_coset_reference": ("3ad2f0c6aa200588ad762aeb", 1025),
        "w3_magic_pass": ("4434e1fabbb0839169d04988", 1479),
        "w3b_extract_faults": ("5647c70754e31f7b693226ee", 22),
        "w3c_magic_pass_external_fault_slots": ("e1afa91ad15ff267b6d03a63", 2426),
        "w3d_magic_pass_selector": ("1722c7a6f181c53a08f6aabf", 3501),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("4f5235cf781568437805cced", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("ce18155754c8b46c3ada63b5", 25),
        "w7_strip": ("48c3bb0e3f1ce6b71de337a0", 27),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_YCX": {
        "w10_unroll": ("967a5fd1d28d3a71404ccd23", 27),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("eabc8ea17a547afeb7d8d2bb", 474),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("b143debb9523b4da2483318f", 522),
        "w1b_tableau_precompile_ucu_false": ("c1da8ead1d6c14af78d004d0", 219),
        "w1c_tableau_precompile_full": ("becd1248d4b9579b97a9ae3c", 917),
        "w2_coset_reference": ("77c31bcf900d66844288a939", 1037),
        "w3_magic_pass": ("cf6b616f12ce9c67cf14181c", 1483),
        "w3b_extract_faults": ("dc818d79763d659d9d9d7980", 26),
        "w3c_magic_pass_external_fault_slots": ("481a09c93c25bd33b6fcb587", 2430),
        "w3d_magic_pass_selector": ("2646ab5d22a2ab326d996753", 3517),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("2056d3cfc7fb1f7037cb0de2", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("550f0973d4b8f4eaa612b0c3", 31),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_YCY": {
        "w10_unroll": ("bb09f23c415aab6b3d5170a1", 27),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("8c2e7a21b8ce4517bce4d3c5", 474),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("8f097c46847c1db69fe3a486", 522),
        "w1b_tableau_precompile_ucu_false": ("08c18d65986eb41fe92da9ef", 219),
        "w1c_tableau_precompile_full": ("3b63aa46b4e73479366da12f", 917),
        "w2_coset_reference": ("8994de2b313c6ae4cc81fe13", 1037),
        "w3_magic_pass": ("de374a7e48ad37909c9a7531", 1483),
        "w3b_extract_faults": ("941cd75a938d9f4410fbb469", 26),
        "w3c_magic_pass_external_fault_slots": ("695814a0d8026754b3245874", 2430),
        "w3d_magic_pass_selector": ("a1e6d71b85ef46c29d11033c", 3517),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("cf1894c69c7f3f0638d378c6", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("d470bd64114366aa5231609b", 31),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_YCZ": {
        "w10_unroll": ("c36c87aa84273e712b4da6e3", 27),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("20b6df95811069c866e657e8", 474),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c2a0bda16831141f7ca12a41", 135),
        "w1_tableau_precompile": ("8ff9680c1108abd8cc207fd8", 522),
        "w1b_tableau_precompile_ucu_false": ("128e00f599fc95888e635d1b", 219),
        "w1c_tableau_precompile_full": ("76e93ddb038f687b80a6c6c4", 917),
        "w2_coset_reference": ("498d8948eb83b318fa75c103", 1037),
        "w3_magic_pass": ("58bee46529416094bf0ba597", 1483),
        "w3b_extract_faults": ("f6bb6c54c3be578d6b2b7f8c", 26),
        "w3c_magic_pass_external_fault_slots": ("96ef0f103a8b9ddb2744603f", 2430),
        "w3d_magic_pass_selector": ("78098bacf672230b3167eefa", 3517),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("10b4d5a8f1209a7f1b6db7f3", 683),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e38beebf9fdb4bf76cb272c3", 25),
        "w7_strip": ("e26935a2334e29786c2c8500", 31),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_Y_ERROR": {
        "w10_unroll": ("d368e4301cc4ddbd90f82b02", 35),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("74fb1601a1d40148afec5482", 510),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("4b9b0a9706be621aa71c23ed", 141),
        "w1_tableau_precompile": ("8e9ee7aa4575abbfa357aa15", 538),
        "w1b_tableau_precompile_ucu_false": ("afacbcddd39c011c163a6532", 227),
        "w1c_tableau_precompile_full": ("0b911e42135c6040912ded97", 925),
        "w2_coset_reference": ("e679d13cb1525376fe3ff296", 1008),
        "w3_magic_pass": ("a3e3093c72d2cd06c6d0d928", 1595),
        "w3b_extract_faults": ("0b2353088c17991e2f62b3a5", 34),
        "w3c_magic_pass_external_fault_slots": ("3c333779a45445584bd60d6e", 2542),
        "w3d_magic_pass_selector": ("95e467d38de74b6bb2a40c0c", 3674),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("47f5f54aa4ee03a293b15c51", 671),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("499d779e6da186b2b15eb9e8", 22),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_Z": {
        "w10_unroll": ("fc9516d1819ffa0eb0713619", 23),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("a6d4464226aad5fd2c8ece14", 454),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("c6df6d47fbc9f947693d7a49", 135),
        "w1_tableau_precompile": ("c3af25f399a2f1ddaf5f6354", 496),
        "w1b_tableau_precompile_ucu_false": ("ba89e6da9513f9021dc68756", 215),
        "w1c_tableau_precompile_full": ("e89681c3dd2235e6a0200ba0", 881),
        "w2_coset_reference": ("0985fc3b143828c8c7934577", 1025),
        "w3_magic_pass": ("2e766d974d95d736b4f45149", 1479),
        "w3b_extract_faults": ("fad343ada22aa5d80477dc41", 22),
        "w3c_magic_pass_external_fault_slots": ("0a1d4df25557cf881651676c", 2426),
        "w3d_magic_pass_selector": ("656329363c7de8cf94a0a9c4", 3501),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("a8ca30efa0e99ec285002464", 703),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("799b8642f10414219226f96d", 27),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "g_Z_ERROR": {
        "w10_unroll": ("88d0aa3daa600b088406eb03", 35),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("7275cb7edc8f8eddc513bb26", 510),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("4b9b0a9706be621aa71c23ed", 141),
        "w1_tableau_precompile": ("e7254fdfda3a199ef0822900", 538),
        "w1b_tableau_precompile_ucu_false": ("b4c124c5a89a1394b382c27f", 227),
        "w1c_tableau_precompile_full": ("a5cbd893183f751b854f491d", 925),
        "w2_coset_reference": ("506c449a481384b1d8d6c142", 1008),
        "w3_magic_pass": ("def697d25d826644c48d5a63", 1595),
        "w3b_extract_faults": ("3cc479645d28356713334f8b", 34),
        "w3c_magic_pass_external_fault_slots": ("1371411d58c008fea6f1cd1d", 2542),
        "w3d_magic_pass_selector": ("a5ba107d6e3b63ae9e0e3086", 3674),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("1254d6a78d43133be69d9959", 758),
        "w4c_marginal_builder_abi": ("47f5f54aa4ee03a293b15c51", 671),
        "w4d_marginal_builder_uc_false": ("1254d6a78d43133be69d9959", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("499d779e6da186b2b15eb9e8", 22),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "m01_magic_multibranch_selector": {
        "w10_unroll": ("39b818b1f59b8a840774161d", 187),
        "w11_u2f_and_swap": ("db3d49f7a92c173e744372c0", 83),
        "w12_ir_fields": ("728965a1bf56b819c42c1018", 1907),
        "w12b_tableau_layout": ("6e081a9caa4ed742de059eb2", 117),
        "w12c_ir_shared_fields": ("d4996e36be8027213cdd8957", 349),
        "w1_tableau_precompile": ("858b516832ee624caab575bd", 1616),
        "w1b_tableau_precompile_ucu_false": ("268a956294c4dd5048a2d6d8", 679),
        "w1c_tableau_precompile_full": ("65daa195f792708474248e08", 2385),
        "w2_coset_reference": ("a5a0ac992dff40a5fbf43bfa", 3054),
        "w3_magic_pass": ("136333b6bfdeec01be462b84", 5949),
        "w3b_extract_faults": ("6f32ecc0119cbe1f3b454c0c", 698),
        "w3c_magic_pass_external_fault_slots": ("33205faaae222ab3ae43cfb3", 6897),
        "w3d_magic_pass_selector": ("520ede20734d5c97ea0dcd7a", 19331),
        "w4_marginal_builder": ("c00519ad5a1177cc0384dc1d", 728),
        "w4b_marginal_builder_oracle": ("15b5d25af3463aabf535cf63", 798),
        "w4c_marginal_builder_abi": ("c272eb149b0fe65cc3e12ca5", 1029),
        "w4d_marginal_builder_uc_false": ("15b5d25af3463aabf535cf63", 798),
        "w5_sync_reference": ("a8158a5a93d9bcf5dc2f66b8", 23),
        "w5b_sync_reference_nohandler": ("a8158a5a93d9bcf5dc2f66b8", 23),
        "w6_known_states": ("78ba98a4ca962b864960cfdc", 40),
        "w7_strip": ("8488083a1a424428656c11bf", 67),
        "w8_parse_magic": ("057b11dcda2329aa9a999adc", 68),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "m02_magic_gauge_steps_and_data_tab": {
        "w10_unroll": ("fc1c1ab48d2afdb6c5a5855d", 544),
        "w11_u2f_and_swap": ("9d12b121884f5f257d03e58d", 244),
        "w12_ir_fields": ("db14a0aca292b5d038704d18", 5326),
        "w12b_tableau_layout": ("ed8af9f6e1b83e86ef26b67c", 469),
        "w12c_ir_shared_fields": ("1602af7e4be42cda5d52c4a8", 1293),
        "w1_tableau_precompile": ("af340b73e98d2f3d549c4314", 5051),
        "w1b_tableau_precompile_ucu_false": ("489829ac5317cb7c89f80fde", 2254),
        "w1c_tableau_precompile_full": ("f48447f02dfdc2cf769d938d", 7655),
        "w2_coset_reference": ("4ee56c0992997d6fec7e3b1d", -1),
        "w3_magic_pass": ("9b2fa2f1be8d3f4d5cb3c762", 11750),
        "w3b_extract_faults": ("3e868eada94aba455c0ac4a3", 610),
        "w3c_magic_pass_external_fault_slots": ("39d7b64ba0d625380f6babef", 12700),
        "w3d_magic_pass_selector": ("5301adf8fb2bbeede4a99d7b", 35493),
        "w4_marginal_builder": ("df7bc772e9482d1c9a4c3088", -1),
        "w4b_marginal_builder_oracle": ("df7bc772e9482d1c9a4c3088", -1),
        "w4c_marginal_builder_abi": ("df7bc772e9482d1c9a4c3088", -1),
        "w4d_marginal_builder_uc_false": ("8a6ee6470ec590fc79b82844", 959),
        "w5_sync_reference": ("8a7dc07a6089156d97aa6b2e", 28),
        "w5b_sync_reference_nohandler": ("8a7dc07a6089156d97aa6b2e", 28),
        "w6_known_states": ("aba70dd4733cff1f2ad2fb1d", 101),
        "w7_strip": ("f33e9ce7b3a9181c8bf7b4d1", 158),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "m03_magic_feedback_all_forms": {
        "w10_unroll": ("10c74afef447b4b2980f9dab", 210),
        "w11_u2f_and_swap": ("161335d276595c75dd16615c", 91),
        "w12_ir_fields": ("d75bf36cf21c45af291cb215", 1883),
        "w12b_tableau_layout": ("1cfd4e24739674fe793dddca", 129),
        "w12c_ir_shared_fields": ("bc4c1fd4e7de858d39dabdaa", 385),
        "w1_tableau_precompile": ("97a68bad2b0f21f5854a59db", 2077),
        "w1b_tableau_precompile_ucu_false": ("0aa3e0dedd92b747093da311", 762),
        "w1c_tableau_precompile_full": ("ff4c3f49c5ec645da75589ce", 3199),
        "w2_coset_reference": ("0f955cf383c2e6d4cd172c3e", 3687),
        "w3_magic_pass": ("c230a3838ac5528efc5c981e", 5056),
        "w3b_extract_faults": ("6a251b50e20e4cb2d0059899", 197),
        "w3c_magic_pass_external_fault_slots": ("0720a2c4e8717820741b2109", 6003),
        "w3d_magic_pass_selector": ("86d868281fa4a1033453d6aa", 11505),
        "w4_marginal_builder": ("a57cdd33c2ed9658d7e2a686", 728),
        "w4b_marginal_builder_oracle": ("219a550bd13867a626b8d11f", 806),
        "w4c_marginal_builder_abi": ("9f0cf24a3ba33c2ed7a2b4a2", 1193),
        "w4d_marginal_builder_uc_false": ("219a550bd13867a626b8d11f", 806),
        "w5_sync_reference": ("ee7d2c70eadb9c41d829f1a6", 25),
        "w5b_sync_reference_nohandler": ("ee7d2c70eadb9c41d829f1a6", 25),
        "w6_known_states": ("4eee502baa2554c72e99498c", 54),
        "w7_strip": ("09d9024e36987b21c195f396", 208),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "m04_magic_feedback_invalid_cx_order": {
        "w10_unroll": ("ed47fc044de947c1c274f277", 36),
        "w11_u2f_and_swap": ("f082a425bca6de5a5839a87a", 51),
        "w12_ir_fields": ("74aba0021d78c1726add04f6", 593),
        "w12b_tableau_layout": ("b5bd5d46bdc906f8d7eaa5ae", 69),
        "w12c_ir_shared_fields": ("7f48ef9575e654e1d477099f", 183),
        "w1_tableau_precompile": ("cec28f855b14e419bd750a23", 625),
        "w1b_tableau_precompile_ucu_false": ("36a8fcf638bc8378b1cf5f46", 288),
        "w1c_tableau_precompile_full": ("31f5521254b6862932bbfc8f", 1071),
        "w2_coset_reference": ("5d4c75671d8038b152667529", 1187),
        "w3_magic_pass": ("1a9ddb5b87b76ed28111a091", -1),
        "w3b_extract_faults": ("c9f3e7259688425fa290e12d", 33),
        "w3c_magic_pass_external_fault_slots": ("1a9ddb5b87b76ed28111a091", -1),
        "w3d_magic_pass_selector": ("1a9ddb5b87b76ed28111a091", -1),
        "w4_marginal_builder": ("bab6acedb9dc95eaa75884fe", -1),
        "w4b_marginal_builder_oracle": ("bab6acedb9dc95eaa75884fe", -1),
        "w4c_marginal_builder_abi": ("bab6acedb9dc95eaa75884fe", -1),
        "w4d_marginal_builder_uc_false": ("bab6acedb9dc95eaa75884fe", -1),
        "w5_sync_reference": ("2f574ba6068a831f476f67ee", 22),
        "w5b_sync_reference_nohandler": ("2f574ba6068a831f476f67ee", 22),
        "w6_known_states": ("2b87b45f376071245e408e59", -1),
        "w7_strip": ("7ff3356d94fb5dad96c131c3", 39),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "m05_magic_nested_repeat_inline_faults": {
        "w10_unroll": ("7db2e539781c94f1c7d9e0f6", 652),
        "w11_u2f_and_swap": ("1433a6e3a98f9a8939400b2a", 260),
        "w12_ir_fields": ("ece90760f6588a67573ccc66", 6219),
        "w12b_tableau_layout": ("9e974b0b4db74142e37fd485", 365),
        "w12c_ir_shared_fields": ("a7bae9d23093736fe763e761", 1353),
        "w1_tableau_precompile": ("380edb1b01ef3defd7bb2d12", 4342),
        "w1b_tableau_precompile_ucu_false": ("d4321917039071841353a2c8", 1997),
        "w1c_tableau_precompile_full": ("108364fb370f6624b060ec6e", 5629),
        "w2_coset_reference": ("02b7ab926d26e5d627155ff2", 7292),
        "w3_magic_pass": ("0f8789250a9c87dc2e4f84f5", 7348),
        "w3b_extract_faults": ("fb613aa144382b0bdb14bd5c", 1227),
        "w3c_magic_pass_external_fault_slots": ("b25e12797b76e78a8bcf98f6", 8298),
        "w3d_magic_pass_selector": ("b0f4e96816bbc426a502c4c9", 19693),
        "w4_marginal_builder": ("1873740e7258cec08598ddca", 980),
        "w4b_marginal_builder_oracle": ("3c33a4c0ccfe47b0ac42aab5", 1227),
        "w4c_marginal_builder_abi": ("5299fe1ea79ecadf340e6499", 2079),
        "w4d_marginal_builder_uc_false": ("3c33a4c0ccfe47b0ac42aab5", 1227),
        "w5_sync_reference": ("bc39bdc8d70acf45561af869", 27),
        "w5b_sync_reference_nohandler": ("bc39bdc8d70acf45561af869", 27),
        "w6_known_states": ("d5cf5f310ed8ccb7da53f4ee", 107),
        "w7_strip": ("1b64f40b9d14e5838e305601", 235),
        "w8_parse_magic": ("9abd194e5eca0c2f63284800", 24),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "m06_magic_pair_meas_inv_and_mpad": {
        "w10_unroll": ("6bf19247e6423cdebde8b13d", 179),
        "w11_u2f_and_swap": ("0efd8d2ab76b1dfa58c17d83", 108),
        "w12_ir_fields": ("44d19af97f39870b820a741e", 1876),
        "w12b_tableau_layout": ("d21410f391f21de54cd8566b", 163),
        "w12c_ir_shared_fields": ("3eabfac7d3b4b6f9b91e7288", 499),
        "w1_tableau_precompile": ("1d5d3b03526aa6ac601073f1", 1879),
        "w1b_tableau_precompile_ucu_false": ("f1c95c412c10ca85e71e7639", 852),
        "w1c_tableau_precompile_full": ("f16294cf93a16357003b50bb", 2901),
        "w2_coset_reference": ("6ac6780f8c325cd01adcd755", 4439),
        "w3_magic_pass": ("8ea3deb46a57e36f5add2a72", 3606),
        "w3b_extract_faults": ("76cb896626c3188320805949", 162),
        "w3c_magic_pass_external_fault_slots": ("543de094c575b171854306ba", 4554),
        "w3d_magic_pass_selector": ("d4fe4f68167be7301512b0bc", 9433),
        "w4_marginal_builder": ("ca1430f85be1951cf5c6f750", 728),
        "w4b_marginal_builder_oracle": ("1d448aaf341121b45779196e", 823),
        "w4c_marginal_builder_abi": ("a4256ca8bd4d9b7f89ea7f15", 1170),
        "w4d_marginal_builder_uc_false": ("1d448aaf341121b45779196e", 823),
        "w5_sync_reference": ("f60b4068d499ab03591e921d", 29),
        "w5b_sync_reference_nohandler": ("f60b4068d499ab03591e921d", 29),
        "w6_known_states": ("5ee0ff023beddc93a5915f33", 50),
        "w7_strip": ("1f0365b840fedb59f24c4a6d", 108),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "m07_magic_selector_multiword_66q": {
        "w10_unroll": ("069be41f456a9d6a7ea78756", 175),
        "w11_u2f_and_swap": ("161335d276595c75dd16615c", 91),
        "w12_ir_fields": ("46f236f62f64bbe95ead2495", 1844),
        "w12b_tableau_layout": ("1cfd4e24739674fe793dddca", 129),
        "w12c_ir_shared_fields": ("595b2d79fad8901cce63e7b9", 407),
        "w1_tableau_precompile": ("bba3e85e82ea4e1d3f142f0c", 1732),
        "w1b_tableau_precompile_ucu_false": ("9c804b3e0974eec9254dc065", 727),
        "w1c_tableau_precompile_full": ("05f703f5ac82cf0f0d617d17", 2667),
        "w2_coset_reference": ("614fdd1f427d6d7b7a2b258c", 41600),
        "w3_magic_pass": ("230ed804cf0742bf8d074e8a", 9082),
        "w3b_extract_faults": ("4d1905de7c94da026a3eb9cc", 400),
        "w3c_magic_pass_external_fault_slots": ("958b765e3a4054c9feb636a5", 10078),
        "w3d_magic_pass_selector": ("f26f86bce7b9f4153e008bc3", 32262),
        "w4_marginal_builder": ("0c0ba562e7d4516cff45fd5c", 728),
        "w4b_marginal_builder_oracle": ("3816e5ccbacae11bb2ff09e7", 806),
        "w4c_marginal_builder_abi": ("c2bd10ae79a3ea14e29eaff5", 1298),
        "w4d_marginal_builder_uc_false": ("3816e5ccbacae11bb2ff09e7", 806),
        "w5_sync_reference": ("0cfc79999f5139c98b05cc85", 22),
        "w5b_sync_reference_nohandler": ("0cfc79999f5139c98b05cc85", 22),
        "w6_known_states": ("feb3c3d0d3933bd85ecd05aa", 675),
        "w7_strip": ("db87d13e1f7d1433ae9d165b", 44),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "s01_rec_feedback_after_asym_readout": {
        "w10_unroll": ("4308e5f85096a1af85926ab2", 138),
        "w11_u2f_and_swap": ("54d80463c222de2ad3c941c3", 67),
        "w12_ir_fields": ("28a65e41c8deb42d3d624654", 1371),
        "w12b_tableau_layout": ("c768be9ef457fecacd6e256a", 69),
        "w12c_ir_shared_fields": ("4c05c28233d54918361a2bbd", 279),
        "w1_tableau_precompile": ("f1bb5f23ec02c74ec013808b", 1002),
        "w1b_tableau_precompile_ucu_false": ("180b48a91ebb3960b2b91a9b", 337),
        "w1c_tableau_precompile_full": ("3d3561220376dc865b2e1b23", 1925),
        "w2_coset_reference": ("b8561fb1d5bc2760eae8fe71", 2043),
        "w3_magic_pass": ("dbd3ff7ff94b9cf61a89e4b1", 2049),
        "w3b_extract_faults": ("8811eb00289448121afca4da", 131),
        "w3c_magic_pass_external_fault_slots": ("69fc48986d4f43265b7dfe36", 2996),
        "w3d_magic_pass_selector": ("5b8c820839f5483701366b99", 4467),
        "w4_marginal_builder": ("62242156334624ba4426e092", 796),
        "w4b_marginal_builder_oracle": ("6bb2c4ffed97807c0583f124", 850),
        "w4c_marginal_builder_abi": ("80fa8444ba157ba3c169b1a3", 798),
        "w4d_marginal_builder_uc_false": ("6bb2c4ffed97807c0583f124", 850),
        "w5_sync_reference": ("1a2324eb321c7b0f6420d098", 23),
        "w5b_sync_reference_nohandler": ("1a2324eb321c7b0f6420d098", 23),
        "w6_known_states": ("5296bec4be143b5c2f5989f0", 34),
        "w7_strip": ("acc9a6004c0866399aeab75d", 87),
        "w8_parse_magic": ("e6968619adfc897a06acb9e1", 212),
        "w8b_parse_tableau": ("ce0a1a7c15b7a68c1e891919", 204),
        "w9_collapses_z": ("3cbc87c7681f34db4617feaa", 4),
    },
    "s02_untagged_1q_2q_with_leak_tag": {
        "w10_unroll": ("ea7e70260f13a30510b72833", 115),
        "w11_u2f_and_swap": ("e6bdd9a5db6e6fac616a0e8c", 75),
        "w12_ir_fields": ("e90c8e1fd340a94fcbddb597", 1313),
        "w12b_tableau_layout": ("3bdb68bf14fd259d1900859a", 93),
        "w12c_ir_shared_fields": ("3f71e0ad3abc6691a0518bf5", 313),
        "w1_tableau_precompile": ("2ebc6e631fd0b5b384caad19", 1359),
        "w1b_tableau_precompile_ucu_false": ("15cbd7b37cacb657c11d31a1", 445),
        "w1c_tableau_precompile_full": ("42c9c495b9e918d129fb1c7f", 2551),
        "w2_coset_reference": ("bdb17a6f8ec5a1abf5912193", 2224),
        "w3_magic_pass": ("1a0c0cd33fb07b89f611da04", 1809),
        "w3b_extract_faults": ("6a1edb4227e5352df4c2a838", 106),
        "w3c_magic_pass_external_fault_slots": ("689e14db986c333cfb37fe21", 2756),
        "w3d_magic_pass_selector": ("13eda74b80ab1d82d68101e1", 4457),
        "w4_marginal_builder": ("b46d77df32d86b4b1eecd0b3", 750),
        "w4b_marginal_builder_oracle": ("9e47fc3b290db2d396a41c2d", 881),
        "w4c_marginal_builder_abi": ("92808ea480f7c5feeccb08ab", 2023),
        "w4d_marginal_builder_uc_false": ("030d996cc4dcaceaa113020d", 829),
        "w5_sync_reference": ("2f574ba6068a831f476f67ee", 22),
        "w5b_sync_reference_nohandler": ("2f574ba6068a831f476f67ee", 22),
        "w6_known_states": ("41213676ff2f62d9bbfdc964", 29),
        "w7_strip": ("149b80ef24a2fa5e057cade1", 74),
        "w8_parse_magic": ("267a469d402ba23604832858", 150),
        "w8b_parse_tableau": ("b482caea5191f2726a8d92fc", 142),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "s03_repeat_fused_boundary": {
        "w10_unroll": ("3902776d8beccacb1522f66e", 367),
        "w11_u2f_and_swap": ("39e09e8e71dcf90330ad8e41", 188),
        "w12_ir_fields": ("e6a4f932804b02f633f99aba", 3836),
        "w12b_tableau_layout": ("bfaf9b71024ed15e5c0fda28", 265),
        "w12c_ir_shared_fields": ("d1833c042fc1c5ee0038a762", 967),
        "w1_tableau_precompile": ("8dd9ac45505ae247bd0f06a9", 3649),
        "w1b_tableau_precompile_ucu_false": ("ab7835c7f0ac2e76b438138b", 1281),
        "w1c_tableau_precompile_full": ("40ea58950471a6d73850c064", 6107),
        "w2_coset_reference": ("f3587d6ace3eeae9abc26289", 5765),
        "w3_magic_pass": ("b260824a7b20e36a4a98dedc", 3844),
        "w3b_extract_faults": ("7dab45d710ce6fd8be5cf6ca", 330),
        "w3c_magic_pass_external_fault_slots": ("5acb87a16805fd71d9237fac", 4793),
        "w3d_magic_pass_selector": ("67a659f4582351b606f7a111", 8810),
        "w4_marginal_builder": ("d091834fc07c74f8a349a63b", 1000),
        "w4b_marginal_builder_oracle": ("9628da427a391535ff8f6b97", 1229),
        "w4c_marginal_builder_abi": ("64f1ce2f9230cccc144ab2b0", 1662),
        "w4d_marginal_builder_uc_false": ("9628da427a391535ff8f6b97", 1229),
        "w5_sync_reference": ("c62c18fd5a3df2a67e7d6165", 24),
        "w5b_sync_reference_nohandler": ("c62c18fd5a3df2a67e7d6165", 24),
        "w6_known_states": ("5e525c78e504e8ec70b57507", 58),
        "w7_strip": ("4f747433253e69d4b6db27a9", 112),
        "w8_parse_magic": ("47086e23e8b22107a3235b93", 150),
        "w8b_parse_tableau": ("e1114ff029e46f83c80cbe3d", 142),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "s04_untagged_1q_on_leaked_qubit": {
        "w10_unroll": ("0da03804843910e3d4ce7867", 162),
        "w11_u2f_and_swap": ("161335d276595c75dd16615c", 91),
        "w12_ir_fields": ("bea91884abca741a3b90dce5", 1717),
        "w12b_tableau_layout": ("2292b379c8f6247c761bd4a9", 105),
        "w12c_ir_shared_fields": ("dda105a47e48731ddcdf04e6", 419),
        "w1_tableau_precompile": ("e950b7f149f0f2112cc415af", 1604),
        "w1b_tableau_precompile_ucu_false": ("81741ae6c3fef472761b0fb8", 545),
        "w1c_tableau_precompile_full": ("a04e388e384f33c64729256e", 2877),
        "w2_coset_reference": ("f66055d8b6fe8fba3907dfc4", 3486),
        "w3_magic_pass": ("bdf9e5fbdc3bd340f66fb7b9", 2666),
        "w3b_extract_faults": ("6601e225572156c9aacffd3c", 149),
        "w3c_magic_pass_external_fault_slots": ("c175b4e6f2edf451b10956c4", 3613),
        "w3d_magic_pass_selector": ("80d3d28f825acaa59cccfefd", 6124),
        "w4_marginal_builder": ("fa3439fe72288a62556695a7", 796),
        "w4b_marginal_builder_oracle": ("5f412dd3406c029a40946868", 998),
        "w4c_marginal_builder_abi": ("159ce92ae2fbc0bee56db961", 1056),
        "w4d_marginal_builder_uc_false": ("005fd32bf2e40ff6f11c3ea2", 891),
        "w5_sync_reference": ("1a2324eb321c7b0f6420d098", 23),
        "w5b_sync_reference_nohandler": ("1a2324eb321c7b0f6420d098", 23),
        "w6_known_states": ("6328442b3e20ffae1db62bec", 44),
        "w7_strip": ("09adbf886d225104e900ab0f", 93),
        "w8_parse_magic": ("dc6b42957dab2a71557799fe", 159),
        "w8b_parse_tableau": ("3396a866c8bd5c20facac499", 151),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "x05_nested_repeat_boundary_fusion": {
        "w10_unroll": ("8c6cbc1dd868f06727ef1798", 159),
        "w11_u2f_and_swap": ("4a2e98e56801612448a20d29", 108),
        "w12_ir_fields": ("d0f530b366ecaca1d7899a24", 1578),
        "w12b_tableau_layout": ("964bdecc49dc5a56d60000ee", 131),
        "w12c_ir_shared_fields": ("e9a58e90a05d41b08d82f2b2", 527),
        "w1_tableau_precompile": ("4d1939ec48679ebc52eac985", 1486),
        "w1b_tableau_precompile_ucu_false": ("370615225d2a4768ae147e17", 711),
        "w1c_tableau_precompile_full": ("1736cde12aa996bfc7fa7780", 2253),
        "w2_coset_reference": ("10e88400f8262d2ca1d6a428", 1965),
        "w3_magic_pass": ("039754601593c9c24f84a234", 2045),
        "w3b_extract_faults": ("55543bb0e657a396a9ae5391", 142),
        "w3c_magic_pass_external_fault_slots": ("d1f87ba2f1543035d9e846cf", 2993),
        "w3d_magic_pass_selector": ("8c056ecb71c0e219ead8ff7a", 4511),
        "w4_marginal_builder": ("03291f18b414a386cd8e3f20", 860),
        "w4b_marginal_builder_oracle": ("36253715681b6a2946144a86", 955),
        "w4c_marginal_builder_abi": ("6cd98faabca054bb5887efd8", 822),
        "w4d_marginal_builder_uc_false": ("36253715681b6a2946144a86", 955),
        "w5_sync_reference": ("2f574ba6068a831f476f67ee", 22),
        "w5b_sync_reference_nohandler": ("2f574ba6068a831f476f67ee", 22),
        "w6_known_states": ("8451eda4b19e50931b9a26cc", 26),
        "w7_strip": ("d64ea9a00eb021273b072742", 84),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "x06_shift_coords_detector_in_repeat": {
        "w10_unroll": ("8059fa2983f4a672848deadc", 238),
        "w11_u2f_and_swap": ("dcbe48a75580de4eb09f0d0e", 140),
        "w12_ir_fields": ("fbfa690ffcbe069b97127850", 2342),
        "w12b_tableau_layout": ("ea3cfdd55c1b213b0ca960b5", 115),
        "w12c_ir_shared_fields": ("79ecc68147ef362531fc7aae", 699),
        "w1_tableau_precompile": ("10e42c663fa3d4aba0ddf4cb", 1401),
        "w1b_tableau_precompile_ucu_false": ("022e95f4478bd9cb73ed45de", 672),
        "w1c_tableau_precompile_full": ("3c96fa35a29bfd77cbc16850", 1677),
        "w2_coset_reference": ("5f4d2109c674914e8be5e6dd", 2492),
        "w3_magic_pass": ("1bf281894302b74edebcd18a", 1727),
        "w3b_extract_faults": ("722b7c7a36b71eba3e6a38da", 213),
        "w3c_magic_pass_external_fault_slots": ("b82e37fc4ff86b0f2f4be13f", 2675),
        "w3d_magic_pass_selector": ("034cc7b7c366a01cde9e18c0", 4035),
        "w4_marginal_builder": ("c74b01ce487d777003ec1032", 834),
        "w4b_marginal_builder_oracle": ("2a8bb2f1e5f54fd672860647", 961),
        "w4c_marginal_builder_abi": ("5fcacc0ceee70c9f145e8939", 1117),
        "w4d_marginal_builder_uc_false": ("2a8bb2f1e5f54fd672860647", 961),
        "w5_sync_reference": ("1a2324eb321c7b0f6420d098", 23),
        "w5b_sync_reference_nohandler": ("1a2324eb321c7b0f6420d098", 23),
        "w6_known_states": ("0de4ea93c946525fa3132458", 46),
        "w7_strip": ("a92985a11bb5f245ef4ec690", 147),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "x07_sweep_feedback": {
        "w10_unroll": ("3a014ff742388494733a47ca", 62),
        "w11_u2f_and_swap": ("54d80463c222de2ad3c941c3", 67),
        "w12_ir_fields": ("e1e65acb2c6c0842ea85734f", 891),
        "w12b_tableau_layout": ("55c55bdb44e9751e4aaa984b", 81),
        "w12c_ir_shared_fields": ("32c84a1f2ad806963deca17f", 273),
        "w1_tableau_precompile": ("0c40a82a20bba37b3dbdf31c", 801),
        "w1b_tableau_precompile_ucu_false": ("b119c8968f7f5c3958883bd2", 374),
        "w1c_tableau_precompile_full": ("2bc2e6289d0eb923082a9e48", 1303),
        "w2_coset_reference": ("29c4f2cdccfdceeefa8fb999", 1699),
        "w3_magic_pass": ("a95e816f8ba9eda995600def", 1812),
        "w3b_extract_faults": ("43f56bcb2458f44589278837", 55),
        "w3c_magic_pass_external_fault_slots": ("196e77c96c980ab401b23aef", 2759),
        "w3d_magic_pass_selector": ("e6ff7187bd8343dfd62cdf8e", 4407),
        "w4_marginal_builder": ("b46d77df32d86b4b1eecd0b3", 750),
        "w4b_marginal_builder_oracle": ("2dee98372589943d673ec77a", 804),
        "w4c_marginal_builder_abi": ("14ace365a551b02fae991c5e", 823),
        "w4d_marginal_builder_uc_false": ("2dee98372589943d673ec77a", 804),
        "w5_sync_reference": ("2f574ba6068a831f476f67ee", 22),
        "w5b_sync_reference_nohandler": ("2f574ba6068a831f476f67ee", 22),
        "w6_known_states": ("3b25c61bf6938038274ba3d2", 27),
        "w7_strip": ("00acaee23d3f6846515593d7", 63),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "x08_tag_copy_fused_across_repeat": {
        "w10_unroll": ("1b02c4a7741f8efc3ae23f25", 123),
        "w11_u2f_and_swap": ("63d315bdf6319033d0b7633b", 51),
        "w12_ir_fields": ("c16576ec423d571f90b4c41d", 1067),
        "w12b_tableau_layout": ("b5bd5d46bdc906f8d7eaa5ae", 69),
        "w12c_ir_shared_fields": ("d8fc07f5f74e6095f61c844f", 193),
        "w1_tableau_precompile": ("aad0b8f046e33647e3afd781", 1156),
        "w1b_tableau_precompile_ucu_false": ("6bf4ea57fcf127518bd099c0", 375),
        "w1c_tableau_precompile_full": ("150e3a54daf67b04865056b0", 2139),
        "w2_coset_reference": ("2618307b5e6c432721530b93", 1393),
        "w3_magic_pass": ("3b98713cbec1e07dc2f8fbc0", 1840),
        "w3b_extract_faults": ("908ff7af7f7757a888295ad6", 120),
        "w3c_magic_pass_external_fault_slots": ("cf08c38984d5da7359f53d86", 2787),
        "w3d_magic_pass_selector": ("0c8022d8b25b549682ed52f3", 4049),
        "w4_marginal_builder": ("73fdcc9af1efe454b20eb045", 796),
        "w4b_marginal_builder_oracle": ("12670db36773428f1e33c8d7", 802),
        "w4c_marginal_builder_abi": ("999427810d492e8433051a43", 730),
        "w4d_marginal_builder_uc_false": ("12670db36773428f1e33c8d7", 802),
        "w5_sync_reference": ("07dce59c607530184d5f5d5c", 84),
        "w5b_sync_reference_nohandler": ("ee7d2c70eadb9c41d829f1a6", 25),
        "w6_known_states": ("23db902f07fadcf04a8bdc3b", 28),
        "w7_strip": ("8ee04af03acb0409ef519a77", 96),
        "w8_parse_magic": ("08a644330c210a45bae417cb", 187),
        "w8b_parse_tableau": ("45b14b2c826757c1abf41393", 179),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "x09_tag_copy_nonmultiple_fused": {
        "w10_unroll": ("fdba303e99976d139d634b2f", 127),
        "w11_u2f_and_swap": ("63d315bdf6319033d0b7633b", 51),
        "w12_ir_fields": ("006a74041c28d614928a41c5", 1136),
        "w12b_tableau_layout": ("b5bd5d46bdc906f8d7eaa5ae", 69),
        "w12c_ir_shared_fields": ("d8fc07f5f74e6095f61c844f", 193),
        "w1_tableau_precompile": ("9c841eec05c16a81139fcf1a", 1184),
        "w1b_tableau_precompile_ucu_false": ("ac3bcf7adf2ae7f149d7c126", 379),
        "w1c_tableau_precompile_full": ("913dac57ed92df23432eacb5", 2183),
        "w2_coset_reference": ("c2651cedf656ec24511d2a92", 1397),
        "w3_magic_pass": ("cc3bc6688d43465472039e90", -1),
        "w3b_extract_faults": ("0d7ee9bb27183bc3563535a7", 124),
        "w3c_magic_pass_external_fault_slots": ("cc3bc6688d43465472039e90", -1),
        "w3d_magic_pass_selector": ("cc3bc6688d43465472039e90", -1),
        "w4_marginal_builder": ("d193c9d857ea4882ca4767af", 830),
        "w4b_marginal_builder_oracle": ("b629fb1514e8f13f768b2212", 820),
        "w4c_marginal_builder_abi": ("9d40c05757e6253dcc03b89b", 804),
        "w4d_marginal_builder_uc_false": ("b629fb1514e8f13f768b2212", 820),
        "w5_sync_reference": ("cc3bc6688d43465472039e90", -1),
        "w5b_sync_reference_nohandler": ("bc39bdc8d70acf45561af869", 27),
        "w6_known_states": ("23db902f07fadcf04a8bdc3b", 28),
        "w7_strip": ("2587e08f061bfb0601f46a61", 98),
        "w8_parse_magic": ("cc3bc6688d43465472039e90", -1),
        "w8b_parse_tableau": ("45f85b0f6491313764853df9", 181),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "y10_lt1_proj_fusion_peephole": {
        "w10_unroll": ("9e16f7637503e27c2f53ece9", 124),
        "w11_u2f_and_swap": ("f082a425bca6de5a5839a87a", 51),
        "w12_ir_fields": ("1e12f1a555b3f0af1f4edd28", 1166),
        "w12b_tableau_layout": ("921303c4356ec7e47525d1b7", 57),
        "w12c_ir_shared_fields": ("e746dc09a533057f277245a8", 187),
        "w1_tableau_precompile": ("4085c2b5e7795c7a68f0a039", 1104),
        "w1b_tableau_precompile_ucu_false": ("977bac89cd8dc61242860fcb", 235),
        "w1c_tableau_precompile_full": ("09e192f95f1685f338a2789b", 2435),
        "w2_coset_reference": ("c467cdf998be26ffe23f50ef", 2061),
        "w3_magic_pass": ("176aedd445240dc63be6f4be", 1981),
        "w3b_extract_faults": ("bd3be1a108c158aeaecde6fb", 121),
        "w3c_magic_pass_external_fault_slots": ("c59b846349c97c64426692df", 2928),
        "w3d_magic_pass_selector": ("d728fc39c2632f08837ab6a0", 4399),
        "w4_marginal_builder": ("c2bce55950daa71f152a2c47", -1),
        "w4b_marginal_builder_oracle": ("c2bce55950daa71f152a2c47", -1),
        "w4c_marginal_builder_abi": ("c2bce55950daa71f152a2c47", -1),
        "w4d_marginal_builder_uc_false": ("c2bce55950daa71f152a2c47", -1),
        "w5_sync_reference": ("a35918e63795a0c933cb95cf", 65),
        "w5b_sync_reference_nohandler": ("c62c18fd5a3df2a67e7d6165", 24),
        "w6_known_states": ("3599bdcd6adec6467df5c084", 33),
        "w7_strip": ("df632c4062c9199c4c907e44", 48),
        "w8_parse_magic": ("5a4c92c1fb8590a84f0f3cb5", 302),
        "w8b_parse_tableau": ("96c093c03f3b9e767b6f63de", 294),
        "w9_collapses_z": ("3cbc87c7681f34db4617feaa", 4),
    },
    "y11_rx_x_known_state_row": {
        "w10_unroll": ("fbfc374be22459029232c9ad", 131),
        "w11_u2f_and_swap": ("54d80463c222de2ad3c941c3", 67),
        "w12_ir_fields": ("8ecea1b0743952deec9e77b9", 1358),
        "w12b_tableau_layout": ("4d68d347b01df71734f226ef", 81),
        "w12c_ir_shared_fields": ("a08346372af4cab800ba63f0", 283),
        "w1_tableau_precompile": ("6dd01a4ce35fc6b014c72072", 1325),
        "w1b_tableau_precompile_ucu_false": ("2f50a9c72c21e173e132f2a8", 395),
        "w1c_tableau_precompile_full": ("20d356b5a79efb67f1c4b04c", 2757),
        "w2_coset_reference": ("4b1afb10cb99d983132bc6d8", 1345),
        "w3_magic_pass": ("9e6aee0bd452fe06a72ca20d", 1492),
        "w3b_extract_faults": ("ef3ecff4ee7624ce2c772c50", 124),
        "w3c_magic_pass_external_fault_slots": ("fc35adc92b660ee16048834a", 2439),
        "w3d_magic_pass_selector": ("cb3ea08d608273c4c373586b", 3875),
        "w4_marginal_builder": ("9defe689507295349d3b1cc7", 767),
        "w4b_marginal_builder_oracle": ("bed2aff81d290c70adac355b", 830),
        "w4c_marginal_builder_abi": ("16d8ebb505ef58c9bcc4fc08", 875),
        "w4d_marginal_builder_uc_false": ("bed2aff81d290c70adac355b", 830),
        "w5_sync_reference": ("2f574ba6068a831f476f67ee", 22),
        "w5b_sync_reference_nohandler": ("2f574ba6068a831f476f67ee", 22),
        "w6_known_states": ("513ac5e00a81cb7335caf01f", 20),
        "w7_strip": ("e52ddf9e5e748500c2e9b380", 51),
        "w8_parse_magic": ("facc5a8f11c4de08157a5098", 305),
        "w8b_parse_tableau": ("aecc829dbe948c0475358ea9", 297),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "y11b_two_qubit_prep_then_leak": {
        "w10_unroll": ("15affca87dec83516e83dfc9", 150),
        "w11_u2f_and_swap": ("e6bdd9a5db6e6fac616a0e8c", 75),
        "w12_ir_fields": ("3aca328c621c90ddc2b93b21", 1537),
        "w12b_tableau_layout": ("414c61869037b86facbf2211", 93),
        "w12c_ir_shared_fields": ("f9f8e033d43efaf1fb88ac26", 325),
        "w1_tableau_precompile": ("376bbc18dea17ee1c10ad9f6", 1502),
        "w1b_tableau_precompile_ucu_false": ("80cf16ba163cf430adc7519c", 474),
        "w1c_tableau_precompile_full": ("5347cdc31be03d1c426c3a01", 3019),
        "w2_coset_reference": ("6f2b40c0a96d34238b1fe6bd", 1976),
        "w3_magic_pass": ("dde0c7b9b9ad583528051133", 1952),
        "w3b_extract_faults": ("5121bd6609965a7cb2c26010", 141),
        "w3c_magic_pass_external_fault_slots": ("aec4da9140156e2a8b4f4ed4", 2899),
        "w3d_magic_pass_selector": ("ed0f3211030ff247f40f4d2f", 4781),
        "w4_marginal_builder": ("404a6eb27b125a3d4cc9be10", 767),
        "w4b_marginal_builder_oracle": ("3886633858a88facf71513e3", 838),
        "w4c_marginal_builder_abi": ("6a3d653800766fbd2b48d95c", 951),
        "w4d_marginal_builder_uc_false": ("3886633858a88facf71513e3", 838),
        "w5_sync_reference": ("1a2324eb321c7b0f6420d098", 23),
        "w5b_sync_reference_nohandler": ("1a2324eb321c7b0f6420d098", 23),
        "w6_known_states": ("cdf778f7b7e5e9c207f6a59a", 29),
        "w7_strip": ("52da9e37dfcfc49923c0b728", 69),
        "w8_parse_magic": ("facc5a8f11c4de08157a5098", 305),
        "w8b_parse_tableau": ("aecc829dbe948c0475358ea9", 297),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "y11c_x_then_leak": {
        "w10_unroll": ("f8377883cb0d8cb0848aacb5", 129),
        "w11_u2f_and_swap": ("54d80463c222de2ad3c941c3", 67),
        "w12_ir_fields": ("57493bee56faa634b801f033", 1348),
        "w12b_tableau_layout": ("d83d929462d67d3dd8888bd4", 81),
        "w12c_ir_shared_fields": ("967e9c56b6f1326dacfd980e", 283),
        "w1_tableau_precompile": ("baaf8e7faf6a6c12580c3168", 1317),
        "w1b_tableau_precompile_ucu_false": ("9474e455354a20b7be4b146b", 393),
        "w1c_tableau_precompile_full": ("22d5a469c2b9a628b24a1dde", 2749),
        "w2_coset_reference": ("7cec5da46ed01cc51fb08ac4", 1190),
        "w3_magic_pass": ("d60eaefcf1eccb54ce7ad420", 1215),
        "w3b_extract_faults": ("8d858ba8e9341a797a0058fc", 122),
        "w3c_magic_pass_external_fault_slots": ("4ece566381215ddfe6c12fdd", 2162),
        "w3d_magic_pass_selector": ("7fcbd009f05b3080c9cacb79", 2749),
        "w4_marginal_builder": ("9defe689507295349d3b1cc7", 767),
        "w4b_marginal_builder_oracle": ("fa94072162be04efad0ace81", 830),
        "w4c_marginal_builder_abi": ("f64a84bd6c411aa8caee940a", 875),
        "w4d_marginal_builder_uc_false": ("fa94072162be04efad0ace81", 830),
        "w5_sync_reference": ("5f81c67df23797ebf5ae7da8", 22),
        "w5b_sync_reference_nohandler": ("5f81c67df23797ebf5ae7da8", 22),
        "w6_known_states": ("28f4bdd37424b215d4fd3e49", 20),
        "w7_strip": ("66ad171db4c22136b17a2b15", 49),
        "w8_parse_magic": ("facc5a8f11c4de08157a5098", 305),
        "w8b_parse_tableau": ("aecc829dbe948c0475358ea9", 297),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "y12_tick_between_steps": {
        "w10_unroll": ("b3e4b72ff3acbe2e628db445", 73),
        "w11_u2f_and_swap": ("e6bdd9a5db6e6fac616a0e8c", 75),
        "w12_ir_fields": ("0a8c0613bd90d2a3d8f4f6e8", 1035),
        "w12b_tableau_layout": ("301fc75998903fd396eb84d2", 69),
        "w12c_ir_shared_fields": ("fdf60ff14b6faebbe0a08d20", 325),
        "w1_tableau_precompile": ("7502f5d98415e250b605d385", 715),
        "w1b_tableau_precompile_ucu_false": ("2ede14076451f07fd8443809", 325),
        "w1c_tableau_precompile_full": ("71046fbb6d52c92ab6aec094", 1103),
        "w2_coset_reference": ("4749d47df67b0be38da2c7f6", 1899),
        "w3_magic_pass": ("cd777c31e40674bc6138a49c", 1692),
        "w3b_extract_faults": ("9a34e63e298873b919512c73", 64),
        "w3c_magic_pass_external_fault_slots": ("9e05c44485b141c2d1ecacdd", 2639),
        "w3d_magic_pass_selector": ("846c24e32108f3b37dc3e6a3", 4261),
        "w4_marginal_builder": ("b46d77df32d86b4b1eecd0b3", 750),
        "w4b_marginal_builder_oracle": ("15239f5d8d797ed8dc4f9ffe", 812),
        "w4c_marginal_builder_abi": ("b72103eb24ef0a0245a7971a", 875),
        "w4d_marginal_builder_uc_false": ("15239f5d8d797ed8dc4f9ffe", 812),
        "w5_sync_reference": ("2f574ba6068a831f476f67ee", 22),
        "w5b_sync_reference_nohandler": ("2f574ba6068a831f476f67ee", 22),
        "w6_known_states": ("8bde364e283e73b566fb1f82", 29),
        "w7_strip": ("4c958ad9bb2ddeaf6e7922d9", 73),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "y13_split_copies_then_other_op": {
        "w10_unroll": ("44fe9878958c25975101fea0", 122),
        "w11_u2f_and_swap": ("e6bdd9a5db6e6fac616a0e8c", 75),
        "w12_ir_fields": ("fcde96e442aaaa239424e09b", 1345),
        "w12b_tableau_layout": ("9ed15d07837fb4afa4032894", 93),
        "w12c_ir_shared_fields": ("ef0032fdcd25886a810d99ed", 325),
        "w1_tableau_precompile": ("0c408e972764c390a2e038fd", 1254),
        "w1b_tableau_precompile_ucu_false": ("d0958a4eef08d381f9a60f03", 494),
        "w1c_tableau_precompile_full": ("2a12815a75d101af7e909ae8", 2313),
        "w2_coset_reference": ("c5f373f9f78046cac862f369", 2527),
        "w3_magic_pass": ("54dc578dd80914d10819ce04", 2675),
        "w3b_extract_faults": ("2c906417f4d24682d404e177", 113),
        "w3c_magic_pass_external_fault_slots": ("b7ed37a61c4ef921c193dd4c", 3622),
        "w3d_magic_pass_selector": ("c0be30b0ccaf64a40243040f", 6717),
        "w4_marginal_builder": ("164d8f4a9e9c41dbce5ac067", 824),
        "w4b_marginal_builder_oracle": ("7791db1e998c67dceb89239c", 870),
        "w4c_marginal_builder_abi": ("1a5ffeed55b156d430396532", 908),
        "w4d_marginal_builder_uc_false": ("7791db1e998c67dceb89239c", 870),
        "w5_sync_reference": ("c2ff7b5d3f80db5c525dcc29", 54),
        "w5b_sync_reference_nohandler": ("6c086a81325eb9d4f53ad326", 25),
        "w6_known_states": ("bbbf753b4366808f3c82ca66", 37),
        "w7_strip": ("a9e1b70afda45b5c0f3068fc", 109),
        "w8_parse_magic": ("08a644330c210a45bae417cb", 187),
        "w8b_parse_tableau": ("45b14b2c826757c1abf41393", 179),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "y13b_split_copies_leak_source": {
        "w10_unroll": ("5c75ed6e67ebd181de9bf17f", 155),
        "w11_u2f_and_swap": ("fd971af69de9ec97197ab780", 59),
        "w12_ir_fields": ("e4edcbb903747f64330b5fd1", 1459),
        "w12b_tableau_layout": ("761655f3dcc3afe3c3ca795e", 69),
        "w12c_ir_shared_fields": ("3a14c74c9014d8ec612c42dd", 241),
        "w1_tableau_precompile": ("6005b4f80ad017a862aa1805", 1377),
        "w1b_tableau_precompile_ucu_false": ("ed79655cb43eda95b605ca29", 357),
        "w1c_tableau_precompile_full": ("a8c0ee6186bee34eab5b0160", 2855),
        "w2_coset_reference": ("9386fbc50b9064f5dc79f495", 1846),
        "w3_magic_pass": ("4c9d678c49934310903689b4", 2190),
        "w3b_extract_faults": ("5f0f541aebfd48dba5b56c10", 150),
        "w3c_magic_pass_external_fault_slots": ("45c3a6e3a74de15c59b74a97", 3137),
        "w3d_magic_pass_selector": ("ac18a2352c5a2055a1fd4234", 4749),
        "w4_marginal_builder": ("07c4b622fb5848b3a4ad2fa8", 818),
        "w4b_marginal_builder_oracle": ("7628464b6d278a5e53c8d345", 867),
        "w4c_marginal_builder_abi": ("fe504c3c5b827a1847dd1e36", 932),
        "w4d_marginal_builder_uc_false": ("7628464b6d278a5e53c8d345", 867),
        "w5_sync_reference": ("4851e90534a6ce76695be2e0", 86),
        "w5b_sync_reference_nohandler": ("bc39bdc8d70acf45561af869", 27),
        "w6_known_states": ("6530ffbe95967dd4f50cbe0d", 31),
        "w7_strip": ("8736d93f14d13e7002dff6b7", 111),
        "w8_parse_magic": ("a58c195bd7632c1584cd89d7", 340),
        "w8b_parse_tableau": ("ac5994a67e44e6049f39fe22", 332),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "y14_memo_hides_second_parse": {
        "w10_unroll": ("8f0b10eb02649910f949062d", 129),
        "w11_u2f_and_swap": ("f082a425bca6de5a5839a87a", 51),
        "w12_ir_fields": ("9083f402c64fdc8b2a17c911", 1218),
        "w12b_tableau_layout": ("e71d44a171289798195487d6", -1),
        "w12c_ir_shared_fields": ("abf733a9ad0d794c6b15b1e1", 191),
        "w1_tableau_precompile": ("e71d44a171289798195487d6", -1),
        "w1b_tableau_precompile_ucu_false": ("e71d44a171289798195487d6", -1),
        "w1c_tableau_precompile_full": ("e71d44a171289798195487d6", -1),
        "w2_coset_reference": ("1e27a94897b871b9e7ac31c6", 1280),
        "w3_magic_pass": ("e71d44a171289798195487d6", -1),
        "w3b_extract_faults": ("3fc5f8666c93cbc01c67fbbd", 126),
        "w3c_magic_pass_external_fault_slots": ("e71d44a171289798195487d6", -1),
        "w3d_magic_pass_selector": ("e71d44a171289798195487d6", -1),
        "w4_marginal_builder": ("0bb9b1afc13c997516c5ca83", 790),
        "w4b_marginal_builder_oracle": ("11dd671dc0e2eafe33cba221", 828),
        "w4c_marginal_builder_abi": ("882bd6da18daaed70e1e529a", 678),
        "w4d_marginal_builder_uc_false": ("11dd671dc0e2eafe33cba221", 828),
        "w5_sync_reference": ("e71d44a171289798195487d6", -1),
        "w5b_sync_reference_nohandler": ("2f574ba6068a831f476f67ee", 22),
        "w6_known_states": ("809afcb9bfc26fba105fcd1a", 23),
        "w7_strip": ("e71d44a171289798195487d6", -1),
        "w8_parse_magic": ("e71d44a171289798195487d6", -1),
        "w8b_parse_tableau": ("e71d44a171289798195487d6", -1),
        "w9_collapses_z": ("e71d44a171289798195487d6", -1),
    },
    "y14b_memo_same_name_tag_diff_args": {
        "w10_unroll": ("79620d767d26ba8f9ea2bd7f", 129),
        "w11_u2f_and_swap": ("f082a425bca6de5a5839a87a", 51),
        "w12_ir_fields": ("ba9cbc7c337c227ba4f5daad", 1218),
        "w12b_tableau_layout": ("e71d44a171289798195487d6", -1),
        "w12c_ir_shared_fields": ("abf733a9ad0d794c6b15b1e1", 191),
        "w1_tableau_precompile": ("e71d44a171289798195487d6", -1),
        "w1b_tableau_precompile_ucu_false": ("e71d44a171289798195487d6", -1),
        "w1c_tableau_precompile_full": ("e71d44a171289798195487d6", -1),
        "w2_coset_reference": ("1cc3503f5e4d8040ef49a048", 1280),
        "w3_magic_pass": ("e71d44a171289798195487d6", -1),
        "w3b_extract_faults": ("5f09c17edfa184874a294548", 126),
        "w3c_magic_pass_external_fault_slots": ("e71d44a171289798195487d6", -1),
        "w3d_magic_pass_selector": ("e71d44a171289798195487d6", -1),
        "w4_marginal_builder": ("e71d44a171289798195487d6", -1),
        "w4b_marginal_builder_oracle": ("e71d44a171289798195487d6", -1),
        "w4c_marginal_builder_abi": ("e71d44a171289798195487d6", -1),
        "w4d_marginal_builder_uc_false": ("e71d44a171289798195487d6", -1),
        "w5_sync_reference": ("e71d44a171289798195487d6", -1),
        "w5b_sync_reference_nohandler": ("2f574ba6068a831f476f67ee", 22),
        "w6_known_states": ("809afcb9bfc26fba105fcd1a", 23),
        "w7_strip": ("e71d44a171289798195487d6", -1),
        "w8_parse_magic": ("e71d44a171289798195487d6", -1),
        "w8b_parse_tableau": ("e71d44a171289798195487d6", -1),
        "w9_collapses_z": ("e71d44a171289798195487d6", -1),
    },
    "y15_nonmultiple_copy_alone": {
        "w10_unroll": ("7fa1b0fa37414ac52e3527d8", 72),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("7550f993475a68e65fa61b93", 763),
        "w12b_tableau_layout": ("7c989caee150c41f3e29490d", 57),
        "w12c_ir_shared_fields": ("4b3ad3a61c84c625376d6fab", 145),
        "w1_tableau_precompile": ("d4b67a97a2da606c0392240c", 778),
        "w1b_tableau_precompile_ucu_false": ("336c7fe0f9a7fcad8ffe8024", 264),
        "w1c_tableau_precompile_full": ("4e40b9c7b3f9c1548aa34281", 1607),
        "w2_coset_reference": ("bc7eb022766c8108966beb1e", 1145),
        "w3_magic_pass": ("cc3bc6688d43465472039e90", -1),
        "w3b_extract_faults": ("2f05bddccbd037032fa0a77b", 71),
        "w3c_magic_pass_external_fault_slots": ("cc3bc6688d43465472039e90", -1),
        "w3d_magic_pass_selector": ("cc3bc6688d43465472039e90", -1),
        "w4_marginal_builder": ("cc3bc6688d43465472039e90", -1),
        "w4b_marginal_builder_oracle": ("cc3bc6688d43465472039e90", -1),
        "w4c_marginal_builder_abi": ("cc3bc6688d43465472039e90", -1),
        "w4d_marginal_builder_uc_false": ("cc3bc6688d43465472039e90", -1),
        "w5_sync_reference": ("cc3bc6688d43465472039e90", -1),
        "w5b_sync_reference_nohandler": ("c62c18fd5a3df2a67e7d6165", 24),
        "w6_known_states": ("d281da47de7cadf3ed201d3b", 25),
        "w7_strip": ("ea90b3073d753fdeb71d089c", 79),
        "w8_parse_magic": ("cc3bc6688d43465472039e90", -1),
        "w8b_parse_tableau": ("45f85b0f6491313764853df9", 181),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "y16_spp_then_badtag": {
        "w10_unroll": ("712c70f189433a0339504d5b", 101),
        "w11_u2f_and_swap": ("fd971af69de9ec97197ab780", 59),
        "w12_ir_fields": ("0858f4d9194d95b251a211ba", 1084),
        "w12b_tableau_layout": ("e71d44a171289798195487d6", -1),
        "w12c_ir_shared_fields": ("caab093130910dbc59817642", 239),
        "w1_tableau_precompile": ("e71d44a171289798195487d6", -1),
        "w1b_tableau_precompile_ucu_false": ("e71d44a171289798195487d6", -1),
        "w1c_tableau_precompile_full": ("e71d44a171289798195487d6", -1),
        "w2_coset_reference": ("4ee56c0992997d6fec7e3b1d", -1),
        "w3_magic_pass": ("e71d44a171289798195487d6", -1),
        "w3b_extract_faults": ("1a39eea8f8403b32b1313e4b", 96),
        "w3c_magic_pass_external_fault_slots": ("e71d44a171289798195487d6", -1),
        "w3d_magic_pass_selector": ("e71d44a171289798195487d6", -1),
        "w4_marginal_builder": ("df7bc772e9482d1c9a4c3088", -1),
        "w4b_marginal_builder_oracle": ("df7bc772e9482d1c9a4c3088", -1),
        "w4c_marginal_builder_abi": ("df7bc772e9482d1c9a4c3088", -1),
        "w4d_marginal_builder_uc_false": ("e71d44a171289798195487d6", -1),
        "w5_sync_reference": ("e71d44a171289798195487d6", -1),
        "w5b_sync_reference_nohandler": ("1a2324eb321c7b0f6420d098", 23),
        "w6_known_states": ("f2accb932a338de95a83ef11", 25),
        "w7_strip": ("e71d44a171289798195487d6", -1),
        "w8_parse_magic": ("e71d44a171289798195487d6", -1),
        "w8b_parse_tableau": ("e71d44a171289798195487d6", -1),
        "w9_collapses_z": ("e71d44a171289798195487d6", -1),
    },
    "y17_badtag_then_spp": {
        "w10_unroll": ("99286241336bfb11e34266d0", 101),
        "w11_u2f_and_swap": ("fd971af69de9ec97197ab780", 59),
        "w12_ir_fields": ("68dc5b017ffbd2efbcf73307", 1084),
        "w12b_tableau_layout": ("e71d44a171289798195487d6", -1),
        "w12c_ir_shared_fields": ("8bf6d22f9d319451fb6b4c64", 239),
        "w1_tableau_precompile": ("e71d44a171289798195487d6", -1),
        "w1b_tableau_precompile_ucu_false": ("e71d44a171289798195487d6", -1),
        "w1c_tableau_precompile_full": ("e71d44a171289798195487d6", -1),
        "w2_coset_reference": ("4ee56c0992997d6fec7e3b1d", -1),
        "w3_magic_pass": ("e71d44a171289798195487d6", -1),
        "w3b_extract_faults": ("d4222b248fa9ac6d97d84339", 96),
        "w3c_magic_pass_external_fault_slots": ("e71d44a171289798195487d6", -1),
        "w3d_magic_pass_selector": ("e71d44a171289798195487d6", -1),
        "w4_marginal_builder": ("e71d44a171289798195487d6", -1),
        "w4b_marginal_builder_oracle": ("e71d44a171289798195487d6", -1),
        "w4c_marginal_builder_abi": ("e71d44a171289798195487d6", -1),
        "w4d_marginal_builder_uc_false": ("e71d44a171289798195487d6", -1),
        "w5_sync_reference": ("e71d44a171289798195487d6", -1),
        "w5b_sync_reference_nohandler": ("1a2324eb321c7b0f6420d098", 23),
        "w6_known_states": ("30575ea5ccc1a239ee0f79c7", 25),
        "w7_strip": ("e71d44a171289798195487d6", -1),
        "w8_parse_magic": ("e71d44a171289798195487d6", -1),
        "w8b_parse_tableau": ("e71d44a171289798195487d6", -1),
        "w9_collapses_z": ("e71d44a171289798195487d6", -1),
    },
    "y18_qubitrange_then_badtag": {
        "w10_unroll": ("f80df7b61c46d5d7fa9add88", 138),
        "w11_u2f_and_swap": ("fd971af69de9ec97197ab780", 59),
        "w12_ir_fields": ("efe540a4778a7f051aabfff6", 1332),
        "w12b_tableau_layout": ("e71d44a171289798195487d6", -1),
        "w12c_ir_shared_fields": ("caab093130910dbc59817642", 239),
        "w1_tableau_precompile": ("e71d44a171289798195487d6", -1),
        "w1b_tableau_precompile_ucu_false": ("e71d44a171289798195487d6", -1),
        "w1c_tableau_precompile_full": ("e71d44a171289798195487d6", -1),
        "w2_coset_reference": ("44686865eb5a0f1bd373c134", 1564),
        "w3_magic_pass": ("c68f94bae204a194e33fd19a", -1),
        "w3b_extract_faults": ("e568b3401c65959d28563805", 133),
        "w3c_magic_pass_external_fault_slots": ("c68f94bae204a194e33fd19a", -1),
        "w3d_magic_pass_selector": ("c68f94bae204a194e33fd19a", -1),
        "w4_marginal_builder": ("c68f94bae204a194e33fd19a", -1),
        "w4b_marginal_builder_oracle": ("c68f94bae204a194e33fd19a", -1),
        "w4c_marginal_builder_abi": ("c68f94bae204a194e33fd19a", -1),
        "w4d_marginal_builder_uc_false": ("c68f94bae204a194e33fd19a", -1),
        "w5_sync_reference": ("e71d44a171289798195487d6", -1),
        "w5b_sync_reference_nohandler": ("c62c18fd5a3df2a67e7d6165", 24),
        "w6_known_states": ("f1f4ca25426e790eec50ae8e", 25),
        "w7_strip": ("e71d44a171289798195487d6", -1),
        "w8_parse_magic": ("e71d44a171289798195487d6", -1),
        "w8b_parse_tableau": ("e71d44a171289798195487d6", -1),
        "w9_collapses_z": ("e71d44a171289798195487d6", -1),
    },
    "y19_unsupported_then_badtag": {
        "w10_unroll": ("68641d28c9c13208afb3f4ea", 137),
        "w11_u2f_and_swap": ("fd971af69de9ec97197ab780", 59),
        "w12_ir_fields": ("a9cb97028365342fb710a716", 1326),
        "w12b_tableau_layout": ("e71d44a171289798195487d6", -1),
        "w12c_ir_shared_fields": ("caab093130910dbc59817642", 239),
        "w1_tableau_precompile": ("e71d44a171289798195487d6", -1),
        "w1b_tableau_precompile_ucu_false": ("e71d44a171289798195487d6", -1),
        "w1c_tableau_precompile_full": ("e71d44a171289798195487d6", -1),
        "w2_coset_reference": ("47117ce54501a536857952b0", 1563),
        "w3_magic_pass": ("e71d44a171289798195487d6", -1),
        "w3b_extract_faults": ("1e5531822bbc1b2fe81f3b15", 132),
        "w3c_magic_pass_external_fault_slots": ("e71d44a171289798195487d6", -1),
        "w3d_magic_pass_selector": ("e71d44a171289798195487d6", -1),
        "w4_marginal_builder": ("c2bce55950daa71f152a2c47", -1),
        "w4b_marginal_builder_oracle": ("c2bce55950daa71f152a2c47", -1),
        "w4c_marginal_builder_abi": ("c2bce55950daa71f152a2c47", -1),
        "w4d_marginal_builder_uc_false": ("c2bce55950daa71f152a2c47", -1),
        "w5_sync_reference": ("e71d44a171289798195487d6", -1),
        "w5b_sync_reference_nohandler": ("1a2324eb321c7b0f6420d098", 23),
        "w6_known_states": ("f1f4ca25426e790eec50ae8e", 25),
        "w7_strip": ("e71d44a171289798195487d6", -1),
        "w8_parse_magic": ("e71d44a171289798195487d6", -1),
        "w8b_parse_tableau": ("e71d44a171289798195487d6", -1),
        "w9_collapses_z": ("e71d44a171289798195487d6", -1),
    },
    "y20_spp_carrying_a_badtag": {
        "w10_unroll": ("80f512ddd971dfe7d9e1d6b1", 90),
        "w11_u2f_and_swap": ("f082a425bca6de5a5839a87a", 51),
        "w12_ir_fields": ("052da46b696d4026e37dc345", 950),
        "w12b_tableau_layout": ("6a41aaf8f90a91276c6c2bcf", -1),
        "w12c_ir_shared_fields": ("140bb0bcec7135d8dcd2681b", 193),
        "w1_tableau_precompile": ("6a41aaf8f90a91276c6c2bcf", -1),
        "w1b_tableau_precompile_ucu_false": ("6a41aaf8f90a91276c6c2bcf", -1),
        "w1c_tableau_precompile_full": ("6a41aaf8f90a91276c6c2bcf", -1),
        "w2_coset_reference": ("4ee56c0992997d6fec7e3b1d", -1),
        "w3_magic_pass": ("6a41aaf8f90a91276c6c2bcf", -1),
        "w3b_extract_faults": ("aede99895f830061c5ca5c0b", 87),
        "w3c_magic_pass_external_fault_slots": ("6a41aaf8f90a91276c6c2bcf", -1),
        "w3d_magic_pass_selector": ("6a41aaf8f90a91276c6c2bcf", -1),
        "w4_marginal_builder": ("df7bc772e9482d1c9a4c3088", -1),
        "w4b_marginal_builder_oracle": ("df7bc772e9482d1c9a4c3088", -1),
        "w4c_marginal_builder_abi": ("df7bc772e9482d1c9a4c3088", -1),
        "w4d_marginal_builder_uc_false": ("6a41aaf8f90a91276c6c2bcf", -1),
        "w5_sync_reference": ("6a41aaf8f90a91276c6c2bcf", -1),
        "w5b_sync_reference_nohandler": ("2f574ba6068a831f476f67ee", 22),
        "w6_known_states": ("2db860f78a830c3c76fe2e0c", 23),
        "w7_strip": ("6a41aaf8f90a91276c6c2bcf", -1),
        "w8_parse_magic": ("6a41aaf8f90a91276c6c2bcf", -1),
        "w8b_parse_tableau": ("6a41aaf8f90a91276c6c2bcf", -1),
        "w9_collapses_z": ("6a41aaf8f90a91276c6c2bcf", -1),
    },
    "y21_state01_tag_and_qubit_out_of_range": {
        "w10_unroll": ("25392d5804a9f4191493dabe", 73),
        "w11_u2f_and_swap": ("f082a425bca6de5a5839a87a", 51),
        "w12_ir_fields": ("79dcf246f0e903167e9263fa", 837),
        "w12b_tableau_layout": ("93f03fc1521db48a8b10a28c", 57),
        "w12c_ir_shared_fields": ("e746dc09a533057f277245a8", 187),
        "w1_tableau_precompile": ("e0e3a744b105fe43c7f1c46e", 631),
        "w1b_tableau_precompile_ucu_false": ("fd878d9fd5ae7e68410f69d7", 235),
        "w1c_tableau_precompile_full": ("3f5542d07d38d132da287054", 1273),
        "w2_coset_reference": ("fcab618255d97e55f62af537", 1513),
        "w3_magic_pass": ("50a333c40244445e8e19f605", -1),
        "w3b_extract_faults": ("2002975ae7ffb83e9c16ab3d", 70),
        "w3c_magic_pass_external_fault_slots": ("50a333c40244445e8e19f605", -1),
        "w3d_magic_pass_selector": ("50a333c40244445e8e19f605", -1),
        "w4_marginal_builder": ("50a333c40244445e8e19f605", -1),
        "w4b_marginal_builder_oracle": ("50a333c40244445e8e19f605", -1),
        "w4c_marginal_builder_abi": ("50a333c40244445e8e19f605", -1),
        "w4d_marginal_builder_uc_false": ("50a333c40244445e8e19f605", -1),
        "w5_sync_reference": ("79650075b7bbb7c11be15d5e", -1),
        "w5b_sync_reference_nohandler": ("2f574ba6068a831f476f67ee", 22),
        "w6_known_states": ("ebfa3bba6257b84f08de6e96", 28),
        "w7_strip": ("bf703340056615b9208bf43d", 75),
        "w8_parse_magic": ("50a333c40244445e8e19f605", -1),
        "w8b_parse_tableau": ("6175e5ae673dadd8b48f472e", 134),
        "w9_collapses_z": ("3cbc87c7681f34db4617feaa", 4),
    },
    "p01_inv_m": {
        "w10_unroll": ("3d6e5ae304e399972cf40563", 14),
        "w11_u2f_and_swap": ("dd1cebcbe68f9c7d8e08f466", 35),
        "w12_ir_fields": ("0c5c758bd04e48ddc7d707d5", 320),
        "w12b_tableau_layout": ("ce6ac9ddc0ce7df6ed718720", 45),
        "w12c_ir_shared_fields": ("4b2546ff6942b800cfc78381", 97),
        "w1_tableau_precompile": ("74944304083ff73064688999", 332),
        "w1b_tableau_precompile_ucu_false": ("1f1d06177bca5da37613030a", 146),
        "w1c_tableau_precompile_full": ("31005ef221f109cf9e822658", 609),
        "w2_coset_reference": ("b68d06b8daf35f3c4fff6f24", 529),
        "w3_magic_pass": ("e92f5fac022648fcb272933b", 911),
        "w3b_extract_faults": ("45efdee707c79551fee3f42c", 15),
        "w3c_magic_pass_external_fault_slots": ("c90a991429fb4086b95cb0e5", 1858),
        "w3d_magic_pass_selector": ("ff53d6404ea68072df6895f9", 2055),
        "w4_marginal_builder": ("5e011bfacca442c9f7f2b1f3", 728),
        "w4b_marginal_builder_oracle": ("cde0b0471a8444b954c9c6db", 750),
        "w4c_marginal_builder_abi": ("da3418befc685ed9df4d2e28", 570),
        "w4d_marginal_builder_uc_false": ("cde0b0471a8444b954c9c6db", 750),
        "w5_sync_reference": ("a03cb4b8ae4a23a7df211ade", 21),
        "w5b_sync_reference_nohandler": ("a03cb4b8ae4a23a7df211ade", 21),
        "w6_known_states": ("f8e4b59d48a3a578ba533ae8", 16),
        "w7_strip": ("e03600c96d8b89a8afa72a47", 19),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "p02_inv_mzz": {
        "w10_unroll": ("782d7bd377a8bc6d2482bbf4", 20),
        "w11_u2f_and_swap": ("dd1cebcbe68f9c7d8e08f466", 35),
        "w12_ir_fields": ("1138cc87420ac7c1c5ce8fc5", 350),
        "w12b_tableau_layout": ("ce6ac9ddc0ce7df6ed718720", 45),
        "w12c_ir_shared_fields": ("6db81a817432ce72ad3e1638", 95),
        "w1_tableau_precompile": ("8a37d3c1f5a3583888adb926", 352),
        "w1b_tableau_precompile_ucu_false": ("9602ad1c0ee0bf6479cf6914", 152),
        "w1c_tableau_precompile_full": ("1e41b164dd4df1e49fb87466", 625),
        "w2_coset_reference": ("f8da8de40fade099132a8ea4", 733),
        "w3_magic_pass": ("54e226988c3ff082427cc457", 1238),
        "w3b_extract_faults": ("4dd335bdd7abf43881f78e04", 21),
        "w3c_magic_pass_external_fault_slots": ("1074dca401c3b22765f06bbd", 2185),
        "w3d_magic_pass_selector": ("c3e17ba6f864afb4f99de79a", 2777),
        "w4_marginal_builder": ("5e011bfacca442c9f7f2b1f3", 728),
        "w4b_marginal_builder_oracle": ("cde0b0471a8444b954c9c6db", 750),
        "w4c_marginal_builder_abi": ("3de686970f0cf8e03bdb0ed0", 582),
        "w4d_marginal_builder_uc_false": ("cde0b0471a8444b954c9c6db", 750),
        "w5_sync_reference": ("a03cb4b8ae4a23a7df211ade", 21),
        "w5b_sync_reference_nohandler": ("a03cb4b8ae4a23a7df211ade", 21),
        "w6_known_states": ("cfa12b6c1ce9e17329dd39b2", 19),
        "w7_strip": ("0ea997b82abd5cfb6237a230", 25),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "p03_mpad1": {
        "w10_unroll": ("45e09d410ed0410efed940ce", 16),
        "w11_u2f_and_swap": ("dd1cebcbe68f9c7d8e08f466", 35),
        "w12_ir_fields": ("1693945aa2906057919b82bb", 328),
        "w12b_tableau_layout": ("ce6ac9ddc0ce7df6ed718720", 45),
        "w12c_ir_shared_fields": ("2386fad683d48e56f68c511e", 99),
        "w1_tableau_precompile": ("a6fc04f413555d91b070279d", 341),
        "w1b_tableau_precompile_ucu_false": ("7dec0eee926fb91d32335697", 148),
        "w1c_tableau_precompile_full": ("f0719ee8e8a35183b4a70b70", 619),
        "w2_coset_reference": ("7263594f16b6e380ed75be7b", 471),
        "w3_magic_pass": ("ec3991e086acc82dc3f8d546", 705),
        "w3b_extract_faults": ("a5361614f8d6296246495d1d", 17),
        "w3c_magic_pass_external_fault_slots": ("ef6b7bd2ef5cdd9fe906d9b8", 1652),
        "w3d_magic_pass_selector": ("32055ec86ba4bd7acd86aa7d", 1711),
        "w4_marginal_builder": ("0c0ba562e7d4516cff45fd5c", 728),
        "w4b_marginal_builder_oracle": ("333dbb7c841e41b7b69f7028", 750),
        "w4c_marginal_builder_abi": ("38d96505fe11561cc56d3679", 566),
        "w4d_marginal_builder_uc_false": ("333dbb7c841e41b7b69f7028", 750),
        "w5_sync_reference": ("5f81c67df23797ebf5ae7da8", 22),
        "w5b_sync_reference_nohandler": ("5f81c67df23797ebf5ae7da8", 22),
        "w6_known_states": ("582f7823deee16e572b47ac0", 19),
        "w7_strip": ("5c5341d266b212f0483a1b8b", 21),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "p04_mpp_rep": {
        "w10_unroll": ("e5f5b2fb2765e36df7b83b95", 24),
        "w11_u2f_and_swap": ("dd1cebcbe68f9c7d8e08f466", 35),
        "w12_ir_fields": ("8afb2f7cf387eaec3e6589d0", 370),
        "w12b_tableau_layout": ("ce6ac9ddc0ce7df6ed718720", 45),
        "w12c_ir_shared_fields": ("cd0913c02f6ea6719d03d366", 97),
        "w1_tableau_precompile": ("ce3ef10532d9c4b7c00cb23c", 364),
        "w1b_tableau_precompile_ucu_false": ("3d98c9530f14feb71e8f44ec", 156),
        "w1c_tableau_precompile_full": ("8c5a7eb214638c134f3e2dd0", 633),
        "w2_coset_reference": ("cc980d96e4e3805c4139b1d2", -1),
        "w3_magic_pass": ("d8eb3ea2bd85040bb74ace7d", -1),
        "w3b_extract_faults": ("fa54ce9aa6c5f1a893828f36", 25),
        "w3c_magic_pass_external_fault_slots": ("d8eb3ea2bd85040bb74ace7d", -1),
        "w3d_magic_pass_selector": ("d8eb3ea2bd85040bb74ace7d", -1),
        "w4_marginal_builder": ("b93a8566e6d4d65630c0aa57", -1),
        "w4b_marginal_builder_oracle": ("b93a8566e6d4d65630c0aa57", -1),
        "w4c_marginal_builder_abi": ("b93a8566e6d4d65630c0aa57", -1),
        "w4d_marginal_builder_uc_false": ("b93a8566e6d4d65630c0aa57", -1),
        "w5_sync_reference": ("0fd7e244b3c1e54208dacc60", -1),
        "w5b_sync_reference_nohandler": ("0fd7e244b3c1e54208dacc60", -1),
        "w6_known_states": ("8190e7d2c08731474624a9a3", -1),
        "w7_strip": ("bb59c292ea5d77cd92ae040b", 29),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "p05_mpp_neg": {
        "w10_unroll": ("fb1928f066237094d1a9a527", 23),
        "w11_u2f_and_swap": ("dd1cebcbe68f9c7d8e08f466", 35),
        "w12_ir_fields": ("96eb5c22892a7219c90b2ed0", 365),
        "w12b_tableau_layout": ("ce6ac9ddc0ce7df6ed718720", 45),
        "w12c_ir_shared_fields": ("1c2a9457b135b9b3e0247914", 97),
        "w1_tableau_precompile": ("6d3abf400f11fdb9461723be", 362),
        "w1b_tableau_precompile_ucu_false": ("136689724860005cd5241c7a", 155),
        "w1c_tableau_precompile_full": ("bb356eeea3b9bb52aefc6e9b", 633),
        "w2_coset_reference": ("a5bfe8a8b275927a0828c09f", 692),
        "w3_magic_pass": ("04fe30ed889cc7223281010c", 1188),
        "w3b_extract_faults": ("d73afdd965a637898227b000", 24),
        "w3c_magic_pass_external_fault_slots": ("8ebf1521bbc60c1ad63ef1d1", 2135),
        "w3d_magic_pass_selector": ("956d1e4f487f9cfda740c34e", 3179),
        "w4_marginal_builder": ("5e011bfacca442c9f7f2b1f3", 728),
        "w4b_marginal_builder_oracle": ("cde0b0471a8444b954c9c6db", 750),
        "w4c_marginal_builder_abi": ("5a189cbe8256c00f10f41a66", 570),
        "w4d_marginal_builder_uc_false": ("cde0b0471a8444b954c9c6db", 750),
        "w5_sync_reference": ("a03cb4b8ae4a23a7df211ade", 21),
        "w5b_sync_reference_nohandler": ("a03cb4b8ae4a23a7df211ade", 21),
        "w6_known_states": ("6abbebd6a63d8dc1fd9795ac", 16),
        "w7_strip": ("c80ca0ac1e02183fcbde6401", 28),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "p06_fb_mpad": {
        "w10_unroll": ("572ab55aa9566594bde276d5", 37),
        "w11_u2f_and_swap": ("f082a425bca6de5a5839a87a", 51),
        "w12_ir_fields": ("77a6c46f094bb730ec792dc5", 592),
        "w12b_tableau_layout": ("b5bd5d46bdc906f8d7eaa5ae", 69),
        "w12c_ir_shared_fields": ("26dbdf915b9aafa5ef078d9f", 185),
        "w1_tableau_precompile": ("77a49e3334e2113fb92ca4e0", 631),
        "w1b_tableau_precompile_ucu_false": ("ff9dfe52817f8b06c84d1231", 289),
        "w1c_tableau_precompile_full": ("fa30d5f5d621bc3a9f322052", 1079),
        "w2_coset_reference": ("47d435405d1c9d2845ed470a", 947),
        "w3_magic_pass": ("748c29c864115e8fba746d53", 1164),
        "w3b_extract_faults": ("30b9083b2ea5aa88e90b3cd6", 34),
        "w3c_magic_pass_external_fault_slots": ("41b2447186be4540ea06dbb9", 2111),
        "w3d_magic_pass_selector": ("6f22f3907f79a59e26205e1b", 2629),
        "w4_marginal_builder": ("0c0ba562e7d4516cff45fd5c", 728),
        "w4b_marginal_builder_oracle": ("6ef5cf8f13212871373ddc42", 766),
        "w4c_marginal_builder_abi": ("87a3aa33fa04075b28a7d65c", 678),
        "w4d_marginal_builder_uc_false": ("6ef5cf8f13212871373ddc42", 766),
        "w5_sync_reference": ("f075b87ec79544607a6ec615", 22),
        "w5b_sync_reference_nohandler": ("f075b87ec79544607a6ec615", 22),
        "w6_known_states": ("39fde4207f8a350351725e14", 23),
        "w7_strip": ("31b2c61adbd9831459c3d844", 40),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "p07_fb_qrec": {
        "w10_unroll": ("d0fb567d77e3a9bb2ae62417", 43),
        "w11_u2f_and_swap": ("fd971af69de9ec97197ab780", 59),
        "w12_ir_fields": ("c1ae0f50e9a1191b5175d7a9", 712),
        "w12b_tableau_layout": ("ddb25e99c6ba64288c294a2f", 81),
        "w12c_ir_shared_fields": ("93431206d28273a2755800ff", 225),
        "w1_tableau_precompile": ("df99dd9cbf337e010647dfb3", 764),
        "w1b_tableau_precompile_ucu_false": ("8d4817c4439ae04e943d1585", 355),
        "w1c_tableau_precompile_full": ("6a41e44738a9089abbdb1f1c", 1305),
        "w2_coset_reference": ("bb83aef304508ffe17fefc41", 1386),
        "w3_magic_pass": ("615b80e9104b19be5104379a", 1622),
        "w3b_extract_faults": ("71ada6c2efae54a2d32f015f", 38),
        "w3c_magic_pass_external_fault_slots": ("6f136628a22fe9fe523df908", 2569),
        "w3d_magic_pass_selector": ("a1c49c3587b6d8af1d218b6c", 3631),
        "w4_marginal_builder": ("0c0ba562e7d4516cff45fd5c", 728),
        "w4b_marginal_builder_oracle": ("2d05922f1500593e9ddcb793", 774),
        "w4c_marginal_builder_abi": ("4c4a30b458448df2934deba9", 775),
        "w4d_marginal_builder_uc_false": ("2d05922f1500593e9ddcb793", 774),
        "w5_sync_reference": ("f075b87ec79544607a6ec615", 22),
        "w5b_sync_reference_nohandler": ("f075b87ec79544607a6ec615", 22),
        "w6_known_states": ("5f73585c72650f343d505a6f", 25),
        "w7_strip": ("6b02852e354a3099f7a6852e", 45),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "p08_fb_cz_qrec": {
        "w10_unroll": ("b1e74f515f05f4676d00972d", 54),
        "w11_u2f_and_swap": ("e6bdd9a5db6e6fac616a0e8c", 75),
        "w12_ir_fields": ("557d1feb979445053c93275d", 935),
        "w12b_tableau_layout": ("7d16bba6913abd2fd7b73baa", 105),
        "w12c_ir_shared_fields": ("642075ef5c20cc5f763e3c94", 309),
        "w1_tableau_precompile": ("d34c930842fc6368f06b3673", 1014),
        "w1b_tableau_precompile_ucu_false": ("1cbd833408a81724df9c4d75", 486),
        "w1c_tableau_precompile_full": ("ee4765a9f6e73844f9e1c300", 1729),
        "w2_coset_reference": ("0342ecb91632c2e7bb3b771f", 1781),
        "w3_magic_pass": ("10bafde41cdf52460935b959", 1710),
        "w3b_extract_faults": ("82c748280d23a9da0da47358", 45),
        "w3c_magic_pass_external_fault_slots": ("decfa4faae044c17dbabf89b", 2657),
        "w3d_magic_pass_selector": ("c041c0dbec2943ce2bf8fb9f", 3983),
        "w4_marginal_builder": ("0c0ba562e7d4516cff45fd5c", 728),
        "w4b_marginal_builder_oracle": ("71b562396a99b960f4000679", 790),
        "w4c_marginal_builder_abi": ("f60ef01baad7298fd4f59e36", 915),
        "w4d_marginal_builder_uc_false": ("71b562396a99b960f4000679", 790),
        "w5_sync_reference": ("f075b87ec79544607a6ec615", 22),
        "w5b_sync_reference_nohandler": ("f075b87ec79544607a6ec615", 22),
        "w6_known_states": ("73b39de5b0ecfcf155041295", 29),
        "w7_strip": ("4051745c1ff9e6e339353eec", 54),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "p09_fb_mixed": {
        "w10_unroll": ("ee7062513b17c342d7d0b880", 50),
        "w11_u2f_and_swap": ("fd971af69de9ec97197ab780", 59),
        "w12_ir_fields": ("b731503a2aa9394459ecaced", 747),
        "w12b_tableau_layout": ("ddb25e99c6ba64288c294a2f", 81),
        "w12c_ir_shared_fields": ("93431206d28273a2755800ff", 225),
        "w1_tableau_precompile": ("3571bc54ba8ba3e80f0062f8", 808),
        "w1b_tableau_precompile_ucu_false": ("432fc736658f0488bed336de", 362),
        "w1c_tableau_precompile_full": ("390a365a49ba6c32619cebf7", 1365),
        "w2_coset_reference": ("b3c307fc9b26a734bc6a845d", 1822),
        "w3_magic_pass": ("cfcc78ea51fabdec899cb02d", 2170),
        "w3b_extract_faults": ("5bf95db11f2928461b429f96", 45),
        "w3c_magic_pass_external_fault_slots": ("ff5f2714bf166c2ae6ddf1f8", 3117),
        "w3d_magic_pass_selector": ("e52e922b4654017e2fae1efc", 4889),
        "w4_marginal_builder": ("c00519ad5a1177cc0384dc1d", 728),
        "w4b_marginal_builder_oracle": ("8a2311830c52d0c69a6c3823", 774),
        "w4c_marginal_builder_abi": ("a3fb4fa164c7e34459462407", 804),
        "w4d_marginal_builder_uc_false": ("8a2311830c52d0c69a6c3823", 774),
        "w5_sync_reference": ("7c917f31de11c94853080396", 23),
        "w5b_sync_reference_nohandler": ("7c917f31de11c94853080396", 23),
        "w6_known_states": ("3a284cd263c30a1f7d13a40c", 31),
        "w7_strip": ("2df2164f8e29da3470a98a94", 52),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "p10_mpad_z_super": {
        "w10_unroll": ("b7dd6eb20d3ea27894fd9969", 78),
        "w11_u2f_and_swap": ("f082a425bca6de5a5839a87a", 51),
        "w12_ir_fields": ("65bd364b366c4715dcfb0235", 877),
        "w12b_tableau_layout": ("b5bd5d46bdc906f8d7eaa5ae", 69),
        "w12c_ir_shared_fields": ("2fdfb8f6cd0c998a7cd33ae3", 187),
        "w1_tableau_precompile": ("821682b251604eae49b00ba6", 901),
        "w1b_tableau_precompile_ucu_false": ("b3f4742d251a75c7ca2a9cf6", 330),
        "w1c_tableau_precompile_full": ("51b60a711e1b3f862edde8f8", 1809),
        "w2_coset_reference": ("f0d5f6b1daefe1d85aed790b", 1620),
        "w3_magic_pass": ("15de25adc2079cda442ea34d", 2109),
        "w3b_extract_faults": ("1b6f59b1876bd7d9a499ad2e", 75),
        "w3c_magic_pass_external_fault_slots": ("f7fdccfcb9674a0a3a11bb9b", 3056),
        "w3d_magic_pass_selector": ("a874102de4770bceadea0392", 5069),
        "w4_marginal_builder": ("7c18cf2a888f37546252b4ff", 762),
        "w4b_marginal_builder_oracle": ("7fab56f7fb2fe490252be2b3", 784),
        "w4c_marginal_builder_abi": ("65afbbc636c874c64a1f6e6f", 747),
        "w4d_marginal_builder_uc_false": ("7fab56f7fb2fe490252be2b3", 784),
        "w5_sync_reference": ("4a0f91c0ba4212f36c67dcd7", 53),
        "w5b_sync_reference_nohandler": ("c62c18fd5a3df2a67e7d6165", 24),
        "w6_known_states": ("ee49fb1b408acc802915aaf2", 28),
        "w7_strip": ("5bf243ca6ac6a863ae1bc2e1", 84),
        "w8_parse_magic": ("08a644330c210a45bae417cb", 187),
        "w8b_parse_tableau": ("45b14b2c826757c1abf41393", 179),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "p11_lt_reset": {
        "w10_unroll": ("d45fcd230126116226633e63", 61),
        "w11_u2f_and_swap": ("f082a425bca6de5a5839a87a", 51),
        "w12_ir_fields": ("3a6654624eb30dfb52c7a88d", 791),
        "w12b_tableau_layout": ("e27342a05ff82ae8f29217d5", 69),
        "w12c_ir_shared_fields": ("34206ff611c646987238292c", 181),
        "w1_tableau_precompile": ("cfc66c12e9a4856a32241fe3", 913),
        "w1b_tableau_precompile_ucu_false": ("cfea1460ee20709376ad4394", 271),
        "w1c_tableau_precompile_full": ("1956c7ee2bb29b67a8e351ea", 1903),
        "w2_coset_reference": ("3335e905eb8b15d22db4fad3", 929),
        "w3_magic_pass": ("58f2bda8e1916c26fe356b3c", 1082),
        "w3b_extract_faults": ("0a629b49a1f1fff7c8f256b3", 58),
        "w3c_magic_pass_external_fault_slots": ("5b1e302ad5be0646cef93c87", 2029),
        "w3d_magic_pass_selector": ("58677fceb2599227de6b0634", 2483),
        "w4_marginal_builder": ("5e011bfacca442c9f7f2b1f3", 728),
        "w4b_marginal_builder_oracle": ("94959f71b97750d3459d92fd", 766),
        "w4c_marginal_builder_abi": ("61df76626e3cffba962e03d2", 771),
        "w4d_marginal_builder_uc_false": ("94959f71b97750d3459d92fd", 766),
        "w5_sync_reference": ("d3e9941fef6b4109272cb579", 37),
        "w5b_sync_reference_nohandler": ("a03cb4b8ae4a23a7df211ade", 21),
        "w6_known_states": ("3e21784119b6f3e416d83209", 18),
        "w7_strip": ("83ff7d1ce74485a8bc375742", 23),
        "w8_parse_magic": ("59f04b77dcfee5735348a1a1", 148),
        "w8b_parse_tableau": ("0d1c38218df8912128d6e8d2", 140),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "p12_bwd_mpp": {
        "w10_unroll": ("f6756ac617fce5bff23320ce", 41),
        "w11_u2f_and_swap": ("fd971af69de9ec97197ab780", 59),
        "w12_ir_fields": ("23d76e7898b8485324a3c068", 697),
        "w12b_tableau_layout": ("ddb25e99c6ba64288c294a2f", 81),
        "w12c_ir_shared_fields": ("6132236b4570fde0a10fa8fb", 231),
        "w1_tableau_precompile": ("8f94234676ed6fa216945226", 726),
        "w1b_tableau_precompile_ucu_false": ("925df6afa4d468dd264badee", 353),
        "w1c_tableau_precompile_full": ("ea0afc2dc81178d3e3f8cd88", 1237),
        "w2_coset_reference": ("b92f0a0384cc5dd355e4e4c7", 1813),
        "w3_magic_pass": ("09a96e4da32aecf6b6d28266", 1957),
        "w3b_extract_faults": ("b0c20c86ef32b851e899fee0", 36),
        "w3c_magic_pass_external_fault_slots": ("44987edad2863f805dc9a66c", 2904),
        "w3d_magic_pass_selector": ("5125af29da16d85bd3ef52e4", 4369),
        "w4_marginal_builder": ("c00519ad5a1177cc0384dc1d", 728),
        "w4b_marginal_builder_oracle": ("8a2311830c52d0c69a6c3823", 774),
        "w4c_marginal_builder_abi": ("a401556db3d0aa240ae869c6", 779),
        "w4d_marginal_builder_uc_false": ("8a2311830c52d0c69a6c3823", 774),
        "w5_sync_reference": ("fc626bca513b3de7ccc049f5", 23),
        "w5b_sync_reference_nohandler": ("fc626bca513b3de7ccc049f5", 23),
        "w6_known_states": ("878037ac0a30314fc0cf1e15", 31),
        "w7_strip": ("edd2eaae3f080084d2b62584", 43),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "p13_rx_order": {
        "w10_unroll": ("4cece47f4b92a2b7b7cecdd5", 42),
        "w11_u2f_and_swap": ("fd971af69de9ec97197ab780", 59),
        "w12_ir_fields": ("bd173129f7550bf9e23cc373", 703),
        "w12b_tableau_layout": ("ddb25e99c6ba64288c294a2f", 81),
        "w12c_ir_shared_fields": ("3f64b7d99807dcf1eb7f9a9c", 227),
        "w1_tableau_precompile": ("aa513e4ccae2ded9e6b02b97", 762),
        "w1b_tableau_precompile_ucu_false": ("6026bb8d3fec4e942658b368", 354),
        "w1c_tableau_precompile_full": ("5fa350ce4f6ab786a088d96c", 1305),
        "w2_coset_reference": ("38ebe8b1cbdb076b91eb5a60", 2116),
        "w3_magic_pass": ("a2b1f4e1630c8bf60e393147", 2796),
        "w3b_extract_faults": ("61e26ae1ee658a53c3a56df8", 37),
        "w3c_magic_pass_external_fault_slots": ("15676384551caf246c4a6bcb", 3743),
        "w3d_magic_pass_selector": ("3e6597ea95644daca9780d3e", 7287),
        "w4_marginal_builder": ("5e011bfacca442c9f7f2b1f3", 728),
        "w4b_marginal_builder_oracle": ("a281405443888b5b6dc3a0ea", 774),
        "w4c_marginal_builder_abi": ("29a72586550b4f5d22bdc52b", 791),
        "w4d_marginal_builder_uc_false": ("a281405443888b5b6dc3a0ea", 774),
        "w5_sync_reference": ("a03cb4b8ae4a23a7df211ade", 21),
        "w5b_sync_reference_nohandler": ("a03cb4b8ae4a23a7df211ade", 21),
        "w6_known_states": ("abefa94cf9437d6dc0304320", 31),
        "w7_strip": ("5897e325d3db9cdef17b4244", 44),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "p14_xcz_sweep_target": {
        "w10_unroll": ("957da901eb6c3435bd899f66", 40),
        "w11_u2f_and_swap": ("f082a425bca6de5a5839a87a", 51),
        "w12_ir_fields": ("74d433a4577bdaf84b2bd1fb", 618),
        "w12b_tableau_layout": ("b5bd5d46bdc906f8d7eaa5ae", 69),
        "w12c_ir_shared_fields": ("54c5bd2ea67543b0217118f1", 179),
        "w1_tableau_precompile": ("bc2bb9ad8d87e641429c7f9b", 658),
        "w1b_tableau_precompile_ucu_false": ("fc502e6610e69be6d59b6065", 292),
        "w1c_tableau_precompile_full": ("ce9942fed0803c6719a59702", 1121),
        "w2_coset_reference": ("648f4f7a6d4a1be8289b299c", 1334),
        "w3_magic_pass": ("7f67802f93515be756ca82e2", -1),
        "w3b_extract_faults": ("c27ff654a0fcc73202a7f43d", 37),
        "w3c_magic_pass_external_fault_slots": ("7f67802f93515be756ca82e2", -1),
        "w3d_magic_pass_selector": ("7f67802f93515be756ca82e2", -1),
        "w4_marginal_builder": ("1a7f3f603d192c758db01723", -1),
        "w4b_marginal_builder_oracle": ("1a7f3f603d192c758db01723", -1),
        "w4c_marginal_builder_abi": ("1a7f3f603d192c758db01723", -1),
        "w4d_marginal_builder_uc_false": ("1a7f3f603d192c758db01723", -1),
        "w5_sync_reference": ("2f574ba6068a831f476f67ee", 22),
        "w5b_sync_reference_nohandler": ("2f574ba6068a831f476f67ee", 22),
        "w6_known_states": ("2b87b45f376071245e408e59", -1),
        "w7_strip": ("0b1a4f012b924cf6f1d39392", 43),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "z01_zero_target_shift_fuse": {
        "w10_unroll": ("4904e4856eac8d527ad7ee86", 198),
        "w11_u2f_and_swap": ("7fc522f0dbecd6f68dc520bb", 156),
        "w12_ir_fields": ("a3f450d5db11a9088e1660b2", 2196),
        "w12b_tableau_layout": ("43ac765f582031a676ba103d", 175),
        "w12c_ir_shared_fields": ("88de3d3f79159dd16b610ec4", 779),
        "w1_tableau_precompile": ("0af518b663951f7c4dad3ce0", 1787),
        "w1b_tableau_precompile_ucu_false": ("51720f1102f94ee760133bc7", 874),
        "w1c_tableau_precompile_full": ("f53653a987437204c256ce33", 2619),
        "w2_coset_reference": ("29dd6fc773e66dd2bb3485df", 2860),
        "w3_magic_pass": ("f37520f8e4a08cd2cc0a2426", -1),
        "w3b_extract_faults": ("a785187ce5d88a19bafe8ad0", 169),
        "w3c_magic_pass_external_fault_slots": ("f37520f8e4a08cd2cc0a2426", -1),
        "w3d_magic_pass_selector": ("f37520f8e4a08cd2cc0a2426", -1),
        "w4_marginal_builder": ("b46d77df32d86b4b1eecd0b3", 750),
        "w4b_marginal_builder_oracle": ("9e2722ec0950d5800f2602bb", 893),
        "w4c_marginal_builder_abi": ("0819060cccc47ece468aa615", 1053),
        "w4d_marginal_builder_uc_false": ("9e2722ec0950d5800f2602bb", 893),
        "w5_sync_reference": ("2f574ba6068a831f476f67ee", 22),
        "w5b_sync_reference_nohandler": ("2f574ba6068a831f476f67ee", 22),
        "w6_known_states": ("2bce00b7f88310e99294d7a3", 50),
        "w7_strip": ("36a4d8df2103f7e6b68e3963", 158),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "z02_zero_target_repeat_fuse": {
        "w10_unroll": ("206370b3a6b9db6b19bdd348", 98),
        "w11_u2f_and_swap": ("e61181618c953816c4169cba", 75),
        "w12_ir_fields": ("8a1bc3fdcf7a4d68d2e8b4f6", 1190),
        "w12b_tableau_layout": ("3bdb68bf14fd259d1900859a", 93),
        "w12c_ir_shared_fields": ("4cc9912fc516d1893f80dc62", 319),
        "w1_tableau_precompile": ("85c600d0bc0ea0ed617fcd98", 1241),
        "w1b_tableau_precompile_ucu_false": ("9ec78182cdbd7a514f5d77ce", 422),
        "w1c_tableau_precompile_full": ("52888718ffb0ecfad6255128", 2407),
        "w2_coset_reference": ("d152d9d01a8ddb955c320ed5", 1929),
        "w3_magic_pass": ("de5e5019024f4423223fd316", 1778),
        "w3b_extract_faults": ("fe1f0e315334456cda25ea6b", 89),
        "w3c_magic_pass_external_fault_slots": ("6024d1050cd14fd3a51b14cb", 2725),
        "w3d_magic_pass_selector": ("0596cc8cd4017045a59fd04c", 4339),
        "w4_marginal_builder": ("b46d77df32d86b4b1eecd0b3", 750),
        "w4b_marginal_builder_oracle": ("e38636c7b24b72c3813c4eec", 829),
        "w4c_marginal_builder_abi": ("f474889983ec0ac91648312d", 811),
        "w4d_marginal_builder_uc_false": ("e38636c7b24b72c3813c4eec", 829),
        "w5_sync_reference": ("2f574ba6068a831f476f67ee", 22),
        "w5b_sync_reference_nohandler": ("2f574ba6068a831f476f67ee", 22),
        "w6_known_states": ("2cc67a11e83011de95992afa", 29),
        "w7_strip": ("c9f135633964872a2d545268", 67),
        "w8_parse_magic": ("9a3fbe1c9eb0fa75f279d9ce", 156),
        "w8b_parse_tableau": ("f0ca39b03de362f79b44373b", 148),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "z03_zero_target_nonfusing_pairs": {
        "w10_unroll": ("e857318b8a9d40d0a745ee3a", 317),
        "w11_u2f_and_swap": ("0736ac67e59a44aa2c6fe582", 180),
        "w12_ir_fields": ("0f1a057e2cf419ee07272cf5", 3035),
        "w12b_tableau_layout": ("1db91d9a8463b5076f3df6a4", 115),
        "w12c_ir_shared_fields": ("454c615e65ffbe6727e1b3b0", 961),
        "w1_tableau_precompile": ("23f992e3b15f8e2b2c2f57f7", 1472),
        "w1b_tableau_precompile_ucu_false": ("e5dedfb3b8d3ef5bbc3bb6ba", 694),
        "w1c_tableau_precompile_full": ("e559cf9b83d7841c57dae660", 1607),
        "w2_coset_reference": ("bad7d5a24b1bbecf6713c2ea", 1979),
        "w3_magic_pass": ("71fc10fd82052ec6b99d6ed3", 1517),
        "w3b_extract_faults": ("88036f1c194d681029fd83a5", 282),
        "w3c_magic_pass_external_fault_slots": ("42f6b145722550a0687125fa", 2466),
        "w3d_magic_pass_selector": ("c08efed2a8216adc24e178e3", 2897),
        "w4_marginal_builder": ("fe8ea68fbcfdd912bd5f0b8c", 862),
        "w4b_marginal_builder_oracle": ("8dfd53e5463d4b8b2c0e1e58", 1029),
        "w4c_marginal_builder_abi": ("1afbc952fb32d8314af0ba64", 1148),
        "w4d_marginal_builder_uc_false": ("8dfd53e5463d4b8b2c0e1e58", 1029),
        "w5_sync_reference": ("dc061b1593c3268e0dcd2823", 21),
        "w5b_sync_reference_nohandler": ("dc061b1593c3268e0dcd2823", 21),
        "w6_known_states": ("29dbb22ab32e41d4600c2fc8", 35),
        "w7_strip": ("68555324131b23f82513e194", 232),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "t01_tab_cond_u_kind8": {
        "w10_unroll": ("fe9e2a5118b13ea2990b1868", 160),
        "w11_u2f_and_swap": ("fd971af69de9ec97197ab780", 59),
        "w12_ir_fields": ("1526f37c7569d56cd9bc0a6d", 1464),
        "w12b_tableau_layout": ("3f4d50b431c8f1481b297f87", 81),
        "w12c_ir_shared_fields": ("8ee0d823446d710929eddde2", 229),
        "w1_tableau_precompile": ("298944416a6557623c4e6683", 1443),
        "w1b_tableau_precompile_ucu_false": ("cc5b601e0dd3483243a7c4cd", 377),
        "w1c_tableau_precompile_full": ("46659efdf5cc475eac35ed9d", 3273),
        "w2_coset_reference": ("38e64b099d735bdd129d20a4", 2105),
        "w3_magic_pass": ("156b0af5828b034b6a58bb20", 2307),
        "w3b_extract_faults": ("7556792f6d86dcee7acc7d1e", 155),
        "w3c_magic_pass_external_fault_slots": ("52c278807ef2d390a7212909", 3254),
        "w3d_magic_pass_selector": ("961aec4bea87edf5303d2d24", 4983),
        "w4_marginal_builder": ("c00519ad5a1177cc0384dc1d", 728),
        "w4b_marginal_builder_oracle": ("b89239a8b42cdffcc0c667e5", 791),
        "w4c_marginal_builder_abi": ("f3bc2c5849ba2368b1fec3b2", 912),
        "w4d_marginal_builder_uc_false": ("b89239a8b42cdffcc0c667e5", 791),
        "w5_sync_reference": ("79650075b7bbb7c11be15d5e", -1),
        "w5b_sync_reference_nohandler": ("1a2324eb321c7b0f6420d098", 23),
        "w6_known_states": ("8c5fd9dc7c50f7fd819d4f10", 31),
        "w7_strip": ("c306fa39328c8167fb59810a", 31),
        "w8_parse_magic": ("21206cf5f3a15c515e84167a", 442),
        "w8b_parse_tableau": ("947ef7e2e596fad425dafd9c", 434),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "t02_tab_lt1_dup_and_cow_fusion": {
        "w10_unroll": ("24c9d60366b01c2fd48d6b7b", 327),
        "w11_u2f_and_swap": ("0efd8d2ab76b1dfa58c17d83", 108),
        "w12_ir_fields": ("8bd054ffcdf0dce098102ec8", 3100),
        "w12b_tableau_layout": ("2f51711a3221fbc753b5e285", 155),
        "w12c_ir_shared_fields": ("b936226cdb75c41a06e6c78d", 483),
        "w1_tableau_precompile": ("167e2a4c9d5cb8539d5c83ec", 3506),
        "w1b_tableau_precompile_ucu_false": ("6bba14d7ed6e14d6e9b855da", 715),
        "w1c_tableau_precompile_full": ("1fd9a9240127869b7bae0d61", 7817),
        "w2_coset_reference": ("3d66b72b3765efa3f2414c9c", 4259),
        "w3_magic_pass": ("c67478d43aa9d51dffd54081", 2771),
        "w3b_extract_faults": ("09662613bef7a594ad01523d", 310),
        "w3c_magic_pass_external_fault_slots": ("499fd3a627caad19a328961f", 3719),
        "w3d_magic_pass_selector": ("952755a835aded6187674542", 6811),
        "w4_marginal_builder": ("c2bce55950daa71f152a2c47", -1),
        "w4b_marginal_builder_oracle": ("c2bce55950daa71f152a2c47", -1),
        "w4c_marginal_builder_abi": ("c2bce55950daa71f152a2c47", -1),
        "w4d_marginal_builder_uc_false": ("c2bce55950daa71f152a2c47", -1),
        "w5_sync_reference": ("1e5b9d27d112d668b490b9c1", 96),
        "w5b_sync_reference_nohandler": ("be082427b224f835f4d44015", 25),
        "w6_known_states": ("c4b47457db5c32fbad59af42", 38),
        "w7_strip": ("f556647899c1c2a54e49cb7e", 41),
        "w8_parse_magic": ("61243429a139773f8d64c747", 863),
        "w8b_parse_tableau": ("d60ecc009f2e8be792477743", 855),
        "w9_collapses_z": ("3cbc87c7681f34db4617feaa", 4),
    },
    "t03_tab_meas_inv_mz_dup": {
        "w10_unroll": ("993f663bc490a75f0e0e118f", 100),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("2db7269a3bac67e7b38f9d28", 962),
        "w12b_tableau_layout": ("00386a85b4ce0edcf0e22529", 57),
        "w12c_ir_shared_fields": ("b1c032899073a7b9b95d0f9b", 139),
        "w1_tableau_precompile": ("1300d3b66832180a6d111f11", 1112),
        "w1b_tableau_precompile_ucu_false": ("2567df34cea6471e8511f84d", 216),
        "w1c_tableau_precompile_full": ("7e66650c57ed60b8ef27206d", 2539),
        "w2_coset_reference": ("dd8218335957875a95e41493", 1477),
        "w3_magic_pass": ("2e2551096a303712af571df3", 1326),
        "w3b_extract_faults": ("cb954d3570d6af9e2dcd51d8", 99),
        "w3c_magic_pass_external_fault_slots": ("5dabae32cfc4c3b82e77c5a4", 2273),
        "w3d_magic_pass_selector": ("696badb7ce022fb25cf40af3", 2953),
        "w4_marginal_builder": ("a57cdd33c2ed9658d7e2a686", 728),
        "w4b_marginal_builder_oracle": ("b9788e5e1456f3c4bab8b659", 793),
        "w4c_marginal_builder_abi": ("d6d7f6a9a51c6d77a1013b9a", 755),
        "w4d_marginal_builder_uc_false": ("b9788e5e1456f3c4bab8b659", 793),
        "w5_sync_reference": ("d4445d82ebde40d96e231ae1", 25),
        "w5b_sync_reference_nohandler": ("d4445d82ebde40d96e231ae1", 25),
        "w6_known_states": ("fc95fd938a1b32dd534ba521", 21),
        "w7_strip": ("e4f33ba1c84868020aea22f4", 29),
        "w8_parse_magic": ("69ac4a98d115dc2cec2c4e16", 303),
        "w8b_parse_tableau": ("616f15dfee248c1dd5ba316a", 295),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "t04_tab_coords_tick_only": {
        "w10_unroll": ("71166c5c7dd0aa423d04296d", 52),
        "w11_u2f_and_swap": ("790db939b089c23b0cca8ad1", 43),
        "w12_ir_fields": ("89b842cb2929bb8b07bfe196", 542),
        "w12b_tableau_layout": ("ba5010540434e08dc8c2c562", 23),
        "w12c_ir_shared_fields": ("523f6278d9911c20dc3e529c", 149),
        "w1_tableau_precompile": ("3f0b905373eba084cb5d42b5", 230),
        "w1b_tableau_precompile_ucu_false": ("87a57a2cf075848c63419f6c", 68),
        "w1c_tableau_precompile_full": ("c989c1289d79bfda6cbf9d1e", 239),
        "w2_coset_reference": ("284860392ca6b111dab207be", 137),
        "w3_magic_pass": ("8bca37a8263a4651066ec147", 446),
        "w3b_extract_faults": ("9a086cb4ad7f50117688a41b", 51),
        "w3c_magic_pass_external_fault_slots": ("97279ffaee0ec07e857821ed", 1392),
        "w3d_magic_pass_selector": ("0b6d41cc9ab4863c4e6f9fef", 1125),
        "w4_marginal_builder": ("69d84d8a9e79e8a64a053ee1", 728),
        "w4b_marginal_builder_oracle": ("6687db62882974807b534d83", 758),
        "w4c_marginal_builder_abi": ("37a3b631d703f87ad5f915d0", 558),
        "w4d_marginal_builder_uc_false": ("6687db62882974807b534d83", 758),
        "w5_sync_reference": ("2bff2a202d44b21d80538df9", 20),
        "w5b_sync_reference_nohandler": ("2bff2a202d44b21d80538df9", 20),
        "w6_known_states": ("e40ae6c13e869c55f6fcb05b", 17),
        "w7_strip": ("bc0e0263c61a654ee20e63e2", 56),
        "w8_parse_magic": ("5ae1625b488b3935122d8dd6", 10),
        "w8b_parse_tableau": ("4f53cda18c2baa0c0354bb5f", 2),
        "w9_collapses_z": ("60a33e6cf5151f2d52eddae9", 5),
    },
    "t05_tab_lt2_01_only": {
        "w10_unroll": ("c418dfc13efd13684b24d18f", 66),
        "w11_u2f_and_swap": ("f7bcb98839bb260435559769", 43),
        "w12_ir_fields": ("e108c0dc3247b766f7313fa7", 740),
        "w12b_tableau_layout": ("00386a85b4ce0edcf0e22529", 57),
        "w12c_ir_shared_fields": ("c2d081e8abc50dc7a20f22d1", 139),
        "w1_tableau_precompile": ("7e681325296a72da0397b4ec", 889),
        "w1b_tableau_precompile_ucu_false": ("e8fb6dcf557865cdab8d0795", 208),
        "w1c_tableau_precompile_full": ("ba448a0d3a0d3873fc0bbfff", 1917),
        "w2_coset_reference": ("fe956dac8fa4d138cc69d2fc", 1112),
        "w3_magic_pass": ("c8dac43d8eb59ae0da277764", 1547),
        "w3b_extract_faults": ("59b7a3262c9171bdfd5feffa", 65),
        "w3c_magic_pass_external_fault_slots": ("ac04fdd2e25a05f0afb4248d", 2494),
        "w3d_magic_pass_selector": ("d412c5c5adcb07b0e84edbde", 3395),
        "w4_marginal_builder": ("c2bce55950daa71f152a2c47", -1),
        "w4b_marginal_builder_oracle": ("c2bce55950daa71f152a2c47", -1),
        "w4c_marginal_builder_abi": ("c2bce55950daa71f152a2c47", -1),
        "w4d_marginal_builder_uc_false": ("c2bce55950daa71f152a2c47", -1),
        "w5_sync_reference": ("14821f94a8e2d482d87ecd3b", 39),
        "w5b_sync_reference_nohandler": ("2f574ba6068a831f476f67ee", 22),
        "w6_known_states": ("fc95fd938a1b32dd534ba521", 21),
        "w7_strip": ("23ff5949193dddf31f6785d6", 22),
        "w8_parse_magic": ("c35ef7614134adecf616d297", 173),
        "w8b_parse_tableau": ("4f157f0dd80084ec77296d76", 165),
        "w9_collapses_z": ("3cbc87c7681f34db4617feaa", 4),
    },
}


def _flatten_expected() -> dict[str, tuple[str, int]]:
    return {
        f"{cname}|{wname}": digest
        for cname, entries in FROZEN_DIGESTS.items()
        for wname, digest in entries.items()
    }


def test_corpus_and_walker_inventory() -> None:
    """Verify exact 24 walkers, 152 self-contained circuits, and 3,648 entries."""
    expected_walkers = {
        "w1_tableau_precompile",
        "w1b_tableau_precompile_ucu_false",
        "w1c_tableau_precompile_full",
        "w2_coset_reference",
        "w3_magic_pass",
        "w3b_extract_faults",
        "w3c_magic_pass_external_fault_slots",
        "w3d_magic_pass_selector",
        "w4_marginal_builder",
        "w4b_marginal_builder_oracle",
        "w4c_marginal_builder_abi",
        "w4d_marginal_builder_uc_false",
        "w5_sync_reference",
        "w5b_sync_reference_nohandler",
        "w6_known_states",
        "w7_strip",
        "w8_parse_magic",
        "w8b_parse_tableau",
        "w9_collapses_z",
        "w10_unroll",
        "w11_u2f_and_swap",
        "w12_ir_fields",
        "w12b_tableau_layout",
        "w12c_ir_shared_fields",
    }
    assert set(WALKERS) == expected_walkers
    assert len(WALKERS) == 24

    corpus = build_corpus()
    names = [name for name, _circ in corpus]
    assert len(names) == len(set(names)), "duplicate circuit names in corpus"
    assert len(corpus) == 152

    expected_gates = {f"g_{n}" for n in stim.gate_data() if n != "REPEAT"}
    assert len(expected_gates) == 80
    assert expected_gates.issubset(set(names))

    assert set(FROZEN_DIGESTS) == set(names)
    for cname, entries in FROZEN_DIGESTS.items():
        assert set(entries) == expected_walkers, f"walker mismatch on {cname}"
    flat = _flatten_expected()
    assert len(flat) == 152 * 24
    assert sum(1 for _sha, nbytes in flat.values() if nbytes == -1) == 255


def test_canon_rejects_memory_addresses_and_is_stable() -> None:
    """Verify ``canon()`` rejects memory addresses and is order/layout invariant."""

    class _AddrRepr:
        def __repr__(self) -> str:
            return "<Dummy 0x7f123456abcd>"

    with pytest.raises(TypeError, match="address"):
        canon(object())
    with pytest.raises(TypeError, match="address"):
        canon([1, {"k": object()}])
    with pytest.raises(TypeError, match="address"):
        canon(_AddrRepr())

    @dataclasses.dataclass
    class _SampleDC:
        a: int
        b: str
        _private: object

    dc1 = _SampleDC(1, "x", object())
    dc2 = _SampleDC(1, "x", object())
    assert canon(dc1) == canon(dc2)

    assert canon({"b": 2, "a": 1}) == canon({"a": 1, "b": 2})
    assert canon({1, 2, 3}) == canon({3, 1, 2})
    assert canon(frozenset({"x", "y"})) == canon(frozenset({"y", "x"}))

    arr = np.arange(10, dtype=np.int32)
    sliced = arr[::2]
    copied = np.array([0, 2, 4, 6, 8], dtype=np.int32)
    assert not sliced.flags["C_CONTIGUOUS"]
    assert canon(sliced) == canon(copied)

    assert canon(np.int64(7)) == canon(7)
    assert canon(np.float64(1.5)) == canon(1.5)
    assert canon(np.bool_(True)) == canon(True)
    assert canon(bytearray(b"abc")) == canon(b"abc")

    class _CStruct(ctypes.Structure):
        _fields_ = (("x", ctypes.c_uint32), ("y", ctypes.c_uint32))

    s1 = _CStruct(x=3, y=4)
    s2 = _CStruct(x=3, y=4)
    assert canon(s1) == canon(s2)
    assert canon((ctypes.c_uint8 * 3)(1, 2, 3)) == canon((ctypes.c_uint8 * 3)(1, 2, 3))

    c = stim.Circuit("H 0\nM 0")
    assert canon(c) == str(c).encode()
    assert canon(c[0]) == str(c[0]).encode()


def test_run_walker_entry_does_not_swallow_canon_address_error() -> None:
    """Memory-address error in ``canon()`` must propagate, never hash as RAISES."""

    def _bad_walker(_c: stim.Circuit) -> object:
        return object()

    with pytest.raises(TypeError, match="address"):
        run_walker_entry(_bad_walker, stim.Circuit("H 0"))


def test_oracle_digests_match_frozen_table() -> None:
    """Every (circuit, walker) pair must match ``FROZEN_DIGESTS`` byte-for-byte."""
    expected = _flatten_expected()
    computed = compute_oracle_digests()
    actual_flat: dict[str, tuple[str, int]] = {}
    raises_info: dict[str, str] = {}
    for cname, entries in computed.items():
        for wname, (sha, nbytes, msg) in entries.items():
            key = f"{cname}|{wname}"
            actual_flat[key] = (sha, nbytes)
            if msg is not None:
                raises_info[key] = msg

    missing = sorted(set(expected) - set(actual_flat))
    extra = sorted(set(actual_flat) - set(expected))
    assert not missing, f"missing oracle keys ({len(missing)}): {missing[:10]}"
    assert not extra, f"unexpected oracle keys ({len(extra)}): {extra[:10]}"

    diffs: list[str] = []
    for key in sorted(expected):
        if actual_flat[key] != expected[key]:
            extra_msg = f" [{raises_info[key]}]" if key in raises_info else ""
            diffs.append(
                f"{key}: expected={expected[key]!r} got={actual_flat[key]!r}{extra_msg}"
            )
    assert not diffs, (
        f"{len(diffs)} oracle digest mismatch(es) (first 20):\n"
        + "\n".join(diffs[:20])
    )


def test_oracle_in_process_repeatability() -> None:
    """Re-running representative circuits in-process must yield identical digests."""
    subset_names = {
        "c01_clifford_noise_feedback",
        "c08_magic_with_leakage",
        "s01_rec_feedback_after_asym_readout",
        "s03_repeat_fused_boundary",
        "x05_nested_repeat_boundary_fusion",
        "y10_lt1_proj_fusion_peephole",
        "y14_memo_hides_second_parse",
        "y16_spp_then_badtag",
        "g_SPP",
    }
    subset = [(n, c) for n, c in build_corpus() if n in subset_names]
    assert len(subset) == 9
    run1 = compute_oracle_digests(circuits=subset)
    run2 = compute_oracle_digests(circuits=subset)
    assert run1 == run2


def test_error_order_circuits_pin_w4_contract_raises() -> None:
    """Circuits y16-y21 must pin raising contract outcomes on w4/w4b/w4c/w4d."""
    w4_names = (
        "w4_marginal_builder",
        "w4b_marginal_builder_oracle",
        "w4c_marginal_builder_abi",
        "w4d_marginal_builder_uc_false",
    )
    corpus_map = dict(build_corpus())
    expected_exc: dict[str, tuple[str, str]] = {
        "y16_spp_then_badtag": ("NotImplementedError", "ValueError"),
        "y17_badtag_then_spp": ("ValueError", "ValueError"),
        "y18_qubitrange_then_badtag": ("ValueError", "ValueError"),
        "y19_unsupported_then_badtag": ("NotImplementedError", "NotImplementedError"),
        "y20_spp_carrying_a_badtag": ("NotImplementedError", "ValueError"),
        "y21_state01_tag_and_qubit_out_of_range": ("ValueError", "ValueError"),
    }
    for cname, (uc_true_exc, uc_false_exc) in expected_exc.items():
        circ = corpus_map[cname]
        for wname in w4_names:
            sha, nbytes, msg = run_walker_entry(WALKERS[wname], circ)
            assert nbytes == -1 and msg is not None, f"{cname}|{wname} did not raise"
            want_type = uc_false_exc if wname.endswith("uc_false") else uc_true_exc
            assert msg.startswith(f"RAISES:{want_type}:"), f"{cname}|{wname}: {msg}"
            assert FROZEN_DIGESTS[cname][wname] == (sha, -1)
