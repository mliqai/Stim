"""Comprehensive invariants and contract test suite for ``circuit_ir`` and ``ir_tags``.

Self-contained pytest suite (zero external ``e2e_tests/`` file dependencies)
consolidating WP-27 RFC rev-2 invariants (I1-I19), rev-3 additions (I18b in both
directions, I20-I23, recursive purity I6/I6w with self-sensitivity tests), and
direct R-f field/pass coverage for ``util/circuit_ir.py`` and
``op_handlers/ir_tags.py``.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
import itertools
import operator
import sys
import time
import types
from collections.abc import Callable, Hashable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import stim

import stimside.dem_generators.marginal_builder as mb
from stimside import golden_test
from stimside._tableau.runner import PrecompiledTablesideCircuit
from stimside.dem_generators.dem_generator_marginal import MarginalLeakageDemGenerator
from stimside.dem_generators.marginal_builder import Builder, _check_supported
from stimside.op_handlers import ir_tags
from stimside.op_handlers.leakage_handlers.leakage_parameters import (
    LeakageConditioningParams,
    LeakageMeasurementParams,
)
from stimside.op_handlers.leakage_handlers.tag_registry import (
    check_tag_qubits,
    parse_leakage_tag,
)
from stimside.util import circuit_ir
from stimside.util.circuit import unroll_circuit
from stimside.util.known_states import compute_known_measurements
from stimside.util.leakage_events import unrolled_to_flattened_indices
from stimside.util.stim_workarounds import split_fused_instruction

_TAG2 = "MPAD[LEAKAGE_MEASUREMENT: (0, 0) (1, 1) : 1 2]"
_PROJ = "LEAKAGE_PROJECTION_Z: (0.05, 0) (0.9, 1)"
_SKIP_COSET = frozenset({"QUBIT_COORDS", "SHIFT_COORDS"})
_SKIP_TAB = frozenset(
    {"DETECTOR", "TICK", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS"}
)
_CORRELATED = frozenset({"E", "ELSE_CORRELATED_ERROR"})
_CONSUMER_EXC = (ValueError, NotImplementedError, KeyError, TypeError, RuntimeError)

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
            "}\n"
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
            "}\n"
        ),
    ),
    (
        "x07_sweep_feedback",
        "R 0 1\nH 0\nCX sweep[0] 1\nM 1\nDETECTOR rec[-1]\nM 0\n",
    ),
    (
        "x08_tag_copy_fused_across_repeat",
        f"R 0 1 2\nREPEAT 2 {{\n  {_TAG2} 0 0\n}}\nM 0\n",
    ),
    (
        "x09_tag_copy_nonmultiple_fused",
        f"R 0 1 2\nREPEAT 2 {{\n  {_TAG2} 0 0 0\n}}\nM 0\n",
    ),
)

_SUPPLEMENT_Y: tuple[tuple[str, str], ...] = (
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
        f"R 0 1 2\nH 0\n{_TAG2} 0 0\nH 1\nX_ERROR(0.1) 2\nM 0 1 2\nDETECTOR rec[-1]",
    ),
    (
        "y13b_split_copies_leak_source",
        (
            "R 0 1 2\nI_ERROR[LEAKAGE_TRANSITION_1: (0.2, U-->2)] 0 1\n"
            f"{_TAG2} 0 0 0 0\nM 0 1 2\nDETECTOR rec[-1] rec[-2]"
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

_EXTRA_EDGE_CIRCUITS: tuple[tuple[str, str], ...] = (
    (
        "z01_fused_mpp_combiner_repeat",
        "R 0 1\nREPEAT 2 {\n  MPP X0*X1\n}\n",
    ),
    (
        "z02_fused_meas_repeat",
        "R 0\nREPEAT 3 {\n  M 0\n}\n",
    ),
    (
        "z03_repeat_gt_50",
        "R 0\nREPEAT 60 {\n  H 0\n  X_ERROR(0.01) 0\n}\nM 0\n",
    ),
    (
        "z04_tagged_annotations",
        (
            "QUBIT_COORDS[MY_TAG](0, 0) 0\n"
            "TICK[MY_TAG]\n"
            "SHIFT_COORDS[MY_TAG](1)\n"
            "H 0\n"
            "M 0\n"
        ),
    ),
)


def _corpus() -> list[tuple[str, stim.Circuit]]:
    """Build the self-contained 41-circuit invariant corpus."""
    out = [
        (name, stim.Circuit(text))
        for name, _tags, _leak, text in golden_test.CORPUS
    ]
    for group in (_SUPPLEMENT_S, _SUPPLEMENT_X, _SUPPLEMENT_Y, _EXTRA_EDGE_CIRCUITS):
        for name, text in group:
            out.append((name, stim.Circuit(text)))
    return out


_CORPUS: tuple[tuple[str, stim.Circuit], ...] = tuple(_corpus())

# ---- Explicit hand-written Stim gate table (RFC §3.4 I22 / gates_rev3.py) ----

_EXPLICIT_CATEGORY_SPEC: dict[str, str] = {
    "annotation": "DETECTOR TICK OBSERVABLE_INCLUDE QUBIT_COORDS SHIFT_COORDS",
    "mpad": "MPAD",
    "herald": "HERALDED_ERASE HERALDED_PAULI_CHANNEL_1",
    "measure_reset": "MR MRX MRY",
    "measure": "M MX MY MXX MYY MZZ MPP",
    "reset": "R RX RY",
    "unitary": (
        "I X Y Z H H_XY H_YZ H_NXY H_NXZ H_NYZ S S_DAG SQRT_X SQRT_X_DAG "
        "SQRT_Y SQRT_Y_DAG C_XYZ C_ZYX C_NXYZ C_NZYX C_XNYZ C_XYNZ C_ZNYX "
        "C_ZYNX CX CY CZ XCX XCY XCZ YCX YCY YCZ SWAP ISWAP ISWAP_DAG CXSWAP "
        "CZSWAP SWAPCX SQRT_XX SQRT_XX_DAG SQRT_YY SQRT_YY_DAG SQRT_ZZ "
        "SQRT_ZZ_DAG II SPP SPP_DAG"
    ),
    "noise": (
        "X_ERROR Y_ERROR Z_ERROR I_ERROR II_ERROR DEPOLARIZE1 DEPOLARIZE2 "
        "PAULI_CHANNEL_1 PAULI_CHANNEL_2 E ELSE_CORRELATED_ERROR"
    ),
    "other": "REPEAT",
}


def _build_explicit_category() -> dict[str, str]:
    table: dict[str, str] = {}
    for cat, names in _EXPLICIT_CATEGORY_SPEC.items():
        for name in names.split():
            if name in table:
                raise AssertionError(f"Duplicate gate in EXPLICIT_CATEGORY: {name}")
            table[name] = cat
    return table


EXPLICIT_CATEGORY: dict[str, str] = _build_explicit_category()

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
    # Use a prefix op that cannot fuse with the probed gate (S 2 + S 0 would fuse).
    pre = "H 2" if name == "S" else "S 2"
    return stim.Circuit(f"R 0 1 2\n{pre}\n" + _gate_text(name))


def _forces_state(name: str) -> bool:
    """Return True iff single-qubit ``name`` forces qubit 0 to one fixed +P state."""
    starts = {
        "+X": "H 0",
        "-X": "H 0\nZ 0",
        "+Y": "H 0\nS 0",
        "-Y": "H 0\nS_DAG 0",
        "+Z": "I 0",
        "-Z": "X 0",
    }
    text = _gate_text(name)
    for pauli in "xyz":
        ends: list[bool] = []
        for pre in starts.values():
            sim = stim.TableauSimulator()
            sim.do_circuit(stim.Circuit(f"{pre}\n{text}"))
            ends.append(bool(getattr(sim, "peek_" + pauli)(0) == 1))
        if all(ends):
            return True
    return False


def _rebuilt_circuit(ops: list[stim.CircuitInstruction]) -> stim.Circuit:
    circ = stim.Circuit()
    for op in ops:
        circ.append(op)
    return circ


# ---- Memo differential helpers (I18b / memo_differential_rev3.py) ------------

_MEMO_TAG2 = "LEAKAGE_MEASUREMENT: (0, 0) (1, 1) : 1 2"
_MEMO_POOL: tuple[str, ...] = (
    f"M[{_PROJ}] 0",
    f"M[{_PROJ}](0.1) 0",
    f"H[{_PROJ}] 0",
    f"M[{_PROJ}] 1",
    "I_ERROR[LEAKAGE_TRANSITION_1: (0.1, U-->2)] 0",
    "I_ERROR[LEAKAGE_TRANSITION_1: (0.1, 1-->2)] 0",
    f"MPAD[{_MEMO_TAG2}] 0 0",
    f"MPAD[{_MEMO_TAG2}] 0 0 0",
    "MPAD[LEAKAGE_MEASUREMENT: (0, 0) (1, 1) : 9] 0",
    "SPP X0*Z1",
    "M 1",
)
_MEMO_TRIPLES: tuple[tuple[int, int, int], ...] = (
    (0, 3, 2),
    (2, 0, 3),
    (1, 0, 3),
    (0, 1, 3),
    (5, 2, 9),
    (9, 2, 5),
    (4, 5, 2),
    (8, 2, 0),
)


def _memo_diff_circuits() -> list[tuple[tuple[int, ...], stim.Circuit]]:
    out: list[tuple[tuple[int, ...], stim.Circuit]] = []
    combos: list[tuple[int, ...]] = list(
        itertools.product(range(len(_MEMO_POOL)), repeat=2)
    )
    combos.extend(_MEMO_TRIPLES)
    for combo in combos:
        body = "\n".join(_MEMO_POOL[i] for i in combo)
        text = f"R 0 1 2\n{body}\nM 0 1\nDETECTOR rec[-1]"
        out.append((combo, stim.Circuit(text)))
    return out


def _head_builder_outcome(c: stim.Circuit, uc: bool) -> tuple[Any, ...]:
    try:
        return ("OK", [repr(p) for p in Builder(c, uc).params])
    except _CONSUMER_EXC as exc:
        return ("RAISES", type(exc).__name__, str(exc)[:90])


def _memo_builder_outcome(
    c: stim.Circuit,
    uc: bool,
    key: Callable[[stim.CircuitInstruction], Hashable],
) -> tuple[Any, ...]:
    memo = ir_tags.TagMemo(
        lambda op: parse_leakage_tag(op, simulator="tableau"), key
    )
    try:
        params_out: list[str] = []
        for op in c.flattened():
            if uc and op.name in ("SPP", "SPP_DAG"):
                raise NotImplementedError(
                    f"MarginalLeakageDemGenerator does not support {op.name} with "
                    "unconditional_condition_on_U=True: it does not model a Pauli "
                    "product term that touches a leaked qubit."
                )
            params = memo.params(op)
            if params is not None:
                check_tag_qubits({op: params}, c.num_qubits)
                _check_supported(params)
            copy_len = 0
            if isinstance(params, LeakageMeasurementParams) and params.targets is not None:
                copy_len = len(params.targets)
            elif (
                isinstance(params, LeakageConditioningParams)
                and params.targets is not None
            ):
                mul = 2 if stim.gate_data(op.name).is_two_qubit_gate else 1
                copy_len = len(params.targets) * mul
            for _copy in split_fused_instruction(op, copy_len):
                params_out.append(repr(params))
        return ("OK", params_out)
    except _CONSUMER_EXC as exc:
        return ("RAISES", type(exc).__name__, str(exc)[:90])


def _head_precompiler_outcome(c: stim.Circuit) -> tuple[str, ...]:
    try:
        PrecompiledTablesideCircuit(c.copy(), True)
        return ("OK",)
    except _CONSUMER_EXC as exc:
        return ("RAISES", type(exc).__name__, str(exc)[:90])


def _memo_precompiler_outcome(
    c: stim.Circuit,
    key: Callable[[stim.CircuitInstruction], Hashable],
) -> tuple[str, ...]:
    memo = ir_tags.TagMemo(
        lambda op: parse_leakage_tag(op, simulator="tableau"), key
    )
    try:
        for op in unroll_circuit(c):
            if op.tag != "":
                memo.params(op)
        return ("OK",)
    except _CONSUMER_EXC as exc:
        return ("RAISES", type(exc).__name__, str(exc)[:90])


# ---- Static purity scanners (I6 / I6w) ---------------------------------------

_FORBIDDEN_ROOT_MODS = frozenset({"ctypes", "importlib", "builtins", "pydoc"})
_FORBIDDEN_CALL_NAMES = frozenset({"exec", "eval", "__import__", "getattr"})


def _resolve_import_from(
    node: ast.ImportFrom, pkg_parts: tuple[str, ...]
) -> tuple[str | None, bool]:
    """Resolve an ``ImportFrom`` module path; returns ``(resolved, escaped)``."""
    mod = node.module or ""
    if node.level == 0:
        return mod, False
    depth = len(pkg_parts) - node.level + 1
    if depth <= 0:
        return None, True
    base = list(pkg_parts[:depth])
    if mod:
        base.extend(mod.split("."))
    return ".".join(base), False


def _is_allowed_stimside_mod(mod: str, allowed_prefixes: tuple[str, ...]) -> bool:
    if not mod.startswith("stimside"):
        return True
    return any(mod == p or mod.startswith(p + ".") for p in allowed_prefixes)


def _scan_common_evasions(
    tree: ast.AST,
    label: str,
    pkg_parts: tuple[str, ...],
    allowed_prefixes: tuple[str, ...],
    prefix: str,
) -> list[str]:
    """Check dynamic-exec, builtins/getattr, sys.modules, TYPE_CHECKING & layering."""
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("stimside"):
                violations.append(f"{prefix}-no-stimside-string@{label}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _FORBIDDEN_ROOT_MODS:
                    violations.append(f"{prefix}-forbidden-import@{label}")
                elif not _is_allowed_stimside_mod(alias.name, allowed_prefixes):
                    violations.append(f"{prefix}-cross-layer-import@{label}")
        elif isinstance(node, ast.ImportFrom):
            resolved, escaped = _resolve_import_from(node, pkg_parts)
            if escaped:
                violations.append(f"{prefix}-escaping-relative-import@{label}")
            elif resolved is not None:
                root = resolved.split(".")[0]
                if root in _FORBIDDEN_ROOT_MODS:
                    violations.append(f"{prefix}-forbidden-import@{label}")
                elif resolved == "sys" and any(a.name == "modules" for a in node.names):
                    violations.append(f"{prefix}-sys-modules@{label}")
                elif not _is_allowed_stimside_mod(resolved, allowed_prefixes):
                    violations.append(f"{prefix}-cross-layer-import@{label}")
                elif resolved == "stimside":
                    for alias in node.names:
                        cand = f"stimside.{alias.name}"
                        if not _is_allowed_stimside_mod(cand, allowed_prefixes):
                            violations.append(f"{prefix}-cross-layer-import@{label}")
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in _FORBIDDEN_CALL_NAMES:
                violations.append(f"{prefix}-forbidden-call-{fn.id}@{label}")
            elif isinstance(fn, ast.Attribute) and fn.attr in ("exec", "eval", "locate"):
                violations.append(f"{prefix}-forbidden-call-{fn.attr}@{label}")
        elif isinstance(node, ast.Attribute) and node.attr == "modules":
            if isinstance(node.value, ast.Name) and node.value.id == "sys":
                violations.append(f"{prefix}-sys-modules@{label}")
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tgt in targets:
                if isinstance(tgt, ast.Name) and tgt.id == "TYPE_CHECKING":
                    violations.append(f"{prefix}-type-checking-rebind@{label}")
    return violations


def _scan_file_i6w(path: Path, label: str) -> list[str]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    violations: list[str] = []
    if "ctypes" in src or "OMEGA" in src:
        violations.append(f"I6w-no-ctypes-OMEGA@{label}")
    if "importlib" in src or "__import__" in src:
        violations.append(f"I6w-no-dynamic-import@{label}")
    rel_parts = Path(label).with_suffix("").parts
    pkg_parts = ("stimside", *rel_parts[:-1])
    allowed: tuple[str, ...] = (
        ("stimside.util",)
        if label.startswith("util/")
        else ("stimside.util", "stimside.op_handlers")
    )
    violations.extend(_scan_common_evasions(tree, label, pkg_parts, allowed, "I6w"))
    return violations


def _scan_tree_i6w(util_dir: Path, ir_tags_path: Path) -> list[str]:
    files = sorted(
        p
        for p in util_dir.rglob("*.py")
        if not p.name.endswith("_test.py") and not p.name.startswith("test_")
    )
    if ir_tags_path.exists():
        files.append(ir_tags_path)
    violations: list[str] = []
    pkg_root = util_dir.parent
    for path in files:
        label = str(path.relative_to(pkg_root))
        violations.extend(_scan_file_i6w(path, label))
    return violations


def _eq_tag_pair_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq)
            and len(node.comparators) == 1
        ):
            left, right = node.left, node.comparators[0]
            if (
                isinstance(left, ast.Attribute)
                and left.attr == "tag"
                and isinstance(left.ctx, ast.Load)
                and isinstance(right, ast.Attribute)
                and right.attr == "tag"
                and isinstance(right.ctx, ast.Load)
            ):
                ids.add(id(left))
                ids.add(id(right))
    return ids


def _scan_circuit_ir_i6(src: str) -> list[str]:
    tree = ast.parse(src)
    violations: list[str] = []
    if "ctypes" in src or "OMEGA" in src:
        violations.append("I6a-no-ctypes-OMEGA")
    if "LEAKAGE" in src or "CONDITIONED_ON" in src or "CLIFFORD_FAULT" in src:
        violations.append("I6b-no-tag-literal")
    if "importlib" in src or "__import__" in src:
        violations.append("I6d-no-dynamic-import")
    violations.extend(
        _scan_common_evasions(
            tree, "util/circuit_ir.py", ("stimside", "util"), ("stimside.util",), "I6d"
        )
    )
    tag_loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "tag"
        and isinstance(node.ctx, ast.Load)
    ]
    direct_ctor_tags = [
        arg
        for call in ast.walk(tree)
        if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "IRInstr"
        for arg in (*call.args, *(kw.value for kw in call.keywords))
        if isinstance(arg, ast.Attribute)
        and arg.attr == "tag"
        and isinstance(arg.ctx, ast.Load)
    ]
    ctor_ids = {id(arg) for arg in direct_ctor_tags}
    allowed_ids = ctor_ids | _eq_tag_pair_ids(tree)
    cond_types = (ast.If, ast.IfExp, ast.Compare, ast.BoolOp, ast.Match)
    ctor_in_cond = [
        inner
        for node in ast.walk(tree)
        if isinstance(node, cond_types)
        for inner in ast.walk(node)
        if id(inner) in ctor_ids
    ]
    bad_loads = [node for node in tag_loads if id(node) not in allowed_ids]
    if not ctor_ids or bad_loads or ctor_in_cond:
        violations.append("I6c-tag-only-copied")
    return violations


# ---- Core Invariants I1-I18 --------------------------------------------------


def test_i1_i2_unrolled_and_flat_mapping() -> None:
    """I1, I2a-d: unrolled ops, flat_ops, SHIFT_COORDS sentinel, monotonicity, u2f."""
    for cname, circ in _CORPUS:
        ir = circuit_ir.build_ir(circ, uc=True)
        unrolled = unroll_circuit(circ)
        assert [i.op for i in ir.instrs] == unrolled, f"I1 op mismatch: {cname}"
        assert [i.unrolled_index for i in ir.instrs] == list(range(len(unrolled))), cname
        assert ir.flat_ops == tuple(circ.flattened()), f"I2a: {cname}"
        assert all(
            (i.flat_index == -1) == (i.op.name == "SHIFT_COORDS") for i in ir.instrs
        ), f"I2b: {cname}"
        flat_indices = [i.flat_index for i in ir.instrs if i.flat_index >= 0]
        assert flat_indices == sorted(flat_indices), f"I2c: {cname}"
        want_u2f = [int(x) for x in unrolled_to_flattened_indices(circ, validate=True)]
        assert circuit_ir.unrolled_to_flat(ir) == want_u2f, f"I2d: {cname}"


def test_i3_has_repeat_and_no_repeat_identity() -> None:
    """I3/I3b: has_repeat flag and no-repeat direct list(circuit) equivalence."""
    saw_repeat = False
    saw_plain = False
    for cname, circ in _CORPUS:
        ir = circuit_ir.build_ir(circ, uc=True)
        expected_repeat = any(isinstance(i, stim.CircuitRepeatBlock) for i in circ)
        assert ir.has_repeat is expected_repeat, f"I3b: {cname}"
        if expected_repeat:
            saw_repeat = True
        else:
            saw_plain = True
            assert [i.op for i in ir.instrs] == list(circ), f"I3: {cname}"
    assert saw_repeat and saw_plain

    # Boundary REPEAT cases where len(ops) <= len(src) (kills naive length heuristics).
    boundary_cases = (
        ("REPEAT 1 {\n H 0\n}\n", 1, True),
        ("REPEAT 2 {\n}\n", 0, True),
        ("REPEAT 1 {\n}\n", 0, True),
        ("H 0\nREPEAT 1 {\n X 1\n}\n", 2, True),
        ("H 0\nX 1\n", 2, False),
    )
    for text, exp_len, exp_repeat in boundary_cases:
        ir = circuit_ir.build_ir(stim.Circuit(text), uc=True)
        assert len(ir.instrs) == exp_len, text
        assert ir.has_repeat is exp_repeat, text


def test_i4_untagged_filter_and_gate_table() -> None:
    """I4a/I4b and exhaustive stim.gate_data() filter table check."""
    for cname, circ in _CORPUS:
        ir_on = circuit_ir.build_ir(circ, uc=True)
        ir_off = circuit_ir.build_ir(circ, uc=False)
        assert all(
            i.untagged_filter == circuit_ir.UntaggedFilter.NONE for i in ir_off.instrs
        ), f"I4a: {cname}"
        for instr in ir_on.instrs:
            gd = stim.gate_data(instr.op.name)
            if instr.untagged_filter != circuit_ir.UntaggedFilter.NONE:
                assert not gd.produces_measurements, f"I4b meas: {cname}/{instr.op.name}"
                assert gd.is_noisy_gate or gd.is_unitary, f"I4b kind: {cname}/{instr.op.name}"
    for name in sorted(stim.gate_data()):
        gd = stim.gate_data(name)
        if gd.produces_measurements or not (gd.is_noisy_gate or gd.is_unitary):
            exp = circuit_ir.UntaggedFilter.NONE
        elif name in _CORRELATED:
            exp = circuit_ir.UntaggedFilter.CORRELATED
        elif name in ("SPP", "SPP_DAG"):
            exp = circuit_ir.UntaggedFilter.PAULI_TERMS
        elif gd.is_single_qubit_gate:
            exp = circuit_ir.UntaggedFilter.ONE_QUBIT
        elif gd.is_two_qubit_gate:
            exp = circuit_ir.UntaggedFilter.TWO_QUBIT
        else:
            exp = circuit_ir.UntaggedFilter.NONE
        assert circuit_ir.classify_untagged(name, uc=True) == exp, name
        assert (
            circuit_ir.classify_untagged(name, uc=False)
            == circuit_ir.UntaggedFilter.NONE
        ), name


def test_i5_i7_measurements_and_prefix_sum() -> None:
    """I5/I7: measurement count conservation and exact first_measurement prefix sum."""
    for cname, circ in _CORPUS:
        ir = circuit_ir.build_ir(circ, uc=True)
        total = sum(i.num_measurements for i in ir.instrs)
        assert total == circ.num_measurements == ir.num_measurements, f"I5: {cname}"
        running = 0
        for instr in ir.instrs:
            assert instr.first_measurement == running, f"I7: {cname}@{instr.unrolled_index}"
            running += instr.num_measurements
        assert running == ir.num_measurements, cname


def test_i8_distinct_ir_instr_identities() -> None:
    """I8: value-equal ops at distinct indices produce distinct IRInstr objects."""
    for cname, circ in _CORPUS:
        ir = circuit_ir.build_ir(circ, uc=True)
        assert len({id(i) for i in ir.instrs}) == len(ir.instrs), f"I8: {cname}"
    rep = stim.Circuit("REPEAT 4 {\n  H 0\n}\n")
    ir_rep = circuit_ir.build_ir(rep)
    assert len(ir_rep.instrs) == 4
    assert len({id(i) for i in ir_rep.instrs}) == 4
    assert len(set(ir_rep.instrs)) == 4


def test_i9_core_never_parses_tags_or_raises() -> None:
    """I9: core never parses or validates tags; malformed tags are copied verbatim."""
    malformed = stim.Circuit(
        "R 0 1\n"
        "M[TOTALLY_INVALID_TAG: ??? garbage !!!] 0\n"
        f"H[{_PROJ}] 0\n"
        f"M[{_PROJ}](0.1) 1\n"
        f"{_TAG2} 0 0 0\n"
    )
    ir = circuit_ir.build_ir(malformed, uc=True)
    assert ir.instrs[1].tag == "TOTALLY_INVALID_TAG: ??? garbage !!!"
    assert ir.instrs[2].tag == _PROJ
    assert ir.instrs[3].tag == _PROJ
    assert ir.instrs[4].tag == _MEMO_TAG2


def test_i10_i17_op_list_source_and_counts() -> None:
    """I10/I17: op-list sources have empty flat_ops, -1 flat_index, and match stim counts."""
    for cname, circ in _CORPUS:
        ir = circuit_ir.build_ir(circ, uc=True)
        ops = [i.op for i in ir.instrs]
        ir_list = circuit_ir.build_ir(ops)
        assert ir_list.from_op_list is True, f"I10: {cname}"
        assert ir_list.flat_ops == (), f"I10: {cname}"
        assert ir_list.has_repeat is False, cname
        assert all(
            i.flat_index == -1 and i.flat_offset == 0 for i in ir_list.instrs
        ), f"I10: {cname}"
        rebuilt = _rebuilt_circuit(ops)
        assert ir_list.num_measurements == rebuilt.num_measurements, f"I17 nm: {cname}"
        assert ir_list.num_qubits == rebuilt.num_qubits, f"I17 nq: {cname}"


def test_i11_flat_offset_tiles_flat_ops() -> None:
    """I11: flat_offset tiles every flat op's targets cleanly from 0."""
    saw_nonzero_offset = False
    for cname, circ in _CORPUS:
        ir = circuit_ir.build_ir(circ, uc=True)
        by_flat: dict[int, list[circuit_ir.IRInstr]] = {}
        for instr in ir.instrs:
            if instr.flat_index >= 0:
                by_flat.setdefault(instr.flat_index, []).append(instr)
        for f_idx, group in by_flat.items():
            offset = 0
            for instr in group:
                assert instr.flat_offset == offset, f"I11: {cname} f={f_idx}"
                if offset > 0:
                    saw_nonzero_offset = True
                offset += len(instr.op.targets_copy())
            assert offset == len(ir.flat_ops[f_idx].targets_copy()), (
                f"I11 total: {cname} f={f_idx}"
            )
    assert saw_nonzero_offset


def test_i12_category_anchors_and_consistency() -> None:
    """I12/I12b: IRInstr.category matches classify_category and category anchors."""
    annotations = frozenset(
        {"DETECTOR", "TICK", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"}
    )
    for cname, circ in _CORPUS:
        ir = circuit_ir.build_ir(circ, uc=True)
        for instr in ir.instrs:
            name = instr.op.name
            assert instr.category == circuit_ir.classify_category(name), cname
            if instr.category == circuit_ir.Category.OTHER:
                gd = stim.gate_data(name)
                assert not (
                    gd.is_unitary or gd.is_noisy_gate or gd.produces_measurements
                ), f"I12 OTHER: {cname}/{name}"
            if name == "MPAD":
                assert instr.category == circuit_ir.Category.MPAD, cname
            if name.startswith("HERALDED"):
                assert instr.category == circuit_ir.Category.HERALD, cname
            if name in annotations:
                assert instr.category == circuit_ir.Category.ANNOTATION, cname


def test_i13_i14_stored_uc_and_tag_copied() -> None:
    """I13/I14: stored uc flag and verbatim op.tag copy."""
    for cname, circ in _CORPUS:
        ir_on = circuit_ir.build_ir(circ, uc=True)
        ir_off = circuit_ir.build_ir(circ, uc=False)
        ir_def = circuit_ir.build_ir([i.op for i in ir_on.instrs])
        assert ir_on.uc is True, f"I13: {cname}"
        assert ir_off.uc is False, f"I13: {cname}"
        assert ir_def.uc is False, f"I13: {cname}"
        assert ir_on.from_op_list is False, cname
        assert all(i.tag == i.op.tag for i in ir_on.instrs), f"I14: {cname}"


def test_i15_split_copies_roundtrip_and_copy_len_one() -> None:
    """I15/I15b: split_copies copy_len, flat_to_op first-copy offsets, and copy_len=1."""
    for cname, circ in _CORPUS:
        ir = circuit_ir.build_ir(circ, uc=True)
        cm = circuit_ir.split_copies(
            ir, lambda op: 2 if (op.name == "MPAD" and op.tag) else 0
        )
        expected_lens = tuple(
            2 if (op.name == "MPAD" and op.tag) else 0 for op in ir.flat_ops
        )
        assert cm.copy_len == expected_lens, f"I15: {cname}"
        assert len(cm.flat_to_op) == len(ir.flat_ops), f"I15b len: {cname}"
        for k, op in enumerate(ir.flat_ops):
            end = cm.flat_to_op[k + 1] if k + 1 < len(cm.flat_to_op) else len(cm.ops)
            split_ops = split_fused_instruction(op, cm.copy_len[k])
            assert end - cm.flat_to_op[k] == len(split_ops), f"I15b slice: {cname}"
            assert list(cm.ops[cm.flat_to_op[k] : end]) == split_ops, cname

    # Explicit copy_len=1 check (kills H14 where copy_len==1 is recorded as 0).
    circ_one = stim.Circuit("H 0 1 2\nM 0 1\n")
    ir_one = circuit_ir.build_ir(circ_one)
    cm_one = circuit_ir.split_copies(ir_one, lambda op: 1 if op.name == "H" else 0)
    assert cm_one.copy_len == (1, 0)
    assert cm_one.flat_to_op == (0, 3)
    assert len(cm_one.ops) == 4
    assert [str(o) for o in cm_one.ops[:3]] == ["H 0", "H 1", "H 2"]

    # Fused across REPEAT boundary: flat-level splitting yields 3 copies for 6 targets.
    x09 = stim.Circuit(dict(_SUPPLEMENT_X)["x09_tag_copy_nonmultiple_fused"])
    ir_x09 = circuit_ir.build_ir(x09)
    cm_x09 = circuit_ir.split_copies(
        ir_x09, lambda op: 2 if op.name == "MPAD" else 0
    )
    mpad_idx = next(i for i, op in enumerate(ir_x09.flat_ops) if op.name == "MPAD")
    assert cm_x09.copy_len[mpad_idx] == 2
    next_off = cm_x09.flat_to_op[mpad_idx + 1]
    assert next_off - cm_x09.flat_to_op[mpad_idx] == 3


def test_i16_step_map_roundtrip() -> None:
    """I16: step_map bidirectional round-trip for Coset skip-2 and Tableau skip-5."""
    for cname, circ in _CORPUS:
        ir = circuit_ir.build_ir(circ, uc=True)
        unrolled = unroll_circuit(circ)
        for skip in (_SKIP_COSET, _SKIP_TAB):
            sm = circuit_ir.step_map(ir, skip)
            assert sm.skip == skip, cname
            want_s2u = [idx for idx, op in enumerate(unrolled) if op.name not in skip]
            assert list(sm.step_to_unrolled) == want_s2u, f"I16 s2u: {cname}"
            assert len(sm.unrolled_to_step) == len(unrolled), cname
            for u_idx, op in enumerate(unrolled):
                assert (sm.unrolled_to_step[u_idx] == -1) == (op.name in skip), cname
            for s_idx, u_idx in enumerate(sm.step_to_unrolled):
                assert sm.unrolled_to_step[u_idx] == s_idx, f"I16 inv: {cname}"


def test_tableau_step_layout_matches_precompiler_and_edge_cases() -> None:
    """Derived tableau_step_layout matches PrecompiledTablesideCircuit and edge cases."""
    checked = 0
    saw_false_bare = False
    for cname, circ in _CORPUS:
        try:
            pre = PrecompiledTablesideCircuit(circ.copy(), True)
        except _CONSUMER_EXC:
            continue
        ir = circuit_ir.build_ir(circ, uc=True)
        sm = circuit_ir.step_map(ir, _SKIP_TAB)
        layout = circuit_ir.tableau_step_layout(ir, sm, pre.step_has_bare_op)
        assert list(layout.step_to_unrolled) == list(pre.step_to_unrolled_idx), cname
        assert list(layout.step_to_quant) == list(pre.step_to_quant_idx), cname
        assert list(layout.step_to_full) == list(pre.step_to_full_idx), cname
        checked += 1
        if any(not b for b in pre.step_has_bare_op):
            saw_false_bare = True
    assert checked >= 20
    assert saw_false_bare

    # Direct synthetic test (kills T1 quant always +1 and T2 full off-by-one).
    c_syn = stim.Circuit("R 0\nTICK\nH 0\nM 0\nDETECTOR rec[-1]\n")
    ir_syn = circuit_ir.build_ir(c_syn)
    sm_syn = circuit_ir.step_map(ir_syn, _SKIP_TAB)
    # Unrolled: [R(0), TICK(1), H(2), M(3), DETECTOR(4)]; steps = [0, 2, 3].
    lay = circuit_ir.tableau_step_layout(ir_syn, sm_syn, [True, False, True, True])
    assert lay.step_to_unrolled == (0, 2, 3)
    assert lay.step_to_quant == (0, 1, 1, 2)
    assert lay.step_to_full == (0, 2, 2, 4)


def test_i18_tag_memo_matches_builder_params() -> None:
    """I18: TagMemo(BUILDER_KEY) matches Builder.params on accepted corpus circuits."""
    checked = 0
    for cname, circ in _CORPUS:
        try:
            builder = Builder(circ, False)
        except _CONSUMER_EXC:
            continue
        ir = circuit_ir.build_ir(circ, uc=False)
        memo = ir_tags.TagMemo(
            lambda op: parse_leakage_tag(op, simulator="tableau"),
            ir_tags.BUILDER_KEY,
        )
        got = [repr(memo.params(op)) for op in ir.flat_ops]
        want = [repr(builder.params[builder.flat_to_op[k]]) for k in range(len(ir.flat_ops))]
        assert got == want, f"I18: {cname}"
        checked += 1
    assert checked >= 15


def test_i18_tag_memo_lazy_none_and_exceptions_not_cached() -> None:
    """I18: lazy evaluation, None caching, key policies, and no exception caching."""
    calls: list[str] = []
    op_m0 = stim.Circuit(f"M[{_PROJ}] 0")[0]
    op_m1 = stim.Circuit(f"M[{_PROJ}] 1")[0]
    op_h0 = stim.Circuit(f"H[{_PROJ}] 0")[0]
    op_m_arg = stim.Circuit(f"M[{_PROJ}](0.1) 0")[0]

    assert ir_tags.BUILDER_KEY(op_m0) == ("M", _PROJ)
    assert ir_tags.VALUE_KEY(op_m0) == op_m0

    def _parse_none(op: stim.CircuitInstruction) -> None:
        calls.append(str(op))

    memo_none = ir_tags.TagMemo(_parse_none, ir_tags.BUILDER_KEY)
    assert calls == []
    assert memo_none.params(op_m0) is None
    assert memo_none.params(op_m1) is None  # same (name, tag), different targets
    assert memo_none.params(op_m_arg) is None  # same (name, tag), different arg
    assert len(calls) == 1
    assert memo_none.params(op_h0) is None  # different name, same tag
    assert len(calls) == 2

    val_calls: list[str] = []

    def _parse_val(op: stim.CircuitInstruction) -> str:
        text = str(op)
        val_calls.append(text)
        return text

    memo_val = ir_tags.TagMemo(_parse_val, ir_tags.VALUE_KEY)
    assert memo_val.params(op_m0) == str(op_m0)
    assert memo_val.params(stim.Circuit(f"M[{_PROJ}] 0")[0]) == str(op_m0)
    assert len(val_calls) == 1
    assert memo_val.params(op_m_arg) == str(op_m_arg)
    assert len(val_calls) == 2

    # Exceptions MUST NOT be cached: repeated call invokes parse again and can recover.
    attempts = 0

    def _flaky_parse(op: stim.CircuitInstruction) -> str:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise ValueError(f"boom {attempts}")
        return f"ok:{op.name}"

    memo_err = ir_tags.TagMemo(_flaky_parse, ir_tags.BUILDER_KEY)
    with pytest.raises(ValueError, match="boom 1"):
        memo_err.params(op_m0)
    with pytest.raises(ValueError, match="boom 2"):
        memo_err.params(op_m0)
    assert memo_err.params(op_m0) == "ok:M"
    assert memo_err.params(op_m1) == "ok:M"
    assert attempts == 3


# ---- Rev-3 Invariants (I18b, I20, I21, I22, I23, I6/I6w, I19) ----------------


def test_i18b_memo_differential_both_directions() -> None:
    """I18b (E1, kills K1: H01, H02): 129 circuits x 3 = 387 accept & reject checks."""
    pairs = _memo_diff_circuits()
    assert len(pairs) == 129
    n_total = 0
    b_ok = b_err = p_ok = p_err = 0
    mismatches: list[str] = []
    for combo, circ in pairs:
        for uc in (True, False):
            head_b = _head_builder_outcome(circ, uc)
            memo_b = _memo_builder_outcome(circ, uc, ir_tags.BUILDER_KEY)
            n_total += 1
            if head_b[0] == "OK":
                b_ok += 1
            else:
                b_err += 1
            if head_b != memo_b:
                mismatches.append(f"builder uc={uc} {combo}: {head_b} != {memo_b}")
        head_p = _head_precompiler_outcome(circ)
        memo_p = _memo_precompiler_outcome(circ, ir_tags.VALUE_KEY)
        n_total += 1
        if head_p[0] == "OK":
            p_ok += 1
        else:
            p_err += 1
        if head_p != memo_p:
            mismatches.append(f"precompiler {combo}: {head_p} != {memo_p}")
    assert n_total == 387
    assert b_ok > 0 and b_err > 0 and p_ok > 0 and p_err > 0
    assert not mismatches, "\n".join(mismatches[:10])


def test_i20_dataclasses_frozen_and_hashable() -> None:
    """I20 (E2, kills K2: G119, G120, G132, G137, G141): all 5 IR records."""
    assert issubclass(circuit_ir.IRInstr, tuple)
    assert hasattr(circuit_ir.IRInstr, "_fields")
    assert circuit_ir.IRInstr.__hash__ is not None
    dc_classes: tuple[Any, ...] = (
        circuit_ir.CircuitIR,
        circuit_ir.StepMap,
        circuit_ir.TableauStepLayout,
        circuit_ir.CopyMap,
    )
    for cls in dc_classes:
        cls_any: Any = cls
        assert dataclasses.is_dataclass(cls), cls_any.__name__
        assert cls_any.__dataclass_params__.frozen is True, f"{cls_any.__name__} not frozen"
        assert cls_any.__hash__ is not None, f"{cls_any.__name__} unhashable"

    ir = circuit_ir.build_ir(
        stim.Circuit("R 0 1\nREPEAT 2 {\nH 0\n}\nM 0 1"), uc=True
    )
    sm = circuit_ir.step_map(ir, frozenset({"QUBIT_COORDS"}))
    lay = circuit_ir.tableau_step_layout(ir, sm, [True] * len(sm.step_to_unrolled))
    cm = circuit_ir.split_copies(ir, lambda _op: 0)

    # All instances hashable and usable as dict keys; duplicate-op indices stay distinct.
    table: dict[Hashable, str] = {
        ir.instrs[0]: "i0",
        ir.instrs[1]: "h0",
        ir.instrs[2]: "h1",
        ir: "ir",
        sm: "sm",
        lay: "lay",
        cm: "cm",
    }
    assert len(table) == 7

    # Mutation must raise on every single one of the 5 record types.
    for attr_name in ("unrolled_index", "flat_index", "flat_offset"):
        with pytest.raises(AttributeError):
            setattr(ir.instrs[0], attr_name, 99)
    dc_mutations: tuple[tuple[Any, str, Any], ...] = (
        (ir, "num_qubits", 99),
        (ir, "flat_ops", ()),
        (sm, "skip", frozenset()),
        (lay, "step_to_unrolled", ()),
        (cm, "copy_len", ()),
    )
    for instance, attr_name, val in dc_mutations:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, attr_name, val)


def test_i21_op_list_edge_cases_vs_stim() -> None:
    """I21 (E2, kills K3: G144, G145, H05): op-list edge cases vs stim.Circuit."""
    cases = (
        ("empty", ""),
        ("coords_only", "QUBIT_COORDS(0, 0) 5"),
        ("coords_then_h", "QUBIT_COORDS(1, 1) 3\nH 0"),
        ("h_then_coords", "H 0\nQUBIT_COORDS(1, 1) 3\nM 0"),
        ("shift_only", "SHIFT_COORDS(1)"),
        ("tick_only", "TICK"),
        ("mpad_only", "MPAD 0 1"),
        ("detector_only", "M 0\nDETECTOR rec[-1]"),
        ("mpp_pauli", "MPP !X0*Y2"),
        ("sweep_feedback", "CX sweep[0] 1"),
    )
    for label, text in cases:
        ops = list(stim.Circuit(text))
        ref = _rebuilt_circuit(ops)
        ir_list = circuit_ir.build_ir(ops)
        assert (
            ir_list.num_qubits,
            ir_list.num_measurements,
            ir_list.has_repeat,
        ) == (ref.num_qubits, ref.num_measurements, False), label
        assert ir_list.from_op_list is True, label
        assert ir_list.flat_ops == (), label
        assert all(i.flat_index == -1 for i in ir_list.instrs), label
        assert all(i.flat_offset == 0 for i in ir_list.instrs), label
        assert tuple(i.op for i in ir_list.instrs) == tuple(ops), label


def test_i22_explicit_category_table_and_behavioural_checks() -> None:
    """I22 (E2, kills K4: H04): 100% stim.gate_data() table, behavioural check, build_ir."""
    all_gates = set(stim.gate_data())
    missing = sorted(all_gates - set(EXPLICIT_CATEGORY))
    extra = sorted(set(EXPLICIT_CATEGORY) - all_gates)
    assert not missing, f"Missing gates in EXPLICIT_CATEGORY: {missing}"
    assert not extra, f"Unknown gates in EXPLICIT_CATEGORY: {extra}"

    skip_behav = frozenset(
        {"REPEAT", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK"}
    )
    for name, cat in sorted(EXPLICIT_CATEGORY.items()):
        if name in skip_behav:
            continue
        circ = stim.Circuit("R 0 1 2\n" + _gate_text(name))
        has_meas = circ.num_measurements > 0
        assert has_meas == (
            cat in ("mpad", "herald", "measure_reset", "measure")
        ), f"{name}: meas={circ.num_measurements} vs cat={cat}"
        if (
            cat in ("reset", "measure_reset", "measure")
            and stim.gate_data(name).is_single_qubit_gate
        ):
            assert _forces_state(name) == (
                cat in ("reset", "measure_reset")
            ), f"{name}: reset probe contradicts cat={cat}"
        if cat == "unitary" and name not in ("SPP", "SPP_DAG"):
            assert stim.Tableau.from_named_gate(name) is not None, name
        if (
            cat in ("noise", "measure", "reset", "measure_reset", "herald", "mpad")
            and name != "MPAD"
        ):
            with pytest.raises((IndexError, ValueError)):
                stim.Tableau.from_named_gate(name)

    assert circuit_ir.classify_category("REPEAT") == circuit_ir.Category.OTHER
    non_repeat = [n for n in sorted(stim.gate_data()) if n != "REPEAT"]
    assert len(non_repeat) == 80
    for name in non_repeat:
        exp = EXPLICIT_CATEGORY[name]
        assert circuit_ir.classify_category(name).value == exp, name
        ir = circuit_ir.build_ir(_gate_circuit(name), uc=True)
        probed = [i for i in ir.instrs if i.op.name == name]
        assert len(probed) >= 1, name
        assert all(i.category.value == exp for i in probed), name


def test_i23_untagged_filter_vs_live_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I23 (E5): independent consumer gate-table oracle over all 80 non-REPEAT gates."""
    real_trace = mb.trace_flows_cpp
    tableau_probes = 0
    dem_probes = 0
    dem_skipped: list[str] = []
    allowed_tab: dict[str, set[int]] = {
        "one_qubit": {1, 8},
        "two_qubit": {2, 8},
        "correlated": {8},
        "pauli_terms": {8},
    }

    captured: list[list[Any]] = []

    def _spy(**kw: Any) -> Any:
        captured.append(list(kw["ops_arr"]))
        return real_trace(**kw)

    monkeypatch.setattr(mb, "trace_flows_cpp", _spy)
    for name in sorted(stim.gate_data()):
        if name == "REPEAT":
            continue
        circ = _gate_circuit(name)
        ir = circuit_ir.build_ir(circ, uc=True)
        probed = [i for i in ir.instrs if i.op.name == name] or [ir.instrs[-1]]
        cls = probed[0].untagged_filter.value

        pre = PrecompiledTablesideCircuit(circ.copy(), True)
        tk = [s[0] for s in pre.steps_py][2:]
        tableau_probes += 1
        kinds = set(tk) if name == "ELSE_CORRELATED_ERROR" else set(tk[:1])
        if cls == "none":
            assert kinds <= {0}, f"T0 {name}: none but tableau kind {tk}"
        else:
            assert tk, f"T0 {name}: {cls} but no tableau step"
            assert set(tk) <= allowed_tab[cls], f"T1 {name}: {cls} vs {tk}"

        captured.clear()
        try:
            mb.Builder(circ, True).build(oracle=True)
        except NotImplementedError:
            dem_skipped.append(name)
            continue

        dk = [o.kind for o in captured[0]][2:] if captured else []
        dem_probes += 1
        if cls in ("one_qubit", "two_qubit"):
            assert set(dk) <= {5, 6, 7}, f"D1 {name}: {cls} vs DEM {dk}"
        if cls == "none":
            assert 5 not in dk and 6 not in dk, f"D2 {name}: none vs DEM {dk}"

    assert tableau_probes == 80
    assert dem_probes == 78
    assert sorted(dem_skipped) == ["SPP", "SPP_DAG"]


def test_i6_circuit_ir_purity_and_single_tag_read() -> None:
    """I6a-d: static purity of util/circuit_ir.py and single .tag copy into IRInstr."""
    src_file = inspect.getsourcefile(circuit_ir)
    assert src_file is not None
    src = Path(src_file).read_text(encoding="utf-8")
    assert _scan_circuit_ir_i6(src) == []


def test_i6w_recursive_util_and_ir_tags_purity() -> None:
    """I6w (E4): recursive scan over all non-test util/**/*.py and op_handlers/ir_tags.py."""
    src_file = inspect.getsourcefile(circuit_ir)
    assert src_file is not None
    util_dir = Path(src_file).parent
    ir_tags_path = util_dir.parent / "op_handlers" / "ir_tags.py"
    assert ir_tags_path.exists()
    assert _scan_tree_i6w(util_dir, ir_tags_path) == []


def test_i6w_and_i6c_self_sensitivity(tmp_path: Path) -> None:
    """Self-test: I6w catches all 4 evasion patterns, and I6c handles positional/keyword."""
    pkg = tmp_path / "stimside"
    util_dir = pkg / "util"
    op_dir = pkg / "op_handlers"
    util_dir.mkdir(parents=True)
    op_dir.mkdir(parents=True)

    clean_ir = (
        "def f(op):\n"
        "    return IRInstr(0, op, op.tag)\n"
    )
    clean_ir_kw = (
        "def f(op):\n"
        "    return IRInstr(unrolled_index=0, op=op, tag=op.tag)\n"
    )
    (util_dir / "circuit_ir.py").write_text(clean_ir, encoding="utf-8")
    (op_dir / "ir_tags.py").write_text("X = 1\n", encoding="utf-8")

    # Clean control passes both I6w and I6c (positional and keyword forms).
    assert _scan_tree_i6w(util_dir, op_dir / "ir_tags.py") == []
    assert _scan_circuit_ir_i6(clean_ir) == []
    assert _scan_circuit_ir_i6(clean_ir_kw) == []

    # I6c catches conditional .tag read, duplicate .tag read, transformed .tag, and literals.
    assert "I6c-tag-only-copied" in _scan_circuit_ir_i6(
        "def f(op):\n    return IRInstr(0, op, op.tag if op.tag else '')\n"
    )
    assert "I6c-tag-only-copied" in _scan_circuit_ir_i6(
        "def f(op):\n    return IRInstr(0, op, op.tag.strip())\n"
    )
    assert "I6b-no-tag-literal" in _scan_circuit_ir_i6(
        clean_ir + 'TAG = "LEAKAGE"\n'
    )

    # Evasion 1: subpkg (util/ir_sub/core.py mentions ctypes + OMEGA)
    sub = util_dir / "ir_sub" / "core.py"
    sub.parent.mkdir()
    sub.write_text("import ctypes\nOMEGA_Q = 1\n", encoding="utf-8")
    assert any("ir_sub/core.py" in v for v in _scan_tree_i6w(util_dir, op_dir / "ir_tags.py"))
    sub.unlink()

    # Evasion 2: sibling (util/circuit_ir_helpers.py mentions ctypes + OMEGA)
    sib = util_dir / "circuit_ir_helpers.py"
    sib.write_text("import ctypes\nOMEGA_Q = 1\n", encoding="utf-8")
    assert any(
        "circuit_ir_helpers.py" in v
        for v in _scan_tree_i6w(util_dir, op_dir / "ir_tags.py")
    )
    sib.unlink()

    # Evasion 3: splitstr (op_handlers/ir_tags.py uses __import__ with split string)
    (op_dir / "ir_tags.py").write_text(
        'X = 1\n_m = __import__("ct" + "ypes")\n', encoding="utf-8"
    )
    assert any("ir_tags.py" in v for v in _scan_tree_i6w(util_dir, op_dir / "ir_tags.py"))
    (op_dir / "ir_tags.py").write_text("X = 1\n", encoding="utf-8")

    # Evasion 4: omega_str (util/circuit_ir.py contains "OMEGA_Q" string literal)
    (util_dir / "circuit_ir.py").write_text(
        clean_ir + '_s = "OMEGA_Q"\n', encoding="utf-8"
    )
    assert any(
        "circuit_ir.py" in v for v in _scan_tree_i6w(util_dir, op_dir / "ir_tags.py")
    )

    # Evasion 5: string constant starting with "stimside"
    (util_dir / "circuit_ir.py").write_text(
        clean_ir + '_s = "stimside.op_handlers"\n', encoding="utf-8"
    )
    assert any(
        "I6w-no-stimside-string" in v
        for v in _scan_tree_i6w(util_dir, op_dir / "ir_tags.py")
    )


def test_i6c_tag_eq_comparison_self_sensitivity() -> None:
    """Self-test: I6c allows symmetric ``.tag == .tag`` and rejects tag inspection."""
    clean_base = "def f(op, cur):\n    return IRInstr(0, op, op.tag)\n"
    assert _scan_circuit_ir_i6(
        clean_base + "def eq(a, b):\n    return a.tag == b.tag\n"
    ) == []
    assert _scan_circuit_ir_i6(
        clean_base + "def neq(a, b):\n    if not (a.tag == b.tag):\n        return False\n"
    ) == []
    bad_snippets = (
        clean_base + 'def b1(op):\n    return op.tag == ""\n',
        clean_base + 'def b2(op):\n    return "" == op.tag\n',
        clean_base + "def b3(a, b):\n    return a.tag != b.tag\n",
        clean_base + "def b4(a, b):\n    return a.tag < b.tag\n",
        clean_base + "def b5(a, b):\n    return a.tag.strip() == b.tag\n",
        clean_base + "def b6(a, b, c):\n    return a.tag == b.tag == c.tag\n",
        "def b7(op, c):\n    if c:\n        return IRInstr(0, op, op.tag)\n",
        "def b8(a, b):\n    return a.tag == b.tag\n",
    )
    for snippet in bad_snippets:
        assert "I6c-tag-only-copied" in _scan_circuit_ir_i6(snippet), snippet


def test_i19_build_ir_linear_complexity() -> None:
    """Guard against O(N^2) regressions across circuit, op-list, and alignment paths."""

    def _best_time(
        src: stim.Circuit | list[stim.CircuitInstruction],
        *,
        uc: bool,
        align: bool = False,
    ) -> float:
        best = float("inf")
        for _ in range(3):
            t0 = time.process_time()
            ir = circuit_ir.build_ir(src, uc=uc)
            if align:
                circuit_ir.unrolled_to_flat(ir)
            best = min(best, time.process_time() - t0)
        return best

    # Guard 1 (kills quadratic prefix sum immediately without waiting on larger N).
    c_small = stim.Circuit("R 0 1\nREPEAT 500 {\n H 0\n CX 0 1\n M 1\n}")
    c_large = stim.Circuit("R 0 1\nREPEAT 4000 {\n H 0\n CX 0 1\n M 1\n}")
    t_small = _best_time(c_small, uc=True)
    t_large = _best_time(c_large, uc=True)
    assert t_large / t_small < 25.0, (
        f"build_ir scaling ratio {t_large / t_small:.1f}x >= 25x "
        f"(500={t_small:.4f}s, 4000={t_large:.4f}s)"
    )

    # Guard 2 (kills low-constant C-level O(N^2) ops[:i] slice copies on circuit path).
    r_small = stim.Circuit("REPEAT 500 {\n TICK\n}")
    r_large = stim.Circuit("REPEAT 16000 {\n TICK\n}")
    rt_small = _best_time(r_small, uc=False)
    rt_large = _best_time(r_large, uc=False)
    assert rt_large / rt_small < 72.0, (
        f"build_ir REPEAT TICK 32x ratio {rt_large / rt_small:.1f}x >= 72x "
        f"(500={rt_small:.4f}s, 16000={rt_large:.4f}s)"
    )

    # Guard 3 (same 32x check on op-list path with minimal linear per-op overhead).
    tick = stim.CircuitInstruction("TICK", [])
    lt_small = _best_time([tick] * 500, uc=False)
    lt_large = _best_time([tick] * 16000, uc=False)
    assert lt_large / lt_small < 80.0, (
        f"build_ir op-list TICK 32x ratio {lt_large / lt_small:.1f}x >= 80x "
        f"(500={lt_small:.4f}s, 16000={lt_large:.4f}s)"
    )

    # Guard 4 (16x fused REPEAT: verifies alignment avoids O(N^2) stringification).
    fh_small = stim.Circuit("REPEAT 500 {\n H 0\n}")
    fh_large = stim.Circuit("REPEAT 8000 {\n H 0\n}")
    ft_small = _best_time(fh_small, uc=False, align=True)
    ft_large = _best_time(fh_large, uc=False, align=True)
    assert ft_large / ft_small < 50.0, (
        f"build_ir fused REPEAT H 16x ratio {ft_large / ft_small:.1f}x >= 50x "
        f"(500={ft_small:.4f}s, 8000={ft_large:.4f}s)"
    )

    # Guard 5 (16x zero-target fused REPEAT: verifies linear scaling on empty REPEAT).
    zh_small = stim.Circuit("REPEAT 500 {\n H\n}")
    zh_large = stim.Circuit("REPEAT 8000 {\n H\n}")
    zt_small = _best_time(zh_small, uc=False, align=True)
    zt_large = _best_time(zh_large, uc=False, align=True)
    assert zt_large / zt_small < 50.0, (
        f"build_ir empty-target REPEAT H 16x ratio {zt_large / zt_small:.1f}x >= 50x "
        f"(500={zt_small:.4f}s, 8000={zt_large:.4f}s)"
    )

    # Guard 6 (wide flat op + many zero-target fused ops: kills O(N*W) regressions).
    def _wide_zero(n: int) -> stim.Circuit:
        q = " ".join(str(i) for i in range(n))
        return stim.Circuit(f"H {q}\nREPEAT {n} {{\n SHIFT_COORDS(1)\n H\n}}\n")

    wz_small = _best_time(_wide_zero(200), uc=False, align=True)
    wz_large = _best_time(_wide_zero(3200), uc=False, align=True)
    assert wz_large / wz_small < 40.0, (
        f"build_ir wide+zero-target 16x ratio {wz_large / wz_small:.1f}x >= 40x "
        f"(200={wz_small:.4f}s, 3200={wz_large:.4f}s)"
    )


# ---- Direct R-f field, error-path, and export coverage -----------------------


def _bad_flat_circuit(
    unrolled_ops: Sequence[stim.CircuitInstruction],
    flat_source: stim.Circuit,
) -> stim.Circuit:
    """Build a stim.Circuit that unrolls to ``unrolled_ops`` with spoofed ``flattened()``."""

    class _SpoofedFlatCircuit(stim.Circuit):
        def copy(self) -> stim.Circuit:
            return self

        def flattened(self) -> stim.Circuit:
            return flat_source.flattened()

    circ = _SpoofedFlatCircuit()
    for op in unrolled_ops:
        sub = stim.Circuit()
        sub.append(op)
        circ.append(stim.CircuitRepeatBlock(1, sub))
    return circ


def test_rf_build_ir_flattened_too_few_ops_raises() -> None:
    """Direct error-branch test for alignment when flat_ops runs out."""
    flat_src = stim.Circuit("H 0\n")
    for extra_ops in (
        [stim.Circuit("H 0")[0], stim.Circuit("X 0")[0]],
        [stim.Circuit("H 0")[0], stim.Circuit("H 1")[0]],
    ):
        bad = _bad_flat_circuit(extra_ops, flat_src)
        ir = circuit_ir.build_ir(bad)
        assert [i.op for i in ir.instrs] == extra_ops
        try:
            circuit_ir.unrolled_to_flat(ir)
        except ValueError as exc:
            assert "too few for the unrolled circuit" in str(exc)
        except IndexError as exc:
            pytest.fail(
                f"expected ValueError for short flattened(), got IndexError: {exc}"
            )
        else:
            pytest.fail("expected ValueError for short flattened(), but build_ir succeeded")


def test_rf_enum_str_mixin_contract() -> None:
    """Verify Category and UntaggedFilter preserve their (str, Enum) mixin contract."""
    assert issubclass(circuit_ir.Category, str)
    assert issubclass(circuit_ir.UntaggedFilter, str)
    for member in circuit_ir.Category:
        assert isinstance(member, str)
        assert member == member.value
        assert hash(member) == hash(member.value)
        assert {member.value: 1}[member] == 1
    for u_member in circuit_ir.UntaggedFilter:
        assert isinstance(u_member, str)
        assert u_member == u_member.value
        assert hash(u_member) == hash(u_member.value)
        assert {u_member.value: 1}[u_member] == 1


def test_rf_classify_untagged_multi_qubit_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify defensive UntaggedFilter.NONE fallback for non-1Q/non-2Q gates."""
    cases = (
        types.SimpleNamespace(
            name="SYNTHETIC_3Q_GATE",
            produces_measurements=False,
            is_noisy_gate=False,
            is_unitary=True,
            is_single_qubit_gate=False,
            is_two_qubit_gate=False,
        ),
        types.SimpleNamespace(
            name="SYNTHETIC_3Q_GATE",
            produces_measurements=False,
            is_noisy_gate=True,
            is_unitary=False,
            is_single_qubit_gate=False,
            is_two_qubit_gate=False,
        ),
    )
    for stub in cases:
        monkeypatch.setattr(stim, "gate_data", lambda _name, _s=stub: _s)
        got_on = circuit_ir.classify_untagged("SYNTHETIC_3Q_GATE", uc=True)
        got_off = circuit_ir.classify_untagged("SYNTHETIC_3Q_GATE", uc=False)
        assert got_on is circuit_ir.UntaggedFilter.NONE
        assert got_off is circuit_ir.UntaggedFilter.NONE


def test_rf_direct_fields_enums_and_empty_circuit() -> None:
    """R-f: verify every exported enum member, dataclass field, and empty circuit."""
    assert {c.value for c in circuit_ir.Category} == {
        "annotation",
        "mpad",
        "herald",
        "measure_reset",
        "measure",
        "reset",
        "unitary",
        "noise",
        "other",
    }
    assert {u.value for u in circuit_ir.UntaggedFilter} == {
        "none",
        "one_qubit",
        "two_qubit",
        "correlated",
        "pauli_terms",
    }
    assert circuit_ir.IRInstr._fields == (
        "unrolled_index",
        "op",
        "tag",
        "category",
        "untagged_filter",
        "num_measurements",
        "first_measurement",
        "basis",
        "is_product_measurement",
        "arity",
        "flat_box",
    )
    assert isinstance(circuit_ir.IRInstr.has_classical_control, property)
    assert isinstance(circuit_ir.IRInstr._flat, property)
    assert tuple(f.name for f in dataclasses.fields(circuit_ir.CircuitIR)) == (
        "instrs",
        "num_qubits",
        "num_measurements",
        "has_repeat",
        "uc",
        "from_op_list",
        "_flat",
    )
    assert isinstance(circuit_ir.IRInstr.flat_index, property)
    assert isinstance(circuit_ir.IRInstr.flat_offset, property)
    assert isinstance(circuit_ir.CircuitIR.flat_ops, property)
    assert tuple(f.name for f in dataclasses.fields(circuit_ir.StepMap)) == (
        "skip",
        "step_to_unrolled",
        "unrolled_to_step",
    )
    assert tuple(
        f.name for f in dataclasses.fields(circuit_ir.TableauStepLayout)
    ) == ("step_to_unrolled", "step_to_quant", "step_to_full")
    assert tuple(f.name for f in dataclasses.fields(circuit_ir.CopyMap)) == (
        "ops",
        "flat_to_op",
        "copy_len",
    )

    empty = circuit_ir.build_ir(stim.Circuit())
    assert empty.instrs == ()
    assert empty.flat_ops == ()
    assert empty.num_qubits == 0
    assert empty.num_measurements == 0
    assert empty.has_repeat is False
    assert empty.uc is False
    assert empty.from_op_list is False
    assert circuit_ir.unrolled_to_flat(empty) == []
    sm = circuit_ir.step_map(empty, _SKIP_TAB)
    assert sm == circuit_ir.StepMap(_SKIP_TAB, (), ())
    lay = circuit_ir.tableau_step_layout(empty, sm, ())
    assert lay == circuit_ir.TableauStepLayout((), (0,), (0,))
    cm = circuit_ir.split_copies(empty, lambda _op: 0)
    assert cm == circuit_ir.CopyMap((), (), ())


def test_i6_and_i6w_extended_evasion_sensitivity(tmp_path: Path) -> None:
    """Self-test: tightened I6/I6w catch exec/eval/getattr/sys.modules/pydoc/TYPE_CHECKING."""
    pkg = tmp_path / "stimside"
    util_dir = pkg / "util"
    op_dir = pkg / "op_handlers"
    util_dir.mkdir(parents=True)
    op_dir.mkdir(parents=True)
    clean_ir = "def f(op):\n    return IRInstr(0, op, op.tag)\n"
    ir_file = util_dir / "circuit_ir.py"
    tag_file = op_dir / "ir_tags.py"
    tag_file.write_text("X = 1\n", encoding="utf-8")

    circuit_evasions = (
        ("E02_fn_body_oph", "def g():\n    from stimside.op_handlers import ir_tags\n"),
        ("E04_exec", "def g():\n    exec('x=1')\n"),
        ("E05_eval", "def g():\n    return eval('1')\n"),
        ("E06_builtins_getattr", "import builtins\ng = getattr(builtins, '__imp' + 'ort__')\n"),
        ("E07_sys_modules", "import sys\ng = sys.modules.get('x')\n"),
        ("E09_rel_oph", "def g():\n    from ..op_handlers import ir_tags\n"),
        ("E10_getattr_lit", 'def g(op):\n    return getattr(op, "tag")\n'),
        ("E10_getattr_split", "def g(op):\n    return getattr(op, 't' + 'ag')\n"),
        ("E11_pydoc", "import pydoc\ng = pydoc.locate('x')\n"),
        ("E12_tc_rebind", "TYPE_CHECKING = True\nif TYPE_CHECKING:\n    pass\n"),
    )
    for label, snippet in circuit_evasions:
        src = clean_ir + "\n" + snippet
        assert _scan_circuit_ir_i6(src), f"I6 missed {label}"
        ir_file.write_text(src, encoding="utf-8")
        assert _scan_tree_i6w(util_dir, tag_file), f"I6w missed {label}"

    # E03: cross-layer import inside util subpackage
    ir_file.write_text(clean_ir, encoding="utf-8")
    sub = util_dir / "sub_evil" / "mod.py"
    sub.parent.mkdir()
    (sub.parent / "__init__.py").write_text("", encoding="utf-8")
    sub.write_text("def g():\n    from stimside.op_handlers import ir_tags\n", encoding="utf-8")
    assert _scan_tree_i6w(util_dir, tag_file), "I6w missed E03"
    sub.unlink()

    # E13 & E14: ir_tags.py importing engine or calling exec
    for label, snippet in (
        ("E13_engine", "def g():\n    from stimside._tableau import runner\n"),
        ("E14_exec", "def g():\n    exec('x=1')\n"),
    ):
        tag_file.write_text(snippet, encoding="utf-8")
        assert _scan_tree_i6w(util_dir, tag_file), f"I6w missed {label}"
    tag_file.write_text("X = 1\n", encoding="utf-8")
    assert _scan_tree_i6w(util_dir, tag_file) == []


def test_rf_alias_and_case_canonicalization() -> None:
    """R8-C-C5-L1: all Stim aliases and lowercase gate names match canonical names."""
    all_gd = stim.gate_data()
    alias_count = 0
    for canon, gd in sorted(all_gd.items()):
        for alias in gd.aliases:
            alias_count += 1
            assert circuit_ir.classify_category(alias) == circuit_ir.classify_category(canon), (
                alias
            )
            assert circuit_ir.classify_untagged(
                alias, uc=True
            ) == circuit_ir.classify_untagged(canon, uc=True), alias
        low = canon.lower()
        assert circuit_ir.classify_category(low) == circuit_ir.classify_category(canon), low
        assert circuit_ir.classify_untagged(
            low, uc=True
        ) == circuit_ir.classify_untagged(canon, uc=True), low
    assert alias_count >= 90
    assert (
        circuit_ir.classify_untagged("CORRELATED_ERROR", uc=True)
        is circuit_ir.UntaggedFilter.CORRELATED
    )
    assert (
        circuit_ir.classify_untagged("correlated_error", uc=True)
        is circuit_ir.UntaggedFilter.CORRELATED
    )
    assert circuit_ir.classify_category("tick") is circuit_ir.Category.ANNOTATION
    assert circuit_ir.classify_category("mpad") is circuit_ir.Category.MPAD
    assert circuit_ir.classify_category("heralded_erase") is circuit_ir.Category.HERALD
    with pytest.raises(IndexError):
        circuit_ir.classify_category("NOT_A_REAL_STIM_GATE")
    with pytest.raises(IndexError):
        circuit_ir.classify_untagged("NOT_A_REAL_STIM_GATE", uc=True)


def _assert_valid_alignment(label: str, circ: stim.Circuit) -> circuit_ir.CircuitIR:
    ir = circuit_ir.build_ir(circ, uc=True)
    assert ir.flat_ops == tuple(circ.flattened()), label
    by_flat: dict[int, list[circuit_ir.IRInstr]] = {}
    prev_f = -1
    for instr in ir.instrs:
        if instr.op.name == "SHIFT_COORDS":
            assert (instr.flat_index, instr.flat_offset) == (-1, 0), label
            continue
        f = instr.flat_index
        assert 0 <= f < len(ir.flat_ops), f"{label}: bad flat_index {f}"
        assert f >= prev_f, f"{label}: non-monotonic {f} < {prev_f}"
        prev_f = f
        fop = ir.flat_ops[f]
        assert fop.name == instr.op.name, f"{label}: {instr.op} vs {fop}"
        assert fop.tag == instr.op.tag, f"{label}: tag {instr.op.tag!r} vs {fop.tag!r}"
        if instr.op.name not in ("DETECTOR", "QUBIT_COORDS"):
            assert fop.gate_args_copy() == instr.op.gate_args_copy(), label
        tgts = instr.op.targets_copy()
        assert fop.targets_copy()[instr.flat_offset : instr.flat_offset + len(tgts)] == tgts
        by_flat.setdefault(f, []).append(instr)
    assert set(by_flat) == set(range(len(ir.flat_ops))), label
    for f, grp in by_flat.items():
        off = 0
        for ins in grp:
            assert ins.flat_offset == off, f"{label}: offset f={f}"
            off += len(ins.op.targets_copy())
        assert off == len(ir.flat_ops[f].targets_copy()), f"{label}: total f={f}"
    return ir


_ZERO_TARGET_WITNESSES: tuple[tuple[str, str, list[int]], ...] = (
    (
        "c_rep",
        "R 0 1\nREPEAT 2 {\n  X\n}\nTICK\nDETECTOR\nTICK\nZ 1\nM 0 1\nDETECTOR rec[-1]\n",
        [0, 1, 1, 2, 3, 4, 5, 6, 7],
    ),
    (
        "c_shift",
        "R 0\nX 0\nSHIFT_COORDS(1)\nX\nZ 0\nM 0\nDETECTOR rec[-1]\n",
        [0, 1, -1, 1, 2, 3, 4],
    ),
    ("W1", "M 0\nREPEAT 1 {\n  M\n}\nH 1\n", [0, 0, 1]),
    ("W2", "H 0\nREPEAT 1 {\n  H\n}\nX 1\n", [0, 0, 1]),
    ("W4", "REPEAT 2 {\n  M\n}\n", [0, 0]),
    ("W5", "REPEAT 2 {\n  H\n}\n", [0, 0]),
    ("W11", "MPAD 0\nREPEAT 1 {\n  MPAD\n}\nH 0\n", [0, 0, 1]),
    ("W12", "H 0\nSHIFT_COORDS(1)\nH\nX 1\n", [0, -1, 0, 1]),
    (
        "W15",
        (
            "R 0 1\nI_ERROR[LEAKAGE_TRANSITION_1: (0.1, U-->2)] 0\n"
            "H 0\nREPEAT 1 {\n  H\n}\nX 1\nM 0 1\nDETECTOR rec[-1]\n"
        ),
        [0, 1, 2, 2, 3, 4, 5],
    ),
    (
        "W16",
        (
            "R 0 1\nI_ERROR[LEAKAGE_TRANSITION_1: (0.1, U-->2)] 0\n"
            "H 0\nSHIFT_COORDS(1)\nH\nX 1\nM 0 1\nDETECTOR rec[-1]\n"
        ),
        [0, 1, 2, -1, 2, 3, 4, 5],
    ),
    ("W17", "H\nSHIFT_COORDS(1)\nH\n", [0, -1, 0]),
    ("W19", "H 0\nSHIFT_COORDS(1)\nH\n", [0, -1, 0]),
    ("w5_repeat2_empty_H", "R 0\nREPEAT 2 {\n  H\n}\nM 0\nDETECTOR rec[-1]\n", [0, 1, 1, 2, 3]),
    ("fuzz_52", "M 0\nREPEAT 2 {\n}\nM\nOBSERVABLE_INCLUDE(0)\n", [0, 0, 1]),
    ("empty_to_nonempty_shift", "H\nSHIFT_COORDS(1)\nH 0\n", [0, -1, 0]),
    ("empty_to_nonempty_rep1", "REPEAT 1 {\n  H\n}\nH 0\nX 1\n", [0, 0, 1]),
    (
        "nonempty_empty_nonempty",
        "H 0\nSHIFT_COORDS(1)\nH\nSHIFT_COORDS(1)\nH 1\n",
        [0, -1, 0, -1, 0],
    ),
    (
        "empty_empty_nonempty",
        "H\nSHIFT_COORDS(1)\nH\nSHIFT_COORDS(1)\nH 0\n",
        [0, -1, 0, -1, 0],
    ),
    (
        "nonempty_empty_empty",
        "H 0\nSHIFT_COORDS(1)\nH\nSHIFT_COORDS(1)\nH\n",
        [0, -1, 0, -1, 0],
    ),
    ("tag_diff_a_vs_empty", "H[a] 0\nSHIFT_COORDS(1)\nH\n", [0, -1, 1]),
    ("tag_diff_a_vs_b", "H[a] 0\nSHIFT_COORDS(1)\nH[b]\n", [0, -1, 1]),
    ("tag_same_a_vs_a", "H[a]\nSHIFT_COORDS(1)\nH[a]\n", [0, -1, 0]),
    ("args_diff_xerr", "X_ERROR(0.01) 0\nSHIFT_COORDS(1)\nX_ERROR(0.02)\n", [0, -1, 1]),
    ("args_same_xerr", "X_ERROR(0.01) 0\nSHIFT_COORDS(1)\nX_ERROR(0.01)\n", [0, -1, 0]),
    ("args_m0_vs_m", "M(0.0) 0\nSHIFT_COORDS(1)\nM\n", [0, -1, 1]),
    ("neg_zero_float", "X_ERROR(0.0) 0\nSHIFT_COORDS(1)\nX_ERROR(-0.0)\n", [0, -1, 0]),
    (
        "mpp_comb_nen",
        "MPP X0*Z1\nSHIFT_COORDS(1)\nMPP\nSHIFT_COORDS(1)\nMPP Y2*X3\n",
        [0, -1, 0, -1, 0],
    ),
    ("spp_comb_ne", "SPP X0*Z1\nSHIFT_COORDS(1)\nSPP\n", [0, -1, 0]),
    ("spp_dag_comb_en", "SPP_DAG\nSHIFT_COORDS(1)\nSPP_DAG X0*Z1\n", [0, -1, 0]),
    (
        "herald_erase_ne",
        "HERALDED_ERASE(0.1) 0\nSHIFT_COORDS(1)\nHERALDED_ERASE(0.1)\n",
        [0, -1, 0],
    ),
    ("shifted_qubit_coords", "SHIFT_COORDS(1, 2)\nQUBIT_COORDS(0, 0) 0\n", [-1, 0]),
    (
        "shifted_qubit_coords_repeat",
        "REPEAT 2 {\n  QUBIT_COORDS(0, 0) 0\n  SHIFT_COORDS(1, 0)\n}\n",
        [0, -1, 1, -1],
    ),
    (
        "fused_tagged_feedback_across_repeat",
        "M 0\nREPEAT 2 {\n  CX[a] rec[-1] 1\n}\n",
        [0, 1, 1],
    ),
    (
        "tagged_detector_after_shift_coords",
        "M 0\nSHIFT_COORDS(1)\nDETECTOR[a](0) rec[-1]\n",
        [0, -1, 1],
    ),
    (
        "fused_tagged_sweep_control",
        "REPEAT 2 {\n  CX[a] sweep[0] 1\n}\n",
        [0, 0],
    ),
)


def test_zero_target_fusion_witnesses_and_offsets() -> None:
    """Exact flat_index and flat_offset for zero-target and tagged fusion cases."""
    for label, txt, want in _ZERO_TARGET_WITNESSES:
        ir = _assert_valid_alignment(label, stim.Circuit(txt))
        assert circuit_ir.unrolled_to_flat(ir) == want, label

    # Escaped special tags must stay bijective ("a]b\\c" != "a\\Cb\\Bc").
    c_esc = stim.Circuit()
    c_esc.append(stim.CircuitInstruction("H", [0], [], tag="a]b\\c"))
    c_esc.append(stim.CircuitInstruction("SHIFT_COORDS", [], [1.0]))
    c_esc.append(stim.CircuitInstruction("H", [], [], tag="a\\Cb\\Bc"))
    c_esc.append(stim.CircuitInstruction("SHIFT_COORDS", [], [1.0]))
    c_esc.append(stim.CircuitInstruction("H", [], [], tag="a\\Cb\\Bc"))
    ir_esc = _assert_valid_alignment("escaped_tags", c_esc)
    assert circuit_ir.unrolled_to_flat(ir_esc) == [0, -1, 1, -1, 1]


def test_zero_target_fusion_all_80_gates() -> None:
    """Verify exact 73 fusing vs 7 non-fusing partition across all 80 Stim gates."""
    non_fusing_seen: list[str] = []
    fusing_seen: list[str] = []
    for name in sorted(stim.gate_data()):
        if name == "REPEAT":
            continue
        gd = stim.gate_data(name)
        k = gd.num_parens_arguments_range.start
        args = [0.0] if name == "OBSERVABLE_INCLUDE" else ([0.01] * k if k else [])
        c = stim.Circuit()
        if name == "ELSE_CORRELATED_ERROR":
            c.append("E", [stim.target_x(0)], [0.1])
        c.append(stim.CircuitInstruction(name, [], args))
        c.append(stim.CircuitInstruction("SHIFT_COORDS", [], [1.0]))
        c.append(stim.CircuitInstruction(name, [], args))
        ir = _assert_valid_alignment(f"gate_{name}", c)
        if name == "SHIFT_COORDS":
            non_fusing_seen.append(name)
            assert ir.flat_ops == ()
        elif len(ir.flat_ops) == (2 if name == "ELSE_CORRELATED_ERROR" else 1):
            fusing_seen.append(name)
        else:
            non_fusing_seen.append(name)
    assert len(fusing_seen) == 73
    assert non_fusing_seen == [
        "DETECTOR",
        "E",
        "ELSE_CORRELATED_ERROR",
        "OBSERVABLE_INCLUDE",
        "QUBIT_COORDS",
        "SHIFT_COORDS",
        "TICK",
    ]


def test_build_ir_alignment_validation_errors() -> None:
    """Verify loud ValueError on every alignment mismatch, overshoot, and extra flat_op."""
    inst = stim.CircuitInstruction
    cases: tuple[tuple[str, str, list[stim.CircuitInstruction]], ...] = (
        ("name_mismatch", "H 0\n", [inst("X", [0])]),
        ("target_mismatch", "H 0\n", [inst("H", [1])]),
        ("fused_second_target_mismatch", "H 0 1\n", [inst("H", [0]), inst("H", [2])]),
        ("overshoot_single", "H 0\n", [inst("H", [0, 1])]),
        ("overshoot_second", "H 0 1\n", [inst("H", [0]), inst("H", [1, 2])]),
        ("tag_mismatch_normal", "H[a] 0\n", [inst("H", [0], tag="b")]),
        ("tag_mismatch_empty_vs_tagged", "H 0\n", [inst("H", [0], tag="a")]),
        (
            "tag_mismatch_fused_second",
            "H[a] 0 1\n",
            [inst("H", [0], tag="a"), inst("H", [1], tag="b")],
        ),
        (
            "tag_mismatch_escaped",
            "H[a\\Cb] 0\n",
            [inst("H", [0], tag="a\\Cb")],
        ),
        (
            "tag_mismatch_shifted_detector",
            "M 0\nSHIFT_COORDS(1)\nDETECTOR[a](2) rec[-1]\n",
            [
                inst("M", [0]),
                inst("SHIFT_COORDS", [], [1.0]),
                inst("DETECTOR", [stim.target_rec(-1)], [2.0], tag="b"),
            ],
        ),
        (
            "tag_mismatch_shifted_qubit_coords",
            "SHIFT_COORDS(1)\nQUBIT_COORDS[a](2) 0\n",
            [
                inst("SHIFT_COORDS", [], [1.0]),
                inst("QUBIT_COORDS", [0], [2.0], tag="b"),
            ],
        ),
        ("args_mismatch_xerr", "X_ERROR(0.01) 0\n", [inst("X_ERROR", [0], [0.02])]),
        ("args_mismatch_m", "M(0.0) 0\n", [inst("M", [0])]),
        (
            "args_mismatch_obs",
            "M 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
            [inst("M", [0]), inst("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], [1.0])],
        ),
        ("trailing_extra_flat_op", "H 0\nX 1\n", [inst("H", [0])]),
        ("empty_ops_nonempty_flat", "H 0\n", []),
        ("shift_only_nonempty_flat", "H 0\n", [inst("SHIFT_COORDS", [], [1.0])]),
        ("partial_flat_leftover", "H 0 1\n", [inst("H", [0])]),
        ("empty_op_partial_flat", "H 0\n", [inst("H", [])]),
        (
            "nonfusing_detector_split",
            "M 0\nDETECTOR rec[-1]\n",
            [inst("M", [0]), inst("DETECTOR", []), inst("DETECTOR", [stim.target_rec(-1)])],
        ),
        (
            "nonfusing_e_split",
            "E(0.1) X0 Z1\n",
            [inst("E", [stim.target_x(0)], [0.1]), inst("E", [stim.target_z(1)], [0.1])],
        ),
        ("nonfusing_tick_too_few", "TICK\n", [inst("TICK", []), inst("TICK", [])]),
        ("nonfusing_det_too_few", "DETECTOR\n", [inst("DETECTOR", []), inst("DETECTOR", [])]),
        ("empty_flat_nonempty_ops", "", [inst("H", [])]),
    )
    for label, text, fake_unrolled in cases:
        bad = _bad_flat_circuit(fake_unrolled, stim.Circuit(text))
        ir = circuit_ir.build_ir(bad)
        assert [i.op for i in ir.instrs] == fake_unrolled, label
        with pytest.raises(ValueError):
            circuit_ir.unrolled_to_flat(ir)

    # Mocked unfused duplicate flat_ops ([H 0, H]) vs [H 0, H]: trailing check catches it.
    h0 = inst("H", [0])
    h_empty = inst("H", [])
    with pytest.raises(ValueError, match="more than the unrolled circuit maps to"):
        circuit_ir._align_flat_ops([h0, h_empty], (h0, h_empty))


def test_legacy_u2f_validate_true_kills_a4_witnesses() -> None:
    """Verify u2f validation on mismatched flattened() and zero-target fusion support."""

    class _BadFlatCircuit(stim.Circuit):
        def flattened(self) -> stim.Circuit:
            return stim.Circuit("X 0")

    bad_circ = _BadFlatCircuit("H 0")
    with pytest.raises(ValueError, match="does not match flattened op"):
        unrolled_to_flattened_indices(bad_circ, validate=True)
    assert unrolled_to_flattened_indices(bad_circ, validate=False).tolist() == [0]

    c_shift = stim.Circuit(
        "R 0\nX 0\nSHIFT_COORDS(1)\nX\nZ 0\nM 0\nDETECTOR rec[-1]\n"
    )
    w12 = stim.Circuit("H 0\nSHIFT_COORDS(1)\nH\nX 1\n")
    w16 = stim.Circuit(
        "R 0 1\n"
        "I_ERROR[LEAKAGE_TRANSITION_1: (0.1, U-->2)] 0\n"
        "H 0\nSHIFT_COORDS(1)\nH\nX 1\nM 0 1\nDETECTOR rec[-1]\n"
    )
    for circ in (c_shift, w12, w16):
        ir = _assert_valid_alignment("a4_witness", circ)
        got = unrolled_to_flattened_indices(circ, validate=True).tolist()
        assert got == circuit_ir.unrolled_to_flat(ir)

    assert Builder(w16, True).build(oracle=True) is not None
    assert MarginalLeakageDemGenerator(loss_oracle=True).base_dem(w16) is not None


def test_known_measurements_herald_inversion_kills_ks22() -> None:
    """Inverted herald targets invert deterministic measurement outcomes."""
    circ = stim.Circuit("R 0\nHERALDED_ERASE(0.1) !0\nM 0\n")
    assert compute_known_measurements(circ) == [True, False]


def test_lazy_alignment_snapshot_and_cross_circuit_equality() -> None:
    """Mutating source post-build and identity equality/hashing across circuits."""
    c_mut = stim.Circuit("H 0\nM 0\n")
    ir_mut = circuit_ir.build_ir(c_mut, uc=True)
    c_mut.append("X", [0])
    assert [str(op) for op in ir_mut.flat_ops] == ["H 0", "M 0"]
    assert circuit_ir.unrolled_to_flat(ir_mut) == [0, 1]

    c1 = stim.Circuit("H 0\nSHIFT_COORDS(1)\nH 1\n")
    c2 = stim.Circuit("X 0\nSHIFT_COORDS(1)\nH 1\n")
    ir1a = circuit_ir.build_ir(c1, uc=True)
    ir1b = circuit_ir.build_ir(c1, uc=True)
    ir2 = circuit_ir.build_ir(c2, uc=True)
    # CircuitIR and IRInstr use identity equality and hashing without forcing alignment.
    ir1a_ref = ir1a
    instr2_ref = ir1a.instrs[2]
    assert ir1a == ir1a_ref
    assert ir1a != ir1b
    assert ir1a != ir2
    assert ir1a.instrs[2] == instr2_ref
    assert ir1a.instrs[2] != ir1b.instrs[2]
    assert ir1a.instrs[2] != ir2.instrs[2]
    assert {ir1a: "a", ir1b: "b"}[ir1a] == "a"
    assert {ir1a.instrs[2]: "a", ir1b.instrs[2]: "b"}[ir1b.instrs[2]] == "b"
    assert ir1a.instrs[2].flat_index == 0 and ir1a.instrs[2].flat_offset == 1
    assert ir1b.instrs[2].flat_index == 0 and ir1b.instrs[2].flat_offset == 1
    assert ir2.instrs[2].flat_index == 1 and ir2.instrs[2].flat_offset == 0

    # Direct construction without _flat defaults to -1/0/() and remains frozen/hashable.
    direct = circuit_ir.IRInstr(
        0,
        c1[0],
        "",
        circuit_ir.Category.UNITARY,
        circuit_ir.UntaggedFilter.ONE_QUBIT,
        0,
        0,
    )
    assert direct.flat_index == -1
    assert direct.flat_offset == 0
    assert hash(direct) == hash(direct)


def test_lazy_alignment_defers_flattened_and_align(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Building IR and reading eager fields/hashes never triggers flat alignment."""
    circ = stim.Circuit("R 0\nREPEAT 3 {\n  H 0\n}\nM 0\n")

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("_align_flat_ops should not run eagerly")

    monkeypatch.setattr(circuit_ir, "_align_flat_ops", _boom)
    ir = circuit_ir.build_ir(circ, uc=True)
    assert len(ir.instrs) == 5
    assert [i.category for i in ir.instrs][1] == circuit_ir.Category.UNITARY
    assert [i.untagged_filter for i in ir.instrs][1] == circuit_ir.UntaggedFilter.ONE_QUBIT
    assert ir.instrs[1] != ir.instrs[2]
    assert {ir: 1, ir.instrs[1]: 2}[ir] == 1
    assert len(ir.flat_ops) == 3  # flat_ops computes flattened() without _align_flat_ops

    monkeypatch.undo()
    assert circuit_ir.unrolled_to_flat(ir) == [0, 1, 1, 1, 2]


def test_precondition_guards_and_negative_copy_len() -> None:
    """R8-C-REV-L3 / R8-C-C6-L3 / F15: input validation guards & negative copy_len."""
    c = stim.Circuit("R 0\nH 0\nM 0\n")
    ir = circuit_ir.build_ir(c)
    sm = circuit_ir.step_map(ir, frozenset())
    with pytest.raises(ValueError, match="shorter than step count"):
        circuit_ir.tableau_step_layout(ir, sm, [True, False])
    with pytest.raises(ValueError, match="shorter than step count"):
        circuit_ir.tableau_step_layout(ir, sm, [])

    rep_block = stim.CircuitRepeatBlock(2, stim.Circuit("H 0\n"))
    bad_seq_only: Any = [rep_block]
    bad_seq_mixed: Any = [stim.Circuit("H 0")[0], rep_block]
    with pytest.raises(TypeError, match="CircuitRepeatBlock"):
        circuit_ir.build_ir(bad_seq_only)
    with pytest.raises(TypeError, match="CircuitRepeatBlock"):
        circuit_ir.build_ir(bad_seq_mixed)

    # split_copies on non-empty from_op_list=True IR returns empty CopyMap.
    ir_list = circuit_ir.build_ir([stim.Circuit("H 0 1")[0]])
    assert circuit_ir.split_copies(ir_list, lambda _op: 1) == circuit_ir.CopyMap((), (), ())

    # Negative copy_len recorded verbatim (kills F15 clamping to 0) and leaves op unsplit.
    ir_h = circuit_ir.build_ir(stim.Circuit("H 0 1\n"))
    cm_neg = circuit_ir.split_copies(ir_h, lambda _op: -2)
    assert cm_neg.copy_len == (-2,)
    assert cm_neg.flat_to_op == (0,)
    assert cm_neg.ops == ir_h.flat_ops


def test_rf_oplist_uc_generator_and_aliasing() -> None:
    """R8-C-C6-TS2 (kills F01, F02, F04): op-list uc=True, generator, and mutation safety."""
    ops = [
        stim.Circuit("H 0")[0],
        stim.Circuit("CX 0 1")[0],
        stim.Circuit("E(0.1) X0")[0],
        stim.Circuit("SPP X0*Z1")[0],
        stim.Circuit("M 0")[0],
    ]
    ir_uc = circuit_ir.build_ir(ops, uc=True)
    assert ir_uc.uc is True
    assert ir_uc.from_op_list is True
    assert [i.untagged_filter for i in ir_uc.instrs] == [
        circuit_ir.UntaggedFilter.ONE_QUBIT,
        circuit_ir.UntaggedFilter.TWO_QUBIT,
        circuit_ir.UntaggedFilter.CORRELATED,
        circuit_ir.UntaggedFilter.PAULI_TERMS,
        circuit_ir.UntaggedFilter.NONE,
    ]
    # Single-pass generator input must produce structurally identical IR fields.
    gen_src: Any = (op for op in ops)
    ir_gen = circuit_ir.build_ir(gen_src, uc=True)
    assert (
        ir_gen.num_qubits,
        ir_gen.num_measurements,
        ir_gen.has_repeat,
        ir_gen.uc,
        ir_gen.from_op_list,
        ir_gen.flat_ops,
    ) == (
        ir_uc.num_qubits,
        ir_uc.num_measurements,
        ir_uc.has_repeat,
        ir_uc.uc,
        ir_uc.from_op_list,
        ir_uc.flat_ops,
    )
    assert len(ir_gen.instrs) == len(ir_uc.instrs)
    for g_ins, u_ins in zip(ir_gen.instrs, ir_uc.instrs, strict=True):
        assert (
            g_ins.unrolled_index,
            g_ins.op,
            g_ins.tag,
            g_ins.category,
            g_ins.untagged_filter,
            g_ins.num_measurements,
            g_ins.first_measurement,
            g_ins.basis,
            g_ins.is_product_measurement,
            g_ins.arity,
            g_ins.has_classical_control,
            g_ins.flat_index,
            g_ins.flat_offset,
        ) == (
            u_ins.unrolled_index,
            u_ins.op,
            u_ins.tag,
            u_ins.category,
            u_ins.untagged_filter,
            u_ins.num_measurements,
            u_ins.first_measurement,
            u_ins.basis,
            u_ins.is_product_measurement,
            u_ins.arity,
            u_ins.has_classical_control,
            u_ins.flat_index,
            u_ins.flat_offset,
        )
    # Mutating caller's list after build_ir must not alter the built IR.
    ops.clear()
    assert len(ir_uc.instrs) == 5


def test_rf_first_measurement_after_mry_and_heralded() -> None:
    """R8-C-C6-TS2 (kills F11, F12): prefix sum after MRY, MRX, and HERALDED_PAULI_CHANNEL_1."""
    circ = stim.Circuit(
        "R 0 1\n"
        "MRY 0\n"
        "M 1\n"
        "MRX 0\n"
        "M 0\n"
        "HERALDED_PAULI_CHANNEL_1(0.05, 0.05, 0.05, 0.05) 1\n"
        "M 1\n"
    )
    ir = circuit_ir.build_ir(circ, uc=True)
    got = [(i.op.name, i.first_measurement, i.num_measurements) for i in ir.instrs]
    assert got == [
        ("R", 0, 0),
        ("MRY", 0, 1),
        ("M", 1, 1),
        ("MRX", 2, 1),
        ("M", 3, 1),
        ("HERALDED_PAULI_CHANNEL_1", 4, 1),
        ("M", 5, 1),
    ]
    assert ir.num_measurements == 6


def test_rf_tableau_layout_numpy_bool_and_int() -> None:
    """R8-C-C6-TS2 (kills F14): np.bool_ and int truthiness in tableau_step_layout."""
    circ = stim.Circuit("R 0\nTICK\nH 0\nM 0\n")
    ir = circuit_ir.build_ir(circ)
    sm = circuit_ir.step_map(ir, _SKIP_TAB)
    want = circuit_ir.tableau_step_layout(ir, sm, [True, False, True])
    np_mask: Any = np.array([True, False, True], dtype=np.bool_)
    int_mask: Any = [1, 0, 1]
    assert circuit_ir.tableau_step_layout(ir, sm, np_mask) == want
    assert circuit_ir.tableau_step_layout(ir, sm, int_mask) == want


def test_rf_tag_memo_falsy_values_and_value_key_targets() -> None:
    """R8-C-C6-TS2 (kills F16, F18): TagMemo falsy caching & VALUE_KEY on X targets."""
    op_x0 = stim.Circuit("X 0")[0]
    op_x1 = stim.Circuit("X 1")[0]
    op_h0 = stim.Circuit("H 0")[0]
    op_h1 = stim.Circuit("H 1")[0]

    falsy_values: tuple[Any, ...] = (None, False, 0, "", (), [], {})
    for fv in falsy_values:
        calls = 0

        def _parse(_op: stim.CircuitInstruction, val: Any = fv) -> Any:
            nonlocal calls
            calls += 1
            return val

        memo = ir_tags.TagMemo(_parse, ir_tags.BUILDER_KEY)
        assert memo.params(op_x0) == fv
        assert memo.params(op_x1) == fv
        assert calls == 1, f"falsy value {fv!r} was not cached (calls={calls})"

    assert ir_tags.VALUE_KEY(op_x0) != ir_tags.VALUE_KEY(op_x1)
    assert ir_tags.VALUE_KEY(op_h0) != ir_tags.VALUE_KEY(op_h1)
    vk_calls: list[str] = []

    def _rec(op: stim.CircuitInstruction) -> str:
        s = str(op)
        vk_calls.append(s)
        return s

    memo_v = ir_tags.TagMemo(_rec, ir_tags.VALUE_KEY)
    assert memo_v.params(op_x0) == "X 0"
    assert memo_v.params(op_x1) == "X 1"
    assert memo_v.params(op_x0) == "X 0"
    assert vk_calls == ["X 0", "X 1"]


def test_shared_fields_all_93_stim_spellings() -> None:
    """Verify basis, is_product_measurement, arity, and classical control on all 93 spellings."""
    expected_x = frozenset({"MX", "MRX", "RX", "MXX"})
    expected_y = frozenset({"MY", "MRY", "RY", "MYY"})
    expected_z = frozenset({"M", "MZ", "MR", "MRZ", "R", "RZ", "MZZ"})
    expected_pm = frozenset({"MPP", "MXX", "MYY", "MZZ"})
    expected_cc_gates = frozenset({"CX", "CNOT", "ZCX", "CY", "ZCY", "CZ", "ZCZ", "XCZ", "YCZ"})

    all_spellings: list[tuple[str, Any]] = []
    for _canon, gd in sorted(stim.gate_data().items()):
        for alias in sorted(gd.aliases):
            all_spellings.append((alias, gd))
    assert len(all_spellings) == 93

    basis_counts: dict[str | None, int] = {"X": 0, "Y": 0, "Z": 0, None: 0}
    pm_true = 0
    arity_counts: dict[int | None, int] = {1: 0, 2: 0, None: 0}
    cc_gates_seen: set[str] = set()

    for spelling, gd in all_spellings:
        exp_basis = (
            "X"
            if spelling in expected_x
            else "Y"
            if spelling in expected_y
            else "Z"
            if spelling in expected_z
            else None
        )
        exp_pm = spelling in expected_pm
        exp_arity = (
            1
            if (gd.is_unitary and gd.is_single_qubit_gate)
            else 2
            if (gd.is_unitary and gd.is_two_qubit_gate)
            else None
        )
        for variant in (spelling, spelling.lower(), spelling.title()):
            assert circuit_ir.classify_basis(variant) == exp_basis, variant
            assert circuit_ir.classify_is_product_measurement(variant) is exp_pm, variant
            assert circuit_ir.classify_arity(variant) == exp_arity, variant

        basis_counts[exp_basis] += 1
        if exp_pm:
            pm_true += 1
        arity_counts[exp_arity] += 1

        if gd.is_two_qubit_gate and gd.takes_measurement_record_targets:
            cc_gates_seen.add(spelling)
            assert not circuit_ir.op_has_classical_control(
                stim.CircuitInstruction(spelling, [0, 1])
            )
            assert not circuit_ir.op_has_classical_control(
                stim.CircuitInstruction(spelling, [0, 1], tag="rec[0]")
            )
            if spelling in ("XCZ", "YCZ"):
                rec_op = stim.CircuitInstruction(spelling, [0, stim.target_rec(-1)])
                swp_op = stim.CircuitInstruction(spelling, [0, stim.target_sweep_bit(0)])
            else:
                rec_op = stim.CircuitInstruction(spelling, [stim.target_rec(-1), 1])
                swp_op = stim.CircuitInstruction(spelling, [stim.target_sweep_bit(0), 1])
            assert circuit_ir.op_has_classical_control(rec_op), spelling
            assert circuit_ir.op_has_classical_control(swp_op), spelling

    assert basis_counts == {"X": 4, "Y": 4, "Z": 7, None: 78}
    assert pm_true == 4
    assert arity_counts == {1: 27, 2: 27, None: 39}
    assert cc_gates_seen == expected_cc_gates
    assert circuit_ir.classify_arity("SPP") is None
    assert circuit_ir.classify_arity("SPP_DAG") is None

    for fn in (
        circuit_ir.classify_basis,
        circuit_ir.classify_is_product_measurement,
        circuit_ir.classify_arity,
    ):
        with pytest.raises(IndexError):
            fn("NOT_A_REAL_STIM_GATE")


def test_shared_fields_build_ir_circuit_repeat_and_oplist_parity() -> None:
    """Verify all 4 shared fields match across circuit, REPEAT, and op-list IR builds."""
    circ = stim.Circuit(
        "R 0 1\nRX 0\nRY 1\nH 0\nCX 0 1\nCX[rec[0\\C] 0 1\nSPP X0*Z1\n"
        "DEPOLARIZE2(0.01) 0 1\nSWAP 0 1\nM 0\nMX 0\nMY 1\nMR 0\nMRX 0\nMRY 1\n"
        "MXX 0 1\nMYY 0 1\nMZZ 0 1\nMPP X0*Z1\nMPAD 0\nHERALDED_ERASE(0.01) 0\n"
        "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
        "REPEAT 2 {\n"
        "  CX rec[-1] 1\n"
        "  CY sweep[0] 1\n"
        "  XCZ 0 rec[-1]\n"
        "  YCZ 0 sweep[0]\n"
        "  CZ rec[-1] 1\n"
        "  CX[sweep[0\\C] 0 1\n"
        "}\n"
    )
    ir_c = circuit_ir.build_ir(circ, uc=True)
    ir_l = circuit_ir.build_ir([i.op for i in ir_c.instrs], uc=True)
    assert len(ir_c.instrs) == len(ir_l.instrs)
    for ic, il in zip(ir_c.instrs, ir_l.instrs, strict=True):
        name = ic.op.name
        assert ic.basis == il.basis == circuit_ir.classify_basis(name), name
        assert (
            ic.is_product_measurement
            == il.is_product_measurement
            == circuit_ir.classify_is_product_measurement(name)
        ), name
        assert ic.arity == il.arity == circuit_ir.classify_arity(name), name
        assert (
            ic.has_classical_control
            == il.has_classical_control
            == circuit_ir.op_has_classical_control(ic.op)
        ), str(ic.op)

    tagged_cx = [i for i in ir_c.instrs if i.op.name == "CX" and i.tag in ("rec[0]", "sweep[0]")]
    assert len(tagged_cx) == 3
    assert all(not i.has_classical_control for i in tagged_cx)
    for ins in ir_c.instrs:
        if ins.op.name in ("DETECTOR", "OBSERVABLE_INCLUDE", "SWAP", "DEPOLARIZE2"):
            assert ins.has_classical_control is False, ins.op.name


def test_shared_fields_immutability_replace_and_deferred_alignment() -> None:
    """Verify AttributeError on fields, _replace, and lazy _FlatAlignment."""
    circ = stim.Circuit("R 0\nREPEAT 2 {\n  H 0\n}\nM 0\n")
    ir_c = circuit_ir.build_ir(circ, uc=True)
    # Alignment helper must not be instantiated eagerly during build_ir.
    assert ir_c._flat is not None and ir_c._flat[1] is None

    sample = ir_c.instrs[0]
    for attr_name, val in (
        ("basis", "X"),
        ("is_product_measurement", True),
        ("arity", 1),
        ("has_classical_control", True),
    ):
        with pytest.raises(AttributeError):
            setattr(sample, attr_name, val)
    replaced = sample._replace(category=circuit_ir.Category.OTHER, basis="X")
    assert replaced.category is circuit_ir.Category.OTHER and replaced.basis == "X"

    # First flat_ops access creates _FlatAlignment; alignment releases circuit snapshot.
    assert len(ir_c.flat_ops) > 0
    assert ir_c._flat[1] is not None and ir_c._flat[0] is None
    assert len(circuit_ir.unrolled_to_flat(ir_c)) == len(ir_c.instrs)
    assert ir_c._flat[1]._circuit is None


def test_u2f_zero_target_fusion_all_conditions() -> None:
    """Compare unrolled_to_flattened_indices and IR unrolled_to_flat on zero-target fusion."""
    witnesses: tuple[tuple[str, str, list[int]], ...] = (
        ("same_gate_fuses", "H 0\nSHIFT_COORDS(1)\nH\n", [0, -1, 0]),
        ("diff_tag_no_fuse", "H[a] 0\nSHIFT_COORDS(1)\nH[b]\n", [0, -1, 1]),
        (
            "diff_args_no_fuse",
            "X_ERROR(0.1) 0\nSHIFT_COORDS(1)\nX_ERROR(0.2)\n",
            [0, -1, 1],
        ),
        ("diff_name_no_fuse", "H 0\nSHIFT_COORDS(1)\nX\n", [0, -1, 1]),
        ("nonfusing_tick", "TICK\nSHIFT_COORDS(1)\nTICK\n", [0, -1, 1]),
        ("nonfusing_detector", "DETECTOR\nSHIFT_COORDS(1)\nDETECTOR\n", [0, -1, 1]),
        (
            "nonfusing_obs",
            "OBSERVABLE_INCLUDE(0)\nSHIFT_COORDS(1)\nOBSERVABLE_INCLUDE(0)\n",
            [0, -1, 1],
        ),
        (
            "nonfusing_qcoords",
            "QUBIT_COORDS(0)\nSHIFT_COORDS(1)\nQUBIT_COORDS(0)\n",
            [0, -1, 1],
        ),
        ("nonfusing_e", "E(0.1)\nSHIFT_COORDS(1)\nE(0.1)\n", [0, -1, 1]),
        (
            "nonfusing_else_e",
            (
                "E(0.1) X0\nELSE_CORRELATED_ERROR(0.1)\nSHIFT_COORDS(1)\n"
                "ELSE_CORRELATED_ERROR(0.1)\n"
            ),
            [0, 1, -1, 2],
        ),
    )
    for label, text, expected in witnesses:
        circ = stim.Circuit(text)
        ir_u2f = circuit_ir.unrolled_to_flat(circuit_ir.build_ir(circ))
        legacy_val = unrolled_to_flattened_indices(circ, validate=True).tolist()
        legacy_noval = unrolled_to_flattened_indices(circ, validate=False).tolist()
        assert ir_u2f == expected, label
        assert legacy_val == expected, label
        assert legacy_noval == expected, label


def test_ir_instr_copy_deepcopy_equality_ordering_and_repr() -> None:
    """Pin copy, deepcopy with Stim memo, simulator copy, tuple eq, ordering, and repr tag."""
    from stimside.op_handlers.abstract_op_handler import CompiledOpHandler
    from stimside.simulator_magic import MagicsideSimulator, _stim_deepcopy_memo

    circ = stim.Circuit("H[my_tag] 0\nCX rec[-1] 1\nM 0 1\n")
    ir1 = circuit_ir.build_ir(circ, uc=True)
    ir2 = circuit_ir.build_ir(circ, uc=True)
    ins1, ins2 = ir1.instrs[0], ir2.instrs[0]

    assert copy.copy(ins1) is ins1
    ins_dc = copy.deepcopy(ins1, _stim_deepcopy_memo(ins1))
    assert isinstance(ins_dc, circuit_ir.IRInstr)
    assert ins_dc.op == ins1.op and ins_dc.tag == ins1.tag and ins_dc.arity == ins1.arity

    # Deepcopy CircuitIR both before and after lazy alignment; _flat stays shared inside copy.
    for src_ir in (ir1, ir2):
        if src_ir is ir2:
            assert len(src_ir.flat_ops) > 0 and src_ir.instrs[0].flat_index == 0
        ir_dc = copy.deepcopy(src_ir, _stim_deepcopy_memo(src_ir))
        assert isinstance(ir_dc, circuit_ir.CircuitIR) and ir_dc is not src_ir
        assert ir_dc._flat is not None and ir_dc._flat is not src_ir._flat
        assert ir_dc._flat is ir_dc.instrs[0]._flat
        assert ir_dc.flat_ops == src_ir.flat_ops
        assert circuit_ir.unrolled_to_flat(ir_dc) == circuit_ir.unrolled_to_flat(src_ir)

    class _IRHoldingHandler(CompiledOpHandler[Any]):
        def __init__(self, held_ir: circuit_ir.CircuitIR) -> None:
            self.ir = held_ir

        def handle_op(self, op: stim.CircuitInstruction, sss: Any) -> None:
            sss._do_bare_instruction(op)

        def clear(self) -> None:
            return

    simple_circ = stim.Circuit("R 0\nH 0\nM 0\n")
    sim = MagicsideSimulator(
        circuit=simple_circ,
        compiled_op_handler=_IRHoldingHandler(circuit_ir.build_ir(simple_circ, uc=True)),
    )
    sim_copy = copy.copy(sim)
    copied_ir = sim_copy.compiled_op_handler.ir
    assert isinstance(copied_ir, circuit_ir.CircuitIR)
    assert copied_ir is not sim.compiled_op_handler.ir
    assert copied_ir._flat is copied_ir.instrs[0]._flat

    # Identity equality without tuple-reflection leaks.
    raw = tuple(ins1)
    same = ins1
    assert (ins1 == same) is True and (ins1 != same) is False
    assert (ins1 == ins2) is False and (ins1 != ins2) is True
    assert (ins1 == raw) is False and (raw == ins1) is False
    assert (ins1 != raw) is True and (raw != ins1) is True

    # Rich ordering is blocked both between IRInstrs and against plain tuples.
    for op_fn in (operator.lt, operator.le, operator.gt, operator.ge):
        for lhs, rhs in ((ins1, ins2), (ins1, raw), (raw, ins1)):
            with pytest.raises(TypeError):
                op_fn(lhs, rhs)

    custom = ins1._replace(tag="DIFFERENT_TAG")
    assert "tag='DIFFERENT_TAG'" in repr(custom)


def _make_sized_flat_circuit(target_len: int) -> stim.Circuit:
    """Build a flat stim.Circuit with exact top-level length ``target_len``."""
    head = [
        stim.CircuitInstruction("R", [0, 1, 2]),
        stim.CircuitInstruction("H", [0], [], tag="my_tag"),
        stim.CircuitInstruction("CX", [0, 1]),
        stim.CircuitInstruction("M", [0, 1]),
        stim.CircuitInstruction("MXX", [0, 1]),
        stim.CircuitInstruction(
            "MPP", [stim.target_x(0), stim.target_combiner(), stim.target_z(1)]
        ),
        stim.CircuitInstruction("CX", [stim.target_rec(-1), 2]),
        stim.CircuitInstruction("MRY", [1, 2]),
        stim.CircuitInstruction("CZ", [2, stim.target_sweep_bit(0)]),
        stim.CircuitInstruction("MPAD", [0, 1]),
    ]
    c = stim.Circuit()
    for i in range(min(target_len, len(head))):
        c.append(head[i])
    alt = (
        stim.CircuitInstruction("H", [0]),
        stim.CircuitInstruction("X", [1]),
        stim.CircuitInstruction("TICK", []),
    )
    for i in range(len(head), target_len):
        c.append(alt[i % 3])
    assert len(c) == target_len
    return c


def _assert_ir_fields_parity(circ: stim.Circuit, uc: bool) -> circuit_ir.CircuitIR:
    """Verify full field, measurement, and lazy-alignment parity for ``build_ir``."""
    ir = circuit_ir.build_ir(circ, uc=uc)
    unrolled = unroll_circuit(circ)
    ir_list = circuit_ir.build_ir(unrolled, uc=uc)
    has_rep = any(isinstance(item, stim.CircuitRepeatBlock) for item in circ)
    assert ir.num_qubits == circ.num_qubits
    assert ir.num_measurements == circ.num_measurements == ir_list.num_measurements
    assert ir.has_repeat is has_rep
    assert ir.uc is uc
    assert ir.from_op_list is False
    assert len(ir.instrs) == len(unrolled) == len(ir_list.instrs)

    running_m = 0
    for idx, (ins, ref) in enumerate(zip(ir.instrs, ir_list.instrs, strict=True)):
        assert ins.unrolled_index == idx == ref.unrolled_index
        assert ins.op == unrolled[idx] == ref.op
        assert ins.tag == ref.tag
        assert ins.category is ref.category
        assert ins.untagged_filter is ref.untagged_filter
        assert ins.num_measurements == ref.num_measurements
        assert ins.first_measurement == running_m == ref.first_measurement
        assert ins.basis == ref.basis
        assert ins.is_product_measurement is ref.is_product_measurement
        assert ins.arity == ref.arity
        assert ins.has_classical_control is ref.has_classical_control
        running_m += ins.num_measurements
    assert running_m == circ.num_measurements

    want_flat = tuple(circ.flattened())
    assert ir.flat_ops == want_flat
    u2f = circuit_ir.unrolled_to_flat(ir)
    assert u2f == unrolled_to_flattened_indices(circ, validate=True).tolist()
    for ins, fi in zip(ir.instrs, u2f, strict=True):
        assert ins.flat_index == fi
        if fi == -1:
            assert ins.flat_offset == 0
        else:
            assert 0 <= ins.flat_offset <= len(want_flat[fi].targets_copy())
    return ir


def test_build_ir_small_large_threshold_parity() -> None:
    """Verify exact IR parity across the 1000-op threshold and small REPEATs."""
    small_prefix_repeat = stim.Circuit(
        "R 0 1\nH[t1] 0\nM 0\n"
        "REPEAT 3 {\n  CX rec[-1] 1\n  MXX 0 1\n  SHIFT_COORDS(1)\n  H 0\n}\n"
        "H 0\nM 1\n"
    )
    small_nested_repeat = stim.Circuit(
        "R 0 1 2\n"
        "REPEAT 2 {\n"
        "  H 0\n"
        "  REPEAT 3 {\n    CX 0 1\n    MPP X0*Z1\n  }\n"
        "  REPEAT 2 {\n  }\n"
        "  M 2\n"
        "}\n"
        "CZ sweep[0] 2\n"
    )
    large_with_repeat = _make_sized_flat_circuit(1000)
    large_with_repeat.append(stim.CircuitRepeatBlock(2, stim.Circuit("H 0\nM 0\n")))

    cases = (
        ("empty", stim.Circuit()),
        ("flat_1", _make_sized_flat_circuit(1)),
        ("flat_150", _make_sized_flat_circuit(150)),
        ("flat_999", _make_sized_flat_circuit(999)),
        ("flat_1000", _make_sized_flat_circuit(1000)),
        ("flat_1001", _make_sized_flat_circuit(1001)),
        ("small_prefix_repeat", small_prefix_repeat),
        ("small_nested_repeat", small_nested_repeat),
        ("large_with_repeat", large_with_repeat),
    )
    for uc in (False, True):
        for label, circ in cases:
            ir = _assert_ir_fields_parity(circ, uc=uc)
            if label in ("flat_999", "flat_1000", "flat_1001"):
                # Tagged H at index 1 stays ONE_QUBIT when uc=True; MXX/MPP count 1 meas.
                want_uf = (
                    circuit_ir.UntaggedFilter.ONE_QUBIT
                    if uc
                    else circuit_ir.UntaggedFilter.NONE
                )
                assert ir.instrs[1].untagged_filter is want_uf, label
                assert ir.instrs[4].num_measurements == 1, label
                assert ir.instrs[5].num_measurements == 1, label
                assert ir.instrs[7].first_measurement == 4, label


def test_build_ir_preserves_gc_state_across_threshold() -> None:
    """Verify build_ir leaves gc.isenabled() unchanged for both < 1000 and >= 1000 ops."""
    import gc

    circs = (
        stim.Circuit(),
        _make_sized_flat_circuit(200),
        _make_sized_flat_circuit(1000),
        stim.Circuit("R 0\nREPEAT 3 {\n  H 0\n  M 0\n}\n"),
    )
    orig = gc.isenabled()
    try:
        for desired in (True, False):
            if desired:
                gc.enable()
            else:
                gc.disable()
            for circ in circs:
                circuit_ir.build_ir(circ, uc=True)
                assert gc.isenabled() is desired
    finally:
        if orig:
            gc.enable()
        else:
            gc.disable()


def _count_c_calls(name: str, fn: Callable[[], Any]) -> int:
    """Count the calls of the C-implemented function ``name`` made while ``fn`` runs."""
    calls = 0

    def _profile(_frame: types.FrameType, event: str, arg: Any) -> None:
        nonlocal calls
        if event == "c_call" and getattr(arg, "__name__", "") == name:
            calls += 1

    previous = sys.getprofile()
    sys.setprofile(_profile)
    try:
        fn()
    finally:
        sys.setprofile(previous)
    return calls


def test_build_ir_builds_a_large_circuit_with_repeat_in_one_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1000 or more top-level items plus a REPEAT block: every IRInstr is created once."""
    circ = _make_sized_flat_circuit(1200)
    circ.append(stim.CircuitRepeatBlock(3, stim.Circuit("H 0\nM 0\n")))
    created = 0
    real_instr = circuit_ir.IRInstr

    def _counting_instr(*args: Any, **kwargs: Any) -> circuit_ir.IRInstr:
        nonlocal created
        created += 1
        return real_instr(*args, **kwargs)

    monkeypatch.setattr(circuit_ir, "IRInstr", _counting_instr)
    ir = circuit_ir.build_ir(circ, uc=True)
    assert ir.has_repeat
    assert created == len(ir.instrs) == len(unroll_circuit(circ))


def test_build_ir_checks_classical_control_without_copying_targets() -> None:
    """No rec[...] or sweep[...] in an instruction's text: its target list is not copied."""
    plain = stim.Circuit()
    for i in range(200):
        plain.append("CX", [i % 7, 7 + i % 5])
        plain.append("CZ", [i % 5, 5 + i % 7])
        plain.append("H", [i % 7])
    assert (
        _count_c_calls(
            "targets_copy",
            lambda: [ins.has_classical_control for ins in circuit_ir.build_ir(plain).instrs],
        )
        == 0
    )
    assert not any(ins.has_classical_control for ins in circuit_ir.build_ir(plain).instrs)

    # A tag that contains "rec[" only sends the instruction to the exact check.
    feedback = stim.Circuit(
        "M 0 1\nCX rec[-1] 1\nCX[rec[0] 0 1\nCZ 0 sweep[2]\nCX 0 1\nCY rec[-2] 1\n"
        "CZ sweep[10] 0\nREPEAT 2 {\n  CY rec[-1] 1\n  H 0\n}\n"
    )
    flags = [ins.has_classical_control for ins in circuit_ir.build_ir(feedback).instrs]
    assert flags == [False, True, False, True, False, True, True, True, False, True, False]


def test_build_ir_never_toggles_gc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building IR never calls gc.disable or gc.enable for any circuit size."""
    import gc

    def _forbid() -> None:
        raise AssertionError("build_ir must not toggle the garbage collector")

    monkeypatch.setattr(gc, "disable", _forbid)
    monkeypatch.setattr(gc, "enable", _forbid)
    for circ in (
        stim.Circuit(),
        _make_sized_flat_circuit(200),
        _make_sized_flat_circuit(1200),
        stim.Circuit("R 0\nREPEAT 5 {\n  H 0\n  CX 0 1\n  M 0\n}\n"),
    ):
        ir = circuit_ir.build_ir(circ, uc=True)
        assert len(ir.instrs) == len(unroll_circuit(circ))


def test_build_ir_defers_classical_control_until_read() -> None:
    """Consumers that do not read has_classical_control pay no str/cache work during build_ir."""
    circ = stim.Circuit("R 0 1\nREPEAT 3 {\n  CX 0 1\n}\nCX rec[-1] 1\nM 0 1\n")
    ir = circuit_ir.build_ir(circ, uc=True)
    assert ir._flat is not None and ir._flat[2] is None
    assert ir.instrs[1].has_classical_control is False
    assert ir._flat[2] is not None
    assert ir.instrs[2].has_classical_control is False
    assert ir.instrs[4].has_classical_control is True


def test_ir_instr_optional_field_defaults() -> None:
    """Direct IRInstr construction defaults optional fields to None/False."""
    op = stim.Circuit("H 0")[0]
    direct = circuit_ir.IRInstr(
        0, op, "", circuit_ir.Category.UNITARY, circuit_ir.UntaggedFilter.ONE_QUBIT, 0, 0
    )
    assert direct.basis is None
    assert direct.is_product_measurement is False
    assert direct.arity is None
    assert direct.flat_box is None
    assert direct.has_classical_control is False
