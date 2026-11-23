"""OOM fault-injection tests for all 11 allocating production `engine_*` calls (13 OOM cases),
zero-allocation `engine_set_rng_states` (12 functions, 14 cases total), and sticky guard."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from stimside._native import coset_kernels as ck

_SRC = Path(ck.__file__).resolve().parents[2]

_FI_CPP = r"""
#include <cstdlib>
#include <new>
static thread_local long g_countdown = -1;
static thread_local long g_count = 0;
extern "C" void fi_arm(long n) { g_countdown = n; g_count = 0; }
extern "C" long fi_count() { return g_count; }
extern "C" void fi_disarm() { g_countdown = -1; }
static void* alloc(size_t sz) {
    g_count++;
    if (g_countdown == 0) { g_countdown = -1; throw std::bad_alloc(); }
    if (g_countdown > 0) --g_countdown;
    void* p = std::malloc(sz ? sz : 1);
    if (!p) throw std::bad_alloc();
    return p;
}
void* operator new(size_t sz) { return alloc(sz); }
void* operator new[](size_t sz) { return alloc(sz); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void operator delete[](void* p, size_t) noexcept { std::free(p); }
"""

_CHILD = r"""
import ctypes, json, sys
import numpy as np
import stim

sys.path.insert(0, sys.argv[1])
fi = ctypes.CDLL(sys.argv[2])
fi.fi_arm.argtypes = [ctypes.c_long]
fi.fi_count.restype = ctypes.c_long
case = sys.argv[3]

from stimside._native import coset_kernels as ck
from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
from stimside.simulator_coset import CosetsideSimulator


def _make_sim(c, batch_size=1, seed=0):
    h = LeakageUint8Coset().compile_op_handler(circuit=c, batch_size=batch_size)
    return CosetsideSimulator(c, compiled_op_handler=h, batch_size=batch_size, seed=seed)


def _make_spy(inner, fn_name, state):
    class Spy:
        def __init__(self, obj):
            self._inner = obj

        def __getattr__(self, attr):
            fn = getattr(self._inner, attr)
            if attr != fn_name or state["k"] is None:
                return fn

            def wrapped(*args):
                state["calls"] += 1
                fi.fi_arm(state["k"])
                try:
                    return fn(*args)
                finally:
                    state["allocs"] += fi.fi_count()
                    fi.fi_disarm()

            return wrapped

    return Spy(inner)


def _sim_with_d(nq, d):
    c = stim.Circuit()
    c.append("H", list(range(nq)))
    sim = _make_sim(c, batch_size=1, seed=7)
    sim.interactive_do(c)
    op = stim.CircuitInstruction("S", list(range(d)))
    sim.do_conditional_clifford(op, np.zeros((d, 1), dtype=bool))
    assert sim._cpp_lib.engine_get_active_shot_indices(
        sim._cpp_eng,
        np.empty(1, np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        np.empty(1, np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    ) == 1
    return sim


def _make_case(name):
    if name == "pub_engine_create":
        c = stim.Circuit("H 0\nCX 0 1")
        return None, "engine_create", lambda: _make_sim(c, batch_size=2, seed=0)
    if name == "pub_set_rng_states":
        sim = _make_sim(stim.Circuit("H 0\nM 0"), batch_size=4, seed=5)
        return sim, "engine_set_rng_states", sim.clear
    if name == "pub_register_unitary_step":
        sim = _make_sim(stim.Circuit("H 0"), batch_size=1, seed=0)
        op = stim.CircuitInstruction("H", [0])
        return sim, "engine_register_unitary_step", lambda: sim.do_conditional_clifford(
            op, np.zeros((1, 1), dtype=bool)
        )
    if name == "pub_register_unitary_group_evict":
        sim = _make_sim(stim.Circuit("H 0 1"), batch_size=1, seed=0)
        op = stim.CircuitInstruction("S", [0, 1])
        gfn = lambda g: np.zeros(1, dtype=bool) if g == 0 else np.ones(1, dtype=bool)
        return sim, "engine_register_unitary_group_evict", lambda: sim.do_conditional_clifford(
            op, condition_group_fn=gfn
        )
    if name == "pub_disjoint_clifford_v3":
        sim = _sim_with_d(66, 64)
        op = stim.CircuitInstruction("S", [64])
        return sim, "engine_batch_inject_unitary_v4", lambda: sim.do_conditional_clifford(
            op, np.zeros((1, 1), dtype=bool)
        )
    if name == "pub_inject_group_v3":
        sim = _sim_with_d(66, 64)
        op = stim.CircuitInstruction("S", [64, 65])
        gfn = lambda g: np.zeros(1, dtype=bool) if g == 0 else np.ones(1, dtype=bool)
        return sim, "engine_inject_group_v3", lambda: sim.do_conditional_clifford(
            op, condition_group_fn=gfn
        )
    if name == "pub_det_meas_step":
        sim = _sim_with_d(262, 260)
        sim.interactive_do(stim.Circuit("S_DAG 0 1\nH 0 1"))
        op = stim.Circuit("M 0 1")
        return sim, "engine_batch_meas_step_v4", lambda: sim.interactive_do(op)
    if name == "pub_meas_step_v4_rand":
        sim = _sim_with_d(262, 260)
        op = stim.Circuit("M 0")
        return sim, "engine_batch_meas_step_v4", lambda: sim.interactive_do(op)
    if name == "pub_mixed_substep_det":
        sim = _sim_with_d(262, 260)
        sim.interactive_do(stim.Circuit("S_DAG 0"))
        op = stim.Circuit("MX 0")
        return sim, "engine_batch_mixed_substep", lambda: sim.interactive_do(op)
    if name == "pub_mixed_substep_rand":
        sim = _sim_with_d(262, 260)
        sim.interactive_do(stim.Circuit("H 0"))
        op = stim.Circuit("MX 0")
        return sim, "engine_batch_mixed_substep", lambda: sim.interactive_do(op)
    if name == "pub_unmatched_reset_z":
        sim = _sim_with_d(262, 260)
        r0 = np.zeros((262, 1), dtype=bool)
        r0[0, 0] = True
        r1 = np.zeros((262, 1), dtype=bool)
        return sim, "engine_batch_unmatched_noisy_reset_z", (
            lambda: sim.do_unmatched_noisy_reset_z(r0, r1, "Z")
        )
    if name == "pub_project_pauli_xy":
        sim = _sim_with_d(262, 260)
        r0 = np.zeros((262, 1), dtype=bool)
        r0[0, 0] = True
        r1 = np.zeros((262, 1), dtype=bool)
        return sim, "engine_project_pauli", lambda: sim.do_unmatched_noisy_reset_z(r0, r1, "X")
    if name == "pub_postselect_batch":
        sim = _sim_with_d(262, 260)
        fm = np.ones(1, dtype=bool)
        db = np.zeros(1, dtype=np.uint8)
        return sim, "engine_postselect_batch", lambda: sim.postselect_z_batch(0, fm, db, "Z")
    if name == "pub_peek_pauli_batch":
        sim = _sim_with_d(262, 260)
        return sim, "engine_peek_pauli_batch", lambda: sim.peek_pauli_batch([0, 1], "Z")
    if name == "sticky_s5_eliminate_zero":
        sim = _sim_with_d(262, 260)
        op = stim.CircuitInstruction("S", [0])
        return sim, "engine_batch_inject_unitary_v4", lambda: sim.do_conditional_clifford(
            op, np.zeros((1, 1), dtype=bool)
        )
    raise SystemExit("unknown case " + name)


def _run_one(name, k):
    state = {"k": k, "calls": 0, "allocs": 0}
    if name == "pub_engine_create":
        lib = ck.get_coset_kernels_lib()
        orig_create = lib.engine_create

        def spy_create(*args):
            state["calls"] += 1
            fi.fi_arm(state["k"])
            try:
                return orig_create(*args)
            finally:
                state["allocs"] += fi.fi_count()
                fi.fi_disarm()

        lib.engine_create = spy_create
        try:
            _, hooked_fn, thunk = _make_case(name)
            thunk()
            return "ok", hooked_fn, state["calls"], state["allocs"]
        except MemoryError:
            return "MemoryError", "engine_create", state["calls"], state["allocs"]
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}", "engine_create", state["calls"], state["allocs"]
        finally:
            lib.engine_create = orig_create

    sim, hooked_fn, thunk = _make_case(name)
    orig_lib = sim._cpp_lib
    sim._cpp_lib = _make_spy(orig_lib, hooked_fn, state)
    try:
        thunk()
        return "ok", hooked_fn, state["calls"], state["allocs"]
    except MemoryError:
        sim._cpp_lib = orig_lib
        if orig_lib.engine_failed(sim._cpp_eng) != 1:
            return "engine_failed!=1", hooked_fn, state["calls"], state["allocs"]
        if name == "sticky_s5_eliminate_zero":
            n_q, n_b = sim._engine_dims()
            pw = sim._current_packed_words()
            xs_b, zs_b, _, _, _ = sim._flip_simulator.to_numpy(output_xs=True, output_zs=True)
            xs_c = np.ascontiguousarray(xs_b, dtype=np.uint8)
            zs_c = np.ascontiguousarray(zs_b, dtype=np.uint8)
            tg = np.zeros(1, dtype=np.int64)
            out_s = np.full((1, n_b), 77, dtype=np.int8)
            from stimside._native import _kernel_loader as kl
            orig_lib._raw.engine_peek_pauli_batch(
                sim._cpp_eng, 1, tg.ctypes.data_as(kl.i64_p), 2,
                xs_c.ctypes.data_as(kl.u8_p), zs_c.ctypes.data_as(kl.u8_p),
                *(a.ctypes.data_as(kl.u64_p) for a in pw[:4]),
                pw[4].ctypes.data_as(kl.u8_p), pw[5].ctypes.data_as(kl.u8_p),
                out_s.ctypes.data_as(kl.i8_p),
            )
            if not np.all(out_s == 77):
                return "raw_peek_touched_sentinel", hooked_fn, state["calls"], state["allocs"]
        for post in (
            lambda: sim.peek_pauli_batch([0], "Z"),
            lambda: sim.active_shots,
            sim.get_detector_flips,
            sim.get_leakage_events,
            sim.clear,
        ):
            try:
                post()
                return "post_call_succeeded", hooked_fn, state["calls"], state["allocs"]
            except RuntimeError as exc:
                msg = str(exc)
                if "MemoryError" not in msg or "new simulator" not in msg:
                    return f"bad_failed_msg:{msg}", hooked_fn, state["calls"], state["allocs"]
        return "MemoryError", hooked_fn, state["calls"], state["allocs"]
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}", hooked_fn, state["calls"], state["allocs"]


if case == "pub_set_rng_states":
    status, hooked_fn, hook_calls, allocs = _run_one(case, 0)
    print(json.dumps({
        "case": case,
        "hooked_fn": hooked_fn,
        "hook_calls": hook_calls,
        "allocs": allocs,
        "results": [status],
    }))
else:
    _, hooked_fn, hook_calls, allocs = _run_one(case, 10**9)
    assert hook_calls >= 1 and allocs >= 1, (case, hooked_fn, hook_calls, allocs)
    results = [_run_one(case, k)[0] for k in range(allocs)]
    print(json.dumps({
        "case": case,
        "hooked_fn": hooked_fn,
        "hook_calls": hook_calls,
        "allocs": allocs,
        "results": results,
    }))
"""


def _under_tsan() -> bool:
    try:
        libc = ctypes.CDLL(None)
        return hasattr(libc, "__tsan_init")
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    _under_tsan() or shutil.which("c++") is None,
    reason="needs a C++ compiler and no TSan runtime (TSan owns operator new)",
)


@pytest.fixture(scope="module")
def fi_so(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("coset_fi")
    cpp, so = d / "fi_new.cpp", d / "libfi_new.so"
    cpp.write_text(_FI_CPP, encoding="utf-8")
    subprocess.run(["c++", "-shared", "-fPIC", "-O1", str(cpp), "-o", str(so)], check=True)
    return so


def _run_child(fi_so: Path, case: str, timeout: int = 300) -> dict[str, Any]:
    existing_preload = os.environ.get("LD_PRELOAD", "").strip()
    ld_preload = f"{fi_so} {existing_preload}" if existing_preload else str(fi_so)
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "LD_PRELOAD": ld_preload,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    if existing_preload:
        asan_opts = env.get("ASAN_OPTIONS", "")
        if "verify_asan_link_order=0" not in asan_opts:
            env["ASAN_OPTIONS"] = (
                f"{asan_opts}:verify_asan_link_order=0"
                if asan_opts
                else "verify_asan_link_order=0"
            )
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(_SRC), str(fi_so), case],
        capture_output=True, text=True, env=env, timeout=timeout, check=False,
    )
    msg = f"{case}: rc={proc.returncode}\n{proc.stdout}\n{proc.stderr[-2500:]}"
    assert proc.returncode == 0, msg
    return json.loads(proc.stdout.strip().splitlines()[-1])  # type: ignore[no-any-return]


_CASES = [
    "pub_engine_create",
    "pub_register_unitary_step",
    "pub_register_unitary_group_evict",
    "pub_disjoint_clifford_v3",
    "pub_inject_group_v3",
    "pub_det_meas_step",
    "pub_meas_step_v4_rand",
    "pub_mixed_substep_det",
    "pub_mixed_substep_rand",
    "pub_unmatched_reset_z",
    "pub_project_pauli_xy",
    "pub_postselect_batch",
    "pub_peek_pauli_batch",
    "sticky_s5_eliminate_zero",
]


@pytest.mark.parametrize("case", _CASES)
def test_coset_engine_oom_becomes_memory_error_and_latches_failed_engine(
    case: str, fi_so: Path
) -> None:
    out = _run_child(fi_so, case)
    print(json.dumps({
        "case": out["case"],
        "hooked_fn": out["hooked_fn"],
        "hook_calls": out["hook_calls"],
        "allocs": out["allocs"],
    }))
    assert out["hook_calls"] >= 1, out
    assert out["allocs"] >= 1, out
    assert out["results"] == ["MemoryError"] * out["allocs"], out


def test_coset_engine_set_rng_states_zero_alloc(fi_so: Path) -> None:
    out = _run_child(fi_so, "pub_set_rng_states")
    assert out["hooked_fn"] == "engine_set_rng_states", out
    assert out["hook_calls"] >= 1, out
    assert out["allocs"] == 0, out
    assert out["results"] == ["ok"], out
