"""Tests for cached op records, conditional Pauli noise v4, unitary v4, and mixed 2Q feedback."""
import copy
import ctypes
import sys

import numpy as np
import pytest
import stim

from stimside import CosetsideSimulator, LeakageUint8Magic, MagicsideSimulator
from stimside._native import _kernel_loader as kl
from stimside.op_handlers.abstract_op_handler import _TrivialCompiledOpHandler
from stimside.op_handlers.leakage_handlers import leakage_uint8_coset as luc
from stimside.simulator_coset_cmeas_test import _C, _CI, _L, _NBLK, _NU8, _mk

_P2 = ",".join(["0.05", "0.06", "0.07", "0.04"] * 3 + ["0.05", "0.06", "0.07"])
_B = (f"R 0 1 2 3\nH 0 1 2 3\nX_ERROR(0.3) 0 1\nDEPOLARIZE2(0.4) 0 1 2 3\n{_L} 0 2\nX_ERROR(0.3)"
      " 0 1 0\nY_ERROR(0.3) 1 2\nZ_ERROR(0.3) 2 3\nX_ERROR(0.0) 0\nX_ERROR(1.0) 1\nDEPOLARIZE1(0.4)"
      f" 0 1 2 3\nPAULI_CHANNEL_1(0.2,0.2,0.2) 0 1 2 3\nDEPOLARIZE2(0.5) 0 1 2 3\nPAULI_CHANNEL_2"
      f"({_P2}) 0 1 2 3\nX_ERROR[CONDITIONED_ON_SELF: U](0.3) 0 2\nDEPOLARIZE2[CONDITIONED_ON_PAIR"
      ": (U, U)](0.5) 0 1\nH 0 1 2 3\nM[LEAKAGE_PROJECTION_Z: (1.0, 2)] 0 1 2 3")
_UB = (f"R 0 1 2 3\nH 2 3\nZ_ERROR(0.5) 0 1 2 3\nX_ERROR(0.4) 0 1 2 3\n{_L} 1 2\nSQRT_X 0 1\n"
       "CX 3 2\nS 0 1\nS 0 1\nSWAP 3 2\nCZ[CONDITIONED_ON_PAIR: (U, U)] 3 2\nH 0 1")
@pytest.mark.parametrize(("b", "fl"), [(1, 0), (7, 0), (37, 0), (64, 0), (64, 1)])
def test_cnoise_and_cunitary_fast_vs_general_bit_exact(b, fl):
    eq, qs = np.array_equal, [0, 1, 2, 3]
    for body, off in ((_B, 50), (_UB, 90)):
        f, g = _mk(c := _C(body), b, off + b, bool(fl)), _mk(c, b, off + b, bool(fl))
        g._use_fast_glue, g._use_cpp = False, body is _B
        assert (f.run(), g.run()) == (None, None) and set(f.active_shots) == set(g.active_shots)
        assert eq(f.get_final_measurement_records(), g.get_final_measurement_records()) and all(
            eq(f.peek_pauli_batch(qs, p), g.peek_pauli_batch(qs, p)) for p in "XYZ")
    m = np.array([[i % 2 == 0 for i in range(b)], [i % 3 != 0 for i in range(b)]], np.bool_)
    f2, g2 = _mk(c2 := _C("R 0 1 2 3\nH 0 1 2 3\nZ_ERROR(0.5) 0 1 2 3"), b, 9), _mk(c2, b, 9)
    g2._use_fast_glue, g2._use_cpp, ss = False, False, _CI("S", [0, 1])
    assert all((s.interactive_do(c2), s.do_conditional_pauli_noise(_CI("DEPOLARIZE2", qs, [.5]), m),
        s.do_conditional_clifford(ss, m), s.do_conditional_clifford(ss, m),
        s.finish_interactive_run()) == (None,) * 5 for s in (f2, g2)) and not f2.active_shots and (
        not g2.active_shots and all(eq(f2.peek_pauli_batch(qs, p), g2.peek_pauli_batch(qs, p))
        for p in "XYZ"))
def test_crecord_hostile_v4_and_reentry():
    c = _C(f"R 0 1\n{_L} 0\nH 0 1\nX_ERROR(0.3) 0 1\nM[LEAKAGE_PROJECTION_Z: (1.0, 2)] 0 1")
    s, s2, sr, sg = _mk(c, 16, 3, True), _mk(c, 8, 3, True), _mk(c, 8, 11), _mk(c, 8, 11)
    assert s.run() is None and s._compiled_op_handler._fast_1q and (
        len(s._compiled_op_handler._op_records) >= 2) and s2.interactive_do(c) is None and not (
        s._compiled_op_handler._get_op_record(_CI("CX", [stim.target_rec(-1), 0])).fast_unitary_ok)
    with pytest.raises(ValueError):
        s2.interactive_do(_CI("M", [0]))
    hm, op_t = LeakageUint8Magic().compile_op_handler(circuit=c, batch_size=4), _CI(
        "X_ERROR", [0], [0.2], tag="CONDITIONED_ON_SELF: U")
    ms, hc, seen, sg._use_fast_glue, r0 = MagicsideSimulator(c, op_handler=LeakageUint8Magic(),
        batch_size=4, seed=1), copy.copy(hm), [0], False, sr.np_rng
    hc.ops_to_params = {**hm.ops_to_params, op_t: next(iter(hm.ops_to_params.values()))}
    hc.claimed_ops_keys, Cl = hm.claimed_ops_keys | {op_t}, luc.CompiledLeakageUint8Coset
    sub = type("_S", (Cl,), {"handle_op": lambda self, op, sss: (seen.__setitem__(0, seen[0] + 1),
        Cl.handle_op(self, op, sss))[1]})
    sr.np_rng = type("_R", (), {"__getattr__": lambda _s, k: getattr(r0, k)})()
    assert CosetsideSimulator(c, compiled_op_handler=sub(2, 8, True, dict(hm.ops_to_params)),
        batch_size=8, seed=3).run() is None and (copy.copy(ms).compiled_op_handler._op_records
        is not ms.compiled_op_handler._op_records) and not hm._get_op_record(op_t).claimed and (
        hc._get_op_record(op_t).claimed and seen[0] == 5 and (sr.run(), sg.run()) == (None, None)
        and np.array_equal(sr.get_final_measurement_records(), sg.get_final_measurement_records()))
    hs, ts = _mk(_C("H 0 1 2 3\nCZ 0 1 2 3"), 8, 1), (kl.u8_p, kl.i64_p, kl.f64_p, kl.void_p)
    ni64, nf64, nv, u8 = *(ctypes.cast(0, t) for t in ts[1:]), np.uint8
    tg, p1, p3 = np.array([0, 1], np.int64), np.array([0.5]), np.array([0.2, 0.2, 0.2])
    p15, ls, cm = np.cumsum(np.full(15, 0.05)), np.zeros((4, 8), u8), np.ones((2, 8), u8)
    (_, fn, st), b15, xp = hs._bitgen_ptrs(), p15.copy(), np.zeros((4, 1), u8)
    b15[5], ptr = b15[4] - 0.01, hs._cpp_owner.ptr
    sid = hs._register_unitary_step_cpp(hs._compiled_ref.steps[1])
    br, raw = ctypes.byref(hs._call_block), hs._cpp_lib._raw
    def _c4(**k):
        p = [x.ctypes.data_as(t) if isinstance(x, np.ndarray) else x for x, t in zip((k.get("t", tg
            ), k.get("pr", p1), k.get("ls", _NU8), k.get("m", cm)), (ts[1], ts[2], ts[0], ts[0]))]
        return int(raw.engine_cond_pauli_noise_v4(k.get("ptr", ptr), k.get("ch", 0), k.get("ng", 1),
            p[0], k.get("n", 1), p[1], p[2], p[3], k.get("f", fn), k.get("s", st), k.get("b", br)))
    def _u4(**k):
        return int(raw.engine_batch_inject_unitary_v4(k.get("ptr", ptr), k.get("sid", sid), k.get(
            "r", 2), k.get("c", 8), *(x.ctypes.data_as(ts[0]) if isinstance(x, np.ndarray) else x
            for x in (k.get("m", cm), k.get("ls", _NU8), k.get("x", xp), k.get("z", xp))),
            k.get("b", br)))
    assert all(_c4(**ov) == rc for rc, ov in ((-1, {"ptr": 0}), (-1, {"b": _NBLK}), (-1, {"f": nv}),
        (-1, {"t": ni64}), (-1, {"s": nv}), (-1, {"m": _NU8}), (-1, {"ls": ls}), (-3, {"ch": -1}),
        (-3, {"ch": 6}), (-4, {"ng": -1}), (-4, {"n": 2}), (-4, {"pr": nf64}), (-4, {"ch": 4,
        "ng": 0x40000000}), (-5, {"ch": 5, "n": 15, "pr": b15}), (-5, {"ch": 3, "n": 3,
        "pr": np.array([0.5] * 3)}), (1, {}), (0, {"ng": 0, "t": ni64, "m": _NU8, "f": nv,
        "s": nv}), (0, {"ch": 4, "pr": np.zeros(1)}), *[(-5, {"pr": np.array([v])}) for v in (
        np.nan, -0.1, 1.1)], *[(-6, {"t": np.array([q], np.int64)}) for q in (-1, 4)]))
    cm[0, 0], zcb = 0, ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_void_p)(lambda _: 0.0)
    s0, x0, n64 = _mk(_C(""), 8, 1), np.zeros((1, 1), np.uint64), ctypes.cast(0, kl.u64_p)
    p0, b0 = s0._cpp_owner.ptr, type(s0._call_block)(0, 8, 1, 1, 0, 0, 0, 0, _NU8, _NU8, _NU8)
    sd0 = int(s0._cpp_lib._raw.engine_register_unitary_step(p0, 0, 1, 0, ni64, *([_NU8] * 3 + [n64]
        * 2 + [_NU8] * 3 + [n64] * 2 + [_NU8]), 0, 1, *(x0.ctypes.data_as(kl.u64_p),) * 2))
    assert _c4(ch=4, pr=np.array([1e-323]), f=ctypes.cast(zcb, ts[3])) == 0 and (
        hs._call_block.any_x, hs._call_block.any_z) == (0, 0) and _c4(ch=3, n=3, pr=p3) == 0 and (
        _c4() == 0 == _c4(ch=5, n=15, pr=p15) == _u4(ptr=p0, sid=sd0, r=0, c=8, m=_NU8,
        b=ctypes.byref(b0))) and all(_u4(**ov) == rc for rc, ov in ((-1, {"ptr": 0}),
        (-1, {"b": _NBLK}), (-1, {"x": _NU8}), (-1, {"z": _NU8}), (-2, {"sid": -1}),
        (-2, {"sid": 99}), (-3, {"ls": ls, "r": 0, "c": 0}), (-3, {"m": _NU8, "ls": ls, "r": 1,
        "c": 0}), (-3, {"r": 1}), (-3, {"c": 7}), (-4, {"m": _NU8}), (0, {}),
        (0, {"m": _NU8, "ls": ls, "r": 0, "c": 0})))
    sr2, fired = _mk(_C(f"R 0 1\n{_L} 0\nH 0 1\nM 0 1"), 8, 5), [0]
    sys.setprofile(lambda fr, ev, _a: (fired.__setitem__(0, 1), setattr(sr2, "_used_interactively",
        True), None)[-1] if not fired[0] and ev == "return"
        and fr.f_code.co_name == "_register_unitary_step_cpp" else None)
    try:
        assert (sr2.run(), fired[0], sr2.get_final_measurement_records().shape) == (None, 1, (8, 2))
    finally:
        sys.setprofile(None)
def test_d74_mixed_2q_feedback_ordering(monkeypatch):
    th = _TrivialCompiledOpHandler()
    for g in ("CX", "CNOT", "ZCX", "CY", "ZCY", "XCZ", "YCZ"):
        p = "1 3 3 rec[-1]" if g in ("XCZ", "YCZ") else "3 1 rec[-1] 3"
        (tb := stim.TableauSimulator(seed=0)).do(
            c := _C(f"R 0 1 2 3\nX 1 2 3\nM 2\n{g} {p}\nM 0 1 3"))
        ref = np.array(tb.current_measurement_record(), np.bool_)
        assert all((setattr(s := CosetsideSimulator(c, compiled_op_handler=th, batch_size=4, seed=3,
            sync_tableside_rng=bool(sy)), "_use_cpp", bool(cp)), s.run(),
            np.all(s.get_final_measurement_records() == ref))[1:] == (None, True)
            for sy in (0, 1) for cp in (1, 0))
    cl = _C(f"R 0 1 2 3\nX 1 2 3\nM 2\n{_L} 1\nCX 3 1 rec[-1] 3\nM 0 1 3")
    sc1, sc2, cnt = _mk(cl, 16, 13, sync=True), _mk(cl, 16, 13, sync=True), [0]
    sc2._use_cpp, sn = False, _mk(_C(f"R 0 1 2 3\nX 0\nM 0\n{_L} 2\nCX rec[-1] 1 2 3\nM 1 2 3"), 8)
    monkeypatch.setattr(sn._cpp_lib, "engine_batch_inject_unitary_v4", lambda *a,
        f=sn._cpp_lib.engine_batch_inject_unitary_v4: (cnt.__setitem__(0, cnt[0] + 1), f(*a))[1])
    assert (sc1.run(), sc2.run(), sn.run()) == (None, None, None) and cnt[0] >= 1 and (
        np.array_equal(r := sc1.get_final_measurement_records(), sc2.get_final_measurement_records()
        ) and bool((r[:, 0] == 1).all()) and bool((r[:, 3] == 0).all()) and not bool(r[:, 2].all()))
