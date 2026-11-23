"""Differential, hostile-input, and re-entrancy tests for batched Z-basis measurement v4."""
import ctypes
import sys

import numpy as np
import pytest
import stim

from stimside import CosetsideSimulator, LeakageUint8Coset
from stimside._native import _kernel_loader as kl
from stimside._native.coset_kernels import CosetCallBlock, call_block_p

_C, _CI, _U8, _I32, _I64 = stim.Circuit, stim.CircuitInstruction, kl.u8_p, kl.i32_p, kl.i64_p
_NU8, _NBLK = ctypes.cast(0, _U8), ctypes.cast(0, call_block_p)
_L, _CZ, _Q = "I[LEAKAGE_TRANSITION_1: (0.4, U-->2)]", "CZ[CONDITIONED_ON_PAIR: (U, U)]", "0 1 2 3"
_P3, _W, _H4 = f"R 0 1 2\nH 0 1\nCX 0 2\n{_L} 0\n{_CZ} 0 1\n", "65 66 68 69", (
    f"R {_Q}\nH 0 2\nCX {_Q}\n{_L} 0\n{_CZ} {_Q}\n")
_SCEN = (("det", f"R {_Q}\nH {_Q}\n{_L} 0 2\n{_CZ} {_Q}\nCZ {_Q}\nH {_Q}\nX 1 3\nM {_Q}\nMZ {_Q}\n"
                 + f"X 0 2\nMR {_Q}\nM {_Q}\nX 1\nMRZ {_Q}\nR {_Q}\nX 2\nRZ {_Q}\nM {_Q}"),
    ("mixed", f"{_H4}H 1 3\nM {_Q}\nH {_Q}\n{_L} 0\n{_CZ} {_Q}\nMR {_Q}\nM {_Q}"),
    ("inv", f"{_P3}X 1\nM !0 1 !2\nMR !0 1 !2\nM 0 1 2"), ("noisy", f"{_H4}M(0.25) {_Q}\n"
     + f"MR(0.25) !0 1 2 !3\nM {_Q}"), ("dups", f"{_P3}M 0 0 1 2 0 1 2\nMR 0 0 1 1 0 2\n"
     + "R 0 0 1 1\nM 0 1 2"), ("mpp", f"{_H4}MPP Z0 !Z1\nTICK\nMPP Z0*Z0\nMX 0 1\nMY 2 3\nMRX 0 1\n"
     + f"MRY 2 3\nMZZ {_Q}\nM {_Q}"), ("mw70", f"R 0 1 {_W}\nH 0 65 68\nCX 0 1 {_W}\n{_L} 65\n"
     + f"{_CZ} {_W}\nM 0 1 {_W}\nH 65 68\nMR {_W}\nM 0 {_W}"))
_TM = {"u": _U8, "3": _I32, "6": _I64, "8": kl.u64_p}
_KS = ("ptr", "num_sub", "is_reset", "is_r", "q_a", "m_r", "p_a", "k_r", "bw", "br", "ep",
       "tot_gx", "gxo", "gxf", "tot_gz", "gzo", "gzf", "x2z", "z2z", "xs", "blk")
def _p(a, t):
    return a.ctypes.data_as(t)
def _mk(c, b=4, seed=1, fl=False, sync=False, cls=CosetsideSimulator, **kw):
    return cls(c, compiled_op_handler=LeakageUint8Coset(fast_leakage=fl).compile_op_handler(
        circuit=c, batch_size=b), batch_size=b, seed=seed, sync_tableside_rng=sync, **kw)
@pytest.mark.parametrize("b", [1, 37, 64, 256], ids=["B1", "B37", "B64", "B256"])
@pytest.mark.parametrize(("name", "body"), _SCEN, ids=[n for n, _ in _SCEN])
def test_fast_vs_general_bit_exact(name, body, b):
    eq, s0, c = np.array_equal, 100 + b + len(name), _C(body)
    f, g, r = _mk(c, b, s0), _mk(c, b, s0), list(range(c.num_qubits))
    g._use_fast_glue = False
    assert (f.run(), g.run()) == (None, None) and set(f.active_shots) == set(g.active_shots) and (
        eq(f.get_final_measurement_records(), g.get_final_measurement_records())) and all(
        a.d == (m := g.active_shots[i]).d and all(eq(x, y) for x, y in ((a.h_words, m.h_words),
        (a.a_union, m.a_union), (a.P[:a.d], m.P[:a.d]), (a.A_buf[:a.d], m.A_buf[:a.d]),
        (a.ell_buf[:a.d], m.ell_buf[:a.d]), (a.Gamma_buf[:a.d, :a.d], m.Gamma_buf[:a.d, :a.d])))
        for i, a in f.active_shots.items()) and all(
        eq(f.peek_pauli_batch(r, q), g.peek_pauli_batch(r, q)) for q in "XYZ")
def test_v4_dispatch_routing_and_buffer_growth(monkeypatch):
    sim = _mk(_C("R 0 1\nH 0\nMPP Z0 !Z1\nTICK\nMPP Z0*Z0\nMX 0 1\nM 0 1 0 1 0"), 16, 7)
    lib, cnt = sim._cpp_lib, {"v4": 0, "mixed": 0}
    for attr, k in (("engine_batch_meas_step_v4", "v4"), ("engine_batch_mixed_substep", "mixed")):
        fn = getattr(lib, attr)
        monkeypatch.setattr(lib, attr, lambda *a, f=fn, k=k: (cnt.update({k: cnt[k]+1}), f(*a))[1])
    assert sim.run() is None and cnt == {"v4": 3, "mixed": 2} and sim._call_block.m_buf_rows == 5
def _v4_fx(sim):
    nq, nb, nw, z1 = sim.num_qubits, sim.batch_size, sim._n_words, np.zeros(1, np.uint8)
    ox, oz, mb = (np.full((r, nb), 0xAA, np.uint8) for r in (nq, nq, 2))
    npb, x2z = (nb + 7) >> 3, (np.uint64(1) << np.arange(nq, dtype=np.uint64))[:, None]
    return {"sim": sim, "lib": sim._cpp_lib, "raw": sim._cpp_lib._raw, "ptr": sim._cpp_owner.ptr,
            "num_sub": 1, "N": nq, "B": nb, "nw": nw, "ox": ox, "oz": oz, "mb": mb, "is_r": z1,
            "xs": np.zeros((nq, npb), np.uint8), "x2z": x2z, "z2z": x2z.copy(), "m_r": z1,
            "blk": CosetCallBlock(nq, nb, nw, npb, 2, 7, 9, 0, _p(ox, _U8), _p(oz, _U8), _p(mb,
            _U8)), "q_a": np.array([0], np.int64), "bw": np.ones((1, nw), np.uint64)}
def _c4(fx, **ov):
    return int(fx["raw"].engine_batch_meas_step_v4(*(v if t == "0" else (_NBLK if v is None else
        ctypes.byref(v)) if t == "1" else ctypes.cast(0, _TM[t]) if v is None else _p(v, _TM[t])
        for k, t in zip(_KS, "000u6u6388803603688u1") for v in (ov.get(k, fx.get(k, 0 if t == "0"
        else None)),))))
def test_hostile_v4_validation_atomicity_and_ubsan():
    u8, i32, i64, ar = np.uint8, np.int32, np.int64, np.array
    fx = _v4_fx(sim := _mk(_C("H 0 1\nCZ 0 1\nI 2 3"), 8, 1))
    blk, nq, nb, nw = fx["blk"], fx["N"], fx["B"], fx["nw"]
    assert _c4(fx, ptr=0) == -1 and (blk.any_x, blk.any_z) == (7, 9)
    for f, bad, ok, rc in (*[(k, getattr(blk, k) + 1, getattr(blk, k), -2) for k in ("n_q", "n_b",
            "n_w", "n_packed_bytes")], ("scratch_x_mask", _NU8, _p(fx["ox"], _U8), -1),
            ("m_buf_rows", -1, 2, -1), ("scratch_z_mask", _NU8, _p(fx["oz"], _U8), -1),
            ("m_buf", _NU8, _p(fx["mb"], _U8), -1)):
        blk.any_x, blk.any_z, _ = 7, 9, setattr(blk, f, bad)
        assert _c4(fx) == rc and (rc != -2 or (blk.any_x, blk.any_z) == (0, 0))
        setattr(blk, f, ok)
    assert _c4(fx, blk=None) == -1 and _c4(fx, num_sub=-1) == -3 == _c4(fx, num_sub=3) and all(
        _c4(fx, **{k: None}) == -1 for k in ("x2z", "z2z", "is_r", "q_a", "m_r", "bw", "xs")) and (
        all(np.all(fx[k] == 0xAA) for k in ("ox", "oz", "mb"))) and (sim.interactive_do(
        _CI("H", [0, 1])), sim.do_conditional_clifford(_CI("CZ", [0, 1]), np.arange(nb)[None, :] > 0
        )) == (None, None)
    idx, ds, gi = np.zeros(nb, i32), np.zeros(nb, i32), fx["lib"].engine_get_active_shot_indices
    assert gi(fx["ptr"], _p(idx, _I32), _p(ds, _I32)) == 1 and ds[0] == 2 and all(
        _c4(fx, q_a=ar([q], i64)) == -4 for q in (-1, nq)) and _c4(fx, num_sub=2, is_r=np.zeros(
        2, u8), q_a=ar([0, nq], i64), m_r=np.zeros(2, u8), bw=np.ones((2, nw), np.uint64)) == -4
    assert gi(fx["ptr"], _p(idx, _I32), _p(ds, _I32)) == 1 and ds[0] == 2 and (fx["mb"]==0xAA).all()
    base = {"is_r": ar([1], u8), "q_a": fx["q_a"], "m_r": ar([0xFF], u8), "bw": fx["bw"],
        "p_a": fx["q_a"], "k_r": ar([2147483647], i32), "br": fx["bw"], "ep": fx["bw"], "tot_gx": 1,
        "gxo": ar([0, 1], i32), "gxf": ar([1], i64), "tot_gz": 1, "gzo": ar([0, 1], i32),
        "gzf": ar([2], i64)}
    assert all(_c4(fx, **{**base, **ov}) == rc for rc, ov in (
        *[(-1, {k: None}) for k in ("p_a", "k_r", "br", "ep", "gxo", "gzo", "gxf", "gzf")],
        *[(c, {k: ar([q], i64)}) for q in (-1, nq) for k, c in (("p_a", -5), ("gxf", -7),
          ("gzf", -7))], *[(-6, {k: -1}) for k in ("tot_gx", "tot_gz")],
        *[(-6, {k: ar(o, i32)}) for o in ((-1, 1), (1, 0), (0, 2)) for k in ("gxo", "gzo")]))
    fx["mb"][:], fx["ox"][:], fx["oz"][:] = 0xFF, 0, 0
    assert _c4(fx, **base) == 0 and set(np.unique(fx["mb"][0]).tolist()) <= {0, 1} and (
        setattr(blk, "m_buf_rows", 0), setattr(blk, "m_buf", _NU8),
        _c4(fx, num_sub=0, is_r=None, q_a=None, m_r=None, bw=None, xs=None))[2] == 0
def _run_hooked(pred, body, action):
    fired = [0]
    sys.setprofile(lambda fr, ev, _a: (fired.__setitem__(0, fired[0] + 1), sys.setprofile(None),
        action(), None)[-1] if ev == "call" and fr.f_code.co_name == "data_as" and (p := fr.f_back)
        and pred(p) else None)
    try:
        body()
    finally:
        sys.setprofile(None)
    assert fired[0] == 1
def test_v4_mbuf_growth_interrupt_reentry_and_rng():
    v4 = "_exec_meas_step_v4"
    for b in (64, 2048):
        sim = _mk(_C("X 0\nM 0\nTICK\nM 0 0 0"), b, 42)
        with pytest.raises(KeyboardInterrupt, match="growth-int"):
            _run_hooked(lambda p: p.f_code.co_name == v4 and "new_buf" in p.f_locals, sim.run,
                        lambda: (_ for _ in ()).throw(KeyboardInterrupt("growth-int")))
        junk = [np.full((1, b), 0x55, np.uint8) for _ in range(100)] if b <= 1024 else []
        assert type(sim).__name__ == "_AbortedCosetsideSimulator" and sim.clear() is None and (
            ctypes.cast(sim._call_block.m_buf, kl.void_p).value == sim._cpp_m_buf.ctypes.data) and (
            sim.run() is None and bool(sim.get_final_measurement_records()[:, 0].all()) and
            all(np.all(a == 0x55) for a in junk))
    for b, pre, out, inn, g, rw, cl in ((2048, "X 0\nM 0", "M 0 0 0", "M 0 0 0 0 0", True, 5, 9),
                                        (64, "X 0", "M 0", "M 0 0 0 0", False, 4, 5)):
        s = _mk(_C(pre), b, 7)
        assert (s.interactive_do(_C(pre)), s._cpp_m_buf.fill(0)) == (None, None)
        _run_hooked(lambda p, g=g: p.f_code.co_name == v4 and (not g or "new_buf" in p.f_locals),
            lambda s=s, o=out: s.interactive_do(_C(o)), lambda s=s, i=inn: s.interactive_do(_C(i)))
        assert (s.finish_interactive_run(), s._call_block.m_buf_rows) == (None, rw) and (
            r := s.get_final_measurement_records()).shape == (b, cl) and bool(r.all())
    for inn, grow in (([2, 2, 2, 2], True), ([2], False)):
        s = _mk(c := _C("H 0 1\nX 2"), 64, 17)
        old_id, rng, drawn, _ = id(s._cpp_m_buf), s.np_rng, [], s.interactive_do(c)
        s.np_rng = type("_Rng", (), {"integers": lambda _s, *a, _i=inn, _r=rng, _sm=s, _d=drawn,
            **kw: (_sm.interactive_do(_CI("M", _i)) if len(_d) == 1 else None,
            _d.append(v := _r.integers(*a, **kw)), v)[2]})()
        assert (s.interactive_do(_CI("M", [0, 1])), s.finish_interactive_run(), len(drawn)) == (
            None, None, 2) and (id(s._cpp_m_buf) != old_id) == grow and bool(
            (r := s.get_final_measurement_records())[:, :-2].all()) and np.array_equal(
            r[:, -2:], np.column_stack(drawn)) and np.array_equal(
            s.peek_pauli_batch([0, 1], "Z"), 1 - 2 * r[:, -2:].T)
    sa, sf = _mk(ca := _C("X_ERROR(1) 0 1"), 64, 21), _mk(cf := _C("H 0\nX_ERROR(1) 1"), 16, 17)
    assert sa.interactive_do(ca) is None and sf.interactive_do(cf) is None
    _run_hooked(lambda p: p.f_code.co_name == v4, lambda: sa.interactive_do(_CI("MR", [0])),
                lambda: sa.interactive_do(_CI("MR", [1])))
    rr, fd, ea = sf.np_rng, [0], type("_EA", (), {"__array__": lambda slf, dtype=None, copy=None: (
        (fd.__setitem__(0, 1), sf.interactive_do(_CI("MR", [1]))) if not fd[0] else None,
        slf._a if dtype is None else slf._a.astype(dtype))[1]})
    sf.np_rng = type("_ER", (), {"integers": lambda _s, *a, **kw: (
        setattr(e := ea(), "_a", rr.integers(*a, **kw)), e)[1]})()
    assert (sf.interactive_do(_CI("M", [0, 1])), sf.finish_interactive_run(),
        sa.finish_interactive_run(), fd[0]) == (None, None, None, 1) and bool(
        (rf := sf.get_final_measurement_records())[:, 0].all()) and not bool(rf[:, 2].any()) and (
        bool((sa.peek_pauli_batch([0, 1], "Z") == 1).all()))
