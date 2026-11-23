"""Tests for persistent engine buffers, measurement step, batch Pauli noise, and guards."""
from __future__ import annotations

import contextlib
import copy
import ctypes
import re
from typing import Any, cast

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

from stimside import MagicsideSimulator, simulator_magic
from stimside._magic import channels, gates, live_store, phase_qfe
from stimside._native import magic_engine
from stimside.util.reference_chp import ReferenceCHPTableau

C, MS, sm, me = stim.Circuit, MagicsideSimulator, simulator_magic, magic_engine
mc, mg, ls, pq = channels, gates, live_store, phase_qfe
_IDS = (r"\bR\d+-", r"\bRL-", r"\bPKG-\d+", r"\bD-\d+", r"\bWP-\d+", r"\bRun \d+")
@pytest.fixture(autouse=True)
def _force_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", True)
def R(exc: type[BaseException], pat: str, fn: Any) -> Any:
    with pytest.raises(exc, match=pat) as ei:
        fn()
    return ei.value
def _t_st(n: int = 4, q: int = 0) -> pq.ShotExpansion:
    a, b = np.array([1 << q], np.uint64), np.zeros(1, np.uint64)
    s0 = pq.apply_quarter_turn(pq.ShotExpansion.zero(n, atol=1e-10), a, b, 0, sign=1, atol=1e-10)
    return pq.apply_pauli_t(s0, b, a, 0, dag=False, merge=True)
def _load(b: int, n: int = 4, st: tuple[pq.ShotExpansion, ...] = ()) -> Any:
    e = me.MagicBatchEngine(b, n, 1e-10)
    [e.import_shot(i, ls.expansion_to_arrays(s)) for i, s in enumerate(st)]
    return contextlib.closing(e)
def _ax(q: int = 0, z: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
    u, w = np.zeros(1, np.uint64), np.array([1 << q], np.uint64)
    return (u, w, 0) if z else (w, u, 0)
def _cnt(mp: pytest.MonkeyPatch, *pairs: tuple[str, str]) -> dict[str, int]:
    lib, cnt = me.get_magic_engine_lib(), {k: 0 for k, _ in pairs}
    for k, nm in pairs:
        o = getattr(lib, nm)
        mp.setattr(lib, nm, lambda *a, _k=k, _o=o: (cnt.__setitem__(_k, cnt[_k] + 1), _o(*a))[1])
    return cnt
def _run(c: stim.Circuit, b: int, sd: int, fast: bool, oh: Any = None) -> tuple[bytes, ...]:
    s = MS(c, op_handler=oh, batch_size=b, seed=sd)
    s._use_fast_glue = fast
    assert copy.copy(s)._use_fast_glue is fast and copy.copy(s)._bitgen_cache is None
    s.run()
    return (s.get_final_measurement_records().tobytes(), s._log_branch_masses.tobytes(),
            s.get_detector_flips().tobytes(), s.get_observable_flips().tobytes())
def test_buffers_owned_axis_and_cond_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    a, b, kap = _ax(0, False)
    am, bm, evil = a.copy(), b.copy(), type("_E", (int,), {"__ne__": lambda s, o: False})
    owned = me.own_axis(am, bm, kap, 1)
    am[:] = bm[:] = 0
    assert int(owned.a_words[0]) == 1 and not owned.a_words.flags.writeable
    R(ValueError, "WRITEABLE", lambda: setattr(owned.a_words.flags, "writeable", True))
    [R(ValueError, "n_words", lambda x=v: me.own_axis(a, b, kap, cast(Any, x)))
     for v in (evil(1), True, 1.0, 0, -1)]
    ft = me.own_factor_table(ls.factor_table(
        [mc.CompiledCliffordFactorList(1, None, (mg.CompiledPauliAxis(*_ax(0, False)),))], 1), 1)
    with _load(2, 4, (_t_st(4, 0), _t_st(4, 1))) as e1, _load(2, 70) as e2:
        z0, x0, s0 = _ax(), _ax(0, False), np.array([0], np.int32)
        sp1, sp2 = e1.measure_split(z0, False, None), e1.measure_split(x0, False, None)
        p1, p2 = e1.postselect(z0, s0, [0]), e1.postselect(z0, [1], [0])
        assert all(not np.shares_memory(u, v) for u, v in (
            (e1.export_shot(0).weights, e1.export_shot(0).weights),
            (e1.partition(True), e1.partition(True)), (e1.peek(z0, s0)[0], e1.peek(x0, s0)[0]),
            (p1.done, p2.done), (p1.z01, p2.z01),
            *((getattr(sp1, f), getattr(sp2, f)) for f in ("cls", "bit", "z", "z01", "chi2"))))
        object.__setattr__(owned, "a_words", np.zeros(1, np.uint64))
        object.__setattr__(owned, "n_words", 2)
        assert np.array_equal(e1.peek(x0, s0)[0], e1.peek(owned, s0)[0])
        called, ent = [False], me._OWNED_AXES[owned]
        monkeypatch.setattr(e2._lib, "mbe_peek", lambda *_: (called.__setitem__(0, True), 0)[1])
        [R(TypeError, "own_axis", lambda x=bad: e2.peek(x, s0)) for bad in (
            me.OwnedAxis(a, b, 0, 2), type("_S", (me.OwnedAxis,), {})(a, b, 0, 2), owned)]
        monkeypatch.setitem(me._OWNED_AXES, owned, (*ent[:5], evil(1)))
        R(TypeError, "own_axis", lambda: e2.peek(owned, s0))
        monkeypatch.setitem(me._OWNED_AXES, owned, ent), setattr(e2, "n_words", evil(2))
        R(TypeError, "own_axis", lambda: e2.peek(owned, s0))
        ic, reent = e1.inject_cliffords, [False]
        assert not called[0] and ic(ft, np.array([-2**31, -5], np.int32), False) is None
        R(RuntimeError, "mbe_inject_cliffords", lambda: ic(ft, np.array([0, 1], np.int32), False))
        e1.apply_pauli_mask(*z0, type("_H", (), {"__array__": lambda s, d=None, copy=None: (
            reent.__setitem__(0, True), e1._busy and 1 / 0, e1.live_shots(),
            np.array([True, False], np.bool_))[3]})())
        assert reent[0] and not e1._busy
    sim = MS(C("H 0 1\nI[T] 0\nMR 0\nM 0 1"), batch_size=4, seed=7)
    st, ax_m = cast(ls.LiveExpansions, sim._live_exps), sim._compile_axis([(0, "Z")])
    assert (ls._OWN_AXES.get(id(ax_m)) is ax_m and type(st._owned_axis(ax_m)) is me.OwnedAxis
            and st.axis_is_encodable(ax_m) and len(st.copy()._axes) == 0)
    monkeypatch.setattr(sm, "_REF_MEMO_MAX_ENTRIES", 0)
    ax0 = MS(C("H 0 1\nI[T] 0\nM 0 1"), batch_size=4, seed=7)._compile_axis([(0, "Z")])
    assert id(ax0) not in ls._OWN_AXES
def test_measure_step_and_pauli_noise_exact_and_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    from stimside import golden_test as gt
    cnt = _cnt(monkeypatch, ("step", "mbe_measure_step"), ("split", "mbe_measure_split"),
               ("fin", "mbe_measure_finish"), ("nb", "mbe_apply_pauli_noise_batch"),
               ("pm", "mbe_apply_pauli_mask"))
    ex = C("H 0 1 2\nI[T] 0 1\nCX 0 2 1 2\nMX 0\nMY 1\nMR 2\nMPP X0*Z1\nM 0 1 2")
    assert all(_run(c, 8, 19, True, gt.LeakageUint8Magic() if lk else None)
               == _run(c, 8, 19, False, gt.LeakageUint8Magic() if lk else None)
               for c, lk in [(C(t), lk) for _, m, lk, t in gt.CORPUS if "magic_ds" in m]
               + [(ex, False)]) and cnt["step"] >= 150
    for cfg in ("fast", "off", "sync", "subgen"):
        cnt["step"] = cnt["split"] = cnt["fin"] = 0
        sim = MS(ex, batch_size=4, seed=5, sync_tableside_rng=(cfg == "sync"))
        sim._use_fast_glue = cfg != "off"
        if cfg == "subgen":
            sim.np_rng = type("_SG", (np.random.Generator,), {})(np.random.PCG64(5))
        sim.run()
        assert ((cnt["step"] > 0 and not (cnt["split"] or cnt["fin"])) if cfg == "fast"
                else (cnt["step"] == 0 and cnt["split"] > 0))
    nc = C("H 0 1 2\nI[T] 0 1\nCX 0 2 1 2\nX_ERROR(0.3) 0 0 1\nY_ERROR(0.2) 1 2\nZ_ERROR(0.4) 0 2\n"
           "DEPOLARIZE1(0.3) 0 1 0\nPAULI_CHANNEL_1(0.0,0.25,0.0) 1 2\nDEPOLARIZE2(0.4) 0 1 1 2\n"
           "PAULI_CHANNEL_2(0,0.1,0,0,0.1,0,0,0,0,0,0.1,0,0,0,0.1) 0 1 0 1\n"
           "CORRELATED_ERROR(0.5) X0*Z1\nELSE_CORRELATED_ERROR(0.5) Y1*Z2\n"
           "HERALDED_ERASE(0.2) 0 1\nHERALDED_PAULI_CHANNEL_1(0.05,0.1,0.1,0.1) 2\nM 0 1 2")
    cnt["nb"] = cnt["pm"] = 0
    o_f = _run(nc, 16, 31, True)
    assert cnt["nb"] >= 1 and cnt["pm"] == 0
    cnt["nb"] = cnt["pm"] = 0
    assert o_f == _run(nc, 16, 31, False) and cnt["nb"] == 0 and cnt["pm"] >= 1
    u_e = np.array([[0.0, 0.25, np.nextafter(0.25, 0.0), 0.75, np.nextafter(0.75, 0.0), 0.99]])
    p1 = stim.CircuitInstruction("PAULI_CHANNEL_1", [0], [0.0, 0.25, 0.5])
    rs: Any = type("_R", (), {"random": lambda s, sh=None: u_e.copy() if isinstance(sh, tuple)
                              else np.full(sh, 0.25)})()
    eo = []
    for f in (True, False):
        s = MS(C("H 0\nI[T] 0\nM 0"), batch_size=6, seed=3)
        s._use_fast_glue, s.np_rng = f, rs
        s.interactive_do(C("H 0\nI[T] 0")), s.do_conditional_pauli_noise(p1)
        s.interactive_do(C("M 0")), s.finish_interactive_run()
        eo.append(s.get_final_measurement_records().tobytes())
    sc = MS(C("H 0\nI[T] 0\nX_ERROR(1) 0 0\nM 0"), batch_size=2, seed=1)
    sc.interactive_do(C("H 0\nI[T] 0\nX_ERROR(1) 0 0"))
    h1, h2 = (type("_S", (), {"_fast_1q": True, "_compiled_len": n, "clear": lambda s: None,
                              "handle_op": lambda s, op, sss: sss._do_bare_instruction(op)})()
              for n in (2, 1))
    ch = mg.CliffordFaultChannel(step_index=0, branches=(mg.FaultBranch(0.5, "X", (0,)),))
    with pytest.warns(UserWarning) as wr:
        MS(C("H 0\nM 0"), compiled_op_handler=h1, batch_size=2, seed=1, fault_channels=(ch,))
    s2 = MS(C("H 0"), compiled_op_handler=h2, batch_size=2, seed=1)
    s2.interactive_do(C("H 0"))
    ev = R(ValueError, "CLIFFORD_FAULT",
           lambda: s2.interactive_do(C("I[CLIFFORD_FAULT:(0.5,X)] 0")))
    assert (eo[0] == eo[1] and len(sc._pending_magic) > 0 and not h1._fast_1q
            and not any(re.search(p, f"{wr[0].message} {ev}") for p in _IDS))
def test_measure_step_and_pauli_noise_hostile_mutants(monkeypatch: pytest.MonkeyPatch) -> None:
    a0, z0 = np.array([1], np.uint64), np.zeros(1, np.uint64)
    bad = pq.ShotExpansion(2, 1, a0.reshape(1, 1), a0.copy(), np.zeros((1, 1), np.uint8),
                           np.array([0], np.int32), np.array([1e6, -1e6 + 0.1], np.complex128),
                           np.zeros((2, 1), np.uint64), np.array([[0], [1]], np.uint8),
                           np.zeros(1, np.uint8), 0.0, 1e-13, True)
    rbs = []
    for f in (True, False):
        sim = MS(C("M 0 1"), batch_size=2, seed=7, atol=1e-13)
        sim._use_fast_glue, sim._live_exps[0], sim._live_exps[1] = f, _t_st(2, 0), bad
        sn0, r0 = sim._live_exps[0].weights.copy(), copy.deepcopy(sim.np_rng.bit_generator.state)
        ev = R(FloatingPointError, "Born mass conservation", sim.run)
        assert (sim.np_rng.bit_generator.state == r0
                and np.array_equal(sim._live_exps[0].weights, sn0))
        rbs.append(f"{type(ev)}:{ev}")
    assert rbs[0] == rbs[1]
    zs = pq.ShotExpansion.zero(4, atol=1e-10)
    with _load(3, 4, (zs, pq.apply_pauli(zs, a0, z0, 0), _t_st(4, 0))) as eng, _load(0, 4) as zeng:
        bg, rng_r = np.random.default_rng(123).bit_generator, np.random.default_rng(123)
        fn = cast(int, ctypes.cast(bg.ctypes.next_double, ctypes.c_void_p).value)
        st_a, ax, kx = int(bg.ctypes.state_address), _ax(), _ax(0, False)
        out_r, z0_r, z1_r, ch_r = ls.born_outcomes(
            eng.measure_split(ax, True, None), 1e-10, np.random.default_rng(123), None)
        lm, d3 = np.zeros(3, np.float64), rng_r.random(3)
        rc, bits, aux = eng.measure_step(ax, True, None, fn, st_a, lm)
        assert (rc == 0 and aux is None and bg.state == rng_r.bit_generator.state
                and bits.tolist() == [False, True, bool(d3[2] < 0.5)] == out_r.tolist()
                and np.array_equal(lm.view(np.uint64),
                                   np.log(ch_r / (z0_r + z1_r)).view(np.uint64)))
        R(RuntimeError, "no pending measure_split", lambda: eng.measure_finish([1] * 3, [0.5] * 3))
        bm: list[Any] = [eng._h(), 3, 1, me.addr(ax[0]), me.addr(ax[1]), 0, 0, 1, 0,
                         me.addr(kx[0]), me.addr(kx[1]), 0, fn, st_a, *eng._ms_a,
                         eng._err_a, eng._errd_a, eng._info_a]
        assert (all(eng._lib.mbe_measure_step(*bm[:i], v, *bm[i + 1 :]) == -2 for i, v in
                    [(2, 2), (1, 4)] + [(j, 0) for j in (0, 3, 4, 9, 10, *range(12, 23))])
                and zeng.measure_step(ax, False, None, fn, st_a)[0] == 0)
        with eng._section():
            R(RuntimeError, "re-entrant", lambda: eng.measure_step(ax, False, None, fn, st_a))
        eng._ms_cls[:], eng._ms_dlog[:] = [me.CLS_DEAD, me.CLS_ACTIVE, me.CLS_TRIVIAL], [0, -0.5, 0]
        def _ms_oom(ret: int, i1: int) -> None:
            monkeypatch.setattr(eng._lib, "mbe_measure_step", lambda *_: (
                eng._info.__setitem__(0, 1), eng._info.__setitem__(1, i1), ret)[2])
        _ms_oom(4, 1)
        lm2 = np.zeros(3, np.float64)
        assert (eng.measure_step(ax, False, None, fn, st_a, lm2)[0] == 4
                and lm2[0] == -np.inf and lm2[1] == -0.5)
        [(_ms_oom(-3, i1), R(MemoryError, p, lambda: eng.measure_step(ax, False, None, fn, st_a)))
         for i1, p in ((0, "no shot was measured"), (1, "no shot was changed"), (2, "committed"))]
        eng.import_shot(2, ls.expansion_to_arrays(_t_st(4, 0)))
        rt, xm, zm = ReferenceCHPTableau(4), np.zeros((4, 3), bool), np.zeros((4, 3), bool)
        zm[0, 2] = True
        b2, apn = eng.export_shot(2).ell.copy(), eng.apply_pauli_noise_batch
        args = (rt.x2x, rt.x2z, rt.x_kap, rt.z2x, rt.z2z, rt.z_kap)
        assert (eng.measure_split(ax, False, None).n_active == 1 and apn(*args, xm, zm) is None
                and not np.array_equal(eng.export_shot(2).ell, b2)
                and eng._noise_bufs is not None)
        R(RuntimeError, "no pending measure_split", lambda: eng.measure_finish([False], [0.5]))
        bn: list[Any] = [eng._h(), 4, 3, 1, *eng._noise_bufs[1], eng._err_a]
        assert all(eng._lib.mbe_apply_pauli_noise_batch(*bn[:i], v, *bn[i + 1 :]) == -2
                   for i, v in [(1, 5), (2, 4), (3, 2)] + [(j, 0) for j in (0, *range(4, 13))])
        zeng.apply_pauli_noise_batch(*args, np.zeros((4, 0), bool), np.zeros((4, 0), bool))
        bk, bx, xm1 = rt.x_kap.copy(), rt.x2x.copy(), np.zeros((4, 3), bool)
        bk[0], bx[0, 0], xm1[0, 0], s2 = 1, bx[0, 0] | np.uint64(1 << 4), True, eng.export_shot(2)
        s2.weights[0] = float("nan")
        R(ValueError, "Hermitian", lambda: apn(rt.x2x, rt.x2z, bk, *args[3:], xm1, zm))
        R(ValueError, "padding", lambda: apn(bx, *args[1:], xm1, zm))
        eng.import_shot(2, s2)
        R(FloatingPointError, "Non-finite weights", lambda: apn(*args, np.ones((4, 3), bool), zm))
        eng.import_shot(2, ls.expansion_to_arrays(_t_st(4, 0)))
        apn(*args, xm.tolist(), zm.tolist())
        with eng._section():
            R(RuntimeError, "re-entrant", lambda: apn(*args, xm, zm))
        monkeypatch.setattr(eng._lib, "mbe_apply_pauli_noise_batch", lambda *_: -3)
        R(MemoryError, "out of memory", lambda: apn(*args, xm, zm))
