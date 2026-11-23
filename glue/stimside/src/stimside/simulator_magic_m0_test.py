"""Tests for Magicside IR validation, fail-closed lowering, copy behavior, and realizations."""
from __future__ import annotations

import copy
import re
import types
import weakref
from typing import Any

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

from stimside import LeakageUint8Magic, MagicsideSimulator, simulator_magic
from stimside._magic import gates, plan
from stimside.util import circuit, circuit_ir, reference_chp

C, MS, sm, mg, mp = stim.Circuit, MagicsideSimulator, simulator_magic, gates, plan
uc, cir, rchp = circuit, circuit_ir, reference_chp

def R(exc: type[BaseException], pat: str, fn: Any) -> Any:
    with pytest.raises(exc, match=pat):
        fn()

def _br(sim: MagicsideSimulator, w: bool = False) -> Any:
    return types.SimpleNamespace(owner=sim, cb=sim.clear, nested=[sim],
                                 weak_owner=weakref.ref(sim) if w else None,
                                 handle_op=lambda o, s: False)
@pytest.mark.parametrize("reals", [[{"measurements": {0: 1}}, {}], [{}, {"measurements": {0: 1}}]])
def test_mixed_realizations_and_public_handler_copy(
    monkeypatch: pytest.MonkeyPatch, reals: list[dict[str, Any]]
) -> None:
    c = C("R 0\nH 0\nM 0")
    [R(ValueError, "Non-fault", lambda k=kw: MS(c, batch_size=2, fault_realizations=reals, **k))
     for kw in ({"sync_tableside_rng": True}, {"op_handler": LeakageUint8Magic()})]
    ch = mg.CliffordFaultChannel(step_index=1, branches=(mg.FaultBranch(0.5, "X", (0,)),))
    sim = MS(c, batch_size=2, seed=5, fault_channels=(ch,),
             fault_realizations=[{"faults": {0: 1}}, {}], sync_tableside_rng=True)
    sim.run()
    assert sim.get_final_measurement_records().shape == (2, 1)
    for eng in (False, True):
        monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", eng)
        s = MS(C("R 0\nH 0\nI[T] 0\nM 0"), batch_size=2, seed=7)
        s.compiled_op_handler = pub = _br(s)
        tw = copy.copy(s)
        cp: Any = tw.compiled_op_handler
        assert (cp not in (pub, tw._compiled_op_handler) and cp.owner is tw
                and cp.cb.__self__ is tw and cp.nested[0] is tw and pub.owner is s)
def test_ir_validation_fail_closed_extract_and_docstring() -> None:
    cb = C("H 0\nM 0")
    irb, bad = cir.build_ir(cb), (C("H 0 1\nM 0"), C("H 0\nM 0 0"), C("H 0\nX 0\nM 0"))
    fns: tuple[Any, ...] = (mg.extract_circuit_faults, mp._compile_circuit_pass,
                            mp.compile_coordinates, mp.compile_direct, mp.compile_selector)
    for f in fns:
        R(TypeError, "Expected CircuitIR or None", lambda g=f: g(cb, ir="x"))
        [R(ValueError, "CircuitIR does not match", lambda g=f, x=o: g(x, ir=irb)) for o in bad]
    rep, re2 = C("R 0\nREPEAT 3 {\n H 0\n}\nM 0"), C("REPEAT 2 {\n}\nM 0")
    for c, ok in ((C(), cir.build_ir(C())), (cb, irb), (rep, cir.build_ir(rep)),
                  (rep, cir.build_ir(uc.unroll_circuit(rep))), (re2, cir.build_ir(re2))):
        ops, flts = mg.extract_circuit_faults(c, ir=ok)
        assert (len(ops) == len(ok.instrs) and not flts
                and mp.compile_direct(c, ir=ok).num_measurements
                == mp.compile_selector(c, ir=ok).num_measurements == c.num_measurements)
    st = mp._PassState(2, False, False, rchp.ReferenceCHPTableau(2), 2)
    for fn_l, cat, nm, args in ((mp._lower_unitary_op, cir.Category.UNITARY, "UNKNOWN_U", []),
                                (mp._lower_noise_op, cir.Category.NOISE, "UNKNOWN_N", [0.1])):
        op: Any = types.SimpleNamespace(name=nm, tag="", targets_copy=lambda: [stim.target_x(0)],
                                        gate_args_copy=lambda a=args: list(a))
        R(ValueError, "Unsupported Stim", lambda g=fn_l, o=op, ct=cat: g(
            st, cir.IRInstr(0, o, "", ct, cir.UntaggedFilter.NONE, 0, 0, None, False, None)))
    v = C("R 0 1\nSPP X0*Z1\nSPP_DAG X0*Z1\nCORRELATED_ERROR(0.1) X0*Z1\n"
          "ELSE_CORRELATED_ERROR(0.2) Y0\nM 0 1")
    sg = mp.compile_selector(v, gadgetize_pauli_noise=True)
    assert (mp.compile_direct(v).num_measurements == sg.num_measurements == 2
            == mp.compile_selector(v, gadgetize_pauli_noise=False).num_measurements
            and len(sg.noise_selector_groups) == 2)
    cr = C("R 0 1\nREPEAT 2 {\n I[CLIFFORD_FAULT:(0.2,X)] 0\n"
           " II[CLIFFORD_FAULT:(0.1,CZ)] 0 1\n}\nM 0 1")
    for c in (C(), C("H 0\nI[CLIFFORD_FAULT:(0.1,S)] 0\nM 0"), cr):
        oa, fa = mg.extract_circuit_faults(c)
        ob, fb = mg.extract_circuit_faults(c, ir=cir.build_ir(c))
        assert ([str(o) for o in oa] == [str(o) for o in ob] and fa == fb
                and all(oa[ch.step_index].name == "I" and not oa[ch.step_index].tag for ch in fa))
    doc, sim = MS.__doc__ or "", MS(C("R 0\nM 0"), batch_size=1, seed=1)
    sim._compiled_op_handler = sim.compiled_op_handler = _br(sim, True)
    cp: Any = copy.copy(sim).compiled_op_handler
    assert (all(p in doc for p in ("direct attributes", "bound methods", "nested containers",
                                   "weakref", "closure", "not remapped"))
            and not any(re.search(p, doc) for p in (r"\bR\d+-", r"\bPKG-\d+", r"\bD-\d+",
                                                    r"\bWP-\d+", r"\bRun \d+"))
            and cp.weak_owner is not None and cp.owner is not sim and cp.weak_owner() is sim)

@pytest.mark.parametrize("cfg", [
    {"backend": "STREAM", "fault_mode": "SELECTOR"}, {"backend": "QUERY", "fault_mode": "DIRECT"},
    {"backend": "QUERY", "fault_mode": "SELECTOR"},
    {"backend": "STREAM", "fault_mode": "DIRECT", "fault_realizations": {"faults": {0: 0}}},
])
def test_copy_preserves_cached_ir(monkeypatch: pytest.MonkeyPatch, cfg: dict[str, Any]) -> None:
    circ = C("R 0 1\nH 0\nI[CLIFFORD_FAULT:(0.2,S)] 0\nCX 0 1\nI[T] 0\nM 0 1")
    calls, orig = [0], cir.build_ir
    for m in (cir, mg, mp, sm):
        monkeypatch.setattr(m, "build_ir",
                            lambda *a, **k: (calls.__setitem__(0, calls[0] + 1), orig(*a, **k))[1])
    sim = MS(circ, batch_size=3, seed=13, **cfg)
    assert calls[0] == 1
    calls[0], tw = 0, copy.copy(sim)
    tw.run(), sim.run()
    r1, r2 = sim.get_final_measurement_records(), tw.get_final_measurement_records()
    assert calls[0] == 0 and tw._cached_ir() is not None and np.array_equal(r1, r2)
    mut, sn = copy.copy(MS(circ.copy(), batch_size=2, seed=13, **cfg)), MS(C("R 0\nM 0"), seed=1)
    mut.circuit = C("R 0\nI[CLIFFORD_FAULT:(0.2,S)] 0\nX 0\nM 0")
    sn.circuit = None  # type: ignore[assignment]
    assert mut._cached_ir() is None and sn._cached_ir() is None and mut.run() is None
    assert np.all(mut.get_final_measurement_records())
