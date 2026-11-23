"""Tests for CosetsideSimulator latch checks, constructor snapshotting, and run-op caching."""
import sys

import pytest

from stimside import CosetsideSimulator
from stimside import simulator_coset as sc
from stimside._coset.reference_compile import CompiledReferenceCircuit
from stimside.simulator_coset_cmeas_test import _C, _mk

def test_metaclass_and_single_base_latch_errors():
    circ, refuse = _C("H 0\nM 0"), [True]
    ms = type("VM", (type,), {"__new__": lambda m, n, b, ns: (
        (_ for _ in ()).throw(ValueError("meta-refused")) if refuse[0] and n.startswith("_Aborted")
        else type.__new__(m, n, b, ns))})("MetaSub", (CosetsideSimulator,), {})
    sb = type("SingleBaseErr", (type("S", (), {"__slots__": ()}), CosetsideSimulator), {
        "__init_subclass__": classmethod(lambda cls: (_ for _ in ()).throw(RuntimeError("hook")))})
    with pytest.raises(ValueError, match="meta-refused"):
        _mk(circ, 2, cls=ms)
    refuse[0] = False
    with pytest.raises(TypeError, match=r"SingleBaseErr.*cannot be latched"):
        _mk(circ, 2, cls=sb)
    assert ms not in sc._SUBCLASS_VERDICTS and sb not in sc._SUBCLASS_VERDICTS and (
        type(_mk(circ, 2, cls=ms)) is ms)
@pytest.mark.parametrize("mode", ["noop", "ignore_failed", "ignore_restore"])
def test_bad_class_setters_rejected_and_cached(mode):
    circ, calls, cs, n0 = _C("H 0\nM 0"), [], object.__dict__["__class__"].__set__, 0
    sub = type("CM", (type,), {"__new__": lambda m, n, b, ns: (calls.append(n),
        type.__new__(m, n, b, ns))[1]})(f"Bad_{mode}", (CosetsideSimulator,), {"__class__":
        property(lambda s: sub if mode == "noop" else type(s), lambda s, v: None if (mode == "noop"
        or (mode == "ignore_failed" and v.__name__.startswith("_Failed")) or (mode ==
        "ignore_restore" and v is sub)) else cs(s, v))})
    for _ in (0, 1):
        with pytest.raises(TypeError, match=rf"Bad_{mode}.*cannot be latched"):
            _mk(circ, 2, cls=sub)
        n0 = n0 or len(calls)
    assert sc._SUBCLASS_VERDICTS.get(sub) and n0 >= 2 and len(calls) == n0
def test_init_mutation_interactive_reuse_and_run_ops_caching():
    x0, x1, mut = _C("X_ERROR(0) 0\nM 0"), _C("X_ERROR(1) 0\nM 0"), [False]
    circ, s2 = x0.copy(), _mk(c2 := x0.copy(), 8, compiled_ref=CompiledReferenceCircuit(c2))
    sys.setprofile(lambda fr, ev, _a: (mut.__setitem__(0, True), circ.clear(), circ.__iadd__(x1),
        None)[-1] if not mut[0] and ev == "call" and fr.f_code.co_name == "_init_runtime_buffers"
        else None)
    try:
        sim = _mk(circ, 8)
    finally:
        sys.setprofile(None)
    s3 = _mk(c3 := _C("QUBIT_COORDS(0, 0) 0\nSHIFT_COORDS(1, 0)\nTICK\nH 0\nM 0"))
    s4 = _mk(c4 := _C("H 0\nM 0"), compiled_ref=CompiledReferenceCircuit(c4))
    assert mut[0] and sim._run_ops_circuit_snap == x0 and sim.run() is None and (
        r0 := sim.get_final_measurement_records()).size == 8 and bool(r0.all()) and (
        s2.interactive_do(c2), s2.finish_interactive_run(), s2.clear(), s2.run()) == (None,) * 4
    assert not bool(s2.get_final_measurement_records().any()) and (s2.circuit.clear(),
        s2.circuit.__iadd__(x1), s2.clear(), s2.run())[2:] == (None, None) and (
        r2 := s2.get_final_measurement_records()).size == 8 and bool(r2.all()) and (
        s3.interactive_do(c3), s3.finish_interactive_run(), s3._unrolled_exact_steps)[2] is True
    assert s4._run_ops is None and s4._run_ops_circuit_snap is None and s4.run() is None and (
        o0 := s4._run_ops) is not None and s4._run_ops_circuit_snap == c4 and (
        s4._run_ops_circuit_snap is not s4.circuit) and (s4.clear(), s4.run(), s4._run_ops)[2] is o0
