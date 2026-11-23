"""Tests for opt-in ``fast_leakage=True`` mode across Tableau, Coset, and Magic handlers."""

from __future__ import annotations

import math
import warnings
from functools import partial
from typing import Any

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

from stimside import (
    CosetsideSimulator,
    LeakageUint8Coset,
    LeakageUint8Magic,
    LeakageUint8Tableau,
    MagicsideSimulator,
    TablesideSimulator,
)

_eq = np.testing.assert_array_equal
_LT1 = "I[LEAKAGE_TRANSITION_1:"
_L0 = f"{_LT1} (1.0, U-->2)] 0"
_MP = "M[LEAKAGE_PROJECTION_Z: (1.0, 2)]"
_CM = ((CosetsideSimulator, LeakageUint8Coset), (MagicsideSimulator, LeakageUint8Magic))
_TCM = ((TablesideSimulator, LeakageUint8Tableau), *_CM)


def _c(*lines: str) -> stim.Circuit:
    return stim.Circuit("\n".join(lines))


def _assert_rates_within_sigma(a: np.ndarray, b: np.ndarray, *, max_sigma: float = 5.0) -> None:
    aa = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    bb = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    pa, pb = aa.mean(axis=0), bb.mean(axis=0)
    p = (aa.sum(axis=0) + bb.sum(axis=0)) / (len(aa) + len(bb))
    se = np.sqrt(p * (1.0 - p) * (1.0 / len(aa) + 1.0 / len(bb)))
    ok = np.where(se == 0.0, pa == pb, np.abs(pa - pb) <= max_sigma * se)
    assert np.all(ok), f"rate mismatch (> {max_sigma} SE): pa={pa} pb={pb} se={se}"


def _u2l_norm(rec: Any) -> list[list[int]]:
    return [rec.tolist()] if isinstance(rec, np.ndarray) else [np.asarray(x).tolist() for x in rec]


def _assert_sims_equal(a: Any, b: Any) -> None:
    for fn in ("get_final_measurement_records", "get_detector_flips", "get_observable_flips"):
        _eq(getattr(a, fn)(), getattr(b, fn)())
    assert _u2l_norm(a.get_unleaked_to_leaked_records()) == _u2l_norm(
        b.get_unleaked_to_leaked_records()
    )


def _make_sim(sim_cls: Any, handler: Any, circ: stim.Circuit, bs: int, seed: int, **kw: Any) -> Any:
    compiled = handler.compile_op_handler(circuit=circ, batch_size=bs)
    return sim_cls(circ, compiled_op_handler=compiled, batch_size=bs, seed=seed, **kw)


def _compile_warnings(handler: Any, circ: stim.Circuit, bs: int = 4) -> list[Any]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        handler.compile_op_handler(circuit=circ, batch_size=bs)
    return caught


def _sample(
    sim_cls: Any, h_fn: Any, circ: stim.Circuit, *, fl: bool, shots: int, bs: int, seed: int,
    det_obs: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    sim = _make_sim(sim_cls, h_fn(fast_leakage=fl), circ, bs, seed)
    pa, pb = [], []
    for idx in range(math.ceil(shots / bs)):
        if idx:
            sim.clear()
        sim.run()
        a = sim.get_detector_flips() if det_obs else sim.get_final_measurement_records()
        b = sim.get_observable_flips() if det_obs else np.zeros((bs, 0), dtype=np.bool_)
        pa.append(np.asarray(a, dtype=np.bool_).reshape(bs, -1))
        pb.append(np.asarray(b, dtype=np.bool_).reshape(bs, -1))
    return np.vstack(pa)[:shots], np.vstack(pb)[:shots]


def _assert_stat_equiv(
    sim_cls: Any, h_fn: Any, circ: stim.Circuit, shots: int, bs: int, s0: int, s1: int,
    det_obs: bool = False,
) -> None:
    a0, b0 = _sample(sim_cls, h_fn, circ, fl=False, shots=shots, bs=bs, seed=s0, det_obs=det_obs)
    a1, b1 = _sample(sim_cls, h_fn, circ, fl=True, shots=shots, bs=bs, seed=s1, det_obs=det_obs)
    _assert_rates_within_sigma(a0, a1, max_sigma=5.0)
    if det_obs:
        _assert_rates_within_sigma(b0, b1, max_sigma=5.0)


def _build_leaky_sc(
    basis: str = "z", *, rounds: int = 3, p_leak: float = 0.03, with_t: bool = False
) -> stim.Circuit:
    base = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}", distance=3, rounds=rounds,
        after_clifford_depolarization=0.005, before_measure_flip_probability=0.005,
    ).flattened()
    meas = {t.qubit_value for op in base if op.name in ("MR", "MRX", "MRY")
            for t in op.targets_copy() if t.qubit_value is not None}
    in2q = {t.qubit_value for op in base if op.name in ("CX", "CZ")
            for t in op.targets_copy() if t.qubit_value is not None}
    data_qs, out, added_t = sorted(in2q - meas), stim.Circuit(), False
    for op in base:
        if op.name in ("M", "MR", "MX", "MRX") and op.tag == "":
            out.append(op.name, op.targets_copy(), [], tag="LEAKAGE_PROJECTION_Z: (1.0, 2)")
        else:
            out.append(op)
        if op.name in ("CX", "CZ"):
            out.append("II", op.targets_copy(), tag=f"LEAKAGE_TRANSITION_2: ({p_leak}, U_U-->2_D)")
        elif op.name == "TICK":
            if with_t and not added_t:
                out.append("I", data_qs[:2], tag="T")
                added_t = True
            out.append("I", data_qs, tag="LEAKAGE_TRANSITION_1: (0.15, 2-->U)")
    return out


def test_default_fast_leakage_is_false_and_bit_exact() -> None:
    """Default ``fast_leakage`` is False and matches explicit ``fast_leakage=False`` bit-exact."""
    assert all(h_cls().fast_leakage is False for _, h_cls in _TCM)
    circ = _build_leaky_sc("z", rounds=2, p_leak=0.04)
    for (s_cls, h_cls), bs in zip(_TCM, (1, 32, 16), strict=True):
        s0 = _make_sim(s_cls, h_cls(), circ, bs, 77, record_unleaked_to_leaked=True)
        s1 = _make_sim(
            s_cls, h_cls(fast_leakage=False), circ, bs, 77, record_unleaked_to_leaked=True
        )
        s0.run()
        s1.run()
        _assert_sims_equal(s0, s1)


@pytest.mark.parametrize("basis", ["z", "x"])
@pytest.mark.parametrize("s_cls,h_cls,rounds,with_t,shots,bs,s0,s1", [
    (CosetsideSimulator, LeakageUint8Coset, 3, False, 8192, 256, 101, 202),
    (MagicsideSimulator, LeakageUint8Magic, 2, True, 2048, 128, 11, 29),
])
def test_leaky_sc_z_and_x_memory_stat_equiv(
    basis: str, s_cls: Any, h_cls: Any, rounds: int, with_t: bool,
    shots: int, bs: int, s0: int, s1: int,
) -> None:
    """Cosetside and Magicside (+I[T]) Z/tagged-MX memory agree within 5 SE."""
    circ = _build_leaky_sc(basis, rounds=rounds, p_leak=0.03, with_t=with_t)
    assert _compile_warnings(h_cls(fast_leakage=True), circ, 64) == []
    _assert_stat_equiv(s_cls, h_cls, circ, shots, bs, s0, s1, det_obs=True)


@pytest.mark.parametrize("circ,shots", [
    (_c("R 0", "H 0", f"{_LT1} (0.02, 1-->2)] 0", "H 0", f"{_MP} 0"), 8192),
    (_c("R 0", "H 0", f"{_LT1} (0.2, 0-->2) (0.5, 1-->3)] 0",
        "M[LEAKAGE_PROJECTION_Z: (1.0, 2) (1.0, 3)] 0"), 8192),
    (_c("R 0", f"{_LT1} (0.4, U-->2)] 0", "H 0",
        f"{_LT1} (0.15, 0-->2) (0.35, 1-->3) (0.4, 2-->U)] 0",
        "M[LEAKAGE_PROJECTION_Z: (1.0, 2) (0.5, 3)] 0"), 8192),
    (_c("R 0", "H 0", f"{_LT1} (0.0, 1-->2)] 0", "H 0", f"{_MP} 0"), 4096),
    (_c("R 0", "H 0", f"{_LT1} (1.0, 1-->2)] 0", f"{_MP} 0"), 4096),
    (_c("R 0", "H 0", f"{_LT1} (0.1, U-->2) (0.2, 1-->3)] 0",
        "M[LEAKAGE_PROJECTION_Z: (1.0, 2) (1.0, 3)] 0"), 4096),
    (_c("R 0", "H 0", f"{_LT1} (0.2, 1-->2)] 0 0", "H 0", f"{_MP} 0"), 4096),
])
def test_lt1_01_conditioned_stat_equiv_coset_and_magic(circ: stim.Circuit, shots: int) -> None:
    """0/1-conditioned LEAKAGE_TRANSITION_1 agrees within 5 SE on Coset and Magic."""
    for s_cls, h_cls in _CM:
        _assert_stat_equiv(s_cls, h_cls, circ, shots, 256, 31, 47)


def test_certain_loss_2q_corr_tags_bell_return_and_disjoint_lt1() -> None:
    """Exact certain-loss, Bell fresh-look, return-to-U/D depol, and disjoint-target LT1."""
    cases = [
        _c("R 0 1", _L0, "X 0", "CX 0 1", f"{_MP} 1"),
        _c("R 0 1", "H 1", _L0, "X 0", "CZ 0 1", "H 1", f"{_MP} 1"),
        _c("R 0 1", _L0, "X 0", "SWAP 0 1", f"{_MP} 1"),
        _c("R 0 1", _L0, "E(1.0) X0 X1", "ELSE_CORRELATED_ERROR(1.0) X1", f"{_MP} 1"),
        _c("R 0 1 2", _L0, "REPEAT 2 {\n  X 0\n  SWAP[LEAKAGE_SWAP] 0 1\n}",
           "MPAD[LEAKAGE_MEASUREMENT: (1.0, 2) : 0] 0",
           "HERALDED_ERASE(0.0) 0 2", "HERALDED_ERASE(1.0) 0", "CX rec[-3] 2", f"{_MP} 0 1 2"),
    ]
    for circ in cases:
        for (s_cls, h_cls), bs in zip(_TCM, (1, 16, 16), strict=True):
            r0, _ = _sample(s_cls, h_cls, circ, fl=False, shots=bs, bs=bs, seed=5)
            r1, _ = _sample(s_cls, h_cls, circ, fl=True, shots=bs, bs=bs, seed=5)
            _eq(r0, r1)

    spp = _c("R 0 1", _L0, "X 0", "SPP X0*X1", f"{_MP} 1")
    t0, _ = _sample(TablesideSimulator, LeakageUint8Tableau, spp, fl=False, shots=4, bs=1, seed=9)
    t1, _ = _sample(TablesideSimulator, LeakageUint8Tableau, spp, fl=True, shots=4, bs=1, seed=9)
    _eq(t0, t1)
    assert not np.any(t1)

    bell = _c("R 0 1", "H 0", "CX 0 1", f"{_LT1} (1.0, 1-->2)] 0 1", f"{_MP} 0 1")
    disjoint = _c("R 0 1", f"{_LT1} (0.01, 1-->2)] 0", "R 0", "H 0",
                  f"{_LT1} (0.01, 1-->2)] 1", "MX[LEAKAGE_PROJECTION_Z: (1.0, 2)] 0")
    for s_cls, h_cls in _CM:
        rec, _ = _sample(s_cls, h_cls, bell, fl=True, shots=128, bs=128, seed=13)
        _eq(rec[:, 0], rec[:, 1])
        assert 0.25 < rec[:, 0].mean() < 0.75
        dj, _ = _sample(s_cls, h_cls, disjoint, fl=True, shots=128, bs=128, seed=1)
        assert not np.any(dj)

    for circ in (
        _c("R 0", _L0, f"{_LT1} (1.0, 2-->U)] 0", f"{_MP} 0"),
        _c("R 0 1", _L0, "II[LEAKAGE_TRANSITION_2: (1.0, 2_U-->D_U)] 0 1", f"{_MP} 0"),
    ):
        rec, _ = _sample(CosetsideSimulator, LeakageUint8Coset, circ, fl=True, shots=2048,
                         bs=256, seed=19)
        assert abs(float(rec[:, 0].mean()) - 0.5) <= 5.0 * math.sqrt(0.25 / 2048)


def test_qualifying_test_warnings_witnesses_and_uc_false() -> None:
    """Disqualifying circuits warn and match baseline; qualifying/uc=False are silent."""
    disq = [
        (_c("R 0", _L0, "X 0", "M 0"), "M 0", [0]),
        (_c("R 0", _L0, "H 0", "MX 0"), "MX 0", [0, 1]),
        (_c("R 0 1", _L0, "X 0", "MPP Z0*Z1"), "MPP", [0]),
        (_c("R 0 1", _L0, "H 0", "MXX 0 1"), "MXX", [0, 1]),
        (_c("R 0 1", _L0, "H 0", "MYY 0 1"), "MYY", [0, 1]),
        (_c("R 0 1", _L0, "X 0", "MZZ 0 1"), "MZZ", [0]),
        (_c("R 0", _L0, "X 0", "REPEAT 2 {\n  M 0\n}"), "M 0", [0]),
        (_c("R 0 1", _L0, "X 0", "CX[CONDITIONED_ON_PAIR: (2, U)] 0 1", f"{_MP} 1"), "CX", [0]),
    ]
    sim_cfgs: list[tuple[Any, Any, dict[str, Any]]] = [(s, h, {}) for s, h in _TCM] + [
        (CosetsideSimulator, LeakageUint8Coset, {"sync_tableside_rng": True}),
        (TablesideSimulator, LeakageUint8Tableau, {"sync_tableside_rng": True}),
    ]
    for circ, needle, expected in disq:
        for s_cls, h_cls, kw in sim_cfgs:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                sim = _make_sim(s_cls, h_cls(fast_leakage=True), circ, 8, 5, **kw)
                sim.run()
            assert any(needle in str(x.message) for x in w), f"missing warning for {needle}"
            rec = np.asarray(sim.get_final_measurement_records(), dtype=np.int64).reshape(8, -1)
            assert sorted(set(rec[:, 0].tolist())) == expected

    ccz = _c("R 0 1 2", _L0, "X 0", "I[CCZ] 0 1 2", f"{_MP} 1")
    w_ccz = _compile_warnings(LeakageUint8Magic(fast_leakage=True), ccz)
    assert any("CCZ" in str(x.message) for x in w_ccz)
    cf = _c("R 0 1", _L0, "X 0", "I[CLIFFORD_FAULT: (1.0, CX)] 0 1", f"{_MP} 1")
    with warnings.catch_warnings(record=True) as w_cf:
        warnings.simplefilter("always")
        r_on, _ = _sample(MagicsideSimulator, LeakageUint8Magic, cf, fl=True, shots=8, bs=8, seed=5)
    assert any("CLIFFORD_FAULT" in str(x.message) for x in w_cf) and not np.any(r_on)
    r_off, _ = _sample(MagicsideSimulator, LeakageUint8Magic, cf, fl=False, shots=8, bs=8, seed=5)
    _eq(r_on, r_off)

    for qc, uc in (
        (_c("R 0 1", _L0, "X 0", "CX[CONDITIONED_ON_PAIR: (U, U)] 0 1", f"{_MP} 1"), True),
        (_c("R 0 1", _L0, "X 0", "SWAP[LEAKAGE_SWAP] 0 1", f"{_MP} 0"), True),
        (_c("R 0", "REPEAT 2 {\n  X 0\n}", f"{_MP} 0"), True),
        (_c("R 0", _L0, "X 0", "M 0"), False),
    ):
        for _, h_cls in _TCM:
            h = h_cls(unconditional_condition_on_U=uc, fast_leakage=True)
            assert _compile_warnings(h, qc) == []

    uc_circ = _c("R 0", "H 0", f"{_LT1} (0.2, 1-->2)] 0", f"{_LT1} (1.0, 2-->U)] 0", "H 0", "M 0")
    h_uc = partial(LeakageUint8Coset, unconditional_condition_on_U=False)
    _assert_stat_equiv(CosetsideSimulator, h_uc, uc_circ, 4096, 256, 71, 73)


def test_late_instruction_raises_and_coset_atomic() -> None:
    """Late frozen-state readers raise ValueError; Cosetside stays atomic; safe ops succeed."""
    base = _c("R 0 1 2", _L0, "REPEAT 2 {\n  X 0\n}")
    late_ops = ("M 0", "MX 0", "MPP Z0", "MXX 0 1")
    for (s_cls, h_cls), bs in zip(_TCM, (1, 4, 4), strict=True):
        for late_op in late_ops:
            sim = _make_sim(s_cls, h_cls(fast_leakage=True), base, bs, 3)
            sim.interactive_do(base)
            with pytest.raises(ValueError, match="fast_leakage"):
                sim.interactive_do(stim.Circuit(late_op))
            sim.interactive_do(stim.Circuit("H 0"))
            sim.finish_interactive_run()
        h_uc = h_cls(unconditional_condition_on_U=False, fast_leakage=True)
        sim_uc = _make_sim(s_cls, h_uc, base, bs, 3)
        sim_uc.interactive_do(base)
        sim_uc.interactive_do(stim.Circuit("M 0"))
        sim_uc.finish_interactive_run()

    msim = _make_sim(MagicsideSimulator, LeakageUint8Magic(fast_leakage=True), base, 4, 3)
    msim.interactive_do(base)
    with pytest.raises(ValueError, match="fast_leakage"):
        msim.interactive_do(stim.Circuit("I[CCZ] 0 1 2"))
    msim.interactive_do(stim.Circuit("I[T] 1"))
    msim.finish_interactive_run()


def test_tableside_cpp_fast_path_and_three_way_sync_lockstep() -> None:
    """Tableside keeps C++ runner active; all three simulators stay in sync lockstep."""
    sc_z = _build_leaky_sc("z", rounds=2, p_leak=0.03)
    h = LeakageUint8Tableau(fast_leakage=True).compile_op_handler(circuit=sc_z, batch_size=1)
    assert h.supports_cpp_fast_path() is True
    sim = TablesideSimulator(sc_z, compiled_op_handler=h, batch_size=1, seed=17)
    runner = sim._cpp_runner
    assert runner is not None
    calls = 0

    class _Spy:
        def __getattr__(self, name: str) -> Any:
            return getattr(runner, name)

        def run(self, tss: Any) -> Any:
            nonlocal calls
            calls += 1
            return runner.run(tss)

    sim._cpp_runner = _Spy()  # type: ignore[assignment]
    for idx in range(3):
        if idx:
            sim.clear()
        sim.run()
    assert calls == 3

    sync_circs = [
        _build_leaky_sc("z", rounds=2, p_leak=0.05),
        _build_leaky_sc("x", rounds=2, p_leak=0.05),
        _c("R 0 1 2", "H 0", "CX 0 1", f"{_LT1} (0.25, 0-->2) (0.35, 1-->3)] 0 1",
           "X 0", "Y 1", "DEPOLARIZE1(0.1) 0 1",
           "II[LEAKAGE_TRANSITION_2: (0.3, 2_U-->D_U) (0.3, 3_U-->U_2)] 0 2",
           "SWAP[LEAKAGE_SWAP] 0 1", "MPAD[LEAKAGE_MEASUREMENT: (0.8, 2) (0.4, 3) : 0 1] 0 0",
           "HERALDED_ERASE(0.2) 0 1", "M[LEAKAGE_PROJECTION_Z: (1.0, 2) (0.5, 3)] 0 1 2"),
        _c("R 0 1", f"{_LT1} (0.5, U-->2)] 0", "X 0",
           "CX[CONDITIONED_ON_PAIR: (2, U)] 0 1", "M 0 1"),
    ]
    kw: dict[str, Any] = {"sync_tableside_rng": True, "record_unleaked_to_leaked": True}
    for circ in sync_circs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sims = [_make_sim(s, hc(fast_leakage=True), circ, 12, 42, **kw) for s, hc in _TCM]
            for s in sims:
                s.run()
        for other in sims[1:]:
            _assert_sims_equal(sims[0], other)
