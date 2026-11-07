"""Unit tests for MagicsideSampler and CompiledMagicsideSampler."""

from __future__ import annotations

import collections

import pytest
import sinter
import stim

from stimside.dem_generators.dem_generator_marginal import MarginalLeakageDemGenerator
from stimside.dem_generators.leakage_decoder import BaseDecoder, CallableDecoder, MarginalDecoder
from stimside.op_handlers.leakage_handlers.leakage_uint8_magic import LeakageUint8Magic
from stimside.sampler_magic import CompiledMagicsideSampler, MagicsideSampler
from stimside._magic.query import QueryBudget, QueryBudgetExceededError


def test_compiled_sampler_counts_and_custom_counts_are_counter() -> None:
    circ = stim.Circuit(
        "H 0\nI[T] 0\nI[T] 0\nI[T] 0\nI[T] 0\nH 0\n"
        "M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    dem = stim.DetectorErrorModel("error(0.01) D0 L0")
    task = sinter.Task(circuit=circ, detector_error_model=dem, decoder="ms")
    sampler = MagicsideSampler(dem_decoder=BaseDecoder(decoder="pymatching"), batch_size=16, seed=101)
    compiled = sampler.compiled_sampler_for_task(task)
    assert isinstance(compiled, CompiledMagicsideSampler)
    stats = compiled.sample(40)
    assert stats.shots == 48 and stats.errors == 0 and stats.discards == 0
    assert isinstance(stats.custom_counts, collections.Counter)
    assert stats.custom_counts == collections.Counter()


def test_sampler_reference_sample_override_and_callable_decoder() -> None:
    circ = stim.Circuit("X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    dem = stim.DetectorErrorModel("error(0.01) D0 L0")
    dec = CallableDecoder(lambda _c, dets, records=None: dets[:, :1].copy())
    sampler = MagicsideSampler(dem_decoder=dec, batch_size=8, seed=7, m2d_reference_mode="zero", reference_sample=[1])
    st = sampler.compiled_sampler_for_task(sinter.Task(circuit=circ, detector_error_model=dem)).sample(8)
    assert st.shots == 8 and st.errors == 0


def test_leakage_sampler_marginal_decoder_and_multiprocess_sinter_collect() -> None:
    circ = stim.Circuit(
        "R 0 1\n"
        "H 0\nI[T] 0\nI[T] 0\nI[T] 0\nI[T] 0\nH 0\n"
        "I_ERROR[LEAKAGE_TRANSITION_1: (0.15, U-->2)] 1\n"
        "CZ[CONDITIONED_ON_PAIR: (U, U)] 0 1\n"
        "CZ[CONDITIONED_ON_PAIR: (U, U)] 0 1\n"
        "M[LEAKAGE_PROJECTION_Z: (0.0, 0) (1.0, 1) (0.5, 2)] 1\n"
        "MPAD[LEAKAGE_MEASUREMENT: (1.0, 2) : 1] 0\n"
        "I_ERROR[LEAKAGE_TRANSITION_1: (1.0, 2-->0)] 1\n"
        "M 0\n"
        "DETECTOR rec[-3]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]"
    )
    dem = stim.DetectorErrorModel("error(0.05) D0\nlogical_observable L0")
    marg_dec = MarginalDecoder(
        MarginalLeakageDemGenerator(unconditional_condition_on_U=False, decompose_errors=True, loss_oracle=True)
    )
    sampler = MagicsideSampler(
        op_handler=LeakageUint8Magic(unconditional_condition_on_U=False),
        dem_decoder=marg_dec,
        batch_size=32,
        seed=202,
    )
    results = sinter.collect(
        num_workers=2,
        max_shots=128,
        tasks=[sinter.Task(circuit=circ, detector_error_model=dem, decoder="ms_leak")],
        custom_decoders={"ms_leak": sampler},
    )
    assert len(results) == 1
    st = results[0]
    assert st.shots >= 128 and st.discards == 0
    assert isinstance(st.custom_counts, collections.Counter)
    assert st.custom_counts.get("leakage_count", 0) > 0


def test_sampler_query_budget_and_invalid_arguments() -> None:
    circ = stim.Circuit("H 0 1 2\nI[CCZ] 0 1 2\nH 0 1 2\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    task = sinter.Task(circuit=circ, detector_error_model=dem)
    tight = QueryBudget(max_terms=1, on_budget_exceeded="raise")
    cs_raise = MagicsideSampler(backend="QUERY", query_budget=tight, seed=3).compiled_sampler_for_task(task)
    with pytest.raises(QueryBudgetExceededError):
        cs_raise.sample(4)

    cs_fb = MagicsideSampler(backend="QUERY", query_budget=tight, on_budget_exceeded="fallback", batch_size=4, seed=3).compiled_sampler_for_task(task)
    assert cs_fb.sample(4).shots == 4

    with pytest.raises(ValueError):
        MagicsideSampler(batch_size=0)
    with pytest.raises(ValueError):
        MagicsideSampler(fault_mode="NOPE")
    with pytest.raises(ValueError):
        MagicsideSampler(backend="NOPE")
    with pytest.raises(ValueError):
        MagicsideSampler(m2d_reference_mode="NOPE")
    with pytest.raises(TypeError):
        MagicsideSampler(dem_decoder=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires a circuit"):
        MagicsideSampler().compiled_sampler_for_task(sinter.Task(circuit_path="dummy.stim", detector_error_model=dem))


def test_sampler_auto_magic_reference_prevents_spurious_logical_flips() -> None:
    circ_det = stim.Circuit(
        "H 0\nI[T] 0\nI[T] 0\nI[T] 0\nI[T] 0\nH 0\n"
        "M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    dem_indep = stim.DetectorErrorModel("error(0.01) D0\nlogical_observable L0")
    st0 = MagicsideSampler(
        dem_decoder=BaseDecoder(decoder="pymatching"),
        batch_size=16,
        seed=301,
        m2d_reference_mode="auto",
    ).compiled_sampler_for_task(sinter.Task(circuit=circ_det, detector_error_model=dem_indep)).sample(32)
    assert st0.shots == 32 and st0.errors == 0

    circ_rep = stim.Circuit(
        "H 0 1 2\nREPEAT 4 {\nI[T] 0 1 2\n}\nH 0 1 2\n"
        "X_ERROR(0.001) 0 1 2\nM 0 1 2\n"
        "DETECTOR rec[-3] rec[-2]\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-3]"
    )
    st1 = MagicsideSampler(
        dem_decoder=BaseDecoder(decoder="pymatching"),
        batch_size=32,
        seed=303,
        m2d_reference_mode="auto",
    ).compiled_sampler_for_task(sinter.Task(circuit=circ_rep)).sample(64)
    assert st1.shots == 64 and st1.errors <= 1


def test_sampler_positional_init_max_components_and_mode_rejection() -> None:
    oh = LeakageUint8Magic()
    dec = BaseDecoder(decoder="pymatching")
    pos_sampler = MagicsideSampler(oh, dec, 64, 99)
    assert pos_sampler.op_handler is oh and pos_sampler.dem_decoder is dec
    assert pos_sampler.batch_size == 64 and pos_sampler.seed == 99

    for bad_mode in ({"backend": "QUERY"}, {"fault_mode": "SELECTOR"}):
        with pytest.raises(ValueError):
            MagicsideSampler(op_handler=oh, **bad_mode)  # type: ignore[arg-type]

    circ = stim.Circuit("H 0 1 2\nI[CCZ] 0 1 2\nH 0 1 2\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    task = sinter.Task(circuit=circ, detector_error_model=stim.DetectorErrorModel("error(0.1) D0 L0"))
    for fm in ("DIRECT", "SELECTOR"):
        for be in ("STREAM", "QUERY"):
            with pytest.raises(QueryBudgetExceededError):
                MagicsideSampler(
                    fault_mode=fm, backend=be, max_components=1, on_budget_exceeded="error", seed=5
                ).compiled_sampler_for_task(task).sample(4)
    st_fb = MagicsideSampler(
        backend="QUERY", max_components=1, on_budget_exceeded="fallback_stream", batch_size=4, seed=5
    ).compiled_sampler_for_task(task).sample(4)
    assert st_fb.shots == 4 and st_fb.discards == 0
    for bad_kw in ({"max_components": 0}, {"max_components": True}, {"on_budget_exceeded": "ignore"}):
        with pytest.raises(ValueError):
            MagicsideSampler(**bad_kw)  # type: ignore[arg-type]


def test_pickled_worker_index_compiled_seeds() -> None:
    import pickle

    sampler = MagicsideSampler(seed=777)
    w0 = pickle.loads(pickle.dumps(sampler))
    w1 = pickle.loads(pickle.dumps(sampler))
    assert w0._worker_index == 0 and w1._worker_index == 1
    assert w0._next_compiled_seed() != w1._next_compiled_seed()
    assert pickle.loads(pickle.dumps(MagicsideSampler(seed=777)))._next_compiled_seed() is not None


def test_compiled_magicside_sampler_compiled_dem_decoder_kwarg() -> None:
    from stimside.sampler_magic import CompiledMagicsideSampler

    circ = stim.Circuit("R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    task = sinter.Task(circuit=circ)
    cdec = BaseDecoder(decoder="pymatching").compile_for_task(task)
    cs_new = CompiledMagicsideSampler(
        circuit=circ, compiled_op_handler=None, compiled_dem_decoder=cdec, needs_records=False, batch_size=8, seed=11
    )
    cs_legacy = CompiledMagicsideSampler(
        circuit=circ, compiled_op_handler=None, compiled_decoder=cdec, needs_records=False, batch_size=8, seed=11
    )
    s_new, s_legacy = cs_new.sample(8), cs_legacy.sample(8)
    assert (s_new.shots, s_new.errors, s_new.discards, s_new.custom_counts) == (
        s_legacy.shots,
        s_legacy.errors,
        s_legacy.discards,
        s_legacy.custom_counts,
    )
    with pytest.raises(TypeError, match="compiled_dem_decoder"):
        CompiledMagicsideSampler(circuit=circ, compiled_op_handler=None, needs_records=False)
    with pytest.raises(TypeError, match="compiled_dem_decoder"):
        CompiledMagicsideSampler(
            circuit=circ, compiled_op_handler=None, compiled_dem_decoder=cdec, compiled_decoder=cdec, needs_records=False
        )


@pytest.mark.parametrize("task_decoder", ["my_magic", None])
def test_sampler_without_dem_decoder_warns_on_pymatching_fallback(task_decoder: str | None) -> None:
    import warnings

    circ = stim.Circuit("R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    sampler = MagicsideSampler(seed=1)
    assert sampler.op_handler is None
    with pytest.warns(UserWarning, match=f"task.decoder={task_decoder!r}.*'pymatching'"):
        sampler.compiled_sampler_for_task(sinter.Task(circuit=circ, decoder=task_decoder))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sampler.compiled_sampler_for_task(sinter.Task(circuit=circ, decoder="pymatching"))
        MagicsideSampler(dem_decoder="pymatching", seed=1).compiled_sampler_for_task(
            sinter.Task(circuit=circ, decoder=task_decoder)
        )


def test_pickled_copies_of_every_sampler_get_distinct_worker_seeds() -> None:
    import pickle

    from stimside import CosetsideSampler, FlipsideSampler, TablesideSampler
    from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
    from stimside.op_handlers.leakage_handlers.leakage_uint8_flip import LeakageUint8Flip
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import LeakageUint8Tableau

    for sampler in (
        MagicsideSampler(seed=777),
        FlipsideSampler(LeakageUint8Flip(), BaseDecoder(), seed=777),
        TablesideSampler(LeakageUint8Tableau(), BaseDecoder(), seed=777),
        CosetsideSampler(LeakageUint8Coset(), BaseDecoder(), seed=777),
    ):
        w0, w1 = pickle.loads(pickle.dumps(sampler)), pickle.loads(pickle.dumps(sampler))
        s0, s1 = w0._next_compiled_seed(), w1._next_compiled_seed()
        assert s0 == 777 and s1 is not None and s1 != s0, type(sampler).__name__
