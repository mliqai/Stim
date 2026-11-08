"""run_query_batch: per-shot realization sweep bits and fault_realizations length contract."""

from __future__ import annotations

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

from stimside._magic.query import PrepareQueries, run_query_batch, run_query_shot


def test_batch_takes_sweep_bits_from_each_shots_own_realization() -> None:
    plan = PrepareQueries(stim.Circuit("CX sweep[0] 0\nM 0"))
    reals = [{"sweep_bits": {0: 0}}, {"sweep_bits": {0: 1}}, None, {"sweep_bits": {0: 1}}]
    records, _ = run_query_batch(plan, 4, rng=0, fault_realizations=reals)
    assert records[:, 0].tolist() == [False, True, False, True]
    for i, real in enumerate(reals):
        shot = run_query_shot(plan, realization=real, rng=0)
        assert records[i].tolist() == shot.measurements.tolist()


def test_batch_explicit_sweep_rows_override_per_shot_realization_sweeps() -> None:
    plan = PrepareQueries(stim.Circuit("CX sweep[0] 0\nM 0"))
    reals = [{"sweep_bits": {0: 1}}, {"sweep_bits": {0: 0}}, {"sweep_bits": {0: 1}}]
    records, _ = run_query_batch(
        plan, 3, rng=0, fault_realizations=reals, sweep_bits=[[0], None, [1]]
    )
    assert records[:, 0].tolist() == [False, False, True]


@pytest.mark.parametrize("length", [0, 1, 3, 5])
def test_batch_fault_realizations_with_wrong_length_raises_value_error(length: int) -> None:
    plan = PrepareQueries(stim.Circuit("H 0\nI[T] 0\nM 0"))
    with pytest.raises(ValueError):
        run_query_batch(plan, 2, rng=1, fault_realizations=[None] * length)


def test_batch_with_wrong_length_realizations_does_not_consume_caller_rng() -> None:
    plan = PrepareQueries(stim.Circuit("H 0\nI[T] 0\nM 0"))
    gen = np.random.default_rng(3)
    state_before = gen.bit_generator.state
    with pytest.raises(ValueError):
        run_query_batch(plan, 2, rng=gen, fault_realizations=[None, None, None])
    assert gen.bit_generator.state == state_before
