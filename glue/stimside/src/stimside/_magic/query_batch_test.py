"""run_query_batch: realizations are validated before any shot runs, and match run_query_shot."""

from __future__ import annotations

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

from stimside._magic.query import PrepareQueries, run_query_batch, run_query_shot

_CIRCUIT = "H 0\nI[T] 0\nH 0\nDEPOLARIZE1(0.2) 0\nM 0\nI[CLIFFORD_FAULT: (0.2, Z)] 1\nM 1"


def test_batch_with_invalid_later_realization_raises_before_consuming_caller_rng() -> None:
    plan = PrepareQueries(stim.Circuit(_CIRCUIT))
    gen = np.random.default_rng(5)
    state_before = gen.bit_generator.state
    bad = [None, None, {"faults": {99: 1}}, None]
    with pytest.raises(ValueError):
        run_query_batch(plan, 4, rng=gen, fault_realizations=bad)
    assert gen.bit_generator.state == state_before


def test_batch_matches_sequential_shots_drawing_from_one_generator() -> None:
    plan = PrepareQueries(stim.Circuit(_CIRCUIT))
    reals = [None, {"measurements": {0: 1}}, {"paulis": {0: 1}}, {"faults": [1]}]
    records, trajs = run_query_batch(
        plan, 4, rng=np.random.default_rng(11), fault_realizations=reals
    )
    gen = np.random.default_rng(11)
    for i, real in enumerate(reals):
        shot = run_query_shot(plan, realization=real, rng=gen)
        assert records[i].tolist() == shot.measurements.tolist()
        assert trajs[i].branch_mass == shot.branch_mass
        assert trajs[i].expansion.chi == shot.expansion.chi


def test_batch_broadcasts_one_realization_and_per_shot_sweep_rows() -> None:
    plan = PrepareQueries(stim.Circuit("CX sweep[0] 0\nM 0"))
    records, _ = run_query_batch(plan, 3, sweep_bits=[[0], [1], [0]], rng=0)
    assert records[:, 0].tolist() == [False, True, False]
    records, _ = run_query_batch(plan, 2, fault_realizations={"sweep_bits": {0: 1}}, rng=0)
    assert records[:, 0].tolist() == [True, True]
