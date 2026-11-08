"""EvaluateQueries tolerates a longer `faults` list than the slot count (WP-22 item 3)."""

from __future__ import annotations

import pytest
import stim  # type: ignore[import-untyped]

from stimside._magic.query import EvaluateQueries, PrepareQueries


def test_query_realization_tolerates_a_longer_faults_list_than_the_slot_count() -> None:
    # EvaluateQueries is lenient (strict_faults=False): extra trailing entries are ignored,
    # unlike MagicsideSimulator, which rejects them (simulator_magic_realization_test.py).
    p = PrepareQueries(stim.Circuit("H 0\nI[CLIFFORD_FAULT: (0.2, Z)] 0\nH 0\nM 0"))
    for faults in ([0], [0, 0], [0, 1, 1]):
        res = EvaluateQueries(p, {"faults": faults, "measurements": {0: 0}}, queries=("mass",))
        assert abs((res.mass or 0.0) - 1.0) < 1e-12
    with pytest.raises(ValueError, match="invalid for 1 slots"):
        EvaluateQueries(p, {"faults": [], "measurements": {0: 0}}, queries=("mass",))
