"""More split-token cases (WP-25 S5a re-gate, reviewer_3's T4 and T6; carried into S5b).

`magic_engine_split_token_test.py` covers a foreign engine's token that is OLDER than the
pending one. A comparison written as `split.token < pending` instead of `!=` would pass that
test; these two kill it:

T4. A foreign engine's split that is NEWER than the one pending here is refused too.
T6. `token == 0` (the default of a hand-built `SplitResult`) is not a wildcard: it is refused
    while a split is pending, so a caller cannot opt out of the guard by zeroing the token.

In both cases the engine state and the pending split are untouched (the rightful owner can
still finish).
"""

from __future__ import annotations

import pytest

from stimside._magic.live_store import LiveExpansions
from stimside._magic.live_store_measure_edges_test import (
    _before,
    _engine_with,
    _t_state,
    _x,
)
from stimside._native import magic_engine as me
from stimside._native.magic_engine_split_token_test import _N, _outcomes


def _split(store: LiveExpansions) -> me.SplitResult:
    a = _x(0)
    return store.measure_split((a.a_words, a.b_words, a.kappa), False, None)


def test_a_foreign_split_newer_than_the_pending_one_is_refused() -> None:
    s1 = _engine_with([_t_state(), _t_state(_N, 3)])
    s2 = _engine_with([_t_state(), _t_state(_N, 3)])
    r2 = _split(s2)
    r1 = _split(s1)  # created after r2: its token is greater than the one pending in s2
    assert r1.token > r2.token
    before = _before(s2)
    with pytest.raises(RuntimeError, match="no pending"):
        s2.measure_finish(*_outcomes(r1), r1)
    assert _before(s2) == before
    assert s2.measure_finish(*_outcomes(r2), r2).failure is None  # the owner still can
    assert s1.measure_finish(*_outcomes(r1), r1).failure is None
    s1.close()
    s2.close()


def test_a_split_with_token_zero_is_refused_while_a_split_is_pending() -> None:
    store = _engine_with([_t_state(), _t_state(_N, 3)])
    res = _split(store)
    assert res.token != 0
    before = _before(store)
    with pytest.raises(RuntimeError, match="no pending"):
        store.measure_finish(*_outcomes(res), res._replace(token=0))
    assert _before(store) == before
    assert store.measure_finish(*_outcomes(res), res).failure is None
    store.close()
