"""`_as_i32` reads the caller's array once (WP-25 S5a re-gate F5, carried into S5b).

Before the fix `_as_i32` range-checked `np.asarray(arr)` (a view of the caller's array), then
`_as(arr, np.int32, ...)` read the caller's array a second time to cast it: a thread that
rewrote the caller's array between the two reads got an int64 value through the check that the
cast then wrapped to int32 (2**40 + 5 ran as 5). The race is closed by copying once, checking
the copy and casting that copy.

The tests are deterministic and need no second thread, with two hooks for the other thread:
the "caller's array" is an object whose `__array_interface__` (numpy's read of the data)
rewrites the underlying buffer on its Nth read; and `_as`, the cast that follows the check, is
wrapped to rewrite the caller's ndarray just before it runs (a view of the caller's memory would
be re-read there, a private copy is not). They count the reads (1 is the contract) and compare
the result with the data of the first read.
"""

from __future__ import annotations

import numpy as np
import pytest

from stimside._native import magic_engine as me

_EVIL = [2**40 + 5, 2**33 + 1, -(2**35)]  # wrap to 5, 1, 0 as int32
_GOOD = [1, 2, 3]


class _Reads:
    """An int64 vector that is replaced by `_EVIL` when numpy reads it for the Nth time."""

    def __init__(self, flip_at_read: int, values: list[int] = _GOOD) -> None:
        self.buf = np.array(values, np.int64)
        self.flip_at_read = flip_at_read
        self.reads = 0

    @property
    def __array_interface__(self) -> dict[str, object]:
        self.reads += 1
        if self.reads == self.flip_at_read:
            self.buf[:] = _EVIL
        return dict(self.buf.__array_interface__)


def test_a_vector_rewritten_after_the_first_read_is_neither_checked_nor_cast_twice() -> None:
    caller = _Reads(flip_at_read=2)
    out = me._as_i32(caller, (3,), "x")
    assert out.tolist() == _GOOD  # the old code returned [5, 1, 0]: checked as _GOOD, cast as _EVIL
    assert out.dtype == np.int32
    assert caller.reads == 1


@pytest.mark.parametrize("flip_at_read", [2, 3, 5])
def test_every_later_read_is_irrelevant_because_there_is_none(flip_at_read: int) -> None:
    caller = _Reads(flip_at_read)
    assert me._as_i32(caller, (3,), "x").tolist() == _GOOD
    assert caller.reads == 1


def test_a_vector_that_is_wide_at_the_first_read_is_refused_whatever_follows() -> None:
    caller = _Reads(flip_at_read=99, values=_EVIL)  # never flips: wide from the start
    with pytest.raises(ValueError, match="int32"):
        me._as_i32(caller, (3,), "x")
    assert caller.reads == 1


def test_the_result_does_not_alias_the_callers_array() -> None:
    src = np.array(_GOOD, np.int32)
    out = me._as_i32(src, (3,), "x")
    src[:] = 9
    assert out.tolist() == _GOOD
    assert out.flags.c_contiguous and out.flags.owndata


def test_an_ndarray_rewritten_between_the_check_and_the_cast_is_cast_as_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller = np.array(_GOOD, np.int64)
    real_cast = me._as

    def rewrite_then_cast(arr: object, dtype: object, shape: tuple[int, ...], what: str) -> object:
        caller[:] = _EVIL  # "the other thread", after the range check and before the cast
        return real_cast(arr, dtype, shape, what)

    monkeypatch.setattr(me, "_as", rewrite_then_cast)
    out = me._as_i32(caller, (3,), "x")
    assert caller.tolist() == _EVIL  # the hook did fire
    assert out.tolist() == _GOOD  # a cast of the caller's memory would give [5, 1, 0]


def test_a_non_integer_dtype_is_still_a_type_error_after_one_read() -> None:
    caller = _Reads(flip_at_read=99)
    caller.buf = caller.buf.astype(np.float64)
    with pytest.raises(TypeError, match="integer"):
        me._as_i32(caller, (3,), "x")
    assert caller.reads == 1


def test_a_wrong_shape_is_still_a_value_error() -> None:
    with pytest.raises(ValueError, match="shape"):
        me._as_i32(np.array(_GOOD, np.int64), (4,), "x")
