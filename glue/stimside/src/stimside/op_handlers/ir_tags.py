"""Per-consumer lazy tag-parse memo.

Consumers parse instruction tags under distinct caching policies:
- The DEM builder parses each ``(op.name, op.tag)`` once (first parse wins) and
  never re-parses a later instruction with the same key.
- The Tableau precompiler memoises by instruction value.

Because parsing every instruction eagerly would raise on circuits that a
first-parse-wins consumer accepts, tag parsing lives outside ``circuit_ir`` as
a lazy policy object: ``TagMemo(parse, key)``.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Generic, TypeVar

import stim

_T = TypeVar("_T")


class TagMemo(Generic[_T]):
    """Lazy, consumer-owned tag-parse memo with an injected key policy.

    ``key(op)`` decides which instructions share one parse. Exceptions raised by
    ``parse(op)`` are never cached and surface at the consumer's access point.
    The two standard key policies are ``BUILDER_KEY`` (``(name, tag)``) and
    ``VALUE_KEY`` (the ``stim.CircuitInstruction`` value itself).
    """

    def __init__(
        self,
        parse: Callable[[stim.CircuitInstruction], _T],
        key: Callable[[stim.CircuitInstruction], Hashable],
    ) -> None:
        self._parse = parse
        self._key = key
        self._memo: dict[Hashable, _T] = {}

    def params(self, op: stim.CircuitInstruction) -> _T:
        """Return the cached parse result for ``op``, parsing lazily on miss."""
        k = self._key(op)
        if k not in self._memo:
            self._memo[k] = self._parse(op)
        return self._memo[k]


def BUILDER_KEY(op: stim.CircuitInstruction) -> tuple[str, str]:
    """Key policy for the DEM builder: ``(op.name, op.tag)``."""
    return (op.name, op.tag)


def VALUE_KEY(op: stim.CircuitInstruction) -> stim.CircuitInstruction:
    """Key policy for the Tableau precompiler: instruction value."""
    return op
