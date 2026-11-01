import stim  # type: ignore[import-untyped]

from stimside.op_handlers.leakage_handlers.leakage_parameters import LeakageParams
from stimside.op_handlers.leakage_handlers.tag_registry import (
    _parse_leakage_in_circuit_recurse as _unified_parse_circuit,
    parse_leakage_tag as _unified_parse_tag,
)

_PARSED_CIRCUIT_CACHE: dict[
    tuple[int, int], tuple[stim.Circuit, dict[stim.CircuitInstruction, LeakageParams]]
] = {}


def parse_leakage_tag(op: stim.CircuitInstruction) -> LeakageParams | None:
    """Parse a leakage tag for the tableau simulator."""
    return _unified_parse_tag(op, simulator="tableau")


def parse_leakage_in_circuit(
    circuit: stim.Circuit,
) -> dict[stim.CircuitInstruction, LeakageParams]:
    """Parse all present leakage tags in a circuit, including inside repeats."""
    key = (id(circuit), len(circuit))
    entry = _PARSED_CIRCUIT_CACHE.get(key)
    if entry is not None and entry[0] is circuit and len(circuit) == key[1]:
        return dict(entry[1])
    parsed = _unified_parse_circuit(circuit, simulator="tableau")
    if len(_PARSED_CIRCUIT_CACHE) >= 16:
        _PARSED_CIRCUIT_CACHE.pop(next(iter(_PARSED_CIRCUIT_CACHE)))
    _PARSED_CIRCUIT_CACHE[key] = (circuit, parsed)
    return dict(parsed)
