import stim  # type: ignore[import-untyped]
from typing import Literal, Any, cast

from stimside.op_handlers.leakage_handlers.common_parsing import (
    LEAKAGE_TRANSITIONS_1_MATCH,
    LEAKAGE_TRANSITIONS_2_MATCH,
    try_as_integer,
    split_arguments,
    parse_list_of_targets,
)

from stimside.op_handlers.leakage_handlers.leakage_parameters import (
    LeakageTransition1Params,
    LeakageTransition2Params,
    LeakageTransitionZParams,
    LeakageConditioningParams,
    LeakageMeasurementParams,
    LeakageControlledErrorParams,
    LeakageSwapParams,
)

# --- PARSER HELPERS ---

def _combine_conditioned_states(states: list[int | str], name: str) -> list[int | str]:
    for st in states:
        if not (
            st == "U"
            or (isinstance(st, int) and not isinstance(st, bool) and 0 <= st <= 9)
        ):
            raise ValueError(
                f"{name} has invalid state {st}, must be 'U' or an int between 0 and 9"
            )
    return list(dict.fromkeys(states))


def _parse_list_of_states(args: str, name: str) -> tuple[str | int, ...]:
    states = [try_as_integer(st.strip()) for st in args.split()]
    return tuple(_combine_conditioned_states(cast(list[int | str], states), name))

# --- TAG-SPECIFIC PARSER FUNCTIONS ---

def _parse_controlled_error(op: stim.CircuitInstruction, match: Any, sim: str) -> LeakageControlledErrorParams:
    args_tuples = split_arguments(match.group("args"))
    errors = []
    for a in args_tuples:
        if len(a) != 2: raise ValueError(f"Malformed argument '{a}' in tag '{op.tag}'.")
        a_parsed = [a[0]] + [s.strip() for s in a[1].split("-->")]
        if len(a_parsed) != 3: raise ValueError(f"Malformed transition '{a_parsed}' in {op.tag}.")
        p, state, pauli = float(a_parsed[0]), try_as_integer(a_parsed[1]), a_parsed[2]
        if not isinstance(state, int) or state <= 1 or state > 9:
            raise ValueError(
                f"{LeakageControlledErrorParams.name} has invalid state {state}, "
                "must be a leaked state between 2 and 9"
            )
        errors.append((p, state, cast(Literal["X", "Y", "Z"], pauli)))
    return LeakageControlledErrorParams(args=tuple(errors), from_tag=op.tag)

def _parse_transitions_1(op: stim.CircuitInstruction, match: Any) -> list[Any]:
    args_tuples = split_arguments(match.group("args"))
    transitions = []
    for a in args_tuples:
        if len(a) != 2: raise ValueError(f"Malformed argument '{a}' in tag '{op.tag}'.")
        p = float(a[0])
        m = LEAKAGE_TRANSITIONS_1_MATCH.fullmatch(a[1])
        if not m: raise ValueError(f"Malformed transition '{a[1]}' in tag '{op.tag}'.")
        s0, s1 = try_as_integer(m.group("s0")), try_as_integer(m.group("s1"))
        transitions.append((p, s0, s1))
        if m.group("direction") == "<->": transitions.append((p, s1, s0))
    return transitions

def _parse_transition_1(op: stim.CircuitInstruction, match: Any, sim: str) -> LeakageTransition1Params:
    return LeakageTransition1Params(args=cast(Any, tuple(_parse_transitions_1(op, match))), from_tag=op.tag)

def _parse_transition_z(op: stim.CircuitInstruction, match: Any, sim: str) -> LeakageTransitionZParams:
    # Use transition 1 parsing (not its validation) but return Z params
    return LeakageTransitionZParams(args=cast(Any, tuple(_parse_transitions_1(op, match))), from_tag=op.tag)

def _parse_transition_2(op: stim.CircuitInstruction, match: Any, sim: str) -> LeakageTransition2Params:
    args_tuples = split_arguments(match.group("args"))
    transitions = []
    for a in args_tuples:
        if len(a) != 2: raise ValueError(f"Malformed argument '{a}' in tag '{op.tag}'.")
        p = float(a[0])
        m = LEAKAGE_TRANSITIONS_2_MATCH.fullmatch(a[1])
        if not m: raise ValueError(f"Malformed transition '{a[1]}' in tag '{op.tag}'.")
        s0, s1, s2, s3 = (try_as_integer(m.group(g)) for g in ["s0", "s1", "s2", "s3"])
        transitions.append((p, (s0, s1), (s2, s3)))
        if m.group("direction") == "<->": transitions.append((p, (s2, s3), (s0, s1)))
    return LeakageTransition2Params(args=cast(Any, tuple(transitions)), from_tag=op.tag)

def _parse_prob_states(op: stim.CircuitInstruction, args: str) -> tuple[tuple[float, Any], ...]:
    projections = []
    for a in split_arguments(args):
        if len(a) != 2:
            raise ValueError(f"Malformed argument '{a}' in tag '{op.tag}'.")
        projections.append((float(a[0]), try_as_integer(a[1])))
    return tuple(projections)

def _parse_projection_z(op: stim.CircuitInstruction, match: Any, sim: str) -> LeakageMeasurementParams:
    return LeakageMeasurementParams(
        args=_parse_prob_states(op, match.group("args")), targets=None, from_tag=op.tag
    )

def _parse_measurement(op: stim.CircuitInstruction, match: Any, sim: str) -> LeakageMeasurementParams:
    return LeakageMeasurementParams(
        args=_parse_prob_states(op, match.group("args")),
        targets=tuple(parse_list_of_targets(match.group("targets"))),
        from_tag=op.tag,
    )

def _parse_conditioned_self(op: stim.CircuitInstruction, match: Any, sim: str) -> LeakageConditioningParams:
    if match.group("targets") or ":" in op.tag.split(":", 1)[1]:
        raise ValueError(f"{match.group('name')} takes no ': targets', got '{op.tag}'.")
    states = _parse_list_of_states(match.group("args"), match.group("name"))
    return LeakageConditioningParams(args=(states,), from_tag=op.tag, targets=None)

def _parse_conditioned_other(op: stim.CircuitInstruction, match: Any, sim: str) -> LeakageConditioningParams:
    args, targets = match.group("args"), match.group("targets")
    if not targets: raise ValueError("CONDITIONED_ON_OTHER tag needs to specify targets.")
    states = _parse_list_of_states(args, match.group("name"))
    return LeakageConditioningParams(
        args=(states,), 
        from_tag=op.tag, 
        targets=tuple(parse_list_of_targets(targets))
    )

def _parse_conditioned_pair(op: stim.CircuitInstruction, match: Any, sim: str) -> LeakageConditioningParams:
    if match.group("targets") or ":" in op.tag.split(":", 1)[1]:
        raise ValueError(f"{match.group('name')} takes no ': targets', got '{op.tag}'.")
    states: list[list[int | str]] = [[], []]
    for pair in split_arguments(match.group("args")):
        if len(pair) != 2: raise ValueError("CONDITIONED_ON_PAIR must have pairs of states.")
        converted = [try_as_integer(s) for s in pair]
        for s in converted:
            if isinstance(s, str) and s != "U": raise ValueError(f"Invalid state {s}.")
        states[0].append(converted[0])
        states[1].append(converted[1])
    return LeakageConditioningParams(args=(tuple(states[0]), tuple(states[1])), from_tag=op.tag, targets=None)

def _parse_swap(op: stim.CircuitInstruction, match: Any, sim: str) -> LeakageSwapParams:
    if op.tag != "LEAKAGE_SWAP":
        raise ValueError(f"LEAKAGE_SWAP takes no arguments, got tag '{op.tag}'.")
    return LeakageSwapParams(from_tag=op.tag)


