from dataclasses import dataclass, field
from typing import Callable, Literal, Any, Sequence
import stim  # type: ignore[import-untyped]
import stimside.op_handlers.leakage_handlers.leakage_tag_parsing_unified as ltp

from stimside.op_handlers.leakage_handlers.common_parsing import (
    LEAKAGE_TAG_MATCH,
    LEAKAGE_SWAP_TAG_MATCH,
    CONDITION_TAG_MATCH,
    MEASURE_TAG_MATCH,
    split_arguments,
    try_as_integer,
)
from stimside.op_handlers.leakage_handlers.leakage_parameters import (
    LeakageMeasurementParams,
    LeakageParams,
)

# --- CORE ARCHITECTURE ---

@dataclass(frozen=True)
class TagDef:
    """Definition and validation rules for a leakage tag."""
    name: str
    parser_func: Callable[[stim.CircuitInstruction, Any, str], LeakageParams]
    allowed_arity: Literal[1, 2] | None = None
    allowed_gates: Sequence[str] | dict[Literal["flip", "tableau"], Sequence[str]] | None = None
    supported_simulators: Sequence[Literal["flip", "tableau"]] = field(
        default_factory=lambda: ("flip", "tableau")
    )

    def validate(self, op: stim.CircuitInstruction, simulator: Literal["flip", "tableau"]):
        """Validate that the tag is used correctly on the given operation."""
        if simulator not in self.supported_simulators:
            raise ValueError(
                f"Tag '{self.name}' is not supported in the {simulator} simulator."
            )

        gd = stim.gate_data(op.name)

        if self.name.startswith("CONDITIONED_ON") and gd.produces_measurements:
            raise ValueError(
                f"Tag '{self.name}' cannot be attached to '{op.name}', which produces "
                "measurement results: a conditioned instruction would drop its results "
                "in the shots where its condition fails."
            )
        if self.name.startswith("CONDITIONED_ON") and gd.is_reset:
            raise ValueError(
                f"Tag '{self.name}' cannot be attached to '{op.name}', which is a reset: "
                "conditioned resets are not supported."
            )
        
        if self.allowed_arity == 1 and not gd.is_single_qubit_gate:
            raise ValueError(f"1Q leakage tag '{op.tag}' attached to not-1Q stim gate '{op.name}'.")
        if self.allowed_arity == 2 and not gd.is_two_qubit_gate:
            raise ValueError(f"2Q leakage tag '{op.tag}' attached to not-2Q stim gate '{op.name}'.")

        if self.allowed_gates is not None:
            if isinstance(self.allowed_gates, dict):
                allowed = self.allowed_gates.get(simulator)
            else:
                allowed = self.allowed_gates
            
            if allowed is not None and op.name not in allowed:
                raise ValueError(
                    f"Leakage tag '{self.name}' must be attached to one of {allowed}, "
                    f"but was attached to '{op.name}'."
                )


# --- THE REGISTRY ---

MAGIC_TAGS: frozenset[str] = frozenset({"T", "T_DAG", "CCZ"})


def validate_magic_op(op: stim.CircuitInstruction, num_qubits: int | None = None) -> list[int]:
    """Validate an `I[T]`, `I[T_DAG]`, or `I[CCZ]` instruction and return its plain qubit targets."""
    if op.tag not in MAGIC_TAGS or op.name != "I" or op.gate_args_copy():
        raise ValueError(f"Invalid magic carrier/tag/args on {op!r}")
    qs: list[int] = []
    for t in op.targets_copy():
        if not t.is_qubit_target or t.is_inverted_result_target:
            raise ValueError(f"Magic gate I[{op.tag}] requires plain qubit targets, got {t!r}")
        q = int(t.qubit_value)
        if q < 0 or (num_qubits is not None and q >= num_qubits):
            raise ValueError(f"Invalid target {q} in magic gate I[{op.tag}] for num_qubits={num_qubits}: {op!r}")
        qs.append(q)
    if op.tag == "CCZ" and (len(qs) % 3 != 0 or any(len(set(qs[i : i + 3])) != 3 for i in range(0, len(qs), 3))):
        raise ValueError(f"I[CCZ] requires distinct triples (multiple of 3 targets), got {qs}")
    return qs

TAG_REGISTRY: dict[str, TagDef] = {
    "LEAKAGE_CONTROLLED_ERROR": TagDef(
        name="LEAKAGE_CONTROLLED_ERROR",
        allowed_arity=2,
        supported_simulators=("flip",),
        allowed_gates=("I", "I_ERROR", "II", "II_ERROR"),
        parser_func=ltp._parse_controlled_error
    ),
    "LEAKAGE_TRANSITION_1": TagDef(
        name="LEAKAGE_TRANSITION_1",
        allowed_arity=1,
        allowed_gates={
            "flip": ("I", "I_ERROR", "II", "II_ERROR"),
            "tableau": ("I", "I_ERROR", "II", "II_ERROR", "DETECTOR", "OBSERVABLE_INCLUDE")
        },
        parser_func=ltp._parse_transition_1
    ),
    "LEAKAGE_TRANSITION_Z": TagDef(
        name="LEAKAGE_TRANSITION_Z",
        allowed_arity=1,
        allowed_gates=("I", "I_ERROR", "R"),
        supported_simulators=("flip",),
        parser_func=ltp._parse_transition_z
    ),
    "LEAKAGE_TRANSITION_2": TagDef(
        name="LEAKAGE_TRANSITION_2",
        allowed_arity=2,
        allowed_gates={
            "flip": ("I", "I_ERROR", "II", "II_ERROR"),
            "tableau": ("I", "I_ERROR", "II", "II_ERROR", "DETECTOR", "OBSERVABLE_INCLUDE")
        },
        parser_func=ltp._parse_transition_2
    ),
    "LEAKAGE_PROJECTION_Z": TagDef(
        name="LEAKAGE_PROJECTION_Z",
        allowed_arity=1,
        allowed_gates=("M", "MZ", "MR", "MRZ", "MX", "MY", "MRX", "MRY"),
        parser_func=ltp._parse_projection_z
    ),
    "LEAKAGE_MEASUREMENT": TagDef(
        name="LEAKAGE_MEASUREMENT",
        allowed_gates=("MPAD",),
        parser_func=ltp._parse_measurement
    ),
    "LEAKAGE_SWAP": TagDef(
        name="LEAKAGE_SWAP",
        allowed_arity=2,
        allowed_gates=("SWAP",),
        parser_func=ltp._parse_swap
    ),
    "CONDITIONED_ON_SELF": TagDef(
        name="CONDITIONED_ON_SELF",
        allowed_arity=1,
        supported_simulators=("tableau",),
        parser_func=ltp._parse_conditioned_self
    ),
    "CONDITIONED_ON_OTHER": TagDef(
        name="CONDITIONED_ON_OTHER",
        allowed_arity=1,
        supported_simulators=("tableau",),
        parser_func=ltp._parse_conditioned_other
    ),
    "CONDITIONED_ON_PAIR": TagDef(
        name="CONDITIONED_ON_PAIR",
        allowed_arity=2,
        supported_simulators=("tableau",),
        parser_func=ltp._parse_conditioned_pair
    ),
    "LEAKAGE_DETECTOR": TagDef(
        name="LEAKAGE_DETECTOR",
        allowed_gates=("I", "I_ERROR", "II", "II_ERROR", "DETECTOR", "OBSERVABLE_INCLUDE"),
        supported_simulators=("tableau",),
        parser_func=lambda op, match, sim: None
    ),
}

# --- UNIFIED ENTRY POINT ---

def parse_leakage_tag(op: stim.CircuitInstruction, simulator: Literal["flip", "tableau"]) -> LeakageParams | None:
    """Parse a leakage tag using the unified registry."""
    tag = op.tag
    if not (tag.startswith("LEAKAGE") or tag.startswith("CONDITIONED_ON")):
        return None

    # 1. Match regex to identify name
    if tag.startswith("CONDITIONED_ON"):
        match = CONDITION_TAG_MATCH.fullmatch(tag)
    elif tag.startswith("LEAKAGE_MEASUREMENT"):
        match = MEASURE_TAG_MATCH.fullmatch(tag)
    elif tag == "LEAKAGE_SWAP":
        match = LEAKAGE_SWAP_TAG_MATCH.fullmatch(tag)
    else:
        match = LEAKAGE_TAG_MATCH.fullmatch(tag)
        
    if not match:
        raise ValueError(f"Malformed leakage tag structure: {tag}")
        
    # 2. Lookup definition
    name = match.group("name")
    tag_def = TAG_REGISTRY.get(name)
    if not tag_def:
        raise ValueError(f"Unrecognised leakage tag name: {name}")

    # 3. Validate & Parse
    tag_def.validate(op, simulator)
    return tag_def.parser_func(op, match, simulator)


def parse_leakage_tag_magic(
    op: stim.CircuitInstruction, *, syntax_only: bool = False
) -> tuple[LeakageParams | None, bool]:
    """Parse a leakage tag under Magic rules (expanding MPAD `(p, U)` to `(p, 0) (p, 1)`).

    When ``syntax_only=True``, tags supported only by the flip simulator are validated
    against their flip grammar instead of being rejected as unsupported.
    """
    tag = op.tag
    if not (tag.startswith("LEAKAGE") or tag.startswith("CONDITIONED_ON")):
        return None, False
    match = (
        MEASURE_TAG_MATCH.fullmatch(tag)
        if op.name == "MPAD" and tag.startswith("LEAKAGE_MEASUREMENT")
        else None
    )
    if match and match.group("args"):
        had_u, parts = False, []
        for pair in split_arguments(match.group("args")):
            if len(pair) != 2:
                raise ValueError(f"Malformed argument '{pair}' in tag '{tag}'.")
            is_u = try_as_integer(pair[1]) == "U"
            had_u |= is_u
            parts.append(f"({pair[0]}, 0) ({pair[0]}, 1)" if is_u else f"({pair[0]}, {pair[1]})")
        if had_u:
            rewritten = stim.CircuitInstruction(
                op.name,
                op.targets_copy(),
                op.gate_args_copy(),
                tag=f"LEAKAGE_MEASUREMENT: {' '.join(parts)} : {match.group('targets')}",
            )
            parsed = parse_leakage_tag(rewritten, simulator="tableau")
            assert isinstance(parsed, LeakageMeasurementParams)
            return LeakageMeasurementParams(parsed.args, parsed.targets, tag), True
    sim: Literal["flip", "tableau"] = "tableau"
    if syntax_only:
        tag_def = TAG_REGISTRY.get(tag.split(":", 1)[0].strip())
        if tag_def is not None and "tableau" not in tag_def.supported_simulators:
            sim = "flip"
    return parse_leakage_tag(op, simulator=sim), False


# --- CIRCUIT PARSING ---

def _parse_leakage_in_circuit_recurse(
    circuit: stim.Circuit, simulator: Literal["flip", "tableau"]
) -> dict[stim.CircuitInstruction, LeakageParams]:
    """Recursive helper to parse all present leakage tags in a circuit."""
    parsed_tags = {}
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            parsed_tags.update(
                _parse_leakage_in_circuit_recurse(op.body_copy(), simulator=simulator)
            )
        elif op.tag != "":
            parsed = parse_leakage_tag(op, simulator=simulator)
            if parsed is not None:
                parsed_tags[op] = parsed

    return parsed_tags


def check_tag_qubits(
    ops_to_params: dict[stim.CircuitInstruction, LeakageParams], num_qubits: int
) -> None:
    """Raise ValueError if a tag lists a qubit index >= the circuit's ``num_qubits``."""
    for op, params in ops_to_params.items():
        qubits = getattr(params, "targets", None) or ()
        if any(q >= num_qubits for q in qubits):
            raise ValueError(
                f"{op} lists qubit {max(qubits)} in its tag, but the circuit has "
                f"num_qubits={num_qubits}."
            )