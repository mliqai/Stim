"""Parameter dataclasses for stimside leakage tags."""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Literal

import numpy as np

from stimside.util.alias_table import (
    build_alias_table,
    is_1_up_to_rounding,
    sample_from_alias_table,
)

Pauli = Literal["X", "Y", "Z"]
InState = int | Literal["U"]
OutState = int | Literal["U", "V", "X", "Y", "Z", "D"]
InPairState = tuple[InState, InState]
OutPairState = tuple[OutState, OutState]
Transition2ArgInput = tuple[float, InPairState, OutPairState]
Transition2Arg = tuple[OutPairState, float]

OUT_ERROR_STATES = ["X", "Y", "Z", "D"]
_OUT_PAIR_STR_STATES = frozenset((*OUT_ERROR_STATES, "U", "V"))


def _is_state_int(x: object, lo: int = 0) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and lo <= x <= 9


@dataclasses.dataclass(frozen=True)
class LeakageControlledErrorParams:
    name: ClassVar[str] = "LEAKAGE_CONTROLLED_ERROR"
    args: tuple[tuple[float, int, Pauli], ...]
    from_tag: str

    args_by_input_state: dict[int, tuple[tuple[Pauli, float], ...]] = dataclasses.field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        by_state: dict[int, list[tuple[Pauli, float]]] = {}
        for prob, state, pauli in self.args:
            by_state.setdefault(state, []).append((pauli, prob))
        args_by_in = {k: tuple(v) for k, v in by_state.items()}
        object.__setattr__(self, "args_by_input_state", args_by_in)

        seen_state_paulis: set[tuple[int, str]] = set()
        probs_by_state: dict[int, float] = {}
        for p, state, pauli in self.args:
            if p < 0 or p > 1:
                raise ValueError(
                    f"{self.name} has probability argument {p} outside [0,1]"
                )
            if not _is_state_int(state, lo=2):
                raise ValueError(
                    f"{self.name} has invalid state {state}, must be a leaked state between 2 and 9"
                )
            if pauli not in ("X", "Y", "Z"):
                raise ValueError(
                    f"{self.name} pauli was {pauli}, must be in ['X','Y','Z']"
                )
            if (state, pauli) in seen_state_paulis:
                raise ValueError(
                    f"{self.name} has repeated state {state} with pauli {pauli}"
                )
            seen_state_paulis.add((state, pauli))
            probs_by_state[state] = probs_by_state.get(state, 0.0) + p

        for state, total_p in probs_by_state.items():
            if total_p > 1 and not is_1_up_to_rounding(
                total_p, [p for _, p in args_by_in[state]]
            ):
                raise ValueError(
                    f"{self.name} total probability {total_p}>1 for state {state}"
                )


@dataclasses.dataclass(frozen=True)
class _TransitionParamsBase:
    name: ClassVar[str]
    args: tuple[Any, ...]
    from_tag: str

    args_by_input_state: dict[Any, tuple[tuple[Any, float], ...]] = dataclasses.field(
        init=False, repr=False, compare=False
    )
    alias_table_by_input_state: dict[
        Any, tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        by_in: dict[Any, list[tuple[Any, float]]] = {}
        for p, s0, s1 in self.args:
            by_in.setdefault(s0, []).append((s1, p))
        args_by_in = {k: tuple(v) for k, v in by_in.items()}
        object.__setattr__(self, "args_by_input_state", args_by_in)

        for s0, transitions in args_by_in.items():
            probs = [p for _, p in transitions]
            total_p = sum(probs)
            if total_p > 1 and not is_1_up_to_rounding(total_p, probs):
                raise ValueError(
                    f"{self.name} total probability {total_p}>1 for input state {s0}"
                )
        self._validate()
        object.__setattr__(
            self,
            "alias_table_by_input_state",
            {
                s0: build_alias_table(
                    [*tr, (s0, max(0.0, 1 - sum(p for _, p in tr)))]
                )
                for s0, tr in args_by_in.items()
            },
        )

    def _check_prob_and_non_identity(self, p: float, s0: Any, s1: Any) -> None:
        if p < 0 or p > 1:
            raise ValueError(
                f"{self.name} has probability argument {p} outside [0,1]"
            )
        if s0 == s1:
            raise ValueError(f"{self.name} has identity transition {s0}->{s1}")

    def _validate(self) -> None:
        raise NotImplementedError

    def sample_transitions_from_state(
        self, input_state: Any, num_samples: int, np_rng: np.random.Generator
    ) -> np.ndarray:
        """For a given input state, return `num_samples` sampled output states."""
        bases, aliases, thresholds = self.alias_table_by_input_state[input_state]
        return sample_from_alias_table(
            num_samples=num_samples,
            bases=bases,
            aliases=aliases,
            thresholds=thresholds,
            np_rng=np_rng,
        )


@dataclasses.dataclass(frozen=True)
class LeakageTransition1Params(_TransitionParamsBase):
    name: ClassVar[str] = "LEAKAGE_TRANSITION_1"
    args: tuple[tuple[float, InState, InState], ...]

    def _validate(self) -> None:
        for p, s0, s1 in self.args:
            if not (_is_state_int(s0) or s0 == "U"):
                raise ValueError(
                    f"{self.name} has invalid input state {s0}, must be an int between 0 and 9 or 'U'"
                )
            if not (_is_state_int(s1) or s1 == "U"):
                raise ValueError(
                    f"{self.name} has invalid output state {s1}, must be an int between 0 and 9 or 'U'"
                )
            self._check_prob_and_non_identity(p, s0, s1)


@dataclasses.dataclass(frozen=True)
class LeakageTransitionZParams(_TransitionParamsBase):
    name: ClassVar[str] = "LEAKAGE_TRANSITION_Z"
    args: tuple[tuple[float, int, int], ...]

    def _validate(self) -> None:
        for p, s0, s1 in self.args:
            if not _is_state_int(s0):
                raise ValueError(
                    f"{self.name} has invalid input state {s0}, must be an int between 0 and 9"
                )
            if not _is_state_int(s1):
                raise ValueError(
                    f"{self.name} has invalid output state {s1}, must be an int between 0 and 9"
                )
            self._check_prob_and_non_identity(p, s0, s1)


@dataclasses.dataclass(frozen=True)
class LeakageTransition2Params(_TransitionParamsBase):
    name: ClassVar[str] = "LEAKAGE_TRANSITION_2"
    args: tuple[Transition2ArgInput, ...]

    def _validate(self) -> None:
        for p, pair0, pair1 in self.args:
            self._check_prob_and_non_identity(p, pair0, pair1)
            for state in pair0:
                if not (_is_state_int(state) or state == "U"):
                    raise ValueError(
                        f"{self.name} has invalid start state {pair0}->{pair1}, "
                        "states must be 'U' or an int between 0 and 9"
                    )
            for state in pair1:
                if not (
                    _is_state_int(state)
                    or (isinstance(state, str) and state in _OUT_PAIR_STR_STATES)
                ):
                    raise ValueError(
                        f"{self.name} has invalid end state {pair0}->{pair1}, "
                        "states must be 'U', 'V', 'X', 'Y', 'Z', 'D' or an int between 0 and 9"
                    )
            if (pair1[0] == "V" and pair0[0] != "U") or (
                pair1[1] == "V" and pair0[1] != "U"
            ):
                raise ValueError(
                    f"{self.name} has invalid transition to V not from U {pair0}->{pair1}"
                )


@dataclasses.dataclass(frozen=True)
class LeakageConditioningParams:
    name: ClassVar[str] = "LEAKAGE_CONDITIONING"
    args: (
        tuple[tuple[int | str, ...]]
        | tuple[tuple[int | str, ...], tuple[int | str, ...]]
    )
    targets: tuple[int, ...] | None
    from_tag: str

    def __post_init__(self) -> None:
        name = self.from_tag.split(":", 1)[0].strip() or self.name
        if not isinstance(self.args, tuple) or not (
            (len(self.args) == 1 and isinstance(self.args[0], tuple))
            or (
                len(self.args) == 2
                and isinstance(self.args[0], tuple)
                and isinstance(self.args[1], tuple)
                and len(self.args[0]) == len(self.args[1])
            )
        ):
            raise ValueError(
                f"{name} args must be a tuple of one or two tuples, got {self.args}"
            )
        for group in self.args:
            for arg in group:
                if isinstance(arg, str):
                    if arg != "U":
                        raise ValueError(
                            f"{name} state must be 'U' or integers, got {arg}"
                        )
                elif isinstance(arg, int) and not isinstance(arg, bool):
                    if arg < 0 or arg > 9:
                        raise ValueError(
                            f"{name} has invalid state {arg}, must be 'U' or an int between 0 and 9"
                        )
                else:
                    raise ValueError(
                        f"{name} args must be a tuple of integers or strings, got {arg}"
                    )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LeakageConditioningParams):
            return False
        if (
            len(self.args) != len(other.args)
            or self.from_tag != other.from_tag
            or self.targets != other.targets
        ):
            return False
        if len(self.args) == 2:
            return set(zip(self.args[0], self.args[1])) == set(
                zip(other.args[0], other.args[1])
            )
        return all(set(a) == set(b) for a, b in zip(self.args, other.args))


@dataclasses.dataclass(frozen=True)
class LeakageMeasurementParams:
    name: ClassVar[str] = "LEAKAGE_MEASUREMENT"
    args: tuple[tuple[float, int], ...]
    targets: tuple[int, ...] | None
    from_tag: str

    prob_for_input_state: dict[int, float] = dataclasses.field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "prob_for_input_state", {state: prob for prob, state in self.args}
        )
        name = self.from_tag.split(":", 1)[0].strip() or self.name
        seen_states: set[int] = set()
        for p, state in self.args:
            if not 0 <= p <= 1:
                raise ValueError(
                    f"{name} has probability argument {p} outside [0,1]"
                )
            if not _is_state_int(state):
                raise ValueError(
                    f"{name} has invalid state {state}, must be an int between 0 and 9"
                )
            if state in seen_states:
                raise ValueError(f"{name} has repeated state {state}")
            seen_states.add(state)

        if self.targets:
            seen_qubits: set[int] = set()
            for qubit in self.targets:
                if qubit < 0:
                    raise ValueError(f"{name} has target qubit index {qubit} < 0")
                if qubit in seen_qubits:
                    raise ValueError(f"{name} has repeated qubit target {qubit}")
                seen_qubits.add(qubit)


@dataclasses.dataclass(frozen=True)
class LeakageSwapParams:
    """SWAP[LEAKAGE_SWAP]: a SWAP that also swaps the leakage states of each target pair."""

    name: ClassVar[str] = "LEAKAGE_SWAP"
    from_tag: str


LeakageParams = (
    LeakageTransition1Params
    | LeakageControlledErrorParams
    | LeakageTransitionZParams
    | LeakageTransition2Params
    | LeakageConditioningParams
    | LeakageMeasurementParams
    | LeakageSwapParams
)
