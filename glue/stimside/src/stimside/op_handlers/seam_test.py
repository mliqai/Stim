"""BatchedLeakageSimulator: the simulators implement it and the handlers use nothing else."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

from stimside.op_handlers.abstract_op_handler import TrivialOpHandler
from stimside.op_handlers.seam import BatchedLeakageSimulator
from stimside.simulator_coset import CosetsideSimulator
from stimside.simulator_magic import MagicsideSimulator

_P = BatchedLeakageSimulator
_METHODS = sorted(n for n, v in vars(_P).items() if inspect.isfunction(v) and n[:2] != "__")
_SIMS = [CosetsideSimulator, MagicsideSimulator]
_HANDLER_DIR = Path(__file__).parent / "leakage_handlers"
_HANDLER_FILES = ["leakage_uint8_coset.py", "leakage_uint8_magic.py", "transition_effects.py"]
# Private simulator attributes the handlers may use besides Protocol members. Keep it empty.
_PRIVATE_ALLOWLIST: frozenset[str] = frozenset()


def _params(fn: Callable[..., Any]) -> list[tuple[str, Any, Any]]:
    return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]


def _sim(cls: type, circuit: str, sync: bool) -> Any:
    c = stim.Circuit(circuit)
    handler = TrivialOpHandler().compile_op_handler(circuit=c, batch_size=2)
    return cls(c, compiled_op_handler=handler, batch_size=2, seed=1, sync_tableside_rng=sync)


@pytest.mark.parametrize("cls", _SIMS)
@pytest.mark.parametrize("name", _METHODS)
def test_simulator_method_matches_protocol(cls: type, name: str) -> None:
    # Names, kinds and defaults must match; mypy checks the annotations.
    assert _params(getattr(cls, name)) == _params(getattr(_P, name))


@pytest.mark.parametrize("cls", _SIMS)
@pytest.mark.parametrize("sync", [False, True])
def test_simulator_has_protocol_attributes_and_live_sync_rngs(cls: type, sync: bool) -> None:
    sim = _sim(cls, "H 0\nM 0", sync)
    assert [a for a in _P.__annotations__ if not hasattr(sim, a)] == []
    rngs = sim.sync_rngs()
    assert (rngs is not None) == sync
    if rngs is not None:  # the live per-shot streams, not copies (sync-mode draw order)
        assert rngs.tab_rngs is sim._tab_rngs and rngs.tab_rngs_x is sim._tab_rngs_x
        assert rngs.shot_np_rngs is sim._shot_np_rngs


@pytest.mark.parametrize("cls", _SIMS)
def test_overwrite_measurement_column_writes_in_place(cls: type) -> None:
    sim = _sim(cls, "X 0\nM 0 0", sync=False)
    sim.run()
    col = sim.measurement_column(-1)
    assert col.tolist() == [True, True]
    sim.overwrite_measurement_column(-1, np.array([False, True]))
    assert sim.measurement_column(-1) is col and col.tolist() == [False, True]
    assert sim.measurement_column(0).tolist() == [True, True]


def _private_sim_accesses(path: Path) -> list[tuple[int, str]]:
    """``sss/tss/css._x`` accesses (the handlers' simulator names) that are not allowed."""
    return sorted(
        (n.lineno, f"{n.value.id}.{n.attr}")
        for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id in ("sss", "tss", "css")
        and n.attr.startswith("_")
        and n.attr not in _METHODS
        and n.attr not in _PRIVATE_ALLOWLIST
    )


def test_protocol_underscore_members_are_only_the_two_seam_methods() -> None:
    # Otherwise a private could be "allowed" by adding it to the Protocol.
    assert [m for m in _METHODS if m.startswith("_")] == ["_do_bare_instruction", "_next_step_meta"]


@pytest.mark.parametrize("fname", _HANDLER_FILES)
def test_handlers_use_no_private_simulator_attributes(fname: str) -> None:
    assert _private_sim_accesses(_HANDLER_DIR / fname) == []
