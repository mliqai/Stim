"""Tests for top-level exports exposed by `stimside.__init__`."""

from __future__ import annotations

import stimside
from stimside import sampler_magic, simulator_magic
from stimside._magic import compile as magic_compile, query as magic_query
from stimside.op_handlers import abstract_op_handler
from stimside.op_handlers.leakage_handlers import (
    leakage_uint8_coset,
    leakage_uint8_flip,
    leakage_uint8_magic,
    leakage_uint8_tableau,
)


def test_top_level_stimside_exports() -> None:
    expected = {
        "__version__",
        "BaseDecoder",
        "BranchAndBoundDecoder",
        "CallableDecoder",
        "CliffordFaultChannel",
        "CompiledLeakageUint8Coset",
        "CompiledLeakageUint8Flip",
        "CompiledLeakageUint8Magic",
        "CompiledLeakageUint8Tableau",
        "CompiledMagicsideSampler",
        "CompiledOpHandler",
        "CosetsideSampler",
        "CosetsideSimulator",
        "EvaluateQueries",
        "FaultBranch",
        "FlipsideSampler",
        "FlipsideSimulator",
        "LeakageDecoder",
        "LeakageUint8Coset",
        "LeakageUint8Flip",
        "LeakageUint8Magic",
        "LeakageUint8Tableau",
        "MagicsideSampler",
        "MagicsideSimulator",
        "MarginalDecoder",
        "MarginalLeakageDemGenerator",
        "OpHandler",
        "PrepareQueries",
        "PreparedQueryPlan",
        "QueryBudget",
        "QueryBudgetExceededError",
        "QueryDiagnostics",
        "QueryResult",
        "TablesideSampler",
        "TablesideSimulator",
        "TrivialOpHandler",
        "evaluate_queries",
        "prepare_queries",
    }
    assert len(expected) == 38
    assert len(stimside.__all__) == len(set(stimside.__all__))
    assert set(stimside.__all__) == expected
    for sym in expected:
        assert hasattr(stimside, sym), f"Missing top-level export {sym}"

    assert stimside.OpHandler is abstract_op_handler.OpHandler
    assert stimside.CompiledOpHandler is abstract_op_handler.CompiledOpHandler
    assert stimside.TrivialOpHandler is abstract_op_handler.TrivialOpHandler
    assert abstract_op_handler._TrivialOpHandler is abstract_op_handler.TrivialOpHandler

    assert stimside.LeakageUint8Flip is leakage_uint8_flip.LeakageUint8
    assert stimside.LeakageUint8Flip is leakage_uint8_flip.LeakageUint8Flip
    assert stimside.CompiledLeakageUint8Flip is leakage_uint8_flip.CompiledLeakageUint8
    assert stimside.CompiledLeakageUint8Flip is leakage_uint8_flip.CompiledLeakageUint8Flip

    assert stimside.LeakageUint8Tableau is leakage_uint8_tableau.LeakageUint8
    assert stimside.LeakageUint8Tableau is leakage_uint8_tableau.LeakageUint8Tableau
    assert stimside.CompiledLeakageUint8Tableau is leakage_uint8_tableau.CompiledLeakageUint8
    assert stimside.CompiledLeakageUint8Tableau is leakage_uint8_tableau.CompiledLeakageUint8Tableau

    assert stimside.LeakageUint8Coset is leakage_uint8_coset.LeakageUint8Coset
    assert stimside.CompiledLeakageUint8Coset is leakage_uint8_coset.CompiledLeakageUint8Coset
    assert stimside.LeakageUint8Magic is leakage_uint8_magic.LeakageUint8Magic
    assert stimside.CompiledLeakageUint8Magic is leakage_uint8_magic.CompiledLeakageUint8Magic

    assert stimside.MagicsideSimulator is simulator_magic.MagicsideSimulator
    assert stimside.MagicsideSampler is sampler_magic.MagicsideSampler
    assert stimside.CompiledMagicsideSampler is sampler_magic.CompiledMagicsideSampler
    assert stimside.CliffordFaultChannel is magic_compile.CliffordFaultChannel
    assert stimside.FaultBranch is magic_compile.FaultBranch
    assert stimside.PrepareQueries is magic_query.PrepareQueries
    assert stimside.EvaluateQueries is magic_query.EvaluateQueries
    assert stimside.prepare_queries is magic_query.prepare_queries
    assert stimside.evaluate_queries is magic_query.evaluate_queries
    assert stimside.QueryBudget is magic_query.QueryBudget
    assert stimside.QueryBudgetExceededError is magic_query.QueryBudgetExceededError
    assert stimside.PreparedQueryPlan is magic_query.PreparedQueryPlan
    assert stimside.QueryResult is magic_query.QueryResult
    assert stimside.QueryDiagnostics is magic_query.QueryDiagnostics
