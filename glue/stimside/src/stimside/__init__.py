from __future__ import annotations

from stimside._magic.compile import CliffordFaultChannel, FaultBranch
from stimside._magic.query import (
    EvaluateQueries,
    PrepareQueries,
    PreparedQueryPlan,
    QueryBudget,
    QueryBudgetExceededError,
    QueryDiagnostics,
    QueryResult,
    evaluate_queries,
    prepare_queries,
)
from stimside.dem_generators.bnb_decoder import BranchAndBoundDecoder
from stimside.dem_generators.dem_generator_marginal import MarginalLeakageDemGenerator
from stimside.dem_generators.leakage_decoder import (
    BaseDecoder,
    CallableDecoder,
    LeakageDecoder,
    MarginalDecoder,
)
from stimside.op_handlers.leakage_handlers.leakage_uint8_magic import (
    CompiledLeakageUint8Magic,
    LeakageUint8Magic,
)
from stimside.sampler_coset import CosetsideSampler
from stimside.sampler_flip import FlipsideSampler
from stimside.sampler_magic import CompiledMagicsideSampler, MagicsideSampler
from stimside.sampler_tableau import TablesideSampler
from stimside.simulator_coset import CosetsideSimulator
from stimside.simulator_flip import FlipsideSimulator
from stimside.simulator_magic import MagicsideSimulator
from stimside.simulator_tableau import TablesideSimulator

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BaseDecoder",
    "BranchAndBoundDecoder",
    "CallableDecoder",
    "CliffordFaultChannel",
    "CompiledLeakageUint8Magic",
    "CompiledMagicsideSampler",
    "CosetsideSampler",
    "CosetsideSimulator",
    "EvaluateQueries",
    "FaultBranch",
    "FlipsideSampler",
    "FlipsideSimulator",
    "LeakageDecoder",
    "LeakageUint8Magic",
    "MagicsideSampler",
    "MagicsideSimulator",
    "MarginalDecoder",
    "MarginalLeakageDemGenerator",
    "PrepareQueries",
    "PreparedQueryPlan",
    "QueryBudget",
    "QueryBudgetExceededError",
    "QueryDiagnostics",
    "QueryResult",
    "TablesideSampler",
    "TablesideSimulator",
    "evaluate_queries",
    "prepare_queries",
]
