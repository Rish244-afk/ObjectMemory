"""
Benchmark and evaluation modules for ObjectMemory.

Provides the synthetic workbench dataset generator and evaluation suite
for comparing pairwise change detection against synthetic ground truth.
"""

from objectmemory.benchmarks.synthetic_dataset import (
    GroundTruthObject,
    SyntheticScenarioPair,
    SyntheticWorkbenchDatasetGenerator,
)
from objectmemory.benchmarks.evaluator import (
    BenchmarkEvaluationReport,
    ObjectEvaluationRecord,
    PairwiseBenchmarkEvaluator,
    ScenarioEvaluationSummary,
)

__all__ = [
    "GroundTruthObject",
    "SyntheticScenarioPair",
    "SyntheticWorkbenchDatasetGenerator",
    "BenchmarkEvaluationReport",
    "ObjectEvaluationRecord",
    "PairwiseBenchmarkEvaluator",
    "ScenarioEvaluationSummary",
]
