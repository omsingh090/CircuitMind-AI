"""Evaluation framework: Benchmarks, metrics, and validation test sets."""

from evaluation.evaluate_retrieval import (
    RelevanceMatcher,
    evaluate_query,
    run_benchmark,
    run_strategy_comparison,
)

__all__ = [
    "RelevanceMatcher",
    "evaluate_query",
    "run_benchmark",
    "run_strategy_comparison",
]
