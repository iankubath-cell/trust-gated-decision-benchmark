"""Domain implementations for TGAI benchmark."""

from .sequential_testing import (
    BenchmarkConfig,
    TrialConfig,
    run_benchmark,
    split_data,
)

__all__ = [
    "BenchmarkConfig",
    "TrialConfig",
    "run_benchmark",
    "split_data",
]
