"""Unified training harness with pluggable ML backends and MLflow lineage."""

from windcast.training.backends import AutoGluonBackend, XGBoostBackend
from windcast.training.harness import TrainingBackend, run_training
from windcast.training.lineage import get_git_info, log_lineage_tags

__all__ = [
    "AutoGluonBackend",
    "TrainingBackend",
    "XGBoostBackend",
    "get_git_info",
    "log_lineage_tags",
    "run_training",
]
