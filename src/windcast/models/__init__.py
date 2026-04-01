"""Wind power forecasting models package."""

from windcast.models.evaluation import (
    CustomMetric,
    compute_metrics,
    compute_skill_score,
    evaluate_with_custom_metrics,
    regime_analysis,
)
from windcast.models.persistence import compute_persistence_metrics, persistence_forecast
from windcast.models.xgboost_model import XGBoostConfig, train_multi_horizon, train_xgboost

__all__ = [
    "CustomMetric",
    "XGBoostConfig",
    "compute_metrics",
    "compute_persistence_metrics",
    "compute_skill_score",
    "evaluate_with_custom_metrics",
    "persistence_forecast",
    "regime_analysis",
    "train_multi_horizon",
    "train_xgboost",
]
