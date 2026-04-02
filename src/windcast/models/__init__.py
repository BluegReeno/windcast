"""Wind power forecasting models package."""

from windcast.models.evaluation import (
    CustomMetric,
    compute_metrics,
    compute_skill_score,
    evaluate_with_custom_metrics,
    regime_analysis,
)
from windcast.models.mlforecast_model import (
    MLForecastConfig,
    create_mlforecast,
    cross_validate_mlforecast,
    predict_mlforecast,
    prepare_mlforecast_df,
    train_mlforecast,
)
from windcast.models.persistence import compute_persistence_metrics, persistence_forecast
from windcast.models.xgboost_model import XGBoostConfig, train_multi_horizon, train_xgboost

__all__ = [
    "CustomMetric",
    "MLForecastConfig",
    "XGBoostConfig",
    "compute_metrics",
    "compute_persistence_metrics",
    "compute_skill_score",
    "create_mlforecast",
    "cross_validate_mlforecast",
    "evaluate_with_custom_metrics",
    "persistence_forecast",
    "predict_mlforecast",
    "prepare_mlforecast_df",
    "regime_analysis",
    "train_mlforecast",
    "train_multi_horizon",
    "train_xgboost",
]
