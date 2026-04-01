"""Naive persistence baseline — last known power = forecast."""

import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

logger = logging.getLogger(__name__)


def persistence_forecast(y_true: np.ndarray, y_lag1: np.ndarray) -> np.ndarray:
    """Return lag1 as the persistence forecast.

    In a direct forecasting setup with separate models per horizon,
    the persistence baseline simply predicts that power stays at
    the last known value (lag1 in the feature matrix).
    """
    return y_lag1.copy()


def compute_persistence_metrics(
    y_true: np.ndarray,
    y_lag1: np.ndarray,
) -> dict[str, float]:
    """Compute MAE, RMSE, and bias for persistence baseline.

    Args:
        y_true: Actual future power values.
        y_lag1: Last known power values (persistence forecast).

    Returns:
        Dict with mae, rmse, bias keys.
    """
    y_pred = persistence_forecast(y_true, y_lag1)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }
