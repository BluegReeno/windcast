"""Evaluation metrics, skill scores, and regime analysis."""

import logging
from collections.abc import Callable

import numpy as np
import polars as pl
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

logger = logging.getLogger(__name__)

CustomMetric = Callable[[np.ndarray, np.ndarray], float]


def compute_skill_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_persistence: np.ndarray,
) -> float:
    """Compute skill score: 1 - RMSE_model / RMSE_persistence.

    Range: (-inf, 1]. 1 = perfect, 0 = same as persistence, <0 = worse.
    """
    rmse_model = root_mean_squared_error(y_true, y_pred)
    rmse_persistence = root_mean_squared_error(y_true, y_persistence)
    if rmse_persistence == 0:
        return 1.0 if rmse_model == 0 else -float("inf")
    return 1.0 - (rmse_model / rmse_persistence)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_persistence: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute standard regression metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Model predictions.
        y_persistence: Persistence baseline predictions (optional).

    Returns:
        Dict with mae, rmse, bias, and optionally skill_score and mape.
    """
    metrics: dict[str, float] = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }

    if y_persistence is not None:
        metrics["skill_score"] = compute_skill_score(y_true, y_pred, y_persistence)

    # MAPE only when no zeros in y_true
    if np.all(y_true != 0):
        metrics["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
    else:
        logger.warning("MAPE skipped: y_true contains zeros")

    return metrics


def regime_analysis(
    df: pl.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    wind_speed_col: str = "wind_speed_ms",
) -> dict[str, dict[str, float]]:
    """Compute metrics per wind speed regime.

    Regimes: low (<5 m/s), medium (5-12 m/s), high (>12 m/s).

    Returns:
        Dict mapping regime name to {mae, rmse} dict.
    """
    regimes = {
        "low": pl.col(wind_speed_col) < 5.0,
        "medium": (pl.col(wind_speed_col) >= 5.0) & (pl.col(wind_speed_col) <= 12.0),
        "high": pl.col(wind_speed_col) > 12.0,
    }

    results: dict[str, dict[str, float]] = {}
    for regime_name, condition in regimes.items():
        subset = df.filter(condition)
        if len(subset) == 0:
            logger.warning("No data for regime %r, skipping", regime_name)
            continue
        y_true = subset.get_column(y_true_col).to_numpy()
        y_pred = subset.get_column(y_pred_col).to_numpy()
        results[regime_name] = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(root_mean_squared_error(y_true, y_pred)),
        }

    return results


def evaluate_with_custom_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    custom_metrics: dict[str, CustomMetric],
) -> dict[str, float]:
    """Evaluate predictions with user-provided metric functions.

    Args:
        y_true: Ground truth values.
        y_pred: Model predictions.
        custom_metrics: Dict mapping metric name to callable(y_true, y_pred) -> float.

    Returns:
        Dict mapping metric name to computed value.
    """
    return {name: float(fn(y_true, y_pred)) for name, fn in custom_metrics.items()}
