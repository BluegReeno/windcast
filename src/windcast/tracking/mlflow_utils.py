"""MLflow setup helpers — experiment creation, artifact logging."""

import logging
import tempfile
from pathlib import Path

import mlflow
import polars as pl

logger = logging.getLogger(__name__)

# Canonical mapping from raw metric keys (as returned by
# `windcast.models.evaluation.compute_metrics` and the per-horizon training
# loops) to stepped metric names logged with `step=horizon_minutes`.
#
# The stepped names unlock MLflow's native "metric vs horizon" line chart
# without custom Charts-tab configuration. The unit `minutes_ahead` is the
# only step unit that is consistent across all EnerCast domains (wind 10 min,
# solar 15 min, demand 60 min, price 60 min). Keep this map in sync with
# `compute_metrics` and the persistence-metrics logging in the training scripts.
STEPPED_METRIC_MAP: dict[str, str] = {
    "mae": "mae_by_horizon_min",
    "rmse": "rmse_by_horizon_min",
    "bias": "bias_by_horizon_min",
    "skill_score": "skill_score_by_horizon_min",
    "mape": "mape_by_horizon_min",
    "persistence_mae": "persistence_mae_by_horizon_min",
    "persistence_rmse": "persistence_rmse_by_horizon_min",
    "persistence_bias": "persistence_bias_by_horizon_min",
}


def setup_mlflow(
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str | None = None,
) -> None:
    """Configure MLflow tracking URI and optionally set experiment.

    Args:
        tracking_uri: MLflow tracking URI (default: local SQLite backend).
        experiment_name: Experiment name to create/set. None = skip.
    """
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s", tracking_uri)

    if experiment_name:
        mlflow.set_experiment(experiment_name)
        logger.info("MLflow experiment: %s", experiment_name)


def log_feature_set(feature_set_name: str, feature_columns: list[str]) -> None:
    """Log feature set as MLflow param + JSON artifact.

    Args:
        feature_set_name: Name of the feature set.
        feature_columns: List of feature column names.
    """
    mlflow.log_param("feature_set", feature_set_name)
    mlflow.log_param("n_features", len(feature_columns))
    mlflow.log_dict(
        {"feature_set": feature_set_name, "columns": feature_columns},
        "feature_set.json",
    )


def log_evaluation_results(
    metrics: dict[str, float],
    horizon: int | None = None,
    horizon_minutes: int | None = None,
) -> None:
    """Log evaluation metrics to MLflow.

    Two complementary logging paths:

    1. **Flat path** (always active). When ``horizon`` is given, metrics are
       prefixed with ``h{horizon}_`` and logged via :func:`mlflow.log_metrics`.
       These are filterable via ``search_runs(filter_string=...)`` and feed
       ``scripts/compare_runs.py``.
    2. **Stepped path** (active when ``horizon_minutes`` is given). Each metric
       key present in :data:`STEPPED_METRIC_MAP` is additionally logged under
       its canonical stepped name (e.g. ``mae_by_horizon_min``) with
       ``step=horizon_minutes``. MLflow then auto-plots "metric vs horizon"
       line charts in the UI — one line per run — without custom config.

    Both paths can fire together: in the per-horizon training loop, the flat
    path produces ``h{n}_mae`` and the stepped path produces one data point
    at ``step=n * data_resolution`` on ``mae_by_horizon_min``.

    Args:
        metrics: Dict of metric name -> value.
        horizon: If provided, prefix flat metrics with ``h{horizon}_``.
        horizon_minutes: If provided, also log each metric in
            :data:`STEPPED_METRIC_MAP` under its stepped name with
            ``step=horizon_minutes`` (integer minutes ahead of forecast time).
    """
    if horizon is not None:
        prefixed = {f"h{horizon}_{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(prefixed)
    else:
        mlflow.log_metrics(metrics)

    if horizon_minutes is not None:
        for metric_name, value in metrics.items():
            stepped_name = STEPPED_METRIC_MAP.get(metric_name)
            if stepped_name is not None:
                mlflow.log_metric(stepped_name, value, step=horizon_minutes)


def log_dataframe_artifact(df: pl.DataFrame, name: str) -> None:
    """Save a Polars DataFrame as CSV and log as MLflow artifact.

    Args:
        df: DataFrame to log.
        name: Artifact name (without extension).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / f"{name}.csv"
        df.write_csv(csv_path)
        mlflow.log_artifact(str(csv_path))
        logger.info("Logged artifact: %s.csv (%d rows)", name, len(df))
