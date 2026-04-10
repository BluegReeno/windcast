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
    # Validation set metrics (flat keys, historical default)
    "mae": "mae_by_horizon_min",
    "rmse": "rmse_by_horizon_min",
    "bias": "bias_by_horizon_min",
    "skill_score": "skill_score_by_horizon_min",
    "mape": "mape_by_horizon_min",
    "persistence_mae": "persistence_mae_by_horizon_min",
    "persistence_rmse": "persistence_rmse_by_horizon_min",
    "persistence_bias": "persistence_bias_by_horizon_min",
    # Held-out test set metrics — the gold standard number for slides and
    # cross-experiment comparison. Test is never seen during training, tuning,
    # or feature selection; computed once per training run, right after val.
    "test_mae": "test_mae_by_horizon_min",
    "test_rmse": "test_rmse_by_horizon_min",
    "test_bias": "test_bias_by_horizon_min",
    "test_skill_score": "test_skill_score_by_horizon_min",
    "test_mape": "test_mape_by_horizon_min",
    "test_persistence_mae": "test_persistence_mae_by_horizon_min",
    "test_persistence_rmse": "test_persistence_rmse_by_horizon_min",
    "test_persistence_bias": "test_persistence_bias_by_horizon_min",
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
) -> None:
    """Log evaluation metrics to the active MLflow run as flat keys.

    When ``horizon`` is given, metrics are prefixed with ``h{horizon}_`` so
    that ``search_runs(filter_string=...)`` can filter on them and so that
    ``scripts/compare_runs.py`` can read them back. These flat metrics are
    logged once per call (one data point per metric name on the run).

    For the native MLflow "metric vs horizon" line chart, the stepped
    companion :func:`log_stepped_horizon_metrics` must be called on a run
    that accumulates **all** horizons — typically the parent of a nested
    parent/child training loop. See that function's docstring for the
    rationale (MLflow issues #2768 and #7060).

    Args:
        metrics: Dict of metric name -> value.
        horizon: If provided, prefix flat metrics with ``h{horizon}_``.
    """
    if horizon is not None:
        prefixed = {f"h{horizon}_{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(prefixed)
    else:
        mlflow.log_metrics(metrics)


def log_stepped_horizon_metrics(
    metrics_by_horizon_minutes: dict[int, dict[str, float]],
) -> None:
    """Log per-horizon metrics as a stepped time series on the active run.

    For each ``(horizon_minutes, metrics)`` pair, every key present in
    :data:`STEPPED_METRIC_MAP` is logged under its canonical stepped name
    (e.g. ``mae_by_horizon_min``) with ``step=horizon_minutes``. Unknown
    keys are silently skipped. Horizons are iterated in ascending order so
    the resulting metric history is monotonic in ``step``.

    **Call this on the parent run**, after all per-horizon children have
    completed. The parent then holds a single time series per stepped
    metric — which MLflow renders natively as "metric vs horizon" in the
    Charts tab and in Compare Runs, one line per parent run. Logging
    stepped metrics on a child run that covers a single horizon produces a
    single-point series that the UI cannot stitch across sibling runs into
    a curve — confirmed by MLflow maintainers in issues
    `#2768 <https://github.com/mlflow/mlflow/issues/2768>`_ ("it's not
    possible to plot metrics that belong to different runs as one curve")
    and `#7060 <https://github.com/mlflow/mlflow/issues/7060>`_ (which
    documents the canonical pattern: one run, N ``log_metric`` calls with
    increasing ``step``).

    Args:
        metrics_by_horizon_minutes: Mapping of horizon (integer minutes
            ahead of forecast time) to the metric dict returned by
            :func:`windcast.models.evaluation.compute_metrics`, optionally
            merged with ``persistence_*`` metrics. The ``minutes_ahead``
            unit is the only one that is consistent across all EnerCast
            domains (wind 10 min, solar 15 min, demand/price 60 min).
    """
    for horizon_minutes in sorted(metrics_by_horizon_minutes):
        metrics = metrics_by_horizon_minutes[horizon_minutes]
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
