"""MLflow setup helpers — experiment creation, artifact logging."""

import logging
import tempfile
from pathlib import Path

import mlflow
import polars as pl

logger = logging.getLogger(__name__)


def setup_mlflow(
    tracking_uri: str = "file:./mlruns",
    experiment_name: str | None = None,
) -> None:
    """Configure MLflow tracking URI and optionally set experiment.

    Args:
        tracking_uri: MLflow tracking URI (default: local file store).
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
    """Log evaluation metrics to MLflow, optionally prefixed by horizon.

    Args:
        metrics: Dict of metric name -> value.
        horizon: If provided, prefix metrics with h{horizon}_.
    """
    if horizon is not None:
        prefixed = {f"h{horizon}_{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(prefixed)
    else:
        mlflow.log_metrics(metrics)


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
