"""Evaluate trained models on test data with MLflow tracking.

Usage:
    uv run python scripts/evaluate.py --turbine-id kwf1 --feature-set wind_baseline
    uv run python scripts/evaluate.py --domain demand --dataset spain_demand
"""

import argparse
import logging
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import xgboost as xgb

from windcast.config import get_settings
from windcast.features import get_feature_set, list_feature_sets
from windcast.models.evaluation import compute_metrics, regime_analysis
from windcast.tracking import log_dataframe_artifact, log_evaluation_results, setup_mlflow

logger = logging.getLogger(__name__)

DOMAIN_CONFIG: dict[str, dict[str, str]] = {
    "wind": {"target": "active_power_kw", "group": "turbine_id", "lag1": "active_power_kw_lag1"},
    "demand": {"target": "load_mw", "group": "zone_id", "lag1": "load_mw_lag1"},
}


def _demand_regime_analysis(
    test_h: pl.DataFrame,
    target_col: str,
    pred_col: str,
) -> dict[str, dict[str, float]]:
    """Time-of-day regime analysis for demand domain."""
    regimes: dict[str, dict[str, float]] = {}
    hour = test_h.get_column("timestamp_utc").dt.hour()

    regime_map = {
        "off_peak": (hour >= 0) & (hour < 8),
        "shoulder": (hour >= 8) & (hour < 18),
        "peak": (hour >= 18) & (hour <= 23),
    }

    for name, mask in regime_map.items():
        subset = test_h.filter(mask)
        if len(subset) == 0:
            continue
        y_true = subset.get_column(target_col).to_numpy()
        y_pred = subset.get_column(pred_col).to_numpy()
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        regimes[name] = {"mae": mae, "rmse": rmse, "n_samples": float(len(subset))}

    return regimes


def _temporal_split(
    df: pl.DataFrame,
    train_years: int,
    val_years: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split DataFrame temporally (same logic as train.py)."""
    ts = df.get_column("timestamp_utc")
    start = ts.min()
    train_end = start.offset_by(f"{train_years}y")  # type: ignore[union-attr]
    val_end = train_end.offset_by(f"{val_years}y")

    train = df.filter(pl.col("timestamp_utc") < train_end)
    val = df.filter((pl.col("timestamp_utc") >= train_end) & (pl.col("timestamp_utc") < val_end))
    test = df.filter(pl.col("timestamp_utc") >= val_end)

    return train, val, test


def _load_models_from_run(run_id: str) -> dict[int, xgb.XGBRegressor]:
    """Load trained models from an MLflow run's child runs."""
    client = mlflow.tracking.MlflowClient()
    parent_run = client.get_run(run_id)

    # Find child runs
    child_runs = client.search_runs(
        experiment_ids=[parent_run.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
    )

    models: dict[int, xgb.XGBRegressor] = {}
    for child in child_runs:
        horizon = int(child.data.params.get("horizon_steps", 0))
        if horizon == 0:
            continue
        model_uri = f"runs:/{child.info.run_id}/model"
        model = mlflow.xgboost.load_model(model_uri)
        models[horizon] = model
        logger.info("Loaded model for horizon h=%d from run %s", horizon, child.info.run_id)

    return models


def _find_latest_run(experiment_name: str) -> str | None:
    """Find the latest parent run ID for an experiment."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.parentRunId = ''",
        order_by=["start_time DESC"],
        max_results=1,
    )
    # If no parentless runs found, get latest run
    if not runs:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
    return runs[0].info.run_id if runs else None


def main() -> None:
    """Run evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate forecast models")
    parser.add_argument(
        "--domain",
        choices=["wind", "demand"],
        default="wind",
        help="Domain: wind or demand. Default: wind",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=None,
        help="Features directory. Default: data/features/",
    )
    parser.add_argument(
        "--feature-set",
        default=None,
        choices=list_feature_sets(),
        help="Feature set. Default: domain-specific baseline",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset ID. Default: domain-specific",
    )
    parser.add_argument("--turbine-id", default="kwf1", help="(Wind) Turbine ID. Default: kwf1")
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="MLflow experiment name. Default: enercast-{dataset}",
    )
    parser.add_argument("--run-id", default=None, help="MLflow run ID to evaluate. Default: latest")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()
    features_dir = args.features_dir or settings.features_dir
    domain = args.domain
    dcfg = DOMAIN_CONFIG[domain]

    feature_set = args.feature_set or ("demand_baseline" if domain == "demand" else "wind_baseline")
    fs = get_feature_set(feature_set)

    dataset = args.dataset or ("spain_demand" if domain == "demand" else "kelmarsh")
    experiment_name = args.experiment_name or f"enercast-{dataset}"

    # Setup MLflow
    setup_mlflow(settings.mlflow_tracking_uri, experiment_name)

    # Find run to evaluate
    run_id = args.run_id or _find_latest_run(experiment_name)
    if run_id is None:
        logger.error("No MLflow runs found for experiment %s", experiment_name)
        return

    logger.info("Evaluating run: %s", run_id)

    # Load models
    models = _load_models_from_run(run_id)
    if not models:
        logger.error("No horizon models found in run %s", run_id)
        return
    logger.info("Loaded %d horizon models: %s", len(models), sorted(models.keys()))

    # Load test data
    if domain == "demand":
        parquet_path = features_dir / "spain_demand_features.parquet"
    else:
        parquet_path = features_dir / f"kelmarsh_{args.turbine_id}.parquet"

    if not parquet_path.exists():
        logger.error("Feature file not found: %s", parquet_path)
        return

    df = pl.read_parquet(parquet_path)
    available_cols = [c for c in fs.columns if c in df.columns]

    # Get test split
    _, _, test_df = _temporal_split(df, settings.train_years, settings.val_years)
    logger.info("Test set: %d rows", len(test_df))

    if len(test_df) == 0:
        logger.error("No test data available")
        return

    target_col_name = dcfg["target"]
    lag1_col = dcfg["lag1"]
    run_label = args.turbine_id if domain == "wind" else dataset

    # Evaluate per horizon
    all_results: list[dict[str, float | int]] = []

    with mlflow.start_run(run_name=f"eval-{run_label}"):
        mlflow.log_params(
            {
                "eval_run_id": run_id,
                "domain": domain,
                "dataset": dataset,
                "feature_set": feature_set,
                "n_test": len(test_df),
            }
        )
        if domain == "wind":
            mlflow.log_param("turbine_id", args.turbine_id)

        for h in sorted(models.keys()):
            # Build target
            target_col = f"target_h{h}"
            test_h = test_df.with_columns(
                pl.col(target_col_name).shift(-h).alias(target_col)
            ).drop_nulls(subset=[target_col])

            if len(test_h) == 0:
                logger.warning("No test data for horizon %d", h)
                continue

            X_test = test_h.select(available_cols)
            y_true = test_h.get_column(target_col).to_numpy()

            # Predict
            y_pred = models[h].predict(X_test)

            # Persistence baseline
            y_persistence = (
                X_test.get_column(lag1_col).to_numpy() if lag1_col in X_test.columns else None
            )

            metrics = compute_metrics(y_true, y_pred, y_persistence=y_persistence)

            with mlflow.start_run(run_name=f"eval-h{h:02d}", nested=True):
                log_evaluation_results(metrics, horizon=h)

                # Domain-specific regime analysis
                pred_col = f"y_pred_h{h}"
                regime_df = test_h.with_columns(pl.Series(name=pred_col, values=y_pred))

                if domain == "wind" and "wind_speed_ms" in test_h.columns:
                    regimes = regime_analysis(regime_df, target_col, pred_col)
                elif domain == "demand" and "timestamp_utc" in test_h.columns:
                    regimes = _demand_regime_analysis(regime_df, target_col, pred_col)
                else:
                    regimes = {}

                for regime_name, regime_metrics in regimes.items():
                    mlflow.log_metrics(
                        {f"regime_{regime_name}_{k}": v for k, v in regime_metrics.items()}
                    )

            result_row: dict[str, float | int] = {"horizon": h, **metrics}
            all_results.append(result_row)

            logger.info(
                "h=%d: MAE=%.1f, RMSE=%.1f, skill=%.3f",
                h,
                metrics["mae"],
                metrics["rmse"],
                metrics.get("skill_score", float("nan")),
            )

        # Summary table
        if all_results:
            results_df = pl.DataFrame(all_results)
            log_dataframe_artifact(results_df, "evaluation_results")

            print("\n=== Evaluation Results ===")
            print(results_df)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
