"""Evaluate trained models on test data with MLflow tracking.

Usage:
    uv run python scripts/evaluate.py --turbine-id kwf1 --feature-set wind_baseline
    uv run python scripts/evaluate.py --turbine-id kwf1 --run-id <mlflow-run-id>
"""

import argparse
import logging
from pathlib import Path

import mlflow
import polars as pl
import xgboost as xgb

from windcast.config import get_settings
from windcast.features import get_feature_set, list_feature_sets
from windcast.models.evaluation import compute_metrics, regime_analysis
from windcast.tracking import log_dataframe_artifact, log_evaluation_results, setup_mlflow

logger = logging.getLogger(__name__)


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
    parser = argparse.ArgumentParser(description="Evaluate wind power forecast models")
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=None,
        help="Features directory. Default: data/features/",
    )
    parser.add_argument(
        "--feature-set",
        default="wind_baseline",
        choices=list_feature_sets(),
        help="Feature set. Default: wind_baseline",
    )
    parser.add_argument("--turbine-id", default="kwf1", help="Turbine ID. Default: kwf1")
    parser.add_argument(
        "--experiment-name",
        default="windcast-kelmarsh",
        help="MLflow experiment name. Default: windcast-kelmarsh",
    )
    parser.add_argument("--run-id", default=None, help="MLflow run ID to evaluate. Default: latest")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()
    features_dir = args.features_dir or settings.features_dir
    fs = get_feature_set(args.feature_set)

    # Setup MLflow
    setup_mlflow(settings.mlflow_tracking_uri, args.experiment_name)

    # Find run to evaluate
    run_id = args.run_id or _find_latest_run(args.experiment_name)
    if run_id is None:
        logger.error("No MLflow runs found for experiment %s", args.experiment_name)
        return

    logger.info("Evaluating run: %s", run_id)

    # Load models
    models = _load_models_from_run(run_id)
    if not models:
        logger.error("No horizon models found in run %s", run_id)
        return
    logger.info("Loaded %d horizon models: %s", len(models), sorted(models.keys()))

    # Load test data
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

    # Evaluate per horizon
    all_results: list[dict[str, float | int]] = []

    with mlflow.start_run(run_name=f"eval-{args.turbine_id}"):
        mlflow.log_params(
            {
                "eval_run_id": run_id,
                "turbine_id": args.turbine_id,
                "feature_set": args.feature_set,
                "n_test": len(test_df),
            }
        )

        for h in sorted(models.keys()):
            # Build target
            target_col = f"target_h{h}"
            test_h = test_df.with_columns(
                pl.col("active_power_kw").shift(-h).alias(target_col)
            ).drop_nulls(subset=[target_col])

            if len(test_h) == 0:
                logger.warning("No test data for horizon %d", h)
                continue

            X_test = test_h.select(available_cols)
            y_true = test_h.get_column(target_col).to_numpy()

            # Predict
            y_pred = models[h].predict(X_test)

            # Persistence baseline
            lag1_col = "active_power_kw_lag1"
            y_persistence = (
                X_test.get_column(lag1_col).to_numpy() if lag1_col in X_test.columns else None
            )

            metrics = compute_metrics(y_true, y_pred, y_persistence=y_persistence)

            with mlflow.start_run(run_name=f"eval-h{h:02d}", nested=True):
                log_evaluation_results(metrics, horizon=h)

                # Regime analysis
                if "wind_speed_ms" in test_h.columns:
                    pred_col = f"y_pred_h{h}"
                    regime_df = test_h.with_columns(pl.Series(name=pred_col, values=y_pred))
                    regimes = regime_analysis(regime_df, target_col, pred_col)
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
