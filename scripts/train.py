"""Train XGBoost models per forecast horizon with MLflow tracking.

Usage:
    uv run python scripts/train.py --turbine-id kwf1 --feature-set wind_baseline
    uv run python scripts/train.py --turbine-id kwf1 --horizons 1 6 12 24 48
"""

import argparse
import logging
from pathlib import Path

import polars as pl

from windcast.config import get_settings
from windcast.features import get_feature_set, list_feature_sets
from windcast.models.evaluation import compute_metrics
from windcast.models.persistence import compute_persistence_metrics
from windcast.models.xgboost_model import XGBoostConfig, train_xgboost
from windcast.tracking import log_evaluation_results, log_feature_set, setup_mlflow

logger = logging.getLogger(__name__)


def _temporal_split(
    df: pl.DataFrame,
    train_years: int,
    val_years: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split DataFrame temporally using year-based boundaries.

    Computes split dates from data timestamp range:
    - train: first train_years
    - val: next val_years
    - test: remainder
    """
    ts = df.get_column("timestamp_utc")
    start = ts.min()
    train_end = start.offset_by(f"{train_years}y")  # type: ignore[union-attr]
    val_end = train_end.offset_by(f"{val_years}y")

    train = df.filter(pl.col("timestamp_utc") < train_end)
    val = df.filter((pl.col("timestamp_utc") >= train_end) & (pl.col("timestamp_utc") < val_end))
    test = df.filter(pl.col("timestamp_utc") >= val_end)

    return train, val, test


def _build_horizon_target(
    df: pl.DataFrame,
    horizon: int,
    feature_cols: list[str],
) -> tuple[pl.DataFrame, pl.Series]:
    """Build target for a given horizon and return X, y without nulls.

    Target at row i = active_power_kw at row i+horizon (shift(-h)).
    """
    target_col = f"target_h{horizon}"
    df_h = df.with_columns(pl.col("active_power_kw").shift(-horizon).alias(target_col)).drop_nulls(
        subset=[target_col]
    )

    X = df_h.select(feature_cols)
    y = df_h.get_column(target_col)
    return X, y


def main() -> None:
    """Run training pipeline."""
    parser = argparse.ArgumentParser(description="Train XGBoost wind power models")
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
        help="Feature set to use. Default: wind_baseline",
    )
    parser.add_argument("--turbine-id", default="kwf1", help="Turbine ID. Default: kwf1")
    parser.add_argument(
        "--experiment-name",
        default="windcast-kelmarsh",
        help="MLflow experiment name. Default: windcast-kelmarsh",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="Forecast horizons in steps. Default: from settings",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()
    features_dir = args.features_dir or settings.features_dir
    horizons = args.horizons or settings.forecast_horizons
    fs = get_feature_set(args.feature_set)

    # Load feature data
    parquet_path = features_dir / f"kelmarsh_{args.turbine_id}.parquet"
    if not parquet_path.exists():
        logger.error("Feature file not found: %s", parquet_path)
        return

    logger.info("Loading features from %s", parquet_path)
    df = pl.read_parquet(parquet_path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # Filter feature columns to those actually present in the data
    available_cols = [c for c in fs.columns if c in df.columns]
    missing_cols = set(fs.columns) - set(available_cols)
    if missing_cols:
        logger.warning("Missing feature columns (skipped): %s", sorted(missing_cols))

    # Temporal split
    train_df, val_df, test_df = _temporal_split(df, settings.train_years, settings.val_years)
    logger.info(
        "Temporal split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df)
    )

    if len(train_df) == 0 or len(val_df) == 0:
        logger.error("Insufficient data for temporal split")
        return

    # Setup MLflow
    import mlflow

    setup_mlflow(settings.mlflow_tracking_uri, args.experiment_name)

    config = XGBoostConfig()

    with mlflow.start_run(run_name=f"{args.turbine_id}-{args.feature_set}"):
        # Log experiment-level params
        mlflow.log_params(
            {
                "turbine_id": args.turbine_id,
                "dataset": "kelmarsh",
                "horizons": str(horizons),
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
            }
        )
        log_feature_set(args.feature_set, available_cols)

        # Train per horizon
        for h in horizons:
            logger.info("=== Horizon h=%d (%d min) ===", h, h * 10)

            X_train, y_train = _build_horizon_target(train_df, h, available_cols)
            X_val, y_val = _build_horizon_target(val_df, h, available_cols)

            if len(X_train) == 0 or len(X_val) == 0:
                logger.warning("Insufficient data for horizon %d, skipping", h)
                continue

            with mlflow.start_run(run_name=f"h{h:02d}-{h * 10}min", nested=True):
                mlflow.log_params({"horizon_steps": h, "horizon_minutes": h * 10})

                # Train XGBoost
                model = train_xgboost(X_train, y_train, X_val, y_val, config)

                # Evaluate on validation set
                y_pred = model.predict(X_val)

                # Persistence baseline (lag1 column)
                lag1_col = "active_power_kw_lag1"
                if lag1_col in X_val.columns:
                    y_persistence = X_val.get_column(lag1_col).to_numpy()
                    metrics = compute_metrics(y_val.to_numpy(), y_pred, y_persistence=y_persistence)
                    persistence_metrics = compute_persistence_metrics(
                        y_val.to_numpy(), y_persistence
                    )
                    mlflow.log_metrics(
                        {f"persistence_{k}": v for k, v in persistence_metrics.items()}
                    )
                else:
                    metrics = compute_metrics(y_val.to_numpy(), y_pred)

                log_evaluation_results(metrics, horizon=h)

                logger.info(
                    "h=%d: MAE=%.1f, RMSE=%.1f, skill=%.3f",
                    h,
                    metrics["mae"],
                    metrics["rmse"],
                    metrics.get("skill_score", float("nan")),
                )

    logger.info("Training complete! Check MLflow UI: %s", settings.mlflow_tracking_uri)


if __name__ == "__main__":
    main()
