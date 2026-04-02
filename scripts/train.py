"""Train XGBoost models per forecast horizon with MLflow tracking.

Usage:
    uv run python scripts/train.py --turbine-id kwf1 --feature-set wind_baseline
    uv run python scripts/train.py --domain demand --dataset spain_demand
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

DOMAIN_CONFIG: dict[str, dict[str, str]] = {
    "wind": {"target": "active_power_kw", "group": "turbine_id", "lag1": "active_power_kw_lag1"},
    "demand": {"target": "load_mw", "group": "zone_id", "lag1": "load_mw_lag1"},
}


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
    target_col_name: str = "active_power_kw",
) -> tuple[pl.DataFrame, pl.Series]:
    """Build target for a given horizon and return X, y without nulls.

    Target at row i = target value at row i+horizon (shift(-h)).
    """
    target_col = f"target_h{horizon}"
    df_h = df.with_columns(pl.col(target_col_name).shift(-horizon).alias(target_col)).drop_nulls(
        subset=[target_col]
    )

    X = df_h.select(feature_cols)
    y = df_h.get_column(target_col)
    return X, y


def main() -> None:
    """Run training pipeline."""
    parser = argparse.ArgumentParser(description="Train XGBoost forecast models")
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
        help="Feature set to use. Default: domain-specific baseline",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset ID for file lookup. Default: domain-specific",
    )
    parser.add_argument("--turbine-id", default="kwf1", help="(Wind) Turbine ID. Default: kwf1")
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="MLflow experiment name. Default: enercast-{dataset}",
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
    domain = args.domain
    dcfg = DOMAIN_CONFIG[domain]

    # Domain-specific defaults
    feature_set = args.feature_set or ("demand_baseline" if domain == "demand" else "wind_baseline")
    fs = get_feature_set(feature_set)

    dataset = args.dataset or ("spain_demand" if domain == "demand" else "kelmarsh")
    experiment_name = args.experiment_name or f"enercast-{dataset}"

    # Resolve feature file path
    if domain == "demand":
        parquet_path = features_dir / "spain_demand_features.parquet"
    else:
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

    setup_mlflow(settings.mlflow_tracking_uri, experiment_name)

    config = XGBoostConfig()
    target_col = dcfg["target"]
    lag1_col = dcfg["lag1"]
    run_label = args.turbine_id if domain == "wind" else dataset

    with mlflow.start_run(run_name=f"{run_label}-{feature_set}"):
        # Log experiment-level params
        mlflow.log_params(
            {
                "domain": domain,
                "dataset": dataset,
                "horizons": str(horizons),
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
            }
        )
        if domain == "wind":
            mlflow.log_param("turbine_id", args.turbine_id)
        log_feature_set(feature_set, available_cols)

        # Train per horizon
        for h in horizons:
            logger.info("=== Horizon h=%d ===", h)

            X_train, y_train = _build_horizon_target(train_df, h, available_cols, target_col)
            X_val, y_val = _build_horizon_target(val_df, h, available_cols, target_col)

            if len(X_train) == 0 or len(X_val) == 0:
                logger.warning("Insufficient data for horizon %d, skipping", h)
                continue

            with mlflow.start_run(run_name=f"h{h:02d}", nested=True):
                mlflow.log_params({"horizon_steps": h})

                # Train XGBoost
                model = train_xgboost(X_train, y_train, X_val, y_val, config)

                # Evaluate on validation set
                y_pred = model.predict(X_val)

                # Persistence baseline (lag1 column)
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
