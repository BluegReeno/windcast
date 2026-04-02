"""Train models via mlforecast with automatic lag management and multi-horizon strategies.

Usage:
    uv run python scripts/train_mlforecast.py --domain wind --turbine-id kwf1
    uv run python scripts/train_mlforecast.py --domain demand --dataset spain_demand
    uv run python scripts/train_mlforecast.py --domain solar --dataset pvdaq_system4
"""

import argparse
import logging
from pathlib import Path

import mlflow
import polars as pl

from windcast.config import get_settings
from windcast.features import list_feature_sets
from windcast.features.exogenous import (
    build_demand_exogenous,
    build_solar_exogenous,
    build_wind_exogenous,
)
from windcast.features.registry import get_feature_set
from windcast.models.evaluation import compute_metrics
from windcast.models.mlforecast_model import (
    MLForecastConfig,
    predict_mlforecast,
    prepare_mlforecast_df,
    train_mlforecast,
)
from windcast.tracking import log_evaluation_results, log_feature_set, setup_mlflow

logger = logging.getLogger(__name__)

EXOG_BUILDERS = {
    "wind": build_wind_exogenous,
    "demand": build_demand_exogenous,
    "solar": build_solar_exogenous,
}


def _temporal_split(
    df: pl.DataFrame,
    train_years: int,
    val_years: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split DataFrame temporally using year-based boundaries on ds column."""
    ts = df.get_column("ds")
    start = ts.min()
    train_end = start.offset_by(f"{train_years}y")  # type: ignore[union-attr]
    val_end = train_end.offset_by(f"{val_years}y")

    train = df.filter(pl.col("ds") < train_end)
    val = df.filter((pl.col("ds") >= train_end) & (pl.col("ds") < val_end))
    test = df.filter(pl.col("ds") >= val_end)

    return train, val, test


def main() -> None:
    """Run mlforecast training pipeline."""
    parser = argparse.ArgumentParser(description="Train models via mlforecast")
    parser.add_argument(
        "--domain",
        choices=["wind", "demand", "solar"],
        default="wind",
        help="Domain: wind, demand, or solar. Default: wind",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Processed data directory. Default: data/processed/",
    )
    parser.add_argument(
        "--feature-set",
        default=None,
        choices=[fs for fs in list_feature_sets() if "exog" in fs],
        help="Exogenous feature set. Default: domain-specific baseline",
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
        help="MLflow experiment name. Default: enercast-mlforecast-{dataset}",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="Forecast horizons in steps. Default: from settings",
    )
    parser.add_argument(
        "--strategy",
        choices=["recursive", "direct", "sparse_direct"],
        default="sparse_direct",
        help="Forecasting strategy. Default: sparse_direct",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()
    processed_dir = args.processed_dir or settings.processed_dir
    horizons = args.horizons or settings.forecast_horizons
    domain = args.domain

    # Domain-specific defaults
    domain_feature_defaults = {
        "wind": "wind_exog_baseline",
        "demand": "demand_exog_baseline",
        "solar": "solar_exog_baseline",
    }
    feature_set = args.feature_set or domain_feature_defaults[domain]
    fs = get_feature_set(feature_set)

    domain_dataset_defaults = {
        "wind": "kelmarsh",
        "demand": "spain_demand",
        "solar": "pvdaq_system4",
    }
    dataset = args.dataset or domain_dataset_defaults[domain]
    experiment_name = args.experiment_name or f"enercast-mlforecast-{dataset}"

    # Resolve processed data file path
    if domain == "demand":
        parquet_path = processed_dir / "spain_demand.parquet"
    elif domain == "solar":
        parquet_path = processed_dir / "pvdaq_system4.parquet"
    else:
        parquet_path = processed_dir / f"kelmarsh_{args.turbine_id}.parquet"

    if not parquet_path.exists():
        logger.error("Processed data file not found: %s", parquet_path)
        return

    # Load processed data (NOT feature data — mlforecast handles lags/rolling)
    logger.info("Loading processed data from %s", parquet_path)
    df = pl.read_parquet(parquet_path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # Build exogenous features
    exog_builder = EXOG_BUILDERS[domain]
    df = exog_builder(df, feature_set)

    # Prepare mlforecast DataFrame (rename to unique_id, ds, y)
    df_ml = prepare_mlforecast_df(df, domain)

    # Filter exogenous columns to those actually present
    available_exog = [c for c in fs.columns if c in df_ml.columns]
    missing_cols = set(fs.columns) - set(available_exog)
    if missing_cols:
        logger.warning("Missing exogenous columns (skipped): %s", sorted(missing_cols))

    # Keep only required columns: unique_id, ds, y + available exogenous
    keep_cols = ["unique_id", "ds", "y", *available_exog]
    df_ml = df_ml.select([c for c in keep_cols if c in df_ml.columns])

    # Temporal split
    train_df, val_df, test_df = _temporal_split(df_ml, settings.train_years, settings.val_years)
    logger.info(
        "Temporal split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df)
    )

    if len(train_df) == 0 or len(val_df) == 0:
        logger.error("Insufficient data for temporal split")
        return

    # Setup MLflow
    setup_mlflow(settings.mlflow_tracking_uri, experiment_name)

    config = MLForecastConfig(strategy=args.strategy)
    run_label = args.turbine_id if domain == "wind" else dataset

    with mlflow.start_run(run_name=f"mlforecast-{run_label}-{feature_set}"):
        mlflow.log_params(
            {
                "domain": domain,
                "dataset": dataset,
                "horizons": str(horizons),
                "strategy": config.strategy,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "backend": "mlforecast",
            }
        )
        if domain == "wind":
            mlflow.log_param("turbine_id", args.turbine_id)
        log_feature_set(feature_set, available_exog)

        # Train mlforecast on training data
        fcst = train_mlforecast(train_df, domain, config, horizons)

        # Predict on validation set using cross-validation approach
        # mlforecast predicts from the end of training data
        h_max = max(horizons)
        preds = predict_mlforecast(fcst, h=h_max)

        # Evaluate per horizon using nested runs
        # For sparse_direct, predictions contain only the specified horizons
        if preds is not None and len(preds) > 0:
            # Join predictions with actuals from validation set
            # preds has columns: unique_id, ds, xgb
            val_actuals = val_df.select(["unique_id", "ds", "y"])
            eval_df = preds.join(val_actuals, on=["unique_id", "ds"], how="inner")

            if len(eval_df) > 0:
                y_true = eval_df.get_column("y").to_numpy()
                y_pred = eval_df.get_column("xgb").to_numpy()
                metrics = compute_metrics(y_true, y_pred)

                with mlflow.start_run(run_name="overall", nested=True):
                    log_evaluation_results(metrics)
                    logger.info(
                        "Overall: MAE=%.1f, RMSE=%.1f",
                        metrics["mae"],
                        metrics["rmse"],
                    )
            else:
                logger.warning(
                    "No overlapping predictions with validation set. "
                    "This is expected if the validation period starts after the prediction window."
                )
        else:
            logger.warning("No predictions generated")

    logger.info("Training complete! Check MLflow UI: %s", settings.mlflow_tracking_uri)


if __name__ == "__main__":
    main()
