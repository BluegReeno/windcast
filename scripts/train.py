"""Train XGBoost models per forecast horizon with MLflow tracking.

Usage:
    uv run python scripts/train.py --turbine-id kwf1 --feature-set wind_baseline
    uv run python scripts/train.py --domain demand --dataset spain_demand
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import polars as pl

from windcast.config import get_settings
from windcast.features import get_feature_set, list_feature_sets
from windcast.models.evaluation import compute_metrics
from windcast.models.persistence import compute_persistence_metrics
from windcast.models.xgboost_model import XGBoostConfig, train_xgboost
from windcast.tracking import log_evaluation_results, setup_mlflow

logger = logging.getLogger(__name__)

DOMAIN_CONFIG: dict[str, dict[str, str]] = {
    "wind": {"target": "active_power_kw", "group": "turbine_id", "lag1": "active_power_kw_lag1"},
    "demand": {"target": "load_mw", "group": "zone_id", "lag1": "load_mw_lag1"},
    "solar": {"target": "power_kw", "group": "system_id", "lag1": "power_kw_lag1"},
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
    start: datetime = ts.min()  # type: ignore[assignment]
    train_end = start.replace(year=start.year + train_years)
    val_end = train_end.replace(year=train_end.year + val_years)

    train = df.filter(pl.col("timestamp_utc") < train_end)
    val = df.filter((pl.col("timestamp_utc") >= train_end) & (pl.col("timestamp_utc") < val_end))
    test = df.filter(pl.col("timestamp_utc") >= val_end)

    return train, val, test


def _resolve_horizon_features(
    available_cols: list[str],
    feature_set_cols: list[str],
    horizon: int,
) -> tuple[list[str], dict[str, str]]:
    """Resolve feature columns for a specific horizon.

    For NWP columns (``nwp_*``), selects the ``_h{h}`` variant matching the
    current horizon and excludes all other horizon variants.  Non-NWP columns
    are passed through unchanged.

    Returns:
        ``(actual_columns, rename_map)`` — *actual_columns* are the DataFrame
        column names to select; *rename_map* maps ``nwp_X_h{h}`` → ``nwp_X``
        so the model sees consistent canonical names across horizons.
    """
    available_set = set(available_cols)
    actual: list[str] = []
    rename_map: dict[str, str] = {}

    for col in feature_set_cols:
        if col.startswith("nwp_"):
            # Look for horizon-specific column nwp_X_h{h}
            horizon_col = f"{col}_h{horizon}"
            if horizon_col in available_set:
                actual.append(horizon_col)
                rename_map[horizon_col] = col
            elif col in available_set:
                # Fallback: use unsuffixed column (no horizon shifting)
                actual.append(col)
        elif col in available_set:
            actual.append(col)

    return actual, rename_map


def _build_horizon_target(
    df: pl.DataFrame,
    horizon: int,
    feature_cols: list[str],
    target_col_name: str = "active_power_kw",
    rename_map: dict[str, str] | None = None,
) -> tuple[pl.DataFrame, pl.Series]:
    """Build target for a given horizon and return X, y without nulls.

    Target at row i = target value at row i+horizon (shift(-h)).
    If *rename_map* is provided, renames columns (e.g. strip ``_h{h}`` suffix)
    so the model sees canonical feature names.
    """
    target_col = f"target_h{horizon}"
    df_h = df.with_columns(pl.col(target_col_name).shift(-horizon).alias(target_col)).drop_nulls(
        subset=[target_col]
    )

    X = df_h.select(feature_cols)
    if rename_map:
        X = X.rename(rename_map)
    y = df_h.get_column(target_col)
    return X, y


def main() -> None:
    """Run training pipeline."""
    parser = argparse.ArgumentParser(description="Train XGBoost forecast models")
    parser.add_argument(
        "--domain",
        choices=["wind", "demand", "solar"],
        default="wind",
        help="Domain: wind, demand, or solar. Default: wind",
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
    domain_feature_defaults = {
        "wind": "wind_baseline",
        "demand": "demand_baseline",
        "solar": "solar_baseline",
    }
    feature_set = args.feature_set or domain_feature_defaults[domain]
    fs = get_feature_set(feature_set)

    domain_dataset_defaults = {
        "wind": "kelmarsh",
        "demand": "spain_demand",
        "solar": "pvdaq_system4",
    }
    dataset = args.dataset or domain_dataset_defaults[domain]
    experiment_name = args.experiment_name or f"enercast-{dataset}"

    # Resolve feature file path
    if domain == "demand":
        parquet_path = features_dir / "spain_demand_features.parquet"
    elif domain == "solar":
        parquet_path = features_dir / "pvdaq_system4_features.parquet"
    else:
        parquet_path = features_dir / f"kelmarsh_{args.turbine_id}.parquet"

    if not parquet_path.exists():
        logger.error("Feature file not found: %s", parquet_path)
        return

    logger.info("Loading features from %s", parquet_path)
    df = pl.read_parquet(parquet_path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # Detect whether NWP horizon columns are present (nwp_*_h{h})
    has_nwp_horizons = any(c for c in df.columns if c.startswith("nwp_") and "_h" in c)
    if has_nwp_horizons:
        logger.info("NWP horizon columns detected — will resolve per horizon")

    # Filter non-NWP feature columns to those present in the data.
    # NWP columns are resolved per-horizon in the training loop.
    non_nwp_cols = [c for c in fs.columns if not c.startswith("nwp_")]
    available_non_nwp = [c for c in non_nwp_cols if c in df.columns]
    missing_non_nwp = set(non_nwp_cols) - set(available_non_nwp)
    if missing_non_nwp:
        logger.warning("Missing feature columns (skipped): %s", sorted(missing_non_nwp))

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
    import mlflow.data
    import mlflow.xgboost  # pyright: ignore[reportPrivateImportUsage]

    setup_mlflow(settings.mlflow_tracking_uri, experiment_name)
    mlflow.xgboost.autolog(log_datasets=True, log_model_signatures=True)  # pyright: ignore[reportPrivateImportUsage]

    config = XGBoostConfig()
    target_col = dcfg["target"]
    lag1_col = dcfg["lag1"]
    run_label = args.turbine_id if domain == "wind" else dataset
    data_resolution = 10 if domain == "wind" else 60

    with mlflow.start_run(run_name=f"{run_label}-{feature_set}"):
        # Tags: searchable metadata
        mlflow.set_tags(
            {
                "enercast.stage": "dev",
                "enercast.domain": domain,
                "enercast.purpose": "baseline",
                "enercast.backend": "xgboost",
                "enercast.data_resolution_min": str(data_resolution),
            }
        )

        # Params: run config + split boundaries
        ts_col = "timestamp_utc"
        mlflow.log_params(
            {
                "domain": domain,
                "dataset": dataset,
                "feature_set": feature_set,
                "n_features_base": len(available_non_nwp),
                "horizons": str(horizons),
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "split.train_start": str(train_df[ts_col].min()),
                "split.train_end": str(train_df[ts_col].max()),
                "split.val_start": str(val_df[ts_col].min()),
                "split.val_end": str(val_df[ts_col].max()),
                "split.test_start": str(test_df[ts_col].min()),
                "data.source_file": str(parquet_path),
                "data.n_rows_total": len(df),
            }
        )
        if domain == "wind":
            mlflow.log_param("turbine_id", args.turbine_id)

        # Dataset provenance: native MLflow tracking with auto-hash
        src = str(parquet_path)
        train_dataset = mlflow.data.from_polars(
            train_df, source=src, name=f"{dataset}-{run_label}-train", targets=target_col
        )
        val_dataset = mlflow.data.from_polars(
            val_df, source=src, name=f"{dataset}-{run_label}-val", targets=target_col
        )
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(val_dataset, context="validation")

        # Train per horizon
        for h in horizons:
            logger.info("=== Horizon h=%d ===", h)

            # Resolve NWP columns for this specific horizon
            if has_nwp_horizons:
                h_cols, rename_map = _resolve_horizon_features(df.columns, fs.columns, h)
                # Combine: non-NWP features + horizon-specific NWP features
                feature_cols_h = [c for c in available_non_nwp if not c.startswith("nwp_")] + [
                    c for c in h_cols if c.startswith("nwp_")
                ]
            else:
                feature_cols_h = available_non_nwp + [
                    c for c in fs.columns if c.startswith("nwp_") and c in df.columns
                ]
                rename_map = {}

            X_train, y_train = _build_horizon_target(
                train_df, h, feature_cols_h, target_col, rename_map
            )
            X_val, y_val = _build_horizon_target(val_df, h, feature_cols_h, target_col, rename_map)

            if len(X_train) == 0 or len(X_val) == 0:
                logger.warning("Insufficient data for horizon %d, skipping", h)
                continue

            with mlflow.start_run(run_name=f"h{h:02d}", nested=True):
                mlflow.log_params({"horizon_steps": h, "n_features": len(feature_cols_h)})

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
                    "h=%d: MAE=%.1f, RMSE=%.1f, skill=%.3f (%d features)",
                    h,
                    metrics["mae"],
                    metrics["rmse"],
                    metrics.get("skill_score", float("nan")),
                    len(feature_cols_h),
                )

    logger.info("Training complete! Check MLflow UI: %s", settings.mlflow_tracking_uri)


if __name__ == "__main__":
    main()
