"""Train AutoGluon-Tabular models per forecast horizon with MLflow tracking.

Usage:
    uv run python scripts/train_autogluon.py --turbine-id kwf1 --feature-set wind_full
    uv run python scripts/train_autogluon.py --domain demand --dataset spain_demand
"""

import argparse
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow
import polars as pl

from windcast.config import get_settings
from windcast.features import get_feature_set, list_feature_sets
from windcast.models.autogluon_model import AutoGluonConfig, train_autogluon
from windcast.models.evaluation import compute_metrics
from windcast.models.persistence import compute_persistence_metrics
from windcast.tracking import (
    log_evaluation_results,
    log_stepped_horizon_metrics,
    setup_mlflow,
)

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
    """Split DataFrame temporally using year-based boundaries."""
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
    current horizon and excludes all other horizon variants.
    """
    available_set = set(available_cols)
    actual: list[str] = []
    rename_map: dict[str, str] = {}

    for col in feature_set_cols:
        if col.startswith("nwp_"):
            horizon_col = f"{col}_h{horizon}"
            if horizon_col in available_set:
                actual.append(horizon_col)
                rename_map[horizon_col] = col
            elif col in available_set:
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
    """Build target for a given horizon and return X, y without nulls."""
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
    """Run AutoGluon training pipeline."""
    parser = argparse.ArgumentParser(description="Train AutoGluon-Tabular forecast models")
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
    parser.add_argument(
        "--time-limit",
        type=int,
        default=120,
        help="AutoGluon time limit per horizon in seconds. Default: 120 (demo speed)",
    )
    parser.add_argument(
        "--presets",
        default="good_quality",
        choices=["best_quality", "high_quality", "good_quality", "medium_quality"],
        help="AutoGluon presets. Default: good_quality (demo); use best_quality for final runs",
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
    if domain in ("demand", "solar"):
        parquet_path = features_dir / f"{dataset}_features.parquet"
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

    # Filter non-NWP feature columns to those present in the data
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
    setup_mlflow(settings.mlflow_tracking_uri, experiment_name)

    ag_config = AutoGluonConfig(
        presets=args.presets,
        time_limit=args.time_limit,
    )
    target_col = dcfg["target"]
    lag1_col = dcfg["lag1"]
    run_label = args.turbine_id if domain == "wind" else dataset
    data_resolution = 10 if domain == "wind" else 60

    # Horizon descriptions for human-readable labels
    horizon_desc: dict[int, str] = {}
    for h in horizons:
        minutes = h * data_resolution
        if minutes < 60:
            horizon_desc[h] = f"{minutes} min ahead"
        elif minutes < 1440:
            horizon_desc[h] = f"{minutes // 60}h ahead"
        else:
            horizon_desc[h] = f"D+{minutes // 1440}"

    # Parent tags shared with child runs
    parent_tags = {
        "enercast.stage": "dev",
        "enercast.domain": domain,
        "enercast.purpose": "baseline",
        "enercast.backend": "autogluon",
        "enercast.data_resolution_min": str(data_resolution),
        "enercast.feature_set": feature_set,
    }

    ts_col = "timestamp_utc"

    with mlflow.start_run(run_name=f"{run_label}-autogluon-{feature_set}"):
        mlflow.set_tags({**parent_tags, "enercast.run_type": "parent"})

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
                "ag.presets": ag_config.presets,
                "ag.time_limit": ag_config.time_limit,
                "ag.eval_metric": ag_config.eval_metric,
            }
        )
        if domain == "wind":
            mlflow.log_param("turbine_id", args.turbine_id)

        # Dataset provenance
        src = str(parquet_path)
        train_dataset = mlflow.data.from_polars(
            train_df, source=src, name=f"{dataset}-{run_label}-train", targets=target_col
        )
        val_dataset = mlflow.data.from_polars(
            val_df, source=src, name=f"{dataset}-{run_label}-val", targets=target_col
        )
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(val_dataset, context="validation")

        # Collect results for parent summary
        results_summary: list[str] = []
        ag_model_summary: list[str] = []
        # Per-horizon metric dicts keyed by horizon_minutes — replayed on the
        # parent run as a stepped time series after the child loop completes,
        # so MLflow's Charts tab renders "metric vs horizon" natively.
        horizon_metrics: dict[int, dict[str, float]] = {}

        # Train per horizon
        for h in horizons:
            logger.info("=== Horizon h=%d (%s) ===", h, horizon_desc[h])

            # Resolve NWP columns for this specific horizon
            if has_nwp_horizons:
                h_cols, rename_map = _resolve_horizon_features(df.columns, fs.columns, h)
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
                # Propagate parent tags + horizon-specific tags
                mlflow.set_tags(
                    {
                        **parent_tags,
                        "enercast.run_type": "child",
                        "enercast.horizon_steps": str(h),
                        "enercast.horizon_desc": horizon_desc[h],
                    }
                )
                mlflow.log_params({"horizon_steps": h, "n_features": len(feature_cols_h)})

                # Train AutoGluon
                ag_path = Path(tempfile.mkdtemp(prefix=f"ag_h{h}_"))
                predictor = train_autogluon(
                    X_train, y_train, X_val, y_val, ag_config, ag_path=ag_path
                )

                # Predict on validation set
                X_val_pd = X_val.to_pandas()
                y_pred = predictor.predict(X_val_pd).values

                # Persistence baseline
                if lag1_col in X_val.columns:
                    y_persistence = X_val.get_column(lag1_col).to_numpy()
                    metrics = compute_metrics(y_val.to_numpy(), y_pred, y_persistence=y_persistence)
                    persistence_metrics = compute_persistence_metrics(
                        y_val.to_numpy(), y_persistence
                    )
                    prefixed_persistence = {
                        f"persistence_{k}": v for k, v in persistence_metrics.items()
                    }
                    mlflow.log_metrics(prefixed_persistence)
                else:
                    metrics = compute_metrics(y_val.to_numpy(), y_pred)
                    prefixed_persistence = {}

                log_evaluation_results(metrics, horizon=h)
                # Stash for the parent-level stepped-metric replay
                horizon_metrics[h * data_resolution] = {**metrics, **prefixed_persistence}

                # Log AutoGluon leaderboard as artifact
                lb = predictor.leaderboard(data=None, silent=True)
                lb_path = ag_path / f"ag_leaderboard_h{h}.csv"
                lb.to_csv(str(lb_path), index=False)
                mlflow.log_artifact(str(lb_path))

                # Log best model info
                best_model = lb.iloc[0]["model"]
                n_ag_models = len(lb)
                mlflow.log_params({"n_ag_models": n_ag_models, "best_ag_model": best_model})

                # Child run description
                mae = metrics["mae"]
                rmse = metrics["rmse"]
                skill = metrics.get("skill_score", float("nan"))
                bias = metrics.get("bias", float("nan"))
                child_desc = (
                    f"## Horizon h{h} — {horizon_desc[h]}\n\n"
                    f"**Backend:** AutoGluon ({ag_config.presets}) | "
                    f"**Features:** {len(feature_cols_h)} | "
                    f"**Models trained:** {n_ag_models} | "
                    f"**Best:** {best_model}\n\n"
                    f"| Metric | Value |\n|--------|-------|\n"
                    f"| MAE | {mae:.1f} kW |\n"
                    f"| RMSE | {rmse:.1f} kW |\n"
                    f"| Skill score | {skill:.3f} |\n"
                    f"| Bias | {bias:+.1f} kW |\n"
                )
                mlflow.set_tag("mlflow.note.content", child_desc)

                logger.info(
                    "h=%d: MAE=%.1f, RMSE=%.1f, skill=%.3f (best=%s, %d models)",
                    h,
                    mae,
                    rmse,
                    skill,
                    best_model,
                    n_ag_models,
                )
                results_summary.append(
                    f"h{h} ({horizon_desc[h]}): MAE={mae:.0f} kW, skill={skill:.3f}"
                )
                ag_model_summary.append(f"h{h}: best={best_model}, {n_ag_models} models trained")

        # Replay per-horizon metrics on the parent as a stepped time series
        # (one point per horizon). This is what unlocks the native "metric vs
        # horizon" line chart in the MLflow UI: one line per parent run, N
        # points per line. Children only hold single-horizon flat metrics.
        log_stepped_horizon_metrics(horizon_metrics)

        # Bubble up flat child metrics to parent (enables compare_runs.py and
        # search_runs filters like `metrics.h48_mae < 300`).
        active = mlflow.active_run()
        parent_run_id = active.info.run_id if active else ""
        client = mlflow.tracking.MlflowClient()  # pyright: ignore[reportPrivateImportUsage]
        exp_obj = client.get_experiment_by_name(experiment_name)
        if exp_obj and parent_run_id:
            children = client.search_runs(
                experiment_ids=[exp_obj.experiment_id],
                filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
            )
            for child in children:
                for k, v in child.data.metrics.items():
                    if k.startswith("h") and (
                        "_mae" in k or "_rmse" in k or "_skill_score" in k
                    ):
                        mlflow.log_metric(k, v)

        # Parent run description
        summary_text = "\n".join(results_summary)
        model_text = "\n".join(ag_model_summary)
        parent_desc = (
            f"## {run_label.upper()} — AutoGluon — {feature_set}\n\n"
            f"**Dataset:** {dataset} | "
            f"**Train:** {len(train_df):,} rows | "
            f"**Val:** {len(val_df):,} rows\n\n"
            f"**Presets:** {ag_config.presets} | "
            f"**Time limit:** {ag_config.time_limit}s/horizon\n\n"
            f"### Feature Set\n{fs.description}\n\n"
            f"### Results (validation set)\n{summary_text}\n\n"
            f"### AutoGluon Model Zoo\n{model_text}\n"
        )
        mlflow.set_tag("mlflow.note.content", parent_desc)

    logger.info("Training complete! Check MLflow UI: %s", settings.mlflow_tracking_uri)


if __name__ == "__main__":
    main()
