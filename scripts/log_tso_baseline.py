"""Log the RTE day-ahead (Prévision J-1) forecast as a benchmark run in MLflow.

This gives us the "we match the TSO" slide: compare our ``demand_full`` model at
h24 against the official forecast that RTE publishes the day before. Evaluates
on BOTH the validation split (so the bar chart stays consistent with the dev
comparison) AND the held-out test split (2024 on the 8/2/1 split) — the latter
is the gold standard number for the presentation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow
import polars as pl

from windcast.config import DATASETS, get_settings
from windcast.models.evaluation import compute_metrics
from windcast.tracking import setup_mlflow
from windcast.training.harness import temporal_split
from windcast.training.lineage import log_lineage_tags

logger = logging.getLogger(__name__)

DATASET_ID = "rte_france"
EXPERIMENT_NAME = f"enercast-{DATASET_ID}"
PARQUET_PATH = Path("data/processed/rte_france.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log RTE TSO day-ahead forecast as MLflow benchmark"
    )
    parser.add_argument("--generation", default=None, help="Generation label (e.g. gen4)")
    parser.add_argument(
        "--data-quality", default="CLEAN", help="Data quality: CLEAN or LEAKED. Default: CLEAN"
    )
    parser.add_argument("--change-reason", default=None, help="Free-text change reason")
    parser.add_argument(
        "--train-years",
        type=int,
        default=None,
        help="Training split in years. Default: from dataset config",
    )
    parser.add_argument(
        "--val-years",
        type=int,
        default=None,
        help="Validation split in years. Default: from dataset config",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings = get_settings()
    setup_mlflow(settings.mlflow_tracking_uri, EXPERIMENT_NAME)

    df = pl.read_parquet(PARQUET_PATH)
    logger.info("Loaded %d rows from %s", len(df), PARQUET_PATH)

    # Resolve split config: CLI > dataset config > global settings
    dataset_cfg = DATASETS[DATASET_ID]
    effective_train_years = args.train_years or dataset_cfg.train_years or settings.train_years
    effective_val_years = args.val_years or dataset_cfg.val_years or settings.val_years

    _, val_raw, test_raw = temporal_split(df, effective_train_years, effective_val_years)
    val = val_raw.drop_nulls(subset=["load_mw", "tso_forecast_mw"])
    test = test_raw.drop_nulls(subset=["load_mw", "tso_forecast_mw"])

    # Compute boundaries for logging
    train_end = val_raw["timestamp_utc"].min()
    val_end = (
        test_raw["timestamp_utc"].min() if len(test_raw) > 0 else val_raw["timestamp_utc"].max()
    )
    logger.info("Val set: %d rows (%s → %s)", len(val), train_end, val_end)
    logger.info("Test set: %d rows (>= %s)", len(test), val_end)

    val_metrics = compute_metrics(val["load_mw"].to_numpy(), val["tso_forecast_mw"].to_numpy())
    logger.info("TSO val metrics: %s", val_metrics)

    test_metrics: dict[str, float] = {}
    if len(test) > 0:
        test_metrics = compute_metrics(
            test["load_mw"].to_numpy(), test["tso_forecast_mw"].to_numpy()
        )
        logger.info("TSO test metrics: %s", test_metrics)

    with mlflow.start_run(run_name=f"{DATASET_ID}-tso_baseline"):
        mlflow.set_tags(
            {
                "enercast.stage": "dev",
                "enercast.domain": "demand",
                "enercast.purpose": "tso_baseline",
                "enercast.backend": "tso",
                "enercast.feature_set": "none",
                "enercast.run_type": "parent",
                "enercast.data_resolution_min": "60",
            }
        )
        log_lineage_tags(
            generation=args.generation,
            nwp_source="none",
            data_quality=args.data_quality,
            change_reason=args.change_reason,
        )
        mlflow.log_params(
            {
                "domain": "demand",
                "dataset": DATASET_ID,
                "n_val": len(val),
                "n_test": len(test),
                "split.train_years": effective_train_years,
                "split.val_years": effective_val_years,
                "split.train_end": str(train_end),
                "split.val_start": str(train_end),
                "split.val_end": str(val_end),
                "split.test_start": str(val_end),
            }
        )
        # Log against the day-ahead horizon (h24 for hourly data)
        for k, v in val_metrics.items():
            mlflow.log_metric(f"h24_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"h24_test_{k}", v)

        val_md = (
            f"**Val rows:** {len(val):,}\n"
            f"- MAE: {val_metrics['mae']:.0f} MW\n"
            f"- RMSE: {val_metrics['rmse']:.0f} MW\n"
            f"- MAPE: {val_metrics['mape'] * 100:.2f} %\n"
        )
        if test_metrics:
            test_md = (
                f"\n**Test rows ({val_end.date()} onward):** {len(test):,}\n"
                f"- MAE: {test_metrics['mae']:.0f} MW\n"
                f"- RMSE: {test_metrics['rmse']:.0f} MW\n"
                f"- MAPE: {test_metrics['mape'] * 100:.2f} %\n"
            )
        else:
            test_md = "\n**Test set:** empty (no data after val_end)\n"

        mlflow.set_tag(
            "mlflow.note.content",
            (
                "## RTE Day-Ahead Forecast Benchmark\n\n"
                "The official Prévision J-1 forecast published by RTE, evaluated "
                "on the same temporal splits as our ML models. The **test** set "
                "is the gold standard — held out from everything during dev.\n\n"
                f"{val_md}{test_md}\n"
                "Our `demand_full` model at h24 should be in the same league on both splits."
            ),
        )

    logger.info("Logged TSO baseline run to experiment %s", EXPERIMENT_NAME)


if __name__ == "__main__":
    main()
