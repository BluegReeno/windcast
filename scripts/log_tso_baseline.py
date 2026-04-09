"""Log the RTE day-ahead (Prévision J-1) forecast as a benchmark run in MLflow.

This gives us the "we match the TSO" slide: compare our ``demand_full`` model at
h24 against the official forecast that RTE publishes the day before.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import mlflow
import polars as pl

from windcast.config import get_settings
from windcast.models.evaluation import compute_metrics
from windcast.tracking import setup_mlflow

logger = logging.getLogger(__name__)

DATASET_ID = "rte_france"
EXPERIMENT_NAME = f"enercast-{DATASET_ID}"
PARQUET_PATH = Path("data/processed/rte_france.parquet")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings = get_settings()
    setup_mlflow(settings.mlflow_tracking_uri, EXPERIMENT_NAME)

    df = pl.read_parquet(PARQUET_PATH)
    logger.info("Loaded %d rows from %s", len(df), PARQUET_PATH)

    # Mirror the temporal split used by train.py (train=8y, val=2y, test=1y)
    ts = df.get_column("timestamp_utc")
    start: datetime = ts.min()  # type: ignore[assignment]
    train_end = start.replace(year=start.year + settings.train_years)
    val_end = train_end.replace(year=train_end.year + settings.val_years)
    val = df.filter(
        (pl.col("timestamp_utc") >= train_end) & (pl.col("timestamp_utc") < val_end)
    ).drop_nulls(subset=["load_mw", "tso_forecast_mw"])
    logger.info("Val set: %d rows (%s → %s)", len(val), train_end, val_end)

    y_true = val["load_mw"].to_numpy()
    y_pred = val["tso_forecast_mw"].to_numpy()
    metrics = compute_metrics(y_true, y_pred)
    logger.info("TSO benchmark metrics: %s", metrics)

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
        mlflow.log_params(
            {
                "domain": "demand",
                "dataset": DATASET_ID,
                "n_val": len(val),
                "split.val_start": str(train_end),
                "split.val_end": str(val_end),
            }
        )
        # Log against the day-ahead horizon (~h24 for hourly data)
        for k, v in metrics.items():
            mlflow.log_metric(f"h24_{k}", v)
        mlflow.set_tag(
            "mlflow.note.content",
            (
                "## RTE Day-Ahead Forecast Benchmark\n\n"
                "The official Prévision J-1 forecast published by RTE, evaluated "
                "on the same validation split as our ML models.\n\n"
                f"**Val rows:** {len(val):,}\n"
                f"**MAE:** {metrics['mae']:.0f} MW\n"
                f"**RMSE:** {metrics['rmse']:.0f} MW\n\n"
                "Our `demand_full` model at h24 should be in the same league."
            ),
        )

    logger.info("Logged TSO baseline run to experiment %s", EXPERIMENT_NAME)


if __name__ == "__main__":
    main()
