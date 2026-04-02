"""Ingest Spain demand data: parse CSVs → QC → save Parquet.

Usage:
    uv run python scripts/ingest_spain_demand.py
    uv run python scripts/ingest_spain_demand.py --energy-path data/raw/spain/energy_dataset.csv
"""

import argparse
import logging
import sys
from pathlib import Path

from windcast.config import get_settings
from windcast.data.demand_qc import demand_qc_summary, run_demand_qc_pipeline
from windcast.data.demand_schema import validate_demand_schema
from windcast.data.spain_demand import parse_spain_demand

logger = logging.getLogger(__name__)


def main() -> None:
    """Run Spain demand ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Ingest Spain demand data")
    parser.add_argument(
        "--energy-path",
        type=Path,
        default=None,
        help="Path to energy_dataset.csv. Default: data/raw/spain/energy_dataset.csv",
    )
    parser.add_argument(
        "--weather-path",
        type=Path,
        default=None,
        help="Path to weather_features.csv. Default: data/raw/spain/weather_features.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for Parquet. Default: data/processed/",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()
    energy_path = args.energy_path or settings.raw_dir / "spain" / "energy_dataset.csv"
    weather_path = args.weather_path or settings.raw_dir / "spain" / "weather_features.csv"
    output_dir = args.output_dir or settings.processed_dir

    # Parse
    logger.info("Parsing Spain demand data...")
    df = parse_spain_demand(energy_path, weather_path)
    logger.info("Parsed %d rows", len(df))

    # Validate schema
    errors = validate_demand_schema(df)
    if errors:
        logger.error("Schema validation failed:\n%s", "\n".join(errors))
        sys.exit(1)
    logger.info("Schema validation passed")

    # QC
    logger.info("Running QC pipeline...")
    df = run_demand_qc_pipeline(df, settings.demand_qc)
    summary = demand_qc_summary(df)
    for key, value in summary.items():
        logger.info("QC: %s = %s", key, value)

    # Write single Parquet file
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "spain_demand.parquet"
    df.write_parquet(out_path, compression="zstd", compression_level=3)
    logger.info(
        "Wrote %d rows to %s (%.1f MB)",
        len(df),
        out_path,
        out_path.stat().st_size / 1e6,
    )


if __name__ == "__main__":
    main()
