"""Ingest PVDAQ solar data: parse CSVs → QC → save Parquet.

Usage:
    uv run python scripts/ingest_pvdaq.py --data-dir data/raw/pvdaq/
    uv run python scripts/ingest_pvdaq.py --data-dir data/raw/pvdaq/ --year 2020
"""

import argparse
import logging
import sys
from pathlib import Path

from windcast.config import get_settings
from windcast.data.pvdaq import parse_pvdaq
from windcast.data.solar_qc import run_solar_qc_pipeline, solar_qc_summary
from windcast.data.solar_schema import validate_solar_schema

logger = logging.getLogger(__name__)


def main() -> None:
    """Run PVDAQ solar ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Ingest PVDAQ solar data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing PVDAQ CSV files. Default: data/raw/pvdaq/",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Filter to this year only. Default: all years.",
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
    data_dir = args.data_dir or settings.raw_dir / "pvdaq"
    output_dir = args.output_dir or settings.processed_dir

    # Parse
    logger.info("Parsing PVDAQ data from %s...", data_dir)
    df = parse_pvdaq(data_dir, year=args.year)
    logger.info("Parsed %d rows", len(df))

    # Validate schema
    errors = validate_solar_schema(df)
    if errors:
        logger.error("Schema validation failed:\n%s", "\n".join(errors))
        sys.exit(1)
    logger.info("Schema validation passed")

    # QC
    logger.info("Running QC pipeline...")
    df = run_solar_qc_pipeline(df, settings.solar_qc)
    summary = solar_qc_summary(df)
    for key, value in summary.items():
        logger.info("QC: %s = %s", key, value)

    # Write single Parquet file
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "pvdaq_system4.parquet"
    df.write_parquet(out_path, compression="zstd", compression_level=3)
    logger.info(
        "Wrote %d rows to %s (%.1f MB)",
        len(df),
        out_path,
        out_path.stat().st_size / 1e6,
    )


if __name__ == "__main__":
    main()
