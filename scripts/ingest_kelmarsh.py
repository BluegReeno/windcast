"""Ingest Kelmarsh SCADA data: parse ZIP → QC → save Parquet.

Usage:
    uv run python scripts/ingest_kelmarsh.py [--raw-path PATH] [--output-dir PATH]

If --raw-path is not specified, looks in data/raw/kelmarsh/ for ZIP files.
"""

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

from windcast.config import KELMARSH, get_settings
from windcast.data.kelmarsh import parse_kelmarsh
from windcast.data.qc import qc_summary, run_qc_pipeline
from windcast.data.schema import validate_schema

logger = logging.getLogger(__name__)


def main() -> None:
    """Run Kelmarsh ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Ingest Kelmarsh SCADA data")
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=None,
        help="Path to Kelmarsh ZIP archive or directory. Default: data/raw/kelmarsh/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for Parquet files. Default: data/processed/",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()

    raw_path = args.raw_path or settings.raw_dir / "kelmarsh"
    output_dir = args.output_dir or settings.processed_dir

    # Find ZIP file if a directory was given
    if raw_path.is_dir():
        zips = list(raw_path.glob("*.zip"))
        if zips:
            raw_path = zips[0]
            logger.info("Found ZIP archive: %s", raw_path)

    logger.info("Parsing Kelmarsh data from %s", raw_path)
    df = parse_kelmarsh(raw_path)
    logger.info("Parsed %d rows from %d turbines", len(df), df["turbine_id"].n_unique())

    # Validate schema
    errors = validate_schema(df)
    if errors:
        logger.error("Schema validation failed:\n%s", "\n".join(errors))
        sys.exit(1)
    logger.info("Schema validation passed")

    # Run QC pipeline
    logger.info("Running QC pipeline...")
    df = run_qc_pipeline(df, rated_power_kw=KELMARSH.rated_power_kw, qc_config=settings.qc)
    summary = qc_summary(df)
    for key, value in summary.items():
        logger.info("QC: %s = %s", key, value)

    # Write per-turbine Parquet files
    output_dir.mkdir(parents=True, exist_ok=True)
    turbine_ids = df["turbine_id"].unique().sort().to_list()

    for tid in turbine_ids:
        turbine_df = df.filter(pl.col("turbine_id") == tid)
        filename = f"kelmarsh_{tid.lower()}.parquet"
        output_path = output_dir / filename
        turbine_df.write_parquet(
            output_path,
            compression="zstd",
            compression_level=3,
            statistics=True,
        )
        logger.info(
            "Wrote %s: %d rows, %.1f MB",
            filename,
            len(turbine_df),
            output_path.stat().st_size / 1024 / 1024,
        )

    logger.info(
        "Done! %d turbines, %d total rows written to %s",
        len(turbine_ids),
        len(df),
        output_dir,
    )


if __name__ == "__main__":
    main()
