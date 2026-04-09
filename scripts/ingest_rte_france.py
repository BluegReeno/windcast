"""Ingest RTE éCO2mix France demand: read annual ZIPs → QC → save Parquet.

Usage:
    uv run python scripts/ingest_rte_france.py
    uv run python scripts/ingest_rte_france.py --data-dir data/ecomix-rte
"""

import argparse
import logging
import sys
from pathlib import Path

from windcast.config import get_settings
from windcast.data.demand_qc import demand_qc_summary, run_demand_qc_pipeline
from windcast.data.demand_schema import validate_demand_schema
from windcast.data.rte_france import parse_rte_france

logger = logging.getLogger(__name__)


def main() -> None:
    """Run RTE éCO2mix France demand ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Ingest RTE éCO2mix France demand")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing eCO2mix annual ZIPs. Default: data/ecomix-rte",
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
    data_dir = args.data_dir or (settings.data_dir / "ecomix-rte")
    output_dir = args.output_dir or settings.processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Parsing RTE éCO2mix annual files from %s", data_dir)
    df = parse_rte_france(data_dir)
    logger.info("Parsed %d rows", len(df))

    errors = validate_demand_schema(df)
    if errors:
        logger.error("Schema validation failed:\n%s", "\n".join(errors))
        sys.exit(1)
    logger.info("Schema validation passed")

    logger.info("Running QC pipeline...")
    df = run_demand_qc_pipeline(df, settings.demand_qc)
    summary = demand_qc_summary(df)
    for key, value in summary.items():
        logger.info("QC: %s = %s", key, value)

    out_path = output_dir / "rte_france.parquet"
    df.write_parquet(out_path, compression="zstd", compression_level=3)
    logger.info(
        "Wrote %d rows to %s (%.1f MB)",
        len(df),
        out_path,
        out_path.stat().st_size / 1e6,
    )


if __name__ == "__main__":
    main()
