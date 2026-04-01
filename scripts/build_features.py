"""Build wind features from processed SCADA Parquet files.

Usage:
    uv run python scripts/build_features.py [--feature-set wind_baseline]
    uv run python scripts/build_features.py --feature-set wind_enriched --turbine-id kwf1
"""

import argparse
import logging
from pathlib import Path

import polars as pl

from windcast.config import get_settings
from windcast.features import build_wind_features, list_feature_sets

logger = logging.getLogger(__name__)


def main() -> None:
    """Run feature building pipeline."""
    parser = argparse.ArgumentParser(description="Build wind features from SCADA Parquet")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Input directory with processed Parquet files. Default: data/processed/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for feature Parquet files. Default: data/features/",
    )
    parser.add_argument(
        "--feature-set",
        default="wind_baseline",
        choices=list_feature_sets(),
        help="Feature set to build. Default: wind_baseline",
    )
    parser.add_argument(
        "--turbine-id",
        default=None,
        help="Process only this turbine (e.g., kwf1). Default: all turbines.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()
    input_dir = args.input_dir or settings.processed_dir
    output_dir = args.output_dir or settings.features_dir

    # Find Parquet files
    pattern = f"kelmarsh_{args.turbine_id}.parquet" if args.turbine_id else "kelmarsh_*.parquet"
    parquet_files = sorted(input_dir.glob(pattern))
    if not parquet_files:
        logger.error("No Parquet files found in %s matching %s", input_dir, pattern)
        return

    logger.info("Found %d Parquet files in %s", len(parquet_files), input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pq_file in parquet_files:
        logger.info("Processing %s", pq_file.name)
        df = pl.read_parquet(pq_file)
        rows_in = len(df)

        df = build_wind_features(df, feature_set=args.feature_set)

        # Drop rows with nulls in feature columns (from lags/rolling at series start)
        df = df.drop_nulls()
        rows_out = len(df)

        output_path = output_dir / pq_file.name
        df.write_parquet(output_path, compression="zstd", compression_level=3, statistics=True)
        logger.info(
            "Wrote %s: %d -> %d rows (%d features), %.1f MB",
            output_path.name,
            rows_in,
            rows_out,
            len(df.columns),
            output_path.stat().st_size / 1024 / 1024,
        )

    logger.info("Done! Feature set: %s, output: %s", args.feature_set, output_dir)


if __name__ == "__main__":
    main()
