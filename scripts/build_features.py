"""Build features from processed Parquet files.

Usage:
    uv run python scripts/build_features.py [--feature-set wind_baseline]
    uv run python scripts/build_features.py --domain demand --feature-set demand_baseline
"""

import argparse
import logging
from pathlib import Path

import polars as pl

from windcast.config import get_settings
from windcast.features import (
    build_demand_features,
    build_solar_features,
    build_wind_features,
    list_feature_sets,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Run feature building pipeline."""
    parser = argparse.ArgumentParser(description="Build features from processed Parquet")
    parser.add_argument(
        "--domain",
        choices=["wind", "demand", "solar"],
        default="wind",
        help="Domain: wind or demand. Default: wind",
    )
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
        default=None,
        choices=list_feature_sets(),
        help="Feature set to build. Default: domain-specific baseline",
    )
    parser.add_argument(
        "--turbine-id",
        default=None,
        help="(Wind only) Process only this turbine (e.g., kwf1). Default: all turbines.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()
    input_dir = args.input_dir or settings.processed_dir
    output_dir = args.output_dir or settings.features_dir

    # Domain-specific defaults
    if args.feature_set is None:
        domain_defaults = {
            "wind": "wind_baseline",
            "demand": "demand_baseline",
            "solar": "solar_baseline",
        }
        feature_set = domain_defaults[args.domain]
    else:
        feature_set = args.feature_set

    # Find Parquet files
    if args.domain == "demand":
        pattern = "spain_demand.parquet"
    elif args.domain == "solar":
        pattern = "pvdaq_system4.parquet"
    elif args.turbine_id:
        pattern = f"kelmarsh_{args.turbine_id}.parquet"
    else:
        pattern = "kelmarsh_*.parquet"

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

        if args.domain == "demand":
            df = build_demand_features(df, feature_set=feature_set)
        elif args.domain == "solar":
            df = build_solar_features(df, feature_set=feature_set)
        else:
            df = build_wind_features(df, feature_set=feature_set)

        # Drop rows with nulls in feature columns (from lags/rolling at series start)
        df = df.drop_nulls()
        rows_out = len(df)

        # Name output: use input name but replace .parquet with _features.parquet for demand
        if args.domain == "demand":
            output_path = output_dir / "spain_demand_features.parquet"
        elif args.domain == "solar":
            output_path = output_dir / "pvdaq_system4_features.parquet"
        else:
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

    logger.info("Done! Feature set: %s, output: %s", feature_set, output_dir)


if __name__ == "__main__":
    main()
