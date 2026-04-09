"""Build features from processed Parquet files.

Usage:
    uv run python scripts/build_features.py [--feature-set wind_baseline]
    uv run python scripts/build_features.py --domain demand --feature-set demand_baseline
    uv run python scripts/build_features.py --feature-set wind_full --weather-db data/weather.db
"""

import argparse
import logging
from pathlib import Path

import polars as pl

from windcast.config import DOMAIN_RESOLUTION, get_settings
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
    parser.add_argument(
        "--weather-db",
        type=Path,
        default=None,
        help="Path to weather SQLite cache. Enables NWP horizon features for *_full sets.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset ID for file lookup. Default: domain-specific.",
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

    domain_dataset_defaults = {
        "wind": "kelmarsh",
        "demand": "spain_demand",
        "solar": "pvdaq_system4",
    }
    dataset = args.dataset or domain_dataset_defaults[args.domain]

    # Find Parquet files
    if args.domain in ("demand", "solar"):
        pattern = f"{dataset}.parquet"
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

    # Load NWP weather data if requested
    weather_df: pl.DataFrame | None = None
    resolution_minutes = DOMAIN_RESOLUTION.get(args.domain, 60)
    horizons = settings.forecast_horizons

    if args.weather_db is not None:
        from windcast.weather import get_weather_weighted
        from windcast.weather.registry import WeightedWeatherConfig, get_weather_config
        from windcast.weather.storage import WeatherStorage

        # For demand, weather config name tracks the dataset; wind/solar are 1:1
        domain_weather_map = {
            "wind": "kelmarsh",
            "demand": dataset,
            "solar": dataset,
        }
        weather_name = domain_weather_map.get(args.domain)
        if weather_name is None:
            logger.warning("No weather config for domain %s", args.domain)
        else:
            wcfg = get_weather_config(weather_name)
            if isinstance(wcfg, WeightedWeatherConfig):
                # Determine range from the first point's cached coverage
                first_point = wcfg.points[0]
                storage = WeatherStorage(args.weather_db)
                try:
                    coverage = storage.get_coverage(
                        f"{first_point.latitude}_{first_point.longitude}"
                    )
                finally:
                    storage.close()
                if coverage is None:
                    logger.error(
                        "No weather data in %s for %s (first point %s)",
                        args.weather_db,
                        weather_name,
                        first_point.name,
                    )
                else:
                    start, end = coverage[0][:10], coverage[1][:10]
                    weather_df = get_weather_weighted(wcfg, start, end, db_path=args.weather_db)
                    logger.info(
                        "Loaded weighted NWP: %d rows, %d variables, %d points from %s",
                        len(weather_df),
                        len(weather_df.columns) - 1,
                        len(wcfg.points),
                        args.weather_db,
                    )
            else:
                storage = WeatherStorage(args.weather_db)
                try:
                    coverage = storage.get_coverage(f"{wcfg.latitude}_{wcfg.longitude}")
                    if coverage is None:
                        logger.error("No weather data in %s for %s", args.weather_db, weather_name)
                    else:
                        start, end = coverage[0][:10], coverage[1][:10]
                        weather_df = storage.query(
                            f"{wcfg.latitude}_{wcfg.longitude}", start, end, wcfg.variables
                        )
                        logger.info(
                            "Loaded NWP: %d rows, %d variables from %s",
                            len(weather_df),
                            len(weather_df.columns) - 1,
                            args.weather_db,
                        )
                finally:
                    storage.close()

    for pq_file in parquet_files:
        logger.info("Processing %s", pq_file.name)
        df = pl.read_parquet(pq_file)
        rows_in = len(df)

        if args.domain == "demand":
            # For *_full feature sets, inject NWP horizon columns BEFORE calling
            # build_demand_features, so HDD/CDD can read nwp_temperature_2m_h1.
            if weather_df is not None and feature_set == "demand_full":
                from windcast.features.weather import join_nwp_horizon_features

                df = join_nwp_horizon_features(
                    df,
                    weather_df,
                    horizons=horizons,
                    resolution_minutes=resolution_minutes,
                )
            df = build_demand_features(df, feature_set=feature_set)
        elif args.domain == "solar":
            df = build_solar_features(df, feature_set=feature_set)
        else:
            df = build_wind_features(
                df,
                feature_set=feature_set,
                weather_df=weather_df,
                horizons=horizons,
                resolution_minutes=resolution_minutes,
            )

        # Drop rows with nulls in feature columns (from lags/rolling at series start).
        # Restrict the null check to (target + feature set columns + NWP horizon columns)
        # so empty placeholder columns in the canonical schema (e.g. temperature_c for
        # RTE, which only populates load + tso forecast) don't wipe the whole DataFrame.
        from windcast.features.registry import get_feature_set as _gfs

        target_by_domain = {
            "wind": "active_power_kw",
            "demand": "load_mw",
            "solar": "power_kw",
        }
        target_col = target_by_domain[args.domain]
        feature_cols = _gfs(feature_set).columns
        null_check_cols = [target_col]
        for col in feature_cols:
            if col.startswith("nwp_"):
                # NWP columns get horizon-suffixed variants; include them all
                null_check_cols.extend(
                    c for c in df.columns if c.startswith(col) and c not in null_check_cols
                )
            elif col in df.columns:
                null_check_cols.append(col)
        null_check_cols = [c for c in null_check_cols if c in df.columns]
        df = df.drop_nulls(subset=null_check_cols)
        rows_out = len(df)

        # Name output: use input name but replace .parquet with _features.parquet for demand/solar
        if args.domain in ("demand", "solar"):
            output_path = output_dir / f"{dataset}_features.parquet"
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
