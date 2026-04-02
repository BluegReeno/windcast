"""Spain ENTSO-E demand dataset parser — energy + weather CSVs to canonical demand schema."""

import logging
from pathlib import Path

import polars as pl

from windcast.data.demand_schema import DEMAND_SCHEMA

logger = logging.getLogger(__name__)

DATASET_ID = "spain_demand"
ZONE_ID = "ES"


def parse_spain_demand(
    energy_path: Path,
    weather_path: Path,
) -> pl.DataFrame:
    """Parse Spain ENTSO-E energy + weather CSVs into canonical demand schema.

    Args:
        energy_path: Path to energy_dataset.csv.
        weather_path: Path to weather_features.csv.

    Returns:
        DataFrame conforming to DEMAND_SCHEMA.
    """
    if not energy_path.exists():
        raise FileNotFoundError(f"Energy CSV not found: {energy_path}")
    if not weather_path.exists():
        raise FileNotFoundError(f"Weather CSV not found: {weather_path}")

    logger.info("Reading energy data from %s", energy_path)
    energy = _read_energy_csv(energy_path)
    logger.info("Energy data: %d rows", len(energy))

    logger.info("Reading weather data from %s", weather_path)
    weather = _read_weather_csv(weather_path)
    logger.info("Weather data: %d rows (before aggregation)", len(weather))

    weather_agg = _aggregate_weather(weather)
    logger.info("Aggregated weather: %d rows", len(weather_agg))

    # Join energy + weather on timestamp
    df = energy.join(weather_agg, on="timestamp_utc", how="inner")
    logger.info("Joined: %d rows", len(df))

    # Add identifier columns
    df = df.with_columns(
        pl.lit(DATASET_ID).alias("dataset_id"),
        pl.lit(ZONE_ID).alias("zone_id"),
    )

    # Add default flag columns
    df = df.with_columns(
        pl.lit(False).alias("is_holiday"),
        pl.lit(False).alias("is_dst_transition"),
        pl.lit(0).cast(pl.UInt8).alias("qc_flag"),
    )

    # Ensure all schema columns exist, cast to schema dtypes, reorder
    for col, dtype in DEMAND_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

    df = df.select([pl.col(col).cast(dtype) for col, dtype in DEMAND_SCHEMA.items()])
    df = df.sort("timestamp_utc")

    logger.info("Final demand DataFrame: %d rows, %d columns", len(df), len(df.columns))
    return df


def _read_energy_csv(path: Path) -> pl.DataFrame:
    """Read energy_dataset.csv and extract relevant columns."""
    df = pl.read_csv(str(path), infer_schema_length=10_000, null_values=["", "NA", "NaN"])

    # Parse timestamp with timezone info → UTC
    df = df.with_columns(
        pl.col("time")
        .str.to_datetime("%Y-%m-%d %H:%M:%S%:z")
        .dt.convert_time_zone("UTC")
        .alias("timestamp_utc")
    )

    # Select and rename relevant columns
    rename_map = {
        "total load actual": "load_mw",
        "price day ahead": "price_eur_mwh",
    }

    cols_to_select = ["timestamp_utc"]
    for src in rename_map:
        if src in df.columns:
            cols_to_select.append(src)
        else:
            logger.warning("Missing column in energy CSV: %s", src)

    df = df.select(cols_to_select).rename({k: v for k, v in rename_map.items() if k in df.columns})

    return df


def _read_weather_csv(path: Path) -> pl.DataFrame:
    """Read weather_features.csv, clean, and convert units."""
    df = pl.read_csv(str(path), infer_schema_length=10_000, null_values=["", "NA", "NaN"])

    # Strip whitespace from city_name (Barcelona has leading space)
    df = df.with_columns(pl.col("city_name").str.strip_chars())

    # Parse timestamp → UTC
    df = df.with_columns(
        pl.col("dt_iso")
        .str.to_datetime("%Y-%m-%d %H:%M:%S%:z")
        .dt.convert_time_zone("UTC")
        .alias("timestamp_utc")
    )

    # Convert Kelvin → Celsius
    df = df.with_columns((pl.col("temp") - 273.15).alias("temperature_c"))

    # Filter Barcelona pressure outliers (keep 900-1100 hPa)
    df = df.with_columns(
        pl.when((pl.col("city_name") == "Barcelona") & (pl.col("pressure") > 1100))
        .then(None)
        .otherwise(pl.col("pressure"))
        .alias("pressure")
    )

    # Filter Valencia wind speed outliers (keep <= 50 m/s)
    df = df.with_columns(
        pl.when((pl.col("city_name") == "Valencia") & (pl.col("wind_speed") > 50))
        .then(None)
        .otherwise(pl.col("wind_speed"))
        .alias("wind_speed")
    )

    df = df.select(
        "timestamp_utc",
        "city_name",
        "temperature_c",
        pl.col("humidity").alias("humidity_pct"),
        pl.col("wind_speed").alias("wind_speed_ms"),
        "pressure",
        "clouds_all",
        "rain_1h",
    )

    return df


def _aggregate_weather(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate weather from 5 cities to mean per timestamp."""
    return df.group_by("timestamp_utc").agg(
        pl.col("temperature_c").mean(),
        pl.col("humidity_pct").mean(),
        pl.col("wind_speed_ms").mean(),
    )
