"""PVDAQ System 4 parser — CSV files to canonical solar schema."""

import logging
from pathlib import Path

import polars as pl

from windcast.data.solar_schema import SOLAR_SCHEMA

logger = logging.getLogger(__name__)

DATASET_ID = "pvdaq_system4"
SYSTEM_ID = "4"

# System 4 column mapping: PVDAQ name → intermediate name
SIGNAL_MAP = {
    "ac_power__315": "power_watts",
    "poa_irradiance__313": "poa_wm2",
    "ambient_temp__320": "ambient_temp_c",
    "module_temp_1__321": "module_temp_c",
}


def parse_pvdaq(
    data_dir: Path,
    year: int | None = None,
) -> pl.DataFrame:
    """Parse PVDAQ System 4 CSV files into canonical solar schema.

    Args:
        data_dir: Directory containing CSV files (pattern: system_4__date_*.csv).
        year: If provided, filter to files for this year only.

    Returns:
        DataFrame conforming to SOLAR_SCHEMA, aggregated to 15-min resolution.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    csv_files = sorted(data_dir.glob("system_4__date_*.csv"))
    if year is not None:
        csv_files = [f for f in csv_files if f"date_{year}_" in f.name]

    if not csv_files:
        raise FileNotFoundError(f"No PVDAQ CSV files found in {data_dir}")

    logger.info("Found %d CSV files in %s", len(csv_files), data_dir)

    dfs = []
    for csv_path in csv_files:
        df = _read_pvdaq_csv(csv_path)
        if df is not None and len(df) > 0:
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No valid data parsed from {data_dir}")

    combined = pl.concat(dfs)
    logger.info("Combined: %d rows (1-min resolution)", len(combined))

    # Aggregate 1-min → 15-min
    df = _aggregate_to_15min(combined)
    logger.info("Aggregated to 15-min: %d rows", len(df))

    # Add identifier columns
    df = df.with_columns(
        pl.lit(DATASET_ID).alias("dataset_id"),
        pl.lit(SYSTEM_ID).alias("system_id"),
    )

    # Add null columns for signals not available in System 4
    df = df.with_columns(
        pl.lit(None).cast(pl.Float64).alias("ghi_wm2"),
        pl.lit(None).cast(pl.Float64).alias("wind_speed_ms"),
    )

    # Add default QC flag
    df = df.with_columns(pl.lit(0).cast(pl.UInt8).alias("qc_flag"))

    # Cast to schema, reorder, sort
    df = df.select([pl.col(col).cast(dtype) for col, dtype in SOLAR_SCHEMA.items()])
    df = df.sort("timestamp_utc")

    logger.info("Final solar DataFrame: %d rows, %d columns", len(df), len(df.columns))
    return df


def _read_pvdaq_csv(path: Path) -> pl.DataFrame | None:
    """Read a single PVDAQ CSV file and map columns."""
    try:
        df = pl.read_csv(str(path), infer_schema_length=10_000, null_values=["", "NA", "NaN"])
    except Exception:
        logger.warning("Failed to read %s, skipping", path.name)
        return None

    # Check that required columns exist
    available = {k: v for k, v in SIGNAL_MAP.items() if k in df.columns}
    if "measured_on" not in df.columns or not available:
        logger.warning("Missing required columns in %s, skipping", path.name)
        return None

    # Parse timestamp: local time (America/Denver) → UTC
    df = df.with_columns(
        pl.col("measured_on")
        .str.to_datetime("%Y-%m-%d %H:%M:%S")
        .dt.replace_time_zone("America/Denver")
        .dt.convert_time_zone("UTC")
        .alias("timestamp_utc")
    )

    # Select and rename signal columns
    select_exprs = [pl.col("timestamp_utc")]
    for pvdaq_name, internal_name in SIGNAL_MAP.items():
        if pvdaq_name in df.columns:
            select_exprs.append(pl.col(pvdaq_name).cast(pl.Float64).alias(internal_name))

    df = df.select(select_exprs)

    # Convert power: Watts → kW, clip negative to 0
    if "power_watts" in df.columns:
        df = df.with_columns(
            pl.max_horizontal(pl.col("power_watts") / 1000.0, pl.lit(0.0)).alias("power_kw")
        ).drop("power_watts")

    return df


def _aggregate_to_15min(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate 1-minute data to 15-minute means."""
    # Floor timestamp to 15-min intervals
    df = df.with_columns(pl.col("timestamp_utc").dt.truncate("15m").alias("timestamp_utc"))

    # Group and mean all signal columns
    signal_cols = [c for c in df.columns if c != "timestamp_utc"]
    return df.group_by("timestamp_utc").agg([pl.col(c).mean() for c in signal_cols])
