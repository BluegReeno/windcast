"""RTE éCO2mix annual definitive files parser — local ZIPs to canonical demand schema.

The éCO2mix definitive files are TSV in ISO-8859-1 (Latin-1) encoding inside a
ZIP, despite the `.xls` extension. Load (Consommation) is at 30-min native
resolution; the TSO day-ahead forecast (Prévision J-1) is at 15-min native
resolution. Both are aggregated to hourly mean in the parser.
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import polars as pl

from windcast.data.demand_schema import DEMAND_SCHEMA

logger = logging.getLogger(__name__)

DATASET_ID = "rte_france"
ZONE_ID = "FR"
ANNUAL_PATTERN = "eCO2mix_RTE_Annuel-Definitif_*.zip"
SOURCE_TZ = "Europe/Paris"


def parse_rte_france(data_dir: Path) -> pl.DataFrame:
    """Read all eCO2mix annual ZIPs in data_dir, return canonical demand DataFrame.

    Args:
        data_dir: Directory containing eCO2mix_RTE_Annuel-Definitif_<YYYY>.zip files.

    Returns:
        DataFrame conforming to DEMAND_SCHEMA, sorted by timestamp_utc, hourly.

    Raises:
        FileNotFoundError: If data_dir doesn't exist or contains no matching files.
        ValueError: If no yearly file could be parsed successfully.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"RTE data directory not found: {data_dir}")

    zip_files = sorted(data_dir.glob(ANNUAL_PATTERN))
    if not zip_files:
        raise FileNotFoundError(
            f"No annual definitive files found in {data_dir} (pattern: {ANNUAL_PATTERN})"
        )

    logger.info(
        "Found %d annual files: %s → %s",
        len(zip_files),
        zip_files[0].name,
        zip_files[-1].name,
    )

    yearly_dfs: list[pl.DataFrame] = []
    for zp in zip_files:
        try:
            yearly_dfs.append(_parse_one_year(zp))
        except Exception as e:
            logger.warning("Failed to parse %s: %s", zp.name, e)
            continue

    if not yearly_dfs:
        raise ValueError("No yearly files could be parsed")

    df = pl.concat(yearly_dfs, how="vertical_relaxed")
    logger.info(
        "Concatenated %d yearly frames: %d rows before hourly resample",
        len(yearly_dfs),
        len(df),
    )

    # Aggregate to hourly (load is 30-min native, forecast is 15-min native)
    df = _resample_hourly(df)
    logger.info("After hourly resample: %d rows", len(df))

    # Identifier + default flag columns
    df = df.with_columns(
        pl.lit(DATASET_ID).alias("dataset_id"),
        pl.lit(ZONE_ID).alias("zone_id"),
        pl.lit(False).alias("is_holiday"),
        pl.lit(False).alias("is_dst_transition"),
        pl.lit(0).cast(pl.UInt8).alias("qc_flag"),
    )

    # Ensure all schema columns exist, cast, reorder
    for col, dtype in DEMAND_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df = df.select([pl.col(col).cast(dtype) for col, dtype in DEMAND_SCHEMA.items()])
    df = df.sort("timestamp_utc").unique(subset=["timestamp_utc"], keep="first")

    logger.info("Final RTE demand DataFrame: %d rows, %d columns", len(df), len(df.columns))
    return df


def _parse_one_year(zip_path: Path) -> pl.DataFrame:
    """Extract and parse one eCO2mix annual ZIP → (timestamp_utc, load_mw, tso_forecast_mw)."""
    with ZipFile(zip_path) as zf:
        xls_names = [n for n in zf.namelist() if n.endswith(".xls")]
        if not xls_names:
            raise ValueError(f"No .xls in {zip_path.name}")
        raw = zf.read(xls_names[0])

    # File is ISO-8859-1 TSV despite the .xls extension
    decoded = raw.decode("latin-1").encode("utf-8")

    raw_df = pl.read_csv(
        BytesIO(decoded),
        separator="\t",
        infer_schema_length=10_000,
        null_values=["ND", "-", "DC", ""],
        truncate_ragged_lines=True,
        ignore_errors=True,
    )

    # Drop trailer rows (warning + empty). They have null Date.
    raw_df = raw_df.filter(pl.col("Date").is_not_null() & pl.col("Heures").is_not_null())

    # Parse Date + Heures → naive datetime → Europe/Paris → UTC.
    # Date format: "YYYY-MM-DD" (ISO, NOT dd/mm/yyyy as the PDF claims).
    # Heures format: "HH:MM"
    df = raw_df.with_columns(
        (pl.col("Date").cast(pl.String) + pl.lit(" ") + pl.col("Heures").cast(pl.String))
        .str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M", strict=False)
        .dt.replace_time_zone(SOURCE_TZ, ambiguous="earliest", non_existent="null")
        .dt.convert_time_zone("UTC")
        .alias("timestamp_utc")
    )

    # Drop rows where timestamp parsing failed (spring DST non-existent hour)
    df = df.filter(pl.col("timestamp_utc").is_not_null())

    # Extract the columns we care about, rename
    df = df.select(
        pl.col("timestamp_utc"),
        pl.col("Consommation").cast(pl.Float64, strict=False).alias("load_mw"),
        pl.col("Prévision J-1").cast(pl.Float64, strict=False).alias("tso_forecast_mw"),
    )

    return df


def _resample_hourly(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate 30-min load + 15-min forecast to hourly mean.

    Polars ``group_by_dynamic`` produces a row for each hour bucket; null
    values (e.g., load_mw at :15 / :45) are skipped by the mean aggregator.
    """
    return (
        df.sort("timestamp_utc")
        .group_by_dynamic("timestamp_utc", every="1h", closed="left")
        .agg(
            pl.col("load_mw").mean(),
            pl.col("tso_forecast_mw").mean(),
        )
    )
