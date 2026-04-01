"""Kelmarsh v4 dataset parser — ZIP archives to canonical SCADA schema."""

import io
import logging
import re
import zipfile
from pathlib import Path

import polars as pl

from windcast.data.schema import SCADA_SCHEMA

logger = logging.getLogger(__name__)

# Greenbyte CSV headers → canonical column names
KELMARSH_SIGNAL_MAP: dict[str, str] = {
    "# Date and time": "timestamp_raw",
    "Wind speed (m/s)": "wind_speed_ms",
    "Wind direction (°)": "wind_direction_deg",
    "Power (kW)": "active_power_kw",
    "Nacelle position (°)": "nacelle_direction_deg",
    "Rotor speed (rpm)": "rotor_rpm",
    "Blade angle (pitch position) A (°)": "pitch_a_deg",
    "Blade angle (pitch position) B (°)": "pitch_b_deg",
    "Blade angle (pitch position) C (°)": "pitch_c_deg",
    "Nacelle ambient temperature (°C)": "ambient_temp_c",
    "Nacelle temperature (°C)": "nacelle_temp_c",
}

KELMARSH_COLUMNS: list[str] = list(KELMARSH_SIGNAL_MAP.keys())
DATASET_ID = "kelmarsh"
TURBINE_IDS = [f"KWF{i}" for i in range(1, 7)]

_TURBINE_NUM_RE = re.compile(r"Kelmarsh_(\d+)")


def parse_kelmarsh(raw_path: Path) -> pl.DataFrame:
    """Parse Kelmarsh v4 ZIP archive into canonical SCADA DataFrame.

    Handles both nested ZIPs (outer ZIP containing annual ZIPs) and
    a directory of extracted CSVs.

    Args:
        raw_path: Path to the Kelmarsh ZIP archive or directory of CSVs.

    Returns:
        DataFrame conforming to SCADA_SCHEMA (before QC — qc_flag/is_curtailed/is_maintenance
        are initialized to defaults).
    """
    if raw_path.is_dir():
        return _parse_from_directory(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Kelmarsh data not found: {raw_path}")
    return _parse_from_zip(raw_path)


def _parse_from_directory(directory: Path) -> pl.DataFrame:
    """Parse Kelmarsh data from a directory of extracted CSVs."""
    csv_files = sorted(directory.glob("**/Turbine_Data_Kelmarsh_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No Turbine_Data CSV files found in {directory}")

    frames: list[pl.DataFrame] = []
    for csv_path in csv_files:
        turbine_id = _extract_turbine_id(csv_path.name)
        logger.info("Parsing %s (turbine %s)", csv_path.name, turbine_id)
        csv_bytes = csv_path.read_bytes()
        df = _read_turbine_csv(csv_bytes, turbine_id)
        frames.append(df)

    return pl.concat(frames).sort("timestamp_utc", "turbine_id")


def _parse_from_zip(zip_path: Path) -> pl.DataFrame:
    """Parse Kelmarsh data from a (potentially nested) ZIP archive."""
    frames: list[pl.DataFrame] = []

    with zipfile.ZipFile(zip_path, "r") as outer_zip:
        for entry in outer_zip.namelist():
            if entry.endswith(".zip"):
                # Nested annual ZIP
                logger.info("Opening inner archive: %s", entry)
                inner_bytes = outer_zip.read(entry)
                with zipfile.ZipFile(io.BytesIO(inner_bytes), "r") as inner_zip:
                    frames.extend(_parse_zip_contents(inner_zip))
            elif _is_turbine_csv(entry):
                # Direct CSV in outer ZIP
                turbine_id = _extract_turbine_id(entry)
                logger.info("Parsing %s (turbine %s)", entry, turbine_id)
                csv_bytes = outer_zip.read(entry)
                df = _read_turbine_csv(csv_bytes, turbine_id)
                frames.append(df)

    if not frames:
        raise ValueError(f"No turbine data CSVs found in {zip_path}")

    return pl.concat(frames).sort("timestamp_utc", "turbine_id")


def _parse_zip_contents(zf: zipfile.ZipFile) -> list[pl.DataFrame]:
    """Extract and parse all turbine CSVs from a ZIP archive."""
    frames: list[pl.DataFrame] = []
    for entry in zf.namelist():
        if _is_turbine_csv(entry):
            turbine_id = _extract_turbine_id(entry)
            logger.info("Parsing %s (turbine %s)", entry, turbine_id)
            csv_bytes = zf.read(entry)
            df = _read_turbine_csv(csv_bytes, turbine_id)
            frames.append(df)
    return frames


def _is_turbine_csv(filename: str) -> bool:
    """Check if a filename is a Kelmarsh turbine data CSV."""
    basename = filename.rsplit("/", 1)[-1] if "/" in filename else filename
    return basename.startswith("Turbine_Data_Kelmarsh_") and basename.endswith(".csv")


def _extract_turbine_id(filename: str) -> str:
    """Extract turbine ID from Kelmarsh filename.

    Example: 'Turbine_Data_Kelmarsh_1_2020-01-01_-_2020-12-31_1234.csv' → 'KWF1'
    """
    match = _TURBINE_NUM_RE.search(filename)
    if not match:
        raise ValueError(f"Cannot extract turbine number from filename: {filename}")
    return f"KWF{match.group(1)}"


def _read_turbine_csv(csv_bytes: bytes, turbine_id: str) -> pl.DataFrame:
    """Read a single Kelmarsh turbine CSV and map to canonical columns.

    Handles column selection, renaming, pitch averaging, timestamp parsing,
    and default flag initialization.
    """
    df = pl.read_csv(
        io.BytesIO(csv_bytes),
        infer_schema_length=10_000,
        null_values=["", "NA", "NaN", "N/A", "-999", "-9999"],
    )

    # Verify expected columns exist
    available = set(df.columns)
    mapped_cols = [c for c in KELMARSH_SIGNAL_MAP if c in available]
    missing = [c for c in KELMARSH_SIGNAL_MAP if c not in available]
    if missing:
        logger.warning(
            "Turbine %s: missing columns %s. Available: %s",
            turbine_id,
            missing,
            sorted(available)[:20],
        )

    # Select and rename mapped columns
    df = df.select(mapped_cols).rename({k: KELMARSH_SIGNAL_MAP[k] for k in mapped_cols})

    # Average pitch angles A/B/C → single pitch_angle_deg
    pitch_cols = [c for c in ["pitch_a_deg", "pitch_b_deg", "pitch_c_deg"] if c in df.columns]
    if pitch_cols:
        df = df.with_columns(
            pl.mean_horizontal(*[pl.col(c) for c in pitch_cols]).alias("pitch_angle_deg")
        ).drop(pitch_cols)
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("pitch_angle_deg"))

    # Parse timestamp → UTC
    if "timestamp_raw" in df.columns:
        df = df.with_columns(
            pl.col("timestamp_raw")
            .str.to_datetime("%Y-%m-%d %H:%M:%S")
            .dt.replace_time_zone("UTC")
            .alias("timestamp_utc")
        ).drop("timestamp_raw")
    else:
        raise ValueError(f"No timestamp column found for turbine {turbine_id}")

    # Check power unit: if max << rated (2050), likely kWh per 10-min → multiply by 6
    max_power = float(df["active_power_kw"].max())  # type: ignore[arg-type]
    if max_power < 500:
        logger.warning(
            "Turbine %s: max power = %.1f, likely kWh per 10-min. Converting to kW (*6).",
            turbine_id,
            max_power,
        )
        df = df.with_columns((pl.col("active_power_kw") * 6).alias("active_power_kw"))

    # Add identifier columns
    df = df.with_columns(
        pl.lit(DATASET_ID).alias("dataset_id"),
        pl.lit(turbine_id).alias("turbine_id"),
    )

    # Initialize QC/flag columns with defaults
    df = df.with_columns(
        pl.lit(0).cast(pl.Int32).alias("status_code"),
        pl.lit(False).alias("is_curtailed"),
        pl.lit(False).alias("is_maintenance"),
        pl.lit(0).cast(pl.UInt8).alias("qc_flag"),
    )

    # Ensure all schema columns exist, adding nulls for any missing
    for col, dtype in SCADA_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

    # Cast to schema dtypes and reorder
    df = df.select([pl.col(col).cast(dtype) for col, dtype in SCADA_SCHEMA.items()])

    return df
