"""Canonical schema for solar PV generation time-series data."""

import polars as pl

from windcast.data.schema import QC_BAD, QC_OK, QC_SUSPECT

__all__ = [
    "QC_BAD",
    "QC_OK",
    "QC_SUSPECT",
    "SOLAR_COLUMNS",
    "SOLAR_SCHEMA",
    "SOLAR_SIGNAL_COLUMNS",
    "empty_solar_frame",
    "validate_solar_schema",
]

SOLAR_SCHEMA: dict[str, type[pl.DataType] | pl.DataType] = {
    "timestamp_utc": pl.Datetime("us", "UTC"),
    "dataset_id": pl.String,
    "system_id": pl.String,
    "power_kw": pl.Float64,
    "ghi_wm2": pl.Float64,
    "poa_wm2": pl.Float64,
    "ambient_temp_c": pl.Float64,
    "module_temp_c": pl.Float64,
    "wind_speed_ms": pl.Float64,
    "qc_flag": pl.UInt8,
}

SOLAR_COLUMNS: list[str] = list(SOLAR_SCHEMA.keys())

SOLAR_SIGNAL_COLUMNS: list[str] = [
    "power_kw",
    "ghi_wm2",
    "poa_wm2",
    "ambient_temp_c",
    "module_temp_c",
    "wind_speed_ms",
]


def validate_solar_schema(
    df: pl.DataFrame,
    *,
    strict: bool = False,
) -> list[str]:
    """Validate DataFrame conforms to canonical solar schema.

    Args:
        df: DataFrame to validate.
        strict: If True, reject extra columns not in schema.

    Returns:
        List of error messages. Empty list = valid.
    """
    errors: list[str] = []
    actual = dict(zip(df.columns, df.dtypes, strict=True))

    for col, expected_dtype in SOLAR_SCHEMA.items():
        if col not in actual:
            errors.append(f"Missing column: {col!r}")
        elif actual[col] != expected_dtype:
            errors.append(f"Column {col!r}: expected {expected_dtype}, got {actual[col]}")

    if strict:
        extra = set(actual) - set(SOLAR_SCHEMA)
        errors.extend(f"Unexpected column: {col!r}" for col in sorted(extra))

    return errors


def empty_solar_frame() -> pl.DataFrame:
    """Create an empty DataFrame with the canonical solar schema. Useful for tests."""
    return pl.DataFrame(schema=SOLAR_SCHEMA)
