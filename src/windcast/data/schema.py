"""Canonical SCADA schema for wind farm time-series data."""

import polars as pl

SCADA_SCHEMA: dict[str, type[pl.DataType] | pl.DataType] = {
    "timestamp_utc": pl.Datetime("us", "UTC"),
    "dataset_id": pl.String,
    "turbine_id": pl.String,
    "active_power_kw": pl.Float64,
    "wind_speed_ms": pl.Float64,
    "wind_direction_deg": pl.Float64,
    "pitch_angle_deg": pl.Float64,
    "rotor_rpm": pl.Float64,
    "nacelle_direction_deg": pl.Float64,
    "ambient_temp_c": pl.Float64,
    "nacelle_temp_c": pl.Float64,
    "status_code": pl.Int32,
    "is_curtailed": pl.Boolean,
    "is_maintenance": pl.Boolean,
    "qc_flag": pl.UInt8,
}

SCADA_COLUMNS: list[str] = list(SCADA_SCHEMA.keys())

SIGNAL_COLUMNS: list[str] = [
    "active_power_kw",
    "wind_speed_ms",
    "wind_direction_deg",
    "pitch_angle_deg",
    "rotor_rpm",
    "nacelle_direction_deg",
    "ambient_temp_c",
    "nacelle_temp_c",
]

QC_OK: int = 0
QC_SUSPECT: int = 1
QC_BAD: int = 2


def validate_schema(
    df: pl.DataFrame,
    *,
    strict: bool = False,
) -> list[str]:
    """Validate DataFrame conforms to canonical SCADA schema.

    Args:
        df: DataFrame to validate.
        strict: If True, reject extra columns not in schema.

    Returns:
        List of error messages. Empty list = valid.
    """
    errors: list[str] = []
    actual = dict(zip(df.columns, df.dtypes, strict=True))

    for col, expected_dtype in SCADA_SCHEMA.items():
        if col not in actual:
            errors.append(f"Missing column: {col!r}")
        elif actual[col] != expected_dtype:
            errors.append(f"Column {col!r}: expected {expected_dtype}, got {actual[col]}")

    if strict:
        extra = set(actual) - set(SCADA_SCHEMA)
        errors.extend(f"Unexpected column: {col!r}" for col in sorted(extra))

    return errors


def empty_scada_frame() -> pl.DataFrame:
    """Create an empty DataFrame with the canonical SCADA schema. Useful for tests."""
    return pl.DataFrame(schema=SCADA_SCHEMA)
