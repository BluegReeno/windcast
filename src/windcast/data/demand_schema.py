"""Canonical schema for demand forecasting time-series data."""

import polars as pl

from windcast.data.schema import QC_BAD, QC_OK, QC_SUSPECT

__all__ = [
    "DEMAND_COLUMNS",
    "DEMAND_SCHEMA",
    "DEMAND_SIGNAL_COLUMNS",
    "QC_BAD",
    "QC_OK",
    "QC_SUSPECT",
    "empty_demand_frame",
    "validate_demand_schema",
]

DEMAND_SCHEMA: dict[str, type[pl.DataType] | pl.DataType] = {
    "timestamp_utc": pl.Datetime("us", "UTC"),
    "dataset_id": pl.String,
    "zone_id": pl.String,
    "load_mw": pl.Float64,
    "temperature_c": pl.Float64,
    "wind_speed_ms": pl.Float64,
    "humidity_pct": pl.Float64,
    "price_eur_mwh": pl.Float64,
    "is_holiday": pl.Boolean,
    "is_dst_transition": pl.Boolean,
    "qc_flag": pl.UInt8,
}

DEMAND_COLUMNS: list[str] = list(DEMAND_SCHEMA.keys())

DEMAND_SIGNAL_COLUMNS: list[str] = [
    "load_mw",
    "temperature_c",
    "wind_speed_ms",
    "humidity_pct",
    "price_eur_mwh",
]


def validate_demand_schema(
    df: pl.DataFrame,
    *,
    strict: bool = False,
) -> list[str]:
    """Validate DataFrame conforms to canonical demand schema.

    Args:
        df: DataFrame to validate.
        strict: If True, reject extra columns not in schema.

    Returns:
        List of error messages. Empty list = valid.
    """
    errors: list[str] = []
    actual = dict(zip(df.columns, df.dtypes, strict=True))

    for col, expected_dtype in DEMAND_SCHEMA.items():
        if col not in actual:
            errors.append(f"Missing column: {col!r}")
        elif actual[col] != expected_dtype:
            errors.append(f"Column {col!r}: expected {expected_dtype}, got {actual[col]}")

    if strict:
        extra = set(actual) - set(DEMAND_SCHEMA)
        errors.extend(f"Unexpected column: {col!r}" for col in sorted(extra))

    return errors


def empty_demand_frame() -> pl.DataFrame:
    """Create an empty DataFrame with the canonical demand schema. Useful for tests."""
    return pl.DataFrame(schema=DEMAND_SCHEMA)
