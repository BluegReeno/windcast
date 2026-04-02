"""Tests for windcast.data.solar_schema module."""

import polars as pl

from windcast.data.solar_schema import (
    SOLAR_COLUMNS,
    SOLAR_SCHEMA,
    SOLAR_SIGNAL_COLUMNS,
    empty_solar_frame,
    validate_solar_schema,
)


def test_solar_schema_has_expected_columns():
    """Schema defines all 10 canonical columns."""
    assert len(SOLAR_SCHEMA) == 10
    assert "timestamp_utc" in SOLAR_SCHEMA
    assert "power_kw" in SOLAR_SCHEMA
    assert "qc_flag" in SOLAR_SCHEMA


def test_empty_solar_frame_matches_schema():
    """empty_solar_frame produces a valid schema-compliant DataFrame."""
    df = empty_solar_frame()
    errors = validate_solar_schema(df)
    assert errors == []
    assert df.shape == (0, 10)


def test_validate_solar_schema_detects_missing_column():
    """Validation catches missing columns."""
    df = pl.DataFrame({"timestamp_utc": [], "dataset_id": []})
    errors = validate_solar_schema(df)
    assert len(errors) > 0
    assert any("Missing column" in e for e in errors)


def test_validate_solar_schema_detects_wrong_type():
    """Validation catches wrong column types."""
    df = empty_solar_frame().cast({"power_kw": pl.Int32})
    errors = validate_solar_schema(df)
    assert any("power_kw" in e for e in errors)


def test_validate_solar_schema_strict_mode():
    """Strict mode catches extra columns."""
    df = empty_solar_frame().with_columns(pl.lit("extra").alias("bonus_col"))
    errors = validate_solar_schema(df, strict=True)
    assert any("Unexpected column" in e for e in errors)


def test_solar_columns_ordered():
    """SOLAR_COLUMNS matches schema keys in order."""
    assert list(SOLAR_SCHEMA.keys()) == SOLAR_COLUMNS


def test_solar_signal_columns():
    """Signal columns are the 6 numeric measurement columns."""
    assert len(SOLAR_SIGNAL_COLUMNS) == 6
    assert "power_kw" in SOLAR_SIGNAL_COLUMNS
    assert "ghi_wm2" in SOLAR_SIGNAL_COLUMNS
    assert "poa_wm2" in SOLAR_SIGNAL_COLUMNS
    assert "ambient_temp_c" in SOLAR_SIGNAL_COLUMNS
    assert "module_temp_c" in SOLAR_SIGNAL_COLUMNS
    assert "wind_speed_ms" in SOLAR_SIGNAL_COLUMNS
