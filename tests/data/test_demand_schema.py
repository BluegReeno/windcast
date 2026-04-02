"""Tests for windcast.data.demand_schema module."""

import polars as pl

from windcast.data.demand_schema import (
    DEMAND_COLUMNS,
    DEMAND_SCHEMA,
    DEMAND_SIGNAL_COLUMNS,
    empty_demand_frame,
    validate_demand_schema,
)


def test_demand_schema_has_expected_columns():
    """Schema defines all 11 canonical columns."""
    assert len(DEMAND_SCHEMA) == 11
    assert "timestamp_utc" in DEMAND_SCHEMA
    assert "load_mw" in DEMAND_SCHEMA
    assert "qc_flag" in DEMAND_SCHEMA


def test_empty_demand_frame_matches_schema():
    """empty_demand_frame produces a valid schema-compliant DataFrame."""
    df = empty_demand_frame()
    errors = validate_demand_schema(df)
    assert errors == []
    assert df.shape == (0, 11)


def test_validate_demand_schema_detects_missing_column():
    """Validation catches missing columns."""
    df = pl.DataFrame({"timestamp_utc": [], "dataset_id": []})
    errors = validate_demand_schema(df)
    assert len(errors) > 0
    assert any("Missing column" in e for e in errors)


def test_validate_demand_schema_detects_wrong_type():
    """Validation catches wrong column types."""
    df = empty_demand_frame().cast({"load_mw": pl.Int32})
    errors = validate_demand_schema(df)
    assert any("load_mw" in e for e in errors)


def test_validate_demand_schema_strict_mode():
    """Strict mode catches extra columns."""
    df = empty_demand_frame().with_columns(pl.lit("extra").alias("bonus_col"))
    errors = validate_demand_schema(df, strict=True)
    assert any("Unexpected column" in e for e in errors)


def test_demand_columns_ordered():
    """DEMAND_COLUMNS matches schema keys in order."""
    assert list(DEMAND_SCHEMA.keys()) == DEMAND_COLUMNS


def test_demand_signal_columns():
    """Signal columns are the 5 numeric measurement columns."""
    assert len(DEMAND_SIGNAL_COLUMNS) == 5
    assert "load_mw" in DEMAND_SIGNAL_COLUMNS
    assert "temperature_c" in DEMAND_SIGNAL_COLUMNS
    assert "wind_speed_ms" in DEMAND_SIGNAL_COLUMNS
    assert "humidity_pct" in DEMAND_SIGNAL_COLUMNS
    assert "price_eur_mwh" in DEMAND_SIGNAL_COLUMNS
