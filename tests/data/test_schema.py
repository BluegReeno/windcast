"""Tests for windcast.data.schema module."""

import polars as pl

from windcast.data.schema import (
    QC_BAD,
    QC_OK,
    QC_SUSPECT,
    SCADA_COLUMNS,
    SCADA_SCHEMA,
    empty_scada_frame,
    validate_schema,
)


def test_schema_has_expected_columns():
    """Schema defines all 15 canonical columns."""
    assert len(SCADA_SCHEMA) == 15
    assert "timestamp_utc" in SCADA_SCHEMA
    assert "active_power_kw" in SCADA_SCHEMA
    assert "qc_flag" in SCADA_SCHEMA


def test_empty_frame_matches_schema():
    """empty_scada_frame produces a valid schema-compliant DataFrame."""
    df = empty_scada_frame()
    errors = validate_schema(df)
    assert errors == []
    assert df.shape == (0, 15)


def test_validate_schema_detects_missing_column():
    """Validation catches missing columns."""
    df = pl.DataFrame({"timestamp_utc": [], "dataset_id": []})
    errors = validate_schema(df)
    assert len(errors) > 0
    assert any("Missing column" in e for e in errors)


def test_validate_schema_detects_wrong_type():
    """Validation catches wrong column types."""
    df = empty_scada_frame().cast({"active_power_kw": pl.Int32})
    errors = validate_schema(df)
    assert any("active_power_kw" in e for e in errors)


def test_validate_schema_strict_mode():
    """Strict mode catches extra columns."""
    df = empty_scada_frame().with_columns(pl.lit("extra").alias("bonus_col"))
    errors = validate_schema(df, strict=True)
    assert any("Unexpected column" in e for e in errors)


def test_scada_columns_ordered():
    """SCADA_COLUMNS matches schema keys in order."""
    assert list(SCADA_SCHEMA.keys()) == SCADA_COLUMNS


def test_qc_flag_constants():
    """QC flag constants are correctly defined."""
    assert QC_OK == 0
    assert QC_SUSPECT == 1
    assert QC_BAD == 2
