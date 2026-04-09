"""Tests for windcast.data.rte_france parser."""

from __future__ import annotations

import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import polars as pl
import pytest

from windcast.data.demand_schema import validate_demand_schema
from windcast.data.rte_france import (
    _parse_one_year,
    _resample_hourly,
    parse_rte_france,
)

MINIMAL_TSV_HEADER = "Périmètre\tNature\tDate\tHeures\tConsommation\tPrévision J-1\tPrévision J\n"


def _make_minimal_xls_bytes(rows: list[tuple[str, ...]]) -> bytes:
    """Build a fake eCO2mix annual TSV (Latin-1) with the minimal columns."""
    text = MINIMAL_TSV_HEADER
    for row in rows:
        text += "\t".join(row) + "\n"
    # Trailer warning line + empty row (mirrors real files)
    text += "RTE ne pourra être tenu responsable...\n\n"
    return text.encode("latin-1")


def _make_zip(xls_bytes: bytes, inner_name: str = "eCO2mix_fake.xls") -> bytes:
    """Wrap XLS bytes in an in-memory ZIP."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(inner_name, xls_bytes)
    return buf.getvalue()


def _write_fake_zip(tmp_path: Path, rows: list[tuple[str, ...]], year: int = 2023) -> Path:
    """Write a fake annual ZIP to tmp_path and return the path."""
    zip_path = tmp_path / f"eCO2mix_RTE_Annuel-Definitif_{year}.zip"
    zip_path.write_bytes(_make_zip(_make_minimal_xls_bytes(rows)))
    return zip_path


def test_parse_one_year_basic(tmp_path: Path):
    """Happy path: parse a few rows and verify columns + null handling."""
    rows = [
        ("France", "Données définitives", "2023-06-01", "00:00", "50000", "49500", "50500"),
        ("France", "Données définitives", "2023-06-01", "00:15", "", "49600", "50600"),
        ("France", "Données définitives", "2023-06-01", "00:30", "50100", "49700", "50700"),
        ("France", "Données définitives", "2023-06-01", "00:45", "", "49800", "50800"),
        ("France", "Données définitives", "2023-06-01", "01:00", "50200", "49900", "50900"),
    ]
    zip_path = _write_fake_zip(tmp_path, rows)
    df = _parse_one_year(zip_path)

    assert df.columns == ["timestamp_utc", "load_mw", "tso_forecast_mw"]
    # Trailer rows dropped by the null-Date filter
    assert len(df) == 5
    # Null load at :15 / :45 rows preserved
    assert df["load_mw"].drop_nulls().len() == 3
    assert df["tso_forecast_mw"].drop_nulls().len() == 5


def test_parse_one_year_drops_trailer(tmp_path: Path):
    """The RTE warning trailer line + empty row must be filtered out."""
    rows = [
        ("France", "Données définitives", "2023-06-01", "00:00", "50000", "49500", "50500"),
    ]
    zip_path = _write_fake_zip(tmp_path, rows)
    df = _parse_one_year(zip_path)
    # If trailer wasn't dropped, we'd see > 1 row here
    assert len(df) == 1


def test_parse_handles_nd_as_null(tmp_path: Path):
    """`ND` markers should be parsed as null."""
    rows = [
        ("France", "Données définitives", "2023-06-01", "00:00", "ND", "49500", "50500"),
        ("France", "Données définitives", "2023-06-01", "00:30", "50100", "ND", "50700"),
    ]
    zip_path = _write_fake_zip(tmp_path, rows)
    df = _parse_one_year(zip_path)
    # Row 0: load null, tso present. Row 1: load present, tso null.
    assert df["load_mw"].null_count() == 1
    assert df["tso_forecast_mw"].null_count() == 1


def test_resample_hourly_aggregates_to_hour():
    """15-min rows collapsed into a single hourly mean row."""
    df = pl.DataFrame(
        {
            "timestamp_utc": [
                datetime(2023, 6, 1, 0, 0),
                datetime(2023, 6, 1, 0, 15),
                datetime(2023, 6, 1, 0, 30),
                datetime(2023, 6, 1, 0, 45),
            ],
            "load_mw": [50000.0, None, 50100.0, None],
            "tso_forecast_mw": [49500.0, 49600.0, 49700.0, 49800.0],
        }
    ).with_columns(pl.col("timestamp_utc").dt.replace_time_zone("UTC"))

    result = _resample_hourly(df)
    assert len(result) == 1
    # Mean of (50000, 50100) ignoring nulls = 50050
    assert result["load_mw"][0] == pytest.approx(50050.0)
    # Mean of (49500, 49600, 49700, 49800) = 49650
    assert result["tso_forecast_mw"][0] == pytest.approx(49650.0)


def test_parse_rte_france_produces_canonical_schema(tmp_path: Path):
    """End-to-end parse yields a schema-compliant DataFrame."""
    rows = [
        ("France", "Données définitives", "2023-06-01", "00:00", "50000", "49500", "50500"),
        ("France", "Données définitives", "2023-06-01", "00:30", "50100", "49700", "50700"),
    ]
    _write_fake_zip(tmp_path, rows)

    df = parse_rte_france(tmp_path)

    errors = validate_demand_schema(df)
    assert errors == [], f"Schema validation failed: {errors}"
    assert df["dataset_id"][0] == "rte_france"
    assert df["zone_id"][0] == "FR"
    assert df["tso_forecast_mw"].drop_nulls().len() >= 1


def test_parse_rte_france_missing_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        parse_rte_france(tmp_path / "nonexistent")


def test_parse_rte_france_empty_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No annual definitive"):
        parse_rte_france(tmp_path)


def test_parse_rte_france_timezone_conversion(tmp_path: Path):
    """2023-06-01 00:00 Europe/Paris (CEST, +02:00) → 2023-05-31 22:00 UTC."""
    rows = [
        ("France", "Données définitives", "2023-06-01", "00:00", "50000", "49500", "50500"),
    ]
    _write_fake_zip(tmp_path, rows)
    df = parse_rte_france(tmp_path)
    first_ts = df["timestamp_utc"][0]
    assert first_ts.hour == 22  # type: ignore[union-attr]
    assert first_ts.day == 31  # type: ignore[union-attr]
    assert first_ts.month == 5  # type: ignore[union-attr]
