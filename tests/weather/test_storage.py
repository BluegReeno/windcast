"""Tests for windcast.weather.storage module."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from windcast.weather.storage import WeatherStorage


def _make_weather_df(n_hours: int = 24, start_iso: str = "2021-01-01T00:00:00") -> pl.DataFrame:
    """Create a sample wide-format weather DataFrame."""
    start = datetime.fromisoformat(start_iso).replace(tzinfo=UTC)
    timestamps = pl.datetime_range(
        start=start,
        end=start,
        interval="1h",
        eager=True,
        closed="left",
    )
    # Generate the correct number of timestamps
    timestamps = pl.datetime_range(
        start=start,
        end=datetime(start.year, start.month, start.day, n_hours - 1, tzinfo=UTC),
        interval="1h",
        eager=True,
    )
    return pl.DataFrame(
        {
            "timestamp_utc": timestamps,
            "wind_speed_100m": [10.0 + i * 0.1 for i in range(len(timestamps))],
            "temperature_2m": [5.0 + i * 0.05 for i in range(len(timestamps))],
        }
    )


def test_upsert_and_query_roundtrip():
    """Insert weather data, query it back, verify values match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = WeatherStorage(Path(tmpdir) / "test.db")
        df = _make_weather_df(n_hours=5)
        count = storage.upsert("test_loc", df)
        assert count == 10  # 5 hours x 2 variables

        result = storage.query("test_loc", "2021-01-01", "2021-01-01")
        assert len(result) == 5
        assert "wind_speed_100m" in result.columns
        assert "temperature_2m" in result.columns
        storage.close()


def test_upsert_is_idempotent():
    """Inserting same data twice doesn't duplicate rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = WeatherStorage(Path(tmpdir) / "test.db")
        df = _make_weather_df(n_hours=3)
        storage.upsert("test_loc", df)
        storage.upsert("test_loc", df)

        result = storage.query("test_loc", "2021-01-01", "2021-01-01")
        assert len(result) == 3
        storage.close()


def test_query_empty_returns_empty_df():
    """Query on empty DB returns empty DataFrame."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = WeatherStorage(Path(tmpdir) / "test.db")
        result = storage.query("nonexistent", "2021-01-01", "2021-01-01")
        assert result.is_empty()
        assert "timestamp_utc" in result.columns
        storage.close()


def test_get_coverage_empty():
    """Coverage on empty DB returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = WeatherStorage(Path(tmpdir) / "test.db")
        assert storage.get_coverage("nonexistent") is None
        storage.close()


def test_get_coverage_returns_range():
    """Coverage returns (min, max) dates after insert."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = WeatherStorage(Path(tmpdir) / "test.db")
        df = _make_weather_df(n_hours=24)
        storage.upsert("test_loc", df)

        coverage = storage.get_coverage("test_loc")
        assert coverage is not None
        min_ts, _max_ts = coverage
        assert "2021-01-01" in min_ts
        storage.close()


def test_query_filters_by_date_range():
    """Query only returns rows within requested range."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = WeatherStorage(Path(tmpdir) / "test.db")

        # Insert 2 days of data
        df1 = _make_weather_df(n_hours=24, start_iso="2021-01-01T00:00:00")
        df2 = _make_weather_df(n_hours=24, start_iso="2021-01-02T00:00:00")
        storage.upsert("test_loc", df1)
        storage.upsert("test_loc", df2)

        # Query only first day
        result = storage.query("test_loc", "2021-01-01", "2021-01-01")
        assert len(result) == 24

        # Query both days
        result_both = storage.query("test_loc", "2021-01-01", "2021-01-02")
        assert len(result_both) == 48
        storage.close()
