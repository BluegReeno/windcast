"""Tests for weather source selection + blend mode in scripts/build_features.

The blend helper lives in ``windcast.weather`` — this test exercises it
end-to-end with two SQLite caches (one ERA5 stub, one forecast stub) and
verifies that the cutoff-based split preserves each source's values.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from windcast.weather import load_blended_weather, load_weather_from_db
from windcast.weather.storage import WeatherStorage


def _make_hourly_df(
    start: datetime,
    n_hours: int,
    temperature: float,
) -> pl.DataFrame:
    """Build a wide-format weather DataFrame with constant temperature."""
    timestamps = pl.datetime_range(
        start=start,
        end=start + timedelta(hours=n_hours - 1),
        interval="1h",
        eager=True,
    )
    return pl.DataFrame(
        {
            "timestamp_utc": timestamps,
            "wind_speed_100m": [5.0] * len(timestamps),
            "temperature_2m": [temperature] * len(timestamps),
        }
    )


def test_load_weather_from_db_returns_none_when_empty():
    """Empty db yields None — caller logs and skips NWP features."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "empty.db"
        # Create empty db
        WeatherStorage(db_path).close()
        result = load_weather_from_db("kelmarsh", db_path)
        assert result is None


def test_load_blended_weather_splits_at_cutoff():
    """Blend must use ERA5 before cutoff and forecast at/after cutoff.

    We write distinguishable values (different temperatures) into each db so
    the blend output is self-verifying: pre-cutoff rows must carry the ERA5
    temperature, post-cutoff rows must carry the forecast temperature.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        era5_db = Path(tmpdir) / "weather.db"
        forecast_db = Path(tmpdir) / "weather_forecast.db"

        # ERA5 covers 2021-12-31 00:00 → 2022-01-01 23:00 with temp = -5.0
        era5_store = WeatherStorage(era5_db)
        era5_df = _make_hourly_df(
            start=datetime(2021, 12, 31, 0, tzinfo=UTC),
            n_hours=48,
            temperature=-5.0,
        )
        loc_key = "52.4016_-0.9436"  # Kelmarsh config
        era5_store.upsert(loc_key, era5_df)
        era5_store.close()

        # Forecast covers the same range with temp = +10.0
        forecast_store = WeatherStorage(forecast_db)
        forecast_df = _make_hourly_df(
            start=datetime(2021, 12, 31, 0, tzinfo=UTC),
            n_hours=48,
            temperature=10.0,
        )
        forecast_store.upsert(loc_key, forecast_df)
        forecast_store.close()

        blended = load_blended_weather(
            "kelmarsh",
            era5_db=era5_db,
            forecast_db=forecast_db,
            cutoff="2022-01-01",
        )

        assert blended is not None
        # Expected: 24 rows from ERA5 (2021-12-31) + 24 from forecast (2022-01-01)
        assert len(blended) == 48

        cutoff_dt = datetime(2022, 1, 1, tzinfo=UTC)
        pre = blended.filter(pl.col("timestamp_utc") < cutoff_dt)
        post = blended.filter(pl.col("timestamp_utc") >= cutoff_dt)

        assert len(pre) == 24
        assert len(post) == 24

        # ERA5 values pre-cutoff
        assert pre["temperature_2m"].to_list() == [-5.0] * 24
        # Forecast values post-cutoff
        assert post["temperature_2m"].to_list() == [10.0] * 24


def test_load_blended_weather_handles_missing_era5():
    """If ERA5 db is empty, blend falls back to forecast db with a warning."""
    with tempfile.TemporaryDirectory() as tmpdir:
        era5_db = Path(tmpdir) / "weather.db"
        forecast_db = Path(tmpdir) / "weather_forecast.db"

        WeatherStorage(era5_db).close()  # empty

        forecast_store = WeatherStorage(forecast_db)
        forecast_df = _make_hourly_df(
            start=datetime(2022, 6, 1, 0, tzinfo=UTC),
            n_hours=6,
            temperature=20.0,
        )
        forecast_store.upsert("52.4016_-0.9436", forecast_df)
        forecast_store.close()

        blended = load_blended_weather(
            "kelmarsh",
            era5_db=era5_db,
            forecast_db=forecast_db,
            cutoff="2022-01-01",
        )

        assert blended is not None
        assert len(blended) == 6
        assert blended["temperature_2m"].to_list() == [20.0] * 6


def test_load_blended_weather_returns_none_when_both_empty():
    """Both dbs empty → None — caller must handle gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        era5_db = Path(tmpdir) / "weather.db"
        forecast_db = Path(tmpdir) / "weather_forecast.db"
        WeatherStorage(era5_db).close()
        WeatherStorage(forecast_db).close()

        result = load_blended_weather(
            "kelmarsh",
            era5_db=era5_db,
            forecast_db=forecast_db,
            cutoff="2022-01-01",
        )
        assert result is None
