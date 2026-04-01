"""Tests for windcast.data.open_meteo module."""

from unittest.mock import MagicMock

import numpy as np
import polars as pl

from windcast.data.open_meteo import (
    ARCHIVE_URL,
    WIND_VARIABLES,
    build_client,
    fetch_historical_weather,
)


def test_wind_variables_defined():
    """Wind variables list has expected length."""
    assert len(WIND_VARIABLES) == 6
    assert "wind_speed_100m" in WIND_VARIABLES
    assert "temperature_2m" in WIND_VARIABLES


def test_archive_url_set():
    """Archive URL points to historical API."""
    assert "archive-api" in ARCHIVE_URL


def test_build_client_returns_client():
    """build_client returns an openmeteo_requests.Client."""
    import openmeteo_requests

    client = build_client(cache_dir=".test_cache")
    assert isinstance(client, openmeteo_requests.Client)


def test_fetch_returns_polars_dataframe():
    """fetch_historical_weather returns Polars DataFrame with expected columns."""
    n_hours = 24
    mock_hourly_var = MagicMock()
    mock_hourly_var.ValuesAsNumpy.return_value = np.random.rand(n_hours).astype(np.float32)

    mock_hourly = MagicMock()
    mock_hourly.Time.return_value = 1609459200  # 2021-01-01 00:00 UTC
    mock_hourly.TimeEnd.return_value = 1609459200 + 3600 * n_hours
    mock_hourly.Interval.return_value = 3600
    mock_hourly.Variables.return_value = mock_hourly_var

    mock_response = MagicMock()
    mock_response.Hourly.return_value = mock_hourly

    mock_client = MagicMock()
    mock_client.weather_api.return_value = [mock_response]

    df = fetch_historical_weather(
        latitude=52.4,
        longitude=-0.9,
        start_date="2021-01-01",
        end_date="2021-01-01",
        client=mock_client,
    )

    assert isinstance(df, pl.DataFrame)
    assert "timestamp_utc" in df.columns
    assert len(df) == n_hours
    # All wind variables should be present
    for var in WIND_VARIABLES:
        assert var in df.columns


def test_fetch_timestamps_are_utc():
    """Verify timestamps in result have UTC timezone."""
    n_hours = 5
    mock_hourly_var = MagicMock()
    mock_hourly_var.ValuesAsNumpy.return_value = np.zeros(n_hours, dtype=np.float32)

    mock_hourly = MagicMock()
    mock_hourly.Time.return_value = 1609459200
    mock_hourly.TimeEnd.return_value = 1609459200 + 3600 * n_hours
    mock_hourly.Interval.return_value = 3600
    mock_hourly.Variables.return_value = mock_hourly_var

    mock_response = MagicMock()
    mock_response.Hourly.return_value = mock_hourly

    mock_client = MagicMock()
    mock_client.weather_api.return_value = [mock_response]

    df = fetch_historical_weather(
        latitude=52.4,
        longitude=-0.9,
        start_date="2021-01-01",
        end_date="2021-01-01",
        client=mock_client,
    )

    assert df["timestamp_utc"].dtype == pl.Datetime("us", "UTC")
