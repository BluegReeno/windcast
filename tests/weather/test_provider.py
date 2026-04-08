"""Tests for windcast.weather.provider module."""

from unittest.mock import MagicMock

import numpy as np
import polars as pl

from windcast.weather.provider import OpenMeteoProvider, WeatherProvider


def test_open_meteo_provider_implements_protocol():
    """Verify OpenMeteoProvider satisfies WeatherProvider protocol."""
    # Protocol compliance via runtime_checkable
    assert isinstance(OpenMeteoProvider(cache_dir=".test_cache"), WeatherProvider)


def test_open_meteo_provider_returns_dataframe():
    """Mock the underlying client and verify DataFrame output."""
    n_hours = 24
    variables = ["wind_speed_100m", "temperature_2m"]

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

    provider = OpenMeteoProvider.__new__(OpenMeteoProvider)
    provider._client = mock_client

    df = provider.fetch(
        latitude=52.4,
        longitude=-0.9,
        start_date="2021-01-01",
        end_date="2021-01-01",
        variables=variables,
    )

    assert isinstance(df, pl.DataFrame)
    assert "timestamp_utc" in df.columns
    assert len(df) == n_hours
    for var in variables:
        assert var in df.columns
