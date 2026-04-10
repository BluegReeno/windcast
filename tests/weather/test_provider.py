"""Tests for windcast.weather.provider module."""

from unittest.mock import MagicMock

import numpy as np
import polars as pl

from windcast.weather.provider import (
    HistoricalForecastProvider,
    OpenMeteoProvider,
    WeatherProvider,
)


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


# ---- HistoricalForecastProvider ---------------------------------


def test_historical_forecast_provider_satisfies_protocol():
    """HistoricalForecastProvider must satisfy WeatherProvider protocol."""
    provider = HistoricalForecastProvider(cache_dir=".test_cache_hf")
    assert isinstance(provider, WeatherProvider)


def test_archive_and_forecast_providers_are_distinct_instances():
    """No accidental shared state between the two providers."""
    p1 = OpenMeteoProvider(cache_dir=".test_cache_era5")
    p2 = HistoricalForecastProvider(cache_dir=".test_cache_hf")
    assert p1 is not p2
    assert type(p1).__name__ != type(p2).__name__


def test_historical_forecast_provider_returns_dataframe():
    """Mock the underlying client and verify DataFrame output with forecast URL."""
    from windcast.data.open_meteo import HISTORICAL_FORECAST_URL

    n_hours = 12
    variables = ["temperature_2m", "wind_speed_10m"]

    mock_hourly_var = MagicMock()
    mock_hourly_var.ValuesAsNumpy.return_value = np.random.rand(n_hours).astype(np.float32)

    mock_hourly = MagicMock()
    mock_hourly.Time.return_value = 1640995200  # 2022-01-01 00:00 UTC
    mock_hourly.TimeEnd.return_value = 1640995200 + 3600 * n_hours
    mock_hourly.Interval.return_value = 3600
    mock_hourly.Variables.return_value = mock_hourly_var

    mock_response = MagicMock()
    mock_response.Hourly.return_value = mock_hourly

    mock_client = MagicMock()
    mock_client.weather_api.return_value = [mock_response]

    provider = HistoricalForecastProvider.__new__(HistoricalForecastProvider)
    provider._client = mock_client

    df = provider.fetch(
        latitude=48.85,
        longitude=2.35,
        start_date="2022-01-01",
        end_date="2022-01-01",
        variables=variables,
    )

    assert isinstance(df, pl.DataFrame)
    assert "timestamp_utc" in df.columns
    assert len(df) == n_hours
    for var in variables:
        assert var in df.columns

    # Must hit forecast URL, not archive URL
    called_url = mock_client.weather_api.call_args[0][0]
    assert called_url == HISTORICAL_FORECAST_URL
