"""Tests for windcast.data.open_meteo module."""

from unittest.mock import MagicMock

import numpy as np
import polars as pl

from windcast.data.open_meteo import (
    ARCHIVE_URL,
    HISTORICAL_FORECAST_URL,
    WIND_VARIABLES,
    build_client,
    fetch_historical_forecast_weather,
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


# ---- fetch_historical_forecast_weather ----------------------------------


def test_historical_forecast_url_is_not_archive_url():
    """Regression guard: the forecast URL must differ from the ERA5 archive URL.

    Context: the ERA5 leak incident — `archive-api.open-meteo.com` returns
    reanalysis (observed truth), not forecasts. If this assertion ever starts
    failing, the fix is to revert whoever aliased the constants.
    """
    assert HISTORICAL_FORECAST_URL != ARCHIVE_URL
    assert "historical-forecast-api" in HISTORICAL_FORECAST_URL
    assert "archive" not in HISTORICAL_FORECAST_URL


def test_fetch_historical_forecast_returns_polars_schema():
    """fetch_historical_forecast_weather returns Polars DataFrame with timestamp + variables."""
    n_hours = 24
    variables = ["wind_speed_100m", "temperature_2m"]

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

    df = fetch_historical_forecast_weather(
        latitude=52.4,
        longitude=-0.9,
        start_date="2022-01-01",
        end_date="2022-01-01",
        variables=variables,
        client=mock_client,
    )

    assert isinstance(df, pl.DataFrame)
    assert "timestamp_utc" in df.columns
    assert df["timestamp_utc"].dtype == pl.Datetime("us", "UTC")
    assert len(df) == n_hours
    for var in variables:
        assert var in df.columns


def test_fetch_historical_forecast_calls_forecast_url():
    """Verify the forecast function hits HISTORICAL_FORECAST_URL, not the archive."""
    mock_hourly_var = MagicMock()
    mock_hourly_var.ValuesAsNumpy.return_value = np.zeros(1, dtype=np.float32)
    mock_hourly = MagicMock()
    mock_hourly.Time.return_value = 1640995200
    mock_hourly.TimeEnd.return_value = 1640995200 + 3600
    mock_hourly.Interval.return_value = 3600
    mock_hourly.Variables.return_value = mock_hourly_var
    mock_response = MagicMock()
    mock_response.Hourly.return_value = mock_hourly
    mock_client = MagicMock()
    mock_client.weather_api.return_value = [mock_response]

    fetch_historical_forecast_weather(
        latitude=52.4,
        longitude=-0.9,
        start_date="2022-01-01",
        end_date="2022-01-01",
        variables=["temperature_2m"],
        client=mock_client,
    )

    called_url = mock_client.weather_api.call_args[0][0]
    assert called_url == HISTORICAL_FORECAST_URL
    assert called_url != ARCHIVE_URL


def test_fetch_historical_forecast_respects_date_range():
    """start_date and end_date are passed through to the client params."""
    mock_hourly_var = MagicMock()
    mock_hourly_var.ValuesAsNumpy.return_value = np.zeros(2, dtype=np.float32)
    mock_hourly = MagicMock()
    mock_hourly.Time.return_value = 1640995200
    mock_hourly.TimeEnd.return_value = 1640995200 + 7200
    mock_hourly.Interval.return_value = 3600
    mock_hourly.Variables.return_value = mock_hourly_var
    mock_response = MagicMock()
    mock_response.Hourly.return_value = mock_hourly
    mock_client = MagicMock()
    mock_client.weather_api.return_value = [mock_response]

    fetch_historical_forecast_weather(
        latitude=48.85,
        longitude=2.35,
        start_date="2022-03-15",
        end_date="2022-03-16",
        variables=["temperature_2m"],
        client=mock_client,
    )

    params = mock_client.weather_api.call_args[1]["params"]
    assert params["start_date"] == "2022-03-15"
    assert params["end_date"] == "2022-03-16"
    assert params["latitude"] == 48.85
    assert params["longitude"] == 2.35
