"""Weather data providers — protocol and implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl

from windcast.data.open_meteo import (
    build_client,
    fetch_historical_forecast_weather,
    fetch_historical_weather,
)


@runtime_checkable
class WeatherProvider(Protocol):
    """Protocol for weather data providers."""

    def fetch(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        variables: list[str],
    ) -> pl.DataFrame:
        """Fetch hourly weather data.

        Returns:
            DataFrame with timestamp_utc + variable columns.
        """
        ...


class OpenMeteoProvider:
    """Open-Meteo Archive API provider — ERA5 reanalysis (observed weather).

    Use for *training-era* features (pre-2022) or for any workflow that wants
    the best available ground-truth weather. Do NOT use for val/test horizons
    where honest forecast-time features are required — see
    :class:`HistoricalForecastProvider`.
    """

    def __init__(self, cache_dir: str = ".cache") -> None:
        self._client = build_client(cache_dir=cache_dir)

    def fetch(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        variables: list[str],
    ) -> pl.DataFrame:
        """Fetch from Open-Meteo Archive API (ERA5 reanalysis)."""
        return fetch_historical_weather(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            variables=variables,
            client=self._client,
        )


class HistoricalForecastProvider:
    """Open-Meteo Historical Forecast API provider — archived NWP *forecasts*.

    Serves archived NWP forecast output (real forecasts made by ECMWF IFS /
    ICON / GFS at issue time), NOT reanalysis. Use this for val/test periods
    where honest forecast-time features are required. Coverage starts
    2022-01-01 for ICON, earlier for ECMWF IFS (2017+) and GFS (2021+).

    Ported from WattCast's production pattern — see
    ``wattcast/src/wattcast/data/open_meteo.py::fetch_historical_forecast_weather``
    and ``wattcast/docs/delivery-time-weather-features.md``.
    """

    def __init__(self, cache_dir: str = ".cache") -> None:
        self._client = build_client(cache_dir=cache_dir)

    def fetch(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        variables: list[str],
    ) -> pl.DataFrame:
        """Fetch from Open-Meteo Historical Forecast API."""
        return fetch_historical_forecast_weather(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            variables=variables,
            client=self._client,
        )
