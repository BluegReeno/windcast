"""Weather data providers — protocol and implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl

from windcast.data.open_meteo import build_client, fetch_historical_weather


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
    """Open-Meteo Archive API provider."""

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
        """Fetch from Open-Meteo Archive API."""
        return fetch_historical_weather(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            variables=variables,
            client=self._client,
        )
