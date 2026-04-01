"""Open-Meteo historical weather data client."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import openmeteo_requests
import polars as pl
import requests_cache
from retry_requests import retry

logger = logging.getLogger(__name__)

WIND_VARIABLES: list[str] = [
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_speed_10m",
    "wind_direction_10m",
    "temperature_2m",
    "pressure_msl",
]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def build_client(
    cache_dir: str = ".cache",
    expire_after: int = -1,
    retries: int = 5,
    backoff_factor: float = 0.2,
) -> openmeteo_requests.Client:
    """Build a cached, auto-retrying Open-Meteo client."""
    cache_session = requests_cache.CachedSession(cache_dir, expire_after=expire_after)
    retry_session = retry(cache_session, retries=retries, backoff_factor=backoff_factor)
    return openmeteo_requests.Client(session=retry_session)  # type: ignore[arg-type]


def fetch_historical_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    variables: list[str] | None = None,
    client: openmeteo_requests.Client | None = None,
) -> pl.DataFrame:
    """Fetch hourly historical weather from Open-Meteo archive.

    Args:
        latitude: Location latitude.
        longitude: Location longitude.
        start_date: Start date ISO format "YYYY-MM-DD".
        end_date: End date ISO format "YYYY-MM-DD".
        variables: Weather variables to fetch. Defaults to WIND_VARIABLES.
        client: Pre-built client. Creates new one if None.

    Returns:
        Polars DataFrame with timestamp_utc + weather variable columns.
    """
    if variables is None:
        variables = WIND_VARIABLES
    if client is None:
        client = build_client()

    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": variables,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }

    responses = client.weather_api(ARCHIVE_URL, params=params)
    response = responses[0]

    hourly = response.Hourly()
    assert hourly is not None, "No hourly data in Open-Meteo response"

    start_ts = datetime.fromtimestamp(hourly.Time(), tz=UTC)
    end_ts = datetime.fromtimestamp(hourly.TimeEnd(), tz=UTC)
    interval_s = hourly.Interval()

    timestamps = pl.datetime_range(
        start=start_ts,
        end=end_ts,
        interval=f"{interval_s}s",
        eager=True,
        closed="left",
    )

    data: dict[str, pl.Series] = {"timestamp_utc": timestamps}
    for i, var_name in enumerate(variables):
        values = hourly.Variables(i).ValuesAsNumpy()  # type: ignore[union-attr]
        data[var_name] = pl.Series(var_name, values, dtype=pl.Float32)

    return pl.DataFrame(data)
