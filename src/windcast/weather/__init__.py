"""Weather data layer — fetch, cache, and serve NWP data for any site."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from windcast.weather.provider import OpenMeteoProvider, WeatherProvider
from windcast.weather.registry import (
    WeatherConfig,
    get_weather_config,
    list_weather_configs,
)
from windcast.weather.storage import WeatherStorage

__all__ = [
    "WeatherConfig",
    "WeatherProvider",
    "WeatherStorage",
    "get_weather",
    "get_weather_config",
    "list_weather_configs",
]

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/weather.db")

ERA5_LAG_DAYS = 5


def _location_key(config: WeatherConfig) -> str:
    """Derive a deterministic location key from config coordinates."""
    return f"{config.latitude}_{config.longitude}"


def _clamp_end_date(end_date: str) -> str:
    """Clamp end_date to ERA5 availability (today - 5 days)."""
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    max_dt = datetime.now(UTC) - timedelta(days=ERA5_LAG_DAYS)

    if end_dt > max_dt:
        clamped = max_dt.strftime("%Y-%m-%d")
        logger.warning(
            "ERA5 has ~%d-day lag. Clamping end_date from %s to %s",
            ERA5_LAG_DAYS,
            end_date,
            clamped,
        )
        return clamped
    return end_date


def get_weather(
    config_name: str,
    start_date: str,
    end_date: str,
    db_path: Path = DEFAULT_DB_PATH,
    provider: WeatherProvider | None = None,
) -> pl.DataFrame:
    """Fetch weather data for a registered config, with SQLite caching.

    1. Check SQLite cache for existing coverage
    2. Fetch only missing date ranges from provider
    3. Upsert new data into cache
    4. Return full requested range from cache

    Args:
        config_name: Registered weather config name (e.g., "kelmarsh").
        start_date: Start date ISO "YYYY-MM-DD".
        end_date: End date ISO "YYYY-MM-DD".
        db_path: Path to SQLite cache file.
        provider: Weather data provider. Defaults to OpenMeteoProvider.

    Returns:
        Polars DataFrame: timestamp_utc + weather variable columns (hourly).
    """
    config = get_weather_config(config_name)
    end_date = _clamp_end_date(end_date)

    if provider is None:
        provider = OpenMeteoProvider()

    storage = WeatherStorage(db_path)
    loc_key = _location_key(config)

    try:
        _fetch_missing(storage, provider, config, loc_key, start_date, end_date)
        return storage.query(loc_key, start_date, end_date, config.variables)
    finally:
        storage.close()


def _fetch_missing(
    storage: WeatherStorage,
    provider: WeatherProvider,
    config: WeatherConfig,
    loc_key: str,
    start_date: str,
    end_date: str,
) -> None:
    """Fetch only date ranges not already in the cache."""
    coverage = storage.get_coverage(loc_key)

    if coverage is None:
        logger.info("No cached data for %s — fetching full range", config.name)
        df = provider.fetch(
            config.latitude, config.longitude, start_date, end_date, config.variables
        )
        storage.upsert(loc_key, df)
        return

    cached_min, cached_max = coverage
    # Compare date portions only (cached timestamps are full ISO with time)
    cached_min_date = cached_min[:10]
    cached_max_date = cached_max[:10]

    # Fetch data before cache range
    if start_date < cached_min_date:
        logger.info("Fetching pre-cache gap: %s to %s", start_date, cached_min_date)
        df = provider.fetch(
            config.latitude, config.longitude, start_date, cached_min_date, config.variables
        )
        storage.upsert(loc_key, df)

    # Fetch data after cache range
    if end_date > cached_max_date:
        logger.info("Fetching post-cache gap: %s to %s", cached_max_date, end_date)
        df = provider.fetch(
            config.latitude, config.longitude, cached_max_date, end_date, config.variables
        )
        storage.upsert(loc_key, df)
