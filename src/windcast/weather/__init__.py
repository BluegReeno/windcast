"""Weather data layer — fetch, cache, and serve NWP data for any site."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from windcast.weather.provider import OpenMeteoProvider, WeatherProvider
from windcast.weather.registry import (
    AnyWeatherConfig,
    WeatherConfig,
    WeatherPoint,
    WeightedWeatherConfig,
    get_weather_config,
    list_weather_configs,
)
from windcast.weather.storage import WeatherStorage

__all__ = [
    "AnyWeatherConfig",
    "WeatherConfig",
    "WeatherPoint",
    "WeatherProvider",
    "WeatherStorage",
    "WeightedWeatherConfig",
    "get_weather",
    "get_weather_config",
    "get_weather_weighted",
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

    Dispatches on config type:

    - :class:`WeatherConfig` (single-point): returns variable columns for one
      lat/lon, as cached in the SQLite store.
    - :class:`WeightedWeatherConfig` (multi-point): fetches each point, then
      returns a population-weighted national mean DataFrame — one row per
      timestamp, one column per variable.

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

    if isinstance(config, WeightedWeatherConfig):
        return get_weather_weighted(config, start_date, end_date, db_path, provider)

    storage = WeatherStorage(db_path)
    loc_key = _location_key(config)

    try:
        _fetch_missing(storage, provider, config, loc_key, start_date, end_date)
        return storage.query(loc_key, start_date, end_date, config.variables)
    finally:
        storage.close()


def get_weather_weighted(
    config: WeightedWeatherConfig,
    start_date: str,
    end_date: str,
    db_path: Path = DEFAULT_DB_PATH,
    provider: WeatherProvider | None = None,
) -> pl.DataFrame:
    """Fetch a multi-point config and return a weighted national DataFrame.

    Each point is cached independently under its own ``{lat}_{lon}`` location
    key, so subsequent calls hit the cache. The result is the weight-normalised
    sum across points per timestamp, per variable — the same pattern as
    wattcast's ``compute_national_temperature``.
    """
    if provider is None:
        provider = OpenMeteoProvider()
    end_date = _clamp_end_date(end_date)

    storage = WeatherStorage(db_path)
    per_point_dfs: list[tuple[float, pl.DataFrame]] = []

    try:
        for point in config.points:
            # Build a single-point WeatherConfig stub to reuse _fetch_missing.
            point_cfg = WeatherConfig(
                name=f"{config.name}:{point.name}",
                latitude=point.latitude,
                longitude=point.longitude,
                variables=config.variables,
            )
            loc_key = _location_key(point_cfg)
            _fetch_missing(storage, provider, point_cfg, loc_key, start_date, end_date)
            df = storage.query(loc_key, start_date, end_date, config.variables)
            if df.is_empty():
                logger.warning(
                    "Empty NWP slice for point %s (%s,%s)",
                    point.name,
                    point.latitude,
                    point.longitude,
                )
                continue
            per_point_dfs.append((point.weight, df))
    finally:
        storage.close()

    if not per_point_dfs:
        logger.warning("No NWP data fetched for any point in %s", config.name)
        return pl.DataFrame(schema={"timestamp_utc": pl.Datetime("us", "UTC")})

    return _weighted_mean(per_point_dfs, config.variables)


def _weighted_mean(
    per_point_dfs: list[tuple[float, pl.DataFrame]],
    variables: list[str],
) -> pl.DataFrame:
    """Combine per-point DataFrames into a single weighted-mean wide DataFrame.

    Each input frame must have ``timestamp_utc`` + variable columns. Missing
    values are handled per (timestamp, variable): weights are renormalised over
    only the points that contributed a value, so gaps in one city don't bias
    the national mean.
    """
    # Label each frame by point index, stack vertically
    labelled: list[pl.DataFrame] = []
    for idx, (weight, df) in enumerate(per_point_dfs):
        labelled.append(
            df.with_columns(
                pl.lit(idx).alias("_point_idx"),
                pl.lit(weight).alias("_w"),
            )
        )
    stacked = pl.concat(labelled, how="vertical_relaxed")

    # For each (timestamp, variable), compute sum(w*x) / sum(w) ignoring null x
    agg_exprs = []
    for var in variables:
        if var not in stacked.columns:
            continue
        weighted_val = (pl.col("_w") * pl.col(var)).filter(pl.col(var).is_not_null()).sum()
        weight_sum = pl.col("_w").filter(pl.col(var).is_not_null()).sum()
        agg_exprs.append((weighted_val / weight_sum).alias(var))

    national = stacked.group_by("timestamp_utc").agg(*agg_exprs).sort("timestamp_utc")
    return national


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
