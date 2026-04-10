"""Populate data/weather_forecast.db with archived NWP forecasts (Open-Meteo).

Fetches the Historical Forecast API — real forecast output at issue time, NOT
ERA5 reanalysis — for the registered weather configs used by EnerCast. Coverage
starts 2022-01-01. Idempotent: re-running only fills missing rows.

Usage:
    uv run python scripts/fetch_historical_forecasts.py                 # all
    uv run python scripts/fetch_historical_forecasts.py --config kelmarsh
    uv run python scripts/fetch_historical_forecasts.py --config rte_france \
        --start 2022-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from windcast.weather import (
    WEATHER_FORECAST_DB_PATH,
    HistoricalForecastProvider,
    get_weather,
)
from windcast.weather.registry import (
    WeatherPoint,
    WeightedWeatherConfig,
    get_weather_config,
    list_weather_configs,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIGS = ["kelmarsh", "rte_france"]


def fetch_one(
    config_name: str,
    start: str,
    end: str,
    db_path: Path,
    provider: HistoricalForecastProvider,
) -> None:
    """Fetch and cache archived forecasts for one registered config."""
    cfg = get_weather_config(config_name)

    if isinstance(cfg, WeightedWeatherConfig):
        logger.info(
            "[%s] weighted config, %d points, %s → %s",
            config_name,
            len(cfg.points),
            start,
            end,
        )
        # get_weather dispatches to get_weather_weighted for WeightedWeatherConfig
        # and fetches each point independently, caching under its own lat_lon key
        df = get_weather(
            config_name=config_name,
            start_date=start,
            end_date=end,
            db_path=db_path,
            provider=provider,
        )
        logger.info("[%s] cached: %d rows weighted mean", config_name, len(df))
    else:
        point = WeatherPoint(
            name=cfg.name,
            latitude=cfg.latitude,
            longitude=cfg.longitude,
            weight=1.0,
        )
        logger.info(
            "[%s] single point %s (%.4f, %.4f), %s → %s",
            config_name,
            cfg.name,
            point.latitude,
            point.longitude,
            start,
            end,
        )
        df = get_weather(
            config_name=config_name,
            start_date=start,
            end_date=end,
            db_path=db_path,
            provider=provider,
        )
        logger.info("[%s] cached: %d rows", config_name, len(df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch archived NWP forecasts")
    parser.add_argument(
        "--config",
        default=None,
        choices=list_weather_configs(),
        help=f"Weather config name. Default: all ({', '.join(DEFAULT_CONFIGS)})",
    )
    parser.add_argument("--start", default="2022-01-01", help="Start date ISO YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="End date ISO YYYY-MM-DD")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=WEATHER_FORECAST_DB_PATH,
        help=f"Forecast SQLite cache. Default: {WEATHER_FORECAST_DB_PATH}",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    configs = [args.config] if args.config else DEFAULT_CONFIGS
    provider = HistoricalForecastProvider()

    for cfg_name in configs:
        try:
            fetch_one(cfg_name, args.start, args.end, args.db_path, provider)
        except Exception as e:
            logger.error("[%s] fetch failed: %s", cfg_name, e, exc_info=True)

    logger.info("Done. Forecast db: %s", args.db_path)


if __name__ == "__main__":
    main()
