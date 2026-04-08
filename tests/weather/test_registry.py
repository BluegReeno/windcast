"""Tests for windcast.weather.registry module."""

import pytest

from windcast.config import KELMARSH, PVDAQ_SYSTEM4, SPAIN_DEMAND
from windcast.weather.registry import (
    WEATHER_REGISTRY,
    get_weather_config,
    list_weather_configs,
)


def test_kelmarsh_weather_config_exists():
    """Kelmarsh weather config is registered."""
    config = get_weather_config("kelmarsh")
    assert config.name == "kelmarsh"
    assert "wind_speed_100m" in config.variables
    assert "temperature_2m" in config.variables


def test_kelmarsh_coords_match_dataset_config():
    """Weather config coordinates must match dataset config."""
    weather = get_weather_config("kelmarsh")
    assert weather.latitude == KELMARSH.latitude
    assert weather.longitude == KELMARSH.longitude


def test_spain_coords_match_dataset_config():
    """Spain weather config coordinates must match dataset config."""
    weather = get_weather_config("spain_demand")
    assert weather.latitude == SPAIN_DEMAND.latitude
    assert weather.longitude == SPAIN_DEMAND.longitude


def test_pvdaq_coords_match_dataset_config():
    """PVDAQ weather config coordinates must match dataset config."""
    weather = get_weather_config("pvdaq_system4")
    assert weather.latitude == PVDAQ_SYSTEM4.latitude
    assert weather.longitude == PVDAQ_SYSTEM4.longitude


def test_get_weather_config_unknown_raises():
    """Unknown config name raises ValueError with available names."""
    with pytest.raises(ValueError, match="Unknown weather config"):
        get_weather_config("nonexistent")


def test_list_weather_configs():
    """list_weather_configs returns all registered names."""
    names = list_weather_configs()
    assert "kelmarsh" in names
    assert "spain_demand" in names
    assert "pvdaq_system4" in names
    assert len(names) == len(WEATHER_REGISTRY)
