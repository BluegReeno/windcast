"""Weather configuration registry — declarative per-dataset weather configs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WeatherConfig:
    """Weather data configuration for a dataset/site."""

    name: str
    latitude: float
    longitude: float
    variables: list[str]
    description: str = ""


KELMARSH_WEATHER = WeatherConfig(
    name="kelmarsh",
    latitude=52.4016,
    longitude=-0.9436,
    variables=[
        "wind_speed_100m",
        "wind_direction_100m",
        "wind_speed_10m",
        "wind_direction_10m",
        "temperature_2m",
        "pressure_msl",
    ],
    description="Kelmarsh wind farm — 6 NWP variables for wind forecasting",
)

SPAIN_WEATHER = WeatherConfig(
    name="spain_demand",
    latitude=40.4168,
    longitude=-3.7038,
    variables=[
        "temperature_2m",
        "wind_speed_10m",
        "relative_humidity_2m",
    ],
    description="Spain (Madrid) — weather for demand forecasting",
)

PVDAQ_WEATHER = WeatherConfig(
    name="pvdaq_system4",
    latitude=39.7407,
    longitude=-105.1686,
    variables=[
        "shortwave_radiation",
        "temperature_2m",
        "wind_speed_10m",
        "cloud_cover",
    ],
    description="PVDAQ System 4 (Golden CO) — weather for solar forecasting",
)

WEATHER_REGISTRY: dict[str, WeatherConfig] = {
    "kelmarsh": KELMARSH_WEATHER,
    "spain_demand": SPAIN_WEATHER,
    "pvdaq_system4": PVDAQ_WEATHER,
}


def get_weather_config(name: str) -> WeatherConfig:
    """Look up a weather config by name.

    Raises:
        ValueError: If name is not in the registry.
    """
    if name not in WEATHER_REGISTRY:
        available = ", ".join(sorted(WEATHER_REGISTRY))
        msg = f"Unknown weather config {name!r}. Available: {available}"
        raise ValueError(msg)
    return WEATHER_REGISTRY[name]


def list_weather_configs() -> list[str]:
    """Return all registered weather config names."""
    return list(WEATHER_REGISTRY.keys())
