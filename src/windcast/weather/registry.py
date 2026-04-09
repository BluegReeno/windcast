"""Weather configuration registry — declarative per-dataset weather configs."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WeatherConfig:
    """Single-point weather data configuration for a dataset/site."""

    name: str
    latitude: float
    longitude: float
    variables: list[str]
    description: str = ""


@dataclass(frozen=True)
class WeatherPoint:
    """One geographic point in a multi-point weighted weather config."""

    name: str
    latitude: float
    longitude: float
    weight: float = 1.0


@dataclass(frozen=True)
class WeightedWeatherConfig:
    """Multi-point weather config that returns population-weighted national means.

    Used for national demand forecasting where a single point (e.g. Paris) is a
    poor proxy for the country-wide load. Each point is fetched separately and
    cached under its own ``{lat}_{lon}`` key, then combined via weighted mean at
    query time.
    """

    name: str
    variables: list[str]
    points: list[WeatherPoint]
    description: str = ""
    _validate: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        if self._validate and self.points:
            total = sum(p.weight for p in self.points)
            if not 0.99 <= total <= 1.01:
                raise ValueError(
                    f"WeightedWeatherConfig {self.name!r}: point weights must sum to 1.0, "
                    f"got {total:.4f}"
                )


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

# French national temperature is a demand-weighted average of 8 cities.
# Weights are a population/demand proxy carried over from wattcast and cover
# ~95% of metropolitan France (see wattcast.config.TEMP_POINTS).
RTE_FRANCE_WEATHER = WeightedWeatherConfig(
    name="rte_france",
    variables=[
        "temperature_2m",
        "wind_speed_10m",
        "relative_humidity_2m",
        "shortwave_radiation",
    ],
    points=[
        WeatherPoint("idf_paris", 48.86, 2.35, 0.30),
        WeatherPoint("cvl_tours", 47.40, 0.70, 0.14),
        WeatherPoint("aura_lyon", 45.76, 4.84, 0.15),
        WeatherPoint("hdf_lille", 50.63, 3.06, 0.10),
        WeatherPoint("naq_bordeaux", 44.84, -0.58, 0.08),
        WeatherPoint("occ_toulouse", 43.60, 1.44, 0.08),
        WeatherPoint("paca_marseille", 43.30, 5.37, 0.07),
        WeatherPoint("gest_strasbourg", 48.58, 7.75, 0.08),
    ],
    description="France national NWP — 8 cities weighted by demand proxy (wattcast TEMP_POINTS)",
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

AnyWeatherConfig = WeatherConfig | WeightedWeatherConfig

WEATHER_REGISTRY: dict[str, AnyWeatherConfig] = {
    "kelmarsh": KELMARSH_WEATHER,
    "spain_demand": SPAIN_WEATHER,
    "rte_france": RTE_FRANCE_WEATHER,
    "pvdaq_system4": PVDAQ_WEATHER,
}


def get_weather_config(name: str) -> AnyWeatherConfig:
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
