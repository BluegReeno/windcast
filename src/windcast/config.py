"""WindCast configuration — Pydantic Settings with env override support."""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetConfig(BaseModel):
    """Per-dataset metadata."""

    dataset_id: str
    rated_power_kw: float
    hub_height_m: float
    rotor_diameter_m: float
    n_turbines: int
    latitude: float
    longitude: float


KELMARSH = DatasetConfig(
    dataset_id="kelmarsh",
    rated_power_kw=2050.0,
    hub_height_m=78.5,
    rotor_diameter_m=92.0,
    n_turbines=6,
    latitude=52.4016,
    longitude=-0.9436,
)

HILL_OF_TOWIE = DatasetConfig(
    dataset_id="hill_of_towie",
    rated_power_kw=2300.0,
    hub_height_m=80.0,
    rotor_diameter_m=82.0,
    n_turbines=21,
    latitude=57.34,
    longitude=-2.65,
)

PENMANSHIEL = DatasetConfig(
    dataset_id="penmanshiel",
    rated_power_kw=2050.0,
    hub_height_m=78.5,
    rotor_diameter_m=82.0,
    n_turbines=13,
    latitude=55.905,
    longitude=-2.29,
)

DATASETS: dict[str, DatasetConfig] = {
    "kelmarsh": KELMARSH,
    "hill_of_towie": HILL_OF_TOWIE,
    "penmanshiel": PENMANSHIEL,
}


class QCConfig(BaseModel):
    """Quality control thresholds."""

    max_wind_speed_ms: float = 40.0
    max_gap_fill_minutes: int = 30
    frozen_sensor_threshold_minutes: int = 60
    rated_power_tolerance: float = 1.05
    min_pitch_curtailment_deg: float = 3.0


class WindCastSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WINDCAST_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    dataset_id: str = "kelmarsh"
    train_years: int = 5
    val_years: int = 1
    test_years: int = 1
    forecast_horizons: list[int] = [1, 6, 12, 24, 48]
    qc: QCConfig = Field(default_factory=QCConfig)
    mlflow_tracking_uri: str = "file:./mlruns"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def dataset_config(self) -> DatasetConfig:
        return DATASETS[self.dataset_id]


@lru_cache(maxsize=1)
def get_settings() -> WindCastSettings:
    return WindCastSettings()
