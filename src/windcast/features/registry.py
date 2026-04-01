"""Feature set registry — declarative definitions of feature columns per set."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSet:
    """Named collection of feature columns for ML training."""

    name: str
    columns: list[str]
    description: str


WIND_BASELINE = FeatureSet(
    name="wind_baseline",
    columns=[
        "wind_speed_ms",
        "wind_dir_sin",
        "wind_dir_cos",
        "active_power_kw_lag1",
        "active_power_kw_lag2",
        "active_power_kw_lag3",
        "active_power_kw_lag6",
        "active_power_kw_roll_mean_6",
        "active_power_kw_roll_mean_12",
        "active_power_kw_roll_mean_24",
        "active_power_kw_roll_std_6",
    ],
    description="Core wind + power lags/rolling stats",
)

WIND_ENRICHED = FeatureSet(
    name="wind_enriched",
    columns=[
        *WIND_BASELINE.columns,
        "wind_speed_cubed",
        "turbulence_intensity",
        "wind_dir_sector",
        "hour_sin",
        "hour_cos",
    ],
    description="Baseline + wind-specific transforms + cyclic hour",
)

WIND_FULL = FeatureSet(
    name="wind_full",
    columns=[
        *WIND_ENRICHED.columns,
        "nwp_wind_speed_100m",
        "nwp_wind_direction_100m",
        "nwp_temperature_2m",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
    ],
    description="Enriched + NWP columns + full cyclic encoding",
)

FEATURE_REGISTRY: dict[str, FeatureSet] = {
    "wind_baseline": WIND_BASELINE,
    "wind_enriched": WIND_ENRICHED,
    "wind_full": WIND_FULL,
}


def get_feature_set(name: str) -> FeatureSet:
    """Look up a feature set by name.

    Raises:
        ValueError: If name is not in the registry.
    """
    if name not in FEATURE_REGISTRY:
        available = ", ".join(sorted(FEATURE_REGISTRY))
        msg = f"Unknown feature set {name!r}. Available: {available}"
        raise ValueError(msg)
    return FEATURE_REGISTRY[name]


def list_feature_sets() -> list[str]:
    """Return all registered feature set names."""
    return list(FEATURE_REGISTRY.keys())
