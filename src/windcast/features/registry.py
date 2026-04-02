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

DEMAND_BASELINE = FeatureSet(
    name="demand_baseline",
    columns=[
        "load_mw_lag1",
        "load_mw_lag2",
        "load_mw_lag24",
        "load_mw_lag168",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ],
    description="Demand baseline: load lags (H-1, H-2, D-1, W-1) + calendar cyclic",
)

DEMAND_ENRICHED = FeatureSet(
    name="demand_enriched",
    columns=[
        *DEMAND_BASELINE.columns,
        "temperature_c",
        "heating_degree_days",
        "cooling_degree_days",
        "load_mw_roll_mean_24",
        "load_mw_roll_std_24",
        "load_mw_roll_mean_168",
    ],
    description="Demand enriched: + temperature, HDD/CDD, rolling load stats",
)

DEMAND_FULL = FeatureSet(
    name="demand_full",
    columns=[
        *DEMAND_ENRICHED.columns,
        "wind_speed_ms",
        "humidity_pct",
        "price_eur_mwh",
        "price_lag1",
        "price_lag24",
        "is_holiday",
    ],
    description="Demand full: + wind, humidity, price lags, holiday flag",
)

SOLAR_BASELINE = FeatureSet(
    name="solar_baseline",
    columns=[
        "poa_wm2",
        "power_kw_lag1",
        "power_kw_lag2",
        "power_kw_lag4",
        "power_kw_lag8",
        "power_kw_lag96",
        "hour_sin",
        "hour_cos",
    ],
    description="Solar baseline: POA irradiance + power lags + hour cyclic",
)

SOLAR_ENRICHED = FeatureSet(
    name="solar_enriched",
    columns=[
        *SOLAR_BASELINE.columns,
        "clearsky_ratio",
        "ambient_temp_c",
        "module_temp_c",
        "power_kw_roll_mean_4",
        "power_kw_roll_mean_16",
        "power_kw_roll_mean_96",
        "power_kw_roll_std_4",
    ],
    description="Solar enriched: + clearsky ratio, temperature, rolling power stats",
)

SOLAR_FULL = FeatureSet(
    name="solar_full",
    columns=[
        *SOLAR_ENRICHED.columns,
        "ghi_wm2",
        "wind_speed_ms",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
    ],
    description="Solar full: + GHI, wind, full cyclic encoding",
)

FEATURE_REGISTRY: dict[str, FeatureSet] = {
    "wind_baseline": WIND_BASELINE,
    "wind_enriched": WIND_ENRICHED,
    "wind_full": WIND_FULL,
    "demand_baseline": DEMAND_BASELINE,
    "demand_enriched": DEMAND_ENRICHED,
    "demand_full": DEMAND_FULL,
    "solar_baseline": SOLAR_BASELINE,
    "solar_enriched": SOLAR_ENRICHED,
    "solar_full": SOLAR_FULL,
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
