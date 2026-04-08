"""Feature engineering package for wind, demand, and solar domains."""

from windcast.features.demand import build_demand_features
from windcast.features.exogenous import (
    build_demand_exogenous,
    build_solar_exogenous,
    build_wind_exogenous,
)
from windcast.features.registry import (
    FEATURE_REGISTRY,
    FeatureSet,
    get_feature_set,
    list_feature_sets,
)
from windcast.features.solar import build_solar_features
from windcast.features.weather import join_nwp_horizon_features
from windcast.features.wind import build_wind_features

__all__ = [
    "FEATURE_REGISTRY",
    "FeatureSet",
    "build_demand_exogenous",
    "build_demand_features",
    "build_solar_exogenous",
    "build_solar_features",
    "build_wind_exogenous",
    "build_wind_features",
    "get_feature_set",
    "join_nwp_horizon_features",
    "list_feature_sets",
]
