"""Exogenous feature builders for mlforecast — domain-specific features only.

These functions produce the same domain features as wind.py/demand.py/solar.py
but WITHOUT lag or rolling features (mlforecast handles those internally).
"""

import logging
import math

import polars as pl

from windcast.data.schema import QC_OK
from windcast.features.registry import get_feature_set

logger = logging.getLogger(__name__)

MAX_EXPECTED_IRRADIANCE = 1200.0  # W/m² for clearsky ratio cap


def build_wind_exogenous(
    df: pl.DataFrame,
    feature_set: str = "wind_exog_baseline",
) -> pl.DataFrame:
    """Build wind exogenous features (no lags/rolling — mlforecast handles those).

    Args:
        df: Canonical SCADA DataFrame with qc_flag column.
        feature_set: Name of exogenous feature set from registry.

    Returns:
        DataFrame with original columns + exogenous feature columns.
    """
    fs = get_feature_set(feature_set)
    logger.info("Building exogenous feature set %r (%d features)", fs.name, len(fs.columns))

    n_before = len(df)
    df = df.filter(pl.col("qc_flag") == QC_OK)
    logger.info("QC filter: %d -> %d rows (dropped %d)", n_before, len(df), n_before - len(df))

    df = df.sort("turbine_id", "timestamp_utc")

    # Baseline: wind direction cyclic encoding
    df = _add_cyclic_wind_direction(df)

    # Enriched: wind-specific + hour cyclic
    if feature_set in ("wind_exog_enriched", "wind_exog_full"):
        df = _add_wind_specific_features(df)
        df = _add_cyclic_hour(df)

    # Full: month + day-of-week cyclic
    if feature_set == "wind_exog_full":
        df = _add_cyclic_calendar(df)

    logger.info("Exogenous feature engineering complete: %d columns", len(df.columns))
    return df


def build_demand_exogenous(
    df: pl.DataFrame,
    feature_set: str = "demand_exog_baseline",
) -> pl.DataFrame:
    """Build demand exogenous features (no lags/rolling — mlforecast handles those).

    Args:
        df: Canonical demand DataFrame with qc_flag column.
        feature_set: Name of exogenous feature set from registry.

    Returns:
        DataFrame with original columns + exogenous feature columns.
    """
    fs = get_feature_set(feature_set)
    logger.info("Building exogenous feature set %r (%d features)", fs.name, len(fs.columns))

    n_before = len(df)
    df = df.filter(pl.col("qc_flag") == QC_OK)
    logger.info("QC filter: %d -> %d rows (dropped %d)", n_before, len(df), n_before - len(df))

    df = df.sort("zone_id", "timestamp_utc")

    # Baseline: calendar cyclic encoding
    df = _add_cyclic_calendar(df)

    # Enriched: temperature features
    if feature_set in ("demand_exog_enriched", "demand_exog_full"):
        df = _add_temperature_features(df)

    # Full: weather, price lags, holiday
    if feature_set == "demand_exog_full":
        df = _add_price_features(df)
        df = df.with_columns(pl.col("is_holiday").cast(pl.Int8).alias("is_holiday"))

    logger.info("Exogenous feature engineering complete: %d columns", len(df.columns))
    return df


def build_solar_exogenous(
    df: pl.DataFrame,
    feature_set: str = "solar_exog_baseline",
) -> pl.DataFrame:
    """Build solar exogenous features (no lags/rolling — mlforecast handles those).

    Args:
        df: Canonical solar DataFrame with qc_flag column.
        feature_set: Name of exogenous feature set from registry.

    Returns:
        DataFrame with original columns + exogenous feature columns.
    """
    fs = get_feature_set(feature_set)
    logger.info("Building exogenous feature set %r (%d features)", fs.name, len(fs.columns))

    n_before = len(df)
    df = df.filter(pl.col("qc_flag") == QC_OK)
    logger.info("QC filter: %d -> %d rows (dropped %d)", n_before, len(df), n_before - len(df))

    df = df.sort("system_id", "timestamp_utc")

    # Baseline: hour cyclic
    df = _add_cyclic_hour(df)

    # Enriched: clearsky ratio
    if feature_set in ("solar_exog_enriched", "solar_exog_full"):
        df = _add_clearsky_ratio(df)

    # Full: full cyclic calendar
    if feature_set == "solar_exog_full":
        df = _add_cyclic_calendar(df)

    logger.info("Exogenous feature engineering complete: %d columns", len(df.columns))
    return df


# --- Shared feature helpers (same logic as wind.py/demand.py/solar.py) ---


def _add_cyclic_wind_direction(df: pl.DataFrame) -> pl.DataFrame:
    """Add sin/cos encoding of wind direction (period = 360 degrees)."""
    return df.with_columns(
        (pl.col("wind_direction_deg") * (math.pi / 180)).sin().alias("wind_dir_sin"),
        (pl.col("wind_direction_deg") * (math.pi / 180)).cos().alias("wind_dir_cos"),
    )


def _add_wind_specific_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add wind-specific derived features: V^3, turbulence intensity, direction sector."""
    return df.with_columns(
        pl.col("wind_speed_ms").pow(3).alias("wind_speed_cubed"),
        (
            pl.col("wind_speed_ms").shift(1).rolling_std(window_size=6).over("turbine_id")
            / pl.col("wind_speed_ms")
        )
        .fill_nan(0.0)
        .fill_null(0.0)
        .alias("turbulence_intensity"),
        (pl.col("wind_direction_deg") / 30).cast(pl.Int32).alias("wind_dir_sector"),
    )


def _add_cyclic_hour(df: pl.DataFrame) -> pl.DataFrame:
    """Add sin/cos encoding of hour of day (period = 24)."""
    return df.with_columns(
        (pl.col("timestamp_utc").dt.hour().cast(pl.Float64) * (2 * math.pi / 24))
        .sin()
        .alias("hour_sin"),
        (pl.col("timestamp_utc").dt.hour().cast(pl.Float64) * (2 * math.pi / 24))
        .cos()
        .alias("hour_cos"),
    )


def _add_cyclic_calendar(df: pl.DataFrame) -> pl.DataFrame:
    """Add sin/cos encoding of hour (24), day-of-week (7), and month (12)."""
    exprs = [
        (pl.col("timestamp_utc").dt.month().cast(pl.Float64) * (2 * math.pi / 12))
        .sin()
        .alias("month_sin"),
        (pl.col("timestamp_utc").dt.month().cast(pl.Float64) * (2 * math.pi / 12))
        .cos()
        .alias("month_cos"),
        (pl.col("timestamp_utc").dt.weekday().cast(pl.Float64) * (2 * math.pi / 7))
        .sin()
        .alias("dow_sin"),
        (pl.col("timestamp_utc").dt.weekday().cast(pl.Float64) * (2 * math.pi / 7))
        .cos()
        .alias("dow_cos"),
    ]
    # Add hour cyclic if not already present (demand adds it here, wind/solar add separately)
    if "hour_sin" not in df.columns:
        exprs.extend(
            [
                (pl.col("timestamp_utc").dt.hour().cast(pl.Float64) * (2 * math.pi / 24))
                .sin()
                .alias("hour_sin"),
                (pl.col("timestamp_utc").dt.hour().cast(pl.Float64) * (2 * math.pi / 24))
                .cos()
                .alias("hour_cos"),
            ]
        )
    return df.with_columns(exprs)


def _add_temperature_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add heating degree days (HDD) and cooling degree days (CDD)."""
    return df.with_columns(
        pl.max_horizontal(pl.lit(0.0), pl.lit(18.0) - pl.col("temperature_c")).alias(
            "heating_degree_days"
        ),
        pl.max_horizontal(pl.lit(0.0), pl.col("temperature_c") - pl.lit(24.0)).alias(
            "cooling_degree_days"
        ),
    )


def _add_price_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add price lag features (exogenous lags, not target lags)."""
    return df.with_columns(
        pl.col("price_eur_mwh").shift(1).over("zone_id").alias("price_lag1"),
        pl.col("price_eur_mwh").shift(24).over("zone_id").alias("price_lag24"),
    )


def _add_clearsky_ratio(df: pl.DataFrame) -> pl.DataFrame:
    """Add clearsky ratio: POA / max expected irradiance, capped at 1.5."""
    return df.with_columns(
        pl.when(pl.col("poa_wm2").is_not_null() & (pl.col("poa_wm2") > 0))
        .then(
            pl.min_horizontal(
                pl.col("poa_wm2") / MAX_EXPECTED_IRRADIANCE,
                pl.lit(1.5),
            )
        )
        .otherwise(pl.lit(0.0))
        .alias("clearsky_ratio")
    )
