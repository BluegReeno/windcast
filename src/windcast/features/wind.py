"""Wind-specific feature engineering functions."""

import logging
import math

import polars as pl

from windcast.data.schema import QC_OK
from windcast.features.registry import get_feature_set

logger = logging.getLogger(__name__)

DEFAULT_LAGS = [1, 2, 3, 6, 12, 24]
DEFAULT_ROLLING_WINDOWS = [6, 12, 24]


def build_wind_features(
    df: pl.DataFrame,
    feature_set: str = "wind_baseline",
    weather_df: pl.DataFrame | None = None,
    horizons: list[int] | None = None,
    resolution_minutes: int = 10,
) -> pl.DataFrame:
    """Build wind features from SCADA DataFrame.

    Filters to QC_OK rows, sorts by timestamp per turbine, then applies
    feature transforms based on the requested feature set.

    Args:
        df: Canonical SCADA DataFrame with qc_flag column.
        feature_set: Name of feature set from registry.
        weather_df: Optional NWP DataFrame (timestamp_utc + variable columns).
            Required for ``wind_full`` to populate NWP horizon features.
        horizons: Forecast horizons in steps.  Required with *weather_df*.
        resolution_minutes: Minutes per step (default 10 for wind SCADA).

    Returns:
        DataFrame with original columns + feature columns.
    """
    fs = get_feature_set(feature_set)
    logger.info("Building feature set %r (%d features)", fs.name, len(fs.columns))

    # Filter to good-quality data only
    n_before = len(df)
    df = df.filter(pl.col("qc_flag") == QC_OK)
    logger.info("QC filter: %d -> %d rows (dropped %d)", n_before, len(df), n_before - len(df))

    # Sort by turbine and timestamp for correct lag/rolling computation
    df = df.sort("turbine_id", "timestamp_utc")

    # Always build baseline features
    df = _add_lag_features(df, "active_power_kw", DEFAULT_LAGS)
    df = _add_rolling_features(df, "active_power_kw", DEFAULT_ROLLING_WINDOWS)
    df = _add_cyclic_wind_direction(df)

    # Enriched adds wind-specific + hour cyclic
    if feature_set in ("wind_enriched", "wind_full"):
        df = _add_wind_specific_features(df)
        df = _add_cyclic_hour(df)

    # Full adds NWP horizon features + month + day-of-week cyclic
    if feature_set == "wind_full":
        df = _add_cyclic_calendar(df)
        if weather_df is not None and horizons is not None:
            from windcast.features.weather import join_nwp_horizon_features

            df = join_nwp_horizon_features(
                df,
                weather_df,
                horizons=horizons,
                resolution_minutes=resolution_minutes,
            )
        elif weather_df is None:
            logger.warning(
                "wind_full requested but no weather_df provided — NWP columns will be missing"
            )

    logger.info("Feature engineering complete: %d columns", len(df.columns))
    return df


def _add_lag_features(
    df: pl.DataFrame,
    col: str,
    lags: list[int],
) -> pl.DataFrame:
    """Add lag columns for a signal, computed per turbine."""
    return df.with_columns(
        [pl.col(col).shift(lag).over("turbine_id").alias(f"{col}_lag{lag}") for lag in lags]
    )


def _add_rolling_features(
    df: pl.DataFrame,
    col: str,
    windows: list[int],
) -> pl.DataFrame:
    """Add rolling mean and std features.

    Uses shift(1) before rolling to prevent look-ahead leakage.
    """
    exprs: list[pl.Expr] = []
    for w in windows:
        exprs.extend(
            [
                pl.col(col)
                .shift(1)
                .rolling_mean(window_size=w)
                .over("turbine_id")
                .alias(f"{col}_roll_mean_{w}"),
                pl.col(col)
                .shift(1)
                .rolling_std(window_size=w)
                .over("turbine_id")
                .alias(f"{col}_roll_std_{w}"),
            ]
        )
    return df.with_columns(exprs)


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
    """Add sin/cos encoding of month (period=12) and day-of-week (period=7)."""
    return df.with_columns(
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
    )
