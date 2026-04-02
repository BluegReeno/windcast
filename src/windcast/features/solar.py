"""Solar-specific feature engineering functions."""

import logging
import math

import polars as pl

from windcast.data.schema import QC_OK
from windcast.features.registry import get_feature_set

logger = logging.getLogger(__name__)

DEFAULT_POWER_LAGS = [1, 2, 4, 8, 96]  # 15m, 30m, 1h, 2h, 24h
DEFAULT_ROLLING_WINDOWS = [4, 16, 96]  # 1h, 4h, 24h
MAX_EXPECTED_IRRADIANCE = 1200.0  # W/m² for clearsky ratio cap


def build_solar_features(
    df: pl.DataFrame,
    feature_set: str = "solar_baseline",
) -> pl.DataFrame:
    """Build solar features from canonical solar DataFrame.

    Filters to QC_OK rows, sorts by timestamp per system, then applies
    feature transforms based on the requested feature set.

    Args:
        df: Canonical solar DataFrame with qc_flag column.
        feature_set: Name of feature set from registry.

    Returns:
        DataFrame with original columns + feature columns.
    """
    fs = get_feature_set(feature_set)
    logger.info("Building feature set %r (%d features)", fs.name, len(fs.columns))

    # Filter to good-quality data only
    n_before = len(df)
    df = df.filter(pl.col("qc_flag") == QC_OK)
    logger.info("QC filter: %d -> %d rows (dropped %d)", n_before, len(df), n_before - len(df))

    # Sort by system and timestamp for correct lag/rolling computation
    df = df.sort("system_id", "timestamp_utc")

    # Always build baseline features
    df = _add_lag_features(df, "power_kw", DEFAULT_POWER_LAGS)
    df = _add_cyclic_hour(df)

    # Enriched adds clearsky ratio, temperature, rolling stats
    if feature_set in ("solar_enriched", "solar_full"):
        df = _add_clearsky_ratio(df)
        df = _add_rolling_features(df, "power_kw", DEFAULT_ROLLING_WINDOWS)

    # Full adds GHI, wind, full cyclic encoding
    if feature_set == "solar_full":
        df = _add_cyclic_calendar(df)

    logger.info("Feature engineering complete: %d columns", len(df.columns))
    return df


def _add_lag_features(
    df: pl.DataFrame,
    col: str,
    lags: list[int],
) -> pl.DataFrame:
    """Add lag columns for a signal, computed per system."""
    return df.with_columns(
        [pl.col(col).shift(lag).over("system_id").alias(f"{col}_lag{lag}") for lag in lags]
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
        exprs.append(
            pl.col(col)
            .shift(1)
            .rolling_mean(window_size=w)
            .over("system_id")
            .alias(f"{col}_roll_mean_{w}")
        )
        if w <= 4:
            exprs.append(
                pl.col(col)
                .shift(1)
                .rolling_std(window_size=w)
                .over("system_id")
                .alias(f"{col}_roll_std_{w}")
            )
    return df.with_columns(exprs)


def _add_cyclic_hour(df: pl.DataFrame) -> pl.DataFrame:
    """Add sin/cos encoding of hour (24)."""
    return df.with_columns(
        (pl.col("timestamp_utc").dt.hour().cast(pl.Float64) * (2 * math.pi / 24))
        .sin()
        .alias("hour_sin"),
        (pl.col("timestamp_utc").dt.hour().cast(pl.Float64) * (2 * math.pi / 24))
        .cos()
        .alias("hour_cos"),
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


def _add_cyclic_calendar(df: pl.DataFrame) -> pl.DataFrame:
    """Add sin/cos encoding of month (12) and day-of-week (7)."""
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
