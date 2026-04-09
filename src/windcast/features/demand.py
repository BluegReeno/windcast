"""Demand-specific feature engineering functions."""

import logging
import math

import polars as pl

from windcast.data.schema import QC_OK
from windcast.features.registry import get_feature_set

logger = logging.getLogger(__name__)

DEFAULT_LOAD_LAGS = [1, 2, 24, 168]
DEFAULT_ROLLING_WINDOWS = [24, 168]


def build_demand_features(
    df: pl.DataFrame,
    feature_set: str = "demand_baseline",
) -> pl.DataFrame:
    """Build demand features from canonical demand DataFrame.

    Filters to QC_OK rows, sorts by timestamp per zone, then applies
    feature transforms based on the requested feature set.

    Args:
        df: Canonical demand DataFrame with qc_flag column.
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

    # Sort by zone and timestamp for correct lag/rolling computation
    df = df.sort("zone_id", "timestamp_utc")

    # Always build baseline features
    df = _add_lag_features(df, "load_mw", DEFAULT_LOAD_LAGS)
    df = _add_cyclic_calendar(df)

    # Enriched adds rolling load stats + holiday flag (numeric)
    if feature_set in ("demand_enriched", "demand_full"):
        df = _add_rolling_features(df, "load_mw", DEFAULT_ROLLING_WINDOWS)
        if "is_holiday" in df.columns:
            df = df.with_columns(pl.col("is_holiday").cast(pl.Int8).alias("is_holiday"))

    # Full adds HDD/CDD computed from the NWP temperature at the shortest horizon
    # (or the observed temperature column as a fallback for legacy datasets).
    if feature_set == "demand_full":
        if "nwp_temperature_2m_h1" in df.columns:
            df = _add_temperature_features(df, source_col="nwp_temperature_2m_h1")
        elif "temperature_c" in df.columns and df["temperature_c"].drop_nulls().len() > 0:
            df = _add_temperature_features(df, source_col="temperature_c")
        else:
            logger.warning("No temperature source for HDD/CDD — skipping")

    logger.info("Feature engineering complete: %d columns", len(df.columns))
    return df


def _add_lag_features(
    df: pl.DataFrame,
    col: str,
    lags: list[int],
) -> pl.DataFrame:
    """Add lag columns for a signal, computed per zone."""
    return df.with_columns(
        [pl.col(col).shift(lag).over("zone_id").alias(f"{col}_lag{lag}") for lag in lags]
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
            .over("zone_id")
            .alias(f"{col}_roll_mean_{w}")
        )
        if w <= 24:
            exprs.append(
                pl.col(col)
                .shift(1)
                .rolling_std(window_size=w)
                .over("zone_id")
                .alias(f"{col}_roll_std_{w}")
            )
    return df.with_columns(exprs)


def _add_cyclic_calendar(df: pl.DataFrame) -> pl.DataFrame:
    """Add sin/cos encoding of hour (24), day-of-week (7), and month (12)."""
    return df.with_columns(
        (pl.col("timestamp_utc").dt.hour().cast(pl.Float64) * (2 * math.pi / 24))
        .sin()
        .alias("hour_sin"),
        (pl.col("timestamp_utc").dt.hour().cast(pl.Float64) * (2 * math.pi / 24))
        .cos()
        .alias("hour_cos"),
        (pl.col("timestamp_utc").dt.weekday().cast(pl.Float64) * (2 * math.pi / 7))
        .sin()
        .alias("dow_sin"),
        (pl.col("timestamp_utc").dt.weekday().cast(pl.Float64) * (2 * math.pi / 7))
        .cos()
        .alias("dow_cos"),
        (pl.col("timestamp_utc").dt.month().cast(pl.Float64) * (2 * math.pi / 12))
        .sin()
        .alias("month_sin"),
        (pl.col("timestamp_utc").dt.month().cast(pl.Float64) * (2 * math.pi / 12))
        .cos()
        .alias("month_cos"),
    )


def _add_temperature_features(
    df: pl.DataFrame,
    source_col: str = "temperature_c",
) -> pl.DataFrame:
    """Add heating degree days (HDD) and cooling degree days (CDD) from the given column."""
    if source_col not in df.columns:
        logger.warning("HDD/CDD source column %s missing, skipping", source_col)
        return df
    return df.with_columns(
        pl.max_horizontal(pl.lit(0.0), pl.lit(18.0) - pl.col(source_col)).alias(
            "heating_degree_days"
        ),
        pl.max_horizontal(pl.lit(0.0), pl.col(source_col) - pl.lit(24.0)).alias(
            "cooling_degree_days"
        ),
    )


def _add_price_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add price lag features."""
    return df.with_columns(
        pl.col("price_eur_mwh").shift(1).over("zone_id").alias("price_lag1"),
        pl.col("price_eur_mwh").shift(24).over("zone_id").alias("price_lag24"),
    )
