"""NWP horizon feature joiner — resolution-agnostic.

Joins weather (NWP) data to a main DataFrame, shifted to match each forecast
horizon. The same function works for 10-min SCADA, hourly demand, or daily
price data — the shift is always computed as ``horizon_steps x resolution_minutes``.

Pattern adapted from WattCast delivery-time features.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import polars as pl

logger = logging.getLogger(__name__)


def join_nwp_horizon_features(
    df: pl.DataFrame,
    nwp_df: pl.DataFrame,
    horizons: list[int],
    resolution_minutes: int,
    nwp_columns: list[str] | None = None,
    prefix: str = "nwp_",
) -> pl.DataFrame:
    """Join NWP data aligned to each forecast horizon.

    For each horizon *h*, the NWP timestamps are shifted **backward** by
    ``h x resolution_minutes`` so that, at observation time *t*, the joined
    NWP values correspond to time *t + offset* — i.e. the weather at the
    forecast target time.

    Columns are named ``{prefix}{variable}_h{h}``
    (e.g. ``nwp_wind_speed_100m_h6``).

    Args:
        df: Main DataFrame with ``timestamp_utc`` column.
        nwp_df: NWP DataFrame with ``timestamp_utc`` + variable columns
            (typically hourly from Open-Meteo).
        horizons: Forecast horizons in **steps** (domain-specific).
        resolution_minutes: Minutes per step (10 for wind, 60 for demand, …).
        nwp_columns: NWP variable columns to join.  If ``None``, all
            non-timestamp columns from *nwp_df* are used.
        prefix: Prefix prepended to NWP column names.  Default ``"nwp_"``.

    Returns:
        *df* with additional columns ``{prefix}{var}_h{h}`` per horizon.
    """
    if nwp_columns is None:
        nwp_columns = [c for c in nwp_df.columns if c != "timestamp_utc"]

    if not nwp_columns:
        logger.warning("No NWP columns to join — returning df unchanged")
        return df

    # Resample NWP to target resolution if needed
    nwp_resampled = _resample_nwp(nwp_df, resolution_minutes)

    for h in horizons:
        offset = timedelta(minutes=h * resolution_minutes)

        # Shift NWP timestamps backward: at observation t we get NWP(t + offset)
        shifted = nwp_resampled.with_columns(
            (pl.col("timestamp_utc") - offset).alias("timestamp_utc")
        )

        # Rename variables → {prefix}{var}_h{h}
        rename_map = {c: f"{prefix}{c}_h{h}" for c in nwp_columns}
        shifted = shifted.rename(rename_map)

        join_cols = ["timestamp_utc", *list(rename_map.values())]
        df = df.join(shifted.select(join_cols), on="timestamp_utc", how="left")

    n_new = len(nwp_columns) * len(horizons)
    logger.info(
        "Joined %d NWP columns x %d horizons = %d new features (resolution=%dmin)",
        len(nwp_columns),
        len(horizons),
        n_new,
        resolution_minutes,
    )
    return df


def _resample_nwp(
    nwp_df: pl.DataFrame,
    target_resolution_minutes: int,
    nwp_resolution_minutes: int = 60,
) -> pl.DataFrame:
    """Resample NWP to target resolution via forward-fill.

    If the target resolution is coarser or equal to the NWP resolution,
    returns the DataFrame unchanged.

    Args:
        nwp_df: NWP DataFrame with ``timestamp_utc`` datetime column.
        target_resolution_minutes: Desired output resolution in minutes.
        nwp_resolution_minutes: Source NWP resolution (default 60 = hourly).

    Returns:
        Resampled DataFrame at *target_resolution_minutes* interval.
    """
    if target_resolution_minutes >= nwp_resolution_minutes:
        return nwp_df

    logger.info(
        "Resampling NWP from %dmin → %dmin (forward-fill)",
        nwp_resolution_minutes,
        target_resolution_minutes,
    )

    interval = f"{target_resolution_minutes}m"

    # Build a complete timestamp grid at target resolution
    ts = nwp_df.get_column("timestamp_utc")
    ts_min: datetime = ts.min()  # type: ignore[assignment]
    ts_max: datetime = ts.max()  # type: ignore[assignment]

    if ts_min is None or ts_max is None:
        return nwp_df

    grid = pl.DataFrame({"timestamp_utc": pl.datetime_range(ts_min, ts_max, interval, eager=True)})

    # Join NWP onto grid, then forward-fill gaps
    result = grid.join(nwp_df, on="timestamp_utc", how="left")
    value_cols = [c for c in result.columns if c != "timestamp_utc"]
    result = result.with_columns([pl.col(c).forward_fill() for c in value_cols])

    return result
