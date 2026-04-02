"""Quality control pipeline for solar PV data."""

import logging

import polars as pl

from windcast.config import SolarQCConfig
from windcast.data.schema import QC_BAD, QC_OK, QC_SUSPECT
from windcast.data.solar_schema import SOLAR_SIGNAL_COLUMNS

logger = logging.getLogger(__name__)


def run_solar_qc_pipeline(
    df: pl.DataFrame,
    qc_config: SolarQCConfig | None = None,
) -> pl.DataFrame:
    """Apply QC rules to canonical solar DataFrame.

    Rules applied in order:
    1. Flag nighttime power (power > threshold when irradiance <= 0)
    2. Flag power outliers (above system capacity)
    3. Flag irradiance outliers (out of physical bounds)
    4. Flag temperature outliers
    5. Flag power-irradiance inconsistency (high irradiance, zero power)
    6. Forward-fill small gaps

    Args:
        df: Canonical solar DataFrame.
        qc_config: QC thresholds. Uses defaults if None.

    Returns:
        DataFrame with updated qc_flag column.
    """
    if qc_config is None:
        qc_config = SolarQCConfig()

    # Reset qc_flag baseline
    df = df.with_columns(pl.lit(QC_OK).cast(pl.UInt8).alias("qc_flag"))

    df = _flag_nighttime_power(df, qc_config.nighttime_power_threshold_kw)
    df = _flag_power_outliers(df, qc_config.max_power_kw)
    df = _flag_irradiance_outliers(df, qc_config.min_irradiance_wm2, qc_config.max_irradiance_wm2)
    df = _flag_temperature_outliers(df, qc_config.min_temperature_c, qc_config.max_temperature_c)
    df = _flag_power_irradiance_inconsistency(df)
    df = _fill_small_gaps(df, qc_config.max_gap_fill_intervals)

    return df


def _flag_nighttime_power(df: pl.DataFrame, threshold_kw: float) -> pl.DataFrame:
    """Flag rows where power > threshold but irradiance <= 0 (nighttime)."""
    return df.with_columns(
        pl.when(
            (pl.col("poa_wm2").is_not_null())
            & (pl.col("poa_wm2") <= 0)
            & (pl.col("power_kw") > threshold_kw)
        )
        .then(pl.max_horizontal(pl.col("qc_flag"), pl.lit(QC_SUSPECT).cast(pl.UInt8)))
        .otherwise(pl.col("qc_flag"))
        .alias("qc_flag")
    )


def _flag_power_outliers(df: pl.DataFrame, max_power_kw: float) -> pl.DataFrame:
    """Flag power values exceeding system capacity."""
    return df.with_columns(
        pl.when(pl.col("power_kw") > max_power_kw)
        .then(pl.max_horizontal(pl.col("qc_flag"), pl.lit(QC_SUSPECT).cast(pl.UInt8)))
        .otherwise(pl.col("qc_flag"))
        .alias("qc_flag")
    )


def _flag_irradiance_outliers(df: pl.DataFrame, min_irr: float, max_irr: float) -> pl.DataFrame:
    """Flag irradiance values outside physical bounds."""
    return df.with_columns(
        pl.when(
            (pl.col("poa_wm2").is_not_null())
            & ((pl.col("poa_wm2") < min_irr) | (pl.col("poa_wm2") > max_irr))
        )
        .then(pl.max_horizontal(pl.col("qc_flag"), pl.lit(QC_SUSPECT).cast(pl.UInt8)))
        .otherwise(pl.col("qc_flag"))
        .alias("qc_flag")
    )


def _flag_temperature_outliers(df: pl.DataFrame, min_temp: float, max_temp: float) -> pl.DataFrame:
    """Flag extreme ambient temperature values."""
    return df.with_columns(
        pl.when(
            (pl.col("ambient_temp_c").is_not_null())
            & ((pl.col("ambient_temp_c") < min_temp) | (pl.col("ambient_temp_c") > max_temp))
        )
        .then(pl.max_horizontal(pl.col("qc_flag"), pl.lit(QC_SUSPECT).cast(pl.UInt8)))
        .otherwise(pl.col("qc_flag"))
        .alias("qc_flag")
    )


def _flag_power_irradiance_inconsistency(df: pl.DataFrame) -> pl.DataFrame:
    """Flag rows with high irradiance but zero power (possible inverter fault)."""
    return df.with_columns(
        pl.when(
            (pl.col("poa_wm2").is_not_null())
            & (pl.col("poa_wm2") > 200)
            & (pl.col("power_kw") <= 0)
        )
        .then(pl.max_horizontal(pl.col("qc_flag"), pl.lit(QC_SUSPECT).cast(pl.UInt8)))
        .otherwise(pl.col("qc_flag"))
        .alias("qc_flag")
    )


def _fill_small_gaps(df: pl.DataFrame, max_intervals: int) -> pl.DataFrame:
    """Forward-fill signal columns for gaps up to max_intervals."""
    existing_cols = [c for c in SOLAR_SIGNAL_COLUMNS if c in df.columns]
    return df.with_columns(
        [
            pl.col(c).forward_fill(limit=max_intervals).over("system_id").alias(c)
            for c in existing_cols
        ]
    )


def solar_qc_summary(df: pl.DataFrame) -> dict[str, int | float]:
    """Generate QC summary statistics for logging."""
    total = len(df)
    if total == 0:
        return {"total_rows": 0}

    qc_ok = df.filter(pl.col("qc_flag") == QC_OK).height
    qc_suspect = df.filter(pl.col("qc_flag") == QC_SUSPECT).height
    qc_bad = df.filter(pl.col("qc_flag") == QC_BAD).height

    return {
        "total_rows": total,
        "qc_ok": qc_ok,
        "qc_ok_pct": round(100 * qc_ok / total, 1),
        "qc_suspect": qc_suspect,
        "qc_suspect_pct": round(100 * qc_suspect / total, 1),
        "qc_bad": qc_bad,
        "qc_bad_pct": round(100 * qc_bad / total, 1),
    }
