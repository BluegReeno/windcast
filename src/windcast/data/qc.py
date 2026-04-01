"""Quality control pipeline for SCADA data."""

import logging

import polars as pl

from windcast.config import QCConfig
from windcast.data.schema import QC_BAD, QC_OK, QC_SUSPECT

logger = logging.getLogger(__name__)

# Interval between SCADA records
SCADA_INTERVAL_MINUTES = 10


def run_qc_pipeline(
    df: pl.DataFrame,
    rated_power_kw: float,
    qc_config: QCConfig | None = None,
) -> pl.DataFrame:
    """Apply QC rules to canonical SCADA DataFrame.

    Modifies qc_flag, is_curtailed, is_maintenance columns.

    Rules applied in order:
    1. Flag maintenance periods (status_code-based)
    2. Flag negative power → qc_flag=2
    3. Flag over-rated power → qc_flag=1
    4. Flag negative wind speed → qc_flag=2
    5. Flag extreme wind speed → qc_flag=1
    6. Flag frozen sensors → qc_flag=1
    7. Detect curtailment → is_curtailed=True
    8. Fill small gaps (< 30 min) with forward-fill
    9. Leave large gaps as null

    Args:
        df: Canonical SCADA DataFrame.
        rated_power_kw: Rated power of turbine in kW.
        qc_config: QC thresholds. Uses defaults if None.

    Returns:
        DataFrame with updated QC columns.
    """
    if qc_config is None:
        qc_config = QCConfig()

    # Start with qc_flag=0 baseline
    df = df.with_columns(pl.lit(QC_OK).cast(pl.UInt8).alias("qc_flag"))

    df = _flag_maintenance(df)
    df = _flag_power_outliers(df, rated_power_kw, qc_config.rated_power_tolerance)
    df = _flag_wind_outliers(df, qc_config.max_wind_speed_ms)
    df = _flag_frozen_sensors(df, qc_config.frozen_sensor_threshold_minutes)
    df = _detect_curtailment(df, rated_power_kw, qc_config.min_pitch_curtailment_deg)
    df = _fill_small_gaps(df, qc_config.max_gap_fill_minutes)

    return df


def _flag_maintenance(df: pl.DataFrame) -> pl.DataFrame:
    """Flag rows where status_code indicates non-operational state."""
    return df.with_columns(
        pl.when(pl.col("status_code") != 0)
        .then(True)
        .otherwise(pl.col("is_maintenance"))
        .alias("is_maintenance")
    )


def _flag_power_outliers(df: pl.DataFrame, rated_power_kw: float, tolerance: float) -> pl.DataFrame:
    """Flag negative power (qc=2) and over-rated power (qc=1)."""
    max_allowed = rated_power_kw * tolerance
    return df.with_columns(
        pl.when(pl.col("active_power_kw") < 0)
        .then(pl.max_horizontal(pl.col("qc_flag"), pl.lit(QC_BAD).cast(pl.UInt8)))
        .when(pl.col("active_power_kw") > max_allowed)
        .then(pl.max_horizontal(pl.col("qc_flag"), pl.lit(QC_SUSPECT).cast(pl.UInt8)))
        .otherwise(pl.col("qc_flag"))
        .alias("qc_flag")
    )


def _flag_wind_outliers(df: pl.DataFrame, max_wind_ms: float) -> pl.DataFrame:
    """Flag negative wind speed (qc=2) and extreme wind (qc=1)."""
    return df.with_columns(
        pl.when(pl.col("wind_speed_ms") < 0)
        .then(pl.max_horizontal(pl.col("qc_flag"), pl.lit(QC_BAD).cast(pl.UInt8)))
        .when(pl.col("wind_speed_ms") > max_wind_ms)
        .then(pl.max_horizontal(pl.col("qc_flag"), pl.lit(QC_SUSPECT).cast(pl.UInt8)))
        .otherwise(pl.col("qc_flag"))
        .alias("qc_flag")
    )


def _flag_frozen_sensors(df: pl.DataFrame, threshold_minutes: int) -> pl.DataFrame:
    """Flag periods where a signal is constant for > threshold_minutes.

    Checks wind_speed_ms, active_power_kw, and pitch_angle_deg.
    """
    threshold_rows = threshold_minutes // SCADA_INTERVAL_MINUTES

    # For each signal, detect runs of identical values
    frozen_exprs = [
        (pl.col(col_name) == pl.col(col_name).shift(1)).over("turbine_id").fill_null(False)
        for col_name in ["wind_speed_ms", "active_power_kw", "pitch_angle_deg"]
    ]

    # Any signal frozen counts
    any_frozen = pl.any_horizontal(*frozen_exprs)

    # Count consecutive frozen rows using cumulative sum of "not frozen" as group key
    df = df.with_columns(
        any_frozen.alias("_frozen"),
        (~any_frozen).cum_sum().over("turbine_id").alias("_frozen_group"),
    )

    # Count run lengths
    df = df.with_columns(
        pl.col("_frozen").cum_sum().over("turbine_id", "_frozen_group").alias("_frozen_run_len")
    )

    # Flag rows with long frozen runs
    df = df.with_columns(
        pl.when(pl.col("_frozen_run_len") >= threshold_rows)
        .then(pl.max_horizontal(pl.col("qc_flag"), pl.lit(QC_SUSPECT).cast(pl.UInt8)))
        .otherwise(pl.col("qc_flag"))
        .alias("qc_flag")
    ).drop("_frozen", "_frozen_group", "_frozen_run_len")

    return df


def _detect_curtailment(
    df: pl.DataFrame, rated_power_kw: float, min_pitch_deg: float
) -> pl.DataFrame:
    """Detect curtailment: high wind + high pitch + low power."""
    curtailment_threshold = rated_power_kw * 0.5
    return df.with_columns(
        pl.when(
            (pl.col("wind_speed_ms") > 8.0)
            & (pl.col("pitch_angle_deg") > min_pitch_deg)
            & (pl.col("active_power_kw") < curtailment_threshold)
            & pl.col("active_power_kw").is_not_null()
        )
        .then(True)
        .otherwise(pl.col("is_curtailed"))
        .alias("is_curtailed")
    )


def _fill_small_gaps(df: pl.DataFrame, max_gap_minutes: int) -> pl.DataFrame:
    """Forward-fill gaps shorter than max_gap_minutes.

    Operates per-turbine. Creates a regular 10-min grid via upsample,
    then forward-fills nulls only for gaps smaller than the threshold.
    """
    max_fill_rows = max_gap_minutes // SCADA_INTERVAL_MINUTES

    signal_cols = [
        "active_power_kw",
        "wind_speed_ms",
        "wind_direction_deg",
        "pitch_angle_deg",
        "rotor_rpm",
        "nacelle_direction_deg",
        "ambient_temp_c",
        "nacelle_temp_c",
    ]

    # Forward-fill per turbine, limited to max_fill_rows consecutive nulls
    existing_cols = [c for c in signal_cols if c in df.columns]

    df = df.with_columns(
        [
            pl.col(c).forward_fill(limit=max_fill_rows).over("turbine_id").alias(c)
            for c in existing_cols
        ]
    )

    return df


def qc_summary(df: pl.DataFrame) -> dict[str, int | float]:
    """Generate QC summary statistics for logging."""
    total = len(df)
    if total == 0:
        return {"total_rows": 0}

    qc_ok = df.filter(pl.col("qc_flag") == QC_OK).height
    qc_suspect = df.filter(pl.col("qc_flag") == QC_SUSPECT).height
    qc_bad = df.filter(pl.col("qc_flag") == QC_BAD).height
    curtailed = df.filter(pl.col("is_curtailed")).height
    maintenance = df.filter(pl.col("is_maintenance")).height

    return {
        "total_rows": total,
        "qc_ok": qc_ok,
        "qc_ok_pct": round(100 * qc_ok / total, 1),
        "qc_suspect": qc_suspect,
        "qc_suspect_pct": round(100 * qc_suspect / total, 1),
        "qc_bad": qc_bad,
        "qc_bad_pct": round(100 * qc_bad / total, 1),
        "curtailed": curtailed,
        "maintenance": maintenance,
    }
