"""Quality control pipeline for demand data."""

import logging
from datetime import date

import polars as pl

from windcast.config import DemandQCConfig
from windcast.data.demand_schema import DEMAND_SIGNAL_COLUMNS
from windcast.data.schema import QC_BAD, QC_OK, QC_SUSPECT

logger = logging.getLogger(__name__)

# Spain public holidays (fixed + movable) for 2015-2018
SPAIN_HOLIDAYS: set[date] = {
    # Fixed holidays
    *[date(y, 1, 1) for y in range(2015, 2019)],  # New Year
    *[date(y, 1, 6) for y in range(2015, 2019)],  # Epiphany
    *[date(y, 5, 1) for y in range(2015, 2019)],  # Labour Day
    *[date(y, 8, 15) for y in range(2015, 2019)],  # Assumption
    *[date(y, 10, 12) for y in range(2015, 2019)],  # National Day
    *[date(y, 11, 1) for y in range(2015, 2019)],  # All Saints
    *[date(y, 12, 6) for y in range(2015, 2019)],  # Constitution Day
    *[date(y, 12, 8) for y in range(2015, 2019)],  # Immaculate Conception
    *[date(y, 12, 25) for y in range(2015, 2019)],  # Christmas
    # Movable holidays (Good Friday)
    date(2015, 4, 3),
    date(2016, 3, 25),
    date(2017, 4, 14),
    date(2018, 3, 30),
}

# DST transition dates (last Sunday of March/October) for 2015-2018
DST_TRANSITION_DATES: set[date] = {
    date(2015, 3, 29),
    date(2015, 10, 25),
    date(2016, 3, 27),
    date(2016, 10, 30),
    date(2017, 3, 26),
    date(2017, 10, 29),
    date(2018, 3, 25),
    date(2018, 10, 28),
}


def run_demand_qc_pipeline(
    df: pl.DataFrame,
    qc_config: DemandQCConfig | None = None,
) -> pl.DataFrame:
    """Apply QC rules to canonical demand DataFrame.

    Rules applied in order:
    1. Flag load outliers (negative, extreme)
    2. Flag temperature outliers
    3. Flag wind speed outliers
    4. Detect holidays
    5. Detect DST transitions
    6. Forward-fill small gaps

    Args:
        df: Canonical demand DataFrame.
        qc_config: QC thresholds. Uses defaults if None.

    Returns:
        DataFrame with updated QC/flag columns.
    """
    if qc_config is None:
        qc_config = DemandQCConfig()

    # Reset qc_flag baseline
    df = df.with_columns(pl.lit(QC_OK).cast(pl.UInt8).alias("qc_flag"))

    df = _flag_load_outliers(df, qc_config.min_load_mw, qc_config.max_load_mw)
    df = _flag_temperature_outliers(df, qc_config.min_temperature_c, qc_config.max_temperature_c)
    df = _flag_wind_outliers(df, qc_config.max_wind_speed_ms)
    df = _detect_holidays(df)
    df = _detect_dst_transitions(df)
    df = _fill_small_gaps(df, qc_config.max_gap_fill_hours)

    return df


def _flag_load_outliers(df: pl.DataFrame, min_load: float, max_load: float) -> pl.DataFrame:
    """Flag negative or extreme load values."""
    return df.with_columns(
        pl.when(pl.col("load_mw") < 0)
        .then(pl.max_horizontal(pl.col("qc_flag"), pl.lit(QC_BAD).cast(pl.UInt8)))
        .when((pl.col("load_mw") < min_load) | (pl.col("load_mw") > max_load))
        .then(pl.max_horizontal(pl.col("qc_flag"), pl.lit(QC_SUSPECT).cast(pl.UInt8)))
        .otherwise(pl.col("qc_flag"))
        .alias("qc_flag")
    )


def _flag_temperature_outliers(df: pl.DataFrame, min_temp: float, max_temp: float) -> pl.DataFrame:
    """Flag extreme temperature values."""
    return df.with_columns(
        pl.when((pl.col("temperature_c") < min_temp) | (pl.col("temperature_c") > max_temp))
        .then(pl.max_horizontal(pl.col("qc_flag"), pl.lit(QC_SUSPECT).cast(pl.UInt8)))
        .otherwise(pl.col("qc_flag"))
        .alias("qc_flag")
    )


def _flag_wind_outliers(df: pl.DataFrame, max_wind: float) -> pl.DataFrame:
    """Flag extreme wind speed values."""
    return df.with_columns(
        pl.when(pl.col("wind_speed_ms") > max_wind)
        .then(pl.max_horizontal(pl.col("qc_flag"), pl.lit(QC_SUSPECT).cast(pl.UInt8)))
        .otherwise(pl.col("qc_flag"))
        .alias("qc_flag")
    )


def _detect_holidays(df: pl.DataFrame) -> pl.DataFrame:
    """Set is_holiday=True for Spain public holidays."""
    holiday_dates = pl.Series("_holiday_dates", list(SPAIN_HOLIDAYS)).implode()
    return df.with_columns(
        pl.col("timestamp_utc").dt.date().is_in(holiday_dates).alias("is_holiday")
    )


def _detect_dst_transitions(df: pl.DataFrame) -> pl.DataFrame:
    """Set is_dst_transition=True for DST change hours (01:00-03:00 UTC)."""
    dst_dates = pl.Series("_dst_dates", list(DST_TRANSITION_DATES)).implode()
    return df.with_columns(
        (
            pl.col("timestamp_utc").dt.date().is_in(dst_dates)
            & pl.col("timestamp_utc").dt.hour().is_between(1, 3)
        ).alias("is_dst_transition")
    )


def _fill_small_gaps(df: pl.DataFrame, max_gap_hours: int) -> pl.DataFrame:
    """Forward-fill signal columns for gaps up to max_gap_hours."""
    existing_cols = [c for c in DEMAND_SIGNAL_COLUMNS if c in df.columns]
    return df.with_columns(
        [
            pl.col(c).forward_fill(limit=max_gap_hours).over("zone_id").alias(c)
            for c in existing_cols
        ]
    )


def demand_qc_summary(df: pl.DataFrame) -> dict[str, int | float]:
    """Generate QC summary statistics for logging."""
    total = len(df)
    if total == 0:
        return {"total_rows": 0}

    qc_ok = df.filter(pl.col("qc_flag") == QC_OK).height
    qc_suspect = df.filter(pl.col("qc_flag") == QC_SUSPECT).height
    qc_bad = df.filter(pl.col("qc_flag") == QC_BAD).height
    holidays = df.filter(pl.col("is_holiday")).height
    dst = df.filter(pl.col("is_dst_transition")).height

    return {
        "total_rows": total,
        "qc_ok": qc_ok,
        "qc_ok_pct": round(100 * qc_ok / total, 1),
        "qc_suspect": qc_suspect,
        "qc_suspect_pct": round(100 * qc_suspect / total, 1),
        "qc_bad": qc_bad,
        "qc_bad_pct": round(100 * qc_bad / total, 1),
        "holidays": holidays,
        "dst_transitions": dst,
    }
