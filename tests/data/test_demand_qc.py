"""Tests for windcast.data.demand_qc module."""

from datetime import UTC, datetime, timedelta

import polars as pl

from windcast.data.demand_qc import (
    _detect_dst_transitions,
    _detect_holidays,
    _fill_small_gaps,
    _flag_load_outliers,
    _flag_temperature_outliers,
    _flag_wind_outliers,
    demand_qc_summary,
    run_demand_qc_pipeline,
)
from windcast.data.schema import QC_BAD, QC_OK, QC_SUSPECT


def _make_demand_df(n_rows: int = 24, **overrides) -> pl.DataFrame:
    """Create a minimal canonical demand DataFrame for testing."""
    data = {
        "timestamp_utc": [
            datetime(2015, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(n_rows)
        ],
        "dataset_id": ["spain_demand"] * n_rows,
        "zone_id": ["ES"] * n_rows,
        "load_mw": [25000.0] * n_rows,
        "temperature_c": [15.0] * n_rows,
        "wind_speed_ms": [5.0] * n_rows,
        "humidity_pct": [65.0] * n_rows,
        "price_eur_mwh": [40.0] * n_rows,
        "is_holiday": [False] * n_rows,
        "is_dst_transition": [False] * n_rows,
        "qc_flag": [0] * n_rows,
    }
    data.update(overrides)

    return pl.DataFrame(data).cast(
        {
            "timestamp_utc": pl.Datetime("us", "UTC"),
            "dataset_id": pl.String,
            "zone_id": pl.String,
            "load_mw": pl.Float64,
            "temperature_c": pl.Float64,
            "wind_speed_ms": pl.Float64,
            "humidity_pct": pl.Float64,
            "price_eur_mwh": pl.Float64,
            "qc_flag": pl.UInt8,
        }
    )


class TestFlagLoadOutliers:
    def test_negative_load_flagged_bad(self):
        df = _make_demand_df(3, load_mw=[25000.0, -100.0, 25000.0])
        result = _flag_load_outliers(df, min_load=10000.0, max_load=50000.0)
        assert result["qc_flag"][1] == QC_BAD

    def test_extreme_load_flagged_suspect(self):
        df = _make_demand_df(3, load_mw=[25000.0, 55000.0, 25000.0])
        result = _flag_load_outliers(df, min_load=10000.0, max_load=50000.0)
        assert result["qc_flag"][1] == QC_SUSPECT

    def test_low_load_flagged_suspect(self):
        df = _make_demand_df(3, load_mw=[25000.0, 5000.0, 25000.0])
        result = _flag_load_outliers(df, min_load=10000.0, max_load=50000.0)
        assert result["qc_flag"][1] == QC_SUSPECT

    def test_normal_load_not_flagged(self):
        df = _make_demand_df(3, load_mw=[20000.0, 30000.0, 25000.0])
        result = _flag_load_outliers(df, min_load=10000.0, max_load=50000.0)
        assert result["qc_flag"].to_list() == [QC_OK, QC_OK, QC_OK]


class TestFlagTemperatureOutliers:
    def test_extreme_cold_flagged(self):
        df = _make_demand_df(3, temperature_c=[15.0, -25.0, 15.0])
        result = _flag_temperature_outliers(df, min_temp=-20.0, max_temp=50.0)
        assert result["qc_flag"][1] == QC_SUSPECT

    def test_extreme_hot_flagged(self):
        df = _make_demand_df(3, temperature_c=[15.0, 55.0, 15.0])
        result = _flag_temperature_outliers(df, min_temp=-20.0, max_temp=50.0)
        assert result["qc_flag"][1] == QC_SUSPECT

    def test_normal_temp_not_flagged(self):
        df = _make_demand_df(3, temperature_c=[0.0, 20.0, 40.0])
        result = _flag_temperature_outliers(df, min_temp=-20.0, max_temp=50.0)
        assert result["qc_flag"].to_list() == [QC_OK, QC_OK, QC_OK]


class TestFlagWindOutliers:
    def test_extreme_wind_flagged(self):
        df = _make_demand_df(3, wind_speed_ms=[5.0, 60.0, 5.0])
        result = _flag_wind_outliers(df, max_wind=50.0)
        assert result["qc_flag"][1] == QC_SUSPECT


class TestDetectHolidays:
    def test_known_holiday_detected(self):
        # Jan 1 2015 is a holiday
        df = _make_demand_df(
            3,
            timestamp_utc=[
                datetime(2015, 1, 1, 12, tzinfo=UTC),
                datetime(2015, 1, 2, 12, tzinfo=UTC),
                datetime(2015, 1, 6, 12, tzinfo=UTC),  # Epiphany
            ],
        )
        result = _detect_holidays(df)
        assert result["is_holiday"][0] is True
        assert result["is_holiday"][1] is False
        assert result["is_holiday"][2] is True

    def test_non_holiday_not_flagged(self):
        df = _make_demand_df(
            2,
            timestamp_utc=[
                datetime(2015, 3, 15, 12, tzinfo=UTC),
                datetime(2015, 6, 20, 12, tzinfo=UTC),
            ],
        )
        result = _detect_holidays(df)
        assert not any(result["is_holiday"].to_list())


class TestDetectDstTransitions:
    def test_march_dst_detected(self):
        # 2015-03-29 is spring-forward
        df = _make_demand_df(
            3,
            timestamp_utc=[
                datetime(2015, 3, 29, 1, tzinfo=UTC),  # DST hour
                datetime(2015, 3, 29, 2, tzinfo=UTC),  # DST hour
                datetime(2015, 3, 29, 12, tzinfo=UTC),  # Normal hour
            ],
        )
        result = _detect_dst_transitions(df)
        assert result["is_dst_transition"][0] is True
        assert result["is_dst_transition"][1] is True
        assert result["is_dst_transition"][2] is False

    def test_normal_date_not_flagged(self):
        df = _make_demand_df(
            2,
            timestamp_utc=[
                datetime(2015, 6, 15, 2, tzinfo=UTC),
                datetime(2015, 6, 15, 3, tzinfo=UTC),
            ],
        )
        result = _detect_dst_transitions(df)
        assert not any(result["is_dst_transition"].to_list())


class TestFillSmallGaps:
    def test_small_gap_filled(self):
        df = _make_demand_df(5, load_mw=[25000.0, None, None, 26000.0, 27000.0])
        result = _fill_small_gaps(df, max_gap_hours=3)
        assert result["load_mw"][1] == 25000.0

    def test_large_gap_preserved(self):
        df = _make_demand_df(6, load_mw=[25000.0, None, None, None, None, 26000.0])
        result = _fill_small_gaps(df, max_gap_hours=3)
        # forward_fill(limit=3) fills first 3, 4th stays null
        assert result["load_mw"][4] is None


class TestRunDemandQcPipeline:
    def test_full_pipeline_runs(self):
        df = _make_demand_df(24)
        result = run_demand_qc_pipeline(df)
        assert len(result) == 24
        assert "qc_flag" in result.columns

    def test_worst_flag_wins(self):
        df = _make_demand_df(3, load_mw=[-100.0, 25000.0, 55000.0])
        result = run_demand_qc_pipeline(df)
        assert result["qc_flag"][0] == QC_BAD
        assert result["qc_flag"][2] == QC_SUSPECT


class TestDemandQcSummary:
    def test_summary_counts(self):
        df = _make_demand_df(
            5,
            qc_flag=[0, 0, 1, 2, 0],
            is_holiday=[False, True, False, False, False],
            is_dst_transition=[False, False, True, False, False],
        )
        summary = demand_qc_summary(df)
        assert summary["total_rows"] == 5
        assert summary["qc_ok"] == 3
        assert summary["qc_suspect"] == 1
        assert summary["qc_bad"] == 1
        assert summary["holidays"] == 1
        assert summary["dst_transitions"] == 1

    def test_empty_summary(self):
        df = _make_demand_df(0)
        summary = demand_qc_summary(df)
        assert summary["total_rows"] == 0
