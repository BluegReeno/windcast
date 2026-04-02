"""Tests for windcast.data.solar_qc module."""

from datetime import UTC, datetime, timedelta

import polars as pl

from windcast.data.schema import QC_OK, QC_SUSPECT
from windcast.data.solar_qc import (
    _fill_small_gaps,
    _flag_irradiance_outliers,
    _flag_nighttime_power,
    _flag_power_irradiance_inconsistency,
    _flag_power_outliers,
    _flag_temperature_outliers,
    run_solar_qc_pipeline,
    solar_qc_summary,
)


def _make_solar_df(n_rows: int = 24, **overrides) -> pl.DataFrame:
    """Create a minimal canonical solar DataFrame for testing."""
    data = {
        "timestamp_utc": [
            datetime(2020, 6, 15, 8, tzinfo=UTC) + timedelta(minutes=15 * i) for i in range(n_rows)
        ],
        "dataset_id": ["pvdaq_system4"] * n_rows,
        "system_id": ["4"] * n_rows,
        "power_kw": [1.5] * n_rows,
        "ghi_wm2": [None] * n_rows,
        "poa_wm2": [600.0] * n_rows,
        "ambient_temp_c": [25.0] * n_rows,
        "module_temp_c": [35.0] * n_rows,
        "wind_speed_ms": [None] * n_rows,
        "qc_flag": [0] * n_rows,
    }
    data.update(overrides)

    return pl.DataFrame(data).cast(
        {
            "timestamp_utc": pl.Datetime("us", "UTC"),
            "dataset_id": pl.String,
            "system_id": pl.String,
            "power_kw": pl.Float64,
            "ghi_wm2": pl.Float64,
            "poa_wm2": pl.Float64,
            "ambient_temp_c": pl.Float64,
            "module_temp_c": pl.Float64,
            "wind_speed_ms": pl.Float64,
            "qc_flag": pl.UInt8,
        }
    )


class TestFlagNighttimePower:
    def test_high_power_zero_irradiance_flagged(self):
        df = _make_solar_df(3, poa_wm2=[600.0, 0.0, 600.0], power_kw=[1.5, 0.5, 1.5])
        result = _flag_nighttime_power(df, threshold_kw=0.01)
        assert result["qc_flag"][1] == QC_SUSPECT

    def test_zero_power_zero_irradiance_not_flagged(self):
        df = _make_solar_df(3, poa_wm2=[0.0, 0.0, 0.0], power_kw=[0.0, 0.0, 0.0])
        result = _flag_nighttime_power(df, threshold_kw=0.01)
        assert result["qc_flag"].to_list() == [QC_OK, QC_OK, QC_OK]

    def test_normal_daytime_not_flagged(self):
        df = _make_solar_df(3)
        result = _flag_nighttime_power(df, threshold_kw=0.01)
        assert result["qc_flag"].to_list() == [QC_OK, QC_OK, QC_OK]


class TestFlagPowerOutliers:
    def test_excessive_power_flagged(self):
        df = _make_solar_df(3, power_kw=[1.5, 10.0, 1.5])
        result = _flag_power_outliers(df, max_power_kw=5.0)
        assert result["qc_flag"][1] == QC_SUSPECT

    def test_normal_power_not_flagged(self):
        df = _make_solar_df(3, power_kw=[1.0, 2.0, 1.5])
        result = _flag_power_outliers(df, max_power_kw=5.0)
        assert result["qc_flag"].to_list() == [QC_OK, QC_OK, QC_OK]


class TestFlagIrradianceOutliers:
    def test_high_irradiance_flagged(self):
        df = _make_solar_df(3, poa_wm2=[600.0, 2000.0, 600.0])
        result = _flag_irradiance_outliers(df, min_irr=-10.0, max_irr=1500.0)
        assert result["qc_flag"][1] == QC_SUSPECT

    def test_negative_irradiance_flagged(self):
        df = _make_solar_df(3, poa_wm2=[600.0, -20.0, 600.0])
        result = _flag_irradiance_outliers(df, min_irr=-10.0, max_irr=1500.0)
        assert result["qc_flag"][1] == QC_SUSPECT

    def test_normal_irradiance_not_flagged(self):
        df = _make_solar_df(3, poa_wm2=[0.0, 500.0, 1000.0])
        result = _flag_irradiance_outliers(df, min_irr=-10.0, max_irr=1500.0)
        assert result["qc_flag"].to_list() == [QC_OK, QC_OK, QC_OK]


class TestFlagTemperatureOutliers:
    def test_extreme_heat_flagged(self):
        df = _make_solar_df(3, ambient_temp_c=[25.0, 70.0, 25.0])
        result = _flag_temperature_outliers(df, min_temp=-30.0, max_temp=60.0)
        assert result["qc_flag"][1] == QC_SUSPECT

    def test_extreme_cold_flagged(self):
        df = _make_solar_df(3, ambient_temp_c=[25.0, -40.0, 25.0])
        result = _flag_temperature_outliers(df, min_temp=-30.0, max_temp=60.0)
        assert result["qc_flag"][1] == QC_SUSPECT

    def test_normal_temp_not_flagged(self):
        df = _make_solar_df(3, ambient_temp_c=[-10.0, 25.0, 50.0])
        result = _flag_temperature_outliers(df, min_temp=-30.0, max_temp=60.0)
        assert result["qc_flag"].to_list() == [QC_OK, QC_OK, QC_OK]


class TestFlagPowerIrradianceInconsistency:
    def test_high_irradiance_zero_power_flagged(self):
        df = _make_solar_df(3, poa_wm2=[600.0, 500.0, 600.0], power_kw=[1.5, 0.0, 1.5])
        result = _flag_power_irradiance_inconsistency(df)
        assert result["qc_flag"][1] == QC_SUSPECT

    def test_low_irradiance_zero_power_not_flagged(self):
        df = _make_solar_df(3, poa_wm2=[100.0, 50.0, 100.0], power_kw=[0.5, 0.0, 0.5])
        result = _flag_power_irradiance_inconsistency(df)
        assert result["qc_flag"].to_list() == [QC_OK, QC_OK, QC_OK]


class TestFillSmallGaps:
    def test_small_gap_filled(self):
        df = _make_solar_df(5, power_kw=[1.5, None, None, 2.0, 2.5])
        result = _fill_small_gaps(df, max_intervals=4)
        assert result["power_kw"][1] == 1.5

    def test_large_gap_preserved(self):
        df = _make_solar_df(7, power_kw=[1.5, None, None, None, None, None, 2.0])
        result = _fill_small_gaps(df, max_intervals=4)
        assert result["power_kw"][5] is None


class TestRunSolarQcPipeline:
    def test_full_pipeline_runs(self):
        df = _make_solar_df(24)
        result = run_solar_qc_pipeline(df)
        assert len(result) == 24
        assert "qc_flag" in result.columns

    def test_worst_flag_wins(self):
        # Power outlier + nighttime power on different rows
        df = _make_solar_df(
            3,
            power_kw=[10.0, 0.5, 1.5],
            poa_wm2=[600.0, 0.0, 600.0],
        )
        result = run_solar_qc_pipeline(df)
        assert result["qc_flag"][0] == QC_SUSPECT  # power > 5.0
        assert result["qc_flag"][1] == QC_SUSPECT  # nighttime power


class TestSolarQcSummary:
    def test_summary_counts(self):
        df = _make_solar_df(5, qc_flag=[0, 0, 1, 1, 0])
        summary = solar_qc_summary(df)
        assert summary["total_rows"] == 5
        assert summary["qc_ok"] == 3
        assert summary["qc_suspect"] == 2
        assert summary["qc_bad"] == 0

    def test_empty_summary(self):
        df = _make_solar_df(0)
        summary = solar_qc_summary(df)
        assert summary["total_rows"] == 0
