"""Tests for windcast.data.qc module."""

from datetime import UTC, datetime, timedelta

import polars as pl

from windcast.data.qc import (
    _detect_curtailment,
    _fill_small_gaps,
    _flag_frozen_sensors,
    _flag_maintenance,
    _flag_power_outliers,
    _flag_wind_outliers,
    qc_summary,
    run_qc_pipeline,
)
from windcast.data.schema import QC_BAD, QC_OK, QC_SUSPECT


def _make_scada_df(n_rows: int = 10, **overrides) -> pl.DataFrame:
    """Create a minimal canonical SCADA DataFrame for testing."""
    data = {
        "timestamp_utc": [
            datetime(2021, 1, 1, tzinfo=UTC) + timedelta(minutes=i * 10) for i in range(n_rows)
        ],
        "dataset_id": ["kelmarsh"] * n_rows,
        "turbine_id": ["KWF1"] * n_rows,
        "active_power_kw": [1000.0] * n_rows,
        "wind_speed_ms": [10.0] * n_rows,
        "wind_direction_deg": [180.0] * n_rows,
        "pitch_angle_deg": [2.0] * n_rows,
        "rotor_rpm": [12.0] * n_rows,
        "nacelle_direction_deg": [180.0] * n_rows,
        "ambient_temp_c": [10.0] * n_rows,
        "nacelle_temp_c": [35.0] * n_rows,
        "status_code": [0] * n_rows,
        "is_curtailed": [False] * n_rows,
        "is_maintenance": [False] * n_rows,
        "qc_flag": [0] * n_rows,
    }
    data.update(overrides)

    return pl.DataFrame(data).cast(
        {
            "timestamp_utc": pl.Datetime("us", "UTC"),
            "dataset_id": pl.String,
            "turbine_id": pl.String,
            "active_power_kw": pl.Float64,
            "wind_speed_ms": pl.Float64,
            "wind_direction_deg": pl.Float64,
            "pitch_angle_deg": pl.Float64,
            "rotor_rpm": pl.Float64,
            "nacelle_direction_deg": pl.Float64,
            "ambient_temp_c": pl.Float64,
            "nacelle_temp_c": pl.Float64,
            "status_code": pl.Int32,
            "qc_flag": pl.UInt8,
        }
    )


class TestFlagMaintenance:
    def test_non_zero_status_flagged(self):
        df = _make_scada_df(3, status_code=[0, -1, 0])
        result = _flag_maintenance(df)
        assert result["is_maintenance"].to_list() == [False, True, False]


class TestFlagPowerOutliers:
    def test_negative_power_flagged_bad(self):
        df = _make_scada_df(3, active_power_kw=[100.0, -50.0, 1000.0])
        result = _flag_power_outliers(df, rated_power_kw=2050.0, tolerance=1.05)
        assert result["qc_flag"][1] == QC_BAD

    def test_over_rated_power_flagged_suspect(self):
        df = _make_scada_df(3, active_power_kw=[100.0, 2200.0, 1000.0])
        result = _flag_power_outliers(df, rated_power_kw=2050.0, tolerance=1.05)
        assert result["qc_flag"][1] == QC_SUSPECT

    def test_normal_power_not_flagged(self):
        df = _make_scada_df(3, active_power_kw=[100.0, 1500.0, 2000.0])
        result = _flag_power_outliers(df, rated_power_kw=2050.0, tolerance=1.05)
        assert result["qc_flag"].to_list() == [QC_OK, QC_OK, QC_OK]


class TestFlagWindOutliers:
    def test_negative_wind_flagged_bad(self):
        df = _make_scada_df(3, wind_speed_ms=[10.0, -1.0, 5.0])
        result = _flag_wind_outliers(df, max_wind_ms=40.0)
        assert result["qc_flag"][1] == QC_BAD

    def test_extreme_wind_flagged_suspect(self):
        df = _make_scada_df(3, wind_speed_ms=[10.0, 45.0, 5.0])
        result = _flag_wind_outliers(df, max_wind_ms=40.0)
        assert result["qc_flag"][1] == QC_SUSPECT


class TestFlagFrozenSensors:
    def test_frozen_wind_speed_detected(self):
        # 8 identical values at 10-min intervals = 70 min > 60-min threshold
        n = 8
        df = _make_scada_df(
            n,
            wind_speed_ms=[10.0] * n,
            active_power_kw=[1000.0 + i for i in range(n)],
            pitch_angle_deg=[2.0 + i * 0.1 for i in range(n)],
        )
        result = _flag_frozen_sensors(df, threshold_minutes=60)
        # At least some rows should be flagged
        assert result["qc_flag"].max() >= QC_SUSPECT

    def test_varying_signals_not_flagged(self):
        n = 8
        df = _make_scada_df(
            n,
            wind_speed_ms=[i * 1.0 for i in range(n)],
            active_power_kw=[100.0 * i for i in range(n)],
            pitch_angle_deg=[i * 0.5 for i in range(n)],
        )
        result = _flag_frozen_sensors(df, threshold_minutes=60)
        assert result["qc_flag"].max() == QC_OK


class TestDetectCurtailment:
    def test_curtailment_detected(self):
        # High wind + high pitch + low power = curtailment
        df = _make_scada_df(
            3,
            wind_speed_ms=[12.0, 12.0, 5.0],
            pitch_angle_deg=[5.0, 5.0, 1.0],
            active_power_kw=[500.0, 500.0, 1000.0],
        )
        result = _detect_curtailment(df, rated_power_kw=2050.0, min_pitch_deg=3.0)
        assert result["is_curtailed"][0] is True
        assert result["is_curtailed"][2] is False

    def test_normal_operation_not_curtailed(self):
        df = _make_scada_df(3, wind_speed_ms=[10.0, 10.0, 10.0])
        result = _detect_curtailment(df, rated_power_kw=2050.0, min_pitch_deg=3.0)
        assert not any(result["is_curtailed"].to_list())


class TestFillSmallGaps:
    def test_small_gap_filled(self):
        df = _make_scada_df(
            5,
            active_power_kw=[1000.0, None, None, 1200.0, 1300.0],
        )
        result = _fill_small_gaps(df, max_gap_minutes=30)
        # 2 consecutive nulls (20 min) < 30 min → should be filled
        assert result["active_power_kw"][1] == 1000.0

    def test_large_gap_preserved(self):
        df = _make_scada_df(
            6,
            active_power_kw=[1000.0, None, None, None, None, 1200.0],
        )
        result = _fill_small_gaps(df, max_gap_minutes=30)
        # 4 consecutive nulls (40 min) > 30 min → last ones should stay null
        # forward_fill(limit=3) fills first 3, leaves 4th as null
        assert result["active_power_kw"][4] is None


class TestRunQcPipeline:
    def test_full_pipeline_runs(self):
        df = _make_scada_df(10)
        result = run_qc_pipeline(df, rated_power_kw=2050.0)
        assert len(result) == 10
        assert "qc_flag" in result.columns

    def test_worst_flag_wins(self):
        # Negative power (qc=2) should dominate
        df = _make_scada_df(3, active_power_kw=[-50.0, 100.0, 2200.0])
        result = run_qc_pipeline(df, rated_power_kw=2050.0)
        assert result["qc_flag"][0] == QC_BAD
        assert result["qc_flag"][2] == QC_SUSPECT


class TestQcSummary:
    def test_summary_counts(self):
        df = _make_scada_df(
            5,
            qc_flag=[0, 0, 1, 2, 0],
            is_curtailed=[False, False, False, False, True],
            is_maintenance=[False, True, False, False, False],
        )
        summary = qc_summary(df)
        assert summary["total_rows"] == 5
        assert summary["qc_ok"] == 3
        assert summary["qc_suspect"] == 1
        assert summary["qc_bad"] == 1
        assert summary["curtailed"] == 1
        assert summary["maintenance"] == 1

    def test_empty_summary(self):
        df = _make_scada_df(0)
        summary = qc_summary(df)
        assert summary["total_rows"] == 0
