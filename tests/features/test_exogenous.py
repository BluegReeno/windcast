"""Tests for windcast.features.exogenous module."""

import math
from datetime import UTC, datetime, timedelta

import polars as pl

from windcast.data.schema import QC_BAD, QC_OK
from windcast.features.exogenous import (
    build_demand_exogenous,
    build_solar_exogenous,
    build_wind_exogenous,
)


def _make_wind_df(n_rows: int = 200) -> pl.DataFrame:
    """Create a SCADA DataFrame for wind exogenous testing."""
    return pl.DataFrame(
        {
            "timestamp_utc": [
                datetime(2021, 6, 15, tzinfo=UTC) + timedelta(minutes=i * 10) for i in range(n_rows)
            ],
            "turbine_id": ["KWF1"] * n_rows,
            "active_power_kw": [500.0 + 100.0 * math.sin(i * 0.1) for i in range(n_rows)],
            "wind_speed_ms": [8.0 + 2.0 * math.sin(i * 0.05) for i in range(n_rows)],
            "wind_direction_deg": [(180.0 + i * 1.5) % 360.0 for i in range(n_rows)],
            "qc_flag": [QC_OK] * n_rows,
        }
    ).cast({"timestamp_utc": pl.Datetime("us", "UTC"), "qc_flag": pl.UInt8})


def _make_demand_df(n_rows: int = 200) -> pl.DataFrame:
    """Create a demand DataFrame for exogenous testing."""
    return pl.DataFrame(
        {
            "timestamp_utc": [
                datetime(2021, 6, 15, tzinfo=UTC) + timedelta(hours=i) for i in range(n_rows)
            ],
            "zone_id": ["ES"] * n_rows,
            "load_mw": [25000.0 + 5000.0 * math.sin(i * 0.1) for i in range(n_rows)],
            "temperature_c": [20.0 + 10.0 * math.sin(i * 0.05) for i in range(n_rows)],
            "wind_speed_ms": [5.0] * n_rows,
            "humidity_pct": [60.0] * n_rows,
            "price_eur_mwh": [50.0 + i * 0.1 for i in range(n_rows)],
            "is_holiday": [False] * n_rows,
            "qc_flag": [QC_OK] * n_rows,
        }
    ).cast({"timestamp_utc": pl.Datetime("us", "UTC"), "qc_flag": pl.UInt8})


def _make_solar_df(n_rows: int = 200) -> pl.DataFrame:
    """Create a solar DataFrame for exogenous testing."""
    return pl.DataFrame(
        {
            "timestamp_utc": [
                datetime(2021, 6, 15, tzinfo=UTC) + timedelta(minutes=i * 15) for i in range(n_rows)
            ],
            "system_id": ["sys4"] * n_rows,
            "power_kw": [50.0 + 30.0 * math.sin(i * 0.1) for i in range(n_rows)],
            "poa_wm2": [500.0 + 300.0 * max(0, math.sin(i * 0.05)) for i in range(n_rows)],
            "ambient_temp_c": [25.0] * n_rows,
            "module_temp_c": [35.0] * n_rows,
            "ghi_wm2": [400.0] * n_rows,
            "wind_speed_ms": [3.0] * n_rows,
            "qc_flag": [QC_OK] * n_rows,
        }
    ).cast({"timestamp_utc": pl.Datetime("us", "UTC"), "qc_flag": pl.UInt8})


class TestBuildWindExogenous:
    def test_baseline_columns(self):
        df = _make_wind_df()
        result = build_wind_exogenous(df, "wind_exog_baseline")
        assert "wind_dir_sin" in result.columns
        assert "wind_dir_cos" in result.columns

    def test_baseline_no_lag_columns(self):
        df = _make_wind_df()
        result = build_wind_exogenous(df, "wind_exog_baseline")
        lag_cols = [c for c in result.columns if "_lag" in c or "_roll_" in c]
        assert lag_cols == [], f"Unexpected lag/rolling columns: {lag_cols}"

    def test_enriched_columns(self):
        df = _make_wind_df()
        result = build_wind_exogenous(df, "wind_exog_enriched")
        assert "wind_speed_cubed" in result.columns
        assert "turbulence_intensity" in result.columns
        assert "wind_dir_sector" in result.columns
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns

    def test_full_columns(self):
        df = _make_wind_df()
        result = build_wind_exogenous(df, "wind_exog_full")
        assert "month_sin" in result.columns
        assert "month_cos" in result.columns
        assert "dow_sin" in result.columns
        assert "dow_cos" in result.columns

    def test_qc_filter(self):
        df = _make_wind_df(100)
        # Set first 10 rows to QC_BAD
        qc_flags = [QC_BAD] * 10 + [QC_OK] * 90
        df = df.with_columns(pl.Series("qc_flag", qc_flags, dtype=pl.UInt8))
        result = build_wind_exogenous(df, "wind_exog_baseline")
        assert len(result) == 90


class TestBuildDemandExogenous:
    def test_baseline_columns(self):
        df = _make_demand_df()
        result = build_demand_exogenous(df, "demand_exog_baseline")
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "dow_sin" in result.columns
        assert "month_sin" in result.columns

    def test_baseline_no_lag_columns(self):
        df = _make_demand_df()
        result = build_demand_exogenous(df, "demand_exog_baseline")
        lag_cols = [c for c in result.columns if "load_mw_lag" in c or "load_mw_roll_" in c]
        assert lag_cols == [], f"Unexpected lag/rolling columns: {lag_cols}"

    def test_enriched_columns(self):
        df = _make_demand_df()
        result = build_demand_exogenous(df, "demand_exog_enriched")
        assert "heating_degree_days" in result.columns
        assert "cooling_degree_days" in result.columns

    def test_full_columns(self):
        df = _make_demand_df()
        result = build_demand_exogenous(df, "demand_exog_full")
        assert "price_lag1" in result.columns
        assert "price_lag24" in result.columns
        assert "is_holiday" in result.columns
        # is_holiday should be Int8 for XGBoost
        assert result["is_holiday"].dtype == pl.Int8

    def test_qc_filter(self):
        df = _make_demand_df(100)
        qc_flags = [QC_BAD] * 20 + [QC_OK] * 80
        df = df.with_columns(pl.Series("qc_flag", qc_flags, dtype=pl.UInt8))
        result = build_demand_exogenous(df, "demand_exog_baseline")
        assert len(result) == 80


class TestBuildSolarExogenous:
    def test_baseline_columns(self):
        df = _make_solar_df()
        result = build_solar_exogenous(df, "solar_exog_baseline")
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns

    def test_baseline_no_lag_columns(self):
        df = _make_solar_df()
        result = build_solar_exogenous(df, "solar_exog_baseline")
        lag_cols = [c for c in result.columns if "power_kw_lag" in c or "power_kw_roll_" in c]
        assert lag_cols == [], f"Unexpected lag/rolling columns: {lag_cols}"

    def test_enriched_columns(self):
        df = _make_solar_df()
        result = build_solar_exogenous(df, "solar_exog_enriched")
        assert "clearsky_ratio" in result.columns

    def test_full_columns(self):
        df = _make_solar_df()
        result = build_solar_exogenous(df, "solar_exog_full")
        assert "month_sin" in result.columns
        assert "dow_sin" in result.columns

    def test_qc_filter(self):
        df = _make_solar_df(100)
        qc_flags = [QC_BAD] * 15 + [QC_OK] * 85
        df = df.with_columns(pl.Series("qc_flag", qc_flags, dtype=pl.UInt8))
        result = build_solar_exogenous(df, "solar_exog_baseline")
        assert len(result) == 85
