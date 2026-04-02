"""Tests for windcast.features.solar module."""

import math
from datetime import UTC, datetime, timedelta

import polars as pl

from windcast.data.schema import QC_BAD, QC_OK
from windcast.features.solar import (
    _add_clearsky_ratio,
    _add_cyclic_hour,
    _add_lag_features,
    _add_rolling_features,
    build_solar_features,
)


def _make_solar_df(n_rows: int = 200, n_systems: int = 1) -> pl.DataFrame:
    """Create a solar DataFrame with realistic 15-min data for feature testing."""
    rows_per_system = n_rows // n_systems
    dfs = []
    for s in range(n_systems):
        system_id = f"S{s + 1}"
        dfs.append(
            pl.DataFrame(
                {
                    "timestamp_utc": [
                        datetime(2020, 6, 15, 6, tzinfo=UTC) + timedelta(minutes=15 * i)
                        for i in range(rows_per_system)
                    ],
                    "dataset_id": ["pvdaq_system4"] * rows_per_system,
                    "system_id": [system_id] * rows_per_system,
                    "power_kw": [
                        max(0.0, 1.5 * math.sin(i * math.pi / 48)) for i in range(rows_per_system)
                    ],
                    "ghi_wm2": [None] * rows_per_system,
                    "poa_wm2": [
                        max(0.0, 800.0 * math.sin(i * math.pi / 48)) for i in range(rows_per_system)
                    ],
                    "ambient_temp_c": [
                        20.0 + 10.0 * math.sin(i * math.pi / 48) for i in range(rows_per_system)
                    ],
                    "module_temp_c": [
                        30.0 + 15.0 * math.sin(i * math.pi / 48) for i in range(rows_per_system)
                    ],
                    "wind_speed_ms": [None] * rows_per_system,
                    "qc_flag": [QC_OK] * rows_per_system,
                }
            )
        )
    return pl.concat(dfs).cast(
        {
            "timestamp_utc": pl.Datetime("us", "UTC"),
            "qc_flag": pl.UInt8,
        }
    )


class TestLagFeatures:
    def test_lag_values_are_shifted(self):
        df = _make_solar_df(n_rows=50)
        df = df.sort("system_id", "timestamp_utc")
        result = _add_lag_features(df, "power_kw", [1])
        power = result.get_column("power_kw").to_list()
        lag1 = result.get_column("power_kw_lag1").to_list()
        assert lag1[0] is None
        for i in range(1, len(power)):
            assert lag1[i] == power[i - 1]

    def test_lag_per_system(self):
        df = _make_solar_df(n_rows=100, n_systems=2)
        df = df.sort("system_id", "timestamp_utc")
        result = _add_lag_features(df, "power_kw", [1])
        s1 = result.filter(pl.col("system_id") == "S1")
        s2 = result.filter(pl.col("system_id") == "S2")
        assert s1["power_kw_lag1"][0] is None
        assert s2["power_kw_lag1"][0] is None


class TestRollingFeatures:
    def test_no_look_ahead(self):
        df = _make_solar_df(n_rows=50)
        df = df.sort("system_id", "timestamp_utc")
        result = _add_rolling_features(df, "power_kw", [4])
        roll_mean = result.get_column("power_kw_roll_mean_4")
        assert roll_mean[0] is None

    def test_rolling_window_columns(self):
        df = _make_solar_df(n_rows=50)
        df = df.sort("system_id", "timestamp_utc")
        result = _add_rolling_features(df, "power_kw", [4, 16])
        assert "power_kw_roll_mean_4" in result.columns
        assert "power_kw_roll_std_4" in result.columns
        assert "power_kw_roll_mean_16" in result.columns


class TestClearskyRatio:
    def test_ratio_computed(self):
        df = _make_solar_df(n_rows=10)
        result = _add_clearsky_ratio(df)
        assert "clearsky_ratio" in result.columns
        # POA ~800 → ratio = 800/1200 ≈ 0.67
        ratios = result.filter(pl.col("poa_wm2") > 0)["clearsky_ratio"]
        assert ratios.max() <= 1.5  # type: ignore[operator]

    def test_ratio_capped(self):
        df = _make_solar_df(n_rows=3)
        df = df.with_columns(pl.Series("poa_wm2", [2000.0, 1500.0, 600.0]))
        result = _add_clearsky_ratio(df)
        assert result["clearsky_ratio"][0] == 1.5  # capped

    def test_zero_irradiance_gives_zero(self):
        df = _make_solar_df(n_rows=3)
        df = df.with_columns(pl.Series("poa_wm2", [0.0, 0.0, 0.0]))
        result = _add_clearsky_ratio(df)
        assert result["clearsky_ratio"].to_list() == [0.0, 0.0, 0.0]


class TestCalendarFeatures:
    def test_hour_cyclic_range(self):
        df = _make_solar_df(n_rows=200)
        result = _add_cyclic_hour(df)
        assert result.get_column("hour_sin").min() >= -1.0  # type: ignore[operator]
        assert result.get_column("hour_sin").max() <= 1.0  # type: ignore[operator]
        assert result.get_column("hour_cos").min() >= -1.0  # type: ignore[operator]
        assert result.get_column("hour_cos").max() <= 1.0  # type: ignore[operator]


class TestBuildSolarFeatures:
    def test_qc_filter_removes_bad_rows(self):
        df = _make_solar_df(n_rows=100)
        flags = [QC_OK] * 50 + [QC_BAD] * 50
        df = df.with_columns(pl.Series("qc_flag", flags).cast(pl.UInt8))
        result = build_solar_features(df, "solar_baseline")
        assert len(result) <= 50

    def test_baseline_produces_expected_columns(self):
        df = _make_solar_df(n_rows=200)
        result = build_solar_features(df, "solar_baseline")
        assert "power_kw_lag1" in result.columns
        assert "power_kw_lag96" in result.columns
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns

    def test_enriched_extends_baseline(self):
        df = _make_solar_df(n_rows=200)
        result = build_solar_features(df, "solar_enriched")
        assert "clearsky_ratio" in result.columns
        assert "power_kw_roll_mean_4" in result.columns
        assert "power_kw_lag1" in result.columns

    def test_full_has_cyclic_calendar(self):
        df = _make_solar_df(n_rows=200)
        result = build_solar_features(df, "solar_full")
        assert "month_sin" in result.columns
        assert "dow_cos" in result.columns
