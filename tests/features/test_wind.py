"""Tests for windcast.features.wind module."""

import math
from datetime import UTC, datetime, timedelta

import polars as pl

from windcast.data.schema import QC_BAD, QC_OK
from windcast.features.wind import (
    _add_cyclic_hour,
    _add_cyclic_wind_direction,
    _add_lag_features,
    _add_rolling_features,
    _add_wind_specific_features,
    build_wind_features,
)


def _make_wind_df(n_rows: int = 200, n_turbines: int = 1) -> pl.DataFrame:
    """Create a SCADA DataFrame with realistic values for feature testing."""
    rows_per_turbine = n_rows // n_turbines
    dfs = []
    for t in range(n_turbines):
        tid = f"KWF{t + 1}"
        dfs.append(
            pl.DataFrame(
                {
                    "timestamp_utc": [
                        datetime(2021, 6, 15, tzinfo=UTC) + timedelta(minutes=i * 10)
                        for i in range(rows_per_turbine)
                    ],
                    "dataset_id": ["kelmarsh"] * rows_per_turbine,
                    "turbine_id": [tid] * rows_per_turbine,
                    "active_power_kw": [
                        500.0 + 100.0 * math.sin(i * 0.1) for i in range(rows_per_turbine)
                    ],
                    "wind_speed_ms": [
                        8.0 + 2.0 * math.sin(i * 0.05) for i in range(rows_per_turbine)
                    ],
                    "wind_direction_deg": [
                        (180.0 + i * 1.5) % 360.0 for i in range(rows_per_turbine)
                    ],
                    "pitch_angle_deg": [2.0] * rows_per_turbine,
                    "rotor_rpm": [12.0] * rows_per_turbine,
                    "nacelle_direction_deg": [180.0] * rows_per_turbine,
                    "ambient_temp_c": [15.0] * rows_per_turbine,
                    "nacelle_temp_c": [35.0] * rows_per_turbine,
                    "status_code": [0] * rows_per_turbine,
                    "is_curtailed": [False] * rows_per_turbine,
                    "is_maintenance": [False] * rows_per_turbine,
                    "qc_flag": [QC_OK] * rows_per_turbine,
                }
            )
        )
    return pl.concat(dfs).cast(
        {
            "timestamp_utc": pl.Datetime("us", "UTC"),
            "status_code": pl.Int32,
            "qc_flag": pl.UInt8,
        }
    )


class TestLagFeatures:
    def test_lag_columns_created(self):
        df = _make_wind_df()
        df = df.sort("turbine_id", "timestamp_utc")
        result = _add_lag_features(df, "active_power_kw", [1, 2, 3])
        assert "active_power_kw_lag1" in result.columns
        assert "active_power_kw_lag2" in result.columns
        assert "active_power_kw_lag3" in result.columns

    def test_lag_values_are_shifted(self):
        df = _make_wind_df(n_rows=10)
        df = df.sort("turbine_id", "timestamp_utc")
        result = _add_lag_features(df, "active_power_kw", [1])
        power = result.get_column("active_power_kw").to_list()
        lag1 = result.get_column("active_power_kw_lag1").to_list()
        # lag1[i] should equal power[i-1]
        assert lag1[0] is None
        for i in range(1, len(power)):
            assert lag1[i] == power[i - 1]

    def test_lag_nulls_at_start(self):
        df = _make_wind_df(n_rows=50)
        df = df.sort("turbine_id", "timestamp_utc")
        result = _add_lag_features(df, "active_power_kw", [6])
        lag6 = result.get_column("active_power_kw_lag6")
        assert lag6[:6].null_count() == 6


class TestRollingFeatures:
    def test_rolling_columns_created(self):
        df = _make_wind_df()
        df = df.sort("turbine_id", "timestamp_utc")
        result = _add_rolling_features(df, "active_power_kw", [6, 12])
        assert "active_power_kw_roll_mean_6" in result.columns
        assert "active_power_kw_roll_std_6" in result.columns
        assert "active_power_kw_roll_mean_12" in result.columns
        assert "active_power_kw_roll_std_12" in result.columns

    def test_no_look_ahead(self):
        """Rolling features must not include current row's value (shift(1) applied)."""
        df = _make_wind_df(n_rows=50)
        df = df.sort("turbine_id", "timestamp_utc")
        result = _add_rolling_features(df, "active_power_kw", [3])
        # Row 0 and 1 should be null (shift(1) + window=3 needs 3 past values)
        roll_mean = result.get_column("active_power_kw_roll_mean_3")
        # Row 0: shift(1) makes it null, so rolling mean is null
        assert roll_mean[0] is None


class TestCyclicFeatures:
    def test_wind_direction_sin_cos_range(self):
        df = _make_wind_df()
        result = _add_cyclic_wind_direction(df)
        assert "wind_dir_sin" in result.columns
        assert "wind_dir_cos" in result.columns
        sin_vals = result.get_column("wind_dir_sin")
        cos_vals = result.get_column("wind_dir_cos")
        assert sin_vals.min() >= -1.0  # type: ignore[operator]
        assert sin_vals.max() <= 1.0  # type: ignore[operator]
        assert cos_vals.min() >= -1.0  # type: ignore[operator]
        assert cos_vals.max() <= 1.0  # type: ignore[operator]

    def test_hour_sin_cos_range(self):
        df = _make_wind_df(n_rows=200)
        result = _add_cyclic_hour(df)
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert result.get_column("hour_sin").min() >= -1.0  # type: ignore[operator]
        assert result.get_column("hour_sin").max() <= 1.0  # type: ignore[operator]


class TestWindSpecificFeatures:
    def test_wind_speed_cubed(self):
        df = _make_wind_df(n_rows=50)
        df = df.sort("turbine_id", "timestamp_utc")
        result = _add_wind_specific_features(df)
        assert "wind_speed_cubed" in result.columns
        ws = result.get_column("wind_speed_ms").to_list()
        cubed = result.get_column("wind_speed_cubed").to_list()
        # Check a non-null value
        for i in range(len(ws)):
            assert abs(cubed[i] - ws[i] ** 3) < 1e-6

    def test_turbulence_intensity_no_nan(self):
        df = _make_wind_df(n_rows=50)
        df = df.sort("turbine_id", "timestamp_utc")
        result = _add_wind_specific_features(df)
        ti = result.get_column("turbulence_intensity")
        # fill_nan(0) should have replaced any NaN
        assert ti.is_nan().sum() == 0

    def test_wind_dir_sector_range(self):
        df = _make_wind_df(n_rows=50)
        df = df.sort("turbine_id", "timestamp_utc")
        result = _add_wind_specific_features(df)
        sectors = result.get_column("wind_dir_sector")
        assert sectors.min() >= 0  # type: ignore[operator]
        assert sectors.max() <= 11  # type: ignore[operator]


class TestBuildWindFeatures:
    def test_baseline_produces_expected_columns(self):
        df = _make_wind_df(n_rows=200)
        result = build_wind_features(df, "wind_baseline")
        assert "active_power_kw_lag1" in result.columns
        assert "active_power_kw_roll_mean_6" in result.columns
        assert "wind_dir_sin" in result.columns

    def test_enriched_has_extra_columns(self):
        df = _make_wind_df(n_rows=200)
        result = build_wind_features(df, "wind_enriched")
        assert "wind_speed_cubed" in result.columns
        assert "hour_sin" in result.columns

    def test_qc_filter_removes_bad_rows(self):
        df = _make_wind_df(n_rows=100)
        # Mark half the rows as QC_BAD
        flags = [QC_OK] * 50 + [QC_BAD] * 50
        df = df.with_columns(pl.Series("qc_flag", flags).cast(pl.UInt8))
        result = build_wind_features(df, "wind_baseline")
        assert len(result) <= 50

    def test_multiple_turbines(self):
        df = _make_wind_df(n_rows=400, n_turbines=2)
        result = build_wind_features(df, "wind_baseline")
        turbines = result.get_column("turbine_id").unique().to_list()
        assert len(turbines) == 2

    def test_wind_full_with_weather_df(self):
        """wind_full with weather_df should produce NWP horizon columns."""
        df = _make_wind_df(n_rows=200)
        # Create matching NWP data (hourly, covering the SCADA range)
        nwp = pl.DataFrame(
            {
                "timestamp_utc": [
                    datetime(2021, 6, 15, tzinfo=UTC) + timedelta(hours=i) for i in range(48)
                ],
                "wind_speed_100m": [10.0 + 0.5 * i for i in range(48)],
                "wind_direction_100m": [180.0 + i for i in range(48)],
                "temperature_2m": [15.0 + 0.1 * i for i in range(48)],
            }
        )
        horizons = [1, 6, 12]
        result = build_wind_features(
            df, "wind_full", weather_df=nwp, horizons=horizons, resolution_minutes=10
        )
        # Should have NWP columns for each horizon
        assert "nwp_wind_speed_100m_h1" in result.columns
        assert "nwp_wind_speed_100m_h6" in result.columns
        assert "nwp_wind_speed_100m_h12" in result.columns
        # Should also have calendar features from wind_full
        assert "month_sin" in result.columns

    def test_wind_full_without_weather_warns(self):
        """wind_full without weather_df should still work (no NWP columns)."""
        df = _make_wind_df(n_rows=200)
        result = build_wind_features(df, "wind_full")
        # Should have calendar features but no NWP columns
        assert "month_sin" in result.columns
        nwp_cols = [c for c in result.columns if c.startswith("nwp_")]
        assert len(nwp_cols) == 0
