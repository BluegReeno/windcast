"""Tests for windcast.features.demand module."""

import math
from datetime import UTC, datetime, timedelta

import polars as pl

from windcast.data.schema import QC_BAD, QC_OK
from windcast.features.demand import (
    _add_cyclic_calendar,
    _add_lag_features,
    _add_rolling_features,
    _add_temperature_features,
    build_demand_features,
)


def _make_demand_df(n_rows: int = 200, n_zones: int = 1) -> pl.DataFrame:
    """Create a demand DataFrame with realistic hourly data for feature testing."""
    rows_per_zone = n_rows // n_zones
    dfs = []
    for z in range(n_zones):
        zone_id = f"Z{z + 1}"
        dfs.append(
            pl.DataFrame(
                {
                    "timestamp_utc": [
                        datetime(2015, 6, 15, tzinfo=UTC) + timedelta(hours=i)
                        for i in range(rows_per_zone)
                    ],
                    "dataset_id": ["spain_demand"] * rows_per_zone,
                    "zone_id": [zone_id] * rows_per_zone,
                    "load_mw": [
                        25000.0 + 5000.0 * math.sin(i * math.pi / 12) for i in range(rows_per_zone)
                    ],
                    "temperature_c": [
                        20.0 + 10.0 * math.sin(i * math.pi / 12) for i in range(rows_per_zone)
                    ],
                    "wind_speed_ms": [5.0 + 2.0 * math.sin(i * 0.1) for i in range(rows_per_zone)],
                    "humidity_pct": [65.0] * rows_per_zone,
                    "price_eur_mwh": [40.0 + i * 0.1 for i in range(rows_per_zone)],
                    "is_holiday": [False] * rows_per_zone,
                    "is_dst_transition": [False] * rows_per_zone,
                    "qc_flag": [QC_OK] * rows_per_zone,
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
        df = _make_demand_df(n_rows=50)
        df = df.sort("zone_id", "timestamp_utc")
        result = _add_lag_features(df, "load_mw", [1])
        load = result.get_column("load_mw").to_list()
        lag1 = result.get_column("load_mw_lag1").to_list()
        assert lag1[0] is None
        for i in range(1, len(load)):
            assert lag1[i] == load[i - 1]

    def test_lag_per_zone(self):
        df = _make_demand_df(n_rows=100, n_zones=2)
        df = df.sort("zone_id", "timestamp_utc")
        result = _add_lag_features(df, "load_mw", [1])
        # First row of each zone should be null
        z1 = result.filter(pl.col("zone_id") == "Z1")
        z2 = result.filter(pl.col("zone_id") == "Z2")
        assert z1["load_mw_lag1"][0] is None
        assert z2["load_mw_lag1"][0] is None

    def test_weekly_lag(self):
        df = _make_demand_df(n_rows=200)
        df = df.sort("zone_id", "timestamp_utc")
        result = _add_lag_features(df, "load_mw", [168])
        lag168 = result.get_column("load_mw_lag168")
        # First 168 rows should be null
        assert lag168[:168].null_count() == 168
        # Row 168 should equal row 0 value
        assert lag168[168] == result["load_mw"][0]


class TestRollingFeatures:
    def test_no_look_ahead(self):
        df = _make_demand_df(n_rows=50)
        df = df.sort("zone_id", "timestamp_utc")
        result = _add_rolling_features(df, "load_mw", [24])
        roll_mean = result.get_column("load_mw_roll_mean_24")
        # Row 0: shift(1) → null
        assert roll_mean[0] is None

    def test_rolling_window_size(self):
        df = _make_demand_df(n_rows=50)
        df = df.sort("zone_id", "timestamp_utc")
        result = _add_rolling_features(df, "load_mw", [24])
        assert "load_mw_roll_mean_24" in result.columns
        assert "load_mw_roll_std_24" in result.columns


class TestCalendarFeatures:
    def test_hour_cyclic_range(self):
        df = _make_demand_df(n_rows=200)
        result = _add_cyclic_calendar(df)
        assert result.get_column("hour_sin").min() >= -1.0  # type: ignore[operator]
        assert result.get_column("hour_sin").max() <= 1.0  # type: ignore[operator]
        assert result.get_column("hour_cos").min() >= -1.0  # type: ignore[operator]
        assert result.get_column("hour_cos").max() <= 1.0  # type: ignore[operator]

    def test_dow_cyclic_range(self):
        df = _make_demand_df(n_rows=200)
        result = _add_cyclic_calendar(df)
        assert result.get_column("dow_sin").min() >= -1.0  # type: ignore[operator]
        assert result.get_column("dow_sin").max() <= 1.0  # type: ignore[operator]


class TestTemperatureFeatures:
    def test_hdd_positive_when_cold(self):
        df = _make_demand_df(n_rows=3)
        df = df.with_columns(pl.Series("temperature_c", [5.0, 10.0, 15.0]))
        result = _add_temperature_features(df)
        hdd = result.get_column("heating_degree_days").to_list()
        assert hdd[0] == 13.0  # 18 - 5
        assert hdd[1] == 8.0  # 18 - 10
        assert hdd[2] == 3.0  # 18 - 15

    def test_cdd_positive_when_hot(self):
        df = _make_demand_df(n_rows=3)
        df = df.with_columns(pl.Series("temperature_c", [25.0, 30.0, 35.0]))
        result = _add_temperature_features(df)
        cdd = result.get_column("cooling_degree_days").to_list()
        assert cdd[0] == 1.0  # 25 - 24
        assert cdd[1] == 6.0  # 30 - 24
        assert cdd[2] == 11.0  # 35 - 24

    def test_hdd_zero_when_warm(self):
        df = _make_demand_df(n_rows=3)
        df = df.with_columns(pl.Series("temperature_c", [20.0, 25.0, 30.0]))
        result = _add_temperature_features(df)
        hdd = result.get_column("heating_degree_days").to_list()
        assert all(h == 0.0 for h in hdd)

    def test_cdd_zero_when_cold(self):
        df = _make_demand_df(n_rows=3)
        df = df.with_columns(pl.Series("temperature_c", [5.0, 10.0, 20.0]))
        result = _add_temperature_features(df)
        cdd = result.get_column("cooling_degree_days").to_list()
        assert all(c == 0.0 for c in cdd)


class TestBuildDemandFeatures:
    def test_qc_filter_removes_bad_rows(self):
        df = _make_demand_df(n_rows=100)
        flags = [QC_OK] * 50 + [QC_BAD] * 50
        df = df.with_columns(pl.Series("qc_flag", flags).cast(pl.UInt8))
        result = build_demand_features(df, "demand_baseline")
        assert len(result) <= 50

    def test_baseline_produces_expected_columns(self):
        df = _make_demand_df(n_rows=200)
        result = build_demand_features(df, "demand_baseline")
        assert "load_mw_lag1" in result.columns
        assert "load_mw_lag24" in result.columns
        assert "hour_sin" in result.columns
        assert "dow_cos" in result.columns

    def test_enriched_extends_baseline(self):
        """Enriched adds rolling stats + holiday flag; no weather yet."""
        df = _make_demand_df(n_rows=200)
        result = build_demand_features(df, "demand_enriched")
        assert "load_mw_roll_mean_24" in result.columns
        assert "load_mw_roll_mean_168" in result.columns
        assert "load_mw_lag1" in result.columns
        assert "is_holiday" in result.columns
        # HDD/CDD belong to `demand_full`, not `demand_enriched`
        assert "heating_degree_days" not in result.columns
        assert "cooling_degree_days" not in result.columns

    def test_full_uses_nwp_for_hdd_cdd(self):
        """demand_full computes HDD/CDD from nwp_temperature_2m_h1 when present."""
        df = _make_demand_df(n_rows=200)
        df = df.with_columns(
            pl.Series("nwp_temperature_2m_h1", [10.0] * 200),
        )
        result = build_demand_features(df, "demand_full")
        assert "heating_degree_days" in result.columns
        assert "cooling_degree_days" in result.columns
        # 18 - 10 = 8 heating degree days
        assert result["heating_degree_days"][0] == 8.0
        assert result["cooling_degree_days"][0] == 0.0

    def test_full_falls_back_to_observed_temperature(self):
        """Without NWP columns, demand_full falls back to the observed temperature_c."""
        df = _make_demand_df(n_rows=200)
        # Force a constant cold observed temperature
        df = df.with_columns(pl.lit(5.0).alias("temperature_c"))
        result = build_demand_features(df, "demand_full")
        assert "heating_degree_days" in result.columns
        # 18 - 5 = 13 heating degree days
        assert result["heating_degree_days"][0] == 13.0

    def test_full_has_holiday_flag(self):
        df = _make_demand_df(n_rows=200)
        result = build_demand_features(df, "demand_full")
        assert "is_holiday" in result.columns
