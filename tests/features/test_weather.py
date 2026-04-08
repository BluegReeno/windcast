"""Tests for windcast.features.weather — NWP horizon feature joiner."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from windcast.features.weather import _resample_nwp, join_nwp_horizon_features


def _make_scada_df(n_rows: int = 48, resolution_min: int = 10) -> pl.DataFrame:
    """Create a minimal SCADA-like DataFrame at given resolution."""
    return pl.DataFrame(
        {
            "timestamp_utc": [
                datetime(2023, 1, 1, tzinfo=UTC) + timedelta(minutes=i * resolution_min)
                for i in range(n_rows)
            ],
            "active_power_kw": [500.0 + i for i in range(n_rows)],
        }
    )


def _make_nwp_df(n_hours: int = 24) -> pl.DataFrame:
    """Create hourly NWP DataFrame with two variables."""
    return pl.DataFrame(
        {
            "timestamp_utc": [
                datetime(2023, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(n_hours)
            ],
            "wind_speed_100m": [10.0 + 0.5 * i for i in range(n_hours)],
            "temperature_2m": [5.0 + 0.1 * i for i in range(n_hours)],
        }
    )


class TestJoinNwpBasic:
    def test_single_horizon_creates_columns(self):
        scada = _make_scada_df(n_rows=48, resolution_min=60)
        nwp = _make_nwp_df(n_hours=48)
        result = join_nwp_horizon_features(scada, nwp, horizons=[1], resolution_minutes=60)

        assert "nwp_wind_speed_100m_h1" in result.columns
        assert "nwp_temperature_2m_h1" in result.columns
        assert len(result) == len(scada)

    def test_single_horizon_values_shifted(self):
        """At row t, NWP should be the value from t + 1 step."""
        scada = _make_scada_df(n_rows=10, resolution_min=60)
        nwp = _make_nwp_df(n_hours=12)
        result = join_nwp_horizon_features(scada, nwp, horizons=[1], resolution_minutes=60)

        # NWP at h1 for row 0 (00:00) should be NWP value at 01:00 = 10.5
        val = result.filter(
            pl.col("timestamp_utc") == datetime(2023, 1, 1, 0, 0, tzinfo=UTC)
        ).get_column("nwp_wind_speed_100m_h1")[0]
        assert val == pytest.approx(10.5)  # 10.0 + 0.5 * 1

    def test_custom_prefix(self):
        scada = _make_scada_df(n_rows=10, resolution_min=60)
        nwp = _make_nwp_df(n_hours=12)
        result = join_nwp_horizon_features(
            scada, nwp, horizons=[1], resolution_minutes=60, prefix="wx_"
        )
        assert "wx_wind_speed_100m_h1" in result.columns

    def test_explicit_nwp_columns(self):
        scada = _make_scada_df(n_rows=10, resolution_min=60)
        nwp = _make_nwp_df(n_hours=12)
        result = join_nwp_horizon_features(
            scada, nwp, horizons=[1], resolution_minutes=60, nwp_columns=["wind_speed_100m"]
        )
        assert "nwp_wind_speed_100m_h1" in result.columns
        assert "nwp_temperature_2m_h1" not in result.columns


class TestJoinNwpMultiHorizon:
    def test_multi_horizon_column_count(self):
        scada = _make_scada_df(n_rows=48, resolution_min=60)
        nwp = _make_nwp_df(n_hours=96)
        horizons = [1, 6, 12, 24, 48]
        result = join_nwp_horizon_features(scada, nwp, horizons=horizons, resolution_minutes=60)

        # 2 NWP vars x 5 horizons = 10 new columns
        new_cols = [c for c in result.columns if c.startswith("nwp_")]
        assert len(new_cols) == 10

    def test_different_horizons_have_different_values(self):
        scada = _make_scada_df(n_rows=10, resolution_min=60)
        nwp = _make_nwp_df(n_hours=50)
        result = join_nwp_horizon_features(scada, nwp, horizons=[1, 6], resolution_minutes=60)

        row0 = result.filter(pl.col("timestamp_utc") == datetime(2023, 1, 1, 0, 0, tzinfo=UTC))
        h1_val = row0.get_column("nwp_wind_speed_100m_h1")[0]
        h6_val = row0.get_column("nwp_wind_speed_100m_h6")[0]
        # h1 = NWP at 01:00, h6 = NWP at 06:00 — different values
        assert h1_val != h6_val
        assert h1_val == pytest.approx(10.5)  # 10.0 + 0.5 * 1
        assert h6_val == pytest.approx(13.0)  # 10.0 + 0.5 * 6


class TestJoinNwpResolution10min:
    def test_hourly_nwp_to_10min_scada(self):
        """NWP is hourly, SCADA is 10min — NWP should be forward-filled."""
        scada = _make_scada_df(n_rows=18, resolution_min=10)  # 3 hours
        nwp = _make_nwp_df(n_hours=6)
        result = join_nwp_horizon_features(scada, nwp, horizons=[6], resolution_minutes=10)

        # h=6 at 10min resolution = 60min offset
        # Row at 00:00 should get NWP value at 01:00 = 10.5
        row0 = result.filter(pl.col("timestamp_utc") == datetime(2023, 1, 1, 0, 0, tzinfo=UTC))
        assert row0.get_column("nwp_wind_speed_100m_h6")[0] == pytest.approx(10.5)

        # Row at 00:10 should also get NWP at 01:10 — forward-filled from 01:00
        row1 = result.filter(pl.col("timestamp_utc") == datetime(2023, 1, 1, 0, 10, tzinfo=UTC))
        assert row1.get_column("nwp_wind_speed_100m_h6")[0] == pytest.approx(10.5)

    def test_no_nwp_data_outside_range_is_null(self):
        """Rows whose shifted NWP falls outside the NWP time range should be null."""
        scada = _make_scada_df(n_rows=6, resolution_min=60)  # 6h
        nwp = _make_nwp_df(n_hours=6)  # 0h-5h
        result = join_nwp_horizon_features(scada, nwp, horizons=[6], resolution_minutes=60)

        # h=6 at 60min = 6h offset. Row at 00:00 needs NWP at 06:00, but NWP ends at 05:00
        nwp_col = result.get_column("nwp_wind_speed_100m_h6")
        assert nwp_col.null_count() > 0


class TestJoinNwpResolution60min:
    def test_hourly_nwp_to_hourly_demand_no_resample(self):
        """When both NWP and data are hourly, no resampling should occur."""
        demand = _make_scada_df(n_rows=24, resolution_min=60)
        nwp = _make_nwp_df(n_hours=48)
        result = join_nwp_horizon_features(demand, nwp, horizons=[1], resolution_minutes=60)

        assert "nwp_wind_speed_100m_h1" in result.columns
        assert len(result) == 24


class TestResampleNwp:
    def test_forward_fill_10min(self):
        nwp = _make_nwp_df(n_hours=3)  # 3 hourly rows
        result = _resample_nwp(nwp, target_resolution_minutes=10)

        # 3 hours at 10min = 13 rows (00:00 to 02:00 inclusive at 10min)
        assert len(result) == 13

        # Forward-fill: 00:10 should have same value as 00:00
        row_10 = result.filter(pl.col("timestamp_utc") == datetime(2023, 1, 1, 0, 10, tzinfo=UTC))
        row_00 = result.filter(pl.col("timestamp_utc") == datetime(2023, 1, 1, 0, 0, tzinfo=UTC))
        assert row_10.get_column("wind_speed_100m")[0] == row_00.get_column("wind_speed_100m")[0]

    def test_no_resample_when_coarser(self):
        """If target >= NWP resolution, return unchanged."""
        nwp = _make_nwp_df(n_hours=5)
        result = _resample_nwp(nwp, target_resolution_minutes=60)
        assert len(result) == len(nwp)

    def test_no_resample_when_equal(self):
        nwp = _make_nwp_df(n_hours=5)
        result = _resample_nwp(nwp, target_resolution_minutes=60, nwp_resolution_minutes=60)
        assert len(result) == len(nwp)

    def test_forward_fill_15min(self):
        """15-min resolution (solar PVDAQ)."""
        nwp = _make_nwp_df(n_hours=2)  # 2 hourly rows
        result = _resample_nwp(nwp, target_resolution_minutes=15)
        # 2 hours at 15min = 5 rows (00:00, 00:15, 00:30, 00:45, 01:00)
        assert len(result) == 5
