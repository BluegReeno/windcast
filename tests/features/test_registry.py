"""Tests for windcast.features.registry module."""

import pytest

from windcast.features.registry import (
    FEATURE_REGISTRY,
    FeatureSet,
    get_feature_set,
    list_feature_sets,
)


class TestGetFeatureSet:
    def test_baseline_returns_correct_columns(self):
        fs = get_feature_set("wind_baseline")
        assert fs.name == "wind_baseline"
        assert "wind_speed_ms" in fs.columns
        assert "active_power_kw_lag1" in fs.columns
        assert "wind_dir_sin" in fs.columns

    def test_enriched_extends_baseline(self):
        baseline = get_feature_set("wind_baseline")
        enriched = get_feature_set("wind_enriched")
        # All baseline columns must be in enriched
        for col in baseline.columns:
            assert col in enriched.columns, f"{col} missing from enriched"
        # Enriched has extra columns
        assert len(enriched.columns) > len(baseline.columns)
        assert "wind_speed_cubed" in enriched.columns
        assert "hour_sin" in enriched.columns

    def test_full_extends_enriched(self):
        enriched = get_feature_set("wind_enriched")
        full = get_feature_set("wind_full")
        for col in enriched.columns:
            assert col in full.columns, f"{col} missing from full"
        assert "month_sin" in full.columns
        assert "dow_sin" in full.columns

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown feature set"):
            get_feature_set("nonexistent")


class TestListFeatureSets:
    def test_returns_all_names(self):
        names = list_feature_sets()
        assert "wind_baseline" in names
        assert "wind_enriched" in names
        assert "wind_full" in names
        assert len(names) == len(FEATURE_REGISTRY)


class TestDemandFeatureSets:
    def test_demand_baseline_exists(self):
        fs = get_feature_set("demand_baseline")
        assert fs.name == "demand_baseline"
        assert "load_mw_lag1" in fs.columns

    def test_demand_enriched_extends_baseline(self):
        baseline = get_feature_set("demand_baseline")
        enriched = get_feature_set("demand_enriched")
        for col in baseline.columns:
            assert col in enriched.columns, f"{col} missing from demand_enriched"
        # Enriched now adds rolling load stats + holiday flag (no weather).
        assert "load_mw_roll_mean_24" in enriched.columns
        assert "is_holiday" in enriched.columns

    def test_demand_full_extends_enriched(self):
        enriched = get_feature_set("demand_enriched")
        full = get_feature_set("demand_full")
        for col in enriched.columns:
            assert col in full.columns, f"{col} missing from demand_full"
        # Full now sources weather from forward-looking NWP at horizon.
        assert "nwp_temperature_2m" in full.columns
        assert "heating_degree_days" in full.columns

    def test_list_includes_demand(self):
        names = list_feature_sets()
        assert "demand_baseline" in names
        assert "demand_enriched" in names
        assert "demand_full" in names


class TestSolarFeatureSets:
    def test_solar_baseline_exists(self):
        fs = get_feature_set("solar_baseline")
        assert fs.name == "solar_baseline"
        assert "poa_wm2" in fs.columns
        assert "power_kw_lag1" in fs.columns

    def test_solar_enriched_extends_baseline(self):
        baseline = get_feature_set("solar_baseline")
        enriched = get_feature_set("solar_enriched")
        for col in baseline.columns:
            assert col in enriched.columns, f"{col} missing from solar_enriched"
        assert "clearsky_ratio" in enriched.columns

    def test_solar_full_extends_enriched(self):
        enriched = get_feature_set("solar_enriched")
        full = get_feature_set("solar_full")
        for col in enriched.columns:
            assert col in full.columns, f"{col} missing from solar_full"
        assert "month_sin" in full.columns
        assert "dow_sin" in full.columns

    def test_list_includes_solar(self):
        names = list_feature_sets()
        assert "solar_baseline" in names
        assert "solar_enriched" in names
        assert "solar_full" in names


class TestFeatureSet:
    def test_is_frozen_dataclass(self):
        fs = get_feature_set("wind_baseline")
        assert isinstance(fs, FeatureSet)
        with pytest.raises(AttributeError):
            fs.name = "modified"  # type: ignore[misc]
