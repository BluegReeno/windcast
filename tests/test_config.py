"""Tests for windcast.config module."""

from windcast.config import DATASETS, WindCastSettings, get_settings


def test_default_settings():
    """Default settings load without errors."""
    get_settings.cache_clear()
    settings = WindCastSettings()
    assert settings.dataset_id == "kelmarsh"
    assert settings.data_dir.name == "data"
    assert settings.qc.max_wind_speed_ms == 40.0


def test_datasets_registry():
    """All expected datasets are registered."""
    assert "kelmarsh" in DATASETS
    assert "hill_of_towie" in DATASETS
    assert "penmanshiel" in DATASETS
    assert DATASETS["kelmarsh"].rated_power_kw == 2050.0


def test_env_override(monkeypatch):
    """Environment variables override defaults."""
    get_settings.cache_clear()
    monkeypatch.setenv("WINDCAST_DATASET_ID", "hill_of_towie")
    settings = WindCastSettings()
    assert settings.dataset_id == "hill_of_towie"


def test_derived_paths():
    """Derived path properties work correctly."""
    settings = WindCastSettings()
    assert settings.raw_dir == settings.data_dir / "raw"
    assert settings.processed_dir == settings.data_dir / "processed"
    assert settings.features_dir == settings.data_dir / "features"
