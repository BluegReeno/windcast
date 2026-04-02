"""Tests for windcast.config module."""

from windcast.config import DATASETS, DemandQCConfig, WindCastSettings, get_settings


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


def test_spain_demand_in_registry():
    """Spain demand dataset is registered."""
    assert "spain_demand" in DATASETS


def test_spain_demand_config_values():
    """Spain demand config has correct values."""
    cfg = DATASETS["spain_demand"]
    assert cfg.dataset_id == "spain_demand"
    assert hasattr(cfg, "zone_id")
    assert cfg.zone_id == "ES"  # type: ignore[union-attr]
    assert cfg.latitude > 40.0


def test_demand_qc_defaults():
    """DemandQCConfig has sensible defaults."""
    qc = DemandQCConfig()
    assert qc.max_load_mw == 50_000.0
    assert qc.min_load_mw == 10_000.0
    assert qc.max_gap_fill_hours == 3


def test_domain_default():
    """Default domain is wind."""
    settings = WindCastSettings()
    assert settings.domain == "wind"


def test_domain_env_override(monkeypatch):
    """Domain can be overridden via environment."""
    get_settings.cache_clear()
    monkeypatch.setenv("WINDCAST_DOMAIN", "demand")
    settings = WindCastSettings()
    assert settings.domain == "demand"
