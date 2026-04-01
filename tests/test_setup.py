"""Smoke tests to verify project setup and imports."""


def test_windcast_importable():
    """Verify the windcast package can be imported."""
    import windcast

    assert windcast is not None


def test_core_dependencies_importable():
    """Verify all core dependencies are installed."""
    import lightgbm
    import mlflow
    import optuna
    import polars
    import pydantic
    import sklearn
    import xgboost

    assert all([polars, xgboost, lightgbm, sklearn, mlflow, optuna, pydantic])


def test_sub_packages_importable():
    """Verify all sub-packages can be imported."""
    import windcast.data
    import windcast.features
    import windcast.models

    assert all([windcast.data, windcast.features, windcast.models])
