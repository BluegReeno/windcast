"""Tests for model logging and registry integration."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import mlflow
import numpy as np
import polars as pl
import pytest

from windcast.training.backends import XGBoostBackend
from windcast.training.harness import run_training


@pytest.fixture
def _wind_features_parquet(tmp_path):
    """Create a minimal wind features parquet for testing."""
    n = 500
    dates = [datetime(2015, 1, 1, h % 24) for h in range(n)]
    for i in range(n):
        year = 2015 + i * 11 // n
        dates[i] = dates[i].replace(year=year)
    dates.sort()

    rng = np.random.default_rng(42)
    df = pl.DataFrame(
        {
            "timestamp_utc": dates,
            "active_power_kw": rng.uniform(0, 1000, n).tolist(),
            "active_power_kw_lag1": rng.uniform(0, 1000, n).tolist(),
            "hour": [d.hour for d in dates],
        }
    )
    path = tmp_path / "kelmarsh_kwf1.parquet"
    df.write_parquet(path)
    return tmp_path


def test_xgboost_log_model(tmp_path):
    """Test that XGBoostBackend.log_model() produces a loadable model."""
    rng = np.random.default_rng(42)
    X = pl.DataFrame({"f1": rng.normal(size=100).tolist(), "f2": rng.normal(size=100).tolist()})
    y = pl.Series("target", rng.normal(size=100).tolist())

    backend = XGBoostBackend()
    model = backend.train(X, y, X, y)
    y_pred = backend.predict(model, X)

    tracking_uri = f"sqlite:///{tmp_path}/test_registry.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-log-model")

    with mlflow.start_run():
        with mlflow.start_run(run_name="h01", nested=True):
            model_uri = backend.log_model(model, X, y_pred, horizon=1)
            assert model_uri is not None
            assert model_uri.startswith("models:/")

        # Load back the model and verify predictions
        loaded = mlflow.pyfunc.load_model(model_uri)
        preds = loaded.predict(X.to_pandas())
        np.testing.assert_allclose(preds, y_pred, rtol=1e-5)


def test_run_training_with_model_logging(_wind_features_parquet, tmp_path):
    """Test run_training logs models when log_models=True."""
    tracking_uri = f"sqlite:///{tmp_path}/mlflow_registry_test.db"

    with patch("windcast.config.get_settings") as mock_settings:
        mock_settings.return_value.mlflow_tracking_uri = tracking_uri
        mock_settings.return_value.train_years = 5
        mock_settings.return_value.val_years = 1
        mock_settings.return_value.features_dir = _wind_features_parquet

        run_training(
            backend=XGBoostBackend(),
            domain="wind",
            dataset="kelmarsh",
            feature_set_name="wind_baseline",
            features_path=_wind_features_parquet / "kelmarsh_kwf1.parquet",
            experiment_name="test-registry",
            horizons=[1],
            turbine_id="kwf1",
            log_models=True,
            train_years=8,
            val_years=2,
        )

    mlflow.set_tracking_uri(tracking_uri)
    # Verify a child run was created
    runs = mlflow.search_runs(
        experiment_names=["test-registry"],
        filter_string="tags.`enercast.run_type` = 'child'",
        output_format="pandas",
    )
    assert len(runs) == 1
    # Verify model was logged via MLflow 3.x LoggedModel API
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("test-registry")
    assert exp is not None
    logged_models = client.search_logged_models(
        experiment_ids=[exp.experiment_id],
    )
    assert len(logged_models) >= 1
    model_names = [m.name for m in logged_models]
    assert "model_h01" in model_names


def test_run_training_with_registration(_wind_features_parquet, tmp_path):
    """Test run_training registers best model when register_model_name is set."""
    tracking_uri = f"sqlite:///{tmp_path}/mlflow_register_test.db"

    with patch("windcast.config.get_settings") as mock_settings:
        mock_settings.return_value.mlflow_tracking_uri = tracking_uri
        mock_settings.return_value.train_years = 5
        mock_settings.return_value.val_years = 1
        mock_settings.return_value.features_dir = _wind_features_parquet

        run_training(
            backend=XGBoostBackend(),
            domain="wind",
            dataset="kelmarsh",
            feature_set_name="wind_baseline",
            features_path=_wind_features_parquet / "kelmarsh_kwf1.parquet",
            experiment_name="test-register",
            horizons=[1],
            turbine_id="kwf1",
            log_models=True,
            register_model_name="test-enercast-kelmarsh-xgboost",
            train_years=8,
            val_years=2,
        )

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    # Verify model was registered
    registered = client.get_registered_model("test-enercast-kelmarsh-xgboost")
    assert registered is not None
    assert len(registered.latest_versions) >= 1
    # Verify champion alias
    alias_mv = client.get_model_version_by_alias("test-enercast-kelmarsh-xgboost", "champion")
    assert alias_mv is not None
