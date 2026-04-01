"""Tests for windcast.models.xgboost_model module."""

from unittest.mock import patch

import numpy as np
import polars as pl

from windcast.models.xgboost_model import XGBoostConfig, train_xgboost


def _make_regression_data(
    n_train: int = 100,
    n_val: int = 30,
    n_features: int = 5,
) -> tuple[pl.DataFrame, pl.Series, pl.DataFrame, pl.Series]:
    """Create synthetic regression data for XGBoost testing."""
    rng = np.random.default_rng(42)

    X_train_np = rng.standard_normal((n_train, n_features))
    y_train_np = X_train_np[:, 0] * 2 + X_train_np[:, 1] + rng.standard_normal(n_train) * 0.1

    X_val_np = rng.standard_normal((n_val, n_features))
    y_val_np = X_val_np[:, 0] * 2 + X_val_np[:, 1] + rng.standard_normal(n_val) * 0.1

    cols = [f"f{i}" for i in range(n_features)]
    X_train = pl.DataFrame(dict(zip(cols, X_train_np.T, strict=True)))
    y_train = pl.Series("target", y_train_np)
    X_val = pl.DataFrame(dict(zip(cols, X_val_np.T, strict=True)))
    y_val = pl.Series("target", y_val_np)

    return X_train, y_train, X_val, y_val


class TestXGBoostConfig:
    def test_defaults(self):
        config = XGBoostConfig()
        assert config.objective == "reg:squarederror"
        assert config.n_estimators == 500
        assert config.early_stopping_rounds == 50

    def test_custom_values(self):
        config = XGBoostConfig(n_estimators=100, max_depth=4)
        assert config.n_estimators == 100
        assert config.max_depth == 4


class TestTrainXGBoost:
    @patch("windcast.models.xgboost_model.mlflow")
    def test_returns_fitted_model(self, mock_mlflow):
        mock_mlflow.active_run.return_value = None
        X_train, y_train, X_val, y_val = _make_regression_data()
        config = XGBoostConfig(n_estimators=10, early_stopping_rounds=5)
        model = train_xgboost(X_train, y_train, X_val, y_val, config)

        preds = model.predict(X_val)
        assert len(preds) == len(y_val)
        assert model.best_iteration >= 0

    @patch("windcast.models.xgboost_model.mlflow")
    def test_default_config(self, mock_mlflow):
        mock_mlflow.active_run.return_value = None
        X_train, y_train, X_val, y_val = _make_regression_data()
        config = XGBoostConfig(n_estimators=10, early_stopping_rounds=5)
        model = train_xgboost(X_train, y_train, X_val, y_val, config)
        assert model is not None
