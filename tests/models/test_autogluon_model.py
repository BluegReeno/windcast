"""Tests for windcast.models.autogluon_model module."""

from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from windcast.models.autogluon_model import AutoGluonConfig, train_autogluon


def _make_regression_data(
    n_train: int = 100,
    n_val: int = 30,
    n_features: int = 5,
) -> tuple[pl.DataFrame, pl.Series, pl.DataFrame, pl.Series]:
    """Create synthetic regression data for AutoGluon testing."""
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


class TestAutoGluonConfig:
    def test_defaults(self):
        config = AutoGluonConfig()
        assert config.presets == "best_quality"
        assert config.time_limit == 300
        assert config.eval_metric == "mean_absolute_error"
        assert "NN_TORCH" in config.excluded_model_types

    def test_custom_values(self):
        config = AutoGluonConfig(presets="medium_quality", time_limit=60)
        assert config.presets == "medium_quality"
        assert config.time_limit == 60


class TestTrainAutoGluon:
    @pytest.mark.slow
    @patch("windcast.models.autogluon_model.mlflow")
    def test_returns_fitted_predictor(self, mock_mlflow, tmp_path):
        from autogluon.tabular import TabularPredictor

        X_train, y_train, X_val, y_val = _make_regression_data()
        config = AutoGluonConfig(time_limit=30, presets="medium_quality")
        predictor = train_autogluon(X_train, y_train, X_val, y_val, config, ag_path=tmp_path / "ag")

        assert isinstance(predictor, TabularPredictor)
        preds = predictor.predict(X_val.to_pandas())
        assert len(preds) == len(y_val)

    @pytest.mark.slow
    @patch("windcast.models.autogluon_model.mlflow")
    def test_autolog_disabled_during_fit(self, mock_mlflow, tmp_path):
        X_train, y_train, X_val, y_val = _make_regression_data()
        config = AutoGluonConfig(time_limit=30, presets="medium_quality")
        train_autogluon(X_train, y_train, X_val, y_val, config, ag_path=tmp_path / "ag")

        # Verify autolog was disabled before fit, then re-enabled
        autolog_calls = mock_mlflow.autolog.call_args_list
        assert len(autolog_calls) >= 2
        assert autolog_calls[0].kwargs.get("disable") is True
        assert autolog_calls[1].kwargs.get("disable") is False
