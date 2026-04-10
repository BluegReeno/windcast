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

    @patch("windcast.models.autogluon_model.mlflow")
    def test_fit_never_receives_val_as_tuning_data(self, mock_mlflow, tmp_path):
        """Regression guard: val_pd must NEVER be passed as tuning_data to fit().

        Rationale: if tuning_data=val_pd, AutoGluon uses the caller's validation
        set for stacker selection + ensemble weighting + early stopping, which
        makes the subsequent "eval on val_pd" step a self-fulfilling in-sample
        score. This was the leak caught on 2026-04-09 that produced a fake
        -74% MAE advantage over XGBoost on RTE demand_full.

        See: feedback_verify_extraordinary_results.md and autogluon_model.py docstring.
        """
        from unittest.mock import MagicMock

        X_train, y_train, X_val, y_val = _make_regression_data()
        config = AutoGluonConfig(time_limit=30, presets="medium_quality")

        # Mock TabularPredictor so we never actually train — we only care about fit() kwargs
        with patch("autogluon.tabular.TabularPredictor") as mock_predictor_cls:
            mock_predictor = MagicMock()
            mock_predictor.leaderboard.return_value = MagicMock(
                iloc=[MagicMock(__getitem__=lambda self, k: 0.0)],
                __len__=lambda self: 1,
            )
            mock_predictor_cls.return_value = mock_predictor

            train_autogluon(X_train, y_train, X_val, y_val, config, ag_path=tmp_path / "ag")

        # Confirm fit() was called exactly once, and tuning_data was NOT in its kwargs
        assert mock_predictor.fit.call_count == 1, "fit() should be called exactly once"
        fit_kwargs = mock_predictor.fit.call_args.kwargs
        assert "tuning_data" not in fit_kwargs, (
            f"LEAK GUARD: tuning_data must not be passed to fit(). "
            f"Got kwargs: {list(fit_kwargs.keys())}. "
            f"AutoGluon would use the caller's validation set for ensemble "
            f"selection, turning subsequent eval on val into an in-sample score."
        )
        # Sanity: the healthy path still passes use_bag_holdout=True so AG
        # carves its own tuning holdout from train_data
        assert fit_kwargs.get("use_bag_holdout") is True, (
            "Expected use_bag_holdout=True so AG holds out a tuning slice from train_data"
        )
