"""Tests for windcast.models.evaluation module."""

import logging

import numpy as np
import polars as pl
import pytest

from windcast.models.evaluation import (
    compute_metrics,
    compute_skill_score,
    evaluate_with_custom_metrics,
    regime_analysis,
)


class TestComputeMetrics:
    def test_perfect_prediction(self):
        y = np.array([100.0, 200.0, 300.0])
        metrics = compute_metrics(y, y)
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["bias"] == 0.0

    def test_known_error(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 210.0, 310.0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["mae"] == 10.0
        assert metrics["bias"] == 10.0  # positive bias = over-prediction


class TestComputeSkillScore:
    def test_perfect_model(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = y_true.copy()
        y_persistence = np.array([90.0, 190.0, 290.0])
        assert compute_skill_score(y_true, y_pred, y_persistence) == 1.0

    def test_same_as_persistence(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_persistence = np.array([90.0, 190.0, 290.0])
        skill = compute_skill_score(y_true, y_persistence, y_persistence)
        assert skill == pytest.approx(0.0)

    def test_worse_than_persistence(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([0.0, 0.0, 0.0])  # terrible prediction
        y_persistence = np.array([90.0, 190.0, 290.0])
        assert compute_skill_score(y_true, y_pred, y_persistence) < 0

    def test_persistence_zero_rmse_model_zero(self):
        y_true = np.array([100.0, 100.0, 100.0])
        y_pred = y_true.copy()
        y_persistence = y_true.copy()
        assert compute_skill_score(y_true, y_pred, y_persistence) == 1.0

    def test_persistence_zero_rmse_model_nonzero(self):
        y_true = np.array([100.0, 100.0, 100.0])
        y_pred = np.array([110.0, 110.0, 110.0])
        y_persistence = y_true.copy()
        assert compute_skill_score(y_true, y_pred, y_persistence) == -float("inf")


class TestMapeHandling:
    def test_mape_computed_without_zeros(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 210.0, 310.0])
        metrics = compute_metrics(y_true, y_pred)
        assert "mape" in metrics

    def test_mape_skipped_with_zeros(self, caplog):
        y_true = np.array([0.0, 200.0, 300.0])
        y_pred = np.array([10.0, 210.0, 310.0])
        with caplog.at_level(logging.WARNING):
            metrics = compute_metrics(y_true, y_pred)
        assert "mape" not in metrics
        assert "MAPE skipped" in caplog.text


class TestRegimeAnalysis:
    def test_three_regimes(self):
        df = pl.DataFrame(
            {
                "wind_speed_ms": [2.0, 3.0, 8.0, 10.0, 15.0, 18.0],
                "y_true": [10.0, 20.0, 500.0, 800.0, 1500.0, 1800.0],
                "y_pred": [12.0, 22.0, 510.0, 790.0, 1510.0, 1790.0],
            }
        )
        result = regime_analysis(df, "y_true", "y_pred")
        assert "low" in result
        assert "medium" in result
        assert "high" in result
        for regime in result.values():
            assert "mae" in regime
            assert "rmse" in regime


class TestCustomMetrics:
    def test_custom_metric_callable(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([100.0, 200.0, 300.0])

        def my_metric(y_t: np.ndarray, y_p: np.ndarray) -> float:
            return float(np.max(np.abs(y_t - y_p)))

        result = evaluate_with_custom_metrics(y_true, y_pred, {"max_abs_error": my_metric})
        assert result["max_abs_error"] == 0.0
