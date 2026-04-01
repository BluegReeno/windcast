"""Tests for windcast.models.persistence module."""

import numpy as np

from windcast.models.persistence import compute_persistence_metrics, persistence_forecast


class TestPersistenceForecast:
    def test_returns_lag1(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_lag1 = np.array([90.0, 100.0, 200.0])
        result = persistence_forecast(y_true, y_lag1)
        np.testing.assert_array_equal(result, y_lag1)

    def test_returns_copy(self):
        y_true = np.array([1.0, 2.0])
        y_lag1 = np.array([0.5, 1.0])
        result = persistence_forecast(y_true, y_lag1)
        result[0] = 999.0
        assert y_lag1[0] != 999.0


class TestPersistenceMetrics:
    def test_metrics_keys(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_lag1 = np.array([90.0, 190.0, 310.0])
        metrics = compute_persistence_metrics(y_true, y_lag1)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "bias" in metrics

    def test_perfect_persistence(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_lag1 = np.array([100.0, 200.0, 300.0])
        metrics = compute_persistence_metrics(y_true, y_lag1)
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["bias"] == 0.0
