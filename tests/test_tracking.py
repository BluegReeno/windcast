"""Tests for windcast.tracking.mlflow_utils.log_evaluation_results."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from windcast.tracking import log_evaluation_results
from windcast.tracking.mlflow_utils import STEPPED_METRIC_MAP


@pytest.fixture
def isolated_mlflow(tmp_path: Path):
    """Per-test isolated MLflow SQLite store + experiment.

    Yields a (client, experiment_id) tuple. The tracking URI is scoped to a
    tmp_path SQLite file so tests do not touch the repo-level ``mlflow.db``
    and never share state with each other.
    """
    db_path = tmp_path / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment(
        "test-tracking", artifact_location=str(tmp_path / "artifacts")
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    client = MlflowClient(tracking_uri=tracking_uri)
    yield client, experiment_id


def _base_metrics() -> dict[str, float]:
    return {
        "mae": 120.5,
        "rmse": 180.3,
        "bias": -4.2,
        "skill_score": 0.234,
    }


class TestLogEvaluationResultsLegacyMode:
    """Legacy path: only ``horizon`` given, no ``horizon_minutes``."""

    def test_flat_metrics_present_stepped_absent(self, isolated_mlflow):
        client, _ = isolated_mlflow
        metrics = _base_metrics()

        with mlflow.start_run() as run:
            log_evaluation_results(metrics, horizon=6)
            run_id = run.info.run_id

        run_data = client.get_run(run_id).data.metrics
        assert run_data["h6_mae"] == pytest.approx(120.5)
        assert run_data["h6_rmse"] == pytest.approx(180.3)
        assert run_data["h6_bias"] == pytest.approx(-4.2)
        assert run_data["h6_skill_score"] == pytest.approx(0.234)

        # Stepped metrics must NOT appear when horizon_minutes is None
        for stepped_name in STEPPED_METRIC_MAP.values():
            assert stepped_name not in run_data, (
                f"Stepped metric {stepped_name!r} leaked into legacy path"
            )


class TestLogEvaluationResultsSteppedMode:
    """Stepped path: both ``horizon`` and ``horizon_minutes`` given."""

    def test_flat_and_stepped_metrics_both_logged(self, isolated_mlflow):
        client, _ = isolated_mlflow
        metrics = _base_metrics()

        with mlflow.start_run() as run:
            log_evaluation_results(metrics, horizon=6, horizon_minutes=60)
            run_id = run.info.run_id

        run_data = client.get_run(run_id).data.metrics
        # Flat path still fires
        assert run_data["h6_mae"] == pytest.approx(120.5)
        assert run_data["h6_rmse"] == pytest.approx(180.3)

        # Stepped path fires with step=60
        mae_history = client.get_metric_history(run_id, "mae_by_horizon_min")
        assert [m.step for m in mae_history] == [60]
        assert mae_history[0].value == pytest.approx(120.5)

        rmse_history = client.get_metric_history(run_id, "rmse_by_horizon_min")
        assert [m.step for m in rmse_history] == [60]
        assert rmse_history[0].value == pytest.approx(180.3)

        bias_history = client.get_metric_history(run_id, "bias_by_horizon_min")
        assert [m.step for m in bias_history] == [60]

        skill_history = client.get_metric_history(run_id, "skill_score_by_horizon_min")
        assert [m.step for m in skill_history] == [60]
        assert skill_history[0].value == pytest.approx(0.234)


class TestLogEvaluationResultsMultiHorizon:
    """Multiple horizons on the same run build up a line-chart-ready history."""

    def test_history_has_all_steps_in_order(self, isolated_mlflow):
        client, _ = isolated_mlflow

        horizons = [1, 6, 12, 24, 48]
        horizon_minutes = [10, 60, 120, 240, 480]
        mae_values = [120.0, 210.0, 260.0, 335.0, 430.0]

        with mlflow.start_run() as run:
            for h, hm, mae in zip(horizons, horizon_minutes, mae_values, strict=True):
                log_evaluation_results(
                    {"mae": mae, "rmse": mae * 1.4, "bias": 0.0, "skill_score": 0.2},
                    horizon=h,
                    horizon_minutes=hm,
                )
            run_id = run.info.run_id

        history = client.get_metric_history(run_id, "mae_by_horizon_min")
        points = sorted((m.step, m.value) for m in history)
        assert [p[0] for p in points] == horizon_minutes
        for (_, logged_value), expected in zip(points, mae_values, strict=True):
            assert logged_value == pytest.approx(expected)


class TestLogEvaluationResultsEdgeCases:
    """Edge cases called out in the plan."""

    def test_unknown_metric_key_skipped_in_stepped_path(self, isolated_mlflow):
        """Keys not in STEPPED_METRIC_MAP are skipped (no crash, no stepped log)."""
        client, _ = isolated_mlflow
        metrics = {"mae": 100.0, "custom_metric": 42.0}

        with mlflow.start_run() as run:
            log_evaluation_results(metrics, horizon=6, horizon_minutes=60)
            run_id = run.info.run_id

        run_data = client.get_run(run_id).data.metrics
        # Flat path logs everything (including custom_metric)
        assert run_data["h6_mae"] == pytest.approx(100.0)
        assert run_data["h6_custom_metric"] == pytest.approx(42.0)

        # Stepped path logs mae but silently skips custom_metric
        mae_history = client.get_metric_history(run_id, "mae_by_horizon_min")
        assert len(mae_history) == 1
        assert client.get_metric_history(run_id, "custom_metric_by_horizon_min") == []

    def test_zero_minutes_is_valid_step(self, isolated_mlflow):
        """``horizon_minutes=0`` is a valid step value (e.g. nowcast)."""
        client, _ = isolated_mlflow

        with mlflow.start_run() as run:
            log_evaluation_results({"mae": 50.0}, horizon=0, horizon_minutes=0)
            run_id = run.info.run_id

        history = client.get_metric_history(run_id, "mae_by_horizon_min")
        assert [m.step for m in history] == [0]
        assert history[0].value == pytest.approx(50.0)

    def test_stepped_without_flat_horizon(self, isolated_mlflow):
        """``horizon=None, horizon_minutes=60`` still fires the stepped path."""
        client, _ = isolated_mlflow
        metrics = {"mae": 100.0, "rmse": 150.0}

        with mlflow.start_run() as run:
            log_evaluation_results(metrics, horizon=None, horizon_minutes=60)
            run_id = run.info.run_id

        run_data = client.get_run(run_id).data.metrics
        # Flat path falls through with no h{n}_ prefix
        assert run_data["mae"] == pytest.approx(100.0)
        assert run_data["rmse"] == pytest.approx(150.0)
        assert "h6_mae" not in run_data

        # Stepped path still fires
        mae_history = client.get_metric_history(run_id, "mae_by_horizon_min")
        assert [m.step for m in mae_history] == [60]

    def test_persistence_metrics_stepped_when_present(self, isolated_mlflow):
        """``persistence_mae`` keys in the input dict are stepped-logged too."""
        client, _ = isolated_mlflow
        metrics = {
            "mae": 100.0,
            "persistence_mae": 150.0,
            "persistence_rmse": 200.0,
        }

        with mlflow.start_run() as run:
            log_evaluation_results(metrics, horizon=6, horizon_minutes=60)
            run_id = run.info.run_id

        history = client.get_metric_history(run_id, "persistence_mae_by_horizon_min")
        assert [m.step for m in history] == [60]
        assert history[0].value == pytest.approx(150.0)
