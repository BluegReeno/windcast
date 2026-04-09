"""Tests for windcast.tracking.mlflow_utils metric logging helpers."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from windcast.tracking import log_evaluation_results, log_stepped_horizon_metrics
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


class TestLogEvaluationResults:
    """``log_evaluation_results`` is the flat-metrics path."""

    def test_flat_metrics_with_horizon_prefix(self, isolated_mlflow):
        client, _ = isolated_mlflow

        with mlflow.start_run() as run:
            log_evaluation_results(_base_metrics(), horizon=6)
            run_id = run.info.run_id

        run_data = client.get_run(run_id).data.metrics
        assert run_data["h6_mae"] == pytest.approx(120.5)
        assert run_data["h6_rmse"] == pytest.approx(180.3)
        assert run_data["h6_bias"] == pytest.approx(-4.2)
        assert run_data["h6_skill_score"] == pytest.approx(0.234)

    def test_flat_metrics_without_horizon(self, isolated_mlflow):
        client, _ = isolated_mlflow

        with mlflow.start_run() as run:
            log_evaluation_results({"mae": 100.0, "rmse": 150.0})
            run_id = run.info.run_id

        run_data = client.get_run(run_id).data.metrics
        assert run_data["mae"] == pytest.approx(100.0)
        assert run_data["rmse"] == pytest.approx(150.0)

    def test_does_not_emit_stepped_metrics(self, isolated_mlflow):
        """The flat path never writes stepped metrics — that is the
        stepped helper's job and must not happen implicitly."""
        client, _ = isolated_mlflow

        with mlflow.start_run() as run:
            log_evaluation_results(_base_metrics(), horizon=6)
            run_id = run.info.run_id

        run_data = client.get_run(run_id).data.metrics
        for stepped_name in STEPPED_METRIC_MAP.values():
            assert stepped_name not in run_data, (
                f"Stepped metric {stepped_name!r} leaked out of the flat path"
            )


class TestLogSteppedHorizonMetrics:
    """``log_stepped_horizon_metrics`` accumulates a time series on one run."""

    def test_history_has_all_steps_in_order(self, isolated_mlflow):
        """Five horizons → five-point metric history on the active run."""
        client, _ = isolated_mlflow

        horizon_minutes = [10, 60, 120, 240, 480]
        mae_values = [120.0, 210.0, 260.0, 335.0, 430.0]
        by_horizon = {
            hm: {"mae": mae, "rmse": mae * 1.4, "bias": 0.0, "skill_score": 0.2}
            for hm, mae in zip(horizon_minutes, mae_values, strict=True)
        }

        with mlflow.start_run() as run:
            log_stepped_horizon_metrics(by_horizon)
            run_id = run.info.run_id

        history = client.get_metric_history(run_id, "mae_by_horizon_min")
        points = sorted((m.step, m.value) for m in history)
        assert [p[0] for p in points] == horizon_minutes
        for (_, logged_value), expected in zip(points, mae_values, strict=True):
            assert logged_value == pytest.approx(expected)

        # Sibling stepped metrics follow the same shape
        assert len(client.get_metric_history(run_id, "rmse_by_horizon_min")) == 5
        assert len(client.get_metric_history(run_id, "bias_by_horizon_min")) == 5
        assert len(client.get_metric_history(run_id, "skill_score_by_horizon_min")) == 5

    def test_logs_on_parent_not_children(self, isolated_mlflow):
        """When called in a nested parent/child pattern, the stepped history
        must live on the parent run. This is the canonical pattern from
        MLflow issue #7060: one run, N steps, one curve."""
        client, _ = isolated_mlflow

        horizon_minutes = [10, 60, 120, 240, 480]
        parent_metrics: dict[int, dict[str, float]] = {}

        with mlflow.start_run(run_name="parent") as parent_run:
            parent_id = parent_run.info.run_id
            child_ids: list[str] = []
            for hm in horizon_minutes:
                with mlflow.start_run(run_name=f"child_{hm}", nested=True) as child_run:
                    child_ids.append(child_run.info.run_id)
                    m = {"mae": float(hm), "rmse": float(hm) * 1.5, "bias": 0.0}
                    log_evaluation_results(m, horizon=hm // 10)
                    parent_metrics[hm] = m
            # Back in parent context after children exit
            log_stepped_horizon_metrics(parent_metrics)

        # Parent holds the full 5-step history
        parent_history = client.get_metric_history(parent_id, "mae_by_horizon_min")
        assert sorted(m.step for m in parent_history) == horizon_minutes

        # Children hold NO stepped metrics
        for cid in child_ids:
            assert client.get_metric_history(cid, "mae_by_horizon_min") == []

    def test_unknown_metric_key_skipped(self, isolated_mlflow):
        """Keys absent from STEPPED_METRIC_MAP are silently skipped."""
        client, _ = isolated_mlflow

        with mlflow.start_run() as run:
            log_stepped_horizon_metrics({60: {"mae": 100.0, "custom_metric": 42.0}})
            run_id = run.info.run_id

        mae_history = client.get_metric_history(run_id, "mae_by_horizon_min")
        assert [m.step for m in mae_history] == [60]
        assert client.get_metric_history(run_id, "custom_metric_by_horizon_min") == []

    def test_zero_minutes_is_valid_step(self, isolated_mlflow):
        """``step=0`` is valid (e.g. a nowcast horizon)."""
        client, _ = isolated_mlflow

        with mlflow.start_run() as run:
            log_stepped_horizon_metrics({0: {"mae": 50.0}})
            run_id = run.info.run_id

        history = client.get_metric_history(run_id, "mae_by_horizon_min")
        assert [m.step for m in history] == [0]
        assert history[0].value == pytest.approx(50.0)

    def test_persistence_metrics_logged_when_present(self, isolated_mlflow):
        """``persistence_*`` keys are included in STEPPED_METRIC_MAP and
        must be logged alongside the model's own metrics."""
        client, _ = isolated_mlflow

        with mlflow.start_run() as run:
            log_stepped_horizon_metrics(
                {
                    60: {
                        "mae": 100.0,
                        "persistence_mae": 150.0,
                        "persistence_rmse": 200.0,
                    },
                    120: {
                        "mae": 130.0,
                        "persistence_mae": 180.0,
                        "persistence_rmse": 230.0,
                    },
                }
            )
            run_id = run.info.run_id

        p_mae_history = client.get_metric_history(run_id, "persistence_mae_by_horizon_min")
        points = sorted((m.step, m.value) for m in p_mae_history)
        assert [p[0] for p in points] == [60, 120]
        assert [p[1] for p in points] == pytest.approx([150.0, 180.0])

    def test_iterates_horizons_in_ascending_order(self, isolated_mlflow):
        """Even if the input dict is unordered, the stepped history must be
        logged in ascending step order so ``get_metric_history`` returns a
        monotonic sequence."""
        client, _ = isolated_mlflow

        unordered = {
            480: {"mae": 430.0},
            10: {"mae": 120.0},
            120: {"mae": 260.0},
            60: {"mae": 210.0},
            240: {"mae": 335.0},
        }
        with mlflow.start_run() as run:
            log_stepped_horizon_metrics(unordered)
            run_id = run.info.run_id

        history = client.get_metric_history(run_id, "mae_by_horizon_min")
        steps = [m.step for m in history]
        assert steps == sorted(steps), f"History not monotonic: {steps}"
