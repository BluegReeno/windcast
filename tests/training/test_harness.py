"""Tests for training harness — shared utilities and mock-backend integration."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import patch

import mlflow
import numpy as np
import polars as pl
import pytest

from windcast.training.harness import (
    build_horizon_desc,
    build_horizon_target,
    resolve_horizon_features,
    run_training,
    temporal_split,
)

# --- temporal_split ---


def _make_ts_df(start_year: int, n_years: int, rows_per_year: int = 100) -> pl.DataFrame:
    """Create a minimal timestamped DataFrame spanning n_years."""
    dates: list[datetime] = []
    for y in range(n_years):
        base = datetime(start_year + y, 1, 1)
        dates.extend(base.replace(day=1 + (i % 28), hour=i % 24) for i in range(rows_per_year))
    return pl.DataFrame(
        {
            "timestamp_utc": dates[: n_years * rows_per_year],
            "active_power_kw": list(range(n_years * rows_per_year)),
        }
    )


def test_temporal_split_boundaries():
    df = _make_ts_df(2015, 11)
    train, val, test = temporal_split(df, train_years=8, val_years=2)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert len(train) + len(val) + len(test) == len(df)

    # Verify no overlap
    train_max = train["timestamp_utc"].max()
    val_min = val["timestamp_utc"].min()
    val_max = val["timestamp_utc"].max()
    test_min = test["timestamp_utc"].min()
    assert train_max < val_min
    assert val_max < test_min


def test_temporal_split_empty_test():
    df = _make_ts_df(2020, 3)
    train, val, test = temporal_split(df, train_years=2, val_years=1)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) == 0


def test_temporal_split_custom_timestamp_col():
    """Test temporal_split with a non-default timestamp column (e.g. 'ds')."""
    df = _make_ts_df(2015, 8)
    df = df.rename({"timestamp_utc": "ds"})
    train, val, test = temporal_split(df, train_years=5, val_years=2, timestamp_col="ds")
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert len(train) + len(val) + len(test) == len(df)


# --- resolve_horizon_features ---


def test_resolve_horizon_features_nwp():
    available = ["hour", "nwp_temp_h1", "nwp_temp_h6", "nwp_wind_h1", "nwp_wind_h6"]
    feature_set = ["hour", "nwp_temp", "nwp_wind"]

    cols, rename_map = resolve_horizon_features(available, feature_set, horizon=1)
    assert cols == ["hour", "nwp_temp_h1", "nwp_wind_h1"]
    assert rename_map == {"nwp_temp_h1": "nwp_temp", "nwp_wind_h1": "nwp_wind"}


def test_resolve_horizon_features_fallback():
    available = ["hour", "nwp_temp"]
    feature_set = ["hour", "nwp_temp"]

    cols, rename_map = resolve_horizon_features(available, feature_set, horizon=1)
    assert cols == ["hour", "nwp_temp"]
    assert rename_map == {}


def test_resolve_horizon_features_missing():
    available = ["hour"]
    feature_set = ["hour", "nwp_temp", "missing_col"]

    cols, rename_map = resolve_horizon_features(available, feature_set, horizon=1)
    assert cols == ["hour"]
    assert rename_map == {}


# --- build_horizon_target ---


def test_build_horizon_target():
    df = pl.DataFrame(
        {
            "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "active_power_kw": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    X, y = build_horizon_target(
        df, horizon=1, feature_cols=["feat1"], target_col_name="active_power_kw"
    )
    assert len(X) == 4  # last row dropped (shift produces null)
    assert len(y) == 4
    assert y.to_list() == [20.0, 30.0, 40.0, 50.0]


def test_build_horizon_target_with_rename():
    df = pl.DataFrame(
        {
            "nwp_temp_h1": [1.0, 2.0, 3.0],
            "active_power_kw": [10.0, 20.0, 30.0],
        }
    )
    X, _y = build_horizon_target(
        df,
        horizon=1,
        feature_cols=["nwp_temp_h1"],
        target_col_name="active_power_kw",
        rename_map={"nwp_temp_h1": "nwp_temp"},
    )
    assert "nwp_temp" in X.columns
    assert "nwp_temp_h1" not in X.columns


# --- build_horizon_desc ---


def test_build_horizon_desc_minutes():
    assert build_horizon_desc(1, 10) == "10 min ahead"
    assert build_horizon_desc(3, 10) == "30 min ahead"


def test_build_horizon_desc_hours():
    assert build_horizon_desc(6, 10) == "1h ahead"
    assert build_horizon_desc(1, 60) == "1h ahead"
    assert build_horizon_desc(12, 60) == "12h ahead"


def test_build_horizon_desc_days():
    assert build_horizon_desc(24, 60) == "D+1"
    assert build_horizon_desc(48, 60) == "D+2"
    assert build_horizon_desc(144, 10) == "D+1"


# --- run_training with mock backend ---


class MockBackend:
    """Minimal backend that returns dummy predictions for testing."""

    @property
    def name(self) -> str:
        return "mock"

    def mlflow_setup(self) -> None:
        pass

    def extra_params(self) -> dict[str, Any]:
        return {"mock_param": "test_value"}

    def train(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: pl.DataFrame,
        y_val: pl.Series,
    ) -> str:
        return "mock_model"

    def predict(self, model: Any, X: pl.DataFrame) -> np.ndarray:
        return np.full(len(X), 100.0)

    def log_child_artifacts(self, model: Any, horizon: int) -> None:
        pass

    def log_model(
        self,
        model: Any,
        X_val: pl.DataFrame,
        y_pred: np.ndarray,
        horizon: int,
    ) -> str | None:
        return f"models:/mock-model-h{horizon:02d}"

    def describe_model(self, model: Any) -> str:
        return "MockModel"


@pytest.fixture
def _wind_features_parquet(tmp_path):
    """Create a minimal wind features parquet for testing."""
    n = 500
    dates = [datetime(2015, 1, 1, h % 24) for h in range(n)]
    # Spread across years for temporal split
    for i in range(n):
        year = 2015 + i * 11 // n
        dates[i] = dates[i].replace(year=year)
    dates.sort()

    df = pl.DataFrame(
        {
            "timestamp_utc": dates,
            "active_power_kw": np.random.default_rng(42).uniform(0, 1000, n).tolist(),
            "active_power_kw_lag1": np.random.default_rng(43).uniform(0, 1000, n).tolist(),
            "hour": [d.hour for d in dates],
        }
    )
    path = tmp_path / "kelmarsh_kwf1.parquet"
    df.write_parquet(path)
    return tmp_path


def test_run_training_mock_backend(_wind_features_parquet, tmp_path):
    """Test that run_training works end-to-end with a mock backend."""
    tracking_uri = f"sqlite:///{tmp_path}/mlflow_test.db"

    with patch("windcast.config.get_settings") as mock_settings:
        mock_settings.return_value.mlflow_tracking_uri = tracking_uri
        mock_settings.return_value.train_years = 5
        mock_settings.return_value.val_years = 1
        mock_settings.return_value.features_dir = _wind_features_parquet

        run_training(
            backend=MockBackend(),
            domain="wind",
            dataset="kelmarsh",
            feature_set_name="wind_baseline",
            features_path=_wind_features_parquet / "kelmarsh_kwf1.parquet",
            experiment_name="test-harness",
            horizons=[1],
            turbine_id="kwf1",
            generation="gen_test",
            nwp_source="forecast",
            data_quality="CLEAN",
            train_years=8,
            val_years=2,
            log_models=False,
        )

    # Verify MLflow run was created with correct tags
    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(
        experiment_names=["test-harness"],
        filter_string="tags.`enercast.run_type` = 'parent'",
        output_format="pandas",
    )
    assert len(runs) == 1
    assert runs.iloc[0]["tags.enercast.backend"] == "mock"
    assert runs.iloc[0]["tags.enercast.data_quality"] == "CLEAN"
    assert runs.iloc[0]["tags.enercast.generation"] == "gen_test"
    assert runs.iloc[0]["tags.enercast.nwp_source"] == "forecast"
