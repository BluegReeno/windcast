"""Tests for windcast.models.mlforecast_model module."""

from datetime import datetime
from unittest.mock import patch

import numpy as np
import polars as pl

from windcast.models.mlforecast_model import (
    DOMAIN_MLFORECAST,
    MLForecastConfig,
    create_mlforecast,
    predict_mlforecast,
    prepare_mlforecast_df,
    train_mlforecast,
)


def _make_mlforecast_data(
    n_rows: int = 300, n_series: int = 2, with_exog: bool = False
) -> pl.DataFrame:
    """Create synthetic time series data in mlforecast format."""
    rng = np.random.default_rng(42)
    dates = pl.datetime_range(
        datetime(2020, 1, 1),
        datetime(2020, 1, 1) + pl.duration(hours=n_rows - 1),
        interval="1h",
        eager=True,
    )
    dfs = []
    for i in range(n_series):
        data: dict[str, object] = {
            "unique_id": [f"s{i}"] * n_rows,
            "ds": dates,
            "y": rng.standard_normal(n_rows).cumsum() + 100,
        }
        if with_exog:
            data["exog1"] = rng.standard_normal(n_rows)
        dfs.append(pl.DataFrame(data))
    return pl.concat(dfs)


def _make_wind_scada_data(n_rows: int = 200) -> pl.DataFrame:
    """Create synthetic SCADA data for prepare_mlforecast_df testing."""
    rng = np.random.default_rng(42)
    dates = pl.datetime_range(
        datetime(2020, 1, 1),
        datetime(2020, 1, 1) + pl.duration(minutes=(n_rows - 1) * 10),
        interval="10m",
        eager=True,
    )
    return pl.DataFrame(
        {
            "turbine_id": ["kwf1"] * n_rows,
            "timestamp_utc": dates,
            "active_power_kw": rng.standard_normal(n_rows) * 500 + 1000,
            "wind_speed_ms": rng.uniform(3, 15, n_rows),
            "wind_dir_sin": rng.standard_normal(n_rows),
            "wind_dir_cos": rng.standard_normal(n_rows),
            "qc_flag": [0] * n_rows,
        }
    )


class TestMLForecastConfig:
    def test_defaults(self):
        config = MLForecastConfig()
        assert config.n_estimators == 500
        assert config.learning_rate == 0.05
        assert config.strategy == "sparse_direct"
        assert config.n_cv_windows == 3

    def test_custom_values(self):
        config = MLForecastConfig(n_estimators=100, strategy="recursive")
        assert config.n_estimators == 100
        assert config.strategy == "recursive"


class TestDomainMLForecast:
    def test_wind_config(self):
        dcfg = DOMAIN_MLFORECAST["wind"]
        assert dcfg["target"] == "active_power_kw"
        assert dcfg["group"] == "turbine_id"
        assert dcfg["freq"] == "10m"
        assert 1 in dcfg["lags"]  # type: ignore[operator]

    def test_demand_config(self):
        dcfg = DOMAIN_MLFORECAST["demand"]
        assert dcfg["target"] == "load_mw"
        assert dcfg["group"] == "zone_id"
        assert dcfg["freq"] == "1h"

    def test_solar_config(self):
        dcfg = DOMAIN_MLFORECAST["solar"]
        assert dcfg["target"] == "power_kw"
        assert dcfg["group"] == "system_id"
        assert dcfg["freq"] == "15m"


class TestPrepareMLForecastDf:
    def test_wind_rename(self):
        df = _make_wind_scada_data(50)
        result = prepare_mlforecast_df(df, "wind")
        assert "unique_id" in result.columns
        assert "ds" in result.columns
        assert "y" in result.columns
        assert "turbine_id" not in result.columns
        assert "timestamp_utc" not in result.columns
        assert "active_power_kw" not in result.columns

    def test_wind_keeps_exogenous(self):
        df = _make_wind_scada_data(50)
        result = prepare_mlforecast_df(df, "wind")
        assert "wind_speed_ms" in result.columns
        assert "wind_dir_sin" in result.columns

    def test_wind_drops_qc_flag(self):
        df = _make_wind_scada_data(50)
        result = prepare_mlforecast_df(df, "wind")
        assert "qc_flag" not in result.columns

    def test_unique_id_is_string(self):
        df = _make_wind_scada_data(50)
        result = prepare_mlforecast_df(df, "wind")
        assert result["unique_id"].dtype == pl.Utf8

    def test_demand_rename(self):
        rng = np.random.default_rng(42)
        n = 50
        dates = pl.datetime_range(
            datetime(2020, 1, 1),
            datetime(2020, 1, 1) + pl.duration(hours=n - 1),
            interval="1h",
            eager=True,
        )
        df = pl.DataFrame(
            {
                "zone_id": ["ES"] * n,
                "timestamp_utc": dates,
                "load_mw": rng.uniform(20000, 40000, n),
                "temperature_c": rng.uniform(5, 35, n),
            }
        )
        result = prepare_mlforecast_df(df, "demand")
        assert "unique_id" in result.columns
        assert "ds" in result.columns
        assert "y" in result.columns
        assert "zone_id" not in result.columns

    def test_solar_rename(self):
        rng = np.random.default_rng(42)
        n = 50
        dates = pl.datetime_range(
            datetime(2020, 1, 1),
            datetime(2020, 1, 1) + pl.duration(minutes=(n - 1) * 15),
            interval="15m",
            eager=True,
        )
        df = pl.DataFrame(
            {
                "system_id": ["sys4"] * n,
                "timestamp_utc": dates,
                "power_kw": rng.uniform(0, 100, n),
                "poa_wm2": rng.uniform(0, 1000, n),
            }
        )
        result = prepare_mlforecast_df(df, "solar")
        assert "unique_id" in result.columns
        assert "y" in result.columns
        assert "system_id" not in result.columns


class TestCreateMLForecast:
    def test_creates_mlforecast_object(self):
        fcst = create_mlforecast("wind")
        assert fcst is not None
        assert fcst.freq == "10m"

    def test_demand_freq(self):
        fcst = create_mlforecast("demand")
        assert fcst.freq == "1h"

    def test_solar_freq(self):
        fcst = create_mlforecast("solar")
        assert fcst.freq == "15m"

    def test_custom_config(self):
        config = MLForecastConfig(n_estimators=100)
        fcst = create_mlforecast("wind", config)
        assert fcst is not None


class TestTrainMLForecast:
    @patch("windcast.models.mlforecast_model.mlflow")
    def test_fit_predict_sparse_direct(self, mock_mlflow):
        mock_mlflow.active_run.return_value = None
        df = _make_mlforecast_data(n_rows=300, n_series=2)
        config = MLForecastConfig(n_estimators=10, strategy="sparse_direct")
        horizons = [1, 2]

        fcst = train_mlforecast(df, "demand", config, horizons)
        preds = predict_mlforecast(fcst, h=max(horizons))

        assert len(preds) > 0
        assert "unique_id" in preds.columns
        assert "ds" in preds.columns
        assert "xgb" in preds.columns
        # sparse_direct: n_series * n_horizons rows
        assert len(preds) == 2 * len(horizons)

    @patch("windcast.models.mlforecast_model.mlflow")
    def test_fit_predict_recursive(self, mock_mlflow):
        mock_mlflow.active_run.return_value = None
        df = _make_mlforecast_data(n_rows=300, n_series=1)
        config = MLForecastConfig(n_estimators=10, strategy="recursive")

        fcst = train_mlforecast(df, "demand", config, horizons=[1, 2, 3])
        preds = predict_mlforecast(fcst, h=3)

        assert len(preds) > 0
        # recursive: n_series * h rows (all intermediate steps)
        assert len(preds) == 1 * 3

    @patch("windcast.models.mlforecast_model.mlflow")
    def test_fit_predict_direct(self, mock_mlflow):
        mock_mlflow.active_run.return_value = None
        df = _make_mlforecast_data(n_rows=300, n_series=1)
        config = MLForecastConfig(n_estimators=10, strategy="direct")

        fcst = train_mlforecast(df, "demand", config, horizons=[1, 2, 3])
        preds = predict_mlforecast(fcst, h=3)

        assert len(preds) > 0
        assert len(preds) == 1 * 3
