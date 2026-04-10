"""ML backend implementations for the training harness."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from windcast.models.xgboost_model import XGBoostConfig, train_xgboost

if TYPE_CHECKING:
    from windcast.models.autogluon_model import AutoGluonConfig


class XGBoostBackend:
    """XGBoost backend — wraps train_xgboost() with MLflow autolog."""

    def __init__(self, config: XGBoostConfig | None = None) -> None:
        self._config = config or XGBoostConfig()

    @property
    def name(self) -> str:
        return "xgboost"

    def mlflow_setup(self) -> None:
        import mlflow.xgboost  # pyright: ignore[reportPrivateImportUsage]

        mlflow.xgboost.autolog(  # pyright: ignore[reportPrivateImportUsage]
            log_datasets=False,
            log_models=False,
            log_model_signatures=False,
        )

    def extra_params(self) -> dict[str, Any]:
        return {}

    def train(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: pl.DataFrame,
        y_val: pl.Series,
    ) -> Any:
        return train_xgboost(X_train, y_train, X_val, y_val, self._config)

    def predict(self, model: Any, X: pl.DataFrame) -> np.ndarray:
        return model.predict(X)

    def log_child_artifacts(self, model: Any, horizon: int) -> None:
        pass

    def log_model(
        self,
        model: Any,
        X_val: pl.DataFrame,
        y_pred: np.ndarray,
        horizon: int,
    ) -> str | None:
        import mlflow.xgboost  # pyright: ignore[reportPrivateImportUsage]
        from mlflow.models import infer_signature

        X_pd = X_val.to_pandas()
        sig = infer_signature(X_pd, y_pred)
        artifact_name = f"model_h{horizon:02d}"
        info = mlflow.xgboost.log_model(  # pyright: ignore[reportPrivateImportUsage]
            xgb_model=model,
            name=artifact_name,
            signature=sig,
            input_example=X_pd.head(5),
        )
        return info.model_uri  # type: ignore[union-attr]

    def describe_model(self, model: Any) -> str:
        best_iter = getattr(model, "best_iteration", "?")
        return f"Trees: {best_iter}"


class AutoGluonBackend:
    """AutoGluon-Tabular backend — ensemble of gradient-boosted models."""

    def __init__(
        self,
        config: AutoGluonConfig | None = None,
        ag_base_path: Path | None = None,
    ) -> None:
        from windcast.models.autogluon_model import AutoGluonConfig

        self._config = config or AutoGluonConfig()
        self._ag_base_path = ag_base_path
        self._current_path: Path | None = None

    @property
    def name(self) -> str:
        return "autogluon"

    def mlflow_setup(self) -> None:
        pass

    def extra_params(self) -> dict[str, Any]:
        return {
            "ag.presets": self._config.presets,
            "ag.time_limit": self._config.time_limit,
            "ag.eval_metric": self._config.eval_metric,
        }

    def train(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: pl.DataFrame,
        y_val: pl.Series,
    ) -> Any:
        from windcast.models.autogluon_model import train_autogluon

        self._current_path = Path(tempfile.mkdtemp(prefix="ag_"))
        return train_autogluon(
            X_train, y_train, X_val, y_val, self._config, ag_path=self._current_path
        )

    def predict(self, model: Any, X: pl.DataFrame) -> np.ndarray:
        return model.predict(X.to_pandas()).values

    def log_child_artifacts(self, model: Any, horizon: int) -> None:
        import mlflow

        lb = model.leaderboard(data=None, silent=True)
        if self._current_path:
            lb_path = self._current_path / f"ag_leaderboard_h{horizon}.csv"
            lb.to_csv(str(lb_path), index=False)
            mlflow.log_artifact(str(lb_path))
            best_model = lb.iloc[0]["model"]
            n_models = len(lb)
            mlflow.log_params({"n_ag_models": n_models, "best_ag_model": best_model})

    def log_model(
        self,
        model: Any,
        X_val: pl.DataFrame,
        y_pred: np.ndarray,
        horizon: int,
    ) -> str | None:
        import mlflow.pyfunc
        from mlflow.models import infer_signature

        from windcast.models.autogluon_pyfunc import AutoGluonPyfuncWrapper

        X_pd = X_val.to_pandas()
        sig = infer_signature(X_pd, y_pred)
        artifact_name = f"model_h{horizon:02d}"

        info = mlflow.pyfunc.log_model(
            name=artifact_name,
            python_model=AutoGluonPyfuncWrapper(),
            artifacts={"ag_predictor": model.path},
            signature=sig,
            input_example=X_pd.head(5),
        )
        return info.model_uri  # type: ignore[union-attr]

    def describe_model(self, model: Any) -> str:
        lb = model.leaderboard(data=None, silent=True)
        return f"Best: {lb.iloc[0]['model']}, {len(lb)} models"
