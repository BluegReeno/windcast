"""AutoGluon-Tabular wrapper for MLflow pyfunc serving."""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc  # pyright: ignore[reportPrivateImportUsage]
import numpy as np
import pandas as pd


class AutoGluonPyfuncWrapper(mlflow.pyfunc.PythonModel):  # pyright: ignore[reportPrivateImportUsage]
    """Wraps a TabularPredictor for MLflow model serving.

    MLflow copies the predictor directory as an artifact.
    load_context reconstructs the predictor from that path.
    """

    def load_context(self, context: Any) -> None:
        from autogluon.tabular import TabularPredictor

        self.predictor = TabularPredictor.load(context.artifacts["ag_predictor"])

    def predict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict | None = None,
    ) -> np.ndarray:
        preds: np.ndarray = self.predictor.predict(model_input).values  # type: ignore[union-attr]
        return preds
