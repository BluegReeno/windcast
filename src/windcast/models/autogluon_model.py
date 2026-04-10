"""AutoGluon-Tabular training wrapper with MLflow-safe autolog handling."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import polars as pl
from pydantic import BaseModel

if TYPE_CHECKING:
    from autogluon.tabular import TabularPredictor

logger = logging.getLogger(__name__)


class AutoGluonConfig(BaseModel):
    """Configuration for AutoGluon-Tabular training."""

    presets: str = "best_quality"
    time_limit: int = 300
    excluded_model_types: list[str] = [
        "NN_TORCH",
        "FASTAI",
        "TABPFNV2",
        "TabICL",
        "MITRA",
    ]
    eval_metric: str = "mean_absolute_error"


def train_autogluon(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_val: pl.DataFrame,
    y_val: pl.Series,
    config: AutoGluonConfig | None = None,
    ag_path: Path | None = None,
) -> TabularPredictor:
    """Train AutoGluon-Tabular on a single regression task.

    Holdout discipline: ``X_val`` / ``y_val`` are NEVER passed to ``fit()``. AutoGluon
    carves its own internal tuning holdout from ``train_data`` via bagging, so the
    caller's validation set stays fully out-of-sample for honest evaluation. Passing
    ``tuning_data=val_pd`` would let AG optimise ensemble weights + stacking layers +
    early stopping against the reported eval set, producing in-sample metrics that
    catastrophically overstate generalisation (observed -74% MAE gap on RTE demand
    before the fix, 2026-04-09).

    Disables MLflow autolog during fit (AutoGluon is incompatible).
    Converts Polars to pandas at the boundary.

    Args:
        X_train, y_train: Training data (Polars).
        X_val, y_val: Validation data (Polars) — used ONLY for the post-fit leaderboard
            report, never for model selection or ensemble weighting.
        config: AutoGluon config. Uses defaults if None.
        ag_path: Directory for AutoGluon artifacts. Uses tempdir if None.

    Returns:
        Fitted TabularPredictor.
    """
    from autogluon.tabular import TabularPredictor

    if config is None:
        config = AutoGluonConfig()

    # Polars -> pandas at the AutoGluon boundary
    label = y_train.name
    train_pd = X_train.to_pandas()
    train_pd[label] = y_train.to_pandas()
    val_pd = X_val.to_pandas()
    val_pd[label] = y_val.to_pandas()

    # CRITICAL: Disable MLflow autolog — AutoGluon breaks with sklearn monkey-patching
    mlflow.autolog(disable=True)

    path = str(ag_path) if ag_path else tempfile.mkdtemp(prefix="ag_")

    predictor = TabularPredictor(
        label=label,
        problem_type="regression",
        eval_metric=config.eval_metric,
        path=path,
        verbosity=1,
    )

    # NOTE: no tuning_data — AG carves its own holdout from train_data via
    # use_bag_holdout=True, so stacker/ensemble selection never touches val_pd.
    # See docstring above for the leak that prompted this discipline.
    predictor.fit(
        train_data=train_pd,
        presets=config.presets,
        time_limit=config.time_limit,
        excluded_model_types=config.excluded_model_types,
        use_bag_holdout=True,
    )

    # Re-enable autolog for other code
    mlflow.autolog(disable=False)

    # Log summary — leaderboard on val_pd is a post-hoc report, NOT used for selection
    lb = predictor.leaderboard(data=val_pd, silent=True)
    best_model = lb.iloc[0]["model"]
    best_score = -lb.iloc[0]["score_val"]  # Negate: AG uses higher-is-better
    n_models = len(lb)

    logger.info(
        "AutoGluon trained: %d models, best=%s (val MAE=%.1f)",
        n_models,
        best_model,
        best_score,
    )

    return predictor
