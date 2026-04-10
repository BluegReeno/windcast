# Feature: Pass 11 — Model Registry Integration

The following plan should be complete, but validate documentation and codebase patterns before implementing.

Pay special attention to naming of existing utils, types and models. Import from the right files etc.

## Feature Description

Make trained EnerCast models **servable** by logging them with MLflow signatures and registering them to the Model Registry. After this pass, `mlflow models serve` can spin up a FastAPI endpoint from any trained model — zero custom server code. This is the foundation for Pass 12 (inference pipeline) and Pass 13 (Streamlit dashboard).

## User Story

As an ML engineer using EnerCast
I want trained models to be automatically logged with MLflow signatures and registered in the Model Registry
So that I can serve them via `mlflow models serve` and get predictions via REST API without writing any server code

## Problem Statement

Currently, `run_training()` trains per-horizon models and logs metrics/params to MLflow, but does NOT log the trained model artifacts themselves. XGBoost autolog has `log_models=False` explicitly. AutoGluon models are saved to temp directories that get cleaned up. There is no way to load a trained model from MLflow or serve it.

## Solution Statement

1. Add a `log_model()` method to the `TrainingBackend` Protocol
2. Implement it in `XGBoostBackend` (using `mlflow.xgboost.log_model()`) and `AutoGluonBackend` (using a custom `mlflow.pyfunc.PythonModel` wrapper)
3. Call `backend.log_model()` in the child run loop inside `run_training()`
4. Register the best-performing model (lowest MAE across horizons) to the Model Registry with an alias
5. Add a `--register` flag to `train.py` to control registration (default: off)

## Feature Metadata

**Feature Type**: Enhancement
**Estimated Complexity**: Medium
**Primary Systems Affected**: `training/backends.py`, `training/harness.py`, `scripts/train.py`
**Dependencies**: MLflow >= 2.10 (already installed — actual version is 3.10.1), xgboost, autogluon

---

## CONTEXT REFERENCES

### Relevant Codebase Files — YOU MUST READ THESE BEFORE IMPLEMENTING!

- `src/windcast/training/harness.py` (lines 22-57) — `TrainingBackend` Protocol definition. Add `log_model()` here.
- `src/windcast/training/harness.py` (lines 154-481) — `run_training()` loop. The child run block (line 324) is where `log_model()` will be called.
- `src/windcast/training/backends.py` (lines 18-58) — `XGBoostBackend`. Implement `log_model()` using `mlflow.xgboost.log_model()`.
- `src/windcast/training/backends.py` (lines 60-120) — `AutoGluonBackend`. Implement `log_model()` using custom pyfunc wrapper.
- `src/windcast/models/autogluon_model.py` — `train_autogluon()` returns `TabularPredictor`. The predictor's `.path` attribute gives the saved directory.
- `src/windcast/models/xgboost_model.py` — `train_xgboost()` returns `xgb.XGBRegressor`. Pass this directly to `mlflow.xgboost.log_model()`.
- `scripts/train.py` (lines 1-180) — CLI script. Add `--register` flag and `--model-name` arg.
- `tests/training/test_harness.py` (lines 164-255) — `MockBackend` class. Add `log_model()` stub.
- `src/windcast/training/__init__.py` — Module exports. No changes needed.
- `src/windcast/tracking/mlflow_utils.py` — MLflow utilities. Add `register_best_model()` helper.

### New Files to Create

- `src/windcast/models/autogluon_pyfunc.py` — Custom `mlflow.pyfunc.PythonModel` wrapper for AutoGluon
- `tests/training/test_model_registry.py` — Tests for model logging and registration

### Relevant Documentation

**MLflow API (verified against installed 3.10.1):**
- `mlflow.xgboost.log_model(xgb_model, name=..., signature=..., input_example=...)` — logs both `xgboost` and `python_function` flavors
- `mlflow.pyfunc.log_model(name=..., python_model=..., artifacts=..., signature=...)` — for custom models (AutoGluon)
- `mlflow.models.infer_signature(model_input, model_output)` — **requires pandas, not Polars** (Polars silently degrades to AnyType)
- `mlflow.register_model(model_uri, name)` — register from `runs:/<run_id>/<name>`
- `MlflowClient().set_registered_model_alias(name, alias, version)` — use aliases, NOT deprecated stages

**Critical gotchas:**
- SQLite backend (`sqlite:///mlflow.db`) supports model registry — confirmed
- `infer_signature()` MUST receive `pandas.DataFrame`, not `polars.DataFrame`
- MLflow 3.x: use `name=` parameter, NOT `artifact_path=` (deprecated)
- AutoGluon has no built-in MLflow flavor — needs custom `PythonModel` wrapper
- The `PythonModel.predict()` signature: `(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray`
- For AutoGluon wrapper: `artifacts={"ag_predictor": predictor.path}` passes the directory to MLflow. At load time, `context.artifacts["ag_predictor"]` gives the resolved path.

### Patterns to Follow

**Backend Protocol pattern** (from `harness.py:22-57`):
```python
class TrainingBackend(Protocol):
    def log_child_artifacts(self, model: Any, horizon: int) -> None: ...
    # New method follows the same pattern:
    def log_model(self, model: Any, X_val: pl.DataFrame, y_pred: np.ndarray, horizon: int) -> None: ...
```

**MLflow child run pattern** (from `harness.py:324`):
```python
with mlflow.start_run(run_name=f"h{h:02d}", nested=True):
    model = backend.train(X_train, y_train, X_val, y_val)
    y_pred = backend.predict(model, X_val)
    # ... metrics logging ...
    backend.log_child_artifacts(model, h)
    # NEW: log model with signature
    backend.log_model(model, X_val, y_pred, h)
```

**XGBoost autolog is currently configured with `log_models=False`** (backends.py:31-35). The new explicit `log_model()` call supersedes this — keep `log_models=False` to avoid double-logging.

**Naming conventions:**
- Model artifact name: `model_h{horizon:02d}` (e.g., `model_h06`)
- Registered model name: `enercast-{dataset}-{backend}-h{horizon:02d}` (e.g., `enercast-kelmarsh-xgboost-h06`)

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation — AutoGluon PythonModel Wrapper

Create the custom pyfunc wrapper that lets MLflow serialize/deserialize AutoGluon predictors. This is the only new file.

### Phase 2: Core — Backend Protocol + Implementations

Add `log_model()` to the Protocol and implement it in both backends. This is the main work.

### Phase 3: Integration — Harness + CLI + Registration

Wire `log_model()` into `run_training()`, add CLI flags, and implement optional model registration.

### Phase 4: Testing & Validation

Test model logging, loading, and prediction for both backends. Test registration flow.

---

## STEP-BY-STEP TASKS

### Task 1: CREATE `src/windcast/models/autogluon_pyfunc.py`

Create the custom PythonModel wrapper for AutoGluon:

```python
"""AutoGluon-Tabular wrapper for MLflow pyfunc serving."""

from __future__ import annotations

import numpy as np
import pandas as pd

import mlflow.pyfunc


class AutoGluonPyfuncWrapper(mlflow.pyfunc.PythonModel):
    """Wraps a TabularPredictor for MLflow model serving.

    MLflow copies the predictor directory as an artifact.
    load_context reconstructs the predictor from that path.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        from autogluon.tabular import TabularPredictor

        self.predictor = TabularPredictor.load(context.artifacts["ag_predictor"])

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict | None = None,
    ) -> np.ndarray:
        return self.predictor.predict(model_input).values
```

- **PATTERN**: Follows MLflow custom pyfunc pattern — stateless `__init__`, state loaded in `load_context`
- **GOTCHA**: Do NOT store `predictor.path` in `__init__` — use `context.artifacts` at load time
- **VALIDATE**: `uv run pyright src/windcast/models/autogluon_pyfunc.py`

### Task 2: UPDATE `src/windcast/training/harness.py` — Add `log_model` to Protocol

Add the new method to the `TrainingBackend` Protocol (after `log_child_artifacts` at line 51):

```python
def log_model(
    self,
    model: Any,
    X_val: pl.DataFrame,
    y_pred: np.ndarray,
    horizon: int,
) -> str | None:
    """Log model artifact to MLflow with signature. Returns artifact name or None."""
    ...
```

- **PATTERN**: Same style as `log_child_artifacts` — takes model + horizon, operates within active child run
- **IMPORTS**: No new imports needed (np and pl already imported)
- **GOTCHA**: Return type is `str | None` — returns the artifact name for downstream registration, or None if logging is skipped
- **VALIDATE**: `uv run pyright src/windcast/training/harness.py`

### Task 3: UPDATE `src/windcast/training/backends.py` — XGBoostBackend.log_model()

Add the `log_model` method to `XGBoostBackend`:

```python
def log_model(
    self,
    model: Any,
    X_val: pl.DataFrame,
    y_pred: np.ndarray,
    horizon: int,
) -> str | None:
    import mlflow.xgboost  # pyright: ignore[reportPrivateImportUsage]
    from mlflow.models import infer_signature

    sig = infer_signature(X_val.to_pandas(), y_pred)
    artifact_name = f"model_h{horizon:02d}"
    mlflow.xgboost.log_model(  # pyright: ignore[reportPrivateImportUsage]
        xgb_model=model,
        name=artifact_name,
        signature=sig,
        input_example=X_val.head(5).to_pandas(),
    )
    return artifact_name
```

- **PATTERN**: Uses `mlflow.xgboost.log_model()` which registers both `xgboost` and `python_function` flavors automatically
- **IMPORTS**: `mlflow.xgboost` and `mlflow.models` imported inside method (lazy, same as existing pattern in `mlflow_setup`)
- **GOTCHA**: `infer_signature` requires pandas — use `.to_pandas()`. `input_example` also needs pandas.
- **GOTCHA**: Use `name=` not `artifact_path=` (deprecated in MLflow 3.x)
- **VALIDATE**: `uv run pyright src/windcast/training/backends.py`

### Task 4: UPDATE `src/windcast/training/backends.py` — AutoGluonBackend.log_model()

Add the `log_model` method to `AutoGluonBackend`:

```python
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

    sig = infer_signature(X_val.to_pandas(), y_pred)
    artifact_name = f"model_h{horizon:02d}"

    mlflow.pyfunc.log_model(
        name=artifact_name,
        python_model=AutoGluonPyfuncWrapper(),
        artifacts={"ag_predictor": model.path},
        signature=sig,
        input_example=X_val.head(5).to_pandas(),
    )
    return artifact_name
```

- **PATTERN**: `model.path` is the directory where AutoGluon saved its predictor (set in `AutoGluonBackend.train()` at line 98)
- **IMPORTS**: `mlflow.pyfunc` and the new wrapper imported inside method
- **GOTCHA**: `artifacts` value is a directory path string — MLflow copies the entire directory tree
- **GOTCHA**: Must call `mlflow.autolog(disable=False)` before this (already done in `train_autogluon()`)
- **VALIDATE**: `uv run pyright src/windcast/training/backends.py`

### Task 5: UPDATE `src/windcast/training/harness.py` — Call log_model in run_training()

In `run_training()`, after the `backend.log_child_artifacts(model, h)` call (line 386), add the model logging call. Also collect child run IDs for registration.

Inside the child run block (after line 386):

```python
backend.log_child_artifacts(model, h)
artifact_name = backend.log_model(model, X_val, y_pred, h)
```

Also add a parameter to `run_training()` signature to control model logging:

```python
def run_training(
    ...,
    log_models: bool = True,
    register_model_name: str | None = None,
) -> None:
```

And conditionally call `log_model`:

```python
artifact_name = None
if log_models:
    artifact_name = backend.log_model(model, X_val, y_pred, h)
```

After the horizon loop (after `log_stepped_horizon_metrics`), if `register_model_name` is set, register the best-performing child model (lowest MAE):

```python
if register_model_name and log_models:
    # Find child run with lowest val MAE
    active = mlflow.active_run()
    parent_run_id = active.info.run_id if active else ""
    if parent_run_id and exp_obj:
        children = client.search_runs(
            experiment_ids=[exp_obj.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
            order_by=["metrics.mae ASC"],
            max_results=1,
        )
        if children:
            best_child = children[0]
            best_horizon = int(best_child.data.params.get("horizon_steps", "0"))
            model_uri = f"runs:/{best_child.info.run_id}/model_h{best_horizon:02d}"
            mv = mlflow.register_model(model_uri, register_model_name)
            client.set_registered_model_alias(
                register_model_name, "champion", str(mv.version)
            )
            logger.info(
                "Registered model %s version %s (h%d, MAE=%.1f)",
                register_model_name,
                mv.version,
                best_horizon,
                best_child.data.metrics.get("mae", float("nan")),
            )
```

- **PATTERN**: Registration uses MlflowClient for aliases (modern, not deprecated stages)
- **GOTCHA**: `exp_obj` and `client` are already created later in the function (lines 457-458). Move their creation before the registration block, or reuse them. Actually, they are created after the horizon loop already — the registration code just needs to be placed AFTER the existing client/exp_obj usage block (lines 457-468).
- **GOTCHA**: The `search_runs()` call with `order_by` returns children sorted by MAE — pick the first.
- **VALIDATE**: `uv run pyright src/windcast/training/harness.py`

### Task 6: UPDATE `scripts/train.py` — Add CLI flags

Add two new arguments to the argument parser:

```python
parser.add_argument(
    "--no-log-models",
    action="store_true",
    default=False,
    help="Skip logging model artifacts to MLflow (faster for experimentation)",
)
parser.add_argument(
    "--register",
    action="store_true",
    default=False,
    help="Register best model to MLflow Model Registry",
)
parser.add_argument(
    "--model-name",
    default=None,
    help="Registered model name. Default: enercast-{dataset}-{backend}",
)
```

Then pass to `run_training()`:

```python
model_name = args.model_name or f"enercast-{dataset}-{args.backend}"

run_training(
    ...,
    log_models=not args.no_log_models,
    register_model_name=model_name if args.register else None,
)
```

- **PATTERN**: Follows existing CLI pattern (argparse + defaults from settings)
- **GOTCHA**: Default is to log models but NOT register. Registration is explicit via `--register`.
- **VALIDATE**: `uv run python scripts/train.py --help` (check new flags appear)

### Task 7: UPDATE `tests/training/test_harness.py` — MockBackend + new tests

Add `log_model` to `MockBackend`:

```python
def log_model(
    self,
    model: Any,
    X_val: pl.DataFrame,
    y_pred: np.ndarray,
    horizon: int,
) -> str | None:
    return f"model_h{horizon:02d}"
```

Update `test_run_training_mock_backend` to pass `log_models=False` (keep existing test fast, no model artifact overhead):

```python
run_training(
    ...,
    log_models=False,
)
```

- **VALIDATE**: `uv run pytest tests/training/test_harness.py -v`

### Task 8: CREATE `tests/training/test_model_registry.py` — Integration tests

Create integration tests that verify model logging and loading for XGBoost:

```python
"""Tests for model logging and registry integration."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import mlflow
import numpy as np
import polars as pl
import pytest
import xgboost as xgb

from windcast.training.backends import XGBoostBackend
from windcast.training.harness import run_training


@pytest.fixture
def _wind_features_parquet(tmp_path):
    """Create a minimal wind features parquet for testing."""
    n = 500
    dates = [datetime(2015, 1, 1, h % 24) for h in range(n)]
    for i in range(n):
        year = 2015 + i * 11 // n
        dates[i] = dates[i].replace(year=year)
    dates.sort()

    rng = np.random.default_rng(42)
    df = pl.DataFrame(
        {
            "timestamp_utc": dates,
            "active_power_kw": rng.uniform(0, 1000, n).tolist(),
            "active_power_kw_lag1": rng.uniform(0, 1000, n).tolist(),
            "hour": [d.hour for d in dates],
        }
    )
    path = tmp_path / "kelmarsh_kwf1.parquet"
    df.write_parquet(path)
    return tmp_path


def test_xgboost_log_model():
    """Test that XGBoostBackend.log_model() produces a loadable model."""
    # Train a small model
    rng = np.random.default_rng(42)
    X = pl.DataFrame({"f1": rng.normal(size=100).tolist(), "f2": rng.normal(size=100).tolist()})
    y = pl.Series("target", rng.normal(size=100).tolist())

    backend = XGBoostBackend()
    model = backend.train(X, y, X, y)
    y_pred = backend.predict(model, X)

    tracking_uri = "sqlite:///test_registry.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-log-model")

    with mlflow.start_run():
        with mlflow.start_run(run_name="h01", nested=True) as child_run:
            artifact_name = backend.log_model(model, X, y_pred, horizon=1)
            assert artifact_name == "model_h01"

        # Load back the model and verify predictions
        model_uri = f"runs:/{child_run.info.run_id}/{artifact_name}"
        loaded = mlflow.pyfunc.load_model(model_uri)
        preds = loaded.predict(X.to_pandas())
        np.testing.assert_allclose(preds, y_pred, rtol=1e-5)


def test_run_training_with_model_logging(_wind_features_parquet, tmp_path):
    """Test run_training logs models when log_models=True."""
    tracking_uri = f"sqlite:///{tmp_path}/mlflow_registry_test.db"

    with patch("windcast.config.get_settings") as mock_settings:
        mock_settings.return_value.mlflow_tracking_uri = tracking_uri
        mock_settings.return_value.train_years = 5
        mock_settings.return_value.val_years = 1
        mock_settings.return_value.features_dir = _wind_features_parquet

        run_training(
            backend=XGBoostBackend(),
            domain="wind",
            dataset="kelmarsh",
            feature_set_name="wind_baseline",
            features_path=_wind_features_parquet / "kelmarsh_kwf1.parquet",
            experiment_name="test-registry",
            horizons=[1],
            turbine_id="kwf1",
            log_models=True,
            train_years=8,
            val_years=2,
        )

    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(
        experiment_names=["test-registry"],
        filter_string="tags.`enercast.run_type` = 'child'",
        output_format="pandas",
    )
    assert len(runs) == 1
    # Verify model artifact was logged
    child_run_id = runs.iloc[0]["run_id"]
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(child_run_id)
    artifact_names = [a.path for a in artifacts]
    assert "model_h01" in artifact_names
```

- **PATTERN**: Mirrors existing `test_run_training_mock_backend` pattern
- **GOTCHA**: Each test creates its own SQLite DB to avoid interference
- **VALIDATE**: `uv run pytest tests/training/test_model_registry.py -v`

---

## TESTING STRATEGY

### Unit Tests

- `test_xgboost_log_model` — XGBoostBackend logs a model, it can be loaded back, predictions match
- `test_run_training_with_model_logging` — Full harness with `log_models=True` creates model artifacts in child runs
- `MockBackend.log_model` — Returns artifact name string, existing harness test still passes

### Integration Tests (manual)

- Train a real model: `uv run python scripts/train.py --feature-set wind_baseline --horizons 1`
- Verify model artifact in MLflow: check `model_h01` directory exists
- Load and serve: `MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow models serve -m "runs:/<run_id>/model_h01" --env-manager local -p 5001`
- Test prediction: `curl -X POST http://127.0.0.1:5001/invocations -H 'Content-Type: application/json' -d '{"dataframe_records": [{"wind_speed_ms": 8.0, ...}]}'`
- Register: `uv run python scripts/train.py --feature-set wind_baseline --horizons 1 --register`
- Serve registered: `MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow models serve -m "models:/enercast-kelmarsh-xgboost@champion" --env-manager local -p 5001`

### Edge Cases

- `--no-log-models` flag skips all model logging (fast experimentation mode)
- AutoGluon model logging with temp directory path
- Registration with no existing registered model (creates new)
- Registration with existing model (creates new version)

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

```bash
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run pyright src/
```

### Level 2: Unit Tests

```bash
# Run only the new/changed tests
uv run pytest tests/training/test_harness.py -v
uv run pytest tests/training/test_model_registry.py -v

# Full test suite (322+ tests)
uv run pytest tests/ -v
```

### Level 3: Integration Test

```bash
# Train with model logging (single horizon for speed)
uv run python scripts/train.py --feature-set wind_baseline --horizons 1

# Train with registration
uv run python scripts/train.py --feature-set wind_baseline --horizons 1 --register --model-name enercast-kelmarsh-xgboost
```

### Level 4: Serve & Predict

```bash
# Serve the registered model
MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow models serve \
  -m "models:/enercast-kelmarsh-xgboost@champion" \
  --env-manager local -p 5001 &

# Test prediction
curl -X POST http://127.0.0.1:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_records": [{"wind_speed_ms": 8.0, "wind_dir_sin": 0.5, "wind_dir_cos": 0.87, "active_power_kw_lag1": 500.0, "hour": 12}]}'
```

---

## ACCEPTANCE CRITERIA

- [ ] `TrainingBackend` Protocol has `log_model()` method
- [ ] `XGBoostBackend.log_model()` logs model with signature via `mlflow.xgboost.log_model()`
- [ ] `AutoGluonBackend.log_model()` logs model via custom pyfunc wrapper
- [ ] `run_training()` calls `backend.log_model()` in child runs (when `log_models=True`)
- [ ] `train.py --no-log-models` skips model logging
- [ ] `train.py --register` registers best model with `@champion` alias
- [ ] `mlflow models serve -m "models:/enercast-kelmarsh-xgboost@champion"` starts successfully
- [ ] `curl /invocations` returns predictions
- [ ] All existing tests still pass (zero regressions)
- [ ] New tests cover model logging and loading
- [ ] `ruff check`, `ruff format --check`, `pyright` all pass

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order (1-8)
- [ ] Level 1: ruff check + format + pyright pass
- [ ] Level 2: pytest tests/training/ pass
- [ ] Level 3: Full test suite passes (322+ tests)
- [ ] Level 4: Integration train with --register works
- [ ] All acceptance criteria met

---

## NOTES

### Design Decisions

1. **`log_model()` on the Backend, not in the harness** — Each backend knows how to serialize its model type. XGBoost uses the native flavor; AutoGluon needs a pyfunc wrapper. The harness stays model-agnostic.

2. **`log_models=True` by default, registration explicit** — Logging model artifacts adds ~2s per horizon (mostly signature inference). Worth it by default for reproducibility. Registration is a deliberate act (creates entries in the Model Registry that persist).

3. **Register the BEST child (lowest MAE), not all children** — For serving, you want the best model. The `@champion` alias always points to the latest best. Previous versions are preserved in the registry.

4. **Registered model naming: `enercast-{dataset}-{backend}`** — One registered model per dataset+backend combo. Each registration creates a new version. The `@champion` alias makes it easy to always serve the latest best.

5. **No changes to XGBoost autolog** — Keep `log_models=False` in `mlflow_setup()` to avoid double-logging. The explicit `log_model()` call gives us control over signature and naming.

### Risk: AutoGluon model size

AutoGluon predictor directories can be 100MB+ (multiple sub-models, ensemble weights). MLflow copies the entire directory as an artifact. For the demo this is fine (local SQLite store). For production, consider artifact storage on S3.

### Risk: MLflow 3.x deprecation warnings

`artifact_path=` is deprecated in favor of `name=`. The plan uses `name=` throughout. If any warnings appear, they're informational only.

### Training time impact

Model logging adds:
- XGBoost: ~1-2s per horizon (serialize XGBRegressor + signature inference)
- AutoGluon: ~3-5s per horizon (copy predictor directory + signature inference)

For 5 horizons, total overhead: ~10-25s. Negligible compared to training time.
