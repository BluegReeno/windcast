# WindCast Python Stack — pyproject.toml Reference

Researched 2026-03-31. All versions verified against PyPI.

---

## Verified Latest Versions

| Package | Latest | Minimum (PRD) |
|---------|--------|---------------|
| polars | 1.39.3 | >=1.0 |
| xgboost | 3.2.0 | >=2.0 (now 3.x) |
| lightgbm | 4.6.0 | >=4.0 |
| scikit-learn | 1.8.0 | >=1.4 |
| mlflow | 3.10.1 | >=2.10 (now 3.x) |
| optuna | 4.8.0 | >=3.0 |
| pydantic | 2.12.5 | >=2.0 |
| pydantic-settings | 2.13.1 | >=2.0 |
| openmeteo-requests | 1.7.5 | — |
| requests-cache | 1.3.1 | — |
| retry-requests | 2.0.0 | — |
| ruff | 0.15.8 | — |
| pyright | 1.1.408 | — |
| pytest | 9.0.2 | — |

---

## Complete pyproject.toml

```toml
[project]
name = "windcast"
version = "0.1.0"
description = "Standardized ML framework for wind power forecasting"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    # Data processing
    "polars>=1.39.3",
    # ML models
    "xgboost>=3.2.0",
    "lightgbm>=4.6.0",
    "scikit-learn>=1.8.0",
    # Experiment tracking
    "mlflow>=3.10.1",
    # Hyperparameter tuning
    "optuna>=4.8.0",
    # Configuration
    "pydantic>=2.12.5",
    "pydantic-settings>=2.13.1",
    # Weather data (Open-Meteo official Python client)
    "openmeteo-requests>=1.7.5",
    "requests-cache>=1.3.1",
    "retry-requests>=2.0.0",
    # Utilities
    "numpy>=2.0",
]

[dependency-groups]
dev = [
    "pytest>=9.0.2",
    "pytest-cov>=6.0.0",
    "ruff>=0.15.8",
    "pyright>=1.1.408",
]

# No [build-system] needed — WindCast is an application (scripts + src),
# not a published package. uv manages the env without a build backend.
# If you add a src/ package to install in editable mode, add:
#   [build-system]
#   requires = ["uv_build>=0.11.2,<0.12"]
#   build-backend = "uv_build"

[tool.ruff]
target-version = "py312"
line-length = 100
indent-width = 4

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # Pyflakes (undefined names, unused imports)
    "I",      # isort (import ordering)
    "N",      # pep8-naming
    "UP",     # pyupgrade (modernize syntax)
    "B",      # flake8-bugbear (subtle bugs)
    "SIM",    # flake8-simplify
    "PERF",   # Perflint (performance anti-patterns)
    "NPY",    # NumPy-specific rules (deprecated patterns)
    "RUF",    # Ruff-specific rules
]
ignore = [
    "E501",   # line too long — handled by formatter
    "B008",   # do not perform function calls in argument defaults (Pydantic uses this)
    "N803",   # argument name should be lowercase (ML often uses X, y)
    "N806",   # variable in function should be lowercase (ML: X, X_train, etc.)
]
fixable = ["ALL"]

[tool.ruff.lint.isort]
known-first-party = ["windcast"]
force-sort-within-sections = true

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"

[tool.pyright]
include = ["src", "scripts"]
exclude = ["**/__pycache__", ".venv"]
pythonVersion = "3.12"
typeCheckingMode = "standard"
reportMissingImports = "error"
reportMissingTypeStubs = false   # many ML libs lack full stubs

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

---

## Key Decisions & Rationale

### Build system: none (application project)

uv distinguishes between **application** and **library** projects. WindCast is an
application (scripts that run, not a package published to PyPI). For applications, uv
does **not** require a `[build-system]` section — it installs dependencies directly.

If you later add a `src/windcast/` package (recommended for testability), add:
```toml
[build-system]
requires = ["uv_build>=0.11.2,<0.12"]
build-backend = "uv_build"
```
`uv_build` is uv's native build backend (as of uv 0.6+), preferred over hatchling for
new uv-managed projects. Hatchling remains a valid alternative if interoperability
with other tools matters.

### Dev dependencies: `[dependency-groups]` not `[tool.uv.dev-dependencies]`

uv uses the PEP 735 `[dependency-groups]` table (standardized). Use:
```
uv add --dev pytest ruff pyright
```
This creates a `dev` group under `[dependency-groups]`. The `dev` group syncs by default.
Multiple groups are supported: `uv add --group lint ruff`.

### openmeteo-requests vs open-meteo

| Package | Maintainer | Style | Use case |
|---------|-----------|-------|----------|
| `openmeteo-requests` 1.7.5 | Open-Meteo (official) | Sync, requests-based | WindCast (simpler) |
| `open-meteo` 0.4.0 | Third-party (frenck) | Async, aiohttp-based | Async applications |

Use `openmeteo-requests` — it is the official Open-Meteo Python client maintained by the
API authors. Requires companion packages `requests-cache` and `retry-requests`.

---

## XGBoost 3.x Quantile Regression

XGBoost 3.x is current (latest 3.2.0). Quantile regression was introduced in 2.0 and is
stable in 3.x. **No breaking changes to the quantile regression API** between 2.0 and 3.0
for Python.

### v3.0 Python breaking changes (relevant to WindCast)
- `feval` parameter removed — use `eval_metric` with a custom `Metric` class instead
- Python minimum raised to 3.10
- Dask must be imported as `from xgboost import dask as dxgb` (not via `xgb.dask`)
- `DeviceQuantileDMatrix` removed — use `QuantileDMatrix` (same API)
- Old binary model format (non-JSON/UBJSON) can no longer be saved (loading still works)

### Low-level API (xgb.train)
```python
import numpy as np
import xgboost as xgb

alpha = np.array([0.05, 0.5, 0.95])  # quantiles to predict simultaneously

# QuantileDMatrix is memory-efficient for hist tree method
Xy_train = xgb.QuantileDMatrix(X_train, y_train)
Xy_val = xgb.QuantileDMatrix(X_val, y_val, ref=Xy_train)  # ref required

params = {
    "objective": "reg:quantileerror",  # the quantile loss objective
    "tree_method": "hist",             # required (exact not supported)
    "quantile_alpha": alpha,           # array of quantile levels
    "learning_rate": 0.05,
    "max_depth": 6,
    "n_estimators": 300,
}

booster = xgb.train(
    params,
    Xy_train,
    num_boost_round=300,
    evals=[(Xy_train, "train"), (Xy_val, "val")],
    early_stopping_rounds=20,
)

# Predict: output shape is (n_samples, n_quantiles)
scores = booster.inplace_predict(X_test)
y_lower = scores[:, 0]   # 5th percentile
y_median = scores[:, 1]  # 50th percentile
y_upper = scores[:, 2]   # 95th percentile
```

### sklearn API (XGBRegressor)
```python
# For single quantile prediction via sklearn interface:
model = xgb.XGBRegressor(
    objective="reg:quantileerror",
    tree_method="hist",
    quantile_alpha=0.9,   # single float for sklearn interface
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20)
y_pred = model.predict(X_test)  # shape: (n_samples,)
```

Note: For **multi-quantile** prediction in one pass, use `xgb.train` with
`quantile_alpha` as a numpy array. The sklearn estimator supports a single quantile.

---

## LightGBM Quantile Regression

```python
import lightgbm as lgb

params = {
    "objective": "quantile",
    "alpha": 0.9,           # quantile level (single value per model)
    "metric": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_estimators": 300,
}

# Train separate models per quantile:
models = {}
for q in [0.05, 0.5, 0.95]:
    p = {**params, "alpha": q}
    models[q] = lgb.train(
        p,
        lgb.Dataset(X_train, y_train),
        valid_sets=[lgb.Dataset(X_val, y_val)],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
    )

# Or via sklearn API:
from lightgbm import LGBMRegressor
model = LGBMRegressor(objective="quantile", alpha=0.9, n_estimators=300)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

LightGBM supports one quantile per model. For prediction intervals, train 3 separate
models (q=0.05, q=0.5, q=0.95).

---

## MLflow 3.x File-Based Tracking

MLflow 3.x (current: 3.10.1) is a major version that focuses on GenAI/LLMOps features
while preserving the classical ML tracking API. The core `mlflow.start_run()`,
`log_param()`, `log_metric()`, `log_artifact()` API is **unchanged**.

### Relevant MLflow 3.x breaking changes
- `mlflow.search_trace()` DataFrame schema changed (GenAI feature, not relevant to WindCast)
- `mlflow.evaluate()` no longer logs SHAP explainer by default
- Prompt registry APIs moved under `mlflow.genai.prompts` (not relevant)

### File-based tracking setup (unchanged from 2.x)
```python
import mlflow

# Set local file store — same API as 2.x
mlflow.set_tracking_uri("file:./mlruns")

# Experiment management
mlflow.set_experiment("windcast-xgb-kelmarsh")

with mlflow.start_run(run_name="xgb_quantile_v1"):
    mlflow.log_params({
        "objective": "reg:quantileerror",
        "quantile_alpha": [0.1, 0.5, 0.9],
        "n_estimators": 300,
        "learning_rate": 0.05,
    })
    mlflow.log_metrics({
        "val_pinball_q50": 0.123,
        "val_coverage_90": 0.91,
    })
    # Log feature set as artifact for reproducibility
    mlflow.log_artifact("data/feature_list.json")
    # Log model
    mlflow.xgboost.log_model(booster, artifact_path="model")

# Tag experiments by dataset for cross-site comparison
mlflow.set_tag("dataset", "kelmarsh")
mlflow.set_tag("turbine", "T01")
```

### autolog (use cautiously)
```python
mlflow.xgboost.autolog(log_models=True, log_datasets=False)
# autolog captures params, metrics, feature importances automatically
# Disable log_datasets to avoid logging large training arrays
```

---

## Open-Meteo Client Usage

```python
import openmeteo_requests
import requests_cache
from retry_requests import retry

# Setup with caching (SQLite) and retry logic
cache_session = requests_cache.CachedSession(".cache/openmeteo", expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
client = openmeteo_requests.Client(session=retry_session)

# Fetch hub-height wind forecast
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 53.505,
    "longitude": 0.648,
    "hourly": [
        "wind_speed_100m",        # hub height (check turbine spec)
        "wind_direction_100m",
        "wind_gusts_10m",
        "temperature_2m",
        "surface_pressure",
    ],
    "wind_speed_unit": "ms",      # m/s, not km/h
    "forecast_days": 7,
    "timezone": "UTC",            # always UTC — SCADA may be local time
}

responses = client.weather_api(url, params=params)
response = responses[0]

hourly = response.Hourly()
wind_speed = hourly.Variables(0).ValuesAsNumpy()   # index matches params["hourly"] order
wind_dir = hourly.Variables(1).ValuesAsNumpy()
```

Gotcha: `openmeteo-requests` 1.7.5 depends on `niquests` (not `requests`). The
`requests-cache` session is compatible because `niquests` is a drop-in replacement.

---

## ruff Rule Selection Rationale

Selected rules for a Python 3.12 ML project:

| Rule | Code | Why |
|------|------|-----|
| pycodestyle errors | E | PEP 8 compliance |
| pycodestyle warnings | W | trailing whitespace, blank lines |
| Pyflakes | F | undefined names, unused imports (critical) |
| isort | I | consistent import ordering |
| pep8-naming | N | PascalCase classes, snake_case functions |
| pyupgrade | UP | Python 3.12 idioms (f-strings, etc.) |
| flake8-bugbear | B | mutable default args, bare `except` |
| flake8-simplify | SIM | redundant `if`, `bool()` wrapping |
| Perflint | PERF | avoid `list()` around comprehensions, etc. |
| NumPy | NPY | deprecated `np.bool`, `np.int` etc. |
| Ruff-native | RUF | ruff-specific improvements |

Ignored:
- `E501` — line length enforced by formatter, not linter
- `N803/N806` — ML convention uses uppercase `X`, `X_train`, `y`

---

## Pyright Configuration Notes

`typeCheckingMode = "standard"` is appropriate for a new project. Use `"strict"` only on
well-typed modules. `reportMissingTypeStubs = false` because xgboost, lightgbm, mlflow
ship partial stubs — setting this to `"error"` causes too much noise.

The `[tool.pyright]` section in `pyproject.toml` is fully supported (pyright reads it).
A separate `pyrightconfig.json` takes precedence if both exist — do not create both.

---

## Commands Quick Reference

```bash
# Install all deps including dev group
uv sync --all-groups

# Install only production deps
uv sync --no-dev

# Add a new dependency
uv add polars>=1.39.3

# Add a dev dependency
uv add --dev pytest

# Lint + format check
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/

# Auto-fix lint issues
uv run ruff check --fix src/ tests/ scripts/
uv run ruff format src/ tests/ scripts/

# Type check
uv run pyright src/

# Run tests
uv run pytest tests/ -v

# MLflow UI
uv run mlflow ui --backend-store-uri file:./mlruns
```
