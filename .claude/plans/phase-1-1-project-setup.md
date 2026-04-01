# Feature: Phase 1.1 — Project Setup

The following plan should be complete, but validate documentation and codebase patterns before implementing.

## Feature Description

Bootstrap the WindCast Python project from scratch: create `pyproject.toml` with all dependencies, configure tooling (ruff, pyright, pytest), set up the `src/windcast/` package layout with `__init__.py` files, configure MLflow local tracking, update `.gitignore` for Python/data/MLflow artifacts, and verify everything works with `uv sync` + `uv run pytest`.

## User Story

As an ML engineer
I want a fully configured Python project with all dependencies, linting, type checking, and testing ready
So that I can immediately start writing data ingestion and ML code without tooling friction

## Problem Statement

The WindCast repo currently contains only documentation (PRD, research docs, CLAUDE.md, README). There is no Python code, no `pyproject.toml`, no package structure. Everything needs to be bootstrapped.

## Solution Statement

Create a minimal but complete project scaffold:
1. `pyproject.toml` with all production + dev dependencies, ruff/pyright config
2. `src/windcast/` package with `__init__.py` stubs for all sub-packages
3. `.gitignore` additions for Python, data, MLflow
4. A smoke test to verify the setup works
5. `uv sync` to generate `uv.lock`

## Feature Metadata

**Feature Type**: New Capability (project bootstrap)
**Estimated Complexity**: Low
**Primary Systems Affected**: Project root, src/windcast/ package
**Dependencies**: uv (must be installed on system)

---

## CONTEXT REFERENCES

### Relevant Codebase Files — READ BEFORE IMPLEMENTING

- `.claude/PRD.md` (lines 136-198) — Directory structure specification
- `.claude/PRD.md` (lines 328-354) — Technology stack with versions
- `.claude/PRD.md` (lines 358-395) — Pydantic Settings config spec
- `CLAUDE.md` (lines 82-100) — Code style, naming conventions, commands
- `.gitignore` — Current file (only has OS/IDE ignores, needs Python/data entries)
- `README.md` — Already references `uv sync` and `uv run pytest`

### New Files to Create

- `pyproject.toml` — Project config, dependencies, tool settings
- `src/windcast/__init__.py` — Root package init
- `src/windcast/data/__init__.py` — Data sub-package init
- `src/windcast/features/__init__.py` — Features sub-package init
- `src/windcast/models/__init__.py` — Models sub-package init
- `tests/__init__.py` — Tests package init
- `tests/test_setup.py` — Smoke test to verify imports work
- `scripts/.gitkeep` — Placeholder for scripts directory

### Files to Modify

- `.gitignore` — Add Python, data, MLflow, uv entries

### Relevant Documentation

**Fallback URLs:**
- [uv pyproject.toml reference](https://docs.astral.sh/uv/reference/pyproject/) — project metadata format
- [ruff configuration](https://docs.astral.sh/ruff/configuration/) — pyproject.toml settings
- [pyright configuration](https://microsoft.github.io/pyright/#/configuration) — pyproject.toml settings
- [XGBoost 3.x changelog](https://xgboost.readthedocs.io/en/stable/gpu/index.html) — breaking changes from 2.x
- [MLflow 3.x docs](https://mlflow.org/docs/latest/index.html) — tracking API

### Patterns to Follow

**Naming Conventions:**
- Python files: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

**Import Groups (ruff `isort`):**
- stdlib → third-party → local (`src/windcast/`)

**Package Layout:**
- `src/` layout (PEP 621 compliant)
- Flat `__init__.py` with version only at root

---

## IMPLEMENTATION PLAN

### Phase 1: pyproject.toml + Dependencies

Create the project configuration file with all production and dev dependencies, plus tool configs for ruff, pyright, and pytest.

**Tasks:**
- Create `pyproject.toml` with metadata, dependencies, dev groups, tool configs
- Pin minimum versions per PRD (Polars ≥1.0, XGBoost ≥2.0, etc.)

### Phase 2: Package Structure

Create the `src/windcast/` package layout with all sub-packages as specified in the PRD directory structure.

**Tasks:**
- Create `src/windcast/` and sub-package directories
- Add `__init__.py` files
- Create `tests/` directory
- Create `scripts/` directory

### Phase 3: Git & Environment

Update `.gitignore` for Python ecosystem, run `uv sync` to install everything.

**Tasks:**
- Update `.gitignore`
- Run `uv sync`
- Verify install succeeded

### Phase 4: Validation

Verify everything works with a smoke test.

**Tasks:**
- Create minimal smoke test
- Run pytest, ruff, pyright

---

## STEP-BY-STEP TASKS

### Task 1: CREATE `pyproject.toml`

```toml
[project]
name = "windcast"
version = "0.1.0"
description = "Standardized ML framework for wind power forecasting"
readme = "README.md"
license = "MIT"
requires-python = ">=3.12"
dependencies = [
    # Data processing
    "polars>=1.0",
    # ML
    "xgboost>=2.0",
    "lightgbm>=4.0",
    "scikit-learn>=1.4",
    # Experiment tracking
    "mlflow>=2.10",
    # Hyperparameter tuning
    "optuna>=3.0",
    # Weather data
    "openmeteo-requests>=1.3",
    "requests-cache>=1.0",
    "retry-requests>=2.0",
    # Config
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.4",
    "pyright>=1.1",
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "SIM",  # flake8-simplify
    "PERF", # perflint
    "RUF",  # ruff-specific
]
ignore = [
    "N803",  # argument name should be lowercase (ML convention: X, X_train)
    "N806",  # variable in function should be lowercase (ML convention: X)
]

[tool.ruff.lint.isort]
known-first-party = ["windcast"]

[tool.pyright]
pythonVersion = "3.12"
venvPath = "."
venv = ".venv"
typeCheckingMode = "standard"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- **GOTCHA**: Use `>=` minimum versions, not exact pins — uv.lock handles exact resolution. The PRD specifies minimums.
- **GOTCHA**: `openmeteo-requests` is the correct PyPI package name (not `open-meteo-requests`). Requires `requests-cache` and `retry-requests` as companions.
- **GOTCHA**: XGBoost 3.x is now available with breaking changes from 2.x (`DeviceQuantileDMatrix` removed, use `QuantileDMatrix`). Keeping `>=2.0` allows both 2.x and 3.x.
- **GOTCHA**: MLflow 3.x is now available. Core tracking API (`log_param`, `log_metric`, `start_run`) is unchanged. `>=2.10` allows both.
- **GOTCHA**: No `[build-system]` needed — this is an application, not a published library. uv handles it.
- **VALIDATE**: File exists and is valid TOML: `python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"`

### Task 2: CREATE package structure

Create directories and `__init__.py` files:

```
src/
└── windcast/
    ├── __init__.py          # """WindCast — ML framework for wind power forecasting."""
    ├── data/
    │   └── __init__.py      # empty
    ├── features/
    │   └── __init__.py      # empty
    └── models/
        └── __init__.py      # empty

scripts/
└── .gitkeep                 # placeholder

tests/
└── __init__.py              # empty
```

- **PATTERN**: Root `__init__.py` has a docstring only. Sub-package `__init__.py` are empty.
- **PATTERN**: `scripts/` gets a `.gitkeep` since it has no Python files yet.
- **VALIDATE**: `python -c "import pathlib; assert pathlib.Path('src/windcast/__init__.py').exists()"`

### Task 3: UPDATE `.gitignore`

Add Python, data, MLflow, and uv entries to the existing `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
*.egg
dist/
build/
.eggs/

# Virtual environment
.venv/

# uv
uv.lock

# Data (large files, downloaded separately)
data/

# MLflow tracking
mlruns/

# Jupyter
.ipynb_checkpoints/
```

- **GOTCHA**: Do NOT remove existing entries (OS, IDE, Claude Code). Append new sections.
- **GOTCHA**: `uv.lock` — debate exists on whether to commit it. For an application (not library), committing is recommended for reproducibility. **Decision: DO commit uv.lock** — remove it from .gitignore. This ensures reproducible installs.
- **VALIDATE**: `grep -q "__pycache__" .gitignore && grep -q "mlruns" .gitignore && echo "OK"`

### Task 4: RUN `uv sync`

```bash
cd /Users/renaud/Projects/windcast && uv sync
```

- **GOTCHA**: This will create `.venv/` and `uv.lock`. May take a minute for XGBoost/LightGBM compilation.
- **GOTCHA**: If `uv` is not installed, install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **VALIDATE**: `uv run python -c "import windcast; print('OK')"` should print "OK"

### Task 5: CREATE `tests/test_setup.py` — Smoke test

```python
"""Smoke tests to verify project setup and imports."""


def test_windcast_importable():
    """Verify the windcast package can be imported."""
    import windcast

    assert windcast is not None


def test_core_dependencies_importable():
    """Verify all core dependencies are installed."""
    import lightgbm
    import mlflow
    import optuna
    import polars
    import pydantic
    import sklearn
    import xgboost

    assert all([polars, xgboost, lightgbm, sklearn, mlflow, optuna, pydantic])


def test_sub_packages_importable():
    """Verify all sub-packages can be imported."""
    import windcast.data
    import windcast.features
    import windcast.models

    assert all([windcast.data, windcast.features, windcast.models])
```

- **PATTERN**: Test file naming: `test_*.py` (pytest convention from pyproject.toml)
- **VALIDATE**: `uv run pytest tests/test_setup.py -v`

### Task 6: RUN full validation suite

```bash
# Lint
uv run ruff check src/ tests/

# Format check
uv run ruff format --check src/ tests/

# Type check
uv run pyright src/

# Tests
uv run pytest tests/ -v
```

- **VALIDATE**: All four commands exit with code 0

---

## TESTING STRATEGY

### Unit Tests

- `tests/test_setup.py` — Verify imports work for all packages and dependencies
- This is the only test for Phase 1.1. Subsequent phases add domain-specific tests.

### Validation

- ruff lint + format: zero errors
- pyright: zero errors
- pytest: all tests pass

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

```bash
# Ruff lint (must pass with 0 errors)
uv run ruff check src/ tests/

# Ruff format check
uv run ruff format --check src/ tests/
```

**Expected**: All commands pass with exit code 0

### Level 2: Type Checking

```bash
# Pyright
uv run pyright src/
```

**Expected**: 0 errors

### Level 3: Tests

```bash
# Pytest
uv run pytest tests/ -v
```

**Expected**: 3 tests pass (test_windcast_importable, test_core_dependencies_importable, test_sub_packages_importable)

### Level 4: Import Verification

```bash
# Verify windcast package
uv run python -c "import windcast; print('windcast OK')"

# Verify MLflow tracking URI works
uv run python -c "import mlflow; mlflow.set_tracking_uri('file:./mlruns'); print('mlflow OK')"

# Verify Polars
uv run python -c "import polars as pl; print(f'polars {pl.__version__} OK')"

# Verify XGBoost quantile regression support
uv run python -c "import xgboost as xgb; print(f'xgboost {xgb.__version__} OK')"
```

**Expected**: All print "OK"

---

## ACCEPTANCE CRITERIA

- [x] `pyproject.toml` exists with all dependencies from PRD tech stack
- [x] `src/windcast/` package with `data/`, `features/`, `models/` sub-packages
- [x] `.gitignore` covers Python, data, MLflow, venv
- [x] `uv sync` succeeds and creates `.venv/` + `uv.lock`
- [x] `uv run pytest tests/ -v` passes (3 smoke tests)
- [x] `uv run ruff check src/ tests/` passes with 0 errors
- [x] `uv run ruff format --check src/ tests/` passes
- [x] `uv run pyright src/` passes with 0 errors
- [x] `uv run python -c "import windcast"` succeeds
- [x] All core dependencies importable (polars, xgboost, lightgbm, mlflow, etc.)

---

## COMPLETION CHECKLIST

- [ ] Task 1: pyproject.toml created
- [ ] Task 2: Package structure created
- [ ] Task 3: .gitignore updated
- [ ] Task 4: uv sync succeeds
- [ ] Task 5: Smoke test created
- [ ] Task 6: Full validation passes (ruff, pyright, pytest)
- [ ] All acceptance criteria met

---

## NOTES

### Design Decisions

1. **`>=` version pins, not exact**: uv.lock handles exact resolution. Minimum versions match PRD requirements while allowing latest compatible releases.

2. **`uv.lock` committed**: This is an application, not a library. Committing the lockfile ensures reproducible environments across machines.

3. **No `[build-system]`**: uv handles this for applications. Adding one is only needed if publishing to PyPI.

4. **ruff N803/N806 ignored**: ML convention uses uppercase for matrices (`X`, `X_train`, `y_pred`). These naming rules would fight the entire sklearn/xgboost ecosystem.

5. **`openmeteo-requests` not `open-meteo`**: The PRD says "open-meteo-requests" but the correct PyPI package is `openmeteo-requests` (no hyphen before requests). It's the official client. Requires `requests-cache` and `retry-requests` companions.

6. **XGBoost ≥2.0 (not ≥3.0)**: The PRD specifies ≥2.0. uv will resolve to latest (3.x). Key breaking change in 3.x: use `QuantileDMatrix` not `DeviceQuantileDMatrix`. Quantile regression API (`reg:quantileerror`) is unchanged.

7. **MLflow ≥2.10 (not ≥3.0)**: Same approach. 3.x core tracking API is unchanged for classical ML.

### Confidence Score: 9/10

This is a straightforward project bootstrap. The only risk is dependency resolution conflicts (XGBoost 3.x + LightGBM 4.x + MLflow 3.x on macOS), which uv handles well. Everything else is mechanical file creation.
