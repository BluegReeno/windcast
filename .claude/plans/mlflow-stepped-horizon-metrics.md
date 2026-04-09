# Feature: Stepped Horizon Metrics for Native MLflow Line Charts

> **CORRECTION (2026-04-09 afternoon)** — the original plan was implemented
> but a design error was discovered after shipping: stepped metrics were
> logged on **child** runs, each with a single `step` point. This does not
> produce a native line chart because MLflow does not stitch metrics across
> sibling runs into one curve. Confirmed by MLflow maintainers in
> [mlflow/mlflow#2768](https://github.com/mlflow/mlflow/issues/2768)
> (*"it's not possible to plot metrics that belong to different runs as one
> curve"*) and [mlflow/mlflow#7060](https://github.com/mlflow/mlflow/issues/7060)
> (canonical pattern: one run, N `log_metric` calls at increasing `step`).
>
> **Fix applied**: stepped metrics are now logged on the **parent** run
> after the per-horizon child loop completes, via the new helper
> `log_stepped_horizon_metrics()`. Each parent accumulates all 5 horizons
> as a single metric history → renders natively as a multi-point line
> chart in the UI, one line per parent run. Anti-pattern #3 below (which
> forbade logging stepped on the parent) was wrong and has been inverted.
>
> See `docs/mlflow-ui-setup.md` → "Native line charts: metric vs horizon"
> for the corrected recipe, and `scripts/backfill_stepped_metrics.py` for
> retroactively fixing pre-refactor runs without retraining.

The following plan should be complete, but validate documentation and codebase patterns before implementing.

Pay special attention to: keep existing `h{n}_mae` metrics (needed for filtering + `compare_runs.py`), add stepped metrics *in addition*, NOT as a replacement. Step unit must be `minutes_ahead` for cross-domain consistency.

## Feature Description

Add a second, complementary metric logging path that uses MLflow's native `step` parameter so the UI can render **"MAE vs horizon" line charts** automatically — without writing custom chart configuration, and without the `compare_runs.py` post-processing script. Each MAE/skill/bias/persistence value is logged both as a flat metric (`h{n}_mae`, today's behaviour, preserved for filtering) **and** as a stepped metric (`mae_by_horizon_min` with `step=minutes_ahead`, new). The stepped variant unlocks MLflow's native line chart that shows one coloured line per run with X=horizon(min), Y=metric — exactly the view missing today.

## User Story

As an energy-forecasting engineer using EnerCast,
I want MLflow's native UI to show "metric vs horizon" line charts out of the box for any domain,
So that I can compare runs visually in one click (and train new users) without depending on a custom comparison script or manual Charts-tab configuration.

## Problem Statement

Today, each horizon is logged as its own metric name (`h1_mae`, `h6_mae`, `h12_mae`, `h24_mae`, `h48_mae`). The MLflow UI auto-generates **one bar chart per metric name** — giving 10+ disconnected charts (5 MAE + 5 skill), each showing a single bar per run. There is no native way in the UI to see "MAE as a function of horizon" for one run, let alone compare multiple runs on that curve. The user cannot add a combined line chart because the UI's Line-chart widget needs a metric with a step dimension, and our metrics have no step.

Workarounds today:
- `scripts/compare_runs.py` — external PNG generator, out-of-band from the UI
- `Compare Runs` button in MLflow UI → parallel-coordinates plot (works but is ad-hoc, not persistent, and shows every metric axis)
- Manual Charts-tab bar-chart-per-horizon — 10 widgets, cluttered, no visual continuity across horizons

## Solution Statement

Extend `log_evaluation_results` in `src/windcast/tracking/mlflow_utils.py` to optionally accept `horizon_minutes: int | None`. When provided, it **also** logs the same metric values under a set of canonical stepped names (`mae_by_horizon_min`, `rmse_by_horizon_min`, `skill_score_by_horizon_min`, `bias_by_horizon_min`, `persistence_mae_by_horizon_min`) with MLflow `step=horizon_minutes`. The existing `h{n}_` prefixed metrics remain unchanged — nothing is removed.

Both `scripts/train.py` and `scripts/train_autogluon.py` compute `h * data_resolution` (already in scope via `data_resolution`) and pass it as `horizon_minutes` to the evaluation logger. Zero other changes to the pipeline, zero config changes, zero schema changes, backwards compatible.

Cross-domain: the unit `minutes_ahead` is consistent across all 4 EnerCast pillars — wind (10 min resolution), demand (60 min), solar (15 min), price/EPEX (60 min, planned). This produces a single comparable metric name that works identically for all current and future domains.

## Feature Metadata

**Feature Type**: Enhancement
**Estimated Complexity**: Low
**Primary Systems Affected**: `src/windcast/tracking/mlflow_utils.py`, `scripts/train.py`, `scripts/train_autogluon.py`, `tests/test_tracking.py` (new or extend existing)
**Dependencies**: None new — uses existing `mlflow.log_metric(name, value, step=...)` API

---

## RESEARCH FINDINGS (Phase 1 output)

### Do all 4 pillars need multi-horizon forecasting? → **YES**

| Pillar | Resolution | Current horizons (steps) | Horizons in **minutes ahead** | Business rationale | Status |
|--------|-----------|--------------------------|-------------------------------|---------------------|--------|
| **Wind** (Kelmarsh) | 10 min | `[1, 6, 12, 24, 48]` | `10, 60, 120, 240, 480` | Short-term dispatch, ramp forecasting, 8h-ahead operations | ✅ Live |
| **Demand** (Spain) | 60 min | `[1, 6, 12, 24, 48]` | `60, 360, 720, 1440, 2880` | Intraday balancing, D+1 gate (demand forecast drives generation dispatch) | ✅ Code complete |
| **Solar** (PVDAQ) | 15 min | `[1, 6, 12, 24, 48]` | `15, 90, 180, 360, 720` | Cloud transients (15 min), intraday ramps, D+1 | ✅ Code complete |
| **Price** (EPEX spot, planned 4th domain) | 60 min | *not yet implemented* | `60, 720, 1440, 2880, 10080` (intraday, H+12, D+1, D+2, W+1) | Day-ahead gate (11:00 UTC), intraday gate closure, week-ahead bilateral contracts | 🔄 Planned (WattCast precedent) |

**All 4 share the need for multi-horizon metrics**, and all 4 can express their horizons as an integer "minutes ahead" — the natural step unit for a stepped metric.

**Price has one wrinkle** that this plan anticipates (but does not fix): a day-ahead price forecast for D+1 is *24 hourly values*, not a single value at horizon H. The "minutes ahead" for each of those 24 values varies between ~780 (11:00 UTC forecast → 00:00 D+1) and ~2160 (11:00 → 23:00 D+1). The stepped-metric design naturally handles this: each of the 24 D+1 hours becomes its own step on the line chart, producing a natural 24-point curve for day-ahead price forecasts. **No framework changes needed** when price is added later.

### Current logging (file:line)

- `src/windcast/tracking/mlflow_utils.py:46-60` — `log_evaluation_results(metrics, horizon)` prefixes with `h{horizon}_` and calls `mlflow.log_metrics(prefixed)`. This is where we add the stepped variant.
- `scripts/train.py:358` — call site: `log_evaluation_results(metrics, horizon=h)`. `data_resolution` is already in scope (line 242).
- `scripts/train_autogluon.py:355` — call site: identical pattern. `data_resolution` in scope (line 228).
- `scripts/train.py:405` / `scripts/train_autogluon.py:414` — parent bubble-up filter `if k.startswith("h") and ("_mae" in k or "_skill_score" in k)`. This needs **no change** — stepped metric names don't match the `h` prefix, so they are not re-logged on the parent. (The parent run gets the latest step automatically from nested child runs via MLflow's normal flow — verified in Phase 4 below.)

### MLflow native line chart behaviour

When a metric is logged multiple times to the same run with distinct `step` values, MLflow stores a time series. The UI's **Line chart** widget (in the experiment's Charts tab and in Compare Runs) automatically plots X=step, Y=value for that metric, with one line per run. This is the intended use of the `step` parameter — it is NOT limited to "training epoch" (the most common example in MLflow docs). Any monotonic integer works: epoch, batch, horizon_minutes, lookahead_samples, etc.

Reference: `mlflow.log_metric(key, value, step=int)` — see MLflow Tracking API docs.

---

## CONTEXT REFERENCES

### Relevant Codebase Files — READ BEFORE IMPLEMENTING

- `src/windcast/tracking/mlflow_utils.py` (full file, 75 lines) — **Primary file to modify.** Understand the existing `log_evaluation_results` signature and how it's used.
- `scripts/train.py` lines 240-260, 305-392 — Current training loop per horizon. Understand where `data_resolution` is defined and where `h` is in scope. Line 358 is the call site for `log_evaluation_results`.
- `scripts/train_autogluon.py` lines 225-260, 295-400 — Same structure, second call site. Line 355 is the call site.
- `src/windcast/models/evaluation.py` lines 35-70 — `compute_metrics()` returns the dict that feeds the logger (`mae`, `rmse`, `bias`, optional `skill_score` and `mape`). Understand the exact keys so the stepped metric mapping is complete.
- `src/windcast/tracking/__init__.py` — exports `log_evaluation_results`. No change here (backward compatible signature).
- `tests/` — look for any existing test of `log_evaluation_results`. Verified via grep: **no test file currently covers `mlflow_utils.py`** (no matches for `log_evaluation_results` in `tests/`). This feature should add one.
- `.claude/plans/pass4-weather-provider-layer.md` — reference plan format + granularity expectations in this codebase.
- `docs/mlflow-ui-setup.md` lines 50-90 — UI walkthrough that will need a new "Line chart recipe" subsection after this feature lands.

### New Files to Create

- `tests/test_tracking.py` — unit tests for `log_evaluation_results` covering both the legacy path (`horizon=h`, no minutes) and the new path (`horizon=h, horizon_minutes=m`). Uses MLflow's file store in a tmp_path fixture.

### Files to Modify

- `src/windcast/tracking/mlflow_utils.py` — extend `log_evaluation_results` signature.
- `scripts/train.py` — pass `horizon_minutes=h * data_resolution` at line 358.
- `scripts/train_autogluon.py` — pass `horizon_minutes=h * data_resolution` at line 355.
- `docs/mlflow-ui-setup.md` — add a "Line charts (native, no config)" section documenting the new metrics and how to use the Line-chart widget.
- `.claude/STATUS.md` — short "done" entry under `Planned Improvements (post-presentation)`.

### Patterns to Follow

**Function signature extension (backward compatible)**:
```python
def log_evaluation_results(
    metrics: dict[str, float],
    horizon: int | None = None,
    horizon_minutes: int | None = None,
) -> None:
```

**Stepped metric names — canonical registry**:
```python
STEPPED_METRIC_MAP: dict[str, str] = {
    "mae": "mae_by_horizon_min",
    "rmse": "rmse_by_horizon_min",
    "bias": "bias_by_horizon_min",
    "skill_score": "skill_score_by_horizon_min",
    "mape": "mape_by_horizon_min",
}
```

**Logging idiom** (mirrors existing `mlflow.log_metrics` call immediately above):
```python
if horizon_minutes is not None:
    for metric_name, value in metrics.items():
        stepped_name = STEPPED_METRIC_MAP.get(metric_name)
        if stepped_name is not None:
            mlflow.log_metric(stepped_name, value, step=horizon_minutes)
```

**Naming convention (from codebase)**: snake_case for Python, `_by_horizon_min` suffix chosen over alternatives (`_stepped`, `_vs_horizon`, `_curve`) because it (a) explicitly names the independent variable and (b) includes the unit, removing ambiguity with future `_by_horizon_hr` or `_by_forecast_day` if we ever need coarser grids for weekly price forecasts.

**Test pattern**: The existing test style in `tests/` uses pytest + tmp_path + polars. Mirror the style of `tests/models/test_evaluation.py` (no MLflow state) but add MLflow file-store setup via `mlflow.set_tracking_uri(f"file://{tmp_path}")` inside the test. Use `mlflow.tracking.MlflowClient().get_metric_history(run_id, metric_name)` to assert step values.

### Relevant Documentation

- **MLflow tracking — logging metrics with steps**: `mlflow.log_metric(key, value, step)` — canonical API. The `step` parameter is a monotonically increasing integer (MLflow stores it as int64). Not epoch-specific.
- **MLflow UI line charts**: the experiment Charts tab auto-detects metrics with multiple logged steps and offers a Line chart type that plots X=step, Y=value, one line per run. No config needed beyond "select the metric".
- **Compare Runs → Parallel coordinates**: not relevant for this feature (works off flat metrics). Worth documenting in `docs/mlflow-ui-setup.md` as complement.
- Fallback URL: <https://mlflow.org/docs/latest/tracking.html> (MLflow tracking overview)

### Anti-patterns to Avoid

- ❌ **Replacing** `h{n}_mae` with stepped-only. The `h{n}_` metrics are filterable (`metrics.h48_mae < 300`), while stepped metrics are NOT — MLflow `search_runs` returns the latest step's value only. `compare_runs.py` also depends on flat names.
- ❌ Using `step=h` (the raw horizon steps like 1/6/12/24/48). Inconsistent across domains (step=48 = 8h for wind vs 48h for demand). Always use `horizon_minutes` (h × data_resolution).
- ❌ Logging stepped metrics on the parent run. MLflow's nested-run semantics mean stepped metrics belong on children; the parent's summary comes from the `h{n}_` bubble-up already in place. If you stepped-log on the parent, the "one line per run" chart breaks because the parent is treated as an extra run.
- ❌ Logging `rmse_by_horizon_min` but not `rmse` in `STEPPED_METRIC_MAP`. The map must cover every key that `compute_metrics()` can return, otherwise metrics silently disappear from the line chart. Verify against `src/windcast/models/evaluation.py:35-70`.

---

## IMPLEMENTATION PLAN

### Phase 1: Tracking utility

Extend the canonical logger so both flat and stepped metrics are emitted from one call site.

### Phase 2: Training script wiring

Pass `horizon_minutes` from both training scripts using existing in-scope `data_resolution`.

### Phase 3: Tests

Add unit tests for both the legacy and the new logging paths. Verify the metric history has the correct step values and the correct metric names.

### Phase 4: Documentation + verification

Update `docs/mlflow-ui-setup.md` with the line-chart recipe. Run a real training pass and visually verify the line chart in the MLflow UI. Update STATUS.md.

---

## STEP-BY-STEP TASKS

Execute every task in order, top to bottom. Each task is atomic and independently testable.

### Task 1 — UPDATE `src/windcast/tracking/mlflow_utils.py`

- **IMPLEMENT**: Add `horizon_minutes: int | None = None` parameter to `log_evaluation_results`. Add a module-level `STEPPED_METRIC_MAP` dict mapping raw metric keys (`mae`, `rmse`, `bias`, `skill_score`, `mape`) to stepped names (`mae_by_horizon_min`, etc.). When `horizon_minutes is not None`, iterate `metrics.items()` and call `mlflow.log_metric(stepped_name, value, step=horizon_minutes)` for each key present in the map. The existing `h{n}_` prefix logic stays untouched.
- **PATTERN**: Mirror the existing `mlflow.log_metrics(prefixed)` call at `mlflow_utils.py:58` — same file, same function, just add a second code path below it.
- **IMPORTS**: No new imports (`mlflow` is already imported).
- **GOTCHA**: The `STEPPED_METRIC_MAP` must be a module-level constant, not built inside the function — pyright will flag unused mappings otherwise, and it's a natural extension point for future metric names. Also: `mlflow.log_metric` (singular) takes a `step` parameter; `mlflow.log_metrics` (plural) does **not**. Use the singular form in the stepped path.
- **VALIDATE**: `uv run ruff check src/windcast/tracking/mlflow_utils.py && uv run pyright src/`

### Task 2 — UPDATE `scripts/train.py`

- **IMPLEMENT**: At the `log_evaluation_results(metrics, horizon=h)` call site (line 358), add a third argument: `horizon_minutes=h * data_resolution`. The variable `data_resolution` is already defined at line 242 in the same function scope.
- **PATTERN**: The call is a one-line change. Do not refactor the surrounding block.
- **IMPORTS**: None.
- **GOTCHA**: There is only ONE call to `log_evaluation_results` in this file. Grep to confirm before editing.
- **VALIDATE**: `uv run ruff check scripts/train.py && grep -c "log_evaluation_results" scripts/train.py` — expect `1`.

### Task 3 — UPDATE `scripts/train_autogluon.py`

- **IMPLEMENT**: Same as Task 2, at line 355. `data_resolution` is defined at line 228.
- **PATTERN**: Identical mirror of Task 2.
- **IMPORTS**: None.
- **GOTCHA**: Same as Task 2 — only one call site.
- **VALIDATE**: `uv run ruff check scripts/train_autogluon.py && grep -c "log_evaluation_results" scripts/train_autogluon.py` — expect `1`.

### Task 4 — CREATE `tests/test_tracking.py`

- **IMPLEMENT**: Three pytest tests:
  1. `test_log_evaluation_results_legacy_mode` — call with `horizon=6` only, assert metrics `h6_mae`, `h6_rmse`, `h6_bias` are present in the run and stepped metrics are NOT present.
  2. `test_log_evaluation_results_stepped_mode` — call with `horizon=6, horizon_minutes=60`, assert both flat metrics (`h6_mae`) AND stepped metrics (`mae_by_horizon_min` at step=60) are present. Use `MlflowClient.get_metric_history(run_id, "mae_by_horizon_min")` and assert `[m.step for m in history] == [60]`, `[m.value for m in history][0] == expected`.
  3. `test_log_evaluation_results_multiple_horizons` — call 5 times with horizons `[1, 6, 12, 24, 48]` and corresponding `horizon_minutes=[10, 60, 120, 240, 480]`, assert the metric history for `mae_by_horizon_min` has exactly 5 points with the expected steps in ascending order.
- **PATTERN**: Use `tmp_path` fixture + `mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")` for isolated test state. Wrap calls in `with mlflow.start_run():`. Use `MlflowClient().get_metric_history()` for assertions.
- **IMPORTS**:
  ```python
  import mlflow
  import pytest
  from mlflow.tracking import MlflowClient
  from windcast.tracking import log_evaluation_results
  ```
- **GOTCHA**: MLflow runs persist across tests unless the tracking URI is scoped per-test. Use the tmp_path pattern. Also: `MlflowClient.get_metric_history` returns `list[Metric]` where each `Metric` has `.step` and `.value` attributes. The list is NOT guaranteed to be sorted — sort by `.step` in assertions.
- **VALIDATE**: `uv run pytest tests/test_tracking.py -v`

### Task 5 — UPDATE `docs/mlflow-ui-setup.md`

- **IMPLEMENT**: Insert a new section titled `## Native line charts: metric vs horizon` between the existing "Charts tab configuration" section and "Alternative: programmatic comparison". Content:
  - Explain that EnerCast logs `mae_by_horizon_min` (and siblings) with `step=minutes_ahead`.
  - Recipe: "Charts tab → New chart → Line chart → Metric: `mae_by_horizon_min` → Axes auto-detect step on X." One line per parent run.
  - Note that these metrics coexist with `h{n}_mae` (filterable) so no migration is needed on old runs.
  - Reference the `STEPPED_METRIC_MAP` keys (`mae_by_horizon_min`, `rmse_by_horizon_min`, `skill_score_by_horizon_min`, `bias_by_horizon_min`, `persistence_mae_by_horizon_min`) in the tag-reference table.
  - Add one sentence in "Troubleshooting" explaining that stepped metrics only populate for runs trained **after** the stepped-logging feature landed — older runs (pre-feature) will only have `h{n}_mae`.
- **PATTERN**: Mirror the existing section style (headings, fenced code blocks, prose).
- **VALIDATE**: `grep -c "mae_by_horizon_min" docs/mlflow-ui-setup.md` — expect ≥ 3.

### Task 6 — VALIDATE end-to-end by re-running one training pass

- **IMPLEMENT**: Run `uv run python scripts/train.py --feature-set wind_baseline --horizons 1 6 12` against the existing `data/features/kelmarsh_kwf1.parquet`. This creates a new parent+3-child nested run in the `enercast-kelmarsh` experiment.
- **MANUAL CHECK**: Launch `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`, open the new run, go to Charts tab, confirm that `mae_by_horizon_min` appears as a line chart with 3 points at X=10, 60, 120.
- **GOTCHA**: This re-writes a run to the main SQLite store and adds a test run. If you want a clean state, use `--experiment-name enercast-test-stepped` instead.
- **VALIDATE**: Programmatic check — `uv run python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///mlflow.db'); client=mlflow.tracking.MlflowClient(); exp=client.get_experiment_by_name('enercast-test-stepped'); runs=client.search_runs(experiment_ids=[exp.experiment_id], filter_string=\"tags.\`enercast.run_type\`='child'\"); h=client.get_metric_history(runs[0].info.run_id, 'mae_by_horizon_min'); print(sorted([(m.step,m.value) for m in h]))"`. Expect a list like `[(10, 120.xx)]` for the first child (h=1).

### Task 7 — UPDATE `.claude/STATUS.md`

- **IMPLEMENT**: Add a short entry under `### Planned Improvements (post-presentation)` (around line 124) noting the feature is done, with a one-line description and a reference to the new line-chart recipe in `docs/mlflow-ui-setup.md`.
- **VALIDATE**: Visual inspection; `grep "mae_by_horizon_min" .claude/STATUS.md`.

### Task 8 — COMMIT

- **IMPLEMENT**: Stage the 5 modified files (`mlflow_utils.py`, `train.py`, `train_autogluon.py`, `mlflow-ui-setup.md`, `STATUS.md`) + the new `tests/test_tracking.py`, commit with message:
  ```
  feat(tracking): stepped horizon metrics for native MLflow line charts

  Add mae_by_horizon_min / rmse_by_horizon_min / skill_score_by_horizon_min /
  bias_by_horizon_min / persistence_mae_by_horizon_min, logged with
  step=minutes_ahead. Unlocks MLflow's native line chart (X=horizon_min,
  Y=metric, one line per run) without custom Charts-tab configuration or
  compare_runs.py post-processing.

  Flat h{n}_mae metrics are preserved — no breaking change for filtering
  or for the existing compare_runs.py script. Cross-domain: minutes_ahead
  is the consistent step unit across wind (10 min), demand (60 min), solar
  (15 min), and the planned price domain.
  ```
- **VALIDATE**: `git log -1 --stat` shows 6 files.

---

## TESTING STRATEGY

### Unit Tests

See Task 4. Three test cases cover:
1. **Backward compatibility** — legacy `horizon=h` only must still log `h{n}_` metrics without introducing stepped metrics.
2. **Stepped logging** — both flat and stepped metrics present when `horizon_minutes` is passed.
3. **Multi-step history** — calling the logger 5 times with different horizons produces a single metric with 5 steps in history.

### Integration Test

Task 6 — run a real mini training pass and verify the line chart manifests in MLflow UI.

### Edge Cases

- Metric key not in `STEPPED_METRIC_MAP` (e.g. a custom metric added by a downstream user) → silently skipped from stepped logging, still present in flat logging. **Test**: add a spurious `"custom_metric": 42.0` to the input dict and assert the flat `h6_custom_metric` exists but `custom_metric_by_horizon_min` does not.
- `horizon_minutes=0` → MLflow accepts `step=0`, it's a valid value. Test once with `horizon_minutes=0` to confirm no crash.
- `horizon=None, horizon_minutes=60` → stepped logging should fire but flat logging should NOT prefix (falls into the `else` branch at line 59 of `mlflow_utils.py`). **Test**: assert `mae` (un-prefixed) and `mae_by_horizon_min` (stepped) both exist, but `h6_mae` does not.

---

## VALIDATION COMMANDS

Execute every command to ensure zero regressions and 100% feature correctness.

### Level 1: Syntax & Style

```bash
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run pyright src/
```

**Expected**: All commands exit 0. Pre-existing pyright noise in `scripts/` (not type-checked in CI) is out of scope.

### Level 2: Unit Tests

```bash
uv run pytest tests/test_tracking.py -v
uv run pytest tests/ -q
```

**Expected**: New test file has 4-5 passing tests. Full suite: 275+ passed (was 271).

### Level 3: Integration (real training run)

```bash
uv run python scripts/train.py \
    --experiment-name enercast-test-stepped \
    --feature-set wind_baseline \
    --horizons 1 6 12
```

**Expected**: Training completes, logs to `enercast-test-stepped` experiment. No warnings about missing `horizon_minutes`.

### Level 4: Manual UI validation

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the `enercast-test-stepped` experiment → Charts tab → Add chart → Line chart → Metric: `mae_by_horizon_min`. **Expected**: a line chart with 3 points at X=10, 60, 120.

### Level 5: Programmatic assertion

```bash
uv run python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name('enercast-test-stepped')
runs = client.search_runs(experiment_ids=[exp.experiment_id])
parent = [r for r in runs if r.data.tags.get('enercast.run_type') == 'parent'][0]
children = [r for r in runs if r.data.tags.get('mlflow.parentRunId') == parent.info.run_id]
for c in children:
    h = client.get_metric_history(c.info.run_id, 'mae_by_horizon_min')
    print(f'{c.data.tags.get(\"mlflow.runName\")}: {sorted((m.step, round(m.value,2)) for m in h)}')
"
```

**Expected**: one line per child run, each with exactly one `(step, value)` tuple (e.g. `h01: [(10, 120.18)]`).

---

## ACCEPTANCE CRITERIA

- [ ] `log_evaluation_results` accepts optional `horizon_minutes` parameter (signature change is backward compatible — `horizon_minutes=None` matches pre-feature behaviour exactly)
- [ ] When `horizon_minutes` is passed, 5 stepped metrics are logged: `mae_by_horizon_min`, `rmse_by_horizon_min`, `bias_by_horizon_min`, `skill_score_by_horizon_min`, `persistence_mae_by_horizon_min` (the latter only when present in the input dict)
- [ ] All existing `h{n}_*` flat metrics still logged — no removal, no renaming
- [ ] `tests/test_tracking.py` exists with at least 3 passing tests
- [ ] Full test suite passes (≥275 tests)
- [ ] `scripts/train.py` and `scripts/train_autogluon.py` pass `horizon_minutes=h * data_resolution` at their respective call sites
- [ ] A real training run shows a native MLflow line chart in the UI with X=minutes_ahead, Y=MAE
- [ ] `docs/mlflow-ui-setup.md` has a "Native line charts" section with the recipe
- [ ] `.claude/STATUS.md` updated with the completion note
- [ ] `compare_runs.py` still works unchanged (verify by running it against the experiment — should produce the same PNGs as before)
- [ ] No breaking change for runs logged before this feature — they still appear in the UI with their `h{n}_*` metrics intact
- [ ] Ruff + pyright clean on `src/`

---

## COMPLETION CHECKLIST

- [ ] Task 1 — `mlflow_utils.py` extended
- [ ] Task 2 — `train.py` call site updated
- [ ] Task 3 — `train_autogluon.py` call site updated
- [ ] Task 4 — `tests/test_tracking.py` created, 3+ tests passing
- [ ] Task 5 — `docs/mlflow-ui-setup.md` updated
- [ ] Task 6 — end-to-end training run validates UI
- [ ] Task 7 — STATUS.md updated
- [ ] Task 8 — committed
- [ ] Level 1 validation passes (ruff + pyright)
- [ ] Level 2 validation passes (unit tests)
- [ ] Level 3 validation passes (real training run)
- [ ] Level 4 validation passes (manual UI check)
- [ ] Level 5 validation passes (programmatic metric history assert)
- [ ] `compare_runs.py` still functional (regression sanity check)
- [ ] MLflow UI line chart visually correct: X axis shows minutes_ahead, Y shows MAE, one coloured line per run

---

## NOTES

**Design decision — why `minutes_ahead` as step unit** (chosen over alternatives):

- ❌ **Raw horizon steps (1, 6, 12, 24, 48)** — inconsistent across domains (step=48 means 8h for wind but 48h for demand). Would make cross-domain line charts misleading.
- ❌ **Hours ahead (0.17, 1, 2, 4, 8)** — MLflow `step` must be an integer. Floats silently cast to 0 for sub-hour horizons.
- ❌ **Samples ahead (identical to raw steps)** — same issue as raw steps.
- ✅ **Minutes ahead (10, 60, 120, 240, 480)** — integer, consistent across all 4 current and planned domains, linearly comparable, natural unit for operational decisions (market gate closures are typically minute-resolved).

**Why keep `h{n}_*` flat metrics**: MLflow `search_runs` filter strings only see the LAST logged value of a metric (MLflow does not support filtering on metric history). The flat `h{n}_mae` names preserve the ability to filter runs with `metrics.h48_mae < 300` — essential for the existing `scripts/compare_runs.py` and any future "find the best run at horizon X" query. Stepped metrics are for **visualization**; flat metrics are for **queries**. Both are needed.

**Future extension — price domain**: When EPEX price forecasting is added, the horizons will be non-uniform (intraday 60 min, then skip to D+1 720-2160 min range representing 24 hourly targets, then D+2-D+7). The stepped metric handles this naturally — the line chart will show a scatter of points at irregular X values, which is the *correct* visualization of a day-ahead-plus-week-ahead forecast portfolio. No framework change needed at that time.

**Confidence score**: 9/10 — this is a pure additive change with a small surface area (one function + two call sites + doc). The one risk is test fixture flakiness with MLflow file store (tmp_path handling); covered by using `sqlite:///{tmp_path}/mlflow.db` which is more reliable than `file://`.

**Out of scope (deliberately)**:
- Changing `compare_runs.py` to consume stepped metrics instead of flat. The flat path is simpler and already works; revisit only if `compare_runs.py` becomes a maintenance burden.
- Adding stepped logging for `mlforecast` script (`scripts/train_mlforecast.py`). It uses a different horizon semantics (direct multi-step forecasting, not per-horizon model). Can be added in a follow-up if mlforecast becomes a presentation topic.
- Cross-domain line charts in MLflow UI. Technically possible (same metric name across experiments) but the Y-axis units differ (kW vs MW vs EUR/MWh), making the chart misleading. A normalized skill score chart is the correct cross-domain view — already natively supported by the `skill_score_by_horizon_min` stepped metric.
