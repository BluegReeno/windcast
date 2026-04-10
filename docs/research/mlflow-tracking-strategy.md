# MLflow Tracking Strategy — EnerCast Pass 8

**Date**: 2026-04-10
**Context**: After fixing the ERA5 leak + AG tuning leak, we have 3 generations
of runs in MLflow with the same names, impossible to distinguish. This document
defines the problem and proposes a solution using MLflow's native features.

**Goal**: Establish a reproducible MLflow workflow for continuous improvement —
track what changed, why, and how it impacted metrics.

---

## The Problem

### What happened

We ran the full EnerCast pipeline 3 times on RTE France + Kelmarsh:

| Gen | Date | What changed | Status |
|-----|------|-------------|--------|
| Gen 1 | 9 avril 20h | First run (Pass 7) | **LEAKED** — ERA5 perfect foresight + AG tuning on val |
| Gen 2 | 9 avril 23h | Fixed AG `tuning_data=val_pd` | **LEAKED** — ERA5 still present |
| Gen 3 | 10 avril 07h | Ported WattCast forecast provider, blend mode | **CLEAN** |

### Why it's a problem in MLflow

- All 3 generations have runs named `rte_france-demand_full` — indistinguishable
- `compare_runs.py` charts show 2-3 lines per feature set because it picks up all gens
- No metadata in the runs tells you *why* Gen 3 exists or *what* changed vs Gen 2
- No way to filter "show me only the clean runs" in the UI
- No way to mark Gen 1/2 as invalid without deleting them (we want the audit trail)

### The deeper question

This is a normal ML workflow: train → find a bug → fix → retrain → compare.
We need MLflow to support this loop natively, not fight against it. Specifically:

1. **Before training**: document WHY this run is being created
2. **After training**: compare against previous runs, identify improvements or regressions
3. **When a problem is found**: annotate the problematic runs, don't delete them
4. **Over time**: see a timeline of how the model improved across generations

---

## MLflow Features We Should Use

### 1. Tags — Machine-Queryable Labels (already partially used)

We already have `enercast.run_type`, `enercast.backend`, `enercast.feature_set`.
We need to add lineage/quality tags:

```python
# Proposed standard tags for every training run
{
    # Existing
    "enercast.run_type": "parent",          # parent | child
    "enercast.backend": "xgboost",          # xgboost | autogluon | tso
    "enercast.feature_set": "demand_full",

    # NEW — lineage
    "enercast.generation": "gen3",           # gen1 | gen2 | gen3 | ...
    "enercast.nwp_source": "forecast",       # forecast | era5 | blend
    "enercast.data_quality": "CLEAN",        # CLEAN | LEAKED | SUSPECT
    "enercast.change_reason": "fix_era5_leak", # free text — WHY this run exists

    # NEW — git state
    "git.commit": "a1b2c3d4",
    "git.dirty": "false",                   # critical: was there uncommitted code?
}
```

**UI filter**: `tags.enercast.data_quality = "CLEAN"` shows only valid runs.

### 2. `mlflow.note.content` — Human-Readable Notes (Markdown)

The special tag `mlflow.note.content` renders as Markdown in the UI's "Notes"
section. This is where you write the narrative.

```python
client.set_tag(run_id, "mlflow.note.content", """
## Gen 3 — ERA5 leak fixed

NWP features now use `historical-forecast-api.open-meteo.com` (archived
forecasts) for val/test periods (2022+). Train uses ERA5 (pre-2022).
Ported from WattCast production pattern.

**What changed vs Gen 2**: `src/windcast/data/open_meteo.py` — added
`fetch_historical_forecast_weather`. `scripts/build_features.py` — added
`--weather-source blend`.

**Previous leaks**: Gen 1 had both ERA5 + AG tuning leaks. Gen 2 had
ERA5 only. Both fixed in this generation.
""")
```

**Can be applied retroactively** to any existing run — useful for annotating
Gen 1/2 as leaked after the fact.

### 3. MLflow Datasets (`mlflow.data`) — Data Fingerprinting

`mlflow.log_input()` stores **metadata only** (name, digest/hash, schema, row
count, source path) — NOT the data itself. The digest is a content hash: same
data = same digest, different data = different digest.

```python
import mlflow

dataset = mlflow.data.from_polars(
    df=train_df,
    source="data/features/rte_france_features.parquet",
    name="rte-france-demand-train",
    targets="load_mw",
)

with mlflow.start_run():
    mlflow.log_input(dataset, context="training")
    mlflow.log_input(val_dataset, context="validation")
```

**Key benefit for our case**: Gen 2 and Gen 3 used different feature parquets
(ERA5 vs forecast NWP). The digest changes automatically → you can prove the
training data was different between generations.

```python
# Audit: did data change between gen2 and gen3?
gen2_digest = client.get_run(gen2_id).inputs.dataset_inputs[0].dataset.digest
gen3_digest = client.get_run(gen3_id).inputs.dataset_inputs[0].dataset.digest
print(f"Same data? {gen2_digest == gen3_digest}")  # False — confirms fix worked
```

**Appears in the UI** under a dedicated "Inputs" section on each run page.

### 4. Artifacts — What to Log

The "Artifacts" tab in the UI is currently empty (`log_models=False`). We should
log lightweight artifacts for reproducibility:

| Artifact | Size | Purpose |
|----------|------|---------|
| `feature_list.json` | <1 KB | Which features were used |
| `feature_importance.json` | <5 KB | XGBoost/AG importance scores |
| `eval_summary.json` | <1 KB | Metrics + split dates + row counts |
| `scatter_actual_vs_pred.png` | ~100 KB | Visual diagnostic per horizon |
| `training_config.json` | <1 KB | Full config snapshot |

**The model itself**: only log for runs tagged `production_ready=true`. XGBoost
models are ~5-50 MB each; with 5 horizons × N runs, disk adds up fast.

```python
# During training
mlflow.log_dict({"features": feature_names}, "feature_list.json")
mlflow.log_dict(dict(zip(feature_names, model.feature_importances_)), "feature_importance.json")
mlflow.log_figure(fig, "scatter_actual_vs_pred.png")

# Only for keeper runs
if production_ready:
    mlflow.xgboost.log_model(model, "model", registered_model_name="xgb-demand-france")
```

### 5. Model Registry — The "Models" Tab

The Models tab is a **versioned catalog of promoted models**, separate from
experiment runs. Think of it as: experiments are your lab notebook, the registry
is your shipping dock.

**How it works**:
- You register a model from a run: `mlflow.register_model(f"runs:/{run_id}/model", "xgb-demand-france")`
- Each registration creates a new **version** (v1, v2, v3...)
- You assign **aliases** to versions: `champion` (current best), `challenger` (candidate)
- Loading at inference: `mlflow.xgboost.load_model("models:/xgb-demand-france@champion")`

**Stages (Staging/Production/Archived) are deprecated** since MLflow 2.9.
Use **aliases** instead:

```python
client = MlflowClient()

# After Gen 3 training beats Gen 2
mv = mlflow.register_model(f"runs:/{gen3_run_id}/model", "xgb-demand-france")
client.set_registered_model_alias("xgb-demand-france", "champion", mv.version)

# Tag version with eval metrics
client.set_model_version_tag(mv.name, mv.version, "test_mae_h24", "1104")
client.set_model_version_tag(mv.name, mv.version, "generation", "gen3")
```

**For EnerCast right now**: we don't deploy models, so the registry is not
urgent. But it becomes essential when we want to answer "which model version
should the presentation use?" or "which model powers WattCast D+1?".

### 6. Tracing — The "Traces" Tab

MLflow Tracing is **primarily for LLM/GenAI observability** (tracking agent
tool calls, prompt chains, etc.). It uses OpenTelemetry spans.

**Can** be used for ML pipeline steps (`@mlflow.trace(span_type="TASK")`) but:
- UI is optimized for multi-turn conversations, not batch pipelines
- Adds DB write overhead per span
- The span types are all LLM-centric (AGENT, CHAIN, RETRIEVER...)

**Recommendation for EnerCast**: skip Tracing for now. Standard run logging +
parent/child runs already gives us pipeline step tracking. Tracing would be
relevant if we later build an LLM-powered forecast explanation system.

---

## Proposed Implementation Plan

### Step 1 — Annotate existing runs retroactively

Write a one-shot script `scripts/annotate_mlflow_runs.py` that:

1. Queries all parent runs from `enercast-rte_france` and `enercast-kelmarsh`
2. Classifies them by date into Gen 1/2/3
3. Sets tags: `enercast.generation`, `enercast.data_quality`, `enercast.nwp_source`
4. Adds `mlflow.note.content` with Markdown explanation
5. Renames Gen 1/2 runs with `[LEAKED]` suffix
6. Optionally soft-deletes Gen 1/2 (`client.delete_run`) — recoverable

### Step 2 — Enrich `train.py` and `train_autogluon.py`

Add to both training scripts:

- Standard lineage tags (generation, nwp_source, data_quality, change_reason)
- Git state capture (commit, branch, dirty flag)
- `mlflow.log_input()` for train/val/test datasets (Polars)
- Lightweight artifacts: feature_list.json, eval_summary.json
- `mlflow.note.content` with a template describing what this run is

### Step 3 — Update `compare_runs.py`

- Default filter: `tags.enercast.data_quality = "CLEAN"` — only show valid runs
- Add `--all-generations` flag to include leaked runs (for audit/comparison)
- Add generation labels to the chart legend

### Step 4 — (Optional) Model Registry for keeper models

- Register the best Gen 3 model per domain/horizon as v1
- Set `champion` alias
- Future runs that beat the champion → new version + alias promotion

### Step 5 — Document the workflow

Add `docs/mlflow-tracking-workflow.md` describing:
- Tag taxonomy (what each tag means, valid values)
- How to annotate a run post-hoc
- How to compare generations
- How to promote a model to champion

---

## What This Looks Like in the MLflow UI

### Before (current state)
```
enercast-rte_france
├── rte_france-demand_full          ← which one? Gen 1? Gen 3?
├── rte_france-demand_full          ← same name, different date
├── rte_france-demand_full          ← same name again
├── rte_france-demand_enriched
├── rte_france-demand_enriched
├── rte_france-demand_enriched
├── ...
└── (50+ runs, no way to filter)
```

### After (with tags + annotations)
```
enercast-rte_france
  [Filter: tags.enercast.data_quality = "CLEAN"]
├── rte_france-demand_full          gen3  CLEAN  forecast  2026-04-10
├── rte_france-demand_enriched      gen3  CLEAN  forecast  2026-04-10
├── rte_france-demand_baseline      gen3  CLEAN  forecast  2026-04-10
├── rte_france-autogluon-demand_full gen3 CLEAN  forecast  2026-04-10
└── rte_france-tso_baseline         gen3  CLEAN  —         2026-04-10

  [Remove filter to see audit trail]
├── rte_france-demand_full [LEAKED] gen2  LEAKED era5     2026-04-09
├── rte_france-demand_full [LEAKED] gen1  LEAKED era5     2026-04-09
└── ...
```

---

## Decision Points for Next Session

1. **Soft-delete Gen 1/2 or just tag them?** Tagging keeps them visible for
   audit; soft-delete hides them but is reversible.

2. **Log model artifacts now or wait?** Adds ~50 MB per domain×backend but
   enables Model Registry immediately.

3. **One generation tag or structured versioning?** Simple `gen1/gen2/gen3`
   works now; a formal `feature_engineering_version` + `nwp_source` combo
   is more future-proof.

4. **Automate or manual?** Should training scripts auto-detect the generation
   from git state, or do we set it manually per session?

---

## Files to Read Before Starting

| File | What it tells you |
|------|-------------------|
| `src/windcast/tracking/mlflow_utils.py` | Current `setup_mlflow()`, `log_evaluation_results()`, `log_stepped_horizon_metrics()` |
| `scripts/train.py` | XGBoost training loop — where tags and artifacts should be added |
| `scripts/train_autogluon.py` | AutoGluon training loop — same changes needed |
| `scripts/compare_runs.py` | Chart generation — needs `data_quality` filter |
| `docs/mlflow-ui-setup.md` | Existing UI guide — update with new filter recipes |
| `docs/WNchallenge/post-era5-fix-results.md` | The results from the ERA5 fix — Gen 3 numbers |

---

## References

- [MLflow Tracking API](https://mlflow.org/docs/latest/ml/tracking/tracking-api/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)
- [MLflow Dataset Tracking](https://mlflow.org/docs/latest/ml/dataset/)
- [MLflow Search Runs syntax](https://mlflow.org/docs/latest/ml/search/search-runs/)
- Stages deprecated → aliases: [mlflow/mlflow#10336](https://github.com/mlflow/mlflow/issues/10336)
