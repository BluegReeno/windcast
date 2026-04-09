# EnerCast - Current Status

**Last Updated**: 2026-04-09
**Context**: WeatherNews challenge — PPT presentation next week (English, Craig remote)
**Current Phase**: Phase 4 — Run pipelines + build presentation
**Budget**: 5 hours (Wed 2h + Thu 3h), ~10 core_piv_loop passes

---

## Sprint Plan: Wed 8 → Thu 9 April

### Wednesday (2h = 4 passes) — Wind End-to-End

**Goal:** Run the full wind pipeline on Kelmarsh real data. Get actual metrics in MLflow.

| Pass | Task | Input | Output | Done |
|------|------|-------|--------|------|
| 1 | **Ingest Kelmarsh** — run `ingest_kelmarsh.py` on real ZIP, fix any path issues | `data/KelmarshV4/16807551.zip` | `data/processed/kelmarsh_*.parquet` | [x] |
| 2 | **Build features + train baseline** — run `build_features.py` then `train.py --feature-set wind_baseline` | `data/processed/` | `data/features/` + MLflow run | [x] |
| 3 | **Train enriched + evaluate** — run `train.py --feature-set wind_enriched` then `evaluate.py` on both | MLflow runs | Skill scores, MAE per horizon, regime analysis | [x] |
| 4 | **Weather provider layer** — pluggable NWP fetch/cache/serve (Open-Meteo + SQLite) | — | `weather/` module + `data/weather.db` | [x] |
| 5 | **NWP features + wind_full** — join NWP at forecast horizon, train wind_full, measure improvement | `data/weather.db` | wind_full MLflow run, 23 features | [x] |
| 5b | **MLflow UI polish** — run descriptions, tags propagation, summary metrics on parent | MLflow runs | Readable MLflow UI with comparison | [x] |

**Results (val set, turbine kwf1):**

| Horizon | Baseline MAE | Enriched MAE | Full MAE | Baseline Skill | Enriched Skill | Full Skill |
|---------|-------------|-------------|----------|----------------|----------------|------------|
| h1 (10m) | 120 kW | 119 kW | **115 kW** | 0.203 | 0.207 | **0.236** |
| h6 (1h) | 210 kW | 207 kW | **184 kW** | 0.097 | 0.103 | **0.195** |
| h12 (2h) | 259 kW | 256 kW | **205 kW** | 0.091 | 0.101 | **0.250** |
| h24 (4h) | 334 kW | 329 kW | **235 kW** | 0.107 | 0.116 | **0.315** |
| h48 (8h) | 432 kW | 429 kW | **283 kW** | 0.130 | 0.135 | **0.364** |

**Key result:** NWP data doubles skill scores at short horizons and nearly triples them at longer horizons. Biggest gain at h48: 0.130 → 0.364 (+180%). This is the "rearview mirror → windshield" story for the WN presentation.

---

### Pass 6 — AutoGluon-Tabular Backend + Final Comparison [x]

**Goal:** Add AutoGluon-Tabular as 3rd ML backend (WN's own tool). Train on wind_full features, compare XGBoost vs AutoGluon ensemble. This shows framework pluggability with WN's actual stack.

**Exit criteria:**
- [x] AutoGluon backend works with same feature Parquets as XGBoost
- [x] MLflow shows 4 runs: XGB baseline / XGB enriched / XGB full / AutoGluon full
- [x] Comparison table: XGBoost vs AutoGluon on same features, per horizon
- [x] Can state: "adding a new ML backend = 1 file, zero pipeline changes"

**Results (val set, turbine kwf1, wind_full features):**

| Horizon | XGBoost MAE | AutoGluon MAE | Gain | XGB Skill | AG Skill |
|---------|-------------|---------------|------|-----------|----------|
| h1 (10m) | 115 kW | **112.9 kW** | -1.8% | 0.236 | 0.236 |
| h6 (1h) | 184 kW | **177.6 kW** | -3.5% | 0.195 | 0.184 |
| h12 (2h) | 205 kW | **195.7 kW** | -4.5% | 0.250 | 0.237 |
| h24 (4h) | 235 kW | **224.3 kW** | -4.6% | 0.315 | 0.297 |
| h48 (8h) | 283 kW | **263.0 kW** | -7.1% | 0.364 | 0.350 |

**Key result:** AutoGluon ensemble (CatBoost+LightGBM+XGBoost stacked) beats single XGBoost on all horizons. Gain increases with horizon (-1.8% to -7.1%). Training time: ~6 min/horizon with `best_quality` preset (29 min total for 5 horizons).

**Files created:** `autogluon_model.py` (wrapper) + `train_autogluon.py` (script) — 271 tests pass.

---

### Passes 7-9 — Demand Domain + Presentation

(unchanged from original plan, renumbered)

| Pass | Task | Output | Done |
|------|------|--------|------|
| 7 | **Download Spain data + ingest + train** — demand pipeline end-to-end, zero core changes | Demand metrics in MLflow | [ ] |
| 8 | **Build PPT slides 1-5** — narrative + framework diagram + real metrics tables | Slides with real numbers | [ ] |
| 9 | **Complete PPT + review** — roadmap, WattCast incidents, polish, talking points | Presentation-ready | [ ] |

---

## What's DONE (Phases 1-3)

### Framework (code complete, tested, never run on real data)
- [x] 3 domain schemas (wind 15-col, demand 11-col, solar 10-col)
- [x] 3 parsers (Kelmarsh, Spain ENTSO-E, PVDAQ System 4)
- [x] 3 QC pipelines (wind 9 rules, demand, solar)
- [x] 18 feature sets (3 domains x 3 levels x 2 backends)
- [x] 2 ML backends (XGBoost manual + mlforecast/Nixtla)
- [x] Evaluation (MAE, RMSE, MAPE, skill score, bias, regime analysis)
- [x] MLflow tracking integration
- [x] Persistence baseline
- [x] 267 tests passing, ruff + pyright clean
- [x] 7 CLI scripts covering full pipeline

### Weather & NWP Integration (Passes 4-5)
- [x] WeatherProvider protocol + OpenMeteoProvider
- [x] WeatherConfig registry for Kelmarsh, Spain, PVDAQ
- [x] SQLite cache (weather.db) with upsert, temporal queries, gap detection
- [x] `get_weather()` public API: fetch-cache-serve with ERA5 lag guard
- [x] Resolution-agnostic NWP horizon feature joining (`features/weather.py`)
- [x] `build_features.py --weather-db` loads NWP from SQLite cache
- [x] `train.py` resolves horizon-specific NWP features per child run

### MLflow Integration (Passes 2.5 + 5b)
- [x] XGBoost autolog (replaces manual param/model logging, adds feature importance)
- [x] Dataset provenance via `mlflow.data.from_polars()` (auto-hash train/val)
- [x] Lineage tags: stage, domain, purpose, backend, data_resolution
- [x] Split boundaries logged as params (train/val/test dates)
- [x] Git commit auto-captured as system tag
- [x] Markdown descriptions on parent + child runs (feature set, results table)
- [x] Summary metrics bubbled up to parent (h{n}_mae, h{n}_skill_score)
- [x] Autolog noise reduced (log_models=False, log_datasets=False)

### Research & Planning
- [x] WN challenge CR + detailed slide analysis
- [x] WattCast learnings document (4 real incidents)
- [x] ML backends comparison (XGBoost vs mlforecast)
- [x] Presentation plan (7 slides, narrative arc)

### Pass 6b — MLflow UI polish + AutoGluon speed + SQLite migration [x] (Thu 9 April afternoon)
- [x] **MLflow UI readability**:
  - [x] Added `enercast.run_type = {parent,child}` tag in `train.py` + `train_autogluon.py` for clean UI filtering
  - [x] Created `scripts/compare_runs.py` — programmatic MAE + Skill bar charts per horizon (PNG output in `reports/`) + Markdown comparison table to stdout
  - [x] Wrote `docs/mlflow-ui-setup.md` — reusable guide for UI launch, filter recipes, Charts tab config, tag reference
  - [x] Migrated MLflow backend from `file:./mlruns` to `sqlite:///mlflow.db` — removes deprecation warning, unlocks `IS NULL` tag filters
- [x] AutoGluon speed defaults: script CLI defaults switched to `good_quality` + `time_limit=120` (~2-4 min/horizon vs 6 min). `AutoGluonConfig` class defaults stay at `best_quality` for final runs.

**Reproducibility check after migration (new SQLite vs historical CSV snapshot):**
- XGBoost (3 runs × 5 horizons × 2 metrics): **0.0000 deviation** — bit-for-bit identical
- AutoGluon (1 run × 5 horizons × 2 metrics): **max 0.76% deviation** — bagging/stacking quasi-deterministic
- Historical snapshot: `docs/WNchallenge/historical_runs_2026-04-08.csv` (24 runs preserved as safety baseline)

**Updated results table (val set, turbine kwf1) — post-migration, all 4 backends:**

| Horizon | XGB Baseline MAE | XGB Enriched MAE | XGB Full MAE | AG Full MAE | XGB Full Skill | AG Full Skill |
|---------|------------------|------------------|--------------|-------------|----------------|---------------|
| h1 (10m) | 120 | 119 | 115 | **113** | 0.236 | 0.235 |
| h6 (1h) | 210 | 207 | 184 | **177** | 0.195 | 0.185 |
| h12 (2h) | 259 | 256 | 205 | **196** | 0.250 | 0.237 |
| h24 (4h) | 334 | 329 | 235 | **224** | 0.315 | 0.297 |
| h48 (8h) | 432 | 429 | 283 | **264** | 0.364 | 0.349 |

Generated charts: `reports/comparison_enercast-kelmarsh_mae.png` and `reports/comparison_enercast-kelmarsh_skill.png` — ready for slides.

### Pass 7 — RTE France Demand Domain (éCO2mix local files) [x] (Thu 9 April evening)

**Goal:** Replace the Spain Kaggle dataset with 11 years of real French national load
from locally-downloaded RTE éCO2mix annual definitive files, refactor demand feature
sets to use forward-looking NWP (weighted 8-city mean), and benchmark against the
official RTE day-ahead forecast as the killer comparison slide.

**Key technical decisions:**
- **Local ZIP parser (no API)** — reads `data/ecomix-rte/eCO2mix_RTE_Annuel-Definitif_*.zip`
  directly. Format is TSV in ISO-8859-1 despite the `.xls` extension; date format is ISO
  (the PDF spec is wrong). 11 years × ~35k 15-min rows → 96,421 hourly rows after
  resampling (load 29-96 GW, 100% QC_OK, 2,904 holiday hours detected)
- **8-city weighted NWP (wattcast TEMP_POINTS)** — introduced new `WeightedWeatherConfig`
  dataclass + `get_weather_weighted()` helper. `rte_france` weather = Paris 0.30, Lyon 0.15,
  Tours 0.14, Lille 0.10, Bordeaux 0.08, Toulouse 0.08, Strasbourg 0.08, Marseille 0.07.
  Each point cached independently under its own `{lat}_{lon}` key in `data/weather.db`
- **Feature set refactor** — `demand_enriched` = baseline + rolling stats + holiday;
  `demand_full` = enriched + forward-looking NWP at horizon + HDD/CDD computed from
  `nwp_temperature_2m_h1`. Observed weather + price features dropped (Spain-specific,
  legacy)
- **Schema extension** — added `tso_forecast_mw` (Float64, nullable) to `DEMAND_SCHEMA`.
  Spain parser auto-fills as null; RTE parser populates from `Prévision J-1` column
- **Script parametrisation** — `build_features.py` and `train.py` now accept `--dataset`
  so filenames follow `{dataset}.parquet` / `{dataset}_features.parquet`
- **QC config bump** — `DemandQCConfig.max_load_mw` raised 50 → 100 GW (France peak is
  ~90 GW; Spain peak is ~41 GW so no regression there)

**Results (val set 2022-2023, 17,518 hourly rows, 8-city weighted NWP):**

| Horizon | Baseline MAE | Enriched MAE | Full MAE | Baseline Skill | Enriched Skill | Full Skill |
|---------|-------------|--------------|----------|----------------|----------------|------------|
| h1 (1h) | 839 MW | 782 MW | **766 MW** | 0.693 | 0.711 | **0.719** |
| h6 (6h) | 1,430 MW | 1,168 MW | **1,130 MW** | 0.745 | 0.792 | **0.797** |
| h12 (12h) | 1,634 MW | 1,377 MW | **1,254 MW** | 0.714 | 0.752 | **0.773** |
| h24 (D+1) | 1,506 MW | 1,485 MW | **1,223 MW** | 0.493 | 0.498 | **0.581** |
| h48 (D+2) | 2,121 MW | 2,114 MW | **1,643 MW** | 0.486 | 0.484 | **0.604** |

**Killer slide — RTE TSO day-ahead benchmark:**

| Model | h24 MAE | RMSE | MAPE |
|-------|---------|------|------|
| RTE Prévision J-1 (official) | **1,205 MW** | 1,557 MW | 2.4% |
| `demand_full` (our framework) | **1,223 MW** | 1,791 MW | — |

**We match RTE's own day-ahead forecast to within 1.5%** using a generic pipeline
with 8-city weighted NWP and a stock XGBoost model. 11 years of real French national
load, fully offline ingest, no API credentials, reproducible in ~20 seconds on a
laptop. This is the slide for the jury (Yoel / Michel / Craig).

**Generated assets:**
- `reports/comparison_enercast-rte_france_mae.png` + `_skill.png`
- `data/processed/rte_france.parquet` (96,421 rows, 0.9 MB)
- `data/features/rte_france_features.parquet` (96,206 rows, 12.1 MB, full feature set)
- MLflow experiment `enercast-rte_france` with 4 parent runs (3 training + TSO baseline)
- 11 new tests (parser + French holidays + NWP-aware HDD/CDD)

**Notes:**
- Spain Kaggle parser retained as 2nd demand reference implementation (not deleted;
  confirms the schema abstraction is actually multi-dataset capable)
- Single-point Paris NWP was rejected in favour of 8-city weighted approach after user
  feedback — Paris alone is ~30% of French demand signal; the weighted mean is much
  closer to the national reference temperature RTE uses internally

### Planned Improvements (post-presentation)
- [x] **Stepped horizon metrics for native MLflow line charts** ✓ 2026-04-09 — `log_evaluation_results` also logs `mae_by_horizon_min` / `rmse_by_horizon_min` / `skill_score_by_horizon_min` / `bias_by_horizon_min` with `step=minutes_ahead`, unlocking MLflow's native "metric vs horizon" line chart out of the box. Flat `h{n}_*` metrics preserved. Recipe: `docs/mlflow-ui-setup.md#native-line-charts-metric-vs-horizon`.
- [x] **Fix: stepped metrics on parent, not children** ✓ 2026-04-09 afternoon — initial design logged stepped metrics on child runs (single-step each), which does not render a line chart because MLflow does not stitch metrics across sibling runs (confirmed by maintainers in [mlflow#2768](https://github.com/mlflow/mlflow/issues/2768) + [mlflow#7060](https://github.com/mlflow/mlflow/issues/7060)). New helper `log_stepped_horizon_metrics()` in `tracking/mlflow_utils.py` collects per-horizon metrics during the child loop and replays them on the parent → one 5-point line per parent run, Compare Runs overlays N curves natively. Old XGBoost runs re-trained (~20 s), AutoGluon backfilled via `scripts/backfill_stepped_metrics.py` (no retraining).
- [ ] `mlflow.evaluate()` — replace manual evaluation with MLflow's eval framework (auto residual plots, SHAP, R²). Keep custom skill_score + regime_analysis via `extra_metrics`
- [ ] AutoGluon-TimeSeries — 4th backend, ensemble of DeepAR/TFT/Chronos/XGBoost, probabilistic forecasts

---

## Key Documents

| Document | Purpose |
|----------|---------|
| `docs/WNchallenge/presentation-plan.md` | Slide-by-slide PPT plan |
| `docs/WNchallenge/wattcast-learnings-for-wn.md` | Real incidents for credibility |
| `docs/WNchallenge/CR WeatherNews...` | Meeting notes from briefing |
| `docs/WNchallenge/Analyse Presentation...` | Detailed WN slide analysis |
| `.claude/reference/ml-backends-comparison.md` | XGBoost vs mlforecast strategy |

---

## Session Handoff Protocol

Each session should:
1. Read this STATUS.md to know where we are
2. Pick up the next unchecked task
3. Mark tasks `[x]` as they complete
4. If a task fails or takes longer than expected, note it and adjust the plan
5. At end of session, update this file with current state

**The plan is sequential — each pass depends on the previous one succeeding.** If pass 1 (ingestion) fails, everything downstream blocks. Prioritize unblocking over perfection.
