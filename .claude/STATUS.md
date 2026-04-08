# EnerCast - Current Status

**Last Updated**: 2026-04-08
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

### Planned Improvements (post-presentation)
- [ ] `mlflow.evaluate()` — replace manual evaluation with MLflow's eval framework (auto residual plots, SHAP, R²). Keep custom skill_score + regime_analysis via `extra_metrics`
- [ ] AutoGluon-TimeSeries — 3rd backend, ensemble of DeepAR/TFT/Chronos/XGBoost, probabilistic forecasts
- [ ] Migrate MLflow backend from `file:./mlruns` to `sqlite:///mlflow.db`

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
