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

**Results so far (val set, turbine kwf1):**

| Horizon | Baseline MAE | Enriched MAE | Baseline Skill | Enriched Skill |
|---------|-------------|-------------|----------------|----------------|
| h1 (10m) | 120 kW | 119 kW | 0.203 | 0.207 |
| h6 (1h) | 210 kW | 207 kW | 0.097 | 0.103 |
| h12 (2h) | 259 kW | 256 kW | 0.091 | 0.101 |
| h24 (4h) | 334 kW | 329 kW | 0.107 | 0.116 |
| h48 (8h) | 432 kW | 429 kW | 0.130 | 0.135 |

**Diagnosis:** Skill scores are low (0.10-0.20) because the model has no future weather information — it only uses SCADA lags (rearview mirror). NWP data is the #1 lever.

---

### Pass 4 — Weather Provider Layer

**Goal:** Build a pluggable weather data provider so the framework can fetch, cache, and serve NWP data for any domain/site. This is the "Wx sources" layer that WN has (ECMWF, ICON, AROME) — EnerCast uses Open-Meteo as free equivalent.

**What to build:**

| # | Deliverable | Detail |
|---|------------|--------|
| 4a | `src/windcast/weather/providers/base.py` | `WeatherProvider` protocol: `fetch(config, start, end) → pl.DataFrame` |
| 4b | `src/windcast/weather/providers/open_meteo.py` | Open-Meteo provider: archive + forecast endpoints. Reuse patterns from WattCast |
| 4c | `src/windcast/weather/registry.py` | `WeatherConfig` per dataset (variables, locations, weights). Declarative like feature sets |
| 4d | `src/windcast/weather/storage.py` | SQLite cache: fetch once, query many. Upsert, temporal queries. File: `data/weather.db` |
| 4e | `src/windcast/weather/__init__.py` | Public API: `get_weather(config_name, start, end) → pl.DataFrame` |
| 4f | Tests | Unit tests for provider, storage, registry |

**Exit criteria:**
- [ ] `WeatherProvider` protocol defined with Open-Meteo implementation
- [ ] `WeatherConfig` for Kelmarsh registered (wind_speed_100m, wind_direction_100m, temperature_2m)
- [ ] SQLite storage works: fetch → store → query returns same data
- [ ] `get_weather("kelmarsh", "2016-01-01", "2024-12-31")` returns hourly Polars DF
- [ ] Tests pass, ruff + pyright clean

**Key decision:** SQLite for local caching (one file, upsert, temporal queries). Same schema as Supabase PostgreSQL — migration to prod is trivial.

---

### Pass 5 — NWP Features for Kelmarsh + Measure Improvement

**Goal:** Join weather data with SCADA, train with `wind_full` feature set, measure the improvement vs baseline/enriched. This is the key result: "adding NWP data improves skill score from 0.20 to 0.XX".

**What to do:**

| # | Step | Detail |
|---|------|--------|
| 5a | **Fetch weather** | `get_weather("kelmarsh", "2016-01-01", "2024-12-31")` → cached in `data/weather.db` |
| 5b | **Update build_features.py** | Add `--weather` flag: load weather, resample hourly→10min (forward-fill), join on timestamp |
| 5c | **Build wind_full features** | `build_features.py --feature-set wind_full --weather` → features with NWP columns |
| 5d | **Train wind_full** | `train.py --feature-set wind_full` → MLflow run with 19 features (enriched + NWP + calendar) |
| 5e | **Compare in MLflow** | 3 runs side by side: baseline (11) vs enriched (16) vs full (19). Extract improvement table |

**Exit criteria:**
- [ ] Weather data fetched and cached in SQLite for Kelmarsh (2016-2024, hourly)
- [ ] Feature Parquets contain NWP columns (nwp_wind_speed_100m, nwp_wind_direction_100m, nwp_temperature_2m)
- [ ] wind_full trained, results in MLflow
- [ ] Improvement table: baseline → enriched → full, skill scores per horizon
- [ ] Skill score improvement measurable (expected: 0.20 → 0.30+ at h1, bigger gains at h12-h48)

**What this proves for WN:** The framework handles the Wx sources integration — same pattern WN uses with ECMWF/ICON/AROME. Adding weather for a new site = add a `WeatherConfig` (3 lines), run the pipeline.

---

### Pass 6 — AutoGluon-Tabular Backend + Final Comparison

**Goal:** Add AutoGluon-Tabular as 3rd ML backend (WN's own tool). Train on wind_full features, compare XGBoost vs AutoGluon ensemble. This shows framework pluggability with WN's actual stack.

**What to build:**

| # | Deliverable | Detail |
|---|------------|--------|
| 6a | `src/windcast/models/autogluon_model.py` | AutoGluon-Tabular trainer: `TabularPredictor.fit(train_df)` with time_limit, presets |
| 6b | `scripts/train_autogluon.py` | Training script: same interface as train.py, MLflow logging |
| 6c | `pyproject.toml` | Add `autogluon.tabular` dependency |
| 6d | **Train + compare** | Run on wind_full features, compare with XGBoost in MLflow |

**Exit criteria:**
- [ ] AutoGluon backend works with same feature Parquets as XGBoost
- [ ] MLflow shows 4 runs: XGB baseline / XGB enriched / XGB full / AutoGluon full
- [ ] Comparison table: XGBoost vs AutoGluon on same features, per horizon
- [ ] Can state: "adding a new ML backend = 1 file, zero pipeline changes"

**What this proves for WN:** The framework is backend-agnostic. WN can plug in their own AutoGluon setup, or any sklearn-compatible model, without touching the pipeline.

**Reference:** AutoGluon-Tabular docs: https://auto.gluon.ai/dev/tutorials/tabular/index.html

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
- [x] 234 tests passing, ruff + pyright clean
- [x] 7 CLI scripts covering full pipeline

### MLflow Integration (Pass 2.5)
- [x] XGBoost autolog (replaces manual param/model logging, adds feature importance)
- [x] Dataset provenance via `mlflow.data.from_polars()` (auto-hash train/val)
- [x] Lineage tags: stage, domain, purpose, backend, data_resolution
- [x] Split boundaries logged as params (train/val/test dates)
- [x] Git commit auto-captured as system tag

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
