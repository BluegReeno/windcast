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
| 2 | **Build features + train baseline** — run `build_features.py` then `train.py --feature-set wind_baseline` | `data/processed/` | `data/features/` + MLflow run | [ ] |
| 3 | **Train enriched + evaluate** — run `train.py --feature-set wind_enriched` then `evaluate.py` on both | MLflow runs | Skill scores, MAE per horizon, regime analysis | [ ] |
| 4 | **mlforecast comparison** — run `train_mlforecast.py` on wind, compare with XGBoost in MLflow | `data/processed/` | XGBoost vs mlforecast metrics side by side | [ ] |

**Known risk:** `ingest_kelmarsh.py` expects `data/raw/kelmarsh/*.zip` but data is at `data/KelmarshV4/16807551.zip`. Pass `--raw-path` or fix the path. This will likely eat some time in pass 1.

**Wednesday exit criteria:**
- [x] `data/processed/` has Kelmarsh Parquets (6 turbines, 473k rows each)
- [ ] `data/features/` has feature Parquets
- [ ] MLflow has at least 2 wind runs (baseline + enriched) with real metrics
- [ ] Skill score > 0 at h1 (model beats persistence)
- [ ] Can articulate "roadmap to improve" based on actual results

### Thursday AM (2h = 4 passes) — Demand Domain

**Goal:** Prove cross-domain by running demand pipeline. Zero core changes.

| Pass | Task | Input | Output | Done |
|------|------|-------|--------|------|
| 5 | **Download Spain data + ingest** — Kaggle download (4 MB CSV), run `ingest_spain_demand.py` | Kaggle CSV | `data/processed/spain_demand.parquet` | [ ] |
| 6 | **Build features + train** — `build_features.py --domain demand` then `train.py --domain demand` | `data/processed/` | Demand metrics in MLflow | [ ] |
| 7 | **Evaluate + compare** — `evaluate.py --domain demand`, screenshot MLflow with wind + demand | MLflow runs | Cross-domain comparison | [ ] |

**Thursday AM exit criteria:**
- [ ] Demand pipeline ran with zero core pipeline changes
- [ ] MLflow shows wind AND demand experiments
- [ ] Can state "adding demand = parser + feature config, zero core changes"

### Thursday PM (1h = 2 passes) — Build PPT

**Goal:** Assemble presentation from real results.

| Pass | Task | Output | Done |
|------|------|--------|------|
| 8 | **Populate slides 4-5** — extract metrics tables, take MLflow screenshots, write talking points | Slides with real numbers | [ ] |
| 9 | **Complete slides 1-3, 6-7** — narrative, framework diagram, WattCast incidents, roadmap | Full 7-slide deck content | [ ] |
| 10 | **Review + polish** — flow, English, key messages per audience (Yoel/Michel/Craig) | Presentation-ready PPT content | [ ] |

**Thursday exit criteria:**
- [ ] 7-slide structure complete with real metrics
- [ ] Key talking points per slide written
- [ ] Ready to build final PPT on Friday

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

### Research & Planning
- [x] WN challenge CR + detailed slide analysis
- [x] WattCast learnings document (4 real incidents)
- [x] ML backends comparison (XGBoost vs mlforecast)
- [x] Presentation plan (7 slides, narrative arc)

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
