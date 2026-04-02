# EnerCast - Current Status

**Last Updated**: 2026-04-02
**Context**: WeatherNews challenge — demo presentation next week (English, Craig present)
**Current Phase**: Phase 2 — Demand Domain DONE
**Priority**: Wind E2E → Demand domain → Solar (stretch) → Presentation

---

## Current Focus

**Deadline**: Presentation next week (date TBC), in English, with Craig West remote

### Priority Order

1. **Phase 1 — Wind End-to-End** - DONE
   - Data ingestion DONE (schema, parser, QC, Open-Meteo)
   - Feature engineering DONE (registry, wind features, build_features CLI)
   - Train + evaluate + MLflow DONE (XGBoost, persistence baseline, evaluation, CLI scripts)
2. **Phase 2 — Demand Domain** - DONE
   - Demand schema (11-col canonical) DONE
   - Spain parser (energy + weather CSVs) DONE
   - Demand QC (load/weather outliers, holidays, DST, gap fill) DONE
   - Demand features (3 sets: baseline/enriched/full) DONE
   - Script integration (--domain flag on build_features/train/evaluate) DONE
   - 148 tests passing, ruff/pyright/pytest all green
3. **Phase 3 — Solar Domain** - STRETCH
4. **Phase 4 — Presentation** - TODO

---

## What's DONE

### Phase 1.1: Project Setup (DONE)
- [x] `pyproject.toml` with all deps + tool configs
- [x] `src/windcast/` package structure
- [x] `.gitignore` for Python/data/MLflow
- [x] `uv sync` — 109 packages installed
- [x] Validation: ruff, pyright, pytest all pass

### Phase 1.2: Data Ingestion (DONE)
- [x] `config.py` — Pydantic Settings, 3 dataset configs, QC thresholds
- [x] `schema.py` — 15-column canonical SCADA schema + validation
- [x] `kelmarsh.py` — Nested ZIP parser, signal mapping, pitch averaging
- [x] `qc.py` — 9 QC rules (maintenance, outliers, curtailment, frozen, gap-fill)
- [x] `open_meteo.py` — Cached Open-Meteo historical weather client
- [x] `ingest_kelmarsh.py` — CLI script: parse → QC → per-turbine Parquet
- [x] 50 tests passing, ruff, pyright, pytest all green

---

## What's NEXT

### Phase 1.3: Wind Feature Engineering + ML (DONE)
- [x] `features/registry.py` — Feature set registry (baseline/enriched/full)
- [x] `features/wind.py` — Wind-specific feature builders
- [x] `scripts/build_features.py` — Feature building CLI
- [x] `models/persistence.py` — Naive persistence benchmark
- [x] `models/xgboost_model.py` — XGBoost with MLflow logging
- [x] `models/evaluation.py` — Metrics, skill scores, regime analysis
- [x] `tracking/mlflow_utils.py` — MLflow tracking utilities
- [x] `scripts/train.py` — Training CLI with MLflow
- [x] `scripts/evaluate.py` — Evaluation CLI
- [x] 39 new tests (89 total), ruff, pyright, pytest all green

### Phase 2: Demand Domain (Day 2-3)
- [ ] Demand schema (timestamp, load_mw, temperature, price, calendar)
- [ ] Spain demand parser (Kaggle CSV → canonical schema)
- [ ] Demand QC rules
- [ ] Demand feature sets (lags H-1/D-1/W-1, calendar, temperature)
- [ ] Demand dataset config
- [ ] `scripts/ingest_spain_demand.py`
- [ ] Same train.py + evaluate.py work with `--domain demand`
- [ ] Tests

### Phase 3: Solar Domain (Day 3-4, stretch)
- [ ] Solar schema
- [ ] PVDAQ parser
- [ ] Solar feature sets
- [ ] Same pipeline, 3 domains in MLflow

### Phase 4: Presentation (Day 5)
- [ ] Slide deck (English): "I understood / Here's proof / Here's the roadmap"
- [ ] Live demo script
- [ ] WN roadmap (3 horizons)
- [ ] Dry run

---

## Quick Commands

```bash
# Setup
uv sync

# Validation (run before every commit)
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run pyright src/
uv run pytest tests/ -v

# Pipeline (once built)
uv run python scripts/ingest_kelmarsh.py      # Wind: parse → QC → Parquet
uv run python scripts/build_features.py        # Features → Parquet
uv run python scripts/train.py                 # Train → MLflow
uv run python scripts/evaluate.py              # Evaluate → MLflow
mlflow ui                                      # View results
```

---

## Key Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-01 | Pivot from WindCast → EnerCast (multi-domain) | WN challenge requires proving cross-domain standardization |
| 2026-04-01 | Wind + Demand priority over Wind + Solar | Demand is WN's biggest pain point (MARS legacy). Structurally different from wind = stronger proof of genericity |
| 2026-04-01 | Spain ENTSO-E dataset for demand | CC0, demand + weather + prices in one CSV, 4 MB, zero friction |
| 2026-04-01 | Solar (PVDAQ) as stretch goal | Nice to have but Wind + Demand alone proves the point |
| 2026-04-01 | Defer Hill of Towie / Penmanshiel | One wind dataset is enough for the demo. Cross-OEM testing is secondary |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.12 |
| Package Manager | uv |
| Data | Polars 1.39 |
| ML | XGBoost 3.2, LightGBM 4.6, scikit-learn 1.8 |
| Tracking | MLflow 3.10 |
| Tuning | Optuna 4.8 |

---

**Next Action**: Build Phase 1.3 — Wind features + train + evaluate + MLflow
