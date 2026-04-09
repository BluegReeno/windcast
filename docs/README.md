# `docs/` — Documentation Index

Non-code documentation for EnerCast / WindCast: research, runbooks, setup guides, and exploration notes. Source-of-truth project docs live in `.claude/PRD.md` and `.claude/STATUS.md`; this folder is for durable references and deep dives.

---

## How-to guides

| Document | Purpose |
|---|---|
| [`data-ingestion.md`](data-ingestion.md) | Running the ingestion scripts (Kelmarsh, Spain, PVDAQ) end-to-end |
| [`mlflow-ui-setup.md`](mlflow-ui-setup.md) | Launching MLflow UI with SQLite backend + filter/chart recipes |
| [`workflow-guide.md`](workflow-guide.md) | Day-to-day dev workflow (ruff, pyright, pytest, uv) |

## Exploration notes (non-code decisions and findings)

| Document | Purpose |
|---|---|
| [`rte-api-notes.md`](rte-api-notes.md) | **RTE Data API exploration (2026-04-09)** — OAuth2 test, endpoint availability, chunk limits, wholesale_market check, productionisation verdict. Kept as trace even though Pass 7 uses the local éCO2mix files instead. |

## Research

Deep research and dataset catalogs used to scope the framework and the WN demo.

| Document | Purpose |
|---|---|
| [`research/brainstorming-2026-03-31.md`](research/brainstorming-2026-03-31.md) | Original feasibility study for multi-domain scope |
| [`research/datasets-catalog-2026-03-31.md`](research/datasets-catalog-2026-03-31.md) | Full dataset evaluation across wind / solar / demand |
| [`research/methodology-scaling-pipeline.md`](research/methodology-scaling-pipeline.md) | Scaling strategy from 1 dataset to N |

## WeatherNews challenge

Material for the WN technical presentation (dated 2026-04, due week 14/04).

| Document | Purpose |
|---|---|
| [`WNchallenge/CR WeatherNews — Yoel Chetboun Michel Kolasinski — 01-04-2026.md`](WNchallenge/CR%20WeatherNews%20%E2%80%94%20Yoel%20Chetboun%20Michel%20Kolasinski%20%E2%80%94%2001-04-2026.md) | Meeting notes from the WN briefing |
| [`WNchallenge/Analyse Présentation WeatherNews — Challenge Technique — 01-04-2026.md`](WNchallenge/Analyse%20Pr%C3%A9sentation%20WeatherNews%20%E2%80%94%20Challenge%20Technique%20%E2%80%94%2001-04-2026.md) | Slide-by-slide analysis of WN's own presentation |
| [`WNchallenge/presentation-plan.md`](WNchallenge/presentation-plan.md) | Our response — slide plan and narrative arc |
| [`WNchallenge/wattcast-learnings-for-wn.md`](WNchallenge/wattcast-learnings-for-wn.md) | Four real-world incidents from the wattcast project, used as credibility evidence |
| [`WNchallenge/Onshore energy services - Current landscape and challenges.pdf`](WNchallenge/Onshore%20energy%20services%20-%20Current%20landscape%20and%20challenges.pdf) | Context PDF on the onshore energy services landscape |
| [`WNchallenge/historical_runs_2026-04-08.csv`](WNchallenge/historical_runs_2026-04-08.csv) | MLflow snapshot taken before the SQLite backend migration (safety baseline) |

---

## Conventions

- **English** for code, technical how-tos, and slide plans aimed at an international audience
- **French** allowed for exploration notes and meeting briefings where the source material is French (RTE, éCO2mix, WN meeting notes)
- Filenames in kebab-case for new docs (`rte-api-notes.md`, not `RteApiNotes.md`)
- When a doc supports a specific pass, cross-reference the plan file in `.claude/plans/` at the top
