---
marp: true
title: "Freeing Brain Time — An ML Framework for Onshore Energy Forecasting"
author: "Renaud Laborbe"
date: "April 2026"
lang: en
---

# Freeing Brain Time

### A Pragmatic ML Framework for Onshore Energy Forecasting

**Renaud Laborbe** — April 2026
*Technical proposal — WeatherNews challenge*

<!-- notes:
Good morning Yoel, Michel, Craig. Thank you for the challenge and for sharing the landscape deck — it gave me a clear starting point.

What you'll see today is not a sales pitch. It's a working framework that I've built, tested, and run on real data — and a proposal for how it could help your team.

Three acts: (1) what I understood from your challenge, (2) the framework — what it automates, how it prevents errors, and how it facilitates analysis, (3) proof with real numbers and a deployment plan.

I'll aim for ~20 minutes and keep ~10 for questions.
-->

---

## 1. I Understood Your Challenge

**Two core tensions you're living with:**

1. **Standardize vs. domain expertise** — automate the plumbing to free brain time, without replacing the engineer
2. **Standardize vs. rapid innovation** — the ML/AI rhythm is fast, the infrastructure must let you test new models in hours, not weeks

**Three pillars, three pain points — consistent across all of them:**

| Pillar | Current stack | Stated pain |
|---|---|---|
| Wind & Solar Generation | R + AWS + Python (AutoGluon, xgboost, catboost) | Productivity + solar accuracy |
| Gas & Power Demand | R + WNI in-house + **MARS only** | Productivity + day-ahead accuracy + migration |
| Specific Use Cases (DLR, Power Loss) | **100% R (even in prod)** | Productivity + migration to robust environment |

**Key insight I took from our call:**

> *"The biggest accuracy gains come from understanding the client's data — not from running 80 Optuna trials."*

-> Every hour spent on plumbing is an hour NOT spent on what actually moves accuracy.
**A framework should solve the plumbing once, so your experts focus on everything else.**

<!-- notes:
I want to start by reflecting your own words back to you, because that's where my proposal is anchored. You said two things during our call that stuck with me.

First: "standardize to free brain time." Not to replace the engineer. That's a subtle but critical distinction.

Second: "80 Optuna trials will never beat an engineer who understands the client's data." I want you to remember this line, because it drives every design choice in what I built.

The consequence is clear: if you want better accuracy, stop making your engineers write QC scripts, rebuild Parquet pipelines, and debug deployment artifacts. Every hour there is an hour stolen from what actually matters.

I also noticed something in your landscape deck: productivity is mentioned in all three domains. Accuracy in two. Migration in two. So the framework needs to help with all three pain points simultaneously — not just one.
-->

---

## 2. I've Lived This Pain — WattCast Production Incidents

**Context:** WattCast is a live EPEX Spot France day-ahead price forecasting system I built and operate. Daily cron, real users, XGBoost + LEAR ensemble.

**Five real incidents — not theoretical risks:**

| # | Incident | Root cause | What a framework prevents |
|---|---|---|---|
| 1 | **Test model overwrites prod** — bad predictions served for days | No naming, no promotion workflow | Model registry, auto-naming, tagged stages |
| 2 | **Artifact split** — model in git, features via scp -> forgotten files | No unified artifact store | MLflow tracks model + features + metrics as one linked run |
| 3 | **Optuna v6 worse than v5 in production** | Overfit to validation period | Strict temporal split, side-by-side comparison makes regression obvious |
| 4 | **Iran crisis -> price spike -> model blind** | Shock outside training distribution | **Regime analysis + rolling monitoring surface the problem in hours — the engineer decides the response** |
| 5 | **ERA5 data leak** *(discovered while building this framework)* | Reanalysis data used as forecast features = perfect foresight | **Framework guardrails caught it — see slide 5** |

**Incident 5 happened last week.** I was getting suspiciously good results on demand — 326 MW MAE at D+1, better than RTE's own forecast by 4x. The framework's audit trail led me straight to the root cause: two compounding leaks (val-set tuning + ERA5 reanalysis instead of actual NWP forecasts). Fixed in hours, honest numbers in this deck.

**Punchline:** *A framework doesn't just prevent known errors. It makes unknown errors discoverable.*

<!-- notes:
I didn't read about these problems in a blog post. I shipped bad predictions to real users, at midnight, debugging broken deployments.

Incidents 1 through 4 are from WattCast — my live price forecasting system. Each one maps directly to a pain point you mentioned during our call. I'll cover them quickly.

But incident 5 is the one I want to spend a moment on, because it happened last week, while building this very framework for you. I was running my demand pipeline on 11 years of French load data, and my AutoGluon model was showing 326 megawatts MAE at day-ahead. That's four times better than RTE's own official forecast. Too good. Way too good.

Because the framework logs everything to MLflow — features, parameters, data sources, split boundaries — I could trace back exactly what went wrong. Two leaks compounding: (1) AutoGluon was tuning on the validation set, then I was scoring on that same set. Classic self-fulfilling prophecy. (2) More subtly, I was using ERA5 reanalysis data as NWP features. ERA5 is a re-analysis — it's the actual observed weather, not a forecast. The model was seeing the real future weather at every horizon. Perfect foresight.

Both leaks were invisible to the naked eye — the pipeline ran, metrics looked great, no errors anywhere. What surfaced them was the framework's audit trail: the MLflow lineage tags told me exactly which weather database was used, the feature list told me exactly which columns went in, and the stepped horizon chart showed an impossibly flat error curve that screamed "something is wrong."

Fixed both in a few hours. The honest number is 1,139 MW at D+1 — still 5.5% better than RTE's official forecast. Less flashy than 326 MW. But real.

This is the message: a framework doesn't just prevent errors you've seen before. It makes errors you haven't seen yet discoverable. That's the difference between "we got lucky" and "we can audit."
-->

---

## 3. The Framework — EnerCast

**One pipeline pattern. Any energy domain. Clearly separated responsibilities.**

```
Raw data (ZIP / CSV / API)
        |
   +----v-----+
   |  Parser  |   1 per dataset, ~100 LOC — only domain-specific code
   +----+-----+
        | canonical schema (wind / demand / solar)
   +----v-----+
   |    QC    |   parameterized rules engine (shared across domains)
   +----+-----+
   +----v-----+
   | Features |   named sets: baseline / enriched / full
   +----+-----+
   +----v-----+
   |  Train   |   swappable backends: XGBoost / AutoGluon / mlforecast
   +----+-----+
   +----v-----+
   | Evaluate |   MAE, RMSE, skill score, regime analysis, custom metrics
   +----+-----+
        v
      MLflow  (experiments, runs, artifacts, lineage, comparison)
```

**Two clearly separated zones:**

| **Framework handles (plumbing)** | **Engineer controls (accuracy)** |
|---|---|
| Ingestion, schema validation, QC engine | Feature sets, QC thresholds |
| Temporal split enforcement | Model choice + hyperparameters |
| MLflow logging, lineage, artifacts | Custom metrics, regime definitions |
| NWP fetch, cache, horizon-aware joining | Weather source selection, city weighting |
| Evaluation + comparison tooling | Domain-specific analysis |

**Adding a new client = parser (~100 LOC) + feature config. Everything else is ready.**

<!-- notes:
This is the framework architecture. One pipeline, any domain.

The whole thing is about 3,000 lines of Python, 322 tests passing, ruff lint clean, pyright type-checked. It's not a toy.

But the architecture is not what I want you to focus on. I want you to focus on the right-hand column. The framework deliberately leaves all the accuracy-driving decisions to the engineer. Feature sets — the engineer picks. Model — the engineer picks. QC thresholds — the engineer adjusts per client. Custom metrics — the engineer plugs in a function.

The left-hand column is the plumbing that no engineer should spend time on. Schema validation, temporal splits, MLflow logging, NWP caching, evaluation infrastructure. This is what wastes hours today and what the framework solves once.

I'll now go deeper into three specific areas where the framework adds value: guardrails that prevent errors, MLflow as analysis infrastructure, and extension points for your team.
-->

---

## 4. Guardrails — How the Framework Prevents Errors

**Six built-in protections — none require engineer discipline, all are automatic:**

| Guardrail | What it prevents | How it works |
|---|---|---|
| **Typed schemas** | Silent column mismatches, wrong dtypes | Every parser must produce a strict Polars schema (wind=15 cols, demand=11, solar=10). Validation fails loudly if a column is missing or has wrong type. |
| **Temporal split enforcement** | Data leakage through shuffled train/val/test | Split is strictly chronological, boundaries logged as MLflow params. No row from the future can leak into training. |
| **ERA5 / forecast separation** | Perfect-foresight leak (incident 5) | Train period uses ERA5 reanalysis (best historical record). Val/test use archived NWP forecasts (what was actually available). Two different APIs, enforced by the weather provider layer. |
| **Horizon-aware NWP joining** | Using h+48 forecast as h+1 feature | NWP features are joined at forecast issuance time, per horizon. The model at h+6 sees only the NWP forecast that was available 6 hours ago — not the latest analysis. |
| **QC before training** | Training on bad data (sensor faults, curtailment, frozen signals) | 9 rules (wind), domain-specific rules (demand, solar). Flagged rows excluded from training. Thresholds configurable per client. |
| **MLflow lineage tags** | "Which data produced this model?" | Every run is tagged: `domain`, `dataset`, `feature_set`, `backend`, `weather_source`, `data_resolution`, `split_boundaries`. Full provenance, queryable. |

**These aren't optional best practices. They're enforced by the code.**

An engineer can't accidentally train on shuffled data, use ERA5 as forecast features, or deploy a model without knowing which features went in. The framework makes the right thing the default and the wrong thing impossible.

<!-- notes:
This is the slide I think matters most for your team's daily work.

Your engineers are skilled. They know they shouldn't shuffle temporal data. They know ERA5 is not a forecast. They know you need to check data quality before training. But under time pressure, during a trial phase, at 2 AM before a client deadline — discipline slips. That's not a people problem. That's an infrastructure problem.

Let me walk through these six guardrails.

Typed schemas — every parser must produce exactly the right columns with exactly the right types. If the Kelmarsh ZIP changes its column naming, the parser fails immediately with a clear error message telling you which column is wrong. No silent data corruption.

Temporal split enforcement — this one is critical. The split is strictly chronological and the boundaries are logged to MLflow as parameters. You can verify, on any past run, that no future data leaked into training. This is not just a convention — it's enforced by the training harness.

ERA5 versus forecast separation — this is what caught incident 5. The weather provider layer uses two different APIs: the archive API for ERA5 reanalysis during the training period, and the historical-forecast API for actual NWP forecasts during validation and test. You can't mix them by accident because they're routed by date range.

Horizon-aware NWP joining — this is subtle but important. When you train a model for h+48, the NWP features must be the forecast that was available 48 hours ago — not the latest analysis. The feature builder enforces this automatically, per horizon, per child run. Your wind engineers know this distinction. The framework makes it automatic.

QC before training — 9 rules for wind, domain-specific rules for demand and solar. Every row gets a QC flag. Flagged rows are excluded from training. Thresholds are configurable per client — your engineers tune them, the framework enforces them.

MLflow lineage tags — every run knows its own provenance. Domain, dataset, feature set, backend, weather source, data resolution, split boundaries. When something looks wrong, you query MLflow and trace back to the exact data and features that produced the result.

None of these require the engineer to remember anything. They're in the code. The right thing is the default. The wrong thing is a code change that would fail review.
-->

---

## 5. MLflow — Experiment Tracking as Infrastructure

**Not just "we log metrics." MLflow is the backbone of the analysis workflow.**

### What's logged automatically (zero engineer effort)

| What | How | Why it matters |
|---|---|---|
| **Parameters** | XGBoost autolog + custom params (horizons, feature count, split dates) | Reproduce any run exactly |
| **Metrics** | MAE, RMSE, MAPE, skill score, bias — per horizon, flat + stepped | Native line charts: "metric vs horizon" overlay across runs |
| **Lineage tags** | domain, dataset, feature_set, backend, weather_source, purpose | Filter/compare across experiments: "show me all wind_full runs" |
| **Dataset provenance** | `mlflow.data.from_polars()` — auto-hash of train/val DataFrames | Detect if the same model was trained on different data |
| **Feature lists** | Logged as run artifacts | Verify exactly which columns went into training |
| **Run descriptions** | Markdown on parent + child runs (feature set, results table) | Human-readable context without opening the code |

### What the engineer gets for free

1. **1-click comparison** — select 2+ parent runs, Compare tab shows MAE and skill side-by-side across all horizons
2. **Stepped horizon line charts** — metric logged with `step=minutes_ahead`, MLflow renders "MAE vs forecast horizon" natively. Overlay N runs to see where one model diverges from another
3. **Cross-domain experiments** — wind and demand experiments in the same MLflow instance. Same tags, same metrics, same comparison workflow
4. **Parent/child structure** — one parent run per experiment (e.g., "XGB wind_full"), 5 child runs (one per horizon). Summary metrics bubbled up to parent for quick scanning

### The comparison workflow in practice

```bash
# Train two feature sets
uv run python scripts/train.py --feature-set wind_baseline
uv run python scripts/train.py --feature-set wind_full --weather-db data/weather.db

# Generate comparison charts (PNG + Markdown)
uv run python scripts/compare_runs.py --experiment enercast-kelmarsh

# Or: open MLflow UI, select both parents, click Compare
mlflow ui
```

**This is the workflow your engineers repeat on every trial.** The framework makes it 1 command + 1 click — not a custom R script every time.

<!-- notes:
I've seen teams where MLflow is installed but nobody uses it because logging is manual and the UI is confusing. That's not what I built.

In EnerCast, MLflow logging is zero-effort. XGBoost autolog captures parameters, metrics, and feature importance automatically. On top of that, the training harness logs custom metadata: lineage tags, split boundaries, dataset provenance hashes, feature lists as artifacts, and Markdown descriptions on every run.

The result is that when you open MLflow, every run tells you its full story. You don't need to go back to the code to figure out what happened.

Let me focus on three things that matter for your daily workflow.

First, the stepped horizon line charts. This took some engineering to get right — MLflow doesn't natively stitch metrics across sibling runs, so I had to collect per-horizon metrics during the training loop and replay them on the parent run with step=minutes_ahead. The result is that you can open MLflow, select three parent runs, and see three overlaid lines showing MAE versus forecast horizon. At a glance, you see where one feature set starts winning — typically at h6 and beyond when NWP kicks in.

Second, the comparison workflow. After training, one command generates a comparison chart — bar chart of MAE and skill score per horizon, per run. PNG files ready for a report, plus a Markdown table to stdout. Or you can do the same thing interactively in MLflow's Compare tab. Either way, the engineer never writes a comparison script. It's built in.

Third, cross-domain experiments. Wind and demand live in the same MLflow instance, with the same tagging convention. An engineer switching from a wind trial to a demand trial doesn't need to learn a new tool or a new workflow. Same tags, same metrics, same comparison. This is the standardization you asked for — applied to the analysis layer, not just the training layer.

One more thing: the parent/child structure. Each experiment has one parent run that summarizes results, and five child runs — one per forecast horizon. The parent carries summary metrics (h1_mae, h6_mae, etc.) so you can scan 20 experiments quickly. The children carry the full detail. This is a deliberate UX choice — it keeps the experiment list clean.
-->

---

## 6. Extension Points — Small, Sharp Interfaces

**The framework is useful because it's extensible. Here's exactly how much code each extension requires:**

### New custom metric — 1 function

*Example from our call — "MAE when spot price > 150 EUR/MWh":*

```python
def mae_high_spot(y_true, y_pred, spot_price):
    mask = spot_price > 150
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))
```

5 lines. Plugs into the evaluation framework. Logged to MLflow. Compared across runs.

### New ML backend — 1 file

AutoGluon integration = **271 LOC**. Implements `fit()` / `predict()` following the Backend Protocol. Immediately available via `--backend autogluon`. Same features, same evaluation, same MLflow logging.

### New dataset — 1 parser + 1 config

RTE France demand = **~120 LOC parser** + 1 `DatasetConfig` entry. Maps raw TSV-in-ZIP to the canonical demand schema. Zero changes to QC, training, evaluation, or MLflow.

### New weather source — 1 provider

Implements the `WeatherProvider` protocol (fetch + cache). The `WeightedWeatherConfig` pattern I used for 8-city French NWP is reusable for any multi-point weather combination.

**Pattern: test on 1 site, extend to all sites — your Use Cases workflow:**

```bash
# Same commands, different dataset
uv run python scripts/train.py --domain wind --dataset kelmarsh    # site 1
uv run python scripts/train.py --domain wind --dataset penmanshiel # site 2
# MLflow shows both side by side — same tags, same metrics
```

<!-- notes:
This slide is for Yoel specifically — the engineer who codes daily.

During our call, you gave the example of a client who wants accuracy measured specifically when the spot price is above 150 euros per MWh. In many frameworks, that's a feature request. In EnerCast, it's five lines of Python. The evaluation framework accepts any callable that takes y_true, y_pred, and optionally context columns. It's logged to MLflow like any other metric.

Now, new models. I added AutoGluon as a third backend — 271 lines in a single file. It implements two methods: fit and predict. The training harness sees it as just another backend. Same features go in, same evaluation comes out, same MLflow logging. The comparison between XGBoost and AutoGluon is one click in MLflow.

New datasets — this is the genericity proof. RTE France was about 120 lines of parser code. That parser handles the encoding quirks, the date format issues, the resampling from 15-min to hourly, the extraction of the TSO forecast column. Everything downstream — QC, features, training, evaluation — worked unchanged.

New weather sources — the WeatherProvider protocol. I built Open-Meteo, but the pattern supports any source: ECMWF MARS, your WNI in-house combination, AI weather models like GraphCast. The 8-city weighted NWP pattern for French demand is itself reusable — it's a WeightedWeatherConfig with configurable city weights.

And the last point — the scale-out pattern you described for Use Cases. Test on one site, extend to all sites. Same command, different dataset flag. MLflow shows both side by side with the same tags and metrics. No custom comparison script needed.
-->

---

## 7. Proof — Results on Two Domains

**Wind (Kelmarsh, 6 turbines, 10-min SCADA, 2016-2024):**

| Horizon | Without NWP (baseline) | With NWP (full) | Skill improvement |
|---|---|---|---|
| h1 (10 min) | 119 kW | **114 kW** | +15% skill |
| h6 (1 h) | 205 kW | **181 kW** | +98% skill |
| h12 (2 h) | 255 kW | **202 kW** | +170% skill |
| h24 (4 h) | 326 kW | **231 kW** | +195% skill |
| h48 (8 h) | 421 kW | **277 kW** | +181% skill |

NWP is the biggest lever — "rearview mirror -> windshield." This is what your engineers iterate on.

**Demand (RTE France, 11y national load, hourly, 2013-2024):**

| Model | h24 val MAE | h24 test MAE |
|---|---|---|
| RTE Prevision J-1 *(official TSO forecast)* | 1,205 MW | 1,428 MW |
| **EnerCast `demand_full`** *(generic pipeline, stock XGBoost)* | **1,139 MW (-5.5%)** | **1,104 MW (-23%)** |

**We beat the French TSO's own day-ahead forecast** with a generic pipeline, stock XGBoost, 8-city weighted Open-Meteo NWP. No hand-tuning. Reproducible in ~20 seconds on a laptop.

**Both domains — same framework. Wind required 0 lines of pipeline changes for demand.**

All numbers use forecast-time weather features (not ERA5 reanalysis). Fully honest — see slide 2, incident 5.

<!-- notes:
I'm keeping results to one slide because the numbers are not the point — the framework is the point. The numbers prove the framework works.

Wind — the story is NWP integration. Without weather forecasts, the model relies on recent observations — a rearview mirror. With NWP from Open-Meteo, it sees the future — a windshield. Skill score nearly triples at h48. This is the lever your wind engineers would optimize further with better weather sources — ECMWF, ICON-EU, your WNI in-house combination.

Demand — the headline number. RTE publishes its own day-ahead demand forecast — it's the official number used for grid balancing and market bids. Our generic framework, with zero domain-specific tuning, beats it by 5.5% on the validation set and 23% on the test set. 1,104 MW MAE versus 1,428 MW.

Two important caveats. First, this uses Open-Meteo NWP, not ECMWF operational forecasts — your WNI weather combination would likely do better. Second, the test set improvement is larger than the val set improvement, which suggests the model generalizes well rather than overfitting.

And critically — all these numbers are post-leak-fix. The ERA5 reanalysis leak I described in slide 2 was discovered and fixed during development. The val/test periods use actual archived NWP forecasts, not reanalysis. What you see here is what you'd get in production.

The bottom line: both domains, same framework. Adding demand required zero changes to the core pipeline. A parser, a feature config, and a dataset config. This is what adding a new client looks like.
-->

---

## 8. Deployment Plan — Incremental, R-Compatible, AWS-Aligned

**Three horizons — chosen to respect your team of ~20 and their habits.**

| Horizon | What | Impact | Risk |
|---|---|---|---|
| **Quick wins — 1 to 3 months** | Wind pipeline standardized on 1 site. MARS -> XGBoost POC on 1 demand client. MLflow as shared experiment store. Existing R code untouched. | First accuracy gains + productivity | **Low** — no disruption |
| **Consolidation — 3 to 6 months** | Shared weather feature store. All 3 domains on the pipeline. Reporting migration (Shiny -> modern BI). CI for training jobs. | Reproducibility + scalability | **Medium** — team adoption |
| **Innovation — 6 to 12 months** | AI weather model integration (GraphCast, AIFS). Drift monitoring + alerting. Automated retraining. Probabilistic (CQR) forecasts. | Innovation unlocked | **Higher** — new capabilities |

**Key principles:**

- **R coexists.** Analysis and exploration stay in R. Only the production pipeline is Python. No forced migration.
- **AWS-native target.** MLflow on S3, SageMaker for training, Step Functions for orchestration — aligned with the Japan recommendation.
- **Start with ONE site.** Prove value on a single wind farm or demand client. Extend only once the skill lift is visible.
- **Opt-in per project.** No big bang — each project chooses when to adopt.

<!-- notes:
Deployment plan. Three horizons, matching what I'd realistically expect to ship at a 20-person team with established habits.

Horizon 1 — quick wins, 1 to 3 months. Pick one wind site, standardize the pipeline there. Pick one demand client, POC an XGBoost replacement for MARS in parallel with the existing system. Deploy MLflow as a shared experiment store — this alone will change how your team talks about experiments. Zero disruption to existing production.

Horizon 2 — consolidation, 3 to 6 months. This is where the framework starts paying off across all three pillars. Shared weather feature store means you stop re-fetching ECMWF ten times a day. All three domains using the same pipeline means an engineer can switch between wind and demand projects without relearning tooling.

Horizon 3 — innovation, 6 to 12 months. AI weather models — the framework has a WeatherProvider protocol, so a GraphCast provider is one file. Drift monitoring, automated retraining, probabilistic forecasts — all buildable on the existing structure.

The four principles at the bottom are what I think your team specifically needs to hear. R coexists. AWS-native. Start with ONE site. Opt-in per project. No big bang, no deadline pressure.
-->

---

## 9. Closing — The Pitch in Three Lines

### **I understood your challenge**
The tension between standardization and brain time is a **time allocation problem**, not a technology problem. Every hour on plumbing is an hour stolen from accuracy.

### **I've built a framework that solves the plumbing**
Built-in guardrails prevent errors (temporal splits, ERA5/forecast separation, typed schemas). MLflow makes every experiment auditable, comparable, reproducible. Extension points are small and sharp — new metric = 1 function, new model = 1 file, new dataset = 1 parser.

### **I've proved it works**
Two domains (wind + demand), real data, honest forecast-time features. **Beats the French TSO's own day-ahead forecast by 5.5%.** 322 tests, typed, linted. Reproducible in ~20 seconds on a laptop.

---

### **Proposed next step**

> *Let's pick one WN wind site. One week of joint work. I show you the skill score lift on your own data, with your own engineers, with the framework running on your own laptop.*

**Thank you.**

<!-- notes:
Three lines to close.

One: I understood the challenge. I'm not pitching a technology — I'm pitching a way to free your engineers' brain time.

Two: I've built a framework that solves the plumbing. And I want to emphasize three things specifically: the guardrails are automatic — they prevent errors without requiring engineer discipline. MLflow is not just installed — it's wired into every step of the pipeline, making every experiment auditable. And the extension points are deliberately small — your team won't need to learn a framework, they'll need to write a function, a file, or a parser.

Three: I've proved it works on real data with honest methodology. The ERA5 leak story is itself proof — the framework caught its own bug. The honest numbers are strong: we beat RTE's official day-ahead forecast with a stock XGBoost and generic NWP.

The proposed next step is the most important thing on this slide. I don't want to sell you a framework in the abstract. I want to spend one week with your wind team, on one of your sites, with your data, and show the skill score lift together.

Thank you.
-->
