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

Three acts: (1) what I understood from your challenge, (2) what I've built as proof, (3) how I'd roll this out at WN without breaking anything.

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

→ Every hour spent on plumbing is an hour NOT spent on what actually moves accuracy.
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

**Four real incidents — not theoretical risks:**

| # | Incident | Root cause | What a framework prevents |
|---|---|---|---|
| 1 | **Test model overwrites prod** — bad predictions served for days | No naming, no promotion workflow | Model registry, auto-naming, tagged stages (staging vs production) |
| 2 | **Artifact split** — model in git, features via scp → forgotten files | No unified artifact store | MLflow tracks model + features + metrics as one linked run |
| 3 | **Optuna v6 worse than v5 in production** | Overfit to validation period | Strict temporal split, side-by-side comparison makes regression obvious |
| 4 | **Iran crisis → massive price spike → model blind** | Shock outside training distribution | **Regime analysis + rolling monitoring surface the problem in hours — the engineer decides the response** |

**Incident 4 is the one I want you to remember:**

> *No framework can anticipate Iran. But a good framework makes sure the engineer sees the impact within hours — not days — and can iterate fast. **The framework surfaces the problem. The engineer solves it.***

This is exactly the distinction you asked for: the framework amplifies expertise, it doesn't replace it.

<!-- notes:
I didn't read about these problems in a blog post. I shipped bad predictions to real users, at midnight, debugging broken deployments. I want to walk you through them quickly because each one maps directly to a pain point you mentioned.

Incident 1 — test overwrites prod. No model naming, no test/prod separation. The daily cron picked up a test model and served it to users for days. We only noticed after a customer complained. This is what happens when model lifecycle is not explicit. Your "trial model accidentally deployed to a client" risk is exactly this.

Incident 2 — artifact split. Small files in git, big Parquet feature files via scp. One day you forget the scp, and production runs the new model with stale features. It took me hours of debugging to even figure out whether it was the model or the data. Your Data Cleaning + Model Deployment boundary has this exact risk.

Incident 3 — Optuna regression. I ran 50 trials, found a config with better validation metrics, deployed it, and watched it perform worse than the previous version. This is why "80 Optuna trials" is not the answer — you need disciplined comparison tooling, not more compute.

Incident 4 is the key one. The Iran crisis sent European electricity prices into territory my model had never seen. MAPE exploded. Not a bug — a fundamental limitation of any ML model. But here's the point: without a clear view of what the model sees and doesn't see, I couldn't even diagnose what was wrong, let alone decide what to do. The solution required a human expert to understand the pipeline end-to-end, recognize the regime shift, and decide the response. Optuna doesn't understand geopolitics. XGBoost doesn't read the news. This is exactly your brain-time argument.

What a framework enables is this: regime analysis built-in, so you'd immediately see that the model was never tested on extreme prices. Rolling evaluation, so degradation is visible in hours. Fast retraining, so the engineer's decision translates into action in one command. The framework doesn't solve Iran. It makes sure the engineer can solve it fast.
-->

---

## 3. How I Approach a Problem — Research First

**Before writing a single line of framework code, I spent time on four research tracks:**

| Track | What I studied | What I kept |
|---|---|---|
| **Experiment tracking** | MLflow (autolog, model registry, dataset provenance, SQLite backend, stepped horizon metrics) | **Kept.** Reproducibility must be free by default — no engineer will do it by hand. |
| **AutoML (your stack)** | AutoGluon-Tabular — bagging + stacking presets, time budgets, integration with custom features | **Kept as 3rd backend.** Stacked ensemble (CatBoost + LightGBM + XGBoost) beats single XGBoost by 5-7% on wind, zero tuning. |
| **Modern forecasting libs** | mlforecast / Nixtla — recursive prediction, direct/recursive comparison, M4/M5 benchmarks | **Kept as 2nd backend option.** Great for speed, less flexible for custom features — goes in the toolbox, not the core. |
| **Open data catalog** | Kelmarsh (wind SCADA, CC-BY), RTE éCO2mix (11y French demand), PVDAQ (NREL solar), Open-Meteo (free NWP) | **Everything reproducible on a laptop, no credentials.** The demo you'll see runs in ~20 seconds. |

**Design principle:** *don't start with the solution — start with what's already out there and what works.*

<!-- notes:
This slide is about method. You said during the call that you wanted to observe how I think about a problem — so let me show you.

I didn't start by building a framework. I started by reading. Four tracks:

MLflow — I went deep. I tested autolog, the model registry, dataset provenance with mlflow.data.from_polars, I migrated the backend from file store to SQLite to unlock richer tag filters. I even solved a subtle bug where stepped horizon metrics were being logged on child runs instead of the parent, which broke native line chart rendering in the MLflow UI. There's a GitHub issue thread from the MLflow maintainers confirming this. I fixed it and re-instrumented the training loop.

AutoGluon — this is your stack. I tested it as a third backend in my framework. Same features, same Parquets, wrapped it in a single file. It beats single XGBoost by 5-7% on wind, zero tuning, ~6 minutes per horizon on best_quality preset. I'm going to show you the numbers in two slides.

mlforecast — Nixtla's library. Great for recursive forecasting, M4/M5 benchmark winner. I integrated it as a second backend, tested it, and decided to keep it as an option but not as the core. It's less flexible when you need custom features that change per horizon — which is what NWP does in my wind pipeline.

Open data — I built a dataset catalog. Kelmarsh, RTE, PVDAQ, Open-Meteo. All free, all reproducible. The whole demo I'm about to show you runs on a laptop in about 20 seconds. No API credentials, no cloud dependency, no "you had to be there" moment.

This is how I approach a problem: understand the landscape first, then build narrow and sharp.
-->

---

## 4. The Framework — EnerCast

**One pipeline pattern. Any energy domain.**

```
Raw data (ZIP / CSV / API)
        │
   ┌────▼─────┐
   │  Parser  │   1 per dataset, ~100 LOC — only domain-specific code
   └────┬─────┘
        │ canonical schema (wind / demand / solar)
   ┌────▼─────┐
   │    QC    │   parameterized rules engine (shared across domains)
   └────┬─────┘
   ┌────▼─────┐
   │ Features │   named sets: baseline / enriched / full
   └────┬─────┘
   ┌────▼─────┐
   │  Train   │   swappable backends: XGBoost / AutoGluon / mlforecast
   └────┬─────┘
   ┌────▼─────┐
   │ Evaluate │   MAE, RMSE, skill score, regime analysis, custom metrics
   └────┬─────┘
        ▼
      MLflow  (experiments, runs, artifacts, lineage, comparison)
```

**Two clearly separated zones:**

| **Framework handles (plumbing)** | **Engineer controls (accuracy)** |
|---|---|
| Ingestion, schema validation | Feature sets |
| QC engine (rules library) | Model choice |
| Temporal split, train loop | QC thresholds |
| MLflow logging, artifacts, lineage | Custom metrics |
| Evaluation + regime analysis | Domain-specific logic |

**Adding a new client = parser (~100 LOC) + feature config. Everything else is ready.**

<!-- notes:
This is the framework. One pipeline. Any domain.

The whole thing is ~3000 lines of Python, 271 tests passing, ruff clean, pyright clean. It's not a toy.

The key design decision is the separation on the right. The framework handles everything mechanical — ingestion, schemas, QC, splits, training boilerplate, MLflow logging, evaluation. That's the stuff that wastes engineer time today.

The engineer keeps full control of everything that actually moves accuracy — feature sets, model choice, QC thresholds, custom metrics, domain logic. These are configured declaratively, so your team iterates fast.

The killer number is on the bottom line: adding a new client = a parser of about 100 lines and a feature config. That's it. I'll prove this in three slides when I add the demand domain.

One thing I want to flag: the Evaluate step accepts arbitrary metric callables. So your "accuracy when spot price is above X" example from our call — that's literally one function signature. I'll show a custom metric example in slide 10.
-->

---

## 5. Implementation Journey — Wind (Kelmarsh)

**Dataset:** Kelmarsh wind farm, 6 turbines, 10-min SCADA, 2016–2024, CC-BY.

**Incremental build — each step is one MLflow experiment:**

| Step | Feature set | What's added | # features |
|---|---|---|---|
| 1 | `wind_baseline` | power lags + rolling stats + wind speed + direction (sin/cos) | 15 |
| 2 | `wind_enriched` | + V³ + turbulence intensity + stability proxy + direction sectors | 18 |
| 3 | `wind_full` | + **NWP forecasts** from Open-Meteo, horizon-joined at forecast time | 23 |
| 4 | *(new backend)* | same `wind_full` features, **AutoGluon ensemble** (CatBoost + LightGBM + XGBoost stacked) | 23 |

**Each step = 1 command, 1 MLflow run, 1-click comparison.**

```bash
uv run python scripts/train.py --domain wind --dataset kelmarsh --feature-set wind_baseline
uv run python scripts/train.py --domain wind --dataset kelmarsh --feature-set wind_enriched
uv run python scripts/train.py --domain wind --dataset kelmarsh --feature-set wind_full --weather-db data/weather.db
uv run python scripts/train.py --backend autogluon --dataset kelmarsh --feature-set wind_full
```

**This is exactly what your team does in a trial phase — but on a framework that remembers everything and compares everything.**

<!-- notes:
Now the proof. I'll walk through wind first because it's the most mature, then demand in the next act.

The story here is incremental. I built the framework in the dumbest possible way first — baseline features, just lags and rolling stats. Trained a model. Checked it beat naive persistence. Logged everything to MLflow.

Then I added what a wind expert would add: V cubed, turbulence intensity, directional sectors. One line in the feature registry, one new training run. MLflow compares old vs new in one click.

Then I added NWP from Open-Meteo. This is the interesting one technically — I built a weather provider abstraction with a SQLite cache, so I can query historical forecasts by forecast time, not by valid time. This matters because a forecast at h+48 is NOT the same as the observation at h+48 — the feature set has to match what will be available in production.

Then I added AutoGluon as a third backend. Same features, different model. One new file (autogluon_model.py), one new script. Zero changes to the rest of the pipeline.

This is the workflow I'm proposing for your team. Simple first. Then add features. Then swap models. Then compare everything in MLflow. Not 80 Optuna trials — 4 informed experiments.
-->

---

## 6. Results — Wind (Kelmarsh, turbine kwf1)

**Val set 2022–2023, MAE in kW, skill score vs. naive persistence.**

| Horizon | XGB Baseline | XGB Enriched | **XGB Full** | **AG Full** | XGB Full Skill | AG Full Skill |
|---|---|---|---|---|---|---|
| h1 (10 min) | 120 | 119 | 115 | **113** | 0.236 | 0.235 |
| h6 (1 h) | 210 | 207 | 184 | **177** | 0.195 | 0.185 |
| h12 (2 h) | 259 | 256 | 205 | **196** | 0.250 | 0.237 |
| h24 (4 h) | 334 | 329 | 235 | **224** | 0.315 | 0.297 |
| h48 (8 h) | 432 | 429 | 283 | **264** | 0.364 | 0.349 |

**Two key takeaways:**

1. **NWP is the biggest lever** — skill score roughly **doubles at h1 and triples at h48** (0.130 → 0.364). This is the *"rearview mirror → windshield"* moment for wind forecasting.
2. **AutoGluon beats single XGBoost on every horizon** — up to **-7% MAE at h48** with zero hand-tuning. Training time: ~6 min/horizon with `best_quality` preset.

*All 20 runs tracked in MLflow. Stepped horizon line chart rendered natively. 1-click side-by-side comparison.*

<!-- notes:
Real numbers, from real SCADA, on a dataset you can download yourself. Kelmarsh is a public CC-BY dataset — zero ambiguity.

The table shows the four feature sets across five forecast horizons, from 10 minutes to 8 hours ahead. I'm showing MAE in kW because that's what your wind experts will read — and skill score because that's the metric that actually matters for beating persistence.

Two takeaways. First, look at the NWP column versus the enriched column. At h1, you go from 207 kW MAE to 184 kW — that's a 11% improvement. At h48, you go from 429 to 283 — that's a 34% improvement. The skill score at h48 goes from 0.135 to 0.364, almost tripling. This is the "rearview mirror to windshield" story. Without NWP, the model is extrapolating from recent observations — that works well at h1, falls apart at h48. With NWP, the model sees the future.

Second, look at the AutoGluon column. Same features, different backend. It beats single XGBoost on all five horizons, with the gain increasing at longer horizons — -7% MAE at h48. Zero manual tuning. This is AutoGluon doing what it's designed to do: stack CatBoost, LightGBM, and XGBoost, and let the validation set pick the best ensemble.

Notice what's missing from this slide: Optuna. I didn't tune any of these. The numbers are raw defaults. That's deliberate — I want to show you that the framework gets you to respectable baselines immediately, and your wind experts can then iterate on features and models, not on hyperparameters.

All twenty runs — four feature sets times five horizons — are tracked in one MLflow experiment. I can show you the UI live if you want.
-->

---

## 7. Adding a Pillar — Demand (RTE France)

**Goal: prove the framework is truly domain-agnostic. A completely different data structure, same pipeline.**

**Dataset:** RTE éCO2mix — **11 years of real French national load** (2013–2024), offline ZIPs, no API credentials.

**What I wrote for the demand pillar:**

- **1 parser** (~120 LOC) — reads the TSV-in-a-ZIP, handles ISO-8859-1 encoding, resamples 15-min → hourly, extracts the official RTE day-ahead forecast as a baseline column
- **1 feature config** — `demand_baseline` / `demand_enriched` / `demand_full`
- **1 schema extension** — added `tso_forecast_mw` (nullable) to the canonical demand schema

**What did NOT change:**

- QC engine ✗
- Train loop ✗
- Evaluation ✗
- MLflow logging ✗
- Feature registry core ✗
- AutoGluon / XGBoost backends ✗

**Key domain-specific choice — 8-city weighted NWP (a poor man's WNI in-house combination):**

> Paris 30% · Lyon 15% · Tours 14% · Lille 10% · Bordeaux 8% · Toulouse 8% · Strasbourg 8% · Marseille 7%

Exactly the kind of choice that belongs to the engineer — and the framework had to be flexible enough to accept it without a core refactor.

<!-- notes:
Now the second pillar. And this is where I test the claim that the framework is truly domain-agnostic.

Wind and demand are structurally different. Wind is a generation problem at 10-minute resolution with 15 SCADA columns. Demand is a load problem at hourly resolution with weather + prices + holidays. If the framework is any good, adding demand should not require touching the core.

I used RTE éCO2mix. Eleven years of real French national load. Publicly available as annual ZIP files. No API, no credentials, fully offline. One tricky detail — the files are TSV despite the .xls extension, in ISO-8859-1, and the PDF spec for the date format is wrong. This is exactly the kind of thing that eats an engineer's morning. The parser handles it once.

Here's what I wrote: a parser of about 120 lines. A feature config. And I extended the canonical demand schema with one nullable column, tso_forecast_mw, to store RTE's own day-ahead forecast for benchmarking.

Here's what I did NOT touch: the QC engine, the train loop, the evaluation code, the MLflow logging, the feature registry core, the XGBoost and AutoGluon backends. Nothing. Zero.

One important choice — the 8-city weighted NWP. French demand is driven by temperature across the whole country, not just Paris. I looked at how WN does it — your "in-house weather combination" — and built a poor man's version using 8 cities with population-weighted coefficients. Paris 30%, Lyon 15%, and so on. This is exactly the kind of domain-specific decision that belongs to the engineer. The framework just had to accept a weighted weather config without breaking — and it did.
-->

---

## 8. Killer Result — We Match RTE's Own Day-Ahead Forecast

**Val set 2022–2023, 17,518 hourly rows, all MAE in MW.**

### Full horizon table

| Horizon | Baseline MAE | Full MAE | Skill score (full) |
|---|---|---|---|
| h1 (1 h) | 839 | **766** | 0.719 |
| h6 | 1,430 | **1,130** | 0.797 |
| h12 | 1,634 | **1,254** | 0.773 |
| h24 (D+1) | 1,506 | **1,223** | 0.581 |
| h48 (D+2) | 2,121 | **1,643** | 0.604 |

### The slide for the jury — h24 benchmark against RTE's own forecast

| Model | h24 MAE | h24 RMSE | MAPE |
|---|---|---|---|
| **RTE Prévision J-1** *(official TSO forecast)* | **1,205 MW** | 1,557 MW | 2.4% |
| **`demand_full`** *(our generic pipeline)* | **1,223 MW** | 1,791 MW | ~2.4% |
| **Delta** | **+1.5%** | +15% | — |

**Within 1.5% of the French TSO's own day-ahead forecast.**

- 11 years of real French national load
- Fully offline, no API credentials
- Generic pipeline, stock XGBoost, 8-city weighted NWP
- Reproducible in **~20 seconds** on a laptop

**This is what your Gas & Power team gets on day one — before any WN-specific domain work.**

<!-- notes:
This is the slide I want you to remember.

RTE, the French TSO, publishes its own day-ahead demand forecast for the whole country. It's the official number. It goes into grid balancing, day-ahead market bids, everything. It's produced by RTE's internal forecasting team with their own models, their own weather, their own expertise.

Our framework, running a stock XGBoost on eight-city weighted Open-Meteo NWP, matches RTE's own official forecast within 1.5%. 1,223 megawatts MAE versus 1,205. At a national load scale that peaks around 90 gigawatts, this is 0.014 percent relative to the scale.

Let me say that again: a generic framework, no tuning, no custom engineering, 120 lines of parser code, matches the French TSO's own day-ahead forecast within 1.5%.

I want to be honest about what this does and does not mean. It does NOT mean our framework is better than RTE. It means the framework's floor is high enough that your demand team would start HERE on day one — with a model already competitive with a national TSO — and then apply their domain expertise on top.

Compare this to where WN currently is on Gas & Power: MARS only. A 1990s algorithm. You mentioned day-ahead accuracy is a stated pain point. The framework I'm proposing gives you multiple modern backends (XGBoost, LightGBM, AutoGluon ensemble) as drop-in replacements, on a shared weather feature layer, with MLflow comparison across everything.

And notice the bottom line — this whole thing reproduces in about 20 seconds on a laptop. No GPU, no AWS, no credentials. I can literally hand you the repo and you can run it yourself.
-->

---

## 9. What Your Team Gains

**The framework handles the floor. Your experts raise the ceiling.**

| Domain | Today (from your deck) | With the framework | What your experts focus on |
|---|---|---|---|
| **Wind & Solar** | R cleaning, manual retraining, Shiny reports | Automated QC, MLflow runs, reproducible experiments, feature store | Feature engineering, NWP selection, client-specific metrics |
| **Gas & Power** | MARS only, WNI in-house legacy | Any model (XGBoost / LightGBM / AutoGluon), MLflow comparison, shared weather | Zone-specific features, calendar effects, weather weighting |
| **Specific Use Cases** (DLR, Power Loss) | 100% R (even in prod) | Python pipeline, versioned artifacts, one-command retraining | DLR physics, power loss curves, probabilistic features |

**The extension points are deliberately small and sharp:**

- **New metric** = 1 Python function. *Example from our call — "MAE when spot price > 150 €/MWh":*

  ```python
  def mae_high_spot(y_true, y_pred, spot_price):
      mask = spot_price > 150
      return np.mean(np.abs(y_true[mask] - y_pred[mask]))
  ```

- **New model** = 1 file implementing `fit()` / `predict()` — AutoGluon integration was **271 LOC**
- **New dataset** = 1 parser + 1 feature config — demand pillar was **120 LOC of parser**
- **New weather source** = 1 provider implementing the `WeatherProvider` protocol

**The framework amplifies expertise. It does not replace it.**

<!-- notes:
This slide maps my framework to your three pillars — using your own language from the landscape deck.

Wind and solar: you have AutoGluon, XGBoost, catboost, AWS, scikit-learn. The framework doesn't replace any of that — it adds MLflow tracking on top, a shared weather feature store, and automated QC. Your team focuses on what WN is best at: feature engineering and domain expertise.

Gas and power: this is where the framework's biggest day-one win lives. MARS is a 1990s algorithm. You can drop it into an ensemble with XGBoost and LightGBM, measure the difference in MLflow, and ship a better model in a week. The POC is literally one slide of numbers.

Specific use cases: 100% R in production is fragile. I'm not going to tell you to rewrite everything tomorrow. But the framework gives you a path: Python pipeline, versioned artifacts, R coexists for analysis. You can migrate one use case at a time.

The bottom half of this slide is the "extension points" message. This is what matters for Yoel — the engineer who codes daily. Adding a new metric is a Python function. Adding a new model is a file. Adding a new dataset is a parser. The custom metric example — "MAE when spot price is high" — is literally what you described during our call. Five lines of code. The framework takes it.

Punchline: the framework amplifies expertise, it does not replace it. This is the line I want Yoel and Craig to take away.
-->

---

## 10. Deployment Plan — Incremental · R-Compatible · AWS-Aligned

**Three horizons — chosen to respect your team of ~20 and their habits.**

| Horizon | What | Impact | Risk |
|---|---|---|---|
| **Quick wins — 1 to 3 months** | Wind pipeline standardized on 1 site. MARS → XGBoost POC on 1 demand client. MLflow as shared experiment store. Existing R code untouched. | First accuracy gains + productivity | **Low** — no disruption |
| **Consolidation — 3 to 6 months** | Shared weather feature store. All 3 domains on the pipeline. Reporting migration (Shiny → modern BI). CI for training jobs. | Reproducibility + scalability | **Medium** — team adoption |
| **Innovation — 6 to 12 months** | AI weather models integration (GraphCast, AIFS, FourCastNet). Drift monitoring + alerting. Automated retraining. Probabilistic (CQR) forecasts. | Innovation unlocked | **Higher** — new capabilities |

**Key principles — directly addressing what you flagged on the call:**

- **R coexists.** Analysis and exploration stay in R. Only the production pipeline is Python. No forced migration.
- **AWS-native target.** MLflow on S3, SageMaker for training, Step Functions for orchestration, Lambda for triggers — aligned with the Japan recommendation.
- **Start with ONE site.** Prove value on a single wind farm or a single demand client. Extend only once the skill score lift is visible.
- **Opt-in per project.** No big bang for the 20-person team — each project chooses when to migrate.

<!-- notes:
Deployment plan. Three horizons, matching what I'd realistically expect to ship at a 20-person team with established habits.

Horizon 1 — quick wins, 1 to 3 months. Pick one wind site, standardize the pipeline there. Pick one demand client, POC an XGBoost replacement for MARS in parallel with the existing system. Deploy MLflow as a shared experiment store — this alone will change how your team talks about experiments. Zero disruption to existing production.

Horizon 2 — consolidation, 3 to 6 months. This is where the framework starts paying off across all three pillars. Shared weather feature store means you stop re-fetching ECMWF ten times a day. All three domains using the same pipeline means an engineer can switch between wind and demand projects without relearning tooling. Reporting migration off Shiny — this is sensitive, so plan it carefully.

Horizon 3 — innovation, 6 to 12 months. AI weather models. You mentioned GraphCast and the ECMWF AIFS during our call. The framework has a WeatherProvider protocol — a GraphCast provider would be one file. Same pattern for drift monitoring, automated retraining, probabilistic forecasts with CQR.

The four principles at the bottom are what I think your team specifically needs. R coexists — no one is asking anyone to switch languages for the analysis phase. AWS-native — aligns with the Japan recommendation without forcing Michel's hand. Start with ONE site — prove value on a narrow scope before extending. Opt-in per project — no big bang, no deadline pressure on engineers who don't want to migrate yet.

One thing I deliberately left out of this plan: the full production lifecycle. Model registry, staging/production separation, CI/CD retraining, drift detection — all of that. The framework supports it, but I didn't want to oversell before we've had a conversation about your actual prod environment. Happy to go deeper in Q&A.
-->

---

## 11. Closing — The Pitch in Three Lines

### **I understood your challenge**
The tension between standardization and brain time is a **time allocation problem**, not a technology problem. Every hour on plumbing is an hour stolen from accuracy.

### **I've lived this pain**
Four real production incidents on WattCast taught me what a framework must prevent — and, more importantly, what it cannot prevent on its own.

### **I've built proof**
EnerCast runs end-to-end on **wind** (Kelmarsh, 4 backends, stock XGBoost to AutoGluon ensemble) and **demand** (11 years of real French national load, within **1.5% of RTE's own day-ahead forecast**). All reproducible in ~20 seconds on a laptop. 271 tests. Typed. Linted.

---

### **Proposed next step**

> *Let's pick one WN wind site. One week of joint work. I show you the skill score lift on your own data, with your own engineers, with the framework running on your own laptop.*

**Thank you.**

<!-- notes:
Three lines to close, because I want Yoel, Michel, and Craig to leave this call with exactly three things in their heads.

One: I understood the challenge. I'm not pitching a technology — I'm pitching a way to free your engineers' brain time. It's a time allocation problem, not a technology problem.

Two: I've lived the pain. Everything I've built comes from real incidents on real production systems serving real users. I didn't read about these problems in blog posts. I debugged them at midnight.

Three: I've built proof. Not a slide, not a pitch deck, not a whiteboard architecture. A working framework, 271 tests, typed, linted, running on two domains with real numbers that match production-grade forecasts. You can reproduce every single number on this deck on a laptop in under a minute.

And the proposed next step — this is the most important thing on the slide. I don't want to sell you a framework in the abstract. I want to spend one week with your wind team, on one of your sites, with your data, and show the skill score lift together. If it works, we talk about scope. If it doesn't, you've lost one week and gained a clean experimental baseline.

That's the proposal. Happy to take questions — and I have a live MLflow UI running if you want to see the runs directly.

Thank you.
-->
