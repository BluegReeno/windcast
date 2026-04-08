# WN Challenge — Presentation Plan

**Date:** Week of April 14, 2026
**Format:** PPT (English), Craig remote, Yoel + Michel on-site
**Duration:** ~20 min + Q&A
**Goal:** Prove that standardizing the ML pipeline frees engineer time for accuracy — with real results

---

## Narrative Arc

**"I understood your tension. I've lived the pain. Here's a framework that solves the plumbing, so your experts focus on what matters."**

The presentation follows a 3-act structure:
1. **I understood** (slides 1-2) — their problem, my credibility
2. **Here's proof** (slides 3-5) — framework + real results on 2 domains
3. **Here's the path** (slides 6-7) — roadmap + deployment plan

---

## Slide-by-Slide Plan

### Slide 1 — "I Understood Your Challenge"

**Key message:** The core tension is not a technology problem — it's a time allocation problem.

Content:
- WN's stated tension: standardize to free brain time, WITHOUT killing flexibility
- The real insight: the biggest accuracy gains come from **understanding client-specific data and context** — not from running 80 Optuna trials
- Therefore: every hour an engineer spends on data plumbing, QC scripts, deployment debugging, or reporting boilerplate is an hour NOT spent on what actually improves accuracy
- **Framework = solve the plumbing, free the brain**

Visual: simple diagram showing "engineer time" split between plumbing (red) vs accuracy work (green). Framework shifts the ratio.

### Slide 2 — "I've Lived This Pain"

**Key message:** These aren't theoretical risks. I shipped bad predictions to real users.

Content — 3 real incidents from WattCast (live electricity price forecasting system):

| Incident | What happened | Time wasted |
|----------|--------------|-------------|
| **Test overwrites prod** | No model naming convention → test model served to users for days | Debugging + trust repair |
| **Artifact split** | Model in git, features via scp → forgotten files, broken deployments | Hours of "is it the model or the data?" |
| **Optuna overfitting** | v6 beat v5 on validation, worse in production | A full model iteration cycle wasted |
| **Geopolitical shock** | Iran crisis → price spike outside training distribution → predictions useless when it matters most | Days of bad predictions during highest-stakes period |

**Punchline on incidents 1-3:** "A framework prevents these failures. Not by being clever — by being disciplined."

**Punchline on incident 4:** "No framework can anticipate Iran. But a good framework makes sure the engineer sees the impact within hours — not days — and can iterate fast. Regime analysis, rolling monitoring, one-command retraining. The framework surfaces the problem. The engineer solves it."

This is the strongest slide for Yoel — it shows that the framework doesn't pretend to replace expertise. It amplifies it.

Reference: `docs/WNchallenge/wattcast-learnings-for-wn.md` for full details.

### Slide 3 — "The Framework"

**Key message:** One pipeline pattern, any energy domain. Domain-specific code = parser + feature config.

Content:
- Pipeline diagram: `Raw data → [Parser] → Schema → [QC] → Clean → [Features] → [Train] → [Evaluate] → MLflow`
- What the framework handles (plumbing): data ingestion, schema validation, QC rules, temporal splits, MLflow logging, artifact tracking, evaluation metrics
- What the engineer controls (accuracy): feature sets, model choice, QC thresholds, custom metrics, regime analysis
- Adding a new client/dataset = **write a parser (~100 lines) + choose a feature set**

Visual: the pipeline with "framework" (grey/automated) vs "engineer" (blue/creative) zones clearly marked.

### Slide 4 — "Applied to Wind — Real Results"

**Key message:** The framework runs end-to-end on real SCADA data. Here are the numbers.

Content (populated after Wednesday runs):
- Dataset: Kelmarsh wind farm, 6 turbines, 10-min SCADA, 2016-2024
- Pipeline: `ingest → QC (9 rules) → features → train → evaluate`
- Results table: MAE, RMSE, Skill Score per horizon (h1, h6, h12, h24, h48)
- Skill score > 0 at all horizons = **model beats naive persistence**
- Feature set comparison: baseline vs enriched (impact of adding V3, turbulence intensity)
- Regime analysis: performance breakdown by wind speed (low/medium/high)
- **Roadmap to improve:** NWP integration (Open-Meteo), Optuna tuning (conservative), mlforecast recursive prediction

Key talking point: "These are baseline results with minimal tuning. Your wind experts would start HERE and iterate on features and models — not on data plumbing."

### Slide 5 — "Applied to Demand — Same Framework, Zero Core Changes"

**Key message:** The same pipeline handles a structurally different domain. This is the proof of genericity.

Content (populated after Thursday runs):
- Dataset: Spain ENTSO-E, hourly load + weather + prices, 2015-2018
- **Same scripts**, different `--domain demand` flag
- Results table: MAE, RMSE, MAPE, Skill Score per horizon
- Regime analysis: peak/shoulder/off-peak performance
- Comparison with wind: different data, different features, same pipeline
- **MARS → XGBoost:** directly relevant to WN's Gas & Power demand migration

Key talking point: "Adding demand required zero changes to the core pipeline. A parser (120 lines) and a feature config. This is what your team does for each new client."

### Slide 6 — "What Your Team Improves"

**Key message:** The framework handles the floor. Your experts raise the ceiling.

Content — concrete improvement axes, mapped to WN's 3 domains:

| Domain | Current (WN) | With framework | Expert focus |
|--------|-------------|----------------|-------------|
| **Wind/Solar** | Manual QC, manual retraining, R reporting | Automated QC, tracked experiments, reproducible runs | Feature engineering, NWP selection, client-specific metrics |
| **Gas/Power** | MARS only, WNI legacy system | XGBoost/LightGBM/any model, MLflow comparison | Demand-specific features, calendar effects, price correlation |
| **Use Cases** | 100% R, fragile deployment | Python pipeline, versioned artifacts, one-command runs | Domain-specific models (DLR physics, power loss curves) |

Key talking point: "I'm not proposing to replace your expertise. I'm proposing to eliminate the time you spend NOT using your expertise."

### Slide 7 — "Deployment Plan"

**Key message:** Incremental, R-compatible, AWS-aligned.

Content — 3 horizons:

| Horizon | What | Impact | Risk |
|---------|------|--------|------|
| **Quick wins (1-3 months)** | Wind pipeline standardized. MARS → XGBoost POC for 1 demand client. MLflow for experiment tracking. | Productivity + first accuracy gains | Low — no disruption to existing production |
| **Consolidation (3-6 months)** | Shared weather feature store. All domains on standardized pipeline. Reporting migration (Shiny → modern). | Reproducibility + scalability | Medium — requires team adoption |
| **Innovation (6-12 months)** | AI weather model integration. CI/CD retraining. Drift monitoring. | Innovation unlocked | Higher — new capabilities |

Key talking points:
- "R coexists — engineers keep their analysis tools, the pipeline is Python"
- "AWS-native: MLflow on S3, training on SageMaker, Step Functions for orchestration"
- "Start with ONE wind farm, prove value, extend"

---

## Presentation Dynamics

**For Yoel (codes daily):** Slides 3-5 matter most. He wants to see that the framework doesn't replace him — it frees him. The feature set comparison and regime analysis are his language.

**For Michel (ops):** Slides 1, 6, 7 matter most. He wants to see that this is incremental, doesn't break production, and makes the team more productive.

**For Craig (tech, remote):** Slides 3, 4, 5 matter most. He wants to see clean architecture, real results, and that the code is solid (234 tests, typed, linted).

---

## Assets Needed

| Asset | Source | When |
|-------|--------|------|
| Wind metrics (Kelmarsh) | `scripts/train.py` + `evaluate.py` results | Wednesday |
| Demand metrics (Spain) | Same scripts, `--domain demand` | Thursday AM |
| MLflow screenshots | `mlflow ui` after runs | Thursday AM |
| Pipeline diagram | Draw from architecture in PRD | Thursday PM |
| Feature set comparison table | MLflow run comparison | Thursday PM |

---

## What NOT to Include

- No live demo (PPT only — too risky with data download/compute during presentation)
- No deep code walkthrough (mention "234 tests, typed, linted" — don't show code)
- No deep learning / transformer discussion (out of scope, mention as future)
- No Optuna results (not tuned yet — that's the point: "your experts tune, framework tracks")
- No solar domain (keep for Q&A: "I also have solar, want to see?")
