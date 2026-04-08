# WattCast Learnings — Real Production Pain Points

**Context:** WattCast is a live electricity price forecasting system (EPEX Spot France, day-ahead) that Renaud built and operates. Daily cron, Supabase backend, XGBoost + LEAR ensemble, Streamlit dashboard. These are real incidents, not theoretical risks.

---

## Incident 1: Test Model Overwrites Production

**What happened:** During model iteration (v5 → v6), a test model was deployed to the same path as the production model. The daily cron job picked up the test model and used it for live predictions. Results were significantly worse. The incident was only detected after the fact — there was no monitoring to flag the accuracy degradation.

**Root cause:** No naming convention, no test/prod separation, no promotion workflow. Model artifacts were just files in a directory.

**Impact:** Bad predictions served to users for days. Trust damage.

**What a framework prevents:**
- Auto-naming: `{domain}_{dataset}_{feature_set}_{version}_{timestamp}` — no overwrite possible
- Test/prod separation: explicit environments, models tagged as `staging` or `production`
- Promotion workflow: promoting a model to prod is a deliberate, reviewable action — not an accidental file overwrite
- Monitoring: skill score tracked per run in MLflow — regression is visible immediately

---

## Incident 2: Deployment Artifact Split

**What happened:** Model artifacts were split between two deployment mechanisms:
- Small files (model weights, config) → committed to git, deployed via `git pull`
- Large files (feature Parquet, evaluation data) → gitignored, deployed via manual `scp`

Inevitably, some `scp` steps were forgotten. Production ran with stale feature data while having a new model, or vice versa. This inconsistency contributed to Incident 1.

**Root cause:** No unified deployment mechanism. ML artifacts don't fit in git, but no alternative was in place.

**Impact:** Broken deployments, debugging nightmare ("is it the model or the data?"), manual process that doesn't scale.

**What a framework prevents:**
- MLflow artifact store: all artifacts (model, features, evaluation) tracked together in one run
- Versioned together: you can't deploy model v6 with features from v5 — they're linked
- One-command deployment: `mlflow models serve` or export to a deployment target
- Reproducibility: any past run can be fully recreated from MLflow metadata

---

## Incident 3: Optuna Overfitting (v5 → v6 Regression)

**What happened:** Hyperparameter tuning with Optuna produced a model (v6) that showed better metrics on validation data but performed worse in production. Excessive tuning overfit to the validation period's specific market conditions.

**Root cause:** No out-of-sample evaluation discipline. Validation metrics were trusted without checking temporal robustness.

**What a framework prevents:**
- Strict temporal split: train / validation / test — test is NEVER seen during tuning
- Conservative defaults first: start with known-good hyperparameters, tune incrementally
- Multiple evaluation windows: compare metrics across different time periods, not just one
- MLflow comparison: v5 vs v6 metrics side by side, per horizon, per regime — regression is obvious

---

## Incident 4: Exogenous Shock — Model Blind to Geopolitical Events

**What happened:** Iran crisis caused a massive spike in European electricity spot prices. The model had never seen this pattern in training data — no geopolitical shock of this magnitude existed in the historical dataset. Predictions were dramatically off for days. MAPE exploded.

**Root cause:** This is NOT a bug. It's the fundamental limitation of any ML model trained on historical data. But the real problem was deeper: without a clear mental model of the pipeline — what the model sees, what it doesn't, how features are built — it was hard to even diagnose WHY predictions were off, let alone decide what to do about it.

**The non-trivial part:** The solution isn't "add a geopolitical feature." And critically — no amount of Bayesian optimization, hyperparameter tuning, or AutoML is going to find it either. Optuna doesn't understand geopolitics. XGBoost doesn't read the news.

The solution requires a **human expert** who:
1. **Understands the pipeline end-to-end** — which features drove the prediction? Was the model extrapolating or interpolating?
2. **Recognizes the regime shift** — the model is operating outside its training distribution, this is not a tuning problem
3. **Decides the right response** — retrain on recent data? Add a manual override? Increase prediction intervals? Fall back to a simpler model? Weight recent observations more heavily?
4. **Acts fast** — because during a price spike, every hour of bad predictions costs the client real money

This is exactly why "80 Optuna trials" is not the answer and why WN is right to value "brain time" over compute time. The engineer who understands the data and the context will always beat the engineer who runs more hyperparameter searches.

**Impact:** Bad predictions during the highest-stakes period (when prices are extreme = when accuracy matters most to clients).

**What a framework enables (but doesn't solve automatically):**
- **Regime analysis built-in**: performance breakdown by price regime reveals that the model was never tested on extreme prices
- **Rolling evaluation**: skill score tracked over time, not just at training time — degradation is visible within hours, not days
- **Fast retraining**: when the engineer decides to retrain, the pipeline is one command — not a multi-day rebuild
- **Feature set comparison**: quickly test "does adding recent price volatility as a feature help?" — run a new experiment, compare in MLflow
- **The framework surfaces the problem. The engineer solves it.** This is the distinction — no framework can anticipate Iran. But a good framework makes sure the engineer sees the impact fast and can iterate fast.

---

## The Pattern: What WN Engineers Face Daily

These four incidents map directly to WN's stated pain points:

| WattCast Incident | WN Equivalent | Framework Solution |
|---|---|---|
| Test overwrites prod | Trial model accidentally deployed to client | Model lifecycle management (MLflow registry) |
| Artifact split | Data cleaning scripts + model + reporting on different systems | Unified pipeline: data → model → evaluation → artifacts in one tracked run |
| Optuna overfitting | "80 Optuna trials" vs "understanding the data" | Disciplined evaluation + comparison tooling that makes regression visible |
| Iran price shock | Cold snap destroys demand model, cloud event wrecks solar | Regime analysis + rolling monitoring surface the problem fast. Engineer decides the response. Framework enables fast iteration. |

---

## Key Message for WN

> "I've lived these problems. I didn't read about them — I shipped bad predictions to real users, debugged broken deployments at midnight, and watched a tuned model perform worse than its predecessor. The framework I'm proposing exists because I've felt the pain of not having one."

The framework doesn't replace engineering judgment. It eliminates the mechanical failures that waste engineer time — so they can focus on understanding the client's data and context. Exactly what WN calls "libérer du temps cerveau."
