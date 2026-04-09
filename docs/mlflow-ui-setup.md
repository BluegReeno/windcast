# MLflow UI Setup

Quick reference for launching and configuring the MLflow UI for EnerCast. The
project uses the SQLite backend by default (`sqlite:///mlflow.db`), which
unlocks `IS NULL` tag filters and removes the deprecation warning of the
legacy file store.

## Launch the UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Opens on <http://127.0.0.1:5000>. Artefacts are stored under
`./mlartifacts/` next to the database file.

> If you previously used the file-based backend (`./mlruns`), that directory
> is no longer read. A CSV snapshot of the pre-migration runs lives in
> `docs/WNchallenge/historical_runs_2026-04-08.csv` for reference.

## Recommended filters

EnerCast tags every run with a set of `enercast.*` tags. The most useful one
for the comparison view is `enercast.run_type`, which is either `parent`
(one per training invocation) or `child` (one per forecast horizon inside the
parent).

| Goal | Filter string |
|------|--------------|
| Only comparable top-level runs | ``tags.`enercast.run_type` = 'parent'`` |
| Only XGBoost runs | ``tags.`enercast.backend` = 'xgboost'`` |
| Cross-backend comparison on the same feature set | ``tags.`enercast.feature_set` = 'wind_full'`` |
| Wind domain only | ``tags.`enercast.domain` = 'wind'`` |
| Parent runs that used NWP features | ``tags.`enercast.run_type` = 'parent' and tags.`enercast.feature_set` = 'wind_full'`` |

Note: in the MLflow UI search bar, tag names with dots must be wrapped in
backticks (e.g. `` `enercast.run_type` ``), otherwise the parser fails with
a cryptic syntax error.

## Charts tab configuration

The Charts tab keeps a persistent chart layout per experiment. Configure it
once and it survives across UI restarts.

1. Open the `enercast-kelmarsh` experiment.
2. Apply the filter ``tags.`enercast.run_type` = 'parent'`` so the charts
   only aggregate over parent runs.
3. Open the **Charts** tab → **Add chart** → **Bar chart**.
4. For the MAE comparison chart:
   - Metric: one bar chart per horizon (`h1_mae`, `h6_mae`, `h12_mae`,
     `h24_mae`, `h48_mae`) — or use multiple metrics in a single grouped chart.
   - Group by: `tags.enercast.backend` (or `tags.enercast.feature_set`).
   - Title: "MAE by horizon".
5. Duplicate the chart for skill scores using the `h{n}_skill_score` metrics
   and title "Skill score by horizon".
6. Pin the chart configuration using the bookmark icon so it survives UI
   restarts.

## Alternative: programmatic comparison

For slide-ready PNG charts that do not depend on the UI, use
`scripts/compare_runs.py`:

```bash
uv run python scripts/compare_runs.py --experiment enercast-kelmarsh
```

This generates `reports/comparison_enercast-kelmarsh_mae.png` and
`reports/comparison_enercast-kelmarsh_skill.png`, and prints a Markdown
comparison table to stdout that can be pasted into slides or STATUS.md.

## Tag reference

All tags set by the EnerCast training scripts:

| Tag | Values | Purpose |
|-----|--------|---------|
| `enercast.run_type` | `parent`, `child` | Distinguish comparable top-level runs from per-horizon children |
| `enercast.stage` | `dev`, `prod` | Lifecycle stage (placeholder — always `dev` for now) |
| `enercast.domain` | `wind`, `demand`, `solar` | Energy domain |
| `enercast.purpose` | `baseline`, `experiment`, … | Why the run exists |
| `enercast.backend` | `xgboost`, `autogluon`, `mlforecast` | ML backend used |
| `enercast.feature_set` | `wind_baseline`, `wind_enriched`, `wind_full`, … | Feature set the model was trained on |
| `enercast.data_resolution_min` | `10` (wind), `60` (demand), `15` (solar) | Raw data resolution |
| `enercast.horizon_steps` | integer | (child runs only) Forecast horizon in resolution steps |
| `enercast.horizon_desc` | `"10 min ahead"`, `"1h ahead"`, … | (child runs only) Human-readable horizon label |

## Troubleshooting

**"No experiments found"** — verify the tracking URI. MLflow UI defaults to
`./mlruns` if no `--backend-store-uri` flag is passed, which will appear
empty after the SQLite migration.

**Backtick parse errors on tag filters** — MLflow's filter parser requires
backticks around dotted tag names. `tags.enercast.run_type = 'parent'`
fails; ``tags.`enercast.run_type` = 'parent'`` works.

**Missing `h{n}_mae` metrics on some runs** — the parent summary metrics are
bubbled up from child runs via `client.search_runs` at the end of training.
Older runs predating this logic will not have those metrics on the parent —
look at the child runs instead, or refer to the historical CSV snapshot.
