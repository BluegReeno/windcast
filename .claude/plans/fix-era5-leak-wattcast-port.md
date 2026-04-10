# Fix ERA5 "Perfect Foresight" Leak — Port WattCast's Weather Pattern

**Status:** Ready for execution
**Created:** 2026-04-09 (Thu night)
**Deadline:** WeatherNews presentation, **Tuesday 2026-04-14** — hard deadline
**Owner:** Renaud (framework) + autonomous agent (port + runs)
**Prior work in session history (now cleared):**
- Fixed AG `tuning_data=val_pd` leak in `src/windcast/models/autogluon_model.py`
- Added regression test `test_fit_never_receives_val_as_tuning_data`
- Added `test_` metric variants to `STEPPED_METRIC_MAP` in `src/windcast/tracking/mlflow_utils.py`
- Updated `scripts/train.py` + `scripts/train_autogluon.py` to always evaluate on train/val/test
- Updated `scripts/log_tso_baseline.py` to compute both val + test metrics
- Committed ERA5 leak audit + WattCast research → this plan

---

## Why This Plan Exists

The EnerCast framework uses `src/windcast/data/open_meteo.py::fetch_historical_weather` which
hits `archive-api.open-meteo.com/v1/archive`. **That endpoint returns ERA5 reanalysis —
observed/analyzed historical weather, not forecasts.** The NWP feature joiner in
`src/windcast/features/weather.py::join_nwp_horizon_features` then shifts these values
backward by `h * resolution_minutes`, so the feature `nwp_temperature_2m_h24` at row `t`
contains the **observed** temperature at `t+24`. The model gets perfect foresight on
future weather.

This inflates every skill score + MAE number that uses `*_full` feature sets (wind AND
demand). The inflation is small at h1 (a few %) and grows with horizon. On RTE France
demand h24, it is large enough that the current "`demand_full` matches RTE's own D+1
forecast within 1.5%" claim is not honest to present — it compares a forecast made with
perfect weather foresight against RTE's real operational forecast.

This leak must be fixed before the Tuesday WeatherNews presentation.

---

## The Discovery Trail (for presentation "Incident 5")

This incident itself is presentation material and must be preserved in the git history
so the narrative survives context resets:

1. Drafted the slide deck claiming XGBoost `demand_full` h24 MAE ≈ 1,223 MW vs RTE
   1,205 MW ("match RTE within 1.5% on val set 2022-2023").
2. Claude Cowork (a reviewer) flagged that the published comparison chart showed AutoGluon
   `demand_full` h24 MAE ≈ 320 MW — a 73% advantage over XGBoost on identical features,
   statistically impossible on clean data.
3. Audit found `tuning_data=val_pd` in `autogluon_model.py`. AG was optimising stacker
   weights + ensemble selection against the caller's validation set, then we were
   reporting the metric on that same set. Classic self-fulfilling prophecy.
4. Fixed by removing `tuning_data` and keeping `use_bag_holdout=True` so AG carves its
   own tuning slice from `train_data`. Clean AG numbers became 714-1633 MW across horizons
   — roughly at parity with XGBoost (±5%).
5. While auditing the fix, a second leak was found: `archive-api.open-meteo.com` ≠ NWP
   forecast. It's ERA5 reanalysis. Every NWP feature is "future weather truth." This
   plan fixes that.
6. WattCast (Renaud's live price-forecasting system) **already solved this exact problem
   in production**. The pattern is: two separate stores, ERA5 for training-era history
   and real archived NWP forecasts (from Open-Meteo's historical-forecast-api) for recent
   horizons. This plan ports that pattern.

Mention this trail in the final commit message and in the WN presentation notes — it is
a live demonstration of the exact engineering discipline the framework is designed to
enable: surface problems fast, iterate in hours, don't ship numbers without auditing.

---

## Success Criteria

This plan succeeds when ALL of the following are true:

1. ✅ EnerCast has two weather data paths, clearly named and tested:
   - **ERA5 reanalysis** via `archive-api.open-meteo.com` → stored in `data/weather.db`
     (existing, unchanged)
   - **Historical NWP forecasts** via `historical-forecast-api.open-meteo.com` → stored
     in a new `data/weather_forecast.db`
2. ✅ `weather/provider.py` has a new `HistoricalForecastProvider` that implements the
   existing `WeatherProvider` protocol.
3. ✅ `build_features.py` accepts a `--weather-source {archive,historical_forecast,blend}`
   flag. The `blend` mode uses ERA5 for rows with `timestamp < 2022-01-01` and forecasts
   for rows `>= 2022-01-01` — this is what the RTE rebuild needs.
4. ✅ RTE France feature parquet (`data/features/rte_france_features.parquet`) rebuilt
   with `--weather-source blend`. Train period (2014-2021) still uses ERA5; val
   (2022-2023) and test (2024) use archived forecasts.
5. ✅ Kelmarsh wind feature parquet rebuilt with `--weather-source blend`. Same split
   logic (train in ERA5, val/test in forecasts) because Kelmarsh train starts in 2016
   so the bulk of the training data also predates historical-forecast coverage.
6. ✅ XGBoost retrained on RTE `demand_baseline` / `demand_enriched` / `demand_full` and
   on Kelmarsh `wind_baseline` / `wind_enriched` / `wind_full`, using the updated
   `scripts/train.py` that already logs test metrics.
7. ✅ AutoGluon retrained on `demand_full` (RTE, `good_quality` preset) and `wind_full`
   (Kelmarsh, `best_quality` preset), using the already-patched `train_autogluon.py`.
8. ✅ TSO baseline re-run via `scripts/log_tso_baseline.py` so that val + test metrics
   exist in MLflow on the same split boundaries as the XGBoost/AutoGluon runs.
9. ✅ All tests pass: `uv run pytest tests/ -q` (target: 300+ tests passing, no
   regressions from the 295 that were passing before the port).
10. ✅ `uv run ruff check src/ tests/ scripts/` and `uv run pyright src/` clean.
11. ✅ An end-of-run markdown table is written to
    `docs/WNchallenge/post-era5-fix-results.md` containing, for RTE h24:
    - XGBoost `demand_full` val MAE + test MAE
    - AutoGluon `demand_full` val MAE + test MAE
    - RTE Prévision J-1 val MAE + test MAE
    - Same shape for Kelmarsh wind h1/h6/h12/h24/h48
12. ✅ A single commit (or a small atomic series) lands the port + results under a clear
    conventional-commit message. Example: `feat(weather): port WattCast historical
    forecast provider to fix ERA5 leak`.

**Explicit non-goals for this plan:**
- Do NOT rewrite the slide deck. That is a separate next-session task after results are
  in. This plan stops at the results table.
- Do NOT touch `presentation-draft-v1.md` or any file under `docs/WNchallenge/` other
  than the new `post-era5-fix-results.md`.
- Do NOT implement solar or WattCast-as-4th-domain. Out of scope.
- Do NOT refactor the existing `weather/` module unless strictly necessary for the port.
- Do NOT try to pull ECMWF MARS / Meteomatics / Herbie. Open-Meteo historical-forecast
  is sufficient for Tuesday's deadline.

---

## Decision Log (Renaud's calls, do not revisit)

1. **Two separate SQLite databases** (`weather.db` + `weather_forecast.db`), not one
   database with a `source` column. Mirrors WattCast's `weather` / `weather_forecast`
   Supabase tables. Rationale: easier to debug, impossible to accidentally mix the two
   when querying.
2. **Copy-paste from WattCast is explicitly authorised.** Renaud built WattCast. Add a
   docstring line in each ported function: `# adapted from wattcast/src/wattcast/data/
   open_meteo.py` (or equivalent path). Do NOT reinvent. Do NOT "improve" while porting
   — verbatim port first, EnerCast-specific adjustments second.
3. **Distribution shift is accepted.** Training on ERA5 (2014-2021) and evaluating on
   archived forecasts (2022-2024) introduces a small train/eval distribution gap. This
   is MORE representative of real production than a pure ERA5-only backtest, and it is
   documented in WattCast's own docs (`docs/delivery-time-weather-features.md`) with a
   measured bias of ~1°C RMSE at D+1, ~3°C at D+7 on temperature. Mention this in the
   commit message and in the results markdown.
4. **No ECMWF MARS detour.** The library-researcher agent confirmed MARS is the gold
   standard but queue-based, GRIB, and impractical for a 5-day sprint. Log as future
   work in the results doc.

---

## Reference: WattCast Source Files to Mine

The following files in `/Users/renaud/Projects/wattcast/` are the source material for the
port. Copy the function bodies verbatim where applicable; adjust imports, types, and
naming conventions to match EnerCast (Polars instead of pandas if needed, `pl.DataFrame`
return types, EnerCast's `WeatherProvider` protocol).

| WattCast file | What to extract |
|---|---|
| `src/wattcast/data/open_meteo.py:59-92` | `fetch_historical_weather()` — already mirrored in EnerCast, verify parity only |
| `src/wattcast/data/open_meteo.py:129-164` | **`fetch_historical_forecast_weather()`** — THE KEY FUNCTION. Ports as-is into EnerCast's `open_meteo.py`. Hits `historical-forecast-api.open-meteo.com/v1/forecast`. |
| `src/wattcast/data/open_meteo.py` (top) | URL constants (`HISTORICAL_FORECAST_URL` or equivalent). Copy the constant name. |
| `src/wattcast/features/pipeline.py:178-211` | Two-stage fetch pattern — inspect to understand how WattCast blends ERA5 + forecast in the same feature matrix. EnerCast's blend mode will reproduce this. |
| `src/wattcast/features/pipeline.py:321-334` | `_join_delivery_weather()` delivery-time shift. EnerCast's `features/weather.py::join_nwp_horizon_features` already does this — just verify semantics match. |
| `src/wattcast/features/weather.py` | Spatial aggregation (capacity-weighted / population-weighted). EnerCast's `WeightedWeatherConfig` already covers this — check for discrepancies in variable naming. |
| `docs/delivery-time-weather-features.md:25` | The ~1°C D+1 / ~3°C D+7 bias note. Quote verbatim in the EnerCast results markdown. |
| `scripts/collect_weather_forecasts.py` (WattCast) | Daily-cron wrapper pattern. Not needed for this plan, but useful reference. |

**If a file path above does not exist in the actual WattCast repo, fall back to
`grep -rn fetch_historical_forecast /Users/renaud/Projects/wattcast/` and adapt.** Paths
are from a codebase snapshot and may have drifted.

---

## Phase 1 — Port the Weather Code (~1h30)

### Step 1.1 — Verify WattCast paths and read the source
- `ls /Users/renaud/Projects/wattcast/src/wattcast/data/open_meteo.py` — confirm exists
- Read `fetch_historical_forecast_weather` fully (function body + docstring + any helper)
- Read the URL constants at the top of the file
- Read `_join_delivery_weather` (or equivalent) in `features/pipeline.py`
- Read `docs/delivery-time-weather-features.md` (or grep for "delivery-time" / "perfect foresight")
- Record the exact function signature and return type in a comment for yourself —
  the port must preserve semantics

### Step 1.2 — Extend `src/windcast/data/open_meteo.py`
Add **alongside** the existing `fetch_historical_weather`:

```python
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

def fetch_historical_forecast_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    variables: list[str] | None = None,
    client: openmeteo_requests.Client | None = None,
) -> pl.DataFrame:
    """Fetch archived NWP forecast output from Open-Meteo.

    Unlike fetch_historical_weather (ERA5 reanalysis = observed weather), this
    endpoint returns the model output that was actually forecast at the time —
    stitched from the short-range portion of successive NWP runs. Available
    from 2022-01-01 for ICON, earlier for ECMWF IFS (2017+) and GFS (2021+).

    Bias vs a fully-issued D+1 forecast: ~1°C RMSE at D+1, ~3°C at D+7 on
    temperature (measured in WattCast production).

    # adapted from wattcast/src/wattcast/data/open_meteo.py::fetch_historical_forecast_weather
    """
    ...
```

Same signature shape as `fetch_historical_weather`. Match parameter names exactly. Return
`pl.DataFrame` with `timestamp_utc` + variable columns (same schema as ERA5 fetch, so
downstream code can be data-source-agnostic).

**Tests:** add to `tests/data/test_open_meteo.py`:
- `test_fetch_historical_forecast_returns_polars_schema` (mock HTTP, verify columns)
- `test_historical_forecast_url_not_archive_url` — assertion that URL constant is NOT the
  ERA5 archive URL (regression guard against reverting to the leak)
- `test_historical_forecast_respects_date_range` (mock response, verify start/end honoured)

### Step 1.3 — Extend `src/windcast/weather/provider.py`

Add a new class:

```python
class HistoricalForecastProvider:
    """Open-Meteo Historical Forecast API provider.

    Serves archived NWP forecast output (real forecasts made by ECMWF IFS / ICON /
    GFS at issue time), NOT reanalysis. Use this for val/test periods where honest
    forecast-time features are required. Coverage starts 2022-01-01.
    """
    def __init__(self, cache_dir: str = ".cache") -> None:
        self._client = build_client(cache_dir=cache_dir)

    def fetch(self, latitude, longitude, start_date, end_date, variables) -> pl.DataFrame:
        return fetch_historical_forecast_weather(
            latitude=latitude, longitude=longitude,
            start_date=start_date, end_date=end_date,
            variables=variables, client=self._client,
        )
```

Both `OpenMeteoProvider` (ERA5) and `HistoricalForecastProvider` must satisfy the same
`WeatherProvider` Protocol. Confirm via `isinstance` check in a test.

**Tests:** add to `tests/weather/test_provider.py`:
- `test_historical_forecast_provider_satisfies_protocol`
- `test_two_providers_are_distinct_instances` (no accidental shared state)

### Step 1.4 — Extend `src/windcast/weather/storage.py`

Current implementation uses a single SQLite database path. Make it parameterisable so
the caller can choose `data/weather.db` or `data/weather_forecast.db`. Do NOT add a
`source` column — two physical databases is the decision.

- Add a `db_path: Path` argument to whatever factory / upsert / query helper exists
- Default remains `data/weather.db` to preserve backwards compatibility with existing
  code paths that do not yet opt in
- Add a module-level constant `WEATHER_FORECAST_DB_PATH = Path("data/weather_forecast.db")`

**Tests:** update existing `tests/weather/test_storage.py` so it parameterises over both
paths (or at least covers the `db_path` argument path).

### Step 1.5 — Extend `src/windcast/weather/__init__.py` public API

Export:
- `HistoricalForecastProvider`
- `WEATHER_FORECAST_DB_PATH`
- Any new `get_forecast_weather()` helper analogous to the existing `get_weather()`

### Step 1.6 — `scripts/build_features.py` accepts `--weather-source`

Add a CLI flag:
```
--weather-source {archive,historical_forecast,blend}  (default: archive)
```

Behaviour:
- `archive` → existing path, no change, uses ERA5 `data/weather.db`
- `historical_forecast` → uses `HistoricalForecastProvider` + `data/weather_forecast.db`,
  raises if any requested timestamp is before 2022-01-01
- `blend` → for rows with `timestamp_utc < 2022-01-01`: use ERA5. For rows `>= 2022-01-01`:
  use historical forecasts. The join layer handles the switch; the resulting parquet
  has one weather column per variable and horizon, sourced from whichever provider is
  appropriate for that row's target time.

The `blend` mode is what RTE France and Kelmarsh rebuilds need.

**Tests:** small integration test with a stubbed provider pair, confirm the right source
is consulted for the right timestamp range.

### Step 1.7 — Run the full test suite

```
uv run pytest tests/ -q
uv run ruff check src/ tests/ scripts/
uv run pyright src/
```

Must be green before starting Phase 2.

---

## Phase 2 — Re-fetch Archived Forecasts (~30 min, mostly network)

### Step 2.1 — Populate `data/weather_forecast.db` for RTE France

Use the 8-city weighted config already defined for `rte_france` in
`src/windcast/weather/registry.py` (or wherever `WeightedWeatherConfig` lives). For EACH
of the 8 cities:

- Fetch historical forecasts from `2022-01-01` to `2024-12-31` (or the last date
  available, whichever is earlier)
- Variables: `temperature_2m`, `wind_speed_10m`, `relative_humidity_2m`
- Upsert into `data/weather_forecast.db` keyed by `(timestamp_utc, latitude, longitude)`

A simple one-off script `scripts/fetch_historical_forecasts_rte.py` is acceptable. It
can live in `scripts/` and does not need to be pretty — it must be reproducible and
leave an idempotent state in the new database.

### Step 2.2 — Populate `data/weather_forecast.db` for Kelmarsh

Kelmarsh is a single location (52.4016, -0.9436). Fetch `2022-01-01` → `2024-12-31`,
variables `wind_speed_100m`, `wind_direction_100m`, `temperature_2m`. Upsert into the
same `data/weather_forecast.db`.

### Step 2.3 — Smoke test the new database

- `sqlite3 data/weather_forecast.db "SELECT COUNT(*), MIN(timestamp_utc), MAX(timestamp_utc) FROM weather;"`
- Confirm row counts roughly match `8 cities × 3 years × 8760 hours` for RTE +
  `1 location × 3 years × 8760 hours` for Kelmarsh
- Spot-check a few timestamps against Open-Meteo's web UI

---

## Phase 3 — Rebuild Features, Retrain, Log (~1h)

### Step 3.1 — Rebuild RTE France features with blend mode

```
WINDCAST_TRAIN_YEARS=8 WINDCAST_VAL_YEARS=2 \
uv run python scripts/build_features.py \
    --domain demand --dataset rte_france \
    --weather-source blend
```

Output: `data/features/rte_france_features.parquet` with ERA5 for pre-2022 rows,
archived forecasts for 2022+. Parent run metadata should include the source mix (add
an informational log line).

### Step 3.2 — Rebuild Kelmarsh features with blend mode

```
uv run python scripts/build_features.py \
    --domain wind --dataset kelmarsh \
    --weather-source blend
```

Note: Kelmarsh's default split (`train_years=5`, `val_years=1`) produces train_end
around 2021-05. Almost all training rows are pre-2022 → ERA5. Val + test are 2022+ →
archived forecasts.

### Step 3.3 — Retrain XGBoost on RTE (sequential, fast)

```
WINDCAST_TRAIN_YEARS=8 WINDCAST_VAL_YEARS=2 \
    uv run python scripts/train.py --domain demand --dataset rte_france \
        --feature-set demand_baseline && \
WINDCAST_TRAIN_YEARS=8 WINDCAST_VAL_YEARS=2 \
    uv run python scripts/train.py --domain demand --dataset rte_france \
        --feature-set demand_enriched && \
WINDCAST_TRAIN_YEARS=8 WINDCAST_VAL_YEARS=2 \
    uv run python scripts/train.py --domain demand --dataset rte_france \
        --feature-set demand_full
```

~1 min total. The updated `train.py` already logs val + test metrics.

### Step 3.4 — Retrain XGBoost on Kelmarsh

```
uv run python scripts/train.py --turbine-id kwf1 --feature-set wind_baseline && \
uv run python scripts/train.py --turbine-id kwf1 --feature-set wind_enriched && \
uv run python scripts/train.py --turbine-id kwf1 --feature-set wind_full
```

### Step 3.5 — Re-run TSO baseline with new splits

```
WINDCAST_TRAIN_YEARS=8 WINDCAST_VAL_YEARS=2 \
    uv run python scripts/log_tso_baseline.py
```

(TSO values are invariant to weather source — RTE's forecast is in the raw `rte_france.parquet`
regardless. But the run must exist with the 8/2 split for chart consistency.)

### Step 3.6 — Retrain AutoGluon on RTE (good_quality, ~5 min)

```
WINDCAST_TRAIN_YEARS=8 WINDCAST_VAL_YEARS=2 \
    uv run python scripts/train_autogluon.py \
        --domain demand --dataset rte_france \
        --feature-set demand_full \
        --presets good_quality --time-limit 180
```

### Step 3.7 — Retrain AutoGluon on Kelmarsh (best_quality, ~30 min, background OK)

```
uv run python scripts/train_autogluon.py \
    --domain wind --dataset kelmarsh --turbine-id kwf1 \
    --feature-set wind_full \
    --presets best_quality --time-limit 360
```

Run this in background (`run_in_background: true`). While it runs, move on to Phase 4.

### Step 3.8 — Regenerate comparison charts

```
uv run python scripts/compare_runs.py --experiment enercast-rte_france
uv run python scripts/compare_runs.py --experiment enercast-kelmarsh
```

This refreshes `reports/comparison_enercast-*.png`. Include both val and test bars if
`compare_runs.py` supports it — if not, leave a TODO and do not block on it.

### Step 3.9 — Write the results markdown

Create `docs/WNchallenge/post-era5-fix-results.md` with:

1. **Context paragraph** (3-5 lines): why we re-ran everything, what the two leaks were,
   what was fixed.
2. **RTE France demand results table** — one row per horizon, columns = XGBoost val MAE,
   XGBoost test MAE, AG val MAE, AG test MAE, persistence test MAE, TSO test MAE.
3. **Kelmarsh wind results table** — same shape, no TSO column.
4. **"What changed vs the pre-fix numbers"** — short commentary comparing to the tables
   in STATUS.md.
5. **Methodology note** — ERA5 for train 2014-2021, historical-forecast for 2022+,
   ~1°C D+1 / ~3°C D+7 bias quote from WattCast's delivery-time-weather-features doc.
6. **Next steps** — line item "update `docs/WNchallenge/presentation-draft-v1.md`
   slides 6/7/8/9 with the new numbers" (explicitly deferred to the next session).

Pull all numbers by querying MLflow directly via sqlite3 on `mlflow.db`, not by scraping
the training script logs. MLflow is the source of truth.

---

## Phase 4 — Commit & Handoff (~20 min)

### Step 4.1 — Final validation

```
uv run pytest tests/ -q
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run pyright src/
```

All green.

### Step 4.2 — Commit

One commit per logical chunk is fine. Suggested atomic breakdown:

1. `feat(weather): add HistoricalForecastProvider for archived NWP forecasts` — Phase 1
   code only, tests, no data.
2. `feat(features): add --weather-source blend mode to build_features` — Phase 1.6 +
   relevant tests.
3. `feat(data): populate weather_forecast.db from Open-Meteo for RTE + Kelmarsh` — the
   fetch script + the generated database (if size permits; otherwise just the script,
   git-ignore the db).
4. `feat(eval): re-run XGBoost/AG/TSO with forecast-time weather` — this is the results
   commit, includes `post-era5-fix-results.md` and updated MLflow charts.

All commit messages must end with the trailer:
```
Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
```

Do NOT push to remote. Do NOT touch the presentation draft. Leave those to Renaud.

### Step 4.3 — Handoff note

Append a single paragraph to the end of `docs/WNchallenge/post-era5-fix-results.md`
titled `## Handoff` that tells next-session-Claude exactly what to do next:

> The ERA5 leak is fixed and clean results are in the table above. Next session should:
> (1) read `presentation-draft-v1.md`, (2) update slides 6, 7, 8, 9 with the new numbers
> from the table above, (3) add a "Methodology" caveat box on slide 9 using the
> WattCast-attributed ~1°C D+1 bias quote, (4) mention "Incident 5 — discovered in
> session, fixed in hours" as an optional addition to slide 3 WattCast incidents. Keep
> the rest of the deck as-is.

---

## Risks & Mitigations

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| 1 | Open-Meteo historical-forecast-api down or rate-limited | Low | `openmeteo-requests` has retries; cache is local. If hard failure, fall back to ERA5 everywhere and write a disclaimer in the results doc. |
| 2 | Coverage gap for one of the 8 RTE cities before 2022 | Medium | `blend` mode handles this — pre-2022 rows always use ERA5. No special handling needed. |
| 3 | Distribution shift between ERA5-train and forecast-eval hurts accuracy beyond acceptable | Medium | Expected and documented. If RTE `demand_full` h24 MAE exceeds 1,800 MW (significantly worse than RTE TSO 1,428), note in results and recommend slide 9 pivots to a "framework delivers a solid starting point, tuning is where experts shine" narrative. Do NOT hide the numbers. |
| 4 | WattCast source paths drifted from the snapshot in this plan | Low | Explicit grep fallback instructions in the reference table above. |
| 5 | Tests break during port | Medium | Fix them before moving to Phase 2. Do not leave red tests in the tree. |
| 6 | AG best_quality wind rerun takes >45 min | Low | If the clock is past 02:00 local and it's still running, kill it and fall back to `good_quality` — document in the results doc with a footnote. |
| 7 | `dynamic_stacking` kills AG fit under new conditions | Medium | Existing fix kept `use_bag_holdout=True` which carves its own holdout and sidesteps dynamic_stacking pitfalls. Verify on first AG run before launching the long wind one. |
| 8 | Token budget blows up on the next session before Phase 4 finishes | Low | The plan is self-contained and sequential. If context fills, `/clear` and resume from the last unchecked step — TASKS.md tracks progress. |

---

## What This Plan Is NOT

- NOT a slide rewrite (that's the next session after this one)
- NOT a solar or WattCast-as-4th-domain implementation
- NOT an ECMWF MARS integration
- NOT a refactor of the weather module beyond the port surface
- NOT a change to XGBoost or AutoGluon hyperparameters
- NOT a split-boundary change (keep `WINDCAST_TRAIN_YEARS=8 WINDCAST_VAL_YEARS=2` for RTE,
  defaults 5/1 for Kelmarsh)

The goal is narrow: **honest test-set numbers for the Tuesday WeatherNews presentation**.

---

## Memory / Feedback to Save on Completion

At the end of a successful run, append one feedback memory update via the memory system
at `/Users/renaud/.claude/projects/-Users-renaud-Projects-windcast/memory/`:

- Either update `feedback_verify_extraordinary_results.md` (already exists from session
  history) with a note that the ERA5 leak audit was the second catch of the same day.
- Or create a new `project_wattcast_as_dogfood.md` memory noting that WattCast's
  production patterns are the canonical reference for EnerCast — "when in doubt, check
  what WattCast does."

Do NOT invent new feedback types. Use the existing memory system at the path above.
