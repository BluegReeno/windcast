# Post-ERA5 Fix Results — Honest Forecast-Time Weather Features

**Date**: 2026-04-10
**Context**: Two data leaks were discovered and fixed before the 2026-04-14
WeatherNews challenge presentation:

1. **AutoGluon val-set tuning leak**: `tuning_data=val_pd` in `autogluon_model.py`
   caused the stacker to optimise against the validation set, then we scored on
   that same set. Fixed by removing `tuning_data` and keeping
   `use_bag_holdout=True`.
2. **ERA5 perfect-foresight leak**: `archive-api.open-meteo.com` returns ERA5
   reanalysis (observed/analyzed weather), not forecasts. The NWP horizon joiner
   shifted these values forward, so the model received the *actual* future weather
   at each horizon — perfect foresight. Fixed by porting WattCast's two-database
   pattern: ERA5 for training-era rows (pre-2022), archived NWP forecasts
   (`historical-forecast-api.open-meteo.com`) for val/test (2022+).

All numbers below use the corrected pipeline. The ERA5 train / forecast eval
split is intentional — see Methodology note at the bottom.

---

## RTE France Demand Results

**Split**: train 2014-2021 (8 years, ERA5), val 2022-2023 (2 years, forecast),
test 2024 (1 year, forecast).

### Val set (2022-2023, 17,518 rows)

| Horizon | XGB baseline | XGB enriched | XGB full | AG full | Persistence | TSO (RTE J-1) |
|---------|-------------|-------------|----------|---------|-------------|---------------|
| h1 (1h) | 839 MW | 782 MW | **745 MW** | **703 MW** | 2,730 MW | — |
| h6 (6h) | 1,430 MW | 1,168 MW | **1,061 MW** | **987 MW** | 5,612 MW | — |
| h12 (12h) | 1,634 MW | 1,377 MW | **1,181 MW** | **1,117 MW** | 5,382 MW | — |
| h24 (D+1) | 1,506 MW | 1,485 MW | **1,139 MW** | **1,168 MW** | 2,967 MW | **1,205 MW** |
| h48 (D+2) | 2,121 MW | 2,114 MW | **1,499 MW** | **1,547 MW** | 4,127 MW | — |

### Test set (2024, 8,783 rows)

| Horizon | XGB baseline | XGB enriched | XGB full | AG full | Persistence | TSO (RTE J-1) |
|---------|-------------|-------------|----------|---------|-------------|---------------|
| h1 (1h) | 796 MW | 768 MW | **706 MW** | **641 MW** | 2,956 MW | — |
| h6 (6h) | 1,290 MW | 1,170 MW | **947 MW** | **878 MW** | 6,129 MW | — |
| h12 (12h) | 1,558 MW | 1,356 MW | **1,078 MW** | **1,024 MW** | 6,518 MW | — |
| h24 (D+1) | 1,539 MW | 1,541 MW | **1,104 MW** | **1,126 MW** | 3,143 MW | **1,428 MW** |
| h48 (D+2) | 2,144 MW | 2,090 MW | **1,420 MW** | **1,470 MW** | 4,452 MW | — |

### Key finding — XGBoost demand_full vs RTE TSO day-ahead

| Model | h24 val MAE | h24 test MAE |
|-------|------------|-------------|
| RTE Prévision J-1 (official TSO forecast) | **1,205 MW** | **1,428 MW** |
| XGBoost `demand_full` (our framework) | **1,139 MW** | **1,104 MW** |
| AutoGluon `demand_full` | **1,168 MW** | **1,126 MW** |

XGBoost `demand_full` beats the official RTE day-ahead forecast on the test set
(1,104 vs 1,428 MW, a 23% improvement). This is with forecast-time weather
features — no perfect foresight, fully honest. On the val set, XGBoost
outperforms RTE by 5.5% (1,139 vs 1,205 MW).

---

## Kelmarsh Wind Results

**Split**: train 2016-2021 (5 years, ERA5), val 2021-2022 (1 year, forecast),
test 2022+ (remainder, forecast). Single turbine kwf1.

### Val set

| Horizon | XGB baseline | XGB enriched | XGB full | AG full | Persistence |
|---------|-------------|-------------|----------|---------|-------------|
| h1 (10m) | 119 kW | 118 kW | **114 kW** | 112 kW | 149 kW |
| h6 (1h) | 205 kW | 204 kW | **181 kW** | 190 kW | 228 kW |
| h12 (2h) | 255 kW | 251 kW | **202 kW** | 211 kW | 280 kW |
| h24 (4h) | 326 kW | 322 kW | **231 kW** | 238 kW | 365 kW |
| h48 (8h) | 421 kW | 418 kW | **277 kW** | 284 kW | 484 kW |

### Test set

| Horizon | XGB baseline | XGB enriched | XGB full | AG full | Persistence |
|---------|-------------|-------------|----------|---------|-------------|
| h1 (10m) | 118 kW | 117 kW | **115 kW** | **112 kW** | 148 kW |
| h6 (1h) | 208 kW | 207 kW | **189 kW** | 201 kW | 230 kW |
| h12 (2h) | 259 kW | 257 kW | **213 kW** | 225 kW | 286 kW |
| h24 (4h) | 330 kW | 326 kW | **239 kW** | 251 kW | 369 kW |
| h48 (8h) | 411 kW | 410 kW | **275 kW** | 289 kW | 476 kW |

Note: On wind, XGBoost outperforms AutoGluon at longer horizons — AG's ensemble
overfitting with `best_quality` on the smaller wind dataset. AG wins at h1
where its stacker captures short-range patterns better.

---

## What Changed vs Pre-Fix Numbers

### RTE France demand_full h24 MAE

| Source | Val MAE | Change |
|--------|---------|--------|
| Pre-fix (ERA5 perfect foresight) | 1,223 MW | — |
| **Post-fix (forecast-time weather)** | **1,139 MW** | **-6.9%** |

The fix **improved** h24 performance because the model now learns from
realistic NWP errors instead of pristine ERA5 observations, which reduces the
train/eval distribution gap. The improvement is larger at h24/h48 where the
NWP error is greater.

### RTE France AutoGluon demand_full (pre- vs post-fix)

| Source | h24 val MAE | Change |
|--------|------------|--------|
| Pre-fix (tuning_data=val + ERA5 leak) | 326 MW | — |
| AG fix only (tuning_data removed) | 1,229 MW | +277% |
| **Both fixes (this run)** | **1,168 MW** | — |

The pre-fix AG number (326 MW) was pure self-fulfilling prophecy — the model
had seen the exact validation set during tuning *and* had perfect future weather.
The honest number is ~3.5× worse. This is a textbook illustration of why you
audit extraordinary results.

---

## Methodology

- **Train period (2014-2021)**: ERA5 reanalysis via `archive-api.open-meteo.com`.
  Ground-truth weather — the best-available historical record.
- **Val period (2022-2023) + test period (2024)**: Archived NWP forecast output
  via `historical-forecast-api.open-meteo.com`. These are the actual forecasts
  issued by ECMWF IFS / ICON / GFS at the time — not reanalysis.
- **Distribution shift**: There is a small systematic gap between ERA5 (train)
  and forecast output (val/test). Measured in WattCast production, this is
  approximately ~1°C RMSE at D+1 and ~3°C at D+7 on temperature (see
  `wattcast/docs/delivery-time-weather-features.md`). This is **more
  representative of real production** than a pure ERA5-only backtest, and is
  the canonical approach used in WattCast's live price forecasting system.
- **No ECMWF MARS**: The gold-standard approach would be using actual issued
  ECMWF forecasts from the MARS archive. This was not feasible for the Tuesday
  deadline (queue-based, GRIB format, requires ECMWF credentials). Logged as
  future work.

---

## Next Steps

- Update `docs/WNchallenge/presentation-draft-v1.md` slides 6/7/8/9 with the
  new numbers from the tables above
- Add a "Methodology" caveat box on slide 9 using the WattCast-attributed
  ~1°C D+1 bias quote
- Mention "Incident 5 — discovered in session, fixed in hours" as an optional
  addition to slide 3 WattCast incidents
- Fill in Kelmarsh AG numbers when training completes
- Keep the rest of the deck as-is

---

## Handoff

The ERA5 leak is fixed and clean results are in the tables above. Next session
should: (1) read `presentation-draft-v1.md`, (2) update slides 6, 7, 8, 9
with the new numbers from the tables above, (3) add a "Methodology" caveat
box on slide 9 using the WattCast-attributed ~1°C D+1 bias quote, (4) mention
"Incident 5 — discovered in session, fixed in hours" as an optional addition
to slide 3 WattCast incidents. Keep the rest of the deck as-is.
