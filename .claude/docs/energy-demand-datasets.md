# Open Energy Demand Datasets for Forecasting

Curated list of freely available time-series energy demand datasets suitable for demand
forecasting demos, particularly those that pair well with weather features.

Last updated: 2026-04-01

---

## Tier 1 — Best for Demand Forecasting Demo (weather-paired, multi-year, easy access)

### 1. GEFCom2014 Load Forecasting Track

| Field | Detail |
|-------|--------|
| Source | http://blog.drhongtao.com/2017/03/gefcom2014-load-forecasting-data.html |
| Download | Dropbox ZIP (linked from above blog post) |
| Region | New England ISO (US Northeast) |
| Resolution | Hourly |
| Period | ~7 years (roughly 2005–2011) |
| Variables | Hourly load (MW) + hourly temperature (°C) |
| Size | ~10 MB unzipped |
| License | Academic use (cite Hong et al., IJF 2016) |
| Format | CSV, structured in 15 task subfolders |

**Why it's good**: Gold standard benchmark dataset. Already includes matching temperature.
Used in hundreds of published papers — baselines are well-documented.

**Gotcha**: Not permissively licensed. Fine for research/demo, not commercial redistribution.
Use GEFCom2014-L_V2.zip specifically (V2 fixes data issues in original release).

---

### 2. Kaggle — Hourly Energy + Weather (Spain, ENTSO-E sourced)

| Field | Detail |
|-------|--------|
| Source | https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather |
| Region | Spain |
| Resolution | Hourly |
| Period | 4 years (~2015–2018) |
| Variables | Actual consumption, TSO forecast, generation by type, electricity price, weather (5 cities) |
| Size | ~4 MB zipped |
| License | CC0 (public domain) |
| Format | 2 CSV files: energy data + weather data |

**Why it's good**: Already merged demand + weather in one clean package. TSO forecast included
so you can compare your model against the official baseline. CC0 = no restrictions.

**Gotcha**: Weather is city-level averages, not grid-level. Spain only.

---

### 3. Open Power System Data (OPSD) — Time Series Package

| Field | Detail |
|-------|--------|
| Source | https://data.open-power-system-data.org/time_series/ |
| Region | 37 European countries (ENTSO-E coverage) |
| Resolution | Hourly (also 15-min and 30-min files available) |
| Period | 2015–2020 |
| Variables | Load (consumption), wind generation, solar generation, electricity prices |
| Size | ~200 MB (full CSV), SQLite and XLSX also available |
| License | CC BY 4.0 |
| Format | CSV, XLSX, SQLite |

**Why it's good**: Pre-cleaned, cross-country, well-documented. Best multi-country European
option. No weather data embedded but pairs naturally with Open-Meteo (same time range).

**Gotcha**: Last update was ~2020. For 2021+ data you need to go directly to ENTSO-E API.
Data gaps exist for some countries/years.

---

### 4. EIA Hourly Electric Grid Monitor (US)

| Field | Detail |
|-------|--------|
| Source | https://www.eia.gov/electricity/gridmonitor/ |
| API | https://www.eia.gov/opendata/ (free API key, register by email) |
| Region | All US balancing authorities (PJM, MISO, CAISO, ERCOT, etc.) |
| Resolution | Hourly |
| Period | July 2015 – present (near real-time, ~1h lag) |
| Variables | Demand, net generation, interchange, generation by fuel type |
| Size | Varies by region/period; CSV and JSON download |
| License | US government open data (public domain) |
| Format | CSV, JSON, XLSX, API |

**Why it's good**: The most comprehensive US demand dataset. Covers all major grid operators.
Near real-time updates. Free API with no quota (just needs registration).

**Gotcha**: No weather data bundled — must join with NOAA or Open-Meteo.
Historical depth varies: some series start 2015, others 2018.

---

### 5. PJM Hourly Energy Consumption (Kaggle mirror)

| Field | Detail |
|-------|--------|
| Source | https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption |
| Official source | https://dataminer2.pjm.com/feed/hrl_load_metered/definition |
| Region | PJM sub-regions (AEP, COMED, DAYTON, DEOK, DOM, DUQ, EKPC, FE, NI, PJME, PJMW...) |
| Resolution | Hourly |
| Period | 2001–2018 (varies by sub-region) |
| Variables | Hourly load in MW |
| Size | ~25 MB |
| License | CC0 on Kaggle mirror |
| Format | CSV (one file per sub-region) |

**Why it's good**: Long history (15+ years for some regions), easy Kaggle download, no
registration. PJME (Eastern PJM) file alone has 145k+ hourly rows.

**Gotcha**: No weather data included. Some sub-regions have gaps. For fresh data go to
PJM DataMiner2 directly.

---

## Tier 2 — Good but More Niche / Setup Required

### 6. French RTE éCO2mix

| Field | Detail |
|-------|--------|
| Source | https://odre.opendatasoft.com/explore/dataset/eco2mix-national-tr/api/ |
| API | https://data.rte-france.com (free registration) |
| Region | France (mainland) |
| Resolution | 15 minutes (quarters of an hour) |
| Period | 2012 – present (national); 2013 – present (regional breakdown) |
| Variables | Actual consumption, J-1 and J forecasts, production by source, cross-border flows, CO2 |
| License | Open License v2.0 (Etalab) |
| Format | CSV, JSON, XLSX via ODRÉ or RTE API |

**Why it's good**: 15-min resolution is rare. Official TSO forecast included for benchmarking.
Regional data allows sub-national analysis.

**Gotcha**: API quota of 50,000 calls/month per user. Bulk historical download easier via
ODRÉ than via RTE API directly.

---

### 7. UK NESO Historic Demand Data

| Field | Detail |
|-------|--------|
| Source | https://www.neso.energy/data-portal/historic-demand-data |
| Region | Great Britain (National Grid ESO / NESO) |
| Resolution | Half-hourly (30 minutes) |
| Period | 2001 – present (updated ~6h lag) |
| Variables | Electricity demand, interconnector flows, hydro storage pumping, wind/solar outturn |
| License | NESO Open Data Licence |
| Format | CSV (one file per year), API via api.neso.energy |

**Why it's good**: 25-year history is exceptional. Half-hourly resolution is industry standard
for GB grid. No weather data but pairs well with Met Office or Open-Meteo for same period.

**Gotcha**: GB-specific. Demand definition changed over time (embedded generation treatment).

---

### 8. ENTSO-E Transparency Platform (Direct API)

| Field | Detail |
|-------|--------|
| Source | https://transparency.entsoe.eu/ |
| API Guide | https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html |
| Python client | https://github.com/EnergieID/entsoe-py |
| Region | All of Europe (37 countries, bidding zones) |
| Resolution | Hourly (or 15-min depending on country) |
| Period | 2014 – present |
| Variables | Actual load, load forecast, generation by type, prices, cross-border flows |
| License | Free, requires API key (email transparency@entsoe.eu) |
| Format | XML/CSV via API, or bulk CSV via File Library |

**Why it's good**: Primary source for all European electricity data. OPSD (above) is derived
from this. Direct access gives you fresher, country-specific data.

**Gotcha**: Requires free API key (email request, not instant). XML format needs parsing.
Use entsoe-py to avoid XML pain. Data quality varies by country TSO.

---

### 9. AEMO NEM Aggregated Demand Data (Australia)

| Field | Detail |
|-------|--------|
| Source | https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data |
| Python tool | https://github.com/ADGEfficiency/nem-data |
| Region | Australia NEM (QLD, NSW, VIC, SA, TAS) |
| Resolution | 5 minutes (trading intervals) |
| Period | 1998 – present |
| Variables | Operational demand, spot price, generation dispatch by technology |
| License | AEMO Terms of Use (free, open) |
| Format | CSV (MMSDM monthly archives) |

**Why it's good**: 5-min resolution — finer than most other sources. Longest open history
(27 years). Strong for forecasting competition research.

**Gotcha**: AEMO's file structure is complex (MMSDM archives). Use NEMOSIS or nem-data
Python packages to avoid manual CSV wrangling.

---

### 10. National Gas Transmission Data Portal (UK Gas)

| Field | Detail |
|-------|--------|
| Source | https://data.nationalgas.com/ |
| Region | Great Britain (NTS — National Transmission System) |
| Resolution | Daily (some data at hourly granularity) |
| Period | Several years of history (JS-rendered portal, exact range varies by dataset) |
| Variables | Gas demand forecast, actual flows, supply/demand balance, storage, linepack |
| License | Open (GSO licence mandate) |
| Format | CSV download from portal |

**Why it's good**: Only major open source for GB gas demand. Regulatory mandate means data
will stay available.

**Gotcha**: Portal requires JavaScript (no direct CSV bulk download URL). Gas demand is
highly weather-sensitive — temperature is the dominant feature, which makes it an
interesting forecasting target. Daily resolution limits ML model sophistication.

---

### 11. ENTSOG Transparency Platform (European Gas)

| Field | Detail |
|-------|--------|
| Source | https://transparency.entsog.eu/ |
| API | https://transparency.entsog.eu/api/v1/operationaldata |
| Region | Europe-wide (NTS entry/exit points) |
| Resolution | Daily |
| Period | 2010 – present |
| Variables | Gas flows at entry/exit points, capacity bookings, nominations |
| License | Free public access |
| Format | CSV via API or download tool |

**Why it's good**: Only pan-European gas flow dataset. Covers all major pipelines.

**Gotcha**: Entry/exit point flows are not the same as end-consumer demand. Requires
aggregation and interpretation. Daily resolution. Country-level demand inference is indirect.

---

### 12. UCI Individual Household Electric Power Consumption

| Field | Detail |
|-------|--------|
| Source | https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption |
| Kaggle mirror | https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set |
| Region | Single household, Sceaux, France |
| Resolution | 1 minute |
| Period | December 2006 – November 2010 (47 months) |
| Variables | Global active power, reactive power, voltage, intensity, 3 sub-metering circuits |
| Size | ~20 MB unzipped (2,075,259 rows) |
| License | CC BY 4.0 |
| Format | Semi-colon delimited CSV |

**Why it's good**: Finest temporal resolution of any public dataset (1-min). Good for
sub-metering disaggregation, short-term forecasting, anomaly detection demos.

**Gotcha**: Single household — not grid-level demand. No weather data. 1.25% missing values.
Voltage fluctuations make it harder to interpret than pure consumption data.

---

## Recommendation Matrix

| Use Case | Best Dataset |
|----------|-------------|
| Quick demo, weather already included | Kaggle Spain (ENTSO-E) — CC0, 4yr, ready to use |
| Academic benchmark with known baselines | GEFCom2014 — gold standard, citable |
| Multi-country European analysis | OPSD Time Series — CC BY 4.0, pre-cleaned |
| US regional/grid-level | EIA Hourly Grid Monitor or PJM Kaggle mirror |
| Long historical depth (25 years) | UK NESO Historic Demand |
| 15-min or sub-hourly resolution | RTE éCO2mix (15-min) or AEMO NEM (5-min) |
| Gas demand forecasting | UK National Gas Portal (daily) |
| Fine-grained building/household | UCI Power Consumption (1-min) |

---

## Weather Pairing

None of these datasets (except the Kaggle Spain set) include weather. Recommended pairing:

- **Open-Meteo** (https://open-meteo.com/) — free, no API key, hourly ERA5 reanalysis back
  to 1940. Already used in WindCast for NWP data. Easiest pairing for any dataset.
- **NOAA ISD** — US weather station data, free, pairs well with EIA/PJM.
- **ECMWF ERA5** via Copernicus — best quality reanalysis, requires free registration.

For demand forecasting, the minimum weather features that matter:
1. Temperature (dominant driver — heating/cooling degree days)
2. Hour of day + day of week (calendar effects)
3. Public holidays (binary flag)
4. Wind speed (heating demand, not just generation)

---

## Quick Start Code Pattern

```python
# Fastest path to a working demo: Kaggle Spain dataset
# Download from: https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather

import polars as pl

energy = pl.read_csv("energy_dataset.csv", try_parse_dates=True)
weather = pl.read_csv("weather_features.csv", try_parse_dates=True)

# Join on time column
df = energy.join(weather, on="time", how="inner")

# Target: "total load actual" column
# Key features: temp, hour, weekday, month, lag features
```

```python
# GEFCom2014 pattern
import polars as pl

df = pl.read_csv("GEFCom2014-L_V2/Task 1/Train.csv")
# Columns: ZONEID, TIMESTAMP, LOAD, T (temperature)
# ZONEID ranges 1-25 (zone 25 = aggregate of all zones)
```

```python
# EIA API pattern (free key required)
import requests

API_KEY = "your_key_here"
url = (
    "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    f"?api_key={API_KEY}"
    "&frequency=hourly"
    "&data[0]=value"
    "&facets[respondent][]=PJM"
    "&facets[type][]=D"   # D = demand
    "&start=2023-01-01"
    "&end=2024-01-01"
    "&sort[0][column]=period&sort[0][direction]=asc"
    "&length=5000"
)
data = requests.get(url).json()
```
