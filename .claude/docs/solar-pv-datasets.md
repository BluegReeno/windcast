# Open Solar PV Datasets for Power Forecasting

Researched 2026-04-01. Focus: time-series power production data suitable for a solar power forecasting demo.

---

## Tier 1 — Best Candidates (production-quality, >1 year, power + irradiance)

### 1. NREL PVDAQ (PV Data Acquisition)

| Field | Value |
|-------|-------|
| URL | https://data.openei.org/submissions/4568 |
| DOI | 10.25984/1846021 |
| Contents | Power production + environmental sensors (irradiance, temp, wind, precip) at inverter/system level |
| Temporal resolution | Varies by site; typically 1-min, 5-min, or 15-min |
| Period | Multi-year per site; database spans ~2000s–present |
| Size | 44 GB (CSV) / 17 GB (Parquet); 512 GB total with 2023 Solar Data Prize subset |
| License | CC-BY 4.0 |
| Access | AWS S3 (no sign-in): `s3://oedi-data-lake/pvdaq/` |
| Format | CSV and Parquet per system |

**Why use it:** Real commercial and experimental PV sites across the US. Some sites have co-located irradiance sensors. Data includes system metadata (technology, tilt, azimuth, capacity). Analogous structure to Kelmarsh wind SCADA — inverter-level time series + site metadata.

**Gotcha:** Temporal resolution is not uniform across sites — always check per-system metadata file first. Environmental sensors are only present at some sites.

**Access pattern:**
```python
import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
# List systems
s3.list_objects_v2(Bucket="oedi-data-lake", Prefix="pvdaq/")
# Or use pvlib iotools
import pvlib
data, meta = pvlib.iotools.get_pvdaq_data(system_id=2, api_key="DEMO_KEY", years=[2011, 2012])
```

---

### 2. DKA Solar Centre — Alice Springs (DKASC)

| Field | Value |
|-------|-------|
| URL | https://dkasolarcentre.com.au/download |
| Contents | Power output per inverter/string, irradiance (GHI, POA), ambient temp, module temp, AC/DC power |
| Temporal resolution | 5-minute (some sensors at 5-second / 1-second) |
| Period | 15+ years of operation (Alice Springs site operational since ~2008) |
| Size | Multiple CSV files per system, moderate size |
| License | Free, open access (Australian Government-funded) |
| Access | Direct CSV download from website, per site |
| Format | CSV |

**Why use it:** One of the longest-running open multi-technology PV demonstration datasets. Includes multiple technologies side-by-side (mono, poly, thin-film, CPV, etc.). Very detailed per-inverter data. Hot desert climate = high irradiance variation.

**Additional datasets from same source:**
- **Yulara Solar System**: 1.8 MW plant, 5 distributed sites, since 2014
- **NT Solar Resource**: High-quality irradiance at 5-second resolution, 4 stations (Darwin, Katherine, Tennant Creek, Alice Springs), Class A instruments

**Gotcha:** Download interface is site-by-site. No bulk API. Some missing timestamps (noted on their site for DKP/BESS). Contact info@dkasolarcentre.com.au for complete datasets.

---

### 3. NREL SRRL BMS (Solar Radiation Research Laboratory — Baseline Measurement System)

| Field | Value |
|-------|-------|
| URL | https://midcdmz.nrel.gov/apps/sitehome.pl?site=BMS |
| DOI | NREL Report No. DA-5500-56488 |
| Contents | 80+ instruments: GHI, DNI, DHI, diffuse, UV, IR, temp, humidity, wind speed/dir, pressure, precipitable water vapor, aerosol optical depth |
| Temporal resolution | 1-minute |
| Period | July 15, 1981 – present |
| Size | Decades of 1-min data; monthly ASCII files |
| License | NREL data policy (free for research, citation required) |
| Access | Web interface + monthly file downloads; `pvlib.iotools.read_srml()` or `get_srml()` |
| Format | ASCII / CSV monthly files |
| Location | Golden, Colorado (39.74°N, 105.18°W, elevation 1829 m) |

**Why use it:** Gold standard for solar resource research since 1981. Irradiance-only (no PV power output), but ideal as the NWP/irradiance companion to a power model. World Meteorological Organization Baseline Surface Radiation Network member.

**Gotcha:** This is irradiance/meteorology only — no PV power output. Use it to train an irradiance-to-power model when combined with PVDAQ.

---

### 4. DWD (Deutscher Wetterdienst) 10-Minute Solar Observations

| Field | Value |
|-------|-------|
| URL | https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/solar/ |
| Contents | Global horizontal irradiance (GHI), diffuse horizontal irradiance (DHI), sunshine duration, direct normal irradiance |
| Temporal resolution | 10-minute |
| Period | ~1990–2024 per station (varies by station) |
| Size | Per-station ZIP files, ~1–10 MB each; hundreds of stations |
| License | Open data, free reuse (DWD Open Data Policy) |
| Access | Direct FTP/HTTP download, no authentication |
| Format | CSV (ZIP archives), one file per station per time period |

**Why use it:** Irradiance ground truth for German solar sites. Pairs well with any German PV production data. No PV power output but excellent irradiance quality. ~500+ stations across Germany.

**File naming pattern:** `10minutenwerte_SOLAR_{station_id}_{start}_{end}_hist.zip`

**Gotcha:** Irradiance only, no PV power output. Station coverage sparse outside Germany.

---

## Tier 2 — Useful for Specific Purposes

### 5. Pedro et al. Solar Forecasting Benchmark Dataset (UCSD Folsom)

| Field | Value |
|-------|-------|
| URL | https://zenodo.org/records/2826939 |
| DOI | 10.5281/zenodo.2826939 |
| Contents | GHI, DNI, DHI (ground measurements), sky imagery, GOES-15 satellite, NAM NWP forecasts, derived features, Python baseline forecasting scripts |
| Temporal resolution | 1-minute |
| Period | 2014–2016 (3 years) |
| Size | ~49.8 GB (dominated by sky image archives 13–18 GB/year) |
| License | CC-BY 4.0 |
| Location | Folsom, California |

**Why use it:** Specifically designed for benchmarking solar forecasting methods. Includes all three standard forecasting horizons (intra-hour, intra-day, day-ahead) with pre-computed features. Reference paper: Pedro et al., 2019, Journal of Renewable and Sustainable Energy.

**Gotcha:** Irradiance only (no PV power output). Sky image archives make it large; can download just irradiance CSVs if images not needed.

---

### 6. NSRDB (National Solar Radiation Database) via NREL API

| Field | Value |
|-------|-------|
| URL | https://nsrdb.nrel.gov / API: https://developer.nrel.gov/api/solar/nsrdb_psm3_download.json |
| Contents | GHI, DNI, DHI, temperature, wind speed, pressure, humidity, cloud type (modeled, not measured) |
| Temporal resolution | 30-minute or 1-hour (PSM3), 5-minute (some years) |
| Period | 1998–present (CONUS); extends globally for some products |
| Size | On-demand API download, per location |
| License | Free with API key (registration required), CC0-like terms |
| Access | Free API key at developer.nrel.gov, then `pvlib.iotools.get_psm3()` |

**Why use it:** Best synthetic solar resource dataset for the US. Free, easy programmatic access via pvlib. Good for generating training inputs when measured irradiance is unavailable.

**Gotcha:** Modeled data (satellite-derived), not ground-measured. Systematic bias vs. measured irradiance. Requires free NREL API key registration.

```python
import pvlib
data, meta = pvlib.iotools.get_psm3(
    latitude=37.7749, longitude=-122.4194,
    api_key="YOUR_KEY", email="your@email.com",
    names="2020", interval=30
)
```

---

### 7. Kaggle — Solar Power Generation (India, 2 plants)

| Field | Value |
|-------|-------|
| URL | https://www.kaggle.com/datasets/anikannal/solar-power-generation-data |
| Contents | Inverter-level DC/AC power, daily yield, total yield; plant-level sensor: ambient temp, module temp, irradiance |
| Temporal resolution | 15-minute (implied from 34-day coverage and file sizes) |
| Period | 34 days (May–June 2020, approximately) |
| Size | ~1.9 MB |
| License | "Data files © Original Authors" (not CC) |

**Why use it:** Clean worked example with inverter + sensor data paired. Ideal for a quick demo or tutorial.

**Gotcha:** Only 34 days — far too short for a serious forecasting model. License is ambiguous. Good for prototyping/testing pipeline logic only.

---

### 8. PVOutput.org (Community Contributions)

| Field | Value |
|-------|-------|
| URL | https://pvoutput.org |
| Contents | Per-system energy generation (Wh), power (W), temperature, voltage; crowdsourced from residential and commercial PV owners globally |
| Temporal resolution | 5–15 minutes (configurable per system) |
| Period | Varies by system; many systems have 5–10+ years of data |
| Size | Millions of systems globally |
| License | API terms of service; bulk download requires "donation" status |
| Access | REST API (60 req/h free, 300 req/h for donors); no free bulk dump |

**Why use it:** Enormous community dataset with global coverage. Some systems have very long histories.

**Gotcha:** No free bulk download. API rate limits make scraping impractical. License unclear for ML training use. Not suitable for a clean reproducible demo without manual data extraction.

---

## Tier 3 — Supplementary / Irradiance Only

### 9. NREL NSRDB Kaggle Subset (Izmir, Turkey)

| Field | Value |
|-------|-------|
| URL | https://www.kaggle.com/datasets/ibrahimkiziloklu/solar-radiation-dataset |
| Contents | GHI, DNI, DHI + meteorological data; derived from NSRDB |
| Temporal resolution | Hourly / half-hourly |
| Period | 3 years |
| Size | 6.37 MB |
| License | CC0 Public Domain |

Small, clean, ready-to-use subset. Useful for quick experiments.

---

### 10. Open Power System Data (OPSD) — European Solar Generation

| Field | Value |
|-------|-------|
| URL | https://data.open-power-system-data.org/time_series/ |
| Contents | Hourly solar generation (aggregated national/regional level), wind, load for European countries |
| Temporal resolution | 15-minute or 60-minute |
| Period | ~2006–2020 (country-dependent) |
| License | CC-BY 4.0 |
| Format | CSV, SQLite |

**Why use it:** Country-level solar generation (Germany, France, etc.) at hourly resolution. Good for national forecasting demos but too aggregated for plant-level forecasting.

**Gotcha:** Aggregated national data, not plant-level. Does not include irradiance or temperature.

---

## Recommendation for WindCast Solar Demo

For a power forecasting demo that mirrors the Kelmarsh wind pipeline:

| Priority | Dataset | Why |
|----------|---------|-----|
| Primary | **PVDAQ** (NREL, via S3/pvlib) | Real PV plant data, multi-year, CC-BY, programmatic access, some sites have irradiance |
| Weather input | **Open-Meteo** (already in codebase) | Free NWP irradiance + temperature, no key required |
| Secondary | **DKA Solar Centre** | Long time series, multi-technology, irradiance co-located |
| Benchmark only | **Pedro et al. (Zenodo 2826939)** | If irradiance forecasting benchmark needed |

**Suggested starter site from PVDAQ:** System ID 2 (NREL, Golden CO) has 15+ years of 1-min data with co-located irradiance — analogous to having a reference SCADA + met mast.

**Pipeline analogy to wind:**
```
SCADA (wind) → PVDAQ system CSV (solar)
Met mast (wind) → PVDAQ environmental sensors OR DKA NT Solar Resource (solar)
Open-Meteo wind forecast → Open-Meteo GHI/temp forecast (solar)
```

---

## Access Cheatsheet

```bash
# PVDAQ via pvlib (requires free NREL API key)
pip install pvlib boto3

# List available PVDAQ systems
import pvlib
meta = pvlib.iotools.get_pvdaq_data(system_id=2, api_key="DEMO_KEY", years=[2015])

# NSRDB irradiance for any US location
data, meta = pvlib.iotools.get_psm3(
    latitude=39.74, longitude=-105.18,
    api_key="YOUR_NREL_KEY", email="you@example.com",
    names="2020", interval=30
)

# DWD 10-min solar (Germany, no auth needed)
import requests, zipfile, io
url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/solar/recent/10minutenwerte_SOLAR_00003_akt.zip"
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
```

---

## Known Issues / Gotchas

| Issue | Detail |
|-------|--------|
| PVDAQ resolution inconsistency | Each system has its own timestep; check metadata before assuming 5-min or 15-min |
| PVDAQ environmental sensors | Only present at ~30% of sites; filter by `has_irradiance=True` in metadata |
| NREL.gov redirects | nrel.gov currently redirects to nlr.gov (DNS issue as of 2026-04-01); use OEDI/data.openei.org instead |
| DKA bulk download | No API; manual download per site per year from website |
| PVOutput license | Not open source; unsuitable for reproducible research |
| Kaggle 34-day dataset | Too short for meaningful forecasting; use only for pipeline prototyping |
| NSRDB = modeled data | Satellite-derived, not ground-measured; expect ±10–20% bias vs. pyranometer |
| Timestamps | PVDAQ is typically local time; normalize to UTC immediately (same lesson as SCADA wind) |

---

## Sources Checked

- NREL PVDAQ: data.openei.org/submissions/4568
- NREL SRRL BMS: midcdmz.nrel.gov
- NSRDB: developer.nrel.gov/docs/solar/nsrdb/
- DKA Solar Centre: dkasolarcentre.com.au
- Pedro et al. benchmark: zenodo.org/records/2826939
- DWD 10-min solar: opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/solar/
- Kaggle solar: kaggle.com/datasets/anikannal/solar-power-generation-data
- OPSD: data.open-power-system-data.org
- PVOutput API: pvoutput.org/help/api_specification.html
- Renewables.ninja: renewables.ninja (simulated output, not measured)
- Ausgrid Solar Home: URL changed/removed from Ausgrid website (as of 2026-04-01)
