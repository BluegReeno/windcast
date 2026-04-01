# WindCast — Dataset Catalog & Selection

**Date:** 2026-03-31
**Sources:** WRAG wiki (groups.io/g/wrag/wiki/13236), Zenodo, NREL, BSH, web research
**Purpose:** Comprehensive evaluation of open datasets for wind power forecasting ML

---

## 1. Selected Datasets (5 datasets, 3 categories)

### 1.1 SCADA — Wind + Production (ML training & evaluation)

#### Kelmarsh v4 (PRIMARY)

| Property | Value |
|----------|-------|
| Turbines | 6 × Senvion MM92 (2.05 MW each, 12.3 MW total) |
| Location | Northamptonshire, UK |
| Period | 2016–2024 (9 years) |
| Resolution | 10-minute |
| Size | 3.8 GB (26 files) |
| Format | CSV/ZIP (annual per turbine group) + KMZ + signal mapping CSV |
| License | CC-BY-4.0 |
| DOI | 10.5281/zenodo.16807551 |
| URL | https://zenodo.org/records/16807551 |
| Status | Online — **v4 published August 2025** |
| Signals | Active power, wind speed (hub), wind direction, rotor speed, pitch angle, nacelle direction, generator torque, grid frequency, temperatures (gearbox, generator, ambient), status/event codes, substation PMU meter data, grid/fiscal meter readings |

**Why primary:** Most cited in wind ML literature, 9-year coverage enables seasonal learning, well-documented signal mapping, manageable size (6 turbines), same OEM as Penmanshiel for cross-site tests.

**Gotchas:**
- v4 is the latest — earlier versions (0.0.3, 0.0.4) only cover 2016–2021
- Signal naming follows Greenbyte secondary SCADA conventions

---

#### Hill of Towie (GENERALIZATION — different OEM)

| Property | Value |
|----------|-------|
| Turbines | 21 × Siemens SWT-2.3-VS-82 (2.3 MW each) |
| Location | Aberdeenshire, Scotland |
| Period | January 2016 — August 2024 (8.5 years) |
| Resolution | 10-minute statistics |
| Size | 12.6 GB |
| Format | CSV (annual ZIP archives) + description CSVs |
| License | CC-BY-4.0 |
| DOI | 10.5281/zenodo.14870023 |
| URL | https://zenodo.org/records/14870023 |
| Status | Online |
| Signals | Active power, rotor speed, grid measurements (voltage, current, frequency), alarm log (fault codes with descriptions) |

**Why included:** Different OEM (Siemens vs Senvion) proves dataset-agnostic design. 21 turbines at one site, 8.5 years, CC-BY-4.0. Replaces La Haute Borne in generalization path.

**Gotchas:**
- Timestamps are **end-of-period** — shift by -10 min when aligning with NWP forecasts
- AeroUp retrofit creates a performance discontinuity per turbine — installation dates provided, treat as covariate or split training
- 12.6 GB requires Polars lazy scanning (`pl.scan_csv()` not `pl.read_csv()`)
- Download description CSVs first (`Hill_of_Towie_turbine_fields_description.csv`, `Hill_of_Towie_grid_fields_description.csv`) to understand signal names before pulling full data
- Wind speed signal availability needs verification from description files

---

#### Penmanshiel v3 (CROSS-SITE — same OEM as Kelmarsh)

| Property | Value |
|----------|-------|
| Turbines | 13 × Senvion MM82 (WT03 absent from dataset) |
| Location | Scottish Borders, UK |
| Period | 2016–2024 (9 years) |
| Resolution | 10-minute |
| Size | ~7.5 GB (27 files) |
| Format | CSV (annual ZIP) + KMZ + XLSX signal mapping |
| License | CC-BY-4.0 |
| DOI | 10.5281/zenodo.16807304 |
| URL | https://zenodo.org/records/16807304 |
| Status | Online — **v3 published August 2025** |
| Signals | Same categories as Kelmarsh (same OEM, same Greenbyte SCADA); PMU data for 2023–2024 only |

**Why included:** Same OEM as Kelmarsh → validates that parser works across sites with minimal changes. Cross-references Kelmarsh on Zenodo (both from Cubico Sustainable Investments Ltd).

---

### 1.2 Wind-Only Offshore (NWP validation)

#### FINO1

| Property | Value |
|----------|-------|
| Type | Offshore meteorological mast (100 m lattice) |
| Location | North Sea, 54°00.9'N 06°35.3'E (45 km NW of Borkum, Germany) |
| Water depth | 30 m |
| Period | 2003–present (22+ years) |
| Resolution | 10-minute averages (some 1 Hz turbulence channels) |
| Heights | Cup anemometers at ~33, 40, 50, 60, 70, 80, 90 m ASL |
| Signals | Wind speed & direction (multi-height), turbulence intensity, air temperature, relative humidity, barometric pressure, precipitation, solar radiation, UV; oceanographic: waves, water level, currents, SST, salinity |
| License | German federal open data (free) |
| Access | BSH-Login registration (free) at https://login.bsh.de |
| Portal | BSH Insitu Portal + Seegangsportal (waves) |
| Contact | fino@bsh.de |

**Why included:** Reference offshore dataset for Europe. 22+ years of multi-height wind profiles — ideal for NWP bias correction, wind shear calibration, atmospheric stability features. No equivalent in terms of duration and height coverage.

**Gotchas:**
- Heights vary between FINO1/2/3 — always verify metadata
- Gaps due to marine fouling, storm damage, sensor replacement — quality flags provided
- Old URL `fino.bsh.de` redirects to BSH MARNET/FINO page
- PANGAEA may host derived FINO datasets separately from BSH operational archive

**Also available:** FINO2 (Baltic Sea, 2007–present), FINO3 (North Sea, 2009–present) — same access process

---

### 1.3 Wind-Only Onshore (Pipeline testing & wind features)

#### NWTC M2 Mast (NREL)

| Property | Value |
|----------|-------|
| Type | 82 m meteorological tower |
| Location | Flatirons Campus, Colorado, USA (39°54'38"N, 105°14'5"W, 1855 m elev.) |
| Period | September 1996–present (30 years) |
| Resolution | **1-minute** averages (2-second raw sampling) |
| Heights | Wind speed & direction at 2, 5, 10, 20, 50, 80 m |
| Signals | Wind speed (mean/peak/std per height), wind direction, temperature (2/50/80 m), dew point, relative humidity, barometric pressure, precipitation, solar radiation; derived: turbulence intensity, Richardson number, friction velocity, power-law shear coefficient |
| Format | CSV daily files |
| License | US public domain |
| Access | No registration required |
| URL | https://midcdmz.nrel.gov/apps/sitehome.pl?site=NWTC |
| API | https://midcdmz.nrel.gov/apps/data_api_doc.pl?NWTC |
| DOI | 10.5439/1052222 |

**Why included:** Zero-friction access, API-available, 30-year record, 1-minute resolution. Ideal for building and testing the wind feature pipeline before handling SCADA. 6 heights enable wind shear profiling.

**Gotchas:**
- Solar radiation data unreliable before June 2024
- Mountain foothill terrain — not representative of flat/offshore conditions
- 80 m max height: insufficient for modern 150+ m hub heights, but covers older turbines and validates shear extrapolation

---

## 2. Generalization Path (revised)

```
Phase 1:  Kelmarsh v4           → prove the pipeline works (primary SCADA)
Phase 2a: Hill of Towie         → different OEM (Siemens vs Senvion), 21 turbines
Phase 2b: Penmanshiel v3        → same OEM, different site (cross-site validation)
Phase 2c: FINO1 + NWTC M2       → NWP validation, wind shear features
```

**Rationale for replacing La Haute Borne with Hill of Towie:**
- ENGIE portal (`opendata-renewables.engie.com`) is offline as of March 2026
- Hill of Towie: more turbines (21 vs 4), longer period (8.5 vs 3+ years), different OEM (Siemens vs unknown), reliable access (Zenodo vs ENGIE portal), CC-BY-4.0 license

---

## 3. Evaluated but Not Selected

### Tier 2 — Useful for specific purposes

| Dataset | Turbines | Period | Why not primary | When to use |
|---------|----------|--------|-----------------|-------------|
| **Brazilian UEPS+UEBB** | 52 (2 farms, Enercon) | 2013-2014 (1 yr) | Too short for seasonal training | Geographic diversity test, NetCDF4 ingestion test |
| **SMARTEOLE** | 7 × Senvion MM82 | 3 months (2020) | Too short | Wake steering, 1-min power curve analysis |
| **SDWPF** | 134 turbines (China) | ~6 months | **CC BY-NC-ND** license blocker | Benchmark only, spatial/wake modeling |
| **Cabauw/CESAR** | Met mast 213m, 6 heights | 1973–present | Portal currently offline | Contact KNMI (datacentrum@knmi.nl) — best European onshore validation if accessible |
| **Tall Tower Dataset** | 181 global towers | Varies | CC BY-NC, heterogeneous | Geographic diversity for wind shear |
| **FINO2** | Offshore Baltic | 2007–present | FINO1 sufficient for POC | Add for Baltic Sea validation |
| **FINO3** | Offshore North Sea | 2009–present | FINO1 sufficient for POC | Cross-platform comparison |

### Tier 3 — Verified dead ends (March 2026)

| Source | Issue |
|--------|-------|
| La Haute Borne (ENGIE) | Portal `opendata-renewables.engie.com` — ECONNREFUSED |
| EDP Open Data | Portal `opendata.edp.com` — ECONNREFUSED |
| Ørsted wind data | Page returns HTTP 403 |
| DTU data portal | `data.dtu.dk` returns 403 on all datasets |
| NREL direct URLs | Redirecting to `nlr.gov` (DNS issue) |
| IEA Wind Tasks | No public operational datasets — modeling tools only |

---

## 4. WRAG Catalog — Full Inventory (for future reference)

### SCADA datasets from WRAG wiki

| Dataset | Source | Notes |
|---------|--------|-------|
| Kelmarsh | Zenodo 16807551 | **Selected** |
| Penmanshiel | Zenodo 16807304 | **Selected** |
| Hill of Towie | Zenodo 14870023 | **Selected** |
| La Haute Borne | ENGIE OpenData | Portal down |
| EDP Renewables | opendata.edp.com | Portal down — 3 anonymized farms when available |
| SMARTEOLE | Zenodo 7342466 | 7 turbines, 3 months, wake steering |
| SDWPF | figshare/Baidu | 134 turbines, CC BY-NC-ND |
| Levenmouth | ORE Catapult POD | 1 turbine, 574 sensors, 1 Hz — cost-based access |
| Brazil Coastal | Zenodo 1475197 | 52 turbines (2 farms), 1 year, NetCDF4, CC BY-SA |
| WILLOW-Norther | Zenodo 11093262 | Offshore, virtual sensing validation |
| AERIS AWIT | awit.aeris-data.fr | Valorem FR, gated access — submit request |
| Alpha Ventus (RAVE) | rave-offshore.de | Research access, German offshore |
| Turkey SCADA | Kaggle | 1 turbine, minimal signals |
| Kaggle CC0 | Kaggle | 1 turbine, 2.5 years, CC0 |
| Vestas V100 | Kaggle | Power-wind-blade load |
| Anholt/WMR (Ørsted) | NDA required | Proprietary |
| Horns Reef etc. (DTU) | NDA required | Via DTU gitlab |
| AWAKEN | DOE Wind Data Hub | US multi-farm, data use agreement |

### Wind measurement datasets from WRAG wiki (selected)

| Dataset | Type | Location | Heights | Period | Access |
|---------|------|----------|---------|--------|--------|
| **FINO1/2/3** | Offshore mast | North/Baltic Sea | 33-100 m | 2003+ | BSH Login (free) |
| **NWTC M2** | Onshore mast 82m | Colorado, USA | 2-80 m | 1996+ | Open, API |
| **Cabauw/CESAR** | Onshore mast 213m | Netherlands | 10-200 m | 1973+ | KNMI registration (portal down) |
| NYSERDA FLS | Floating LiDAR | NY coast, USA | Multi-height | Recent | Open |
| Belgium Offshore | Various | North Sea | Varies | Various | Portal |
| Ørsted atm | Offshore masts | DK/UK | Hub-height | Multi-year | 403 currently |
| Iowa Atm. Obs. | Onshore | Iowa, USA | Tower | Ongoing | Open |
| DTU ALEX17 | Complex terrain | Navarra, Spain | 40-100 m | 2017 | Contact DTU |
| Frøya | Onshore coastal | Norway | Tower | Multi-year | Zenodo (1 Hz available) |
| Tall Tower Dataset | Global 181 masts | Various | >10 m | Varies | B2SHARE, CC BY-NC |

### Reanalysis & NWP (for features, not training truth)

| Source | Resolution | Period | Access | Notes |
|--------|-----------|--------|--------|-------|
| **Open-Meteo** | ~11 km (ECMWF IFS) | Forecast + historical | Free API | Primary NWP source for WindCast |
| **ERA5** | 31 km, hourly | 1940–present | Copernicus CDS (free) | Reanalysis baseline, 100 m wind |
| NORA3 | 3 km | Norway region | THREDDS | High-res Norwegian reanalysis |
| COSMO-R6G2 | 6 km | Germany/Europe | DWD OpenData | ERA5-forced regional reanalysis |
| New European Wind Atlas | 3 km | Europe (statistics) | NEWA portal | Mesoscale climatology |

---

## 5. Access Action Items

| Priority | Action | Effort |
|----------|--------|--------|
| **Now** | Download Kelmarsh v4 from Zenodo | 5 min |
| **Now** | Bookmark NWTC M2 API endpoint | 2 min |
| **This week** | Register BSH-Login for FINO1 | 15 min |
| **This week** | Download Hill of Towie description CSVs (small) to verify signals | 10 min |
| **Later** | Download Penmanshiel v3 when Phase 2b starts | 5 min |
| **If needed** | Email KNMI (datacentrum@knmi.nl) for Cabauw access | 10 min |
| **If needed** | Submit AERIS AWIT access request | 15 min |
