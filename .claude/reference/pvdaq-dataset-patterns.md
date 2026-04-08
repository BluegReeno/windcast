# PVDAQ Dataset — Implementation Reference

## Critical Correction: "System 2" Is NOT NREL Golden

System 2 in PVDAQ is **"Residential 1a" in Lakewood, CO** — a 2.9 kW residential system.
It is NOT at NREL Golden. QA status: **fail** (insufficient data duration).

The NREL research systems at Golden, CO are different IDs (see below).

---

## Recommended NREL Golden Systems (QA: pass)

| system_id | Name | Channels | Date Range | Key Signals |
|-----------|------|----------|------------|-------------|
| **1283** | NREL Research Support Facility II | 25 | 1997-02 to 2024-05 | AC power (metered kW), POA, ambient_temp, wind_speed |
| **1332** | NREL Parking Garage | 26 | 2013-03 to 2024-05 | AC power, POA |
| **50** | NREL x-Si 6 | 23 | 1994-05 to 2025-07 | AC power, POA, ambient_temp, module_temp |
| **51** | NREL x-Si 7 | 23 | 1994-05 to 2025-07 | AC power, POA, ambient_temp, module_temp |
| **4** | NREL x-Si -1 | 15 | 2007-08 to 2025-07 | AC power, POA, ambient_temp, module_temp |
| **10** | NREL CIS -1 | 14 | 2006-01 to 2025-07 | AC power, POA, ambient_temp, module_temp |

**Best choice for a complete solar schema: system_id=1283** (has wind_speed, two POA sensors, metered AC power).
**Best for long history: system_id=50 or 51** (1994–2025, 1-min resolution).

---

## Data Access — NO pvlib Required

### pvlib `get_pvdaq_data()` status
The NREL PVDAQ v3 REST API has been **decommissioned**. `pvlib.iotools.get_pvdaq_data()` calls that API and is **broken**. pvlib is also not in this project's dependencies.

### Direct S3 Access (recommended)

**Public bucket, no AWS credentials needed.** Region: `us-west-2`.

Base URL: `https://oedi-data-lake.s3.amazonaws.com/`

#### CSV format (wide table — easier to parse)

```
s3://oedi-data-lake/pvdaq/csv/pvdata/system_id={id}/year={Y}/month={M}/day={D}/system_{id}__date_{YYYY}_{MM}_{DD}.csv
```

Example:
```
https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/pvdata/system_id%3D4/year%3D2020/month%3D1/day%3D23/system_4__date_2020_01_23.csv
```

Note: `=` must be URL-encoded as `%3D` in HTTP paths.

#### Parquet format (long/normalized — metric_id + value)

```
s3://oedi-data-lake/pvdaq/parquet/pvdata/system_id={id}/year={Y}/month={M}/day={D}/system_{id}__date_{YYYY}_{MM}_{DD}.snappy.000.parquet
```

#### Metadata files

```
# List all systems with QA status, date ranges, channel counts
https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/systems_20250729.csv

# Metric ID -> sensor_name, units lookup (per system)
https://oedi-data-lake.s3.amazonaws.com/pvdaq/parquet/metrics/metrics__system_{id}__part000.parquet
```

---

## File Format Details

### CSV format (wide)
- **Column `measured_on`**: local time (no TZ info, UTC offset in systems CSV)
- **Column `utc_measured_on`**: NOT present in CSV — only in parquet
- **Remaining columns**: `{sensor_name}__{metric_id}` (e.g., `ac_power__315`)
- **Column `system_id`**: repeated in every row
- Time resolution: **1-minute** (confirmed for systems 2, 4, 50, 1283)

```
measured_on,ac_current__319,ac_power__315,ac_voltage__318,ambient_temp__320,...,system_id
2020-01-23 09:50:00,1.251,139.35,121.60,-0.19,...,4
```

### Parquet format (long/normalized)
Schema: `measured_on (datetime[ns]), utc_measured_on (datetime[ns]), metric_id (int32), value (float64)`

- Has `utc_measured_on` — use this for timestamp_utc
- Must join with metrics table to get sensor names
- One row per metric per timestamp → pivot needed

---

## System 1283 — Signals for Canonical Solar Schema

```
metric_id | sensor_name           | units | maps_to
----------|----------------------|-------|--------
1137      | ac_power              | W     | power_kw (÷1000)
1040      | ac_power_metered_kW   | W     | power_kw (preferred, metered)
1055      | POA_irradiance        | W/m^2 | poa_wm2
1054      | POA_irradiance_refcell| W/m^2 | (secondary POA)
1053      | ambient_temp          | C     | ambient_temp_c
1056      | module_temp           | C     | module_temp_c
1051      | wind_speed            | m/s   | wind_speed_ms
```

**No GHI sensor in PVDAQ** — GHI is not available for any system.
Must be derived from POA + geometry or fetched from external source (SRRL/NREL or Open-Meteo).

## System 4 — Signals (simpler, also good)

```
metric_id | sensor_name     | units | maps_to
----------|----------------|-------|--------
315       | ac_power        | W     | power_kw (÷1000)
313       | poa_irradiance  | W/m^2 | poa_wm2
320       | ambient_temp    | C     | ambient_temp_c
321       | module_temp_1   | C     | module_temp_c
```

No wind_speed for system 4.

## System 2 — Signals (DO NOT USE — QA:fail, residential, Lakewood)

```
metric_id | sensor_name     | units
345       | poa_irradiance  | W/m^2
346       | dc_power        | W     ← DC only, no AC
349       | module_temp_1   | C
```

---

## Parser Implementation Pattern (Polars, no pvlib)

```python
import polars as pl
import urllib.request
import io

def _s3_url(system_id: int, year: int, month: int, day: int) -> str:
    """Build HTTPS URL for PVDAQ CSV file."""
    base = "https://oedi-data-lake.s3.amazonaws.com"
    path = (
        f"/pvdaq/csv/pvdata/system_id%3D{system_id}"
        f"/year%3D{year}/month%3D{month}/day%3D{day}"
        f"/system_{system_id}__date_{year:04d}_{month:02d}_{day:02d}.csv"
    )
    return base + path


def fetch_pvdaq_day_csv(system_id: int, year: int, month: int, day: int) -> pl.DataFrame:
    """Fetch one day of PVDAQ data as wide CSV into a Polars DataFrame."""
    url = _s3_url(system_id, year, month, day)
    with urllib.request.urlopen(url) as f:
        return pl.read_csv(f.read())


def build_canonical(df: pl.DataFrame, system_id: int, utc_offset_hours: int) -> pl.DataFrame:
    """Map PVDAQ system 1283 wide CSV to canonical solar schema."""
    return (
        df
        .with_columns([
            # measured_on is local time — convert to UTC
            (
                pl.col("measured_on")
                .str.to_datetime("%Y-%m-%d %H:%M:%S")
                - pl.duration(hours=utc_offset_hours)
            ).alias("timestamp_utc"),
            pl.lit(system_id).alias("system_id"),
            pl.lit("pvdaq").alias("dataset_id"),
            # ac_power_metered_kW column is in Watts — divide by 1000
            (pl.col("ac_power_metered_kw__1040") / 1000.0).alias("power_kw"),
            pl.col("poa_irradiance__1055").alias("poa_wm2"),
            pl.lit(None).cast(pl.Float64).alias("ghi_wm2"),  # not available
            pl.col("ambient_temp__1053").alias("ambient_temp_c"),
            pl.col("module_temp__1056").alias("module_temp_c"),
            pl.col("wind_speed__1051").alias("wind_speed_ms"),
            pl.lit(0).alias("qc_flag"),
        ])
        .select([
            "timestamp_utc", "dataset_id", "system_id",
            "power_kw", "ghi_wm2", "poa_wm2",
            "ambient_temp_c", "module_temp_c", "wind_speed_ms",
            "qc_flag",
        ])
    )
```

### UTC offset for Golden, CO
From `systems_20250729.csv`: `timezone_or_utc_offset = 7` → UTC-7 (Mountain Standard Time).
Note: the field stores the offset as a positive integer meaning "UTC minus N hours".

---

## Known Gotchas

### 1. Column names embed metric_id
CSV columns are `{sensor_name}__{metric_id}`. The suffix changes across systems.
Build column maps from the metrics parquet per system rather than hardcoding names.

```python
# Load metric_id -> sensor_name for a system
metrics_url = f"https://oedi-data-lake.s3.amazonaws.com/pvdaq/parquet/metrics/metrics__system_{system_id}__part000.parquet"
```

### 2. Parquet is long format — pivot required
Parquet has `metric_id` + `value` (EAV schema). Must pivot on `measured_on` + `utc_measured_on`.
Use the CSV format for simpler parsing unless you need `utc_measured_on` from the file itself.

### 3. No GHI in any PVDAQ system
GHI is not available. Use Open-Meteo or NREL SRRL BMS data as external source:
- NREL SRRL BMS (Golden CO): https://data.nrel.gov/submissions/7

### 4. Measured_on is local time in CSV
UTC offset is in the systems metadata file (`timezone_or_utc_offset` field).
For Golden CO: UTC-7 (MST) or UTC-6 (MDT) — the CSV does not indicate DST transitions.
**Use parquet format if UTC is critical** — it includes `utc_measured_on`.

### 5. Power units: Watts, not kW
`ac_power` columns are in Watts. Divide by 1000 for kW. Exception: system 1283 has
`ac_power_metered_kW` which confusingly is ALSO in Watts (sensor name is misleading, units field says "W").

### 6. Negative power values at night
Power readings go slightly negative at night (inverter standby current). Apply `power_kw.clip(lower=0)`.

### 7. Missing days
Not every day has a file. Wrap fetch in try/except and return empty DataFrame.
System 2 has gaps in 2015, 2018 (missing years entirely in the S3 partitioning).

### 8. S3 URL encoding
The `=` in `system_id=4` must be `%3D` in HTTP URLs. Python's `urllib.parse.quote` handles this.

### 9. System 1283 end date is 2024-05
NREL RSF II data stopped in May 2024. For active systems, use system 4 or 10 (data through 2025).

---

## Listing Available Systems

```python
import polars as pl

# Download systems catalog
systems_url = "https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/systems_20250729.csv"
systems = pl.read_csv(systems_url)

# Filter Golden CO, QA pass
golden = systems.filter(
    pl.col("site_location").str.contains("Golden") &
    (pl.col("qa_status") == "pass")
).select([
    "system_id", "system_public_name", "available_sensor_channels",
    "first_timestamp", "last_timestamp", "timezone_or_utc_offset"
])
```

---

## Dependency Requirements

No pvlib needed. Only stdlib + project dependencies:
- `polars` — already in pyproject.toml
- `pyarrow` — already installed (23.0.1) as polars dependency
- `urllib.request` — stdlib (for simple fetching)
- `requests` — already in pyproject.toml (for more robust fetching with retries)
