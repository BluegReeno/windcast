# Spain Energy Dataset — nicholasjhana (Kaggle)

**Kaggle URL:** https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather  
**License:** CC0 Public Domain  
**Source data verified from:** https://github.com/TBorges99/Electricity_Price_Forecast_-ARIMA_SVR_XGBOOST- (contains the actual CSV files)

---

## Download

```bash
# Requires kaggle CLI + ~/.kaggle/kaggle.json credentials
pip install kaggle
kaggle datasets download -d nicholasjhana/energy-consumption-generation-prices-and-weather --unzip -p data/raw/spain/
```

The zip contains exactly **2 CSV files**:
- `energy_dataset.csv` (~6 MB)
- `weather_features.csv` (~19 MB)

---

## File 1: energy_dataset.csv

### Structure
- **Rows:** 35,064 (header + 35,064 data rows — exactly 4 years × 8,766 hours)
- **Columns:** 29
- **Time period:** 2015-01-01 00:00:00+01:00 → 2018-12-31 23:00:00+01:00
- **Granularity:** Hourly
- **No duplicate timestamps**

### Timestamp column
| Column | Type | Format | Notes |
|--------|------|--------|-------|
| `time` | string → datetime | `2015-01-01 00:00:00+01:00` | Europe/Madrid timezone with UTC offset (+01:00 winter, +02:00 summer). Must parse with `pd.to_datetime(df['time'], utc=True)` |

### Generation columns (MW, float64)
All generation values are in **megawatts (MW)**.

| Column | Null count | Always zero? | Notes |
|--------|-----------|--------------|-------|
| `generation biomass` | 19 | No | Useful |
| `generation fossil brown coal/lignite` | 18 | No | Useful |
| `generation fossil coal-derived gas` | 18 | **YES — all zeros** | Drop |
| `generation fossil gas` | 18 | No | Useful |
| `generation fossil hard coal` | 18 | No | Useful |
| `generation fossil oil` | 19 | No | Useful |
| `generation fossil oil shale` | 18 | **YES — all zeros** | Drop |
| `generation fossil peat` | 18 | **YES — all zeros** | Drop |
| `generation geothermal` | 18 | **YES — all zeros** | Drop |
| `generation hydro pumped storage aggregated` | **35,064 (100%)** | — | **ENTIRELY NULL — Drop** |
| `generation hydro pumped storage consumption` | 19 | No | Useful |
| `generation hydro run-of-river and poundage` | 19 | No | Useful |
| `generation hydro water reservoir` | 18 | No | Useful |
| `generation marine` | 19 | **YES — all zeros** | Drop |
| `generation nuclear` | 17 | No | Useful |
| `generation other` | 18 | No | Useful |
| `generation other renewable` | 18 | No | Useful |
| `generation solar` | 18 | No | Useful |
| `generation waste` | 19 | No | Useful |
| `generation wind offshore` | 18 | **YES — all zeros** | Drop (Spain has no offshore) |
| `generation wind onshore` | 18 | No | **Key target/feature** |

### Forecast columns (MW, float64)
| Column | Null count | Notes |
|--------|-----------|-------|
| `forecast solar day ahead` | 0 | TSO day-ahead solar forecast |
| `forecast wind offshore eday ahead` | **35,064 (100%)** | **ENTIRELY NULL — Drop** (typo in name too) |
| `forecast wind onshore day ahead` | 0 | TSO day-ahead wind forecast |
| `total load forecast` | 0 | TSO day-ahead load forecast |

### Load and price columns
| Column | Type | Null count | Range | Units |
|--------|------|-----------|-------|-------|
| `total load actual` | float64 | 36 | 18,041 – 41,015 | MW |
| `price day ahead` | float64 | 0 | 2.06 – 101.99 | EUR/MWh |
| `price actual` | float64 | 0 | 9.33 – 116.80 | EUR/MWh |

### Loading pattern
```python
import polars as pl

energy = pl.read_csv(
    "data/raw/spain/energy_dataset.csv",
    try_parse_dates=False,  # handle manually — timezone-aware string
)
# Parse timestamp with timezone
energy = energy.with_columns(
    pl.col("time").str.to_datetime(format="%Y-%m-%d %H:%M:%S%z", use_earliest=True)
)

# Drop useless columns (all-null or all-zero)
DROP_ENERGY = [
    "generation fossil coal-derived gas",
    "generation fossil oil shale",
    "generation fossil peat",
    "generation geothermal",
    "generation hydro pumped storage aggregated",
    "generation marine",
    "generation wind offshore",
    "forecast wind offshore eday ahead",
]
energy = energy.drop(DROP_ENERGY)
```

---

## File 2: weather_features.csv

### Structure
- **Rows:** 178,396 (5 cities × ~35,679 rows each — counts differ slightly per city)
- **Columns:** 17
- **Time period:** 2015-01-01 00:00:00+01:00 → 2018-12-31 23:00:00+01:00
- **Granularity:** Hourly per city
- **Format:** Long (one row per city per hour — NOT wide)

### Cities
| city_name value | Row count | Note |
|----------------|-----------|------|
| ` Barcelona` | 35,476 | **Leading space in name — must strip** |
| `Bilbao` | 35,951 | |
| `Madrid` | 36,267 | |
| `Seville` | 35,557 | |
| `Valencia` | 35,145 | |

Cities do NOT have exactly 35,064 rows each — row counts differ slightly. When joining to energy_dataset, pivot/aggregate first.

### Columns
| Column | Type | Units | Notes |
|--------|------|-------|-------|
| `dt_iso` | string → datetime | — | Same format as `time` in energy_dataset: `2015-01-01 00:00:00+01:00` |
| `city_name` | string | — | Strip whitespace — Barcelona has leading space |
| `temp` | float64 | **Kelvin** | Subtract 273.15 for Celsius |
| `temp_min` | float64 | **Kelvin** | Often == `temp` (hourly snap) |
| `temp_max` | float64 | **Kelvin** | Often == `temp` (hourly snap) |
| `pressure` | float64 | hPa | **Barcelona has 45 extreme outliers** (values ~100,000–1,008,371). These appear to be data errors, not Pa units (10,084 hPa would still be wrong). Filter pressure > 1100 |
| `humidity` | float64 | % (0–100) | 63 rows with humidity=0 (suspicious) |
| `wind_speed` | float64 | m/s | **3 outliers in Valencia > 50 m/s** (max=133 m/s — clearly erroneous). Filter > 50 m/s |
| `wind_deg` | float64 | degrees (0–360) | |
| `rain_1h` | float64 | mm | 10.9% of rows non-zero |
| `rain_3h` | float64 | mm | Cumulative over 3h |
| `snow_3h` | float64 | mm | 0.1% non-zero |
| `clouds_all` | float64 | % (0–100) | Cloud cover |
| `weather_id` | int | — | OpenWeatherMap condition code |
| `weather_main` | string | — | Categories: clear, clouds, rain, mist, fog, drizzle, thunderstorm, haze, dust, snow, smoke, squall |
| `weather_description` | string | — | Detailed text description |
| `weather_icon` | string | — | OWM icon code (e.g. "01n") |

No null values in weather_features.csv.

### Loading pattern
```python
import polars as pl

weather = pl.read_csv("data/raw/spain/weather_features.csv")

# Parse timestamp
weather = weather.with_columns(
    pl.col("dt_iso").str.to_datetime(format="%Y-%m-%d %H:%M:%S%z", use_earliest=True),
    pl.col("city_name").str.strip_chars(),  # fix Barcelona leading space
    # Convert Kelvin to Celsius
    (pl.col("temp") - 273.15).alias("temp_c"),
    (pl.col("temp_min") - 273.15).alias("temp_min_c"),
    (pl.col("temp_max") - 273.15).alias("temp_max_c"),
)

# Filter outliers
weather = weather.filter(
    (pl.col("pressure").is_between(900, 1100)) &
    (pl.col("wind_speed") <= 50)
)
```

### Pivot to wide format for joining with energy_dataset
```python
# Create city-prefixed columns, then join to energy on timestamp
weather_wide = weather.pivot(
    values=["temp_c", "wind_speed", "wind_deg", "humidity", "pressure", "clouds_all", "rain_1h"],
    index="dt_iso",
    columns="city_name",
    aggregate_function="mean",  # handles the slight count differences
)
```

---

## Joining Both Files

```python
combined = energy.join(
    weather_wide,
    left_on="time",
    right_on="dt_iso",
    how="left",
)
# Expected: 35,064 rows
```

---

## Data Quality Summary

### Critical issues (will break modeling if ignored)

| Issue | File | Column | Fix |
|-------|------|--------|-----|
| 6 columns entirely null or zero | energy_dataset | `generation hydro pumped storage aggregated`, `forecast wind offshore eday ahead` + 4 always-zero generation cols | Drop |
| Barcelona leading space | weather_features | `city_name` | `.str.strip_chars()` |
| Barcelona pressure outliers | weather_features | `pressure` | Filter: keep 900–1100 hPa |
| Valencia wind speed outliers | weather_features | `wind_speed` | Filter: keep ≤ 50 m/s |
| Temperature in Kelvin | weather_features | `temp`, `temp_min`, `temp_max` | Subtract 273.15 |
| Cities have unequal row counts | weather_features | — | Use `aggregate_function="mean"` when pivoting |

### Minor issues (low impact)

| Issue | File | Column | Note |
|-------|------|--------|------|
| ~18–19 NaN per generation column | energy_dataset | most generation cols | ~0.05% — safe to forward-fill |
| 36 NaN in total load actual | energy_dataset | `total load actual` | Linear interpolation |
| 63 rows humidity=0 | weather_features | `humidity` | Plausible in dry conditions — leave |
| Timezone-aware timestamps | both | `time` / `dt_iso` | UTC offsets vary (+01/+02) — normalize to UTC on load |

---

## Relevance to WindCast

This dataset is useful for:
- **Grid-level wind generation forecasting** (not turbine-level SCADA)
- `generation wind onshore` (MW, national total Spain) = target variable
- `forecast wind onshore day ahead` = TSO benchmark to beat
- Weather features (5 cities) serve as NWP proxies
- Price signals useful for market-aware forecasting experiments

**Not suitable for:** individual turbine performance modeling, power curve analysis, wake effects, or anything requiring spatial resolution below national level.

**Time period:** 4 full years (2015–2018), 35,064 hourly observations.
