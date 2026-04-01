# Open-Meteo Historical Weather API — WindCast Reference

Package: `openmeteo-requests>=1.3`, `requests-cache>=1.0`, `retry-requests>=2.0`.

---

## 1. API Endpoints

| Use case | URL |
|----------|-----|
| Historical reanalysis (ERA5) | `https://archive-api.open-meteo.com/v1/archive` |
| Weather forecast (7–16 days) | `https://api.open-meteo.com/v1/forecast` |
| Historical forecast (model runs) | `https://historical-forecast-api.open-meteo.com/v1/forecast` |

For training data: **always use the archive endpoint** (`archive-api`). It provides ERA5 reanalysis, consistent and gap-free.

---

## 2. Rate Limits (Free Tier)

- **10,000 API calls/day** (fractional counting applies).
- A request for >2 weeks of data or >10 variables counts as multiple calls proportionally.
- Example: 4 weeks of 15 variables = 3.0 calls.
- The `429` response means `Daily API request limit exceeded`.
- **Historical data does not change** — use `expire_after=-1` in cache to cache indefinitely.

---

## 3. Client Setup with Cache and Retry

```python
import openmeteo_requests
import requests_cache
from retry_requests import retry


def build_openmeteo_client(
    cache_dir: str = ".cache",
    cache_expire_after: int = -1,   # -1 = never expire (historical data is immutable)
    retries: int = 5,
    backoff_factor: float = 0.2,
) -> openmeteo_requests.Client:
    """Build a cached, auto-retrying Open-Meteo client."""
    cache_session = requests_cache.CachedSession(
        cache_dir,
        expire_after=cache_expire_after,
    )
    retry_session = retry(cache_session, retries=retries, backoff_factor=backoff_factor)
    return openmeteo_requests.Client(session=retry_session)
```

Cache is stored as a SQLite file at `{cache_dir}.sqlite`. First call fetches from API; subsequent identical calls return cached data instantly without counting against rate limits.

---

## 4. Fetching Historical Wind Data

### Parameters for Wind Power Forecasting

```python
WIND_VARIABLES = [
    "wind_speed_100m",       # m/s at hub height (typical turbine hub: 80–120m)
    "wind_direction_100m",   # degrees
    "wind_speed_10m",        # m/s (used to compute wind shear exponent)
    "wind_direction_10m",
    "temperature_2m",        # °C (air density correction)
    "pressure_msl",          # hPa (air density correction)
]
```

### Full Fetch Function

```python
import polars as pl
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry


def fetch_historical_weather(
    latitude: float,
    longitude: float,
    start_date: str,          # ISO format: "2020-01-01"
    end_date: str,            # ISO format: "2023-12-31"
    variables: list[str] | None = None,
    cache_dir: str = ".cache",
) -> pl.DataFrame:
    """
    Fetch hourly historical weather from Open-Meteo archive.

    Returns a Polars DataFrame with a UTC timestamp column and one column
    per requested variable.
    """
    if variables is None:
        variables = WIND_VARIABLES

    # Build client
    cache_session = requests_cache.CachedSession(cache_dir, expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": variables,
        "wind_speed_unit": "ms",     # m/s (default is km/h — always set explicitly)
        "timezone": "UTC",           # force UTC timestamps in response
    }

    responses = client.weather_api(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params,
    )
    response = responses[0]   # one location = one response

    # --- Extract hourly data ---
    hourly = response.Hourly()

    # Build time axis from Unix timestamps
    # hourly.Time() = start (Unix seconds), hourly.Interval() = seconds between steps
    timestamps = pl.datetime_range(
        start=pl.from_epoch(hourly.Time(), time_unit="s").replace_time_zone("UTC"),
        end=pl.from_epoch(hourly.TimeEnd(), time_unit="s").replace_time_zone("UTC"),
        interval=f"{hourly.Interval()}s",
        eager=True,
        closed="left",
    )

    # Build columns dict: variable name -> numpy array
    data: dict[str, pl.Series] = {"timestamp_utc": timestamps}
    for i, var_name in enumerate(variables):
        values = hourly.Variables(i).ValuesAsNumpy()
        data[var_name] = pl.Series(var_name, values, dtype=pl.Float32)

    return pl.DataFrame(data)
```

---

## 5. Response Object Structure

```
responses[0]                     -> WeatherApiResponse
  .Latitude() / .Longitude()     -> float
  .Elevation()                   -> float
  .Timezone()                    -> bytes (b"UTC")
  .UtcOffsetSeconds()            -> int

  .Hourly()                      -> VariablesWithTime
    .Time()                      -> int  (Unix seconds, start)
    .TimeEnd()                   -> int  (Unix seconds, end)
    .Interval()                  -> int  (seconds, e.g. 3600 for hourly)
    .VariablesLength()           -> int  (number of requested variables)
    .Variables(i)                -> VariableWithValues  (0-indexed, same order as params["hourly"])
      .Variable()                -> int  (variable enum ID)
      .ValuesAsNumpy()           -> np.ndarray[float32]
      .Value()                   -> float  (single value, for Current())
```

**Critical:** `Variables(i)` index corresponds to the order of variables in your `params["hourly"]` list. If you request `["wind_speed_100m", "temperature_2m"]`, then `Variables(0)` = wind speed, `Variables(1)` = temperature.

---

## 6. Multi-Turbine Bulk Fetch

For fetching multiple turbine locations, batch requests by passing a list of lat/lon:

```python
# NOT SUPPORTED: open-meteo processes one location per request by default
# For multiple locations, make separate requests (all cached after first run)

turbines = [
    {"id": "T01", "lat": 57.1, "lon": -2.5},
    {"id": "T02", "lat": 57.2, "lon": -2.6},
]

dfs = []
for t in turbines:
    df = fetch_historical_weather(
        latitude=t["lat"],
        longitude=t["lon"],
        start_date="2020-01-01",
        end_date="2023-12-31",
    )
    df = df.with_columns(pl.lit(t["id"]).alias("turbine_id"))
    dfs.append(df)

all_weather = pl.concat(dfs)
```

Since each (lat, lon, date_range, variables) combo is cached, re-running the script is free.

---

## 7. Handling the `429` Rate Limit Error

The `openmeteo_requests.OpenMeteoRequestsError` is raised on `400` and `429`. The retry session handles transient network errors but **not** `429` (daily limit). Handle it explicitly:

```python
from openmeteo_requests import OpenMeteoRequestsError
import time

def fetch_with_backoff(client, url, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.weather_api(url, params=params)
        except OpenMeteoRequestsError as e:
            if "429" in str(e) or "limit" in str(e).lower():
                wait = 60 * (attempt + 1)
                print(f"Rate limit hit, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Exceeded retry attempts after rate limit")
```

In practice, with `expire_after=-1` caching, you hit the rate limit only on the first cold run. Subsequent runs are free.

---

## 8. Gotchas

- **`wind_speed_unit` defaults to `km/h`** — always set `"wind_speed_unit": "ms"` explicitly for m/s output.
- **`timezone`**: if not set, response timestamps are in local time of the location. Always set `"timezone": "UTC"` to get consistent UTC output.
- **Variable order matters**: `Variables(0)`, `Variables(1)`, etc. follow the exact order of `params["hourly"]`.
- **Historical archive data starts from 1940-01-01** (ERA5 reanalysis). For dates after ~5 days ago, use the forecast endpoint.
- **ERA5 resolution**: ~9 km horizontal, 1-hour temporal. Expect systematic bias vs SCADA wind speed at hub height — plan a bias correction step.
- The `openmeteo_requests` package uses `niquests` (not `requests`) internally — `requests_cache.CachedSession` is still compatible as a session argument.
