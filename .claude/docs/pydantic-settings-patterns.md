# Pydantic Settings v2 Patterns — WindCast Reference

pydantic-settings version: 2.13.1, pydantic version: 2.12.5.

---

## 1. Basic `BaseSettings` with `WINDCAST_` prefix

```python
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WindCastSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WINDCAST_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",   # WINDCAST_MLFLOW__TRACKING_URI -> mlflow.tracking_uri
        case_sensitive=False,        # WINDCAST_LOG_LEVEL = windcast_log_level
        extra="ignore",              # don't error on unknown env vars
    )

    # Paths — resolved at validation time, not runtime
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")

    # Runtime config
    log_level: str = "INFO"
    n_jobs: int = -1               # -1 = all CPUs (scikit-learn convention)
    random_seed: int = 42

    @property
    def kelmarsh_dir(self) -> Path:
        return self.raw_dir / "kelmarsh"

    @property
    def hill_of_towie_dir(self) -> Path:
        return self.raw_dir / "hill_of_towie"
```

**Loading:**
```python
settings = WindCastSettings()         # reads .env + env vars
settings = WindCastSettings(_env_file=".env.test")   # override env file
```

---

## 2. Nested Settings

Use `BaseModel` (not `BaseSettings`) for nested groups. The parent `BaseSettings` handles env loading.

```python
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class MLflowConfig(BaseModel):
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "windcast"
    artifact_location: str | None = None


class OpenMeteoConfig(BaseModel):
    cache_dir: str = ".cache"
    cache_expire_after: int = -1     # -1 = never expire (historical data doesn't change)
    retries: int = 5
    backoff_factor: float = 0.2


class WindCastSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WINDCAST_",
        env_nested_delimiter="__",
    )

    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    openmeteo: OpenMeteoConfig = Field(default_factory=OpenMeteoConfig)
    data_dir: Path = Path("data")
    log_level: str = "INFO"
```

**Environment variable mapping with nested delimiter `__`:**
```
WINDCAST_MLFLOW__TRACKING_URI=http://localhost:5000
WINDCAST_MLFLOW__EXPERIMENT_NAME=kelmarsh_v2
WINDCAST_OPENMETEO__CACHE_EXPIRE_AFTER=3600
WINDCAST_DATA_DIR=/mnt/datasets
```

---

## 3. Path Fields with Validation

Pydantic v2 resolves `Path` fields as-is (no auto-mkdir). Add a validator to create dirs:

```python
from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class WindCastSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WINDCAST_")

    data_dir: Path = Path("data")
    processed_dir: Path = Path("data/processed")

    @field_validator("data_dir", "processed_dir", mode="after")
    @classmethod
    def ensure_dir_exists(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v
```

**Warning:** Do not use `ensure_dir_exists` on `raw_dir` — raw data dirs should already exist; creating them silently hides misconfiguration.

---

## 4. `.env` File Example

```bash
# .env  (do not commit — add to .gitignore)
WINDCAST_DATA_DIR=data
WINDCAST_RAW_DIR=data/raw
WINDCAST_PROCESSED_DIR=data/processed
WINDCAST_LOG_LEVEL=DEBUG
WINDCAST_RANDOM_SEED=42
WINDCAST_MLFLOW__TRACKING_URI=file:./mlruns
WINDCAST_MLFLOW__EXPERIMENT_NAME=windcast_kelmarsh
WINDCAST_OPENMETEO__CACHE_EXPIRE_AFTER=-1
```

---

## 5. Singleton Pattern for Scripts

```python
# src/windcast/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class WindCastSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WINDCAST_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> WindCastSettings:
    return WindCastSettings()
```

```python
# Usage in any script or module
from windcast.config import get_settings

cfg = get_settings()
print(cfg.raw_dir / "kelmarsh")
```

`lru_cache` ensures the `.env` file is read only once. In tests, call `get_settings.cache_clear()` before each test that sets env vars.

---

## 6. Gotchas

- `env_prefix` applies to top-level fields only, not nested model fields. Nested fields are accessed via `__` delimiter after the group name: `WINDCAST_MLFLOW__TRACKING_URI`, not `WINDCAST_MLFLOW_TRACKING_URI`.
- `Path` fields accept both strings and `Path` objects from env vars — pydantic coerces `str` to `Path` automatically.
- `case_sensitive=False` (default) means `WINDCAST_LOG_LEVEL` and `windcast_log_level` are equivalent.
- `.env` file takes lower priority than actual environment variables. Setting `WINDCAST_LOG_LEVEL=INFO` in shell overrides `.env`.
- `extra="ignore"` prevents `ValidationError` when other `WINDCAST_*` env vars exist that are not defined in the model.
