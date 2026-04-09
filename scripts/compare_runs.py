"""Compare MLflow parent runs across backends/feature sets.

Queries parent runs (``tags.enercast.run_type = 'parent'``) from one or more
experiments, extracts per-horizon MAE and skill scores, renders grouped bar
charts to ``reports/`` as PNGs, and prints a Markdown comparison table to
stdout (ready to paste into slides or STATUS.md).

Usage:
    uv run python scripts/compare_runs.py --experiment enercast-kelmarsh
    uv run python scripts/compare_runs.py \
        --experiment enercast-kelmarsh \
        --output reports/kelmarsh_final.png
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from windcast.config import get_settings

logger = logging.getLogger(__name__)

HORIZON_METRIC_RE = re.compile(r"^h(\d+)_(mae|skill_score)$")


def _fetch_parent_runs(experiment_names: list[str]) -> pd.DataFrame:
    """Return a DataFrame of parent runs across the given experiments.

    Filters on ``tags.enercast.run_type = 'parent'``. Returns the raw
    ``mlflow.search_runs`` output (pandas DataFrame).
    """
    df = mlflow.search_runs(
        experiment_names=experiment_names,
        filter_string="tags.`enercast.run_type` = 'parent'",
        output_format="pandas",
    )
    assert isinstance(df, pd.DataFrame)
    return df


def _extract_horizon_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """Pivot the mlflow runs DataFrame into one row per run with h{n}_mae /
    h{n}_skill_score columns.

    Returns the trimmed DataFrame (indexed by run label) and the sorted list
    of horizons that appear in at least one run.
    """
    horizons: set[int] = set()
    mae_cols: list[str] = []
    skill_cols: list[str] = []
    for col in df.columns:
        if not col.startswith("metrics."):
            continue
        m = HORIZON_METRIC_RE.match(col.removeprefix("metrics."))
        if m is None:
            continue
        h = int(m.group(1))
        horizons.add(h)
        if m.group(2) == "mae":
            mae_cols.append(col)
        else:
            skill_cols.append(col)

    backend_col = "tags.enercast.backend"
    fs_col = "tags.enercast.feature_set"
    name_col = "tags.mlflow.runName"

    for needed in (backend_col, fs_col, name_col):
        if needed not in df.columns:
            df[needed] = ""

    keep = [name_col, backend_col, fs_col, *sorted(mae_cols), *sorted(skill_cols)]
    trimmed = df[keep].copy()
    trimmed = trimmed.rename(
        columns={name_col: "run_name", backend_col: "backend", fs_col: "feature_set"}
    )
    # Label = "backend/feature_set" (e.g. "xgboost/wind_full")
    trimmed["label"] = trimmed["backend"].fillna("?") + "/" + trimmed["feature_set"].fillna("?")
    # Sort for deterministic chart ordering: xgboost first then autogluon;
    # within each, baseline -> enriched -> full -> other.
    backend_order = {"xgboost": 0, "autogluon": 1, "mlforecast": 2}
    fs_order = {"wind_baseline": 0, "wind_enriched": 1, "wind_full": 2}
    trimmed = trimmed.assign(
        _b_ord=trimmed["backend"].map(backend_order).fillna(99),
        _f_ord=trimmed["feature_set"].map(fs_order).fillna(99),
    ).sort_values(["_b_ord", "_f_ord"], kind="stable")
    trimmed = trimmed.drop(columns=["_b_ord", "_f_ord"]).reset_index(drop=True)
    return trimmed, sorted(horizons)


def _plot_grouped_bars(
    runs: pd.DataFrame,
    horizons: list[int],
    metric_prefix: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Render one grouped bar chart (x=horizon, group=run) and save to PNG."""
    n_runs = len(runs)
    if n_runs == 0 or not horizons:
        logger.warning("No data for %s; skipping chart %s", title, out_path)
        return

    bar_width = 0.8 / n_runs
    x = range(len(horizons))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (_, row) in enumerate(runs.iterrows()):
        values = [row.get(f"metrics.h{h}_{metric_prefix}", float("nan")) for h in horizons]
        offset = (i - (n_runs - 1) / 2) * bar_width
        ax.bar(
            [xi + offset for xi in x],
            values,
            width=bar_width,
            label=row["label"],
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"h{h}" for h in horizons])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def _print_markdown_table(runs: pd.DataFrame, horizons: list[int]) -> None:
    """Print a Markdown comparison table to stdout."""
    header = ["Run"] + [f"h{h} MAE" for h in horizons] + [f"h{h} Skill" for h in horizons]
    print("\n## Comparison table\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for _, row in runs.iterrows():
        cells = [row["label"]]
        for h in horizons:
            mae = row.get(f"metrics.h{h}_mae", float("nan"))
            cells.append(f"{mae:.0f}" if pd.notna(mae) else "—")
        for h in horizons:
            sk = row.get(f"metrics.h{h}_skill_score", float("nan"))
            cells.append(f"{sk:.3f}" if pd.notna(sk) else "—")
        print("| " + " | ".join(cells) + " |")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MLflow parent runs across experiments")
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="MLflow experiment name. Can be repeated for multi-experiment comparison.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Base output path. Default: reports/comparison_<first-experiment>.png",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    logger.info("MLflow tracking URI: %s", settings.mlflow_tracking_uri)

    runs_raw = _fetch_parent_runs(args.experiment)
    if runs_raw.empty:
        logger.error(
            "No parent runs found in %s. Did you tag runs with enercast.run_type='parent'?",
            args.experiment,
        )
        raise SystemExit(1)

    runs, horizons = _extract_horizon_metrics(runs_raw)
    logger.info("Found %d parent runs, horizons=%s", len(runs), horizons)

    base = args.output or Path(f"reports/comparison_{args.experiment[0]}.png")
    if base.suffix != ".png":
        raise SystemExit("--output must end in .png")
    mae_path = base.with_name(base.stem + "_mae.png")
    skill_path = base.with_name(base.stem + "_skill.png")

    _plot_grouped_bars(
        runs,
        horizons,
        metric_prefix="mae",
        title="MAE by horizon — parent runs",
        ylabel="MAE (kW)",
        out_path=mae_path,
    )
    _plot_grouped_bars(
        runs,
        horizons,
        metric_prefix="skill_score",
        title="Skill score by horizon — parent runs",
        ylabel="Skill score (vs persistence)",
        out_path=skill_path,
    )

    _print_markdown_table(runs, horizons)


if __name__ == "__main__":
    main()
