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

HORIZON_METRIC_RE = re.compile(r"^h(\d+)_(mae|rmse|skill_score)$")

# Cross-domain color palette keyed by feature-set stem so that, in every chart,
# "baseline" is always blue, "enriched" always orange, "full" always green —
# regardless of domain (wind / demand / solar) or backend. This keeps side-by-side
# slides legible without a domain-specific legend lookup.
FEATURE_SET_COLORS: dict[str, str] = {
    "baseline": "#1f77b4",  # blue
    "enriched": "#ff7f0e",  # orange
    "full": "#2ca02c",  # green
    "tso_baseline": "#7f7f7f",  # gray — external benchmark, no feature set
    "other": "#9467bd",  # purple fallback
}

# Backends are distinguished by hatch pattern so identical feature sets across
# ML libraries remain comparable on color alone.
BACKEND_HATCHES: dict[str, str] = {
    "xgboost": "",
    "autogluon": "///",
    "mlforecast": "xxx",
    "tso": "",
}


def _feature_set_stem(feature_set: str) -> str:
    """Return the domain-agnostic stem of a feature set name.

    ``wind_full`` → ``"full"``, ``demand_enriched`` → ``"enriched"``, etc.
    Unknown / external runs (e.g. TSO baseline) fall back to ``"other"``.
    """
    if not feature_set or feature_set == "none":
        return "tso_baseline"
    for prefix in ("wind_", "demand_", "solar_"):
        if feature_set.startswith(prefix):
            return feature_set.removeprefix(prefix)
    return "other"


def _bar_style(backend: str, feature_set: str) -> tuple[str, str]:
    """Return ``(color, hatch)`` for a given (backend, feature_set) pair."""
    stem = _feature_set_stem(feature_set)
    color = FEATURE_SET_COLORS.get(stem, FEATURE_SET_COLORS["other"])
    hatch = BACKEND_HATCHES.get(backend, "")
    return color, hatch


# Domain → (target variable label, unit) used in chart titles and ylabels so a
# reader can never mistake a load MAE of 1,200 MW for a price error of 1,200 €.
DOMAIN_TARGET: dict[str, tuple[str, str]] = {
    "wind": ("active power", "kW"),
    "demand": ("load", "MW"),
    "solar": ("power", "kW"),
}


def _infer_domain(runs: pd.DataFrame) -> str | None:
    """Return the dominant domain tag across parent runs, or None if absent."""
    if "tags.enercast.domain" not in runs.columns:
        return None
    vals = runs["tags.enercast.domain"].dropna().unique().tolist()
    if not vals:
        return None
    # If runs mix domains (unusual), prefer the most common one.
    return str(runs["tags.enercast.domain"].mode().iat[0])


def _fetch_parent_runs(
    experiment_names: list[str],
    data_quality: str | None = None,
    show_all: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame of parent runs across the given experiments.

    Filters on ``tags.enercast.run_type = 'parent'``. When *show_all* is False
    and *data_quality* is set, also filters on ``enercast.data_quality``.
    Returns the raw ``mlflow.search_runs`` output (pandas DataFrame).
    """
    filter_str = "tags.`enercast.run_type` = 'parent'"
    if not show_all and data_quality:
        filter_str += f" AND tags.`enercast.data_quality` = '{data_quality}'"
    df = mlflow.search_runs(
        experiment_names=experiment_names,
        filter_string=filter_str,
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
    rmse_cols: list[str] = []
    skill_cols: list[str] = []
    for col in df.columns:
        if not col.startswith("metrics."):
            continue
        m = HORIZON_METRIC_RE.match(col.removeprefix("metrics."))
        if m is None:
            continue
        h = int(m.group(1))
        horizons.add(h)
        kind = m.group(2)
        if kind == "mae":
            mae_cols.append(col)
        elif kind == "rmse":
            rmse_cols.append(col)
        else:
            skill_cols.append(col)

    backend_col = "tags.enercast.backend"
    fs_col = "tags.enercast.feature_set"
    domain_col = "tags.enercast.domain"
    name_col = "tags.mlflow.runName"

    for needed in (backend_col, fs_col, domain_col, name_col):
        if needed not in df.columns:
            df[needed] = ""

    keep = [
        name_col,
        backend_col,
        fs_col,
        domain_col,
        *sorted(mae_cols),
        *sorted(rmse_cols),
        *sorted(skill_cols),
    ]
    trimmed = df[keep].copy()
    trimmed = trimmed.rename(
        columns={name_col: "run_name", backend_col: "backend", fs_col: "feature_set"}
    )
    # keep the tag column under its dotted name so _infer_domain can find it
    trimmed[domain_col] = df[domain_col].to_numpy()
    # Label = "backend/feature_set" (e.g. "xgboost/wind_full")
    trimmed["label"] = trimmed["backend"].fillna("?") + "/" + trimmed["feature_set"].fillna("?")
    # Sort for deterministic chart ordering: xgboost first, then autogluon, then
    # mlforecast, then TSO benchmarks; within each, baseline -> enriched -> full.
    backend_order = {"xgboost": 0, "autogluon": 1, "mlforecast": 2, "tso": 9}
    stem_order = {"baseline": 0, "enriched": 1, "full": 2, "tso_baseline": 8, "other": 9}
    trimmed = trimmed.assign(
        _b_ord=trimmed["backend"].map(backend_order).fillna(99),
        _f_ord=trimmed["feature_set"].map(lambda fs: stem_order[_feature_set_stem(fs)]),
    ).sort_values(["_b_ord", "_f_ord"], kind="stable")
    trimmed = trimmed.drop(columns=["_b_ord", "_f_ord"]).reset_index(drop=True)
    return trimmed, sorted(horizons)


def _add_skill_vs_baseline(runs: pd.DataFrame, horizons: list[int]) -> str | None:
    """Compute a per-horizon ``skill_vs_baseline`` column for every run.

    The reference is the xgboost run whose feature set has the ``baseline`` stem
    (typically ``xgboost/{domain}_baseline``). For each run and each horizon:

        skill_vs_baseline = 1 - metric_run / metric_xgb_baseline

    where *metric* is RMSE if available, otherwise MAE as a fallback. In
    practice the ratio is nearly identical across the two (Gaussian residuals
    give RMSE/MAE ≈ 1.25 uniformly) and the MAE fallback unblocks runs that
    only logged ``h{n}_mae`` on the parent. Future runs should bubble
    ``h{n}_rmse`` too — see ``scripts/train.py``.

    Interpretation:

    - **0.0**  → identical to our baseline (the reference itself sits here)
    - **> 0**  → error lower than the baseline, i.e. the run adds value
    - **< 0**  → error worse than the baseline, i.e. the run hurts

    This is the "marginal gain vs our own starting point" metric, distinct
    from the classical skill score which uses persistence as reference.

    Returns the label of the baseline run used, or ``None`` if no baseline
    was found (in which case no new columns are added).
    """
    baseline_mask = (runs["backend"] == "xgboost") & runs["feature_set"].apply(
        lambda fs: _feature_set_stem(fs) == "baseline"
    )
    baseline_rows = runs[baseline_mask]
    if baseline_rows.empty:
        logger.warning("No xgboost baseline run found — skipping skill_vs_baseline computation")
        return None
    if len(baseline_rows) > 1:
        logger.warning(
            "Found %d xgboost baseline runs, using the first one (%s)",
            len(baseline_rows),
            baseline_rows.iloc[0]["label"],
        )
    baseline_row = baseline_rows.iloc[0]

    # Prefer RMSE when present; fall back to MAE for runs that did not bubble
    # RMSE up to the parent. Log which metric is driving the computation so
    # the provenance of the chart is clear.
    def _pick_metric(h: int) -> str | None:
        rmse_col = f"metrics.h{h}_rmse"
        mae_col = f"metrics.h{h}_mae"
        if rmse_col in runs.columns and pd.notna(baseline_row.get(rmse_col, float("nan"))):
            return rmse_col
        if mae_col in runs.columns and pd.notna(baseline_row.get(mae_col, float("nan"))):
            return mae_col
        return None

    metrics_used: set[str] = set()
    for h in horizons:
        ref_col = _pick_metric(h)
        new_col = f"metrics.h{h}_skill_vs_baseline"
        if ref_col is None:
            runs[new_col] = float("nan")
            continue
        ref_val = baseline_row[ref_col]
        if pd.isna(ref_val) or ref_val == 0:
            runs[new_col] = float("nan")
            continue
        runs[new_col] = 1.0 - runs[ref_col] / ref_val
        metrics_used.add("rmse" if ref_col.endswith("_rmse") else "mae")

    if metrics_used:
        logger.info("skill_vs_baseline computed from: %s", ", ".join(sorted(metrics_used)))
    return str(baseline_row["label"])


def _plot_grouped_bars(
    runs: pd.DataFrame,
    horizons: list[int],
    metric_prefix: str,
    title: str,
    ylabel: str,
    out_path: Path,
    draw_zero_line: bool = False,
) -> None:
    """Render one grouped bar chart (x=horizon, group=run) and save to PNG.

    If *draw_zero_line* is True, a horizontal line is drawn at y=0 — used for
    relative-skill charts where 0 means "same as the reference run".
    """
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
        color, hatch = _bar_style(row["backend"], row["feature_set"])
        ax.bar(
            [xi + offset for xi in x],
            values,
            width=bar_width,
            label=row["label"],
            color=color,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"h{h}" for h in horizons])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    if draw_zero_line:
        ax.axhline(0, color="black", linewidth=1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def _print_markdown_table(
    runs: pd.DataFrame,
    horizons: list[int],
    has_skill_vs_baseline: bool = False,
) -> None:
    """Print a Markdown comparison table to stdout."""
    header = ["Run"] + [f"h{h} MAE" for h in horizons] + [f"h{h} Skill" for h in horizons]
    if has_skill_vs_baseline:
        header += [f"h{h} SkillVsBase" for h in horizons]
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
        if has_skill_vs_baseline:
            for h in horizons:
                svb = row.get(f"metrics.h{h}_skill_vs_baseline", float("nan"))
                cells.append(f"{svb:+.3f}" if pd.notna(svb) else "—")
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
    parser.add_argument(
        "--data-quality",
        default="CLEAN",
        help="Filter by enercast.data_quality tag. Default: CLEAN",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="show_all",
        help="Show all runs regardless of data_quality tag",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    logger.info("MLflow tracking URI: %s", settings.mlflow_tracking_uri)

    runs_raw = _fetch_parent_runs(
        args.experiment,
        data_quality=args.data_quality,
        show_all=args.show_all,
    )
    if runs_raw.empty:
        logger.error(
            "No parent runs found in %s. Did you tag runs with enercast.run_type='parent'?",
            args.experiment,
        )
        raise SystemExit(1)

    runs, horizons = _extract_horizon_metrics(runs_raw)
    logger.info("Found %d parent runs, horizons=%s", len(runs), horizons)

    baseline_label = _add_skill_vs_baseline(runs, horizons)
    if baseline_label:
        logger.info("Skill vs baseline reference: %s", baseline_label)

    domain = _infer_domain(runs)
    target_name, unit = DOMAIN_TARGET.get(domain or "", ("target", ""))
    experiment_label = args.experiment[0].removeprefix("enercast-")
    # Title makes the target variable explicit so a reader can never mistake a
    # load MAE of 1,200 MW for a price error of 1,200 €.
    variable_tag = f"{target_name} ({unit})" if unit else target_name
    mae_title = f"{experiment_label} — MAE on {variable_tag} by forecast horizon"
    skill_title = (
        f"{experiment_label} — Skill score on {variable_tag} by forecast horizon (vs persistence)"
    )
    mae_ylabel = f"MAE ({unit})" if unit else "MAE"

    base = args.output or Path(f"reports/comparison_{args.experiment[0]}.png")
    if base.suffix != ".png":
        raise SystemExit("--output must end in .png")
    mae_path = base.with_name(base.stem + "_mae.png")
    skill_path = base.with_name(base.stem + "_skill.png")
    skill_vs_base_path = base.with_name(base.stem + "_skill_vs_baseline.png")

    _plot_grouped_bars(
        runs,
        horizons,
        metric_prefix="mae",
        title=mae_title,
        ylabel=mae_ylabel,
        out_path=mae_path,
    )
    _plot_grouped_bars(
        runs,
        horizons,
        metric_prefix="skill_score",
        title=skill_title,
        ylabel="Skill score (vs persistence)",
        out_path=skill_path,
    )

    if baseline_label:
        svb_title = f"{experiment_label} — Marginal skill on {variable_tag} (vs {baseline_label})"
        _plot_grouped_bars(
            runs,
            horizons,
            metric_prefix="skill_vs_baseline",
            title=svb_title,
            ylabel=f"1 - RMSE / RMSE({baseline_label})",
            out_path=skill_vs_base_path,
            draw_zero_line=True,
        )

    _print_markdown_table(runs, horizons, has_skill_vs_baseline=baseline_label is not None)


if __name__ == "__main__":
    main()
