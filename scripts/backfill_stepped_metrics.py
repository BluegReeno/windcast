"""Backfill stepped horizon metrics on existing parent MLflow runs.

Reads per-horizon flat metrics from each child run (``h{n}_mae``,
``h{n}_rmse``, ``h{n}_skill_score``, ``h{n}_bias``, plus the unprefixed
``persistence_*`` siblings) and logs them as stepped metrics on the parent
run (``mae_by_horizon_min`` with ``step=h * data_resolution_min``, etc.).
This retroactively unlocks MLflow's native "metric vs horizon" line chart
for runs trained before the stepped-on-parent refactor, without retraining.

Typical use: backfill the AutoGluon ``wind_full`` parent, which is expensive
to retrain (~30 min with ``best_quality``).

Usage:
    uv run python scripts/backfill_stepped_metrics.py \\
        --parent-run-name kwf1-autogluon-wind_full \\
        --experiment enercast-kelmarsh \\
        --data-resolution-min 10
"""

from __future__ import annotations

import argparse
import logging
import re

from mlflow.tracking import MlflowClient

from windcast.tracking.mlflow_utils import STEPPED_METRIC_MAP, setup_mlflow

logger = logging.getLogger(__name__)

# Matches keys like "h48_mae", "h6_skill_score", "h12_bias", "h1_rmse".
H_PREFIXED_METRIC_RE = re.compile(r"^h(\d+)_(.+)$")


def _horizon_metrics_from_child(
    child_metrics: dict[str, float],
) -> tuple[int | None, dict[str, float]]:
    """Extract ``(horizon_steps, metrics_dict)`` from a child run's metrics.

    Parses ``h{n}_{metric}`` keys to recover the horizon and the base metric
    names, and picks up unprefixed ``persistence_*`` keys as-is. Returns
    ``(None, {})`` if the child has no recognizable horizon metrics.
    """
    horizon_steps: int | None = None
    extracted: dict[str, float] = {}

    for key, value in child_metrics.items():
        m = H_PREFIXED_METRIC_RE.match(key)
        if m is not None:
            h = int(m.group(1))
            base = m.group(2)
            if horizon_steps is None:
                horizon_steps = h
            elif horizon_steps != h:
                logger.warning(
                    "Child has mixed horizons in its metrics (%d vs %d), skipping",
                    horizon_steps,
                    h,
                )
                return None, {}
            extracted[base] = value
        elif key.startswith("persistence_") and not key.startswith("persistence_mape"):
            # Persistence metrics are unprefixed on the child run
            extracted[key] = value

    return horizon_steps, extracted


def backfill_parent(
    client: MlflowClient,
    parent_run_id: str,
    experiment_id: str,
    data_resolution_min: int,
    dry_run: bool = False,
) -> int:
    """Backfill stepped horizon metrics on a single parent run.

    Returns the number of stepped metric points written.
    """
    children = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        max_results=100,
    )
    if not children:
        logger.warning("No children found for parent %s", parent_run_id)
        return 0

    logger.info("Found %d child runs", len(children))

    # horizon_steps -> {base_metric_name: value}
    per_horizon: dict[int, dict[str, float]] = {}
    for child in children:
        h, metrics = _horizon_metrics_from_child(child.data.metrics)
        if h is None or not metrics:
            logger.warning("Skipping child %s (no horizon metrics)", child.info.run_id[:8])
            continue
        if h in per_horizon:
            logger.warning("Duplicate horizon %d, skipping child %s", h, child.info.run_id[:8])
            continue
        per_horizon[h] = metrics
        logger.info(
            "  child %s: h=%d, %d metrics (%s)",
            child.info.run_id[:8],
            h,
            len(metrics),
            ", ".join(sorted(metrics)),
        )

    # Log stepped metrics on the parent
    count = 0
    for h in sorted(per_horizon):
        horizon_minutes = h * data_resolution_min
        for metric_name, value in per_horizon[h].items():
            stepped_name = STEPPED_METRIC_MAP.get(metric_name)
            if stepped_name is None:
                continue
            if dry_run:
                logger.info("[DRY] parent.%s[step=%d] = %.4f", stepped_name, horizon_minutes, value)
            else:
                client.log_metric(parent_run_id, stepped_name, value, step=horizon_minutes)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-run-name",
        required=True,
        help="Run name of the parent to backfill (e.g. kwf1-autogluon-wind_full)",
    )
    parser.add_argument(
        "--experiment",
        default="enercast-kelmarsh",
        help="MLflow experiment name (default: enercast-kelmarsh)",
    )
    parser.add_argument(
        "--data-resolution-min",
        type=int,
        default=10,
        help="Data resolution in minutes (default: 10 for wind SCADA)",
    )
    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///mlflow.db",
        help="MLflow tracking URI (default: sqlite:///mlflow.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be logged without writing to MLflow",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    setup_mlflow(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        raise SystemExit(f"Experiment {args.experiment!r} not found")

    parents = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.`enercast.run_type` = 'parent'",
        order_by=["start_time DESC"],
        max_results=100,
    )
    matching = [p for p in parents if p.info.run_name == args.parent_run_name]
    if not matching:
        raise SystemExit(
            f"No parent run named {args.parent_run_name!r} found "
            f"(searched {len(parents)} parents in {args.experiment!r})"
        )
    if len(matching) > 1:
        logger.warning("Multiple parents match — using the most recent")

    parent = matching[0]
    logger.info("Backfilling parent run %s (%s)", parent.info.run_id, parent.info.run_name)
    count = backfill_parent(
        client,
        parent.info.run_id,
        exp.experiment_id,
        args.data_resolution_min,
        dry_run=args.dry_run,
    )
    action = "Would write" if args.dry_run else "Wrote"
    logger.info("%s %d stepped metric points to parent", action, count)

    if not args.dry_run and count > 0:
        # Sanity-check: read back the stepped history
        hist = client.get_metric_history(parent.info.run_id, "mae_by_horizon_min")
        steps = sorted((m.step, round(m.value, 1)) for m in hist)
        logger.info("Parent.mae_by_horizon_min now has %d points: %s", len(steps), steps)


if __name__ == "__main__":
    main()
