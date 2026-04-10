"""Git state capture and MLflow lineage tag logging."""

import logging
import subprocess

import mlflow

logger = logging.getLogger(__name__)


def get_git_info() -> dict[str, str]:
    """Return git commit, branch, dirty state as a dict of MLflow tag values.

    Runs git commands via subprocess. Returns an empty dict if not in a git repo.
    """
    tags: dict[str, str] = {}
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        tags["mlflow.source.git.branch"] = branch

        dirty_result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True,
        )
        tags["enercast.git.dirty"] = str(dirty_result.returncode != 0).lower()
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.debug("Not in a git repo or git not available — skipping git tags")

    return tags


def log_lineage_tags(
    generation: str | None = None,
    nwp_source: str = "forecast",
    data_quality: str = "CLEAN",
    change_reason: str | None = None,
) -> None:
    """Log lineage + git tags on the active MLflow run.

    Sets enercast.* lineage tags and git state tags. Call inside an active
    ``mlflow.start_run()`` context.

    Args:
        generation: Semantic generation label (e.g. "gen4"). None = skip.
        nwp_source: NWP data source — "forecast", "era5", or "none".
        data_quality: Data quality flag — "CLEAN" or "LEAKED".
        change_reason: Free-text explanation of what changed. None = skip.
    """
    tags: dict[str, str] = {
        "enercast.nwp_source": nwp_source,
        "enercast.data_quality": data_quality,
    }
    if generation:
        tags["enercast.generation"] = generation
    if change_reason:
        tags["enercast.change_reason"] = change_reason

    tags.update(get_git_info())
    mlflow.set_tags(tags)
