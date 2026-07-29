from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def code_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


@dataclass(frozen=True)
class ExperimentLineage:
    source_versions: dict[str, str]
    training_cutoff: int
    validation_cutoff: int
    test_cutoff: int
    eligible_domains: list[str]
    user_count: int
    item_count: int
    interaction_count: int
    negative_sampling_policy: str
    seed: int
    code_revision: str
    compute_configuration: str
    estimated_cost: str

    def write(self, path: str | Path, model_state: dict[str, Any]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {"lineage": asdict(self), "model_state": model_state}, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )

    def log_mlflow(self, metrics: dict[str, float], artifact_path: str | Path) -> None:
        """Log a completed integration experiment when the optional MLflow extra is installed."""
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError("Install the 'databricks' extra to log MLflow experiments") from exc
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "training_cutoff": self.training_cutoff,
                    "validation_cutoff": self.validation_cutoff,
                    "test_cutoff": self.test_cutoff,
                    "eligible_domains": ",".join(self.eligible_domains),
                    "negative_sampling_policy": self.negative_sampling_policy,
                    "seed": self.seed,
                    "code_revision": self.code_revision,
                    "compute_configuration": self.compute_configuration,
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(artifact_path))
