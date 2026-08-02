from __future__ import annotations

from dataclasses import asdict
from typing import Any

from marketplace_recommender.retrieval.sasrec_model import SasRecConfig


def _log_mlflow_model(
    mlflow: Any,
    torch: Any,
    model: Any,
    source_frame: Any,
    source_table: str,
    source_delta_version: int,
    config: SasRecConfig,
    summary: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    experiment_name = "/Shared/marketplace-recommender-scale"
    registered_model_name = "workspace.scale_serving.marketplace_sasrec_encoder"
    mlflow.set_experiment(experiment_name)
    mlflow.set_registry_uri("databricks-uc")
    with mlflow.start_run(run_name=f"sasrec-{summary['benchmark_id'][:12]}") as run:
        mlflow.set_tags(
            {
                "benchmark_id": summary["benchmark_id"],
                "contract_version": summary["contract_version"],
                "source_delta_table": source_table,
                "source_delta_version": str(source_delta_version),
                "deployment_purpose": "sequence encoder for governed ANN candidate retrieval",
                "promotion_basis": "validation-only",
                "serialization": "torchscript",
            }
        )
        mlflow.log_params({key: value for key, value in asdict(config).items()})
        mlflow.log_metrics(
            {
                "validation_sasrec_ndcg_at_10": summary["validation"]["sasrec"]["ndcg_at_10"],
                "validation_popularity_ndcg_at_10": summary["validation"]["popularity"][
                    "ndcg_at_10"
                ],
                "test_sasrec_ndcg_at_10": summary["test"]["sasrec"]["ndcg_at_10"],
                "test_popularity_ndcg_at_10": summary["test"]["popularity"]["ndcg_at_10"],
            }
        )
        dataset = mlflow.data.from_spark(
            source_frame,
            table_name=source_table,
            version=str(source_delta_version),
            name="positive-sequential-interactions",
        )
        mlflow.log_input(dataset, context="training")
        mlflow.log_dict(summary, "evidence/sequence_benchmark.json")
        example = np.zeros((2, config.max_sequence_length), dtype=np.int64)
        example[0, :2] = [1, 2]
        example[1, :3] = [2, 3, 4]
        with torch.no_grad():
            output = model(torch.from_numpy(example)).detach().numpy()
            serving_model = torch.jit.trace(
                model.eval(), torch.from_numpy(example), check_trace=False
            )
        signature = mlflow.models.infer_signature(example, output)
        model_info = mlflow.pytorch.log_model(
            pytorch_model=serving_model,
            name="sasrec_encoder",
            input_example=example,
            signature=signature,
            registered_model_name=registered_model_name,
            serialization_format="pickle",
            metadata={"serialization": "torchscript", "purpose": "ANN query encoder"},
        )
        version = str(model_info.registered_model_version)
        client = mlflow.MlflowClient(registry_uri="databricks-uc")
        client.set_model_version_tag(
            registered_model_name, version, "benchmark_id", summary["benchmark_id"]
        )
        client.set_model_version_tag(
            registered_model_name, version, "source_delta_version", str(source_delta_version)
        )
        validation_promoted = (
            summary["validation"]["sasrec"]["ndcg_at_10"]
            > summary["validation"]["popularity"]["ndcg_at_10"]
        )
        if validation_promoted:
            client.set_registered_model_alias(registered_model_name, "candidate", version)
        return {
            "experiment_name": experiment_name,
            "run_id": run.info.run_id,
            "registered_model_name": registered_model_name,
            "registered_model_version": version,
            "model_uri": model_info.model_uri,
            "alias": "candidate" if validation_promoted else None,
            "promotion_basis": "validation-only",
        }
