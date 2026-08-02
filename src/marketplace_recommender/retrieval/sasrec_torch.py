from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from marketplace_recommender.evaluation.paired_bootstrap import (
    paired_bootstrap_mean_difference,
)


@dataclass(frozen=True)
class SasRecConfig:
    hidden_size: int = 64
    attention_heads: int = 4
    transformer_layers: int = 2
    max_sequence_length: int = 20
    dropout: float = 0.20
    batch_size: int = 512
    epochs: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    negative_samples: int = 4
    maximum_training_users: int = 30_000
    maximum_training_examples: int = 60_000
    evaluation_user_limit: int = 4_000
    candidate_k: int = 100
    recommendation_k: int = 10
    minimum_item_users: int = 2
    seed: int = 20250308
    bootstrap_samples: int = 5_000

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.attention_heads <= 0 or self.transformer_layers <= 0:
            raise ValueError("transformer dimensions must be positive")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if self.max_sequence_length < 2 or self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("sequence length, batch size, and epochs must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.negative_samples <= 0:
            raise ValueError("learning rate and negative samples must be positive")
        if self.candidate_k < self.recommendation_k:
            raise ValueError("candidate_k must be at least recommendation_k")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")


try:
    import torch as _torch
except ImportError:  # Core ETL and local CI do not require the sequence extra.
    _torch = None


if _torch is not None:

    class SasRecEncoder(_torch.nn.Module):
        """Causal self-attention encoder whose output is an ANN query vector."""

        def __init__(self, item_count: int, config: SasRecConfig) -> None:
            super().__init__()
            self.item_embedding = _torch.nn.Embedding(
                item_count + 1, config.hidden_size, padding_idx=0
            )
            self.position_embedding = _torch.nn.Embedding(
                config.max_sequence_length, config.hidden_size
            )
            layer = _torch.nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.attention_heads,
                dim_feedforward=config.hidden_size * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = _torch.nn.TransformerEncoder(layer, num_layers=config.transformer_layers)
            self.output_norm = _torch.nn.LayerNorm(config.hidden_size)

        def forward(self, sequences: Any) -> Any:
            positions = _torch.arange(sequences.shape[1], device=sequences.device).unsqueeze(0)
            values = self.item_embedding(sequences) + self.position_embedding(positions)
            padding = sequences.eq(0)
            causal = _torch.triu(
                _torch.ones(
                    sequences.shape[1],
                    sequences.shape[1],
                    dtype=_torch.bool,
                    device=sequences.device,
                ),
                diagonal=1,
            )
            encoded = self.encoder(
                values,
                mask=causal,
                src_key_padding_mask=padding,
                is_causal=True,
            )
            final_positions = (~padding).sum(dim=1).sub(1).clamp_min(0)
            query = encoded[
                _torch.arange(sequences.shape[0], device=sequences.device), final_positions
            ]
            return self.output_norm(query)

else:

    class SasRecEncoder:  # pragma: no cover - optional dependency error path
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("Install the 'sequence' extra to construct SASRec")


def build_next_item_examples(
    histories: Iterable[list[int]], max_sequence_length: int
) -> list[tuple[list[int], int]]:
    """Create causal next-novel-item examples from time-ordered histories."""
    examples: list[tuple[list[int], int]] = []
    for history in histories:
        for target_position in range(2, len(history)):
            prefix = history[max(0, target_position - max_sequence_length) : target_position]
            target = history[target_position]
            if target not in prefix:
                examples.append((prefix, target))
    return examples


def ranking_metrics_from_rank(rank: int | None, candidate_k: int = 100) -> dict[str, float]:
    return {
        "recall_at_10": float(rank is not None and rank <= 10),
        "ndcg_at_10": 1.0 / math.log2(rank + 1) if rank is not None and rank <= 10 else 0.0,
        "mrr_at_10": 1.0 / rank if rank is not None and rank <= 10 else 0.0,
        "candidate_recall_at_100": float(rank is not None and rank <= candidate_k),
    }


def _build_model(torch: Any, item_count: int, config: SasRecConfig) -> Any:
    if _torch is None or torch is not _torch:
        raise RuntimeError("the imported torch runtime does not match the SASRec module runtime")
    return SasRecEncoder(item_count, config)


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def _pad(histories: list[list[int]], max_length: int, np: Any) -> Any:
    values = np.zeros((len(histories), max_length), dtype=np.int64)
    for row_index, history in enumerate(histories):
        clipped = history[-max_length:]
        values[row_index, : len(clipped)] = clipped
    return values


def _negative_samples(
    examples: list[tuple[list[int], int]], item_count: int, count: int, seed: int, np: Any
) -> Any:
    rng = random.Random(seed)
    negatives = np.zeros((len(examples), count), dtype=np.int64)
    for row_index, (history, target) in enumerate(examples):
        excluded = set(history)
        excluded.add(target)
        for column in range(count):
            value = rng.randint(1, item_count)
            while value in excluded:
                value = rng.randint(1, item_count)
            negatives[row_index, column] = value
    return negatives


def _train(
    torch: Any,
    model: Any,
    examples: list[tuple[list[int], int]],
    item_count: int,
    config: SasRecConfig,
    epochs: int,
    validation: tuple[list[str], list[list[int]], list[int]] | None = None,
) -> tuple[Any, int, list[dict[str, float]]]:
    import numpy as np

    if not examples:
        raise RuntimeError("SASRec training produced no causal next-item examples")
    histories = _pad([history for history, _ in examples], config.max_sequence_length, np)
    targets = np.asarray([target for _, target in examples], dtype=np.int64)
    negatives = _negative_samples(examples, item_count, config.negative_samples, config.seed, np)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(histories), torch.from_numpy(targets), torch.from_numpy(negatives)
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_epoch = 1
    best_metric = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epoch_rows: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for sequences, positives, sampled_negatives in loader:
            optimizer.zero_grad(set_to_none=True)
            query = model(sequences)
            positive_score = (query * model.item_embedding(positives)).sum(dim=1, keepdim=True)
            negative_score = (query.unsqueeze(1) * model.item_embedding(sampled_negatives)).sum(
                dim=2
            )
            loss = -torch.nn.functional.logsigmoid(positive_score - negative_score).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {"epoch": float(epoch), "training_loss": sum(losses) / len(losses)}
        if validation is not None:
            metrics, _, _ = _evaluate(torch, model, *validation, config=config)
            row["validation_ndcg_at_10"] = metrics["ndcg_at_10"]
            if metrics["ndcg_at_10"] > best_metric:
                best_metric = metrics["ndcg_at_10"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
        epoch_rows.append(row)
        print(json.dumps({"event": "sasrec_epoch", **row}, sort_keys=True))
    if validation is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, epoch_rows


def _evaluate(
    torch: Any,
    model: Any,
    user_ids: list[str],
    histories: list[list[int]],
    targets: list[int],
    config: SasRecConfig,
) -> tuple[dict[str, float], list[dict[str, float]], list[list[tuple[int, float]]]]:
    import numpy as np

    model.eval()
    padded = _pad(histories, config.max_sequence_length, np)
    metric_rows: list[dict[str, float]] = []
    recommendations: list[list[tuple[int, float]]] = []
    item_matrix = model.item_embedding.weight[1:].detach()
    with torch.no_grad():
        for offset in range(0, len(histories), config.batch_size):
            batch_values = torch.from_numpy(padded[offset : offset + config.batch_size])
            queries = model(batch_values)
            scores = queries @ item_matrix.T
            batch_histories = histories[offset : offset + config.batch_size]
            for row_index, history in enumerate(batch_histories):
                seen = [value - 1 for value in set(history) if value > 0]
                if seen:
                    scores[row_index, torch.tensor(seen, dtype=torch.long)] = float("-inf")
            top_scores, top_columns = torch.topk(scores, k=config.candidate_k, dim=1)
            for row_index in range(top_columns.shape[0]):
                ranked_items = [int(value) + 1 for value in top_columns[row_index].tolist()]
                target = targets[offset + row_index]
                rank = ranked_items.index(target) + 1 if target in ranked_items else None
                metric_rows.append(ranking_metrics_from_rank(rank, config.candidate_k))
                recommendations.append(
                    list(zip(ranked_items, [float(value) for value in top_scores[row_index]]))
                )
    metrics = {
        key: sum(row[key] for row in metric_rows) / len(metric_rows) for key in metric_rows[0]
    }
    return metrics, metric_rows, recommendations


def _popularity_evaluate(
    histories: list[list[int]],
    targets: list[int],
    counts: dict[int, int],
    config: SasRecConfig,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    ranking = sorted(counts, key=lambda item: (-counts[item], item))
    rows: list[dict[str, float]] = []
    for history, target in zip(histories, targets, strict=True):
        seen = set(history)
        candidates = [item for item in ranking if item not in seen][: config.candidate_k]
        rank = candidates.index(target) + 1 if target in candidates else None
        rows.append(ranking_metrics_from_rank(rank, config.candidate_k))
    return (
        {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]},
        rows,
    )


def _indexed_sequences(
    positive: Any,
    vocabulary: Any,
    cutoff: int,
    maximum_users: int,
) -> list[tuple[str, list[int]]]:
    from pyspark.sql import functions as F

    return [
        (row.user_id, [int(event.item_index) for event in row.events])
        for row in (
            positive.join(vocabulary, "parent_asin", "inner")
            .where(F.col("review_timestamp") < F.lit(cutoff))
            .groupBy("user_id")
            .agg(
                F.sort_array(F.collect_list(F.struct("review_timestamp", "item_index"))).alias(
                    "events"
                )
            )
            .where(F.size("events") >= 3)
            .orderBy(F.xxhash64("user_id"))
            .limit(maximum_users)
            .collect()
        )
    ]


def _temporal_targets(
    positive: Any,
    vocabulary: Any,
    history_cutoff: int,
    end: int | None,
    limit: int,
) -> tuple[list[str], list[list[int]], list[int]]:
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    indexed = positive.join(vocabulary, "parent_asin", "inner")
    histories = (
        indexed.where(F.col("review_timestamp") < F.lit(history_cutoff))
        .groupBy("user_id")
        .agg(
            F.sort_array(F.collect_list(F.struct("review_timestamp", "item_index"))).alias("events")
        )
        .where(F.size("events") >= 2)
    )
    future = indexed.where(F.col("review_timestamp") >= F.lit(history_cutoff))
    if end is not None:
        future = future.where(F.col("review_timestamp") < F.lit(end))
    future = future.alias("future")
    seen = histories.select(
        "user_id", F.explode("events.item_index").alias("seen_item_index")
    ).alias("seen")
    unseen = future.join(
        seen,
        (F.col("future.user_id") == F.col("seen.user_id"))
        & (F.col("future.item_index") == F.col("seen.seen_item_index")),
        "left_anti",
    )
    targets = (
        unseen.join(histories, "user_id", "inner")
        .withColumn(
            "target_order",
            F.row_number().over(
                Window.partitionBy("user_id").orderBy("review_timestamp", "parent_asin")
            ),
        )
        .where(F.col("target_order") == 1)
        .orderBy(F.xxhash64("user_id"))
        .limit(limit)
        .select("user_id", "events", F.col("item_index").alias("target_item_index"))
        .collect()
    )
    return (
        [row.user_id for row in targets],
        [[int(event.item_index) for event in row.events] for row in targets],
        [int(row.target_item_index) for row in targets],
    )


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


def train_sasrec_benchmark(
    spark: Any,
    catalog: str,
    schema_prefix: str,
    model_artifact_path: str,
    job_run_id: str,
    job_id: str,
    config: SasRecConfig | None = None,
) -> dict[str, Any]:
    """Train and certify a causal self-attention sequential recommender."""
    import mlflow
    import torch
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    config = config or SasRecConfig()
    config.validate()
    _seed_everything(torch, config.seed)
    prefix = f"{schema_prefix}_" if schema_prefix else ""
    silver_table = f"{catalog}.{prefix}silver.silver_interactions"
    features_schema = f"{catalog}.{prefix}features"
    serving_schema = f"{catalog}.{prefix}serving"
    monitoring_schema = f"{catalog}.{prefix}monitoring"
    for schema in (features_schema, serving_schema, monitoring_schema):
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    positive = (
        spark.table(silver_table)
        .where(F.col("verified_purchase") & (F.col("rating") >= 4))
        .select("user_id", "parent_asin", "review_timestamp")
    )
    train_cutoff, validation_cutoff = (
        int(value) for value in positive.approxQuantile("review_timestamp", [0.80, 0.90], 0.0001)
    )
    source_delta_version = int(
        spark.sql(f"DESCRIBE HISTORY {silver_table} LIMIT 1").first().version
    )
    item_support = (
        positive.where(F.col("review_timestamp") < F.lit(train_cutoff))
        .groupBy("parent_asin")
        .agg(F.countDistinct("user_id").alias("user_count"))
        .where(F.col("user_count") >= config.minimum_item_users)
    )
    vocabulary = item_support.select("parent_asin").withColumn(
        "item_index",
        F.row_number().over(Window.orderBy("parent_asin")).cast("int"),
    )
    vocabulary.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(f"{features_schema}.sasrec_item_vocabulary")
    vocabulary = spark.table(f"{features_schema}.sasrec_item_vocabulary")
    item_count = vocabulary.count()
    if item_count < 100:
        raise RuntimeError("sequential vocabulary is unexpectedly small")

    training_sequences = _indexed_sequences(
        positive, vocabulary, train_cutoff, config.maximum_training_users
    )
    training_examples = build_next_item_examples(
        [history for _, history in training_sequences], config.max_sequence_length
    )
    random.Random(config.seed).shuffle(training_examples)
    training_examples = training_examples[: config.maximum_training_examples]
    validation = _temporal_targets(
        positive,
        vocabulary,
        train_cutoff,
        validation_cutoff,
        config.evaluation_user_limit,
    )
    if not validation[0]:
        raise RuntimeError("sequential validation cohort is empty")

    model = _build_model(torch, item_count, config)
    model, best_epoch, epoch_rows = _train(
        torch,
        model,
        training_examples,
        item_count,
        config,
        config.epochs,
        validation,
    )
    validation_sasrec, validation_sasrec_rows, _ = _evaluate(
        torch, model, *validation, config=config
    )
    training_counts: dict[int, int] = {}
    for _, history in training_sequences:
        for item in history:
            training_counts[item] = training_counts.get(item, 0) + 1
    validation_popularity, validation_popularity_rows = _popularity_evaluate(
        validation[1], validation[2], training_counts, config
    )

    final_sequences = _indexed_sequences(
        positive, vocabulary, validation_cutoff, config.maximum_training_users
    )
    final_examples = build_next_item_examples(
        [history for _, history in final_sequences], config.max_sequence_length
    )
    random.Random(config.seed).shuffle(final_examples)
    final_examples = final_examples[: config.maximum_training_examples]
    _seed_everything(torch, config.seed)
    final_model = _build_model(torch, item_count, config)
    final_model, _, _ = _train(torch, final_model, final_examples, item_count, config, best_epoch)
    test = _temporal_targets(
        positive,
        vocabulary,
        validation_cutoff,
        None,
        config.evaluation_user_limit,
    )
    if not test[0]:
        raise RuntimeError("sequential test cohort is empty")
    test_sasrec, test_sasrec_rows, test_recommendations = _evaluate(
        torch, final_model, *test, config=config
    )
    final_counts: dict[int, int] = {}
    for _, history in final_sequences:
        for item in history:
            final_counts[item] = final_counts.get(item, 0) + 1
    test_popularity, test_popularity_rows = _popularity_evaluate(
        test[1], test[2], final_counts, config
    )
    uncertainty = paired_bootstrap_mean_difference(
        (row["ndcg_at_10"] for row in test_popularity_rows),
        (row["ndcg_at_10"] for row in test_sasrec_rows),
        samples=config.bootstrap_samples,
        seed=config.seed,
    ).as_dict()
    implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    definition = {
        "contract_version": "sasrec-temporal-benchmark/v1",
        "implementation_sha256": implementation_sha256,
        "source_table": silver_table,
        "source_delta_version": source_delta_version,
        "train_cutoff": train_cutoff,
        "validation_cutoff": validation_cutoff,
        "config": asdict(config),
        "best_epoch_selected_on_validation": best_epoch,
    }
    benchmark_id = hashlib.sha256(
        json.dumps(definition, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    model_path = f"{model_artifact_path.rstrip('/')}/sasrec_encoder.pt"
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "config": asdict(config),
            "item_count": item_count,
            "benchmark_id": benchmark_id,
        },
        model_path,
    )
    vocabulary_rows = vocabulary.orderBy("item_index").collect()
    embedding_values = final_model.item_embedding.weight.detach().numpy()
    embedding_rows = [
        {
            "benchmark_id": benchmark_id,
            "parent_asin": row.parent_asin,
            "item_index": int(row.item_index),
            "features": [float(value) for value in embedding_values[int(row.item_index)]],
        }
        for row in vocabulary_rows
    ]
    spark.createDataFrame(embedding_rows).write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(f"{features_schema}.sasrec_item_embeddings")
    item_lookup = {int(row.item_index): row.parent_asin for row in vocabulary_rows}
    recommendation_rows: list[dict[str, Any]] = []
    for user_id, target, recommendations in zip(
        test[0], test[2], test_recommendations, strict=True
    ):
        for rank, (item_index, score) in enumerate(
            recommendations[: config.recommendation_k], start=1
        ):
            recommendation_rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "user_id": user_id,
                    "parent_asin": item_lookup[item_index],
                    "rank": rank,
                    "model_score": score,
                    "target_parent_asin": item_lookup[target],
                    "is_heldout_target": item_index == target,
                }
            )
    spark.createDataFrame(recommendation_rows).write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(f"{serving_schema}.sasrec_offline_recommendations")
    checks = {
        "global_temporal_cutoffs_are_ordered": train_cutoff < validation_cutoff,
        "training_examples_are_nonempty": bool(training_examples and final_examples),
        "validation_and_test_are_nonempty": bool(validation[0] and test[0]),
        "epoch_selection_uses_validation_only": 1 <= best_epoch <= config.epochs,
        "test_users_have_two_or_more_history_events": all(len(history) >= 2 for history in test[1]),
        "test_targets_are_novel_to_history": all(
            target not in history for history, target in zip(test[1], test[2], strict=True)
        ),
        "item_embeddings_cover_vocabulary": len(embedding_rows) == item_count,
        "recommendations_have_fixed_width": len(recommendation_rows)
        == len(test[0]) * config.recommendation_k,
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    summary: dict[str, Any] = {
        **definition,
        "benchmark_id": benchmark_id,
        "job_id": str(job_id),
        "job_run_id": str(job_run_id),
        "item_count": item_count,
        "training_users": len(training_sequences),
        "training_examples": len(training_examples),
        "final_training_users": len(final_sequences),
        "final_training_examples": len(final_examples),
        "validation_users": len(validation[0]),
        "test_users": len(test[0]),
        "validation": {
            "sasrec": validation_sasrec,
            "popularity": validation_popularity,
            "epoch_search": epoch_rows,
        },
        "test": {
            "sasrec": test_sasrec,
            "popularity": test_popularity,
            "paired_ndcg_uncertainty": uncertainty,
        },
        "artifacts": {
            "model_path": model_path,
            "vocabulary_table": f"{features_schema}.sasrec_item_vocabulary",
            "item_embeddings_table": f"{features_schema}.sasrec_item_embeddings",
            "recommendations_table": f"{serving_schema}.sasrec_offline_recommendations",
        },
        "checks": checks,
        "failed_checks": failed_checks,
    }
    summary_path = f"{model_artifact_path.rstrip('/')}/sasrec_benchmark_summary.json"
    Path(summary_path).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["artifacts"]["summary_path"] = summary_path
    lineage = _log_mlflow_model(
        mlflow,
        torch,
        final_model,
        positive,
        silver_table,
        source_delta_version,
        config,
        summary,
    )
    summary["mlflow_lineage"] = lineage
    summary["deployment_decision"] = {
        "status": "candidate" if lineage["alias"] == "candidate" else "rejected",
        "basis": "validation SASRec NDCG@10 must exceed popularity",
        "test_was_used_for_promotion": False,
    }
    Path(summary_path).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lineage_row = {
        "benchmark_id": benchmark_id,
        "job_id": str(job_id),
        "job_run_id": str(job_run_id),
        "source_table": silver_table,
        "source_delta_version": source_delta_version,
        "mlflow_run_id": lineage["run_id"],
        "registered_model_name": lineage["registered_model_name"],
        "registered_model_version": lineage["registered_model_version"],
        "model_alias": lineage["alias"] or "",
        "deployment_status": summary["deployment_decision"]["status"],
        "promotion_basis": lineage["promotion_basis"],
        "summary_json": json.dumps(summary, sort_keys=True, separators=(",", ":")),
        "created_at_epoch_ms": int(time.time() * 1_000),
    }
    spark.createDataFrame([lineage_row]).write.format("delta").mode("append").saveAsTable(
        f"{monitoring_schema}.model_deployment_lineage"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failed_checks:
        raise RuntimeError("SASRec certification failed: " + ", ".join(failed_checks))
    return summary
