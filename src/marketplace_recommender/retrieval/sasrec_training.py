from __future__ import annotations

import copy
import json
from typing import Any

from marketplace_recommender.retrieval.sasrec_model import (
    SasRecConfig,
    _negative_samples,
    _pad,
    ranking_metrics_from_rank,
)


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
