from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable


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

    class SasRecEncoder:  # type: ignore[no-redef]  # pragma: no cover
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
