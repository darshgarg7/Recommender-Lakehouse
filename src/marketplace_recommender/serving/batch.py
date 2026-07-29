from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from marketplace_recommender.ranking.features import candidate_features
from marketplace_recommender.ranking.lambdamart import PairwiseLinearRanker
from marketplace_recommender.ranking.reranker import rerank
from marketplace_recommender.retrieval.ann import ExactANNIndex
from marketplace_recommender.retrieval.candidates import CandidateGenerator
from marketplace_recommender.retrieval.capabilities import resolve_capabilities
from marketplace_recommender.retrieval.content import (
    build_content_embeddings,
    content_candidates,
    user_content_profile,
)
from marketplace_recommender.retrieval.popularity import popularity_scores, recommend_popular
from marketplace_recommender.retrieval.sasrec import SequentialCooccurrenceTeacher
from marketplace_recommender.retrieval.two_tower import HybridTwoTower
from marketplace_recommender.retrieval.vectors import dot
from marketplace_recommender.schemas import cold_start_bucket


@dataclass
class RecommendationSystem:
    interactions: list[dict[str, Any]]
    products: dict[str, dict[str, Any]]
    content_embeddings: dict[str, list[float]]
    cutoff: int
    tower: HybridTwoTower
    generator: CandidateGenerator
    ranker: PairwiseLinearRanker

    @classmethod
    def fit(
        cls,
        interactions: list[dict[str, Any]],
        products: list[dict[str, Any]],
        content_rows: list[dict[str, Any]],
        graph_rows: Iterable[dict[str, Any]],
        cutoff: int,
        seed: int,
    ) -> "RecommendationSystem":
        product_map = {row["parent_asin"]: row for row in products}
        content = build_content_embeddings(content_rows)
        has_content = {row["parent_asin"]: bool(row["has_content"]) for row in content_rows}
        teacher = SequentialCooccurrenceTeacher().fit(interactions, cutoff)
        tower = HybridTwoTower.fit(content, teacher.transitions, interactions, cutoff, has_content)
        item_vectors = {item: tower.item_embedding(item) for item in product_map}
        domains = {item: row["domain"] for item, row in product_map.items()}
        graph: dict[str, set[str]] = defaultdict(set)
        for row in graph_rows:
            graph[row["parent_asin"]].add(row["related_parent_asin"])
        popularity = popularity_scores(interactions, cutoff)
        generator = CandidateGenerator(
            tower=tower,
            ann=ExactANNIndex(item_vectors, domains),
            teacher=teacher,
            popularity=popularity,
            graph=graph,
        )
        return cls(
            interactions=interactions,
            products=product_map,
            content_embeddings=content,
            cutoff=cutoff,
            tower=tower,
            generator=generator,
            ranker=PairwiseLinearRanker(seed=seed),
        )

    def history(self, user_id: str, timestamp: int) -> list[str]:
        rows = sorted(
            (
                row
                for row in self.interactions
                if row["user_id"] == user_id
                and row["review_timestamp"] < timestamp
                and row["verified_purchase"]
                and row["rating"] >= 4
            ),
            key=lambda row: (row["review_timestamp"], row["interaction_id"]),
        )
        return [row["parent_asin"] for row in rows][-100:]

    def preferences(self, history: list[str]) -> dict[str, list[str]]:
        domains: Counter[str] = Counter()
        categories: Counter[str] = Counter()
        brands: Counter[str] = Counter()
        for item in history:
            product = self.products.get(item, {})
            domains.update([product.get("domain", "")])
            categories.update(product.get("category_path", []))
            brands.update([product.get("brand_or_store", "")])
        return {
            "preferred_domains": [value for value, _ in domains.most_common(3) if value],
            "preferred_categories": [value for value, _ in categories.most_common(5) if value],
            "preferred_brands": [value for value, _ in brands.most_common(5) if value],
        }

    def feature_candidates(
        self, user_id: str, timestamp: int, candidate_limit: int, domain: str | None = None
    ) -> list[dict[str, Any]]:
        history = self.history(user_id, timestamp)
        preferences = self.preferences(history)
        candidates = self.generator.generate(history, candidate_limit, domain)
        output = []
        for candidate in candidates:
            item = candidate["parent_asin"]
            product = self.products[item]
            features = candidate_features(
                candidate, history, self.tower, product, preferences, self.generator.popularity
            )
            capabilities = resolve_capabilities(
                content_available=self.tower.has_content.get(item, False),
                behavioral_events=self.tower.interaction_counts.get(item, 0),
                observed_retrieval_channels=tuple(candidate.get("retrieval_scores", {})),
            )
            output.append(
                {**candidate, **product, **capabilities.as_record(), "features": features}
            )
        return output

    def fit_ranker(
        self,
        labels: Iterable[dict[str, Any]],
        candidate_limit: int,
        negative_count: int | None = None,
    ) -> int:
        label_rows = list(labels)
        horizon_end = max((row["label_timestamp"] for row in label_rows), default=self.cutoff)
        training_rows = []
        for label in label_rows:
            if label["label"] != 1:
                continue
            future_positives = {
                row["parent_asin"]
                for row in self.interactions
                if row["user_id"] == label["user_id"]
                and label["label_timestamp"] <= row["review_timestamp"] <= horizon_end
                and row["verified_purchase"]
                and row["rating"] >= 4
            }
            candidates = self.feature_candidates(
                label["user_id"], label["label_timestamp"], candidate_limit
            )
            if not any(row["parent_asin"] == label["parent_asin"] for row in candidates):
                continue
            kept_negatives = 0
            for candidate in candidates:
                if (
                    candidate["parent_asin"] != label["parent_asin"]
                    and candidate["parent_asin"] in future_positives
                ):
                    continue
                is_positive = candidate["parent_asin"] == label["parent_asin"]
                if not is_positive:
                    if negative_count is not None and kept_negatives >= negative_count:
                        continue
                    kept_negatives += 1
                training_rows.append(
                    {
                        "group_id": label["interaction_id"],
                        "label": int(is_positive),
                        "features": candidate["features"],
                    }
                )
        self.ranker.fit(training_rows)
        return len(training_rows)

    def recommend(
        self,
        user_id: str,
        timestamp: int,
        candidate_limit: int,
        limit: int,
        rerank_config: dict[str, Any],
        domain: str | None = None,
        model_version: str = "local-hybrid-v1",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        candidates = self.feature_candidates(user_id, timestamp, candidate_limit, domain)
        for candidate in candidates:
            item = candidate["parent_asin"]
            candidate["ranking_score"] = self.ranker.predict(candidate["features"])
            candidate["novelty"] = candidate["features"]["novelty"]
            evidence_count = self.tower.interaction_counts.get(item, 0)
            candidate["behavioral_evidence_count"] = evidence_count
            candidate["collaboration_weight"] = candidate["features"]["cold_start_gate"]
            candidate["cold_start_bucket"] = cold_start_bucket(evidence_count)
        relevance_order = sorted(
            candidates, key=lambda row: (-row["ranking_score"], row["parent_asin"])
        )
        final = rerank(
            relevance_order,
            self.generator.ann.vectors,
            limit=limit,
            novelty_weight=float(rerank_config["novelty_weight"]),
            long_tail_weight=float(rerank_config["long_tail_weight"]),
            redundancy_weight=float(rerank_config["redundancy_weight"]),
            max_per_brand=int(rerank_config["max_per_brand"]),
            max_score_regret=float(rerank_config["max_score_regret"]),
        )
        generated_at = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).isoformat()
        for row in final:
            row["user_id"] = user_id
            row["model_version"] = model_version
            row["feature_timestamp"] = timestamp
            row["generated_at"] = generated_at
        return relevance_order, final

    def recommend_champion(
        self,
        champion: str,
        user_id: str,
        timestamp: int,
        candidate_limit: int,
        limit: int,
        rerank_config: dict[str, Any],
        promotion_policy_id: str,
    ) -> list[dict[str, Any]]:
        """Generate batch rows from the model selected by the promotion decision."""
        generated_at = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).isoformat()
        if champion in {"full_reranked", "hybrid_ranker"}:
            relevance, reranked = self.recommend(
                user_id,
                timestamp,
                candidate_limit,
                limit,
                rerank_config,
                model_version=f"local-{champion}-v1",
            )
            rows = reranked if champion == "full_reranked" else relevance[:limit]
            for rank, row in enumerate(rows, start=1):
                row["rank"] = rank
                row.setdefault("final_score", row["ranking_score"])
                row.setdefault("score_regret", 0.0)
                row.setdefault("max_score_regret", 0.0)
                row.setdefault("decision_reason", "promoted_learned_relevance")
        elif champion in {"popularity", "content_similarity"}:
            history = self.history(user_id, timestamp)
            ranked = self.baseline_rankings(user_id, timestamp, limit)[champion]
            profile = user_content_profile(history, self.content_embeddings)
            rows = []
            for rank, item in enumerate(ranked, start=1):
                score = (
                    self.generator.popularity.get(item, 0.0)
                    if champion == "popularity"
                    else dot(profile, self.content_embeddings[item])
                )
                evidence_count = self.tower.interaction_counts.get(item, 0)
                capabilities = resolve_capabilities(
                    content_available=self.tower.has_content.get(item, False),
                    behavioral_events=evidence_count,
                    observed_retrieval_channels=(champion,),
                )
                rows.append(
                    {
                        **self.products[item],
                        **capabilities.as_record(),
                        "parent_asin": item,
                        "rank": rank,
                        "retrieval_score": score,
                        "ranking_score": score,
                        "final_score": score,
                        "recommendation_channel": champion,
                        "cold_start_bucket": cold_start_bucket(evidence_count),
                        "behavioral_evidence_count": evidence_count,
                        "collaboration_weight": (
                            0.0
                            if champion == "content_similarity"
                            else self.tower.interaction_counts.get(item, 0)
                            / (self.tower.interaction_counts.get(item, 0) + 10.0)
                        ),
                        "score_regret": 0.0,
                        "max_score_regret": 0.0,
                        "decision_reason": "promotion_fallback",
                    }
                )
        else:
            raise ValueError(f"unsupported serving champion: {champion}")
        for row in rows:
            row["user_id"] = user_id
            row["model_version"] = f"local-{champion}-v1"
            row["serving_champion"] = champion
            row["promotion_policy_id"] = promotion_policy_id
            row["feature_timestamp"] = timestamp
            row["generated_at"] = generated_at
        return rows

    def baseline_rankings(self, user_id: str, timestamp: int, limit: int) -> dict[str, list[str]]:
        history = self.history(user_id, timestamp)
        seen = set(history)
        popular = [item for item, _ in recommend_popular(self.generator.popularity, seen, limit)]
        content_profile = user_content_profile(history, self.content_embeddings)
        content = [
            item
            for item, _ in content_candidates(content_profile, self.content_embeddings, seen, limit)
        ]
        return {"popularity": popular, "content_similarity": content}


def serving_projection(row: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "user_id",
        "rank",
        "parent_asin",
        "retrieval_score",
        "ranking_score",
        "final_score",
        "recommendation_channel",
        "cold_start_bucket",
        "behavioral_evidence_count",
        "collaboration_weight",
        "content_available",
        "observed_retrieval_channels",
        "representation_strategy",
        "evidence_capabilities",
        "serving_champion",
        "promotion_policy_id",
        "decision_reason",
        "score_regret",
        "max_score_regret",
        "model_version",
        "feature_timestamp",
        "generated_at",
    ]
    return {field: row.get(field) for field in fields}
