from __future__ import annotations

import unittest

from marketplace_recommender.retrieval.ann import (
    ANNIndexFactory,
    ExactANNIndex,
    SubspacePartitionedANNIndex,
)
from marketplace_recommender.retrieval.vectors import hashed_text_vector


class AnnScalabilityTests(unittest.TestCase):
    def test_subspace_partitioned_ann_index_query(self) -> None:
        items = {
            f"item_{i}": hashed_text_vector(f"product category description {i}") for i in range(200)
        }
        domains = {f"item_{i}": "Electronics" if i % 2 == 0 else "Home" for i in range(200)}

        exact_index = ExactANNIndex(vectors=items, domains=domains)
        partitioned_index = SubspacePartitionedANNIndex(
            vectors=items, domains=domains, num_buckets=8, n_probes=4, min_exact_threshold=32
        )

        query_vec = hashed_text_vector("product category description 10")
        exact_res = exact_index.query(query_vec, limit=5, domain="Electronics")
        partitioned_res = partitioned_index.query(query_vec, limit=5, domain="Electronics")

        self.assertTrue(partitioned_res)
        self.assertEqual(partitioned_res[0][0], exact_res[0][0])
        self.assertLess(partitioned_index.query_stats["evaluations"], len(items))

    def test_ann_factory_creation(self) -> None:
        items = {f"item_{i}": [float(i)] * 4 for i in range(100)}
        domains = {f"item_{i}": "default" for i in range(100)}

        index_exact = ANNIndexFactory.create(items, domains, scalable=False)
        self.assertIsInstance(index_exact, ExactANNIndex)

        index_scalable = ANNIndexFactory.create(items, domains, scalable=True)
        self.assertIsInstance(index_scalable, SubspacePartitionedANNIndex)


if __name__ == "__main__":
    unittest.main()
