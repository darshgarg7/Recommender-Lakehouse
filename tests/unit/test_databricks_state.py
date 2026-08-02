from __future__ import annotations

import unittest

from marketplace_recommender.pipelines.databricks import (
    BRONZE_IDENTITY_CONTRACT,
    BRONZE_STATE_CONTRACT,
    _bronze_state_paths,
)


class DatabricksStateTests(unittest.TestCase):
    def test_auto_loader_state_is_versioned_by_source_contract(self) -> None:
        schema, checkpoint = _bronze_state_paths(
            "/Volumes/workspace/default/marketplace_landing/reviews/",
            "reviews",
        )

        expected_root = (
            f"/Volumes/workspace/default/marketplace_landing/_state/{BRONZE_STATE_CONTRACT}"
        )
        self.assertEqual(schema, f"{expected_root}/schemas/reviews")
        self.assertEqual(checkpoint, f"{expected_root}/checkpoints/reviews")
        self.assertEqual(BRONZE_STATE_CONTRACT, "raw_text_v1")
        self.assertEqual(BRONZE_IDENTITY_CONTRACT, "content-sha256-generation-v2")


if __name__ == "__main__":
    unittest.main()
