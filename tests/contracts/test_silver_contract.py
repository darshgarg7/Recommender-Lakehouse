import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from marketplace_recommender.cli import run_demo
from marketplace_recommender.storage import read_jsonl

from tests.integration.test_vertical_slice import CONFIG


class SilverContractTests(unittest.TestCase):
    def test_required_keys_and_ranges(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.yml"
            config.write_text(CONFIG.format(output=root / "output"), encoding="utf-8")
            with redirect_stdout(StringIO()):
                run_demo(config)
            rows = list(read_jsonl(root / "output/silver/silver_interactions.jsonl"))
            required = {"interaction_id", "user_id", "asin", "parent_asin", "rating", "source_file"}
            self.assertTrue(rows)
            self.assertTrue(all(required <= row.keys() for row in rows))
            self.assertTrue(all(1 <= row["rating"] <= 5 for row in rows))
            self.assertEqual(len({row["interaction_id"] for row in rows}), len(rows))


if __name__ == "__main__":
    unittest.main()
