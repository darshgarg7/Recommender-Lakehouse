import tempfile
import unittest
from pathlib import Path

from marketplace_recommender.governance.code_identity import source_tree_sha256


class SourceTreeIdentityTests(unittest.TestCase):
    def test_identity_is_order_independent_and_content_bound(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.py"
            second = root / "second.py"
            first.write_text("VALUE = 1\n", encoding="utf-8")
            second.write_text("VALUE = 2\n", encoding="utf-8")

            baseline = source_tree_sha256([first, second])
            self.assertEqual(baseline, source_tree_sha256([second, first]))

            second.write_text("VALUE = 3\n", encoding="utf-8")
            self.assertNotEqual(baseline, source_tree_sha256([first, second]))

    def test_identity_binds_file_names_and_rejects_an_empty_tree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.py"
            renamed = root / "renamed.py"
            first.write_text("VALUE = 1\n", encoding="utf-8")
            renamed.write_text("VALUE = 1\n", encoding="utf-8")
            self.assertNotEqual(source_tree_sha256([first]), source_tree_sha256([renamed]))
        with self.assertRaises(ValueError):
            source_tree_sha256([])


if __name__ == "__main__":
    unittest.main()
