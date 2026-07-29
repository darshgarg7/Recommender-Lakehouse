import tempfile
import unittest
from pathlib import Path

from marketplace_recommender.storage import read_jsonl, records_fingerprint, write_jsonl_atomic


class StorageTests(unittest.TestCase):
    def test_atomic_table_is_byte_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "table.jsonl"
            rows = [{"b": 2, "a": 1}, {"a": 2, "b": 1}]
            write_jsonl_atomic(path, rows)
            first = path.read_bytes()
            write_jsonl_atomic(path, rows)
            self.assertEqual(first, path.read_bytes())
            self.assertEqual(records_fingerprint(rows), records_fingerprint(read_jsonl(path)))


if __name__ == "__main__":
    unittest.main()
