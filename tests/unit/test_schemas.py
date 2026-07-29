import unittest

from marketplace_recommender.schemas import cold_start_bucket, normalize_timestamp, validate_rating


class SchemaTests(unittest.TestCase):
    def test_timestamp_normalization_accepts_seconds_milliseconds_and_iso(self):
        self.assertEqual(normalize_timestamp(1_600_000_000), 1_600_000_000_000)
        self.assertEqual(normalize_timestamp(1_600_000_000_000), 1_600_000_000_000)
        self.assertEqual(normalize_timestamp("2020-09-13T12:26:40Z"), 1_600_000_000_000)

    def test_rating_contract(self):
        self.assertEqual(validate_rating("4"), 4.0)
        with self.assertRaises(ValueError):
            validate_rating(6)

    def test_cold_start_boundaries(self):
        expected = {
            0: "zero-history",
            1: "sparse",
            10: "sparse",
            11: "developing",
            100: "developing",
            101: "warm",
        }
        self.assertEqual({value: cold_start_bucket(value) for value in expected}, expected)


if __name__ == "__main__":
    unittest.main()
