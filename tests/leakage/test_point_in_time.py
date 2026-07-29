import unittest

from marketplace_recommender.features.point_in_time import assert_point_in_time, latest_before


class LeakageTests(unittest.TestCase):
    def test_asof_join_is_strict(self):
        features = [
            {"id": "x", "feature_timestamp": 10, "value": "past"},
            {"id": "x", "feature_timestamp": 20, "value": "equal"},
        ]
        joined = latest_before(features, [{"id": "x", "label_timestamp": 20}], "id")
        self.assertEqual(joined[0]["value"], "past")

    def test_future_history_is_rejected(self):
        with self.assertRaises(AssertionError):
            assert_point_in_time(
                [
                    {
                        "label_timestamp": 20,
                        "feature_timestamp": 10,
                        "historical_event_times": [19, 21],
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
