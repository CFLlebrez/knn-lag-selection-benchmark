import unittest

import pandas as pd

from src.experiment_config import ForecastSpec, SplitSpec
from src.temporal_protocol import make_development_cv, make_temporal_holdout


class TemporalHoldoutTests(unittest.TestCase):
    def test_positive_horizon_purges_crossing_labels(self):
        frame = pd.DataFrame({"origin": range(10)})

        holdout = make_temporal_holdout(
            frame,
            ForecastSpec.path(2),
            SplitSpec(test_fraction=0.2),
        )

        self.assertEqual(holdout.test_start, 8)
        self.assertEqual(holdout.purged_rows, 2)
        self.assertEqual(holdout.development["origin"].tolist(), list(range(6)))
        self.assertEqual(holdout.test["origin"].tolist(), [8, 9])
        self.assertLess(
            holdout.development["origin"].max() + max(ForecastSpec.path(2).steps),
            holdout.test["origin"].min(),
        )

    def test_immediate_target_needs_no_purge(self):
        frame = pd.DataFrame({"origin": range(10)})

        holdout = make_temporal_holdout(
            frame,
            ForecastSpec.immediate(),
            SplitSpec(test_fraction=0.2),
        )

        self.assertEqual(holdout.purged_rows, 0)
        self.assertEqual(len(holdout.development), 8)
        self.assertEqual(len(holdout.test), 2)

    def test_rejects_split_when_purge_consumes_development_data(self):
        frame = pd.DataFrame({"origin": range(5)})

        with self.assertRaises(ValueError):
            make_temporal_holdout(
                frame,
                ForecastSpec.direct(4),
                SplitSpec(test_fraction=0.4),
            )


class DevelopmentCVTests(unittest.TestCase):
    def test_cv_gap_matches_largest_forecast_step(self):
        forecast = ForecastSpec(steps=(1, 3, 6))
        splitter = make_development_cv(forecast, SplitSpec(cv_splits=3))
        frame = pd.DataFrame({"value": range(40)})

        for train_indices, validation_indices in splitter.split(frame):
            self.assertLess(
                train_indices.max() + max(forecast.steps),
                validation_indices.min(),
            )


if __name__ == "__main__":
    unittest.main()
