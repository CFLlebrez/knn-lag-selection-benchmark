import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.feature_selection import create_feature_selector
from src.tuner import get_best_fs_params, quick_cv_eval


class SelectorParameterTests(unittest.TestCase):
    def test_explicit_lasso_alpha_is_used(self):
        selector = create_feature_selector("lasso", n_features=1, alpha=0.25)
        X = pd.DataFrame({"x": np.arange(12, dtype=float)})
        y = pd.Series(np.arange(12, dtype=float))

        selector.fit(X, y)

        self.assertEqual(selector.model.alpha, 0.25)

    def test_embedded_selector_cv_is_temporal(self):
        for method in ("lasso", "elastic_net", "sequential"):
            with self.subTest(method=method):
                selector = create_feature_selector(method, n_features=2)
                self.assertEqual(selector.cv.__class__.__name__, "TimeSeriesSplit")

    def test_mutual_information_receives_tuned_parameters(self):
        selector = create_feature_selector(
            "mutual_info",
            n_features=3,
            n_neighbors=11,
            random_state=123,
        )

        self.assertEqual(selector.n_neighbors, 11)
        self.assertEqual(selector.random_state, 123)


class SelectorTuningTests(unittest.TestCase):
    def test_selector_is_fitted_inside_each_temporal_training_fold(self):
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=48, freq="h"),
                "x": np.linspace(0, 1, 48),
                "target": np.linspace(0, 2, 48),
            }
        )
        observed_training_lengths = []

        def fake_select_features(input_file, *args, **kwargs):
            observed_training_lengths.append(len(pd.read_csv(input_file)))
            return {"selected_features": ["x"]}

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "development.csv"
            frame.to_csv(input_path, index=False)

            with patch("src.tuner.select_features", side_effect=fake_select_features):
                params, _ = get_best_fs_params(
                    "mutual_info",
                    str(input_path),
                    target_col="target",
                    time_col="date",
                    n_features=1,
                    n_trials=1,
                    cv_gap=2,
                    seed=7,
                )

        self.assertEqual(len(observed_training_lengths), 3)
        self.assertTrue(all(length < len(frame) for length in observed_training_lengths))
        self.assertEqual(params["random_state"], 7)
        self.assertIn("n_neighbors", params)

    def test_knn_cv_is_reproducible_with_a_temporal_gap(self):
        X = pd.DataFrame({"x": np.arange(40, dtype=float)})
        y = pd.Series(np.arange(40, dtype=float))

        first = quick_cv_eval(X, y, k=3, gap=2)
        second = quick_cv_eval(X, y, k=3, gap=2)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
