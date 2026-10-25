import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.feature_selection import prepare_data_for_selection, select_features


class FeatureTargetTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=12, freq="h"),
                "x_t-1": range(12),
                "target_t-1": range(10, 22),
                "x": range(20, 32),
                "target": range(30, 42),
                "target_t+1": range(40, 52),
                "target_t+2": range(50, 62),
            }
        )

    def test_direct_horizon_is_label_and_all_targets_are_excluded_from_features(self):
        X, y = prepare_data_for_selection(
            self.frame.drop(columns=["date"]),
            target_col="target",
            selection_target_col="target_t+2",
        )

        self.assertEqual(y.name, "target_t+2")
        self.assertEqual(X.columns.tolist(), ["x_t-1", "target_t-1", "x"])

    def test_metadata_records_only_the_requested_direct_horizon(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "data.csv"
            output_path = Path(directory) / "selection"
            self.frame.to_csv(input_path, index=False)

            result = select_features(
                str(input_path),
                str(output_path),
                target_col="target",
                method="pearson",
                n_features=2,
                time_col="date",
                selection_target_col="target_t+2",
                evaluation_target_cols=["target_t+2"],
                generate_report=False,
                generate_filtered_csv=False,
                verbose=False,
            )

            with open(result["json_metadata_path"], encoding="utf-8") as stream:
                metadata = json.load(stream)

        self.assertEqual(metadata["target_columns"], ["target_t+2"])
        self.assertEqual(
            metadata["candidate_features"],
            ["x_t-1", "target_t-1", "x"],
        )
        self.assertNotIn("target", metadata["selected_features"])
        self.assertNotIn("target_t+1", metadata["selected_features"])
        self.assertNotIn("target_t+2", metadata["selected_features"])


if __name__ == "__main__":
    unittest.main()
