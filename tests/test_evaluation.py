import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.evaluation import PredictiveEvaluator


class EvaluationFeatureSpaceTests(unittest.TestCase):
    def test_baseline_uses_recorded_candidate_features(self):
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=12, freq="h"),
                "x": range(12),
                "target_t-1": range(10, 22),
                "target": range(20, 32),
                "target_t+1": range(30, 42),
                "target_t+2": range(40, 52),
            }
        )
        metadata = {
            "experiment_info": {"method": "pearson"},
            "selected_features": ["x"],
            "candidate_features": ["x", "target_t-1"],
            "target_columns": ["target_t+2"],
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.csv"
            test_path = root / "test.csv"
            metadata_path = root / "metadata.json"
            frame.iloc[:8].to_csv(train_path, index=False)
            frame.iloc[8:].to_csv(test_path, index=False)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            evaluator = PredictiveEvaluator()
            with patch.object(
                evaluator,
                "_evaluate_model",
                return_value={
                    "rmse": 1.0,
                    "mae": 1.0,
                    "r2": 0.0,
                    "t_train": 0.1,
                    "t_inference": 0.1,
                },
            ) as evaluate:
                evaluator.run_full_evaluation(
                    str(test_path),
                    str(train_path),
                    str(metadata_path),
                    "date",
                )

        baseline_features = evaluate.call_args_list[0].args[2]
        self.assertEqual(baseline_features, ["x", "target_t-1"])
        self.assertNotIn("target", baseline_features)
        self.assertNotIn("target_t+1", baseline_features)


if __name__ == "__main__":
    unittest.main()
