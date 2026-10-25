import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.transformer import transform_time_series


class TransformerAlignmentTests(unittest.TestCase):
    def _transform(self, frame, fv, fh, ph, time_col="date"):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            output_path = Path(directory) / "output.csv"
            frame.to_csv(input_path, index=False)
            transform_time_series(
                str(input_path),
                str(output_path),
                fv=fv,
                fh=fh,
                ph=ph,
                time_col=time_col,
            )
            return pd.read_csv(output_path)

    def test_lags_present_and_future_targets_are_aligned(self):
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=5, freq="h"),
                "x": [10, 11, 12, 13, 14],
                "target": [20, 21, 22, 23, 24],
            }
        )

        result = self._transform(frame, fv=2, fh=2, ph=1)

        self.assertEqual(len(result), 2)
        self.assertEqual(result.loc[0, "x_t-1"], 10)
        self.assertEqual(result.loc[0, "target_t-1"], 20)
        self.assertEqual(result.loc[0, "x"], 11)
        self.assertEqual(result.loc[0, "target"], 21)
        self.assertEqual(result.loc[0, "target_t+1"], 22)
        self.assertEqual(result.loc[0, "target_t+2"], 23)

    def test_target_index_does_not_assume_time_column_is_first(self):
        frame = pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "target": [10, 20, 30, 40],
                "date": pd.date_range("2026-01-01", periods=4, freq="h"),
            }
        )

        result = self._transform(frame, fv=1, fh=1, ph=1)

        self.assertIn("target_t+1", result.columns)
        self.assertNotIn("x_t+1", result.columns)
        self.assertEqual(result.loc[0, "target_t+1"], 30)

    def test_rows_are_sorted_before_windows_are_created(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2026-01-01 02:00", "2026-01-01 00:00", "2026-01-01 01:00"]
                ),
                "x": [2, 0, 1],
                "target": [12, 10, 11],
            }
        )

        result = self._transform(frame, fv=2, fh=0, ph=1)

        self.assertEqual(result.loc[0, "x_t-1"], 0)
        self.assertEqual(result.loc[0, "x"], 1)
        self.assertEqual(result.loc[0, "target"], 11)


if __name__ == "__main__":
    unittest.main()
