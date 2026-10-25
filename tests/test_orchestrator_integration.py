import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class OrchestratorIntegrationTests(unittest.TestCase):
    def test_evaluation_without_selection_fails_fast(self):
        command = [
            sys.executable,
            "main.py",
            "unused.csv",
            "unused-results",
            "--fv", "1",
            "--fh", "1",
            "--ph", "2",
            "--time_col", "date",
            "--evaluation",
        ]
        process = subprocess.run(command, capture_output=True, text=True, timeout=30)

        self.assertNotEqual(process.returncode, 0)
        self.assertIn(
            "--evaluation requires --feature_selection",
            process.stderr,
        )

    def test_existing_cli_completes_with_development_only_tuning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "synthetic.csv"
            output_path = root / "results"
            samples = 96
            signal = np.sin(np.linspace(0, 8, samples))
            pd.DataFrame(
                {
                    "date": pd.date_range("2026-01-01", periods=samples, freq="h"),
                    "x": signal,
                    "target": np.roll(signal, 1),
                }
            ).to_csv(input_path, index=False)

            command = [
                sys.executable,
                "main.py",
                str(input_path),
                str(output_path),
                "--fv", "2",
                "--fh", "1",
                "--ph", "2",
                "--time_col", "date",
                "--feature_selection",
                "--fs_method", "pearson",
                "--fs_n_features", "2",
                "--evaluation",
                "--seed", "42",
                "--run_name", "integration",
            ]
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            run_dir = output_path / "synthetic_output" / "integration_s42"
            self.assertTrue((run_dir / "evaluation_report_pearson_knn.csv").exists())
            self.assertTrue(
                (run_dir / "MASTER_COMPARATIVE_REPORT_knn__fv2_fh1_ph2.csv").exists()
            )


if __name__ == "__main__":
    unittest.main()
