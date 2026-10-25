#!/usr/bin/env python3
"""
Complete Smoke Test Battery for Feature Selection Framework.
Validates all implemented selection methods against synthetic ground truth.
"""

import sys
import pandas as pd
import numpy as np
import os
import subprocess
import json

from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))

class SmokeTester:
    def __init__(self):
        self.current_dir = str(root_path)
        self.input_dir = os.path.join(self.current_dir, "input_csv_files")
        self.output_base = os.path.join(self.current_dir, "results", "smoke_tests")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_base, exist_ok=True)

    def generate_synthetic_data(self):
        """Generates linear and non-linear datasets for validation."""
        n_samples = 500
        
        # Dataset 1: Linear (x1, x2 are the ground truth)
        df_lin = pd.DataFrame({
            'fecha': pd.date_range('2023-01-01', periods=n_samples, freq='h'),
            'x1': np.random.randn(n_samples),
            'x2': np.random.randn(n_samples),
            'x3_noise': np.random.uniform(0, 100, n_samples),
            'x4_noise': np.random.normal(0, 50, n_samples),
            'target': 0.0
        })
        df_lin['target'] = 2.5 * df_lin['x1'] - 1.2 * df_lin['x2'] + np.random.normal(0, 0.05, n_samples)
        df_lin.to_csv(os.path.join(self.input_dir, 'test_lineal.csv'), index=False)

        # Dataset 2: Non-Linear (x_val is the ground truth)
        t = np.linspace(0, 100, n_samples)
        df_nonlin = pd.DataFrame({
            'fecha': pd.date_range('2023-01-01', periods=n_samples, freq='h'),
            'x_val': np.sin(t),
            'x_random': np.random.randn(n_samples),
            'target': np.sin(t) + np.random.normal(0, 0.02, n_samples)
        })
        df_nonlin.to_csv(os.path.join(self.input_dir, 'test_sinusoidal.csv'), index=False)

    def run_single_test(self, name, method, n_expected, fv, weka_submethod=None):
        """Executes the framework for a specific method and validates output."""
        input_path = os.path.abspath(os.path.join(self.input_dir, f"{name}.csv"))
        run_id = f"smoke_test_{method}"
        if weka_submethod:
            run_id += f"_{weka_submethod}"

        main_script = os.path.join(self.current_dir, "main.py")

        cmd = [
            sys.executable, main_script,
            input_path, self.output_base,
            "--fv", str(fv), "--fh", "0", "--ph", "1",
            "--time_col", "fecha",
            "--feature_selection",
            "--run_name", run_id,
            "--fs_method", method,
            "--fs_n_features", str(n_expected),
            "--evaluation"
        ]

        # Add specific sub-parameter for Weka methods
        if method == "weka_inspired" and weka_submethod:
            cmd.extend(["--weka_inspired_method", weka_submethod])
        
        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                print(f"Error executing {method}: {process.stderr}")
                return False, []
            
            json_path = os.path.join(
                self.output_base, f"{name}_output", run_id, 
                f"feature_selection_{method}", f"selection_metadata_{method}.json"
            )

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                selected = meta.get('selected_features', [])
                # Filter out the time column from count if present
                selected = [f for f in selected if f.lower() != 'fecha']
                return (len(selected) == n_expected), selected
            return False, []
        except Exception as e:
            print(f"Exception: {e}")
            return False, []

    def run_battery(self):
        """Runs the complete suite of tests including all implemented methods."""
        self.generate_synthetic_data()

        # Matrix: (Dataset, FV, Method, Expected_N, Weka_Submethod)
        tests = [
            # --- Linear Methods ---
            ("test_lineal", 5, "lasso", 2, None),
            ("test_lineal", 5, "elastic_net", 2, None),
            ("test_lineal", 5, "pearson", 2, None),
            ("test_lineal", 5, "ccf", 2, None),
            
            # --- Weka-Inspired (CFS) ---
            ("test_lineal", 5, "weka_inspired", 2, "cfs"),
            
            # --- Wrappers / Sequential ---
            ("test_lineal", 5, "sequential", 2, None),
            ("test_lineal", 5, "rfe", 2, None),
            
            # --- Non-Linear / Tree-Based ---
            ("test_sinusoidal", 3, "random_forest", 1, None),
            ("test_sinusoidal", 3, "mutual_info", 1, None),
            
            # --- Generic Sklearn Filter ---
            ("test_lineal", 5, "sklearn_filter", 2, None)
        ]

        results = []
        for d_name, fv, method, n_exp, weka_sub in tests:
            display_name = f"{method} ({weka_sub})" if weka_sub else method
            print(f"Running: {display_name} on {d_name}...", end=" ", flush=True)
            
            success, selected = self.run_single_test(d_name, method, n_exp, fv, weka_sub)
            status = "PASS ✅" if success else "FAIL ❌"
            print(status)
            
            results.append({
                "Dataset": d_name,
                "Method": display_name,
                "Status": status,
                "Features": ", ".join(selected) if selected else "None"
            })
        
        print("\n" + "="*75)
        print("FINAL SMOKE TEST SUMMARY")
        print("="*75)
        print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    tester = SmokeTester()
    tester.run_battery()