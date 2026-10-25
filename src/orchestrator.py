#!/usr/bin/env python3
"""
Main Time Series Framework.
Orchestrates data transformation, feature selection, and predictive evaluation.
"""

import os
import json
import time
import argparse
import tempfile
import random
import pandas as pd
import numpy as np
from datetime import datetime

from .config import Paths
from .transformer import transform_time_series
from .tuner import get_best_fs_params
from .tuner import tune_knn_k
from .feature_selection import select_features
from .evaluation import PredictiveEvaluator
from .logger_setup import setup_framework_logger


logger = setup_framework_logger(__name__)
# Reproducibility settings
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_unique_output_dir(output_arg, input_file, run_name=None, seed=None):
    """Generates a unique output path. Includes seed in name for multi-seed tracking."""
    base_path = Paths.RESULTS / output_arg
    dataset_name = os.path.splitext(os.path.basename(input_file))[0]
    final_base = base_path / f"{dataset_name}_output"

    suffix = f"_s{seed}" if seed is not None else ""
    if run_name:
        target_dir = final_base / f"{run_name}{suffix}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = final_base / f"run_{timestamp}{suffix}"
    
    return str(Paths.ensure_dir(target_dir))

def main():
    parser = argparse.ArgumentParser(description='Time Series Framework with Feature Selection.')
    
    # --- File and Path Arguments ---
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    parser.add_argument('output_dir', type=str, help='Base directory for results')
    parser.add_argument('--run_name', type=str, default=None, help='Custom experiment name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # --- Time Series Parameters ---
    parser.add_argument('--fv', type=int, required=True, help='Forecast Variable index (target column)')
    parser.add_argument('--fh', type=int, required=True, help='Forecast Horizon (future steps)')
    parser.add_argument('--ph', type=int, required=True, help='Past History (lag steps)')
    parser.add_argument('--time_col', type=str, required=True, help='Name of the timestamp column')
    
    # --- Feature Selection Configuration ---
    parser.add_argument('--feature_selection', action='store_true', help='Enable feature selection')
    parser.add_argument('--fs_method', type=str, default='random_forest',
                        choices=['pearson', 'ccf', 'mutual_info', 'random_forest', 'lasso', 
                                 'elastic_net', 'rfe', 'sequential', 'sklearn_filter', 'weka_inspired'],
                        help='Selection algorithm (Methods genetic, granger, info_gain, pca, spectral removed)')
    parser.add_argument('--fs_n_features', type=int, default=None, help='Number of features to select')
    
    # --- Specific Parameters ---
    parser.add_argument('--sklearn_method', type=str, default='selectkbest', 
                        choices=['selectkbest', 'selectpercentile', 'genericunivariateselect', 'variancethreshold'])
    parser.add_argument('--weka_inspired_method', type=str, default='cfs', 
                        choices=['cfs']) # infogain and relieff removed
    parser.add_argument('--alpha', type=float, default=None, help='Alpha for Lasso/ElasticNet')
    parser.add_argument('--seq_direction', type=str, default='forward', choices=['forward', 'backward'],
                        help='Direction for sequential feature selection')
    
    # --- Evaluation ---
    parser.add_argument('--evaluation', action='store_true', help='Execute final predictive evaluation')
   
    parser.add_argument('--evaluation_model', type=str, default='knn', choices=['knn', 'mlp', 'hgb'], 
                    help='Select model for final predictive evaluation')    

    args = parser.parse_args()

    # Reproducibility
    set_seed(args.seed)

    # Directory setup
    input_path = os.path.join("input_csv_files", args.input_file)
    output_path = get_unique_output_dir(args.output_dir, args.input_file, args.run_name, args.seed)
    os.makedirs(output_path, exist_ok=True)
    
    logger.info(f"Seed: {args.seed} | Experiment: {output_path}")

    # --- DATA LOADING ---
    df = pd.read_csv(input_path)
    
    # Timestamp column management
    if args.time_col not in df.columns:
        logger.warning(f"Column '{args.time_col}' not found. Generating artificial dates...")
        df[args.time_col] = pd.date_range(start='2000-01-01', periods=len(df), freq='h')
    else:
        df[args.time_col] = pd.to_datetime(df[args.time_col], errors='coerce')
        if df[args.time_col].isna().all():
            df[args.time_col] = pd.date_range(start='2000-01-01', periods=len(df), freq='h')
    
    target_col = df.columns[args.fv]
    logger.info(f"Target variable: {target_col}")

    # --- STEP 1: TRANSFORMATION (LAGS) ---
    dataset_output_root = os.path.dirname(output_path)
    transformed_file = os.path.join(dataset_output_root, f'transformed_fv{args.fv}_fh{args.fh}_ph{args.ph}.csv')
    time_meta_file = transformed_file + ".time.json"
    
    t_transform = 0
    if not os.path.exists(transformed_file):
        logger.info("Transforming time series...")
        start_t = time.time()
        transform_time_series(input_path, transformed_file, args.fv, args.fh, args.ph, args.time_col)
        t_transform = time.time() - start_t
        with open(time_meta_file, 'w') as f:
            json.dump({'t_transform': t_transform}, f)
    else:
        logger.info(f"Using existing transformation: {transformed_file}")
        if os.path.exists(time_meta_file):
            with open(time_meta_file, 'r') as f:
                t_transform = json.load(f).get('t_transform', 0)

    # --- TRAIN/TEST SPLIT (80/20 Temporal) ---
    df_full = pd.read_csv(transformed_file)
    split_idx = int(len(df_full) * 0.8)
    df_train_val = df_full.iloc[:split_idx]
    df_test = df_full.iloc[split_idx:]

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_csv = os.path.join(tmp_dir, "train.csv")
        test_csv = os.path.join(tmp_dir, "test.csv")
        df_train_val.to_csv(train_csv, index=False)
        df_test.to_csv(test_csv, index=False)

        # --- STEP 2: FEATURE SELECTION ---
        json_path = None
        selected_features = []
        t_optimization = 0

        if args.feature_selection:

            logger.info("Optimizing selection parameters...")
            # Note: We pass 'direction' explicitly for sequential in tuner
            tuned_params, t_optimization = get_best_fs_params(
                args.fs_method, train_csv, target_col, args.time_col,
                n_features=args.fs_n_features,
                sklearn_method=args.sklearn_method,
                weka_inspired_method=args.weka_inspired_method,
                direction=args.seq_direction
            )

            # Clean parameters for the final selector
            fs_kwargs = tuned_params.copy()
            fs_kwargs.pop('n_features', None)
            
            # --- PARAMETER CLEANUP BY METHOD ---
            # Define specific parameters to remove for simple methods to avoid conflicts
            redundant_params = ['sklearn_method', 'weka_inspired_method', 'n_neighbors', 
                               'alpha', 'direction']
            
            if args.fs_method in ['pearson', 'mutual_info', 'ccf', 'rfe']:
                for key in redundant_params:
                    fs_kwargs.pop(key, None)
            
            elif args.fs_method == 'sequential':
                # Sequential only needs 'direction' (kept from tuned_params) and maybe estimator
                # We remove the others
                for key in ['sklearn_method', 'weka_inspired_method', 'alpha']:
                    fs_kwargs.pop(key, None)

            logger.info(f"Running {args.fs_method}...")
            fs_results = select_features(
                train_csv,
                os.path.join(output_path, f'feature_selection_{args.fs_method}'),
                target_col,
                method=args.fs_method,
                time_col=args.time_col,
                n_features=args.fs_n_features,
                **fs_kwargs
            )
            
            selected_features = fs_results['selected_features']
            json_path = fs_results['json_metadata_path']
            logger.info(f"Selected features: {len(selected_features)}")

        # --- STEP 3: EVALUATION ---
        if args.evaluation and json_path:
            try:
                logger.info(f"{args.evaluation_model.upper()} Arbiter Evaluation")
                method_label = f"weka_{args.weka_inspired_method}" if args.fs_method == 'weka_inspired' else args.fs_method
                df_num = df_test.select_dtypes(include=[np.number])
                
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                
                target_cols = meta['target_columns']
                all_features = [c for c in df_num.columns if c not in target_cols]

                k_base, k_meth = 5, 5 # Default values
                if args.evaluation_model == 'knn':
                    # Tuning K for KNN
                    logger.info("Tuning K for Baseline and Method...")
                    k_base = tune_knn_k(df_num, all_features, target_cols)
                    k_meth = tune_knn_k(df_num, selected_features, target_cols)
                    logger.info(f"Optimal K values -> Baseline: {k_base}, Method: {k_meth}")
                else:
                    logger.info(f"Using default architectures for {args.evaluation_model.upper()} (Neutral mode)")

                evaluator = PredictiveEvaluator(model_type=args.evaluation_model, n_neighbors_baseline=k_base, n_neighbors_method=k_meth, seed=args.seed)
                extra_times = {'t_transform': t_transform, 't_optimization': t_optimization}
                
                report = evaluator.run_full_evaluation(test_csv, train_csv, json_path, args.time_col, extra_times=extra_times)
                
                # Save reports
                report.to_csv(os.path.join(output_path, f"evaluation_report_{method_label}_{args.evaluation_model}.csv"), index=False)
                evaluator.update_master_report(report, output_path, f"_fv{args.fv}_fh{args.fh}_ph{args.ph}")
                
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")

    logger.info("Process completed successfully.")

if __name__ == '__main__':
    main()