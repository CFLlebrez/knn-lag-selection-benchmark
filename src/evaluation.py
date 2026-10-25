#!/usr/bin/env python3
"""
Predictive performance evaluator for feature selection methods.
Compares baseline (all features) vs. filtered subsets using KNN.
"""

import pandas as pd
import numpy as np
import json
import os
import time
from .logger_setup import setup_framework_logger
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = setup_framework_logger(__name__)
class PredictiveEvaluator:
    def __init__(self, model_type='knn', n_neighbors_baseline=5, n_neighbors_method=5, seed=42):
        """
        Initializes the evaluator with temporal validation settings.
        """
        self.model_type = model_type
        self.n_neighbors_baseline = n_neighbors_baseline
        self.n_neighbors_method = n_neighbors_method
        self.seed = seed
        self.method_times = {}
        self.baseline_times = {}

    def _get_model(self, is_baseline=False):
        """
        Factory method to return the regressor based on the selected model_type.
        """
        logger.info(f"INITIALIZING {self.model_type} WITH SEED {self.seed}")
        if self.model_type == 'knn':
            k = self.n_neighbors_baseline if is_baseline else self.n_neighbors_method
            return KNeighborsRegressor(n_neighbors=k)
        
        elif self.model_type == 'mlp':
            return MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=self.seed)
            
        elif self.model_type == 'hgb':
            return HistGradientBoostingRegressor(random_state=self.seed)
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
    def run_full_evaluation(self, csv_test_path, csv_train_path, json_metadata_path, time_col, extra_times=None):
        """
        Loads metadata and data to execute a comparative evaluation.
        """
        if extra_times is None: 
            extra_times = {}

        if not os.path.exists(json_metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {json_metadata_path}")
            
        with open(json_metadata_path, 'r') as f:
            meta = json.load(f)
        
        selected_features = meta['selected_features']
        target_columns = meta['target_columns']
        # Mantenemos la estructura de acceso a meta
        method_name = meta['experiment_info']['method']
        t_selection = meta.get('selection_time_seconds', 0)
        
        df_test = pd.read_csv(csv_test_path)
        df_train = pd.read_csv(csv_train_path)

        time_col_lower = time_col.lower() if time_col else None

        # Exclude targets and time column
        all_features = [
            c for c in df_train.columns 
            if c not in target_columns and c.lower() != time_col_lower
        ]
        
        selected_features = [
            c for c in selected_features 
            if c.lower() != time_col_lower
        ]
        
        logger.info(f"===PREDICTION EVALUATION SYSTEM (Arbiter: {self.model_type.upper()})===")
        
        # 4. Evaluate Baseline
        logger.info(f"Evaluating Baseline...")
        baseline_results = self._evaluate_model(df_train, df_test, all_features, target_columns, is_baseline=True)
        
        # 5. Evaluate Method
        logger.info(f"Evaluating Selection: {method_name}...")
        method_results = self._evaluate_model(df_train, df_test, selected_features, target_columns, is_baseline=False)
        
        self.method_times = {
            'T_Transform': extra_times.get('t_transform', 0),
            'T_Optimization': extra_times.get('t_optimization', 0),
            'T_Selection': t_selection,
            'T_Train': method_results['t_train'],
            'T_Inference': method_results['t_inference']
        }

        self.baseline_times = {
            'T_Transform': extra_times.get('t_transform', 0),
            'T_Optimization': extra_times.get('t_optimization', 0),
            'T_Selection': t_selection,
            'T_Train': baseline_results['t_train'],
            'T_Inference': baseline_results['t_inference']
        }

        return self._generate_final_report(
            baseline_results, 
            method_results, 
            method_name, 
            len(all_features), 
            len(selected_features)
        )

    def _evaluate_model(self, df_train, df_test, feature_cols, target_cols, is_baseline=False):
        """Trains on train set and evaluates on test set using the factory model."""
        X_train, y_train = df_train[feature_cols], df_train[target_cols]
        X_test, y_test = df_test[feature_cols], df_test[target_cols]
        
        # For MLP and HGB
        if len(target_cols) == 1:
            y_train = y_train.values.ravel()
            y_test = y_test.values.ravel()
        else:
            y_train = y_train.values
            y_test = y_test.values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = self._get_model(is_baseline=is_baseline)
        
        start_train = time.perf_counter()
        model.fit(X_train_s, y_train)
        t_train = time.perf_counter() - start_train
        
        start_inf = time.perf_counter()
        preds = model.predict(X_test_s)
        t_inference = (time.perf_counter() - start_inf) / len(X_test)
        
        return {
            'rmse': np.sqrt(mean_squared_error(y_test, preds)),
            'mae': mean_absolute_error(y_test, preds),
            'r2': r2_score(y_test, preds, multioutput='uniform_average'),
            't_train': t_train,
            't_inference': t_inference
        }

    def _generate_final_report(self, base, meth, method_name, n_base, n_meth):
        """Creates comparison table and calculates efficiency gains."""
        data = {
            'Metric': ['Features Count', 'RMSE', 'MAE', 'R2', 'Inf. Time (s/sample)'],
            f'Baseline ({self.model_type})': [n_base, base['rmse'], base['mae'], base['r2'], self.baseline_times['T_Inference']],
            f'Filtered ({method_name}_{self.model_type})': [n_meth, meth['rmse'], meth['mae'], meth['r2'], self.method_times['T_Inference']]
        }
        
        report_df = pd.DataFrame(data)
        
        error_reduction = ((base['rmse'] - meth['rmse']) / base['rmse']) * 100
        dim_reduction = (1 - (n_meth / n_base)) * 100
        time_inf_improvement = ((self.baseline_times['T_Inference'] - self.method_times['T_Inference']) / self.baseline_times['T_Inference']) * 100
        
        logger.info(f"===PERFORMANCE SUMMARY (Model: {self.model_type})===")
        logger.info(f"Dimensionality Reduction: {dim_reduction:.2f}%")
        logger.info(f"Error Improvement (RMSE): {error_reduction:.2f}%")
        logger.info(f"Inference Speedup:         {time_inf_improvement:.2f}%")
        
        return report_df
    
    def update_master_report(self, report_df, output_dir, params_str):
        """Maintains a master CSV tracking all experiment results including model type."""
        master_path = os.path.join(output_dir, f"MASTER_COMPARATIVE_REPORT_{self.model_type}_{params_str}.csv")
        method_column_name = report_df.columns[2]
        
        method_id = method_column_name.replace('Filtered (', '').replace(')', '')
        
        new_row = {
            'Method': method_id,
            'Model': self.model_type,
            'Num_Vars': report_df.iloc[0, 2],
            'RMSE': report_df.iloc[1, 2],
            'MAE': report_df.iloc[2, 2],
            'R2': report_df.iloc[3, 2],
            'T_Trans': self.method_times['T_Transform'],
            'T_Opt': self.method_times['T_Optimization'],
            'T_Sel': self.method_times['T_Selection'],
            'T_Train': self.method_times['T_Train'],
            'T_Inf': self.method_times['T_Inference'],
            'Best_K': self.n_neighbors_method if self.model_type == 'knn' else 'N/A'
        }
        
        if not os.path.exists(master_path):
            baseline_row = {
                'Method': f'BASELINE_{self.model_type}',
                'Model': self.model_type,
                'Num_Vars': report_df.iloc[0, 1],
                'RMSE': report_df.iloc[1, 1],
                'MAE': report_df.iloc[2, 1],
                'R2': report_df.iloc[3, 1],
                'T_Trans': self.baseline_times['T_Transform'],
                'T_Opt': self.baseline_times['T_Optimization'],
                'T_Sel': self.baseline_times['T_Selection'],
                'T_Train': self.baseline_times['T_Train'],
                'T_Inf': self.baseline_times['T_Inference'],
                'Best_K': self.n_neighbors_baseline if self.model_type == 'knn' else 'N/A'
            }
            master_df = pd.DataFrame([baseline_row, new_row])
        else:
            master_df = pd.read_csv(master_path)
            master_df = master_df[master_df['Method'] != new_row['Method']]
            master_df = pd.concat([master_df, pd.DataFrame([new_row])], ignore_index=True)
        
        master_df.to_csv(master_path, index=False)
        return master_path