#!/usr/bin/env python3
"""
Hyperparameter Optimization Module for Feature Selection and Evaluation.
Uses Optuna for Bayesian optimization of K-neighbors and selector parameters.
"""

import numpy as np
import pandas as pd
import os
import optuna
import tempfile
import shutil
import time
from .logger_setup import setup_framework_logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from .feature_selection import select_features

logger = setup_framework_logger(__name__)
def tune_knn_k(df, feature_cols, target_cols, n_trials=15):
    """
    Bayesian optimization of the K parameter for KNN using Optuna.
    """
    logger.info(f"[TUNING] Starting KNN optimization with Optuna ({n_trials} trials)...")
    
    def objective(trial):
        # Suggest K within a wide range
        k = trial.suggest_int('n_neighbors', 3, 50)
        
        # Internal fast evaluation
        X = df[feature_cols]
        y = df[target_cols]
        return quick_cv_eval(X, y, k=k)

    # Set Optuna verbosity to minimize console noise
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    best_k = study.best_params['n_neighbors']
    logger.info(f"[TUNING] Finished. Best K found: {best_k} (RMSE: {study.best_value:.4f})")
    return best_k

def get_best_fs_params(method, transformed_csv, target_col, time_col, n_features, n_trials=10, **kwargs):
    """
    Finds optimal hyperparameters for feature selection methods before final execution.
    """
    start_opt = time.time() # Start timer
    
    # Methods that do not require external tuning (e.g., pearson, cfs, or CV methods)
    methods_to_skip = ['pearson', 'weka_inspired', 'lasso', 'elastic_net', 'ccf']
    if method in methods_to_skip:
        return {**kwargs, 'n_features': n_features}, 0.0

    logger.info(f"[TUNING] Optimizing hyperparameters for {method} with Optuna...")
    
    df = pd.read_csv(transformed_csv)
    y = df[target_col]

    def objective(trial):
        test_params = kwargs.copy()
        test_params['n_features'] = n_features
        test_params.pop('verbose', None)

        # Dynamic search space definition based on the method
        if method == 'random_forest':
            test_params['max_depth'] = trial.suggest_int('max_depth', 3, 20)
            test_params['n_estimators'] = trial.suggest_int('n_estimators', 50, 200, step=50)
        
        elif method == 'mutual_info':
            test_params['n_neighbors'] = trial.suggest_int('n_neighbors', 3, 15)

        try:
            # TEMPORARY DIRECTORY
            with tempfile.TemporaryDirectory(prefix=f"optuna_{method}_") as tmp_dir:
                
                res = select_features(
                    transformed_csv, 
                    tmp_dir, # The framework will write in this temp file
                    target_col, 
                    method=method, 
                    time_col=time_col, 
                    generate_report=False, 
                    generate_filtered_csv=False,
                    verbose=False,
                    **test_params
                )
            
            selected_features = res['selected_features']
            importances = res.get('feature_importances', {})

            if len(selected_features) == 0:
                return float('inf')

            # Trim to n_features based on importance
            if len(selected_features) > n_features:
                sorted_feats = sorted(selected_features, 
                                    key=lambda x: abs(importances.get(x, 0)), 
                                    reverse=True)
                current_selected = sorted_feats[:n_features]
            else:
                current_selected = selected_features

            # Evaluate subset quality
            return quick_cv_eval(df[current_selected], y)
                
        except Exception:
            return float('inf')

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    # Merge original parameters with the best ones found
    final_params = {**kwargs, **study.best_params, 'n_features': n_features}
    elapsed_opt = time.time() - start_opt # End timer
    print(f"[TUNING] Best configuration for {method}: {study.best_params}")
    return final_params, elapsed_opt

def quick_cv_eval(X, y, k=5):
    """
    Robust but fast evaluation using TimeSeriesSplit.
    """
    tscv = TimeSeriesSplit(n_splits=3)
    errors = []
    
    # Ensure X is 2D even with a single feature
    if len(X.shape) == 1:
        X = X.values.reshape(-1, 1)
        
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        actual_k = min(k, len(X_train))

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        
        model = KNeighborsRegressor(n_neighbors=actual_k)
        model.fit(X_tr_s, y_train)
        preds = model.predict(X_te_s)
        
        errors.append(np.sqrt(mean_squared_error(y_test, preds)))
        
    return np.mean(errors)