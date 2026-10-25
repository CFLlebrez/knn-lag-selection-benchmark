#!/usr/bin/env python3
"""
Hyperparameter Optimization Module for Feature Selection and Evaluation.
Uses Optuna for Bayesian optimization of K-neighbors and selector parameters.
"""

import numpy as np
import pandas as pd
import optuna
import tempfile
import time
from .logger_setup import setup_framework_logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from .feature_selection import select_features

logger = setup_framework_logger(__name__)
def tune_knn_k(df, feature_cols, target_cols, n_trials=15, cv_gap=0, seed=42):
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
        return quick_cv_eval(X, y, k=k, gap=cv_gap)

    # Set Optuna verbosity to minimize console noise
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    
    best_k = study.best_params['n_neighbors']
    logger.info(f"[TUNING] Finished. Best K found: {best_k} (RMSE: {study.best_value:.4f})")
    return best_k

def get_best_fs_params(method, transformed_csv, target_col, time_col, n_features,
                       n_trials=10, cv_gap=0, seed=42,
                       selection_target_col=None, **kwargs):
    """
    Finds optimal hyperparameters for feature selection methods before final execution.
    """
    start_opt = time.time() # Start timer
    
    # Methods that do not require external tuning (e.g., pearson, cfs, or CV methods)
    methods_to_skip = [
        'pearson', 'weka_inspired', 'lasso', 'elastic_net', 'ccf',
        'rfe', 'sequential', 'sklearn_filter'
    ]
    if method in methods_to_skip:
        return {**kwargs, 'n_features': n_features}, 0.0

    logger.info(f"[TUNING] Optimizing hyperparameters for {method} with Optuna...")
    
    df = pd.read_csv(transformed_csv)
    selection_target_col = selection_target_col or target_col
    cv = TimeSeriesSplit(n_splits=3, gap=cv_gap)

    def objective(trial):
        test_params = kwargs.copy()
        test_params['n_features'] = n_features
        test_params['random_state'] = seed
        test_params.pop('verbose', None)

        # Dynamic search space definition based on the method
        if method == 'random_forest':
            test_params['max_depth'] = trial.suggest_int('max_depth', 3, 20)
            test_params['n_estimators'] = trial.suggest_int('n_estimators', 50, 200, step=50)
        
        elif method == 'mutual_info':
            test_params['n_neighbors'] = trial.suggest_int('n_neighbors', 3, 15)

        fold_errors = []
        for fold_number, (train_idx, validation_idx) in enumerate(cv.split(df)):
            fold_train = df.iloc[train_idx]
            fold_validation = df.iloc[validation_idx]

            with tempfile.TemporaryDirectory(
                prefix=f"optuna_{method}_fold{fold_number}_"
            ) as tmp_dir:
                fold_train_csv = f"{tmp_dir}/train.csv"
                fold_train.to_csv(fold_train_csv, index=False)
                res = select_features(
                    fold_train_csv,
                    tmp_dir,
                    target_col,
                    method=method,
                    time_col=time_col,
                    selection_target_col=selection_target_col,
                    evaluation_target_cols=[selection_target_col],
                    generate_report=False,
                    generate_filtered_csv=False,
                    verbose=False,
                    **test_params,
                )

            selected_features = res['selected_features']
            if not selected_features:
                raise ValueError(f"{method} selected no features")

            fold_errors.append(
                evaluate_temporal_holdout(
                    fold_train[selected_features],
                    fold_train[selection_target_col],
                    fold_validation[selected_features],
                    fold_validation[selection_target_col],
                    k=5,
                )
            )

        return float(np.mean(fold_errors))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    # Merge original parameters with the best ones found
    final_params = {
        **kwargs,
        **study.best_params,
        'n_features': n_features,
        'random_state': seed,
    }
    elapsed_opt = time.time() - start_opt # End timer
    logger.info(f"[TUNING] Best configuration for {method}: {study.best_params}")
    return final_params, elapsed_opt

def evaluate_temporal_holdout(X_train, y_train, X_validation, y_validation, k=5):
    """Fit preprocessing and k-NN on one temporal fold and return RMSE."""
    actual_k = min(k, len(X_train))
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)

    model = KNeighborsRegressor(n_neighbors=actual_k)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_validation_scaled)
    return np.sqrt(mean_squared_error(y_validation, predictions))


def quick_cv_eval(X, y, k=5, gap=0):
    """
    Robust but fast evaluation using TimeSeriesSplit.
    """
    tscv = TimeSeriesSplit(n_splits=3, gap=gap)
    errors = []
    
    # Ensure X is 2D even with a single feature
    if len(X.shape) == 1:
        X = X.values.reshape(-1, 1)
        
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        errors.append(
            evaluate_temporal_holdout(X_train, y_train, X_test, y_test, k=k)
        )
        
    return np.mean(errors)
