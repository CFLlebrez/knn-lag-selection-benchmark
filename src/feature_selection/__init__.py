#!/usr/bin/env python3
"""
Main module for time series feature selection.

This module integrates all feature selection methods and provides 
a unified interface for their use.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from ..logger_setup import setup_framework_logger

# Import feature selectors
from .correlation_based import create_correlation_selector
from .model_based import create_model_selector
from .automatic_selection import create_automatic_selector
from .sklearn_filter import create_sklearn_filter_selector
from .weka_inspired import create_weka_inspired_selector

logger = setup_framework_logger(__name__)

def create_feature_selector(method, n_features=None, threshold=None, **kwargs):
    """
    Creates a feature selector based on the specified method.
    
    Args:
        method (str): Selection method name.
        n_features (int, optional): Number of features to select.
        threshold (float, optional): Threshold for feature selection.
        **kwargs: Additional method-specific arguments.
        
    Returns:
        BaseFeatureSelector: The corresponding feature selector instance.
    """
    # Correlation-based methods
    if method in ['pearson', 'ccf', 'mutual_info']:
        return create_correlation_selector(method, n_features, threshold, **kwargs)
    
    # Model-based methods
    elif method in ['random_forest', 'lasso', 'elastic_net', 'rfe']:
        return create_model_selector(method, n_features, threshold, **kwargs)
    
    # Automatic selection methods
    elif method in ['sequential']:
        return create_automatic_selector(method, n_features, **kwargs)
    
    # Scikit-learn filter methods
    elif method == 'sklearn_filter':
        return create_sklearn_filter_selector(
            method=kwargs.get('sklearn_method', 'selectkbest'),
            n_features=n_features,
            percentile=kwargs.get('percentile'),
            score_func_name=kwargs.get('score_func_name', 'f_regression'),
            strategy=kwargs.get('strategy', 'k_best'),
            param=kwargs.get('param'),
            threshold=kwargs.get('sklearn_threshold', 0.0),
            verbose=kwargs.get('verbose', False)
        )
    
    # Weka-inspired methods
    elif method == 'weka_inspired':
        return create_weka_inspired_selector(
            method=kwargs.get('weka_inspired_method', 'cfs'),
            n_features=n_features,
            threshold=kwargs.get('weka_threshold', 0.0),
            # CFS
            max_backtrack=kwargs.get('max_backtrack', 5)
        )
    
    else:
        raise ValueError(f"Unknown method: {method}")


def prepare_data_for_selection(df, target_col, include_target=False):
    """
    Prepares data for feature selection by removing the target 
    and its future horizons from the predictors (X).

    Args:
        df (DataFrame): Transformed DataFrame (with lags and horizons).
        target_col (str): Name of the target column.
        include_target (bool): Whether to include the target in X.

    Returns:
        tuple: (X, y) where X are predictors and y is the target variable.
    """
    data = df.copy()

    # Identify columns to exclude
    cols_to_exclude = []
    if not include_target:
        cols_to_exclude.append(target_col)

    # Exclude future horizons (target_col_t+N)
    future_cols = [c for c in data.columns if c.startswith(f"{target_col}_t+")]
    cols_to_exclude.extend(future_cols)
    
    # Build X and y
    X = data.drop(columns=cols_to_exclude)
    y = data[target_col].copy()

    return X, y


def generate_feature_importance_report(importances, selected_list, output_dir, prefix):
    """
    Generates a visual report and CSV of feature importances.
    """
    report_df = pd.DataFrame({
        'Feature': importances.index,
        'Importance': importances.values,
        'Selected': [f in selected_list for f in importances.index]
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    
    # Dynamic palette definition
    palette = {True: 'darkblue', False: 'lightgray'}
    
    sns.barplot(x='Importance', y='Feature', data=report_df.head(30), 
                hue='Selected', palette=palette, legend=False)
    
    plt.title(f'Feature Importance - {prefix}')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"{prefix}_plot.png")
    plt.savefig(plot_path)
    plt.close()
    
    report_path = os.path.join(output_dir, f"{prefix}_report.csv")
    report_df.to_csv(report_path, index=False)
    return report_path, plot_path


def save_selection_json(output_path, method_name, selected_features, target_cols, elapsed_time, input_file):
    """
    Generates the metadata JSON file for the evaluator.
    """
    metadata = {
        "experiment_info": {
            "method": method_name,
            "input_file": os.path.basename(input_file),
            "date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        "selected_features": list(selected_features),
        "target_columns": list(target_cols),
        "selection_time_seconds": round(elapsed_time, 4)
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)


def select_features(input_file, output_dir, target_col, method='random_forest', 
                   n_features=None, threshold=None, time_col=None, include_target=False,
                   generate_report=True, generate_filtered_csv=True, **kwargs):
    """
    Main function for feature selection with strict n_features control.
    """
    verbose = kwargs.get('verbose', True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(input_file)
    dates = None
    if time_col and time_col in df.columns:
        dates = df[time_col].copy()
        df = df.drop(columns=[time_col])

    # Prepare X and y
    X, y = prepare_data_for_selection(df, target_col, include_target=include_target)
    
    # Create selector
    selector = create_feature_selector(method, n_features, threshold, verbose=verbose, **kwargs)
    
    if verbose:
        logger.info(f"Creating feature selector using method '{method}'...")
        logger.info("Fitting feature selector...")
    
    # Measure execution time
    start_time = time.time()
    selector.fit(X, y)
    elapsed = time.time() - start_time
    
    # 1. Get initial features and importances
    selected_features = list(selector.get_selected_features())
    importances = selector.get_feature_importances()
    
    # 2. STRICT CONTROL OF FEATURE COUNT (Post-processing)
    if n_features is not None and len(selected_features) > n_features:
        if verbose:
            logger.info(f"Selector returned {len(selected_features)} variables. "
                  f"Trimming to the top {n_features} features based on importance.")
        
        # Absolute importance dictionary for selected features
        feat_imp = {f: abs(importances.get(f, 0)) for f in selected_features}
        
        # Sort by importance and take exactly n_features
        selected_features = sorted(feat_imp, key=feat_imp.get, reverse=True)[:n_features]

    if verbose:
        logger.info(f"Final result: {len(selected_features)} features selected.")

    # Define method label for filenames
    if method == 'sklearn_filter':
        method_label = f"sklearn_filter_{kwargs.get('sklearn_method')}"
    elif method == 'weka_inspired':
        method_label = f"weka_inspired_{kwargs.get('weka_inspired_method')}"
    else:
        method_label = method

    # 3. Generate visual report and importance CSV
    report_path, plot_path = None, None
    if generate_report:
        report_path, plot_path = generate_feature_importance_report(
            importances, selected_features, output_dir, prefix=f"{method_label}_feature_importance"
        )
    
    # 4. Generate filtered CSV (Optional)
    filtered_csv_path = None
    if generate_filtered_csv:
        if verbose:
            logger.info("Generating filtered CSV with selected features + target...")
        filtered_csv_path = os.path.join(output_dir, f"filtered_data_{method}.csv")
        
        X_selected = X[selected_features]
        df_filtered = pd.concat([X_selected, y], axis=1)

        if dates is not None:
            df_filtered.insert(0, time_col, dates)

        # Optional: Uncomment to save the actual CSV file
        # df_filtered.to_csv(filtered_csv_path, index=False)

    # 5. Identify target columns (future steps) for the JSON
    future_targets = [c for c in df.columns if c.startswith(f"{target_col}_t+")]
    all_targets = [target_col] + future_targets 

    # 6. Save metadata JSON (critical for PredictiveEvaluator)
    json_path = os.path.join(output_dir, f"selection_metadata_{method}.json")
    save_selection_json(json_path, method_label, selected_features, all_targets, elapsed, input_file)
    
    return {
        'selected_features': selected_features,
        'target_columns': all_targets,
        'json_metadata_path': json_path,
        'n_selected': len(selected_features),
        'feature_importances': importances,
        'report_path': report_path,
        'plot_path': plot_path,
        'filtered_csv_path': filtered_csv_path
    }