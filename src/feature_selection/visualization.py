#!/usr/bin/env python3
"""
Visualization module for time series feature selection methods.

This module provides functions to visualize feature selection results,
including importance plots, correlation matrices, and more.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def plot_feature_importance(importances, selected_features=None, top_n=30, 
                           figsize=(12, 8), save_path=None, title='Feature Importance'):
    """
    Visualizes feature importance rankings.
    
    Args:
        importances (Series): Series with feature importance values.
        selected_features (list, optional): List of selected features to highlight.
        top_n (int, optional): Number of top features to display.
        figsize (tuple, optional): Figure size.
        save_path (str, optional): Path to save the generated figure.
        title (str, optional): Plot title.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    # Sort importances
    sorted_importances = importances.sort_values(ascending=False)
    
    # Limit to top_n
    if top_n is not None and len(sorted_importances) > top_n:
        sorted_importances = sorted_importances.iloc[:top_n]
    
    # Create DataFrame for visualization
    df_plot = pd.DataFrame({
        'Feature': sorted_importances.index,
        'Importance': sorted_importances.values
    })
    
    # Add selection column if selected features are provided
    if selected_features is not None:
        df_plot['Selected'] = df_plot['Feature'].isin(selected_features)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    if selected_features is not None:
        ax = sns.barplot(x='Importance', y='Feature', data=df_plot, 
                        hue='Selected', palette=['lightgray', 'darkblue'])
        ax.legend(title='Selected')
    else:
        ax = sns.barplot(x='Importance', y='Feature', data=df_plot)
    
    # Add titles and labels
    ax.set_title(title)
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Features')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is specified
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_correlation_matrix(X, selected_features=None, figsize=(12, 10), 
                           save_path=None, title='Correlation Matrix'):
    """
    Visualizes the correlation matrix between features.
    
    Args:
        X (DataFrame): Input data.
        selected_features (list, optional): List of selected features to highlight.
        figsize (tuple, optional): Figure size.
        save_path (str, optional): Path to save the figure.
        title (str, optional): Plot title.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    # Calculate correlation matrix
    corr_matrix = X.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
               square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    
    # Highlight selected features if provided
    if selected_features is not None:
        # Get indices of selected features
        selected_indices = [i for i, feature in enumerate(corr_matrix.columns) 
                           if feature in selected_features]
        
        # Highlight selected features with rectangles
        for idx in selected_indices:
            ax.add_patch(plt.Rectangle((idx, 0), 1, corr_matrix.shape[0], 
                                      fill=False, edgecolor='red', lw=2))
            ax.add_patch(plt.Rectangle((0, idx), corr_matrix.shape[1], 1, 
                                      fill=False, edgecolor='red', lw=2))
    
    # Add title
    ax.set_title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is specified
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_feature_distribution(X, selected_features=None, max_features=10, 
                             figsize=(15, 10), save_path=None):
    """
    Visualizes the distribution of selected features.
    
    Args:
        X (DataFrame): Input data.
        selected_features (list, optional): List of features to visualize.
        max_features (int, optional): Maximum number of features to display.
        figsize (tuple, optional): Figure size.
        save_path (str, optional): Path to save the figure.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    # Select features to visualize
    if selected_features is None:
        features_to_plot = X.columns[:max_features].tolist()
    else:
        features_to_plot = selected_features[:max_features]
    
    # Determine rows and columns for subplots
    n_features = len(features_to_plot)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array if necessary
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    
    # Plot distribution for each feature
    for i, feature in enumerate(features_to_plot):
        if i < len(axes):
            row = i // n_cols
            col = i % n_cols
            
            if n_rows == 1 and n_cols == 1:
                ax = axes[0]
            elif n_rows == 1 or n_cols == 1:
                ax = axes[i]
            else:
                ax = axes[row, col]
            
            # Plot histogram and KDE
            sns.histplot(X[feature].dropna(), kde=True, ax=ax)
            
            # Add title
            ax.set_title(feature)
            
            # Adjust labels
            ax.set_xlabel('')
            if col == 0:
                ax.set_ylabel('Frequency')
            else:
                ax.set_ylabel('')
    
    # Hide unused axes
    for i in range(n_features, len(axes.flatten())):
        fig.delaxes(axes.flatten()[i])
    
    # Add main title
    fig.suptitle('Distribution of Selected Features', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure if path is specified
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_lag_importance(importances, max_lag=10, figsize=(12, 8), save_path=None):
    """
    Visualizes importance of different lags for each variable.
    
    Args:
        importances (Series): Series with feature importance values.
        max_lag (int, optional): Maximum lag to consider.
        figsize (tuple, optional): Figure size.
        save_path (str, optional): Path to save the figure.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    # Extract base variables and their lags
    lag_pattern = '_lag'
    base_vars = set()
    for feature in importances.index:
        if lag_pattern in feature:
            base_var = feature.split(lag_pattern)[0]
            base_vars.add(base_var)
    
    # Store importance by variable and lag
    lag_importances = {}
    
    for base_var in base_vars:
        var_importances = []
        
        for lag in range(1, max_lag + 1):
            lag_feature = f"{base_var}{lag_pattern}{lag}"
            
            if lag_feature in importances:
                var_importances.append((lag, importances[lag_feature]))
            else:
                var_importances.append((lag, 0))
        
        if var_importances:
            lag_importances[base_var] = var_importances
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot importance per lag for each variable
    for base_var, values in lag_importances.items():
        lags, imps = zip(*values)
        ax.plot(lags, imps, 'o-', label=base_var)
    
    # Add titles and labels
    ax.set_title('Importance per Lag for each Variable')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Importance Score')
    ax.set_xticks(range(1, max_lag + 1))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is specified
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_parallel_coordinates(X, selected_features=None, n_samples=100, figsize=(15, 8), save_path=None):
    """
    Visualizes parallel coordinates for selected features.
    
    Args:
        X (DataFrame): Input data.
        selected_features (list, optional): List of features to visualize.
        n_samples (int, optional): Number of samples to display.
        figsize (tuple, optional): Figure size.
        save_path (str, optional): Path to save the figure.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    # Select features to visualize
    if selected_features is None:
        features_to_plot = X.columns.tolist()
    else:
        features_to_plot = selected_features
    
    # Subsample data if it contains too many rows
    if len(X) > n_samples:
        X_sample = X.sample(n_samples, random_state=42)
    else:
        X_sample = X
    
    # Filter features
    X_plot = X_sample[features_to_plot].copy()
    
    # Normalize data for better visualization
    scaler = MinMaxScaler()
    X_normalized = pd.DataFrame(
        scaler.fit_transform(X_plot),
        columns=X_plot.columns,
        index=X_plot.index
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot parallel coordinates
    pd.plotting.parallel_coordinates(
        X_normalized.reset_index(), 'index', 
        colormap=plt.cm.tab20, ax=ax
    )
    
    # Add titles and labels
    ax.set_title('Parallel Coordinates of Selected Features')
    ax.set_xlabel('')
    ax.set_ylabel('Normalized Value')
    
    # Remove legend if there are many samples
    if len(X_normalized) > 20:
        ax.get_legend().remove()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is specified
    if save_path:
        plt.savefig(save_path)
    
    return fig


def generate_comprehensive_report(X, importances, selected_features, output_dir, prefix='feature_selection'):
    """
    Generates a comprehensive report with multiple visualizations.
    
    Args:
        X (DataFrame): Input data.
        importances (Series): Series with feature importance values.
        selected_features (list): List of selected features.
        output_dir (str): Output directory for saving results.
        prefix (str, optional): Prefix for filenames.
        
    Returns:
        dict: Dictionary containing paths to the generated files.
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate CSV report
    report_df = pd.DataFrame({
        'Feature': importances.index,
        'Importance': importances.values,
        'Selected': [feature in selected_features for feature in importances.index]
    }).sort_values('Importance', ascending=False)
    
    report_path = os.path.join(output_dir, f"{prefix}_report.csv")
    report_df.to_csv(report_path, index=False)
    
    # Generate visualizations
    plots = {}
    
    # 1. Feature Importance Plot
    importance_plot_path = os.path.join(output_dir, f"{prefix}_importance.png")
    plot_feature_importance(
        importances, selected_features, 
        save_path=importance_plot_path
    )
    plots['importance'] = importance_plot_path
    
    # 2. Correlation Matrix
    correlation_plot_path = os.path.join(output_dir, f"{prefix}_correlation.png")
    plot_correlation_matrix(
        X, selected_features, 
        save_path=correlation_plot_path
    )
    plots['correlation'] = correlation_plot_path
    
    # 3. Selected Features Distribution
    if selected_features:
        distribution_plot_path = os.path.join(output_dir, f"{prefix}_distribution.png")
        plot_feature_distribution(
            X, selected_features, 
            save_path=distribution_plot_path
        )
        plots['distribution'] = distribution_plot_path
    
    # 4. Importance per Lag
    lag_plot_path = os.path.join(output_dir, f"{prefix}_lag_importance.png")
    try:
        plot_lag_importance(
            importances, 
            save_path=lag_plot_path
        )
        plots['lag_importance'] = lag_plot_path
    except Exception:
        # Fails if no lag pattern is detected in features
        pass
    
    # 5. Parallel Coordinates
    if len(selected_features) <= 20:  # Only if there are not too many features
        parallel_plot_path = os.path.join(output_dir, f"{prefix}_parallel.png")
        plot_parallel_coordinates(
            X, selected_features, 
            save_path=parallel_plot_path
        )
        plots['parallel'] = parallel_plot_path
    
    return {
        'report': report_path,
        'plots': plots
    }