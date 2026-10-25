#!/usr/bin/env python3
"""
Correlation-based feature selection methods for time series.

This module implements correlation-based feature selection methods
for multivariate time series, including:
- Pearson Correlation
- Cross-Correlation Function (CCF) analysis
- Mutual Information
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from ..logger_setup import setup_framework_logger

logger = setup_framework_logger(__name__)
class BaseFeatureSelector:
    """Base class for all feature selection methods."""
    
    def __init__(self, n_features=None, threshold=None, verbose=False):
        """
        Initializes the feature selector.
        
        Args:
            n_features (int, optional): Number of features to select.
            threshold (float, optional): Threshold for feature selection.
            verbose (bool, optional): If True, shows detailed information.
        """
        self.n_features = n_features
        self.threshold = threshold
        self.verbose = verbose
        self.feature_importances_ = None
        self.selected_features_ = None
        
    def fit(self, X, y=None):
        """
        Fits the selector to the data.
        
        Args:
            X (DataFrame): Input data.
            y (Series, optional): Target variable.
            
        Returns:
            self: The fitted selector.
        """
        raise NotImplementedError("Subclasses must implement this method")
        
    def transform(self, X):
        """
        Transforms the data using only the selected features.
        
        Args:
            X (DataFrame): Input data.
            
        Returns:
            DataFrame: Transformed data with only selected features.
        """
        if self.selected_features_ is None:
            raise ValueError("The selector must be fitted before transforming")
        return X[self.selected_features_]
        
    def fit_transform(self, X, y=None):
        """
        Fits the selector and transforms the data.
        
        Args:
            X (DataFrame): Input data.
            y (Series, optional): Target variable.
            
        Returns:
            DataFrame: Transformed data with only selected features.
        """
        self.fit(X, y)
        return self.transform(X)
        
    def get_feature_importances(self):
        """
        Returns feature importances.
        
        Returns:
            Series: Feature importances.
        """
        if self.feature_importances_ is None:
            raise ValueError("The selector must be fitted before obtaining importances")
        return self.feature_importances_
        
    def get_selected_features(self):
        """
        Returns the selected features.
        
        Returns:
            list: Names of the selected features.
        """
        if self.selected_features_ is None:
            raise ValueError("The selector must be fitted before obtaining features")
        return self.selected_features_
    
    def _select_features(self):
        """
        Selects features based on their importances.
        
        Returns:
            list: Names of the selected features.
        """
        if self.feature_importances_ is None:
            raise ValueError("Feature importances must be calculated before selection")
        
        # Sort features by importance
        sorted_features = self.feature_importances_.sort_values(ascending=False)
        
        # Select features based on n_features or threshold
        if self.n_features is not None:
            selected = sorted_features.iloc[:self.n_features].index.tolist()
        elif self.threshold is not None:
            selected = sorted_features[sorted_features >= self.threshold].index.tolist()
        else:
            # By default, select features with positive importance
            selected = sorted_features[sorted_features > 0].index.tolist()
        
        return selected
    
    def plot_feature_importances(self, top_n=None, figsize=(10, 8), save_path=None):
        """
        Visualizes feature importances.
        
        Args:
            top_n (int, optional): Number of top features to show.
            figsize (tuple, optional): Figure size.
            save_path (str, optional): Path to save the figure.
            
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        if self.feature_importances_ is None:
            raise ValueError("The selector must be fitted before visualizing importances")
        
        # Sort features by importance
        sorted_importances = self.feature_importances_.sort_values(ascending=False)
        
        # Limit to top_n if specified
        if top_n is not None:
            sorted_importances = sorted_importances.iloc[:top_n]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bar plot
        sorted_importances.plot(kind='bar', ax=ax)
        
        # Add titles and labels
        ax.set_title('Feature Importance')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=90)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path is specified
        if save_path:
            plt.savefig(save_path)
        
        return fig


class PearsonCorrelationSelector(BaseFeatureSelector):
    """Feature selector based on Pearson correlation."""
    
    def __init__(self, n_features=None, threshold=None, absolute=True, verbose=False):
        """
        Initializes the Pearson correlation-based selector.
        
        Args:
            n_features (int, optional): Number of features to select.
            threshold (float, optional): Correlation threshold for selecting features.
            absolute (bool, optional): If True, uses the absolute value of the correlation.
            verbose (bool, optional): If True, shows detailed information.
        """
        super().__init__(n_features, threshold, verbose)
        self.absolute = absolute
    
    def fit(self, X, y=None):
        """
        Fits the selector by calculating Pearson correlations with the target variable.
        
        Args:
            X (DataFrame): Input data.
            y (Series): Target variable.
            
        Returns:
            self: The fitted selector.
        """
        if y is None:
            raise ValueError("Target variable (y) is required for this selector")
        
        if self.verbose:
            logger.info("Calculating Pearson correlations...")
        
        # Ensure y is a pandas Series
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Calculate correlation of each feature with the target variable
        correlations = {}
        for col in X.columns:
            if X[col].dtype.kind in 'bifc':  # Check if column is numeric
                corr = stats.pearsonr(X[col], y)[0]
                correlations[col] = abs(corr) if self.absolute else corr
        
        # Convert to Series
        self.feature_importances_ = pd.Series(correlations)
        
        # Select features
        self.selected_features_ = self._select_features()
        
        if self.verbose:
            logger.info(f"Selected {len(self.selected_features_)} features based on Pearson correlation")
        
        return self


class CrossCorrelationSelector(BaseFeatureSelector):
    """Feature selector based on Cross-Correlation Function (CCF) analysis."""
    
    def __init__(self, n_features=None, threshold=None, max_lag=10, absolute=True, verbose=False):
        """
        Initializes the cross-correlation based selector.
        
        Args:
            n_features (int, optional): Number of features to select.
            threshold (float, optional): Correlation threshold for selecting features.
            max_lag (int, optional): Maximum lag to consider.
            absolute (bool, optional): If True, uses the absolute value of the correlation.
            verbose (bool, optional): If True, shows detailed information.
        """
        super().__init__(n_features, threshold, verbose)
        self.max_lag = max_lag
        self.absolute = absolute
        self.best_lags_ = None
    
    def fit(self, X, y=None):
        """
        Fits the selector by calculating cross-correlations with the target variable.
        
        Args:
            X (DataFrame): Input data.
            y (Series): Target variable.
            
        Returns:
            self: The fitted selector.
        """
        if y is None:
            raise ValueError("Target variable (y) is required for this selector")
        
        if self.verbose:
            logger.info(f"Calculating cross-correlations with maximum lag {self.max_lag}...")
        
        # Ensure y is a pandas Series
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Calculate cross-correlation for each feature and lag
        correlations = {}
        best_lags = {}
        
        for col in tqdm(X.columns, disable=not self.verbose):
            if X[col].dtype.kind in 'bifc':  # Check if column is numeric
                max_corr = 0
                best_lag = 0
                best_corr = 0
                
                # Test different lags
                for lag in range(self.max_lag + 1):
                    # Align series with lag
                    x_lagged = X[col].shift(lag)
                    # Remove NaNs
                    valid_idx = ~x_lagged.isna() 
                    if valid_idx.sum() > 10:  # Ensure sufficient data
                        corr = stats.pearsonr(x_lagged[valid_idx], y[valid_idx])[0]
                        corr_abs = abs(corr)
                        if corr_abs > max_corr:
                            max_corr = corr_abs
                            best_lag = lag
                            best_corr = corr
                
                # Store best correlation and lag
                correlations[col] = abs(best_corr) if self.absolute else best_corr
                best_lags[col] = best_lag
        
        # Convert to Series
        self.feature_importances_ = pd.Series(correlations)
        self.best_lags_ = pd.Series(best_lags)
        
        # Select features
        self.selected_features_ = self._select_features()
        
        if self.verbose:
            logger.info(f"Selected {len(self.selected_features_)} features based on cross-correlation")
        
        return self
    
    def get_best_lags(self):
        """
        Returns the best lags for each feature.
        
        Returns:
            Series: Best lags for each feature.
        """
        if self.best_lags_ is None:
            raise ValueError("The selector must be fitted before obtaining best lags")
        return self.best_lags_
    
    def plot_ccf_heatmap(self, X, y, top_n=10, figsize=(12, 8), save_path=None):
        """
        Visualizes a cross-correlation heatmap for the main features.
        
        Args:
            X (DataFrame): Input data.
            y (Series): Target variable.
            top_n (int, optional): Number of top features to show.
            figsize (tuple, optional): Figure size.
            save_path (str, optional): Path to save the figure.
            
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        if self.feature_importances_ is None:
            raise ValueError("The selector must be fitted before visualizing")
        
        # Select top_n features
        top_features = self.feature_importances_.sort_values(ascending=False).iloc[:top_n].index
        
        # Calculate cross-correlation matrix
        ccf_matrix = np.zeros((len(top_features), self.max_lag + 1))
        
        for i, feature in enumerate(top_features):
            for lag in range(self.max_lag + 1):
                x_lagged = X[feature].shift(lag)
                valid_idx = ~x_lagged.isna()
                if valid_idx.sum() > 10:
                    corr = stats.pearsonr(x_lagged[valid_idx], y[valid_idx])[0]
                    ccf_matrix[i, lag] = corr
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(ccf_matrix, 
                   xticklabels=range(self.max_lag + 1),
                   yticklabels=top_features,
                   cmap='coolwarm',
                   center=0,
                   ax=ax)
        
        # Add titles and labels
        ax.set_title('Cross-Correlation by Lag')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Feature')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path is specified
        if save_path:
            plt.savefig(save_path)
        
        return fig


class MutualInformationSelector(BaseFeatureSelector):
    """Feature selector based on mutual information."""
    
    def __init__(self, n_features=None, threshold=None, n_neighbors=3, verbose=False):
        """
        Initializes the mutual information-based selector.
        
        Args:
            n_features (int, optional): Number of features to select.
            threshold (float, optional): Mutual information threshold for selecting features.
            n_neighbors (int, optional): Number of neighbors for estimating mutual information.
            verbose (bool, optional): If True, shows detailed information.
        """
        super().__init__(n_features, threshold, verbose)
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y=None):
        """
        Fits the selector by calculating mutual information with the target variable.
        
        Args:
            X (DataFrame): Input data.
            y (Series): Target variable.
            
        Returns:
            self: The fitted selector.
        """
        if y is None:
            raise ValueError("Target variable (y) is required for this selector")
        
        if self.verbose:
            logger.info("Calculating mutual information...")
        
        # Filter only numeric columns
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        
        # Calculate mutual information
        mi_values = mutual_info_regression(X_numeric, y, n_neighbors=self.n_neighbors)
        
        # Convert to Series
        self.feature_importances_ = pd.Series(mi_values, index=numeric_cols)
        
        # Select features
        self.selected_features_ = self._select_features()
        
        if self.verbose:
            logger.info(f"Selected {len(self.selected_features_)} features based on mutual information")
        
        return self


def create_correlation_selector(method='pearson', n_features=None, threshold=None, max_lag=10, absolute=True, verbose=False):
    """
    Creates a correlation-based feature selector according to the specified method.
    
    Args:
        method (str): Selection method ('pearson', 'ccf', 'mutual_info').
        n_features (int, optional): Number of features to select.
        threshold (float, optional): Threshold for feature selection.
        max_lag (int, optional): Maximum lag to consider for CCF.
        absolute (bool, optional): If True, uses the absolute value of the correlation.
        verbose (bool, optional): If True, shows detailed information.
        
    Returns:
        BaseFeatureSelector: The corresponding feature selector.
    """
    if method == 'pearson':
        return PearsonCorrelationSelector(n_features, threshold, absolute, verbose)
    elif method == 'ccf':
        return CrossCorrelationSelector(n_features, threshold, max_lag, absolute, verbose)
    elif method == 'mutual_info':
        return MutualInformationSelector(n_features, threshold, verbose=verbose)
    else:
        raise ValueError(f"Unknown method: {method}. Valid options: 'pearson', 'ccf', 'mutual_info'")