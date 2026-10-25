#!/usr/bin/env python3
"""
Scikit-learn filter-based feature selection methods for time series.

This module implements feature selection methods based on scikit-learn filters,
including:
- SelectKBest with different scoring functions
- SelectPercentile
- GenericUnivariateSelect
- VarianceThreshold
"""

import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, GenericUnivariateSelect,
    VarianceThreshold, f_regression, mutual_info_regression
)

# Import base class from the correlation module
from .correlation_based import BaseFeatureSelector

from ..logger_setup import setup_framework_logger


logger = setup_framework_logger(__name__)
class SKLearnFilterSelector(BaseFeatureSelector):
    """Feature selector based on scikit-learn filter methods."""
    
    def __init__(self, method='selectkbest', n_features=None, percentile=None, 
                 score_func=f_regression, strategy='k_best', param=None, 
                 threshold=0.0, verbose=False):
        """
        Initializes the scikit-learn filter-based selector.
        
        Args:
            method (str): Filter method ('selectkbest', 'selectpercentile', 
                         'genericunivariateselect', 'variancethreshold').
            n_features (int, optional): Number of features to select (for SelectKBest).
            percentile (int, optional): Percentile of features to select (for SelectPercentile).
            score_func (callable, optional): Scoring function for univariate methods.
            strategy (str, optional): Strategy for GenericUnivariateSelect.
            param (float, optional): Parameter for the GenericUnivariateSelect strategy.
            threshold (float, optional): Threshold for VarianceThreshold.
            verbose (bool, optional): If True, shows detailed information.
        """
        super().__init__(n_features, None, verbose)
        self.method = method
        self.percentile = percentile
        self.score_func = score_func
        self.strategy = strategy
        self.param = param
        self.threshold = threshold
        self.selector = None
        
    def fit(self, X, y=None):
        """
        Fits the selector to the data.
        
        Args:
            X (DataFrame): Input data.
            y (Series, optional): Target variable.
            
        Returns:
            self: The fitted selector.
            
        Raises:
            ValueError: If y is missing for supervised methods or if required parameters are None.
        """
        if self.method in ['selectkbest', 'selectpercentile', 'genericunivariateselect'] and y is None:
            raise ValueError(f"Target variable (y) is required for method {self.method}")
        
        # Filter only numeric columns
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        
        if self.verbose:
            logger.info(f"Applying scikit-learn filter method: {self.method}...")
        
        # Create selector based on the method
        if self.method == 'selectkbest':
            if self.n_features is None:
                raise ValueError("Number of features (n_features) is required for SelectKBest")
            self.selector = SelectKBest(score_func=self.score_func, k=self.n_features)
            
        elif self.method == 'selectpercentile':
            if self.percentile is None:
                raise ValueError("Percentile is required for SelectPercentile")
            self.selector = SelectPercentile(score_func=self.score_func, percentile=self.percentile)
            
        elif self.method == 'genericunivariateselect':
            self.selector = GenericUnivariateSelect(
                score_func=self.score_func, 
                mode=self.strategy,
                param=self.param if self.param is not None else (
                    self.n_features if self.strategy == 'k_best' else 
                    self.percentile if self.strategy == 'percentile' else None
                )
            )
            
        elif self.method == 'variancethreshold':
            self.selector = VarianceThreshold(threshold=self.threshold)
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Fit selector
        if self.method == 'variancethreshold':
            self.selector.fit(X_numeric)
        else:
            self.selector.fit(X_numeric, y)
        
        # Get selected features
        support = self.selector.get_support()
        self.selected_features_ = [feature for feature, selected in zip(numeric_cols, support) if selected]
        
        # Get scores (if available)
        if hasattr(self.selector, 'scores_'):
            self.feature_importances_ = pd.Series(
                self.selector.scores_,
                index=numeric_cols
            )
        else:
            # For VarianceThreshold, use variances as importances
            if self.method == 'variancethreshold':
                variances = X_numeric.var()
                self.feature_importances_ = variances
            else:
                # Create binary importances (1 for selected, 0 for unselected)
                self.feature_importances_ = pd.Series(
                    [1.0 if s else 0.0 for s in support],
                    index=numeric_cols
                )
        
        if self.verbose:
            logger.info(f"Selected {len(self.selected_features_)} features using {self.method}")
        
        return self
    
    def transform(self, X):
        """
        Transforms the data using only the selected features.
        
        Args:
            X (DataFrame): Input data.
            
        Returns:
            DataFrame: Transformed data with only selected features.
            
        Raises:
            ValueError: If the selector has not been fitted yet.
        """
        if self.selector is None:
            raise ValueError("The selector must be fitted before transforming")
        
        # Filter only numeric columns
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        
        # Apply transformation
        X_transformed = self.selector.transform(X_numeric)
        
        # Convert to DataFrame
        X_transformed_df = pd.DataFrame(
            X_transformed,
            columns=self.selected_features_,
            index=X.index
        )
        
        return X_transformed_df


def create_sklearn_filter_selector(method='selectkbest', n_features=None, percentile=None, 
                                  score_func_name='f_regression', strategy='k_best', 
                                  param=None, threshold=0.0, verbose=False):
    """
    Creates a feature selector based on scikit-learn filters.
    
    Args:
        method (str): Filter method ('selectkbest', 'selectpercentile', 
                     'genericunivariateselect', 'variancethreshold').
        n_features (int, optional): Number of features to select.
        percentile (int, optional): Percentile of features to select.
        score_func_name (str, optional): Name of the scoring function.
        strategy (str, optional): Strategy for GenericUnivariateSelect.
        param (float, optional): Parameter for the GenericUnivariateSelect strategy.
        threshold (float, optional): Threshold for VarianceThreshold.
        verbose (bool, optional): If True, shows detailed information.
        
    Returns:
        SKLearnFilterSelector: The corresponding feature selector instance.
        
    Raises:
        ValueError: If the score function name is unknown.
    """
    # Determine scoring function
    if score_func_name == 'f_regression':
        score_func = f_regression
    elif score_func_name == 'mutual_info_regression':
        score_func = mutual_info_regression
    else:
        raise ValueError(f"Unknown scoring function: {score_func_name}")
    
    return SKLearnFilterSelector(
        method=method,
        n_features=n_features,
        percentile=percentile,
        score_func=score_func,
        strategy=strategy,
        param=param,
        threshold=threshold,
        verbose=verbose
    )