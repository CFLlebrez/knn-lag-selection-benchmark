#!/usr/bin/env python3
"""
Model-based feature selection methods for time series.

This module implements model-based feature selection methods
for multivariate time series, including:
- Random Forest Feature Importance
- Lasso (L1)
- Elastic Net
- Recursive Feature Elimination (RFE)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..logger_setup import setup_framework_logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV
from sklearn.model_selection import TimeSeriesSplit

# Import base class from the correlation module
from .correlation_based import BaseFeatureSelector

logger = setup_framework_logger(__name__)

class RandomForestSelector(BaseFeatureSelector):
    """Feature selector based on Random Forest importance."""
    
    def __init__(self, n_features=None, threshold=None, n_estimators=100, 
                 max_depth=None, random_state=42, verbose=False):
        """
        Initializes the selector based on Random Forest.
        
        Args:
            n_features (int, optional): Number of features to select.
            threshold (float, optional): Importance threshold for selecting features.
            n_estimators (int, optional): Number of trees in the forest.
            max_depth (int, optional): Maximum depth of the trees.
            random_state (int, optional): Seed for reproducibility.
            verbose (bool, optional): If True, shows detailed information.
        """
        super().__init__(n_features, threshold, verbose)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
    
    def fit(self, X, y=None):
        """
        Fits the selector by training a Random Forest model and extracting importances.
        
        Args:
            X (DataFrame): Input data.
            y (Series): Target variable.
            
        Returns:
            self: The fitted selector.
        """
        if y is None:
            raise ValueError("Target variable (y) is required for this selector")
        
        if self.verbose:
            logger.info(f"Training Random Forest with {self.n_estimators} trees...")
        
        # Filter only numeric columns
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        
        # Create and train the model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1 if self.verbose else 0
        )
        self.model.fit(X_numeric, y)
        
        # Extract importances
        self.feature_importances_ = pd.Series(
            self.model.feature_importances_,
            index=numeric_cols
        )
        
        # Select features
        self.selected_features_ = self._select_features()
        
        if self.verbose:
            logger.info(f"Selected {len(self.selected_features_)} features based on Random Forest")
        
        return self
    
    def plot_feature_importances(self, top_n=None, figsize=(10, 8), save_path=None):
        """
        Visualizes the feature importances of the Random Forest.
        
        Args:
            top_n (int, optional): Number of top features to show.
            figsize (tuple, optional): Size of the figure.
            save_path (str, optional): Path to save the figure.
            
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        return super().plot_feature_importances(top_n, figsize, save_path)


class LassoSelector(BaseFeatureSelector):
    """Feature selector based on Lasso (L1)."""
    
    def __init__(self, n_features=None, threshold=None, alpha=None,
                 max_iter=1000, random_state=42, cv=None, verbose=False):
        """
        Initializes the selector based on Lasso.
        
        Args:
            n_features (int, optional): Number of features to select.
            threshold (float, optional): Coefficient threshold for selecting features.
            alpha (float, optional): Regularization parameter.
            max_iter (int, optional): Maximum number of iterations.
            random_state (int, optional): Seed for reproducibility.
            verbose (bool, optional): If True, shows detailed information.
        """
        super().__init__(n_features, threshold, verbose)
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.cv = cv or TimeSeriesSplit(n_splits=5)
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        """
        Fits the selector by training a Lasso model and extracting coefficients.
        
        Args:
            X (DataFrame): Input data.
            y (Series): Target variable.
            
        Returns:
            self: The fitted selector.
        """
        if y is None:
            raise ValueError("Target variable (y) is required for this selector")
        
        if self.verbose:
            logger.info(f"Training Lasso with alpha={self.alpha}...")
        
        # Filter only numeric columns
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X_numeric)
        
        # Create and train the model
        if self.alpha is None:
            self.model = LassoCV(
                cv=self.cv,
                random_state=self.random_state,
                max_iter=self.max_iter,
            )
        else:
            self.model = Lasso(
                alpha=self.alpha,
                random_state=self.random_state,
                max_iter=self.max_iter,
            )
        self.model.fit(X_scaled, y)
        if self.verbose:
            fitted_alpha = (
                self.model.alpha_ if hasattr(self.model, 'alpha_') else self.model.alpha
            )
            logger.info(f"Lasso alpha: {fitted_alpha:.6f}")
        # Extract coefficients (absolute value for importance)
        self.feature_importances_ = pd.Series(
            np.abs(self.model.coef_),
            index=numeric_cols
        )
        
        # Select features
        self.selected_features_ = self._select_features()
        
        if self.verbose:
            logger.info(f"Selected {len(self.selected_features_)} features based on Lasso")
        
        return self
    
    def get_coefficients(self):
        """
        Returns the coefficients of the Lasso model.
        
        Returns:
            Series: Coefficients of the model.
        """
        if self.model is None:
            raise ValueError("The selector must be fitted before obtaining coefficients")
        return pd.Series(self.model.coef_, index=self.feature_importances_.index)


class ElasticNetSelector(BaseFeatureSelector):
    """Feature selector based on Elastic Net."""
    
    def __init__(self, n_features=None, threshold=None, alpha=None, l1_ratio=0.5,
                 max_iter=1000, random_state=42, cv=None, verbose=False):
        """
        Initializes the selector based on Elastic Net.
        
        Args:
            n_features (int, optional): Number of features to select.
            threshold (float, optional): Coefficient threshold for selecting features.
            alpha (float, optional): Regularization parameter.
            l1_ratio (float, optional): Mixing ratio between L1 and L2 (1 = Lasso, 0 = Ridge).
            max_iter (int, optional): Maximum number of iterations.
            random_state (int, optional): Seed for reproducibility.
            verbose (bool, optional): If True, shows detailed information.
        """
        super().__init__(n_features, threshold, verbose)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.random_state = random_state
        self.cv = cv or TimeSeriesSplit(n_splits=5)
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        """
        Fits the selector by training an Elastic Net model and extracting coefficients.
        
        Args:
            X (DataFrame): Input data.
            y (Series): Target variable.
            
        Returns:
            self: The fitted selector.
        """
        if y is None:
            raise ValueError("Target variable (y) is required for this selector")
        
        if self.verbose:
            logger.info(f"Training Elastic Net with alpha={self.alpha}, l1_ratio={self.l1_ratio}...")
        
        # Filter only numeric columns
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X_numeric)
        
        # Create and train the model
        if self.alpha is None:
            self.model = ElasticNetCV(
                cv=self.cv,
                l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                random_state=self.random_state,
                max_iter=self.max_iter,
            )
        else:
            self.model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                random_state=self.random_state,
                max_iter=self.max_iter,
            )
        self.model.fit(X_scaled, y)
        if self.verbose:
            fitted_alpha = (
                self.model.alpha_ if hasattr(self.model, 'alpha_') else self.model.alpha
            )
            logger.info(f"Elastic Net alpha: {fitted_alpha:.6f}")
        # Extract coefficients (absolute value for importance)
        self.feature_importances_ = pd.Series(
            np.abs(self.model.coef_),
            index=numeric_cols
        )
        
        # Select features
        self.selected_features_ = self._select_features()
        
        if self.verbose:
            logger.info(f"Selected {len(self.selected_features_)} features based on Elastic Net")
        
        return self
    
    def get_coefficients(self):
        """
        Returns the coefficients of the Elastic Net model.
        
        Returns:
            Series: Coefficients of the model.
        """
        if self.model is None:
            raise ValueError("The selector must be fitted before obtaining coefficients")
        return pd.Series(self.model.coef_, index=self.feature_importances_.index)


class RFESelector(BaseFeatureSelector):
    """Feature selector based on Recursive Feature Elimination (RFE)."""
    
    def __init__(self, n_features=None, step=1, estimator=None, verbose=False):
        """
        Initializes the selector based on RFE.
        
        Args:
            n_features (int, optional): Number of features to select.
            step (int, optional): Number of features to remove in each iteration.
            estimator (object, optional): Base estimator for RFE.
            verbose (bool, optional): If True, shows detailed information.
        """
        super().__init__(n_features, None, verbose)  # RFE does not use threshold
        self.step = step
        self.estimator = estimator or RandomForestRegressor(n_estimators=100, random_state=42)
        self.rfe = None
    
    def fit(self, X, y=None):
        """
        Fits the selector using RFE.
        
        Args:
            X (DataFrame): Input data.
            y (Series): Target variable.
            
        Returns:
            self: The fitted selector.
        """
        if y is None:
            raise ValueError("Target variable (y) is required for this selector")
        
        if self.n_features is None:
            raise ValueError("Number of features (n_features) is required for RFE")
        
        if self.verbose:
            logger.info(f"Running RFE to select {self.n_features} features...")
        
        # Filter only numeric columns
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        
        # Create and fit RFE
        self.rfe = RFE(
            estimator=self.estimator,
            n_features_to_select=self.n_features,
            step=self.step,
            verbose=1 if self.verbose else 0
        )
        self.rfe.fit(X_numeric, y)
        
        # Extract importances (1 for selected, 0 for not selected)
        self.feature_importances_ = pd.Series(
            self.rfe.ranking_,
            index=numeric_cols
        )
        # Invert ranking so higher values are more important
        self.feature_importances_ = 1 / self.feature_importances_
        
        # Select features
        self.selected_features_ = [
            feature for feature, selected in zip(numeric_cols, self.rfe.support_)
            if selected
        ]
        
        if self.verbose:
            logger.info(f"Selected {len(self.selected_features_)} features using RFE")
        
        return self


def create_model_selector(method='random_forest', n_features=None, threshold=None, **kwargs):
    """
    Creates a model-based feature selector according to the specified method.
    
    Args:
        method (str): Selection method ('random_forest', 'lasso', 'elastic_net', 'rfe').
        n_features (int, optional): Number of features to select.
        threshold (float, optional): Threshold for feature selection.
        **kwargs: Additional arguments specific to each method.
        
    Returns:
        BaseFeatureSelector: The corresponding feature selector.
    """
    verbose = kwargs.get('verbose', False)
    
    if method == 'random_forest':
        return RandomForestSelector(
            n_features=n_features,
            threshold=threshold,
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            random_state=kwargs.get('random_state', 42),
            verbose=verbose
        )
    elif method == 'lasso':
        return LassoSelector(
            n_features=n_features,
            threshold=threshold,
            alpha=kwargs.get('alpha'),
            max_iter=kwargs.get('max_iter', 1000),
            random_state=kwargs.get('random_state', 42),
            cv=kwargs.get('cv'),
            verbose=verbose
        )
    elif method == 'elastic_net':
        return ElasticNetSelector(
            n_features=n_features,
            threshold=threshold,
            alpha=kwargs.get('alpha'),
            l1_ratio=kwargs.get('l1_ratio', 0.5),
            max_iter=kwargs.get('max_iter', 1000),
            random_state=kwargs.get('random_state', 42),
            cv=kwargs.get('cv'),
            verbose=verbose
        )
    elif method == 'rfe':
        return RFESelector(
            n_features=n_features,
            step=kwargs.get('step', 1),
            estimator=kwargs.get('estimator', None),
            verbose=verbose
        )
    else:
        raise ValueError(f"Unknown method: {method}. Valid options: 'random_forest', 'lasso', 'elastic_net', 'rfe'")
