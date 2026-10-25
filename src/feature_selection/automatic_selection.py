#!/usr/bin/env python3
"""
Automatic feature selection methods for time series.
Focuses on efficient sequential selection (Forward/Backward).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import warnings
from ..logger_setup import setup_framework_logger

# Import base class from the correlation module
from .correlation_based import BaseFeatureSelector

logger = setup_framework_logger(__name__)
class SequentialFeatureSelector(BaseFeatureSelector):
    """Feature selector based on sequential selection (Forward/Backward)."""
    
    def __init__(self, n_features=None, direction='forward', scoring='neg_mean_squared_error', 
                 cv=5, estimator=None, verbose=False):
        """
        Initializes the sequential selection selector.
        """
        super().__init__(n_features, None, verbose)
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.estimator = estimator or LinearRegression()
        self.feature_scores_ = None
        self.selection_history_ = None
    
    def fit(self, X, y=None):
        """Fits the selector using sequential selection."""
        if y is None:
            raise ValueError("Target variable (y) is required")
        
        if self.n_features is None:
            raise ValueError("n_features is required for sequential selection")
        
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        n_total_features = len(numeric_cols)
        
        if self.direction == 'forward':
            if self.verbose:
                logger.info(f"Forward Selection: target {self.n_features} features...")
            selected_features = []
            remaining_features = numeric_cols.copy()
        elif self.direction == 'backward':
            if self.verbose:
                logger.info(f"Backward Selection: target {self.n_features} features...")
            selected_features = numeric_cols.copy()
            remaining_features = []
        else:
            raise ValueError(f"Unknown direction: {self.direction}")
        
        selection_history = []
        feature_scores = {feature: 0.0 for feature in numeric_cols}
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if self.direction == 'forward':
                pbar = tqdm(range(self.n_features), disable=not self.verbose)
                for _ in pbar:
                    if not remaining_features: break
                    best_score, best_feature = float('-inf'), None
                    
                    for feature in remaining_features:
                        candidate_features = selected_features + [feature]
                        score = self._evaluate_feature_set(X_numeric[candidate_features], y)
                        if score > best_score:
                            best_score, best_feature = score, feature
                    
                    if best_feature:
                        selected_features.append(best_feature)
                        remaining_features.remove(best_feature)
                        feature_scores[best_feature] = best_score
                        selection_history.append((best_feature, best_score))
            
            else:  # backward
                pbar = tqdm(range(n_total_features - self.n_features), disable=not self.verbose)
                for _ in pbar:
                    if len(selected_features) <= self.n_features: break
                    best_score, worst_feature = float('-inf'), None
                    
                    for feature in selected_features:
                        candidate_features = [f for f in selected_features if f != feature]
                        score = self._evaluate_feature_set(X_numeric[candidate_features], y)
                        if score > best_score:
                            best_score, worst_feature = score, feature
                    
                    if worst_feature:
                        selected_features.remove(worst_feature)
                        remaining_features.append(worst_feature)
                        feature_scores[worst_feature] = -best_score
                        selection_history.append((worst_feature, -best_score))
        
        self.selected_features_ = selected_features
        self.feature_importances_ = self._calculate_importances(selection_history, numeric_cols)
        return self
    
    def _calculate_importances(self, history, all_cols):
        n_selected = len(self.selected_features_)
        if self.direction == 'forward':
            order_importance = {feat: (n_selected - i) / n_selected for i, (feat, _) in enumerate(history)}
        else:
            order_importance = {feat: (i + 1) / max(len(history), 1) for i, (feat, _) in enumerate(history)}
        return pd.Series({f: order_importance.get(f, 0.0) for f in all_cols})

    def _evaluate_feature_set(self, X_subset, y):
        return np.mean(cross_val_score(self.estimator, X_subset, y, cv=self.cv, scoring=self.scoring))

def create_automatic_selector(method='sequential', n_features=None, **kwargs):
    """Factory function for automatic selection."""
    if method == 'sequential':
        return SequentialFeatureSelector(n_features=n_features, **kwargs)
    raise ValueError(f"Method '{method}' is no longer supported or unknown.")