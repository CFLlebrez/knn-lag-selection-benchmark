#!/usr/bin/env python3
"""
Weka-inspired feature selection methods for time series.

This module implements feature selection methods popularized by Weka,
specifically adapted for time series data, including:
- CFS (Correlation-based Feature Selection)
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Import base class from correlation module
from .correlation_based import BaseFeatureSelector


class CFSSelector(BaseFeatureSelector):
    """
    Correlation-based Feature Selection (CFS) based on Hall (1999).
    Optimized for time series using Pearson correlation to evaluate
    feature-class relevance and feature-feature redundancy.
    """
    def __init__(self, n_features=None, threshold=None, max_backtrack=5, verbose=False):
        super().__init__(n_features, threshold, verbose)
        self.max_backtrack = max_backtrack

    def _calculate_merit(self, subset, corr_matrix, target_corr):
        """Hall's Merit Equation: Merit = (k * r_cf) / sqrt(k + k*(k-1)*r_ff)"""
        k = len(subset)
        if k == 0: return 0
        
        # Average feature-class correlation (relevance)
        avg_rcf = np.mean([abs(target_corr[f]) for f in subset])
        
        if k == 1: return avg_rcf
        
        # Average feature-feature correlation (redundancy)
        ff_corrs = []
        for i in range(k):
            for j in range(i + 1, k):
                ff_corrs.append(abs(corr_matrix.loc[subset[i], subset[j]]))
        
        avg_rff = np.mean(ff_corrs) if ff_corrs else 0
        
        # Merit formula application
        numerator = k * avg_rcf
        denominator = np.sqrt(k + k * (k - 1) * avg_rff)
        return numerator / denominator

    def fit(self, X, y):
        """Fits CFS by maximizing the merit of the feature subset."""
        if isinstance(y, pd.DataFrame): y = y.iloc[:, 0]
        
        # 1. Pre-calculate correlation matrix (including all lags)
        all_features = X.columns.tolist()
        data_combined = X.assign(target_internal=y.values)
        full_corr = data_combined.corr(method='pearson').fillna(0)
        
        target_corr = full_corr['target_internal'].drop('target_internal')
        feature_matrix_corr = full_corr.drop('target_internal', axis=0).drop('target_internal', axis=1)

        # 2. Greedy Forward Search
        selected = []
        remaining = all_features.copy()
        limit = self.n_features if self.n_features else len(all_features)

        pbar = tqdm(total=limit, desc="CFS Selection", disable=not self.verbose)
        while len(selected) < limit and remaining:
            best_merit = -np.inf
            best_feat = None
            
            for feat in remaining:
                merit = self._calculate_merit(selected + [feat], feature_matrix_corr, target_corr)
                if merit > best_merit:
                    best_merit = merit
                    best_feat = feat
            
            if best_feat:
                selected.append(best_feat)
                remaining.remove(best_feat)
                pbar.update(1)
            else:
                break
        pbar.close()

        self.selected_features_ = selected
        # Use individual correlation as importance for visualization purposes
        self.feature_importances_ = pd.Series(target_corr.abs(), index=target_corr.index)
        return self


def create_weka_inspired_selector(method='cfs', n_features=None, threshold=None, **kwargs):
    """Factory function for Weka-inspired selectors."""
    verbose = kwargs.get('verbose', False)
    if method == 'cfs':
        return CFSSelector(n_features, threshold, kwargs.get('max_backtrack', 5), verbose)
    else:
        raise ValueError(f"Unknown method: {method}")