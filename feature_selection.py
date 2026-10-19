"""
Feature selection module for biomarker identification and dimensionality reduction.

Implements statistical testing (Mann-Whitney U), Mutual Information score ranking,
collinearity removal, and Recursive Feature Elimination (RFE).
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier

from src.config import FeatureConfig


class FeatureSelector:
    """Feature selection pipeline for EHG diagnostic classification."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.selected_feature_names_: List[str] = []

    def remove_collinear_features(
        self, X: pd.DataFrame, threshold: float = 0.90
    ) -> pd.DataFrame:
        """Remove highly correlated redundant features."""
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return X.drop(columns=to_drop)

    def select_statistical_pval(
        self, X: pd.DataFrame, y: np.ndarray, alpha: float = 0.05, method: str = 'mannwhitney'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Rank and filter features based on p-values from statistical hypothesis testing.
        
        Returns:
            Tuple of (Filtered DataFrame, Ranking DataFrame)
        """
        results = []
        for col in X.columns:
            g0 = X[col][y == 0]
            g1 = X[col][y == 1]

            if method == 'mannwhitney':
                stat, pval = mannwhitneyu(g0, g1, alternative='two-sided')
            else:
                stat, pval = ttest_ind(g0, g1, equal_var=False)

            results.append({'feature': col, 'statistic': stat, 'p_value': pval})

        rank_df = pd.DataFrame(results).sort_values('p_value')
        sig_features = rank_df[rank_df['p_value'] <= alpha]['feature'].tolist()

        if not sig_features:
            # If no p-value passes alpha, retain top K lowest p-values
            sig_features = rank_df.head(self.config.top_k_features)['feature'].tolist()

        return X[sig_features], rank_df

    def select_mutual_info(
        self, X: pd.DataFrame, y: np.ndarray, top_k: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Rank features using Mutual Information classification score."""
        top_k = top_k or self.config.top_k_features
        mi_scores = mutual_info_classif(X, y, random_state=42)

        mi_df = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores})
        mi_df = mi_df.sort_values('mi_score', ascending=False)

        selected_cols = mi_df.head(top_k)['feature'].tolist()
        return X[selected_cols], mi_df

    def select_rfe(
        self, X: pd.DataFrame, y: np.ndarray, top_k: Optional[int] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Recursive Feature Elimination using Random Forest classifier."""
        top_k = top_k or self.config.top_k_features
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)

        # Handle top_k larger than available features
        top_k = min(top_k, X.shape[1])

        rfe = RFE(estimator=estimator, n_features_to_select=top_k, step=1)
        rfe.fit(X, y)

        selected_cols = X.columns[rfe.support_].tolist()
        self.selected_feature_names_ = selected_cols
        return X[selected_cols], selected_cols

    def fit_transform(
        self, X: pd.DataFrame, y: np.ndarray, top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """Full feature selection workflow: collinearity removal -> RFE selection."""
        # 1. Remove collinear features
        X_clean = self.remove_collinear_features(X, threshold=0.90)

        # 2. Apply RFE selection
        X_selected, selected_names = self.select_rfe(X_clean, y, top_k=top_k)
        self.selected_feature_names_ = selected_names
        return X_selected
