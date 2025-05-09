#sklearn_transformers.py
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Clips numerical features at specified lower and upper quantiles.
    """
    def __init__(self, cols, lower_quantile=0.01, upper_quantile=0.99):
        self.cols = cols
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        # Compute bounds per column
        self.bounds_ = {
            col: (
                X[col].quantile(self.lower_quantile),
                X[col].quantile(self.upper_quantile)
            )
            for col in self.cols
        }
        return self

    def transform(self, X):
        X = X.copy()
        for col, (low, high) in self.bounds_.items():
            X[col] = X[col].clip(lower=low, upper=high)
        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Applies a log1p transform to specified numerical columns.
    """
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = np.log1p(X[col])
        return X


class NumericImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values in numerical columns using the median,
    and optionally creates a missing-value flag.
    """
    def __init__(self, cols, create_flag=True):
        self.cols = cols
        self.create_flag = create_flag

    def fit(self, X, y=None):
        self.medians_ = {col: X[col].median() for col in self.cols}
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            if self.create_flag:
                X[f"{col}_miss"] = X[col].isnull().astype(int)
            X[col] = X[col].fillna(self.medians_[col])
        return X


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Fills missing values in categorical columns with a placeholder.
    """
    def __init__(self, cols, fill_value='Missing'):
        self.cols = cols
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].fillna(self.fill_value)
        return X


class RareGrouper(BaseEstimator, TransformerMixin):
    """
    Groups rare categories (below a frequency threshold) into a single 'Other' label.
    """
    def __init__(self, cols, min_freq=0.01):
        self.cols = cols
        self.min_freq = min_freq

    def fit(self, X, y=None):
        self.frequent_categories_ = {}
        for col in self.cols:
            freq = X[col].value_counts(normalize=True)
            self.frequent_categories_[col] = set(freq[freq >= self.min_freq].index)
        return self

    def transform(self, X):
        X = X.copy()
        for col, freq_set in self.frequent_categories_.items():
            X[col] = X[col].where(X[col].isin(freq_set), other='Other')
        return X
