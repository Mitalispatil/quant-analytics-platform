"""
Analytics module for quantitative analysis
"""

from .price_stats import compute_price_stats
from .regression import compute_ols_hedge_ratio, compute_robust_regression
from .spread import compute_spread, compute_spread_stats
from .zscore import compute_zscore
from .adf_test import compute_adf_test, compute_adf_test_spread
from .correlation import compute_rolling_correlation, compute_correlation_heatmap
from .advanced import kalman_filter_hedge, mean_reversion_backtest

__all__ = [
    'compute_price_stats',
    'compute_ols_hedge_ratio',
    'compute_robust_regression',
    'compute_spread',
    'compute_spread_stats',
    'compute_zscore',
    'compute_adf_test',
    'compute_adf_test_spread',
    'compute_rolling_correlation',
    'compute_correlation_heatmap',
    'kalman_filter_hedge',
    'mean_reversion_backtest'
]

