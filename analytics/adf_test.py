"""
Augmented Dickey-Fuller test for stationarity
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from typing import Dict, Optional


def compute_adf_test(series: pd.Series, maxlag: int = None) -> Dict:
    """
    Compute Augmented Dickey-Fuller test for stationarity
    
    Args:
        series: Time series to test
        maxlag: Maximum lag order (None for automatic selection)
    
    Returns:
        Dictionary with ADF test results
    """
    if series.empty or len(series) < 10:
        return {'error': 'Insufficient data for ADF test'}
    
    # Remove NaN values
    series_clean = series.dropna()
    
    if len(series_clean) < 10:
        return {'error': 'Insufficient data after cleaning'}
    
    try:
        result = adfuller(series_clean, maxlag=maxlag, autolag='AIC')
        
        adf_statistic = result[0]
        pvalue = result[1]
        usedlag = result[2]
        nobs = result[3]
        critical_values = result[4]
        icbest = result[5] if len(result) > 5 else None
        
        # Determine if stationary (p-value < 0.05)
        is_stationary = pvalue < 0.05
        
        return {
            'adf_statistic': float(adf_statistic),
            'pvalue': float(pvalue),
            'usedlag': int(usedlag),
            'nobs': int(nobs),
            'critical_values': {k: float(v) for k, v in critical_values.items()},
            'icbest': float(icbest) if icbest else None,
            'is_stationary': is_stationary,
            'interpretation': 'Stationary' if is_stationary else 'Non-stationary'
        }
    except Exception as e:
        return {'error': str(e)}


def compute_adf_test_spread(spread_df: pd.DataFrame) -> Dict:
    """
    Compute ADF test on spread data
    
    Args:
        spread_df: DataFrame with spread column
    
    Returns:
        Dictionary with ADF test results
    """
    if spread_df.empty or 'spread' not in spread_df.columns:
        return {'error': 'No spread data available'}
    
    spread = spread_df['spread'].dropna()
    
    if spread.empty:
        return {'error': 'Empty spread series'}
    
    return compute_adf_test(spread)

