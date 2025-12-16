"""
Z-score computation for mean reversion strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def compute_zscore(spread_df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """
    Compute z-score of spread
    
    Args:
        spread_df: DataFrame with spread column
        window: Rolling window for mean and std calculation
    
    Returns:
        DataFrame with z-score
    """
    if spread_df.empty or 'spread' not in spread_df.columns:
        return pd.DataFrame()
    
    spread = spread_df['spread'].dropna()
    
    if spread.empty:
        return pd.DataFrame()
    
    result = spread_df.copy()
    
    # Compute rolling mean and std
    rolling_mean = spread.rolling(window=min(window, len(spread))).mean()
    rolling_std = spread.rolling(window=min(window, len(spread))).std()
    
    # Compute z-score
    result['zscore'] = (spread - rolling_mean) / rolling_std
    
    return result


def get_current_zscore(spread_df: pd.DataFrame, window: int = 100) -> Optional[float]:
    """
    Get the current z-score value
    
    Args:
        spread_df: DataFrame with spread column
        window: Rolling window for mean and std calculation
    
    Returns:
        Current z-score value or None
    """
    zscore_df = compute_zscore(spread_df, window)
    
    if zscore_df.empty or 'zscore' not in zscore_df.columns:
        return None
    
    zscore = zscore_df['zscore'].dropna()
    
    if zscore.empty:
        return None
    
    return float(zscore.iloc[-1])

