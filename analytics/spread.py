"""
Spread computation between two assets
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def compute_spread(df1: pd.DataFrame, df2: pd.DataFrame,
                   hedge_ratio: float = 1.0,
                   price_col: str = 'price') -> pd.DataFrame:
    """
    Compute spread between two assets
    
    Args:
        df1: DataFrame for asset 1
        df2: DataFrame for asset 2
        hedge_ratio: Hedge ratio to use (default 1.0)
        price_col: Name of the price column
    
    Returns:
        DataFrame with spread data
    """
    if df1.empty or df2.empty:
        return pd.DataFrame()
    
    if price_col not in df1.columns or price_col not in df2.columns:
        return pd.DataFrame()
    
    # Align dataframes by timestamp
    merged = pd.merge(df1[['timestamp', price_col]], 
                     df2[['timestamp', price_col]], 
                     on='timestamp', 
                     suffixes=('_1', '_2'))
    
    if merged.empty:
        return pd.DataFrame()
    
    result = pd.DataFrame()
    result['timestamp'] = merged['timestamp']
    result['price_1'] = merged[f'{price_col}_1']
    result['price_2'] = merged[f'{price_col}_2']
    result['spread'] = merged[f'{price_col}_1'] - hedge_ratio * merged[f'{price_col}_2']
    
    return result


def compute_spread_stats(spread_df: pd.DataFrame) -> Dict:
    """
    Compute statistics for spread
    
    Args:
        spread_df: DataFrame with spread column
    
    Returns:
        Dictionary with spread statistics
    """
    if spread_df.empty or 'spread' not in spread_df.columns:
        return {}
    
    spread = spread_df['spread'].dropna()
    
    if spread.empty:
        return {}
    
    return {
        'mean': float(spread.mean()),
        'std': float(spread.std()),
        'min': float(spread.min()),
        'max': float(spread.max()),
        'median': float(spread.median()),
        'current': float(spread.iloc[-1]) if len(spread) > 0 else None
    }

