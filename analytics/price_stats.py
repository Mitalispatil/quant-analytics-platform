"""
Price statistics computation
"""

import pandas as pd
import numpy as np
from typing import Dict


def compute_price_stats(df: pd.DataFrame, price_col: str = 'price', 
                       window: int = None) -> Dict:
    """
    Compute price statistics
    
    Args:
        df: DataFrame with price data
        price_col: Name of the price column
        window: Rolling window size (None for overall stats)
    
    Returns:
        Dictionary with statistics
    """
    if df.empty or price_col not in df.columns:
        return {}
    
    prices = df[price_col].dropna()
    
    if prices.empty:
        return {}
    
    if window and len(prices) > window:
        prices = prices.rolling(window=window).mean()
        prices = prices.dropna()
    
    if prices.empty:
        return {}
    
    stats = {
        'mean': float(prices.mean()),
        'std': float(prices.std()),
        'min': float(prices.min()),
        'max': float(prices.max()),
        'median': float(prices.median()),
        'q25': float(prices.quantile(0.25)),
        'q75': float(prices.quantile(0.75)),
        'count': int(len(prices))
    }
    
    return stats


def compute_rolling_stats(df: pd.DataFrame, price_col: str = 'price',
                         window: int = 100) -> pd.DataFrame:
    """
    Compute rolling statistics
    
    Args:
        df: DataFrame with price data
        price_col: Name of the price column
        window: Rolling window size
    
    Returns:
        DataFrame with rolling statistics
    """
    if df.empty or price_col not in df.columns:
        return pd.DataFrame()
    
    prices = df[price_col]
    
    result = pd.DataFrame(index=df.index)
    result['mean'] = prices.rolling(window=window).mean()
    result['std'] = prices.rolling(window=window).std()
    result['min'] = prices.rolling(window=window).min()
    result['max'] = prices.rolling(window=window).max()
    
    return result

