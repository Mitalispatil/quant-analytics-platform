"""
Correlation analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import timedelta
import config


def compute_rolling_correlation(df1: pd.DataFrame, df2: pd.DataFrame,
                               window: int = 60,
                               price_col: str = 'price') -> pd.DataFrame:
    """
    Compute rolling correlation between two assets
    
    Args:
        df1: DataFrame for asset 1
        df2: DataFrame for asset 2
        window: Rolling window size (in seconds or number of observations)
        price_col: Name of the price column
    
    Returns:
        DataFrame with rolling correlation
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
    
    # Sort by timestamp
    merged = merged.sort_values('timestamp')
    
    # Compute rolling correlation
    # If window is small (< 100), treat as number of observations
    # Otherwise, treat as seconds and use time-based window
    if window < 100:
        # Number of observations
        merged['correlation'] = merged[f'{price_col}_1'].rolling(
            window=min(window, len(merged))
        ).corr(merged[f'{price_col}_2'])
    else:
        # Time-based window (in seconds)
        merged = merged.set_index('timestamp')
        merged['correlation'] = merged[f'{price_col}_1'].rolling(
            window=f'{window}s'
        ).corr(merged[f'{price_col}_2'])
        merged = merged.reset_index()
    
    result = pd.DataFrame()
    result['timestamp'] = merged['timestamp']
    result['correlation'] = merged['correlation']
    
    return result


def compute_correlation_heatmap(df_dict: Dict[str, pd.DataFrame],
                                price_col: str = 'price') -> pd.DataFrame:
    """
    Compute correlation matrix for multiple assets
    
    Args:
        df_dict: Dictionary mapping symbol names to DataFrames
        price_col: Name of the price column
    
    Returns:
        DataFrame with correlation matrix
    """
    if not df_dict:
        return pd.DataFrame()
    
    # Prepare data: align all series by timestamp
    all_data = {}
    
    for symbol, df in df_dict.items():
        if df.empty or price_col not in df.columns:
            continue
        
        # Resample to common frequency (1 second)
        df_resampled = df.set_index('timestamp')[price_col].resample('1S').last()
        all_data[symbol] = df_resampled
    
    if len(all_data) < 2:
        return pd.DataFrame()
    
    # Combine into single DataFrame
    combined = pd.DataFrame(all_data)
    
    # Compute correlation matrix
    corr_matrix = combined.corr()
    
    return corr_matrix


def get_current_correlation(df1: pd.DataFrame, df2: pd.DataFrame,
                            window: int = 60,
                            price_col: str = 'price') -> float:
    """
    Get the current rolling correlation value
    
    Args:
        df1: DataFrame for asset 1
        df2: DataFrame for asset 2
        window: Rolling window size
        price_col: Name of the price column
    
    Returns:
        Current correlation value or NaN
    """
    corr_df = compute_rolling_correlation(df1, df2, window, price_col)
    
    if corr_df.empty or 'correlation' not in corr_df.columns:
        return np.nan
    
    corr = corr_df['correlation'].dropna()
    
    if corr.empty:
        return np.nan
    
    return float(corr.iloc[-1])

