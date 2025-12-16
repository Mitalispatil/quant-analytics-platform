"""
Helper utility functions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def resample_data(df: pd.DataFrame, timeframe: str, 
                 price_col: str = 'price',
                 quantity_col: str = 'quantity') -> pd.DataFrame:
    """
    Resample tick data to OHLC format
    
    Args:
        df: DataFrame with tick data
        timeframe: Timeframe string ('1s', '1m', '5m')
        price_col: Name of the price column
        quantity_col: Name of the quantity column
    
    Returns:
        DataFrame with OHLC data
    """
    if df.empty or 'timestamp' not in df.columns:
        return pd.DataFrame()
    
    if price_col not in df.columns:
        return pd.DataFrame()
    
    # Convert timeframe to pandas frequency
    timeframe_map = {
        '1s': '1S',
        '1m': '1T',
        '5m': '5T'
    }
    
    freq = timeframe_map.get(timeframe, '1T')
    
    # Set timestamp as index
    df_indexed = df.set_index('timestamp')
    
    # Resample
    ohlc = df_indexed[price_col].resample(freq).ohlc()
    volume = df_indexed[quantity_col].resample(freq).sum() if quantity_col in df_indexed.columns else None
    
    ohlc['volume'] = volume if volume is not None else 0
    
    # Reset index
    ohlc = ohlc.reset_index()
    
    # Remove rows with all NaN
    ohlc = ohlc.dropna(subset=['open', 'high', 'low', 'close'], how='all')
    
    return ohlc


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number for display"""
    if value is None or (isinstance(value, float) and (value != value)):  # Check for NaN
        return "N/A"
    
    return f"{value:,.{decimals}f}"


def validate_symbol(symbol: str) -> bool:
    """Validate a trading symbol format"""
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Basic validation: should be alphanumeric, typically uppercase
    return symbol.isalnum() and len(symbol) >= 3 and len(symbol) <= 20


def get_timeframe_seconds(timeframe: str) -> int:
    """Convert timeframe string to seconds"""
    timeframe_map = {
        '1s': 1,
        '1m': 60,
        '5m': 300
    }
    return timeframe_map.get(timeframe, 60)


def prepare_data_for_export(df: pd.DataFrame, 
                           include_timestamp: bool = True) -> pd.DataFrame:
    """
    Prepare DataFrame for CSV export
    
    Args:
        df: DataFrame to prepare
        include_timestamp: Whether to include timestamp column
    
    Returns:
        Prepared DataFrame
    """
    if df.empty:
        return df
    
    export_df = df.copy()
    
    # Format timestamp if present
    if 'timestamp' in export_df.columns and include_timestamp:
        export_df['timestamp'] = pd.to_datetime(export_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Round numeric columns
    numeric_cols = export_df.select_dtypes(include=[np.number]).columns
    export_df[numeric_cols] = export_df[numeric_cols].round(6)
    
    return export_df

