"""
Advanced analytics: Kalman Filter, Backtesting, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import linalg


def kalman_filter_hedge(df1: pd.DataFrame, df2: pd.DataFrame,
                        price_col: str = 'price',
                        initial_hedge: float = 1.0,
                        process_noise: float = 0.01,
                        measurement_noise: float = 0.1) -> pd.DataFrame:
    """
    Estimate dynamic hedge ratio using Kalman Filter
    
    Args:
        df1: DataFrame for asset 1 (dependent)
        df2: DataFrame for asset 2 (independent)
        price_col: Name of the price column
        initial_hedge: Initial hedge ratio estimate
        process_noise: Process noise variance
        measurement_noise: Measurement noise variance
    
    Returns:
        DataFrame with dynamic hedge ratios
    """
    if df1.empty or df2.empty:
        return pd.DataFrame()
    
    if price_col not in df1.columns or price_col not in df2.columns:
        return pd.DataFrame()
    
    # Align dataframes
    merged = pd.merge(df1[['timestamp', price_col]], 
                     df2[['timestamp', price_col]], 
                     on='timestamp', 
                     suffixes=('_1', '_2'))
    
    if merged.empty:
        return pd.DataFrame()
    
    merged = merged.sort_values('timestamp')
    
    y = merged[f'{price_col}_1'].values
    x = merged[f'{price_col}_2'].values
    
    n = len(y)
    if n < 2:
        return pd.DataFrame()
    
    # Kalman Filter parameters
    hedge_ratio = initial_hedge
    P = 1.0  # State covariance
    
    hedge_ratios = []
    
    for i in range(n):
        # Prediction step
        # State: hedge_ratio
        # Measurement: y[i] = hedge_ratio * x[i] + noise
        
        # Predict measurement
        y_pred = hedge_ratio * x[i]
        
        # Innovation (measurement residual)
        innovation = y[i] - y_pred
        
        # Innovation covariance
        S = x[i]**2 * P + measurement_noise
        
        # Kalman gain
        K = (x[i] * P) / S
        
        # Update state
        hedge_ratio = hedge_ratio + K * innovation
        
        # Update covariance
        P = (1 - K * x[i]) * P + process_noise
        
        hedge_ratios.append(hedge_ratio)
    
    result = merged.copy()
    result['hedge_ratio'] = hedge_ratios
    
    return result


def mean_reversion_backtest(spread_df: pd.DataFrame,
                            zscore_df: pd.DataFrame,
                            entry_threshold: float = 2.0,
                            exit_threshold: float = 0.0,
                            initial_capital: float = 100000.0) -> Dict:
    """
    Simple mean reversion backtest
    
    Strategy: Enter when z-score > entry_threshold, exit when z-score < exit_threshold
    
    Args:
        spread_df: DataFrame with spread column
        zscore_df: DataFrame with zscore column
        entry_threshold: Z-score threshold for entry
        exit_threshold: Z-score threshold for exit
        initial_capital: Initial capital
    
    Returns:
        Dictionary with backtest results
    """
    if spread_df.empty or zscore_df.empty:
        return {}
    
    if 'spread' not in spread_df.columns or 'zscore' not in zscore_df.columns:
        return {}
    
    # Merge dataframes
    merged = pd.merge(spread_df[['timestamp', 'spread']],
                     zscore_df[['timestamp', 'zscore']],
                     on='timestamp')
    
    if merged.empty:
        return {}
    
    merged = merged.sort_values('timestamp')
    merged = merged.dropna(subset=['zscore', 'spread'])
    
    if merged.empty:
        return {}
    
    # Initialize backtest variables
    position = 0  # 0 = no position, 1 = long spread, -1 = short spread
    capital = initial_capital
    trades = []
    equity_curve = [initial_capital]
    
    entry_price = None
    
    for i in range(len(merged)):
        zscore = merged.iloc[i]['zscore']
        spread = merged.iloc[i]['spread']
        timestamp = merged.iloc[i]['timestamp']
        
        # Entry logic
        if position == 0:
            if zscore > entry_threshold:
                # Short spread (expect mean reversion)
                position = -1
                entry_price = spread
            elif zscore < -entry_threshold:
                # Long spread
                position = 1
                entry_price = spread
        
        # Exit logic
        elif position != 0:
            if (position == -1 and zscore < exit_threshold) or \
               (position == 1 and zscore > -exit_threshold):
                # Close position
                exit_price = spread
                pnl = (entry_price - exit_price) * position  # For short: profit when spread decreases
                
                trades.append({
                    'entry_time': entry_price,
                    'exit_time': timestamp,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': position,
                    'pnl': pnl
                })
                
                capital += pnl
                position = 0
                entry_price = None
        
        equity_curve.append(capital)
    
    # Close any open position
    if position != 0 and len(merged) > 0:
        exit_price = merged.iloc[-1]['spread']
        pnl = (entry_price - exit_price) * position
        capital += pnl
        equity_curve[-1] = capital
    
    # Compute statistics
    if not trades:
        return {
            'total_trades': 0,
            'final_capital': initial_capital,
            'total_return': 0.0,
            'win_rate': 0.0
        }
    
    trades_df = pd.DataFrame(trades)
    winning_trades = trades_df[trades_df['pnl'] > 0]
    
    total_return = (capital - initial_capital) / initial_capital * 100
    
    return {
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(trades) - len(winning_trades),
        'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
        'total_pnl': capital - initial_capital,
        'final_capital': capital,
        'total_return': total_return,
        'avg_pnl': trades_df['pnl'].mean(),
        'max_pnl': trades_df['pnl'].max(),
        'min_pnl': trades_df['pnl'].min(),
        'trades': trades,
        'equity_curve': equity_curve
    }

