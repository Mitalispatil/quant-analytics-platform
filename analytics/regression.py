"""
Regression analysis for hedge ratio computation
"""

import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from sklearn.linear_model import HuberRegressor, TheilSenRegressor
from typing import Dict, Tuple, Optional


def compute_ols_hedge_ratio(df1: pd.DataFrame, df2: pd.DataFrame,
                            price_col: str = 'price') -> Dict:
    """
    Compute hedge ratio using OLS regression
    
    Args:
        df1: DataFrame for asset 1 (dependent variable)
        df2: DataFrame for asset 2 (independent variable)
        price_col: Name of the price column
    
    Returns:
        Dictionary with regression results
    """
    if df1.empty or df2.empty or price_col not in df1.columns or price_col not in df2.columns:
        return {}
    
    # Align dataframes by timestamp
    merged = pd.merge(df1[['timestamp', price_col]], 
                     df2[['timestamp', price_col]], 
                     on='timestamp', 
                     suffixes=('_1', '_2'))
    
    if merged.empty:
        return {}
    
    y = merged[f'{price_col}_1'].values
    x = merged[f'{price_col}_2'].values
    
    if len(y) < 2:
        return {}
    
    # Add constant for intercept
    x_with_const = add_constant(x)
    
    try:
        model = OLS(y, x_with_const).fit()
        
        hedge_ratio = model.params[1]  # Beta coefficient
        intercept = model.params[0]  # Alpha
        r_squared = model.rsquared
        pvalue = model.pvalues[1] if len(model.pvalues) > 1 else None
        
        return {
            'hedge_ratio': float(hedge_ratio),
            'intercept': float(intercept),
            'r_squared': float(r_squared),
            'pvalue': float(pvalue) if pvalue else None,
            'n_observations': int(len(y)),
            'residuals': model.resid.tolist()
        }
    except Exception as e:
        return {'error': str(e)}


def compute_robust_regression(df1: pd.DataFrame, df2: pd.DataFrame,
                              price_col: str = 'price',
                              method: str = 'huber') -> Dict:
    """
    Compute hedge ratio using robust regression
    
    Args:
        df1: DataFrame for asset 1 (dependent variable)
        df2: DataFrame for asset 2 (independent variable)
        price_col: Name of the price column
        method: 'huber' or 'theilsen'
    
    Returns:
        Dictionary with regression results
    """
    if df1.empty or df2.empty or price_col not in df1.columns or price_col not in df2.columns:
        return {}
    
    # Align dataframes by timestamp
    merged = pd.merge(df1[['timestamp', price_col]], 
                     df2[['timestamp', price_col]], 
                     on='timestamp', 
                     suffixes=('_1', '_2'))
    
    if merged.empty:
        return {}
    
    y = merged[f'{price_col}_1'].values.reshape(-1, 1)
    x = merged[f'{price_col}_2'].values.reshape(-1, 1)
    
    if len(y) < 2:
        return {}
    
    try:
        if method.lower() == 'huber':
            model = HuberRegressor(epsilon=1.35)
        elif method.lower() == 'theilsen':
            model = TheilSenRegressor()
        else:
            return {'error': f'Unknown method: {method}'}
        
        model.fit(x, y.ravel())
        
        hedge_ratio = float(model.coef_[0])
        intercept = float(model.intercept_)
        
        # Compute R-squared
        y_pred = model.predict(x)
        ss_res = np.sum((y.ravel() - y_pred) ** 2)
        ss_tot = np.sum((y.ravel() - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'hedge_ratio': hedge_ratio,
            'intercept': intercept,
            'r_squared': float(r_squared),
            'method': method,
            'n_observations': int(len(y))
        }
    except Exception as e:
        return {'error': str(e)}

