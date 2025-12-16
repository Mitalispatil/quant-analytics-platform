"""
Chart creation functions using Plotly
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional, Dict, List


def create_price_chart(df: pd.DataFrame, symbol: str, 
                      price_col: str = 'price',
                      timeframe: str = '1m') -> go.Figure:
    """
    Create price chart
    
    Args:
        df: DataFrame with price data
        symbol: Symbol name
        price_col: Name of the price column
        timeframe: Timeframe string
    
    Returns:
        Plotly figure
    """
    if df.empty or price_col not in df.columns:
        return go.Figure()
    
    fig = go.Figure()
    
    if 'timestamp' in df.columns:
        x_data = df['timestamp']
    else:
        x_data = df.index
    
    fig.add_trace(go.Scatter(
        x=x_data,
        y=df[price_col],
        mode='lines',
        name=f'{symbol} Price',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Time: %{x}<br>' +
                      'Price: $%{y:,.2f}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{symbol} Price Chart ({timeframe})',
        xaxis_title='Time',
        yaxis_title='Price (USDT)',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        showlegend=True
    )
    
    return fig


def create_spread_chart(spread_df: pd.DataFrame, symbol1: str, symbol2: str) -> go.Figure:
    """
    Create spread chart
    
    Args:
        spread_df: DataFrame with spread data
        symbol1: First symbol
        symbol2: Second symbol
    
    Returns:
        Plotly figure
    """
    if spread_df.empty or 'spread' not in spread_df.columns:
        return go.Figure()
    
    fig = go.Figure()
    
    if 'timestamp' in spread_df.columns:
        x_data = spread_df['timestamp']
    else:
        x_data = spread_df.index
    
    fig.add_trace(go.Scatter(
        x=x_data,
        y=spread_df['spread'],
        mode='lines',
        name='Spread',
        line=dict(color='#ff7f0e', width=2),
        hovertemplate='<b>Spread</b><br>' +
                      'Time: %{x}<br>' +
                      'Spread: %{y:,.4f}<br>' +
                      '<extra></extra>'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                  annotation_text="Zero Line")
    
    fig.update_layout(
        title=f'Spread: {symbol1} - {symbol2}',
        xaxis_title='Time',
        yaxis_title='Spread',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        showlegend=True
    )
    
    return fig


def create_zscore_chart(zscore_df: pd.DataFrame, 
                       entry_threshold: float = 2.0,
                       exit_threshold: float = 0.0) -> go.Figure:
    """
    Create z-score chart with thresholds
    
    Args:
        zscore_df: DataFrame with zscore column
        entry_threshold: Entry threshold line
        exit_threshold: Exit threshold line
    
    Returns:
        Plotly figure
    """
    if zscore_df.empty or 'zscore' not in zscore_df.columns:
        return go.Figure()
    
    fig = go.Figure()
    
    if 'timestamp' in zscore_df.columns:
        x_data = zscore_df['timestamp']
    else:
        x_data = zscore_df.index
    
    fig.add_trace(go.Scatter(
        x=x_data,
        y=zscore_df['zscore'],
        mode='lines',
        name='Z-Score',
        line=dict(color='#2ca02c', width=2),
        hovertemplate='<b>Z-Score</b><br>' +
                      'Time: %{x}<br>' +
                      'Z-Score: %{y:,.2f}<br>' +
                      '<extra></extra>'
    ))
    
    # Add threshold lines
    fig.add_hline(y=entry_threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Entry ({entry_threshold})")
    fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Entry (-{entry_threshold})")
    fig.add_hline(y=exit_threshold, line_dash="dot", line_color="blue",
                  annotation_text=f"Exit ({exit_threshold})")
    fig.add_hline(y=-exit_threshold, line_dash="dot", line_color="blue",
                  annotation_text=f"Exit (-{exit_threshold})")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title='Z-Score Chart',
        xaxis_title='Time',
        yaxis_title='Z-Score',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        showlegend=True
    )
    
    return fig


def create_correlation_chart(corr_df: pd.DataFrame, symbol1: str, symbol2: str) -> go.Figure:
    """
    Create rolling correlation chart
    
    Args:
        corr_df: DataFrame with correlation data
        symbol1: First symbol
        symbol2: Second symbol
    
    Returns:
        Plotly figure
    """
    if corr_df.empty or 'correlation' not in corr_df.columns:
        return go.Figure()
    
    fig = go.Figure()
    
    if 'timestamp' in corr_df.columns:
        x_data = corr_df['timestamp']
    else:
        x_data = corr_df.index
    
    fig.add_trace(go.Scatter(
        x=x_data,
        y=corr_df['correlation'],
        mode='lines',
        name='Correlation',
        line=dict(color='#9467bd', width=2),
        hovertemplate='<b>Correlation</b><br>' +
                      'Time: %{x}<br>' +
                      'Correlation: %{y:,.4f}<br>' +
                      '<extra></extra>'
    ))
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=1, line_dash="dot", line_color="green", opacity=0.3)
    fig.add_hline(y=-1, line_dash="dot", line_color="red", opacity=0.3)
    
    fig.update_layout(
        title=f'Rolling Correlation: {symbol1} vs {symbol2}',
        xaxis_title='Time',
        yaxis_title='Correlation',
        yaxis_range=[-1.1, 1.1],
        hovermode='x unified',
        template='plotly_white',
        height=400,
        showlegend=True
    )
    
    return fig


def create_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """
    Create correlation heatmap
    
    Args:
        corr_matrix: Correlation matrix DataFrame
    
    Returns:
        Plotly figure
    """
    if corr_matrix.empty:
        return go.Figure()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        hovertemplate='<b>%{y} vs %{x}</b><br>' +
                      'Correlation: %{z:,.4f}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title='Correlation Heatmap',
        xaxis_title='Symbol',
        yaxis_title='Symbol',
        template='plotly_white',
        height=500,
        width=600
    )
    
    return fig


def create_equity_curve(equity_curve: List[float], timestamps: List = None) -> go.Figure:
    """
    Create equity curve chart for backtest
    
    Args:
        equity_curve: List of equity values
        timestamps: Optional list of timestamps
    
    Returns:
        Plotly figure
    """
    if not equity_curve:
        return go.Figure()
    
    fig = go.Figure()
    
    x_data = timestamps if timestamps else list(range(len(equity_curve)))
    
    fig.add_trace(go.Scatter(
        x=x_data,
        y=equity_curve,
        mode='lines',
        name='Equity Curve',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)',
        hovertemplate='<b>Equity</b><br>' +
                      'Value: $%{y:,.2f}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title='Backtest Equity Curve',
        xaxis_title='Time',
        yaxis_title='Capital ($)',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        showlegend=True
    )
    
    return fig

