"""
Widget creation functions for Streamlit
"""

import streamlit as st
from typing import List, Optional, Dict
from alerts.alert_manager import AlertRule, AlertManager


def create_symbol_selector(available_symbols: List[str], 
                          default_symbols: List[str] = None,
                          key: str = "symbol_selector") -> List[str]:
    """
    Create symbol selector widget
    
    Args:
        available_symbols: List of available symbols
        default_symbols: Default selected symbols
        key: Streamlit widget key
    
    Returns:
        List of selected symbols
    """
    if not available_symbols:
        return []
    
    default = default_symbols if default_symbols else [available_symbols[0]]
    
    selected = st.multiselect(
        "Select Symbols",
        options=available_symbols,
        default=default,
        key=key
    )
    
    return selected


def create_timeframe_selector(default: str = "1m",
                              key: str = "timeframe_selector") -> str:
    """
    Create timeframe selector widget
    
    Args:
        default: Default timeframe
        key: Streamlit widget key
    
    Returns:
        Selected timeframe
    """
    timeframes = ["1s", "1m", "5m"]
    
    selected = st.selectbox(
        "Timeframe",
        options=timeframes,
        index=timeframes.index(default) if default in timeframes else 0,
        key=key
    )
    
    return selected


def create_rolling_window_selector(default: int = 100,
                                   key: str = "rolling_window") -> int:
    """
    Create rolling window selector
    
    Args:
        default: Default window size
        key: Streamlit widget key
    
    Returns:
        Selected window size
    """
    window = st.slider(
        "Rolling Window",
        min_value=10,
        max_value=1000,
        value=default,
        step=10,
        key=key
    )
    
    return window


def create_regression_type_selector(default: str = "ols",
                                   key: str = "regression_type") -> str:
    """
    Create regression type selector
    
    Args:
        default: Default regression type
        key: Streamlit widget key
    
    Returns:
        Selected regression type
    """
    types = ["OLS", "Huber", "TheilSen"]
    
    selected = st.selectbox(
        "Regression Type",
        options=types,
        index=types.index(default.upper()) if default.upper() in types else 0,
        key=key
    )
    
    return selected.lower()


def create_alert_widget(alert_manager: AlertManager) -> Dict:
    """
    Create alert management widget
    
    Args:
        alert_manager: AlertManager instance
    
    Returns:
        Dictionary with alert actions
    """
    st.subheader("Alert Management")
    
    # Display existing alerts
    rules = alert_manager.get_rules()
    
    if rules:
        st.write("**Current Alerts:**")
        for rule in rules:
            status = "✅ Enabled" if rule.enabled else "❌ Disabled"
            st.write(f"- {rule.name}: {rule.condition} {rule.threshold} ({status})")
    
    # Create new alert
    with st.expander("Create New Alert"):
        alert_name = st.text_input("Alert Name", key="alert_name")
        alert_condition = st.selectbox(
            "Condition",
            options=["zscore_gt", "zscore_lt", "spread_gt", "spread_lt", "price_gt", "price_lt"],
            key="alert_condition"
        )
        alert_threshold = st.number_input("Threshold", value=2.0, key="alert_threshold")
        alert_symbol = st.text_input("Symbol (optional, leave empty for all)", key="alert_symbol")
        
        if st.button("Add Alert"):
            if alert_name:
                rule = AlertRule(
                    name=alert_name,
                    condition=alert_condition,
                    threshold=alert_threshold,
                    symbol=alert_symbol if alert_symbol else None
                )
                alert_manager.add_rule(rule)
                st.success(f"Alert '{alert_name}' added!")
                st.rerun()
    
    # Remove alert
    if rules:
        with st.expander("Remove Alert"):
            alert_to_remove = st.selectbox(
                "Select Alert to Remove",
                options=[r.name for r in rules],
                key="alert_to_remove"
            )
            
            if st.button("Remove Alert"):
                alert_manager.remove_rule(alert_to_remove)
                st.success(f"Alert '{alert_to_remove}' removed!")
                st.rerun()
    
    return {"rules": rules}

