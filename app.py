"""
Main Streamlit application for Quant Analytics Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import threading
import time
import io
import logging
import plotly.graph_objects as go

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modules
import config
from storage.database import get_db
from data_ingestion.websocket_client import WebSocketClient
from analytics import (
    compute_price_stats,
    compute_ols_hedge_ratio,
    compute_robust_regression,
    compute_spread,
    compute_spread_stats,
    compute_zscore,
    compute_adf_test,
    compute_adf_test_spread,
    compute_rolling_correlation,
    compute_correlation_heatmap,
    kalman_filter_hedge,
    mean_reversion_backtest
)
from alerts.alert_manager import AlertManager, AlertRule
from frontend.charts import (
    create_price_chart,
    create_spread_chart,
    create_zscore_chart,
    create_correlation_chart,
    create_heatmap,
    create_equity_curve
)
from frontend.widgets import (
    create_symbol_selector,
    create_timeframe_selector,
    create_rolling_window_selector,
    create_regression_type_selector,
    create_alert_widget
)
from utils.helpers import resample_data, format_number, prepare_data_for_export

# Page configuration
st.set_page_config(
    page_title="Quant Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'ws_client' not in st.session_state:
    st.session_state.ws_client = None
    st.session_state.ws_loop = None
    st.session_state.ws_task = None
    st.session_state.alert_manager = AlertManager()
    st.session_state.ingestion_started = False
    st.session_state.last_update = {}

# Initialize database
db = get_db()


def start_websocket_ingestion(symbols):
    """Start WebSocket ingestion in background"""
    if st.session_state.ingestion_started:
        return
    
    def run_ws():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            client = WebSocketClient(symbols)
            
            # Add callback for alerts
            def tick_callback(tick_info):
                # Check alerts on new tick
                symbol = tick_info['symbol']
                # Get latest metrics for alert checking
                # This is simplified - in production, you'd cache recent metrics
                pass
            
            client.add_callback(tick_callback)
            
            # Store client reference before starting
            st.session_state.ws_client = client
            st.session_state.ws_loop = loop
            
            # Start the client as a task
            logger.info(f"Starting WebSocket client for symbols: {symbols}")
            loop.create_task(client.start())
            
            # Run the event loop forever
            loop.run_forever()
        except Exception as e:
            logger.error(f"WebSocket thread error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            try:
                st.session_state.ingestion_started = False
            except:
                pass
    
    thread = threading.Thread(target=run_ws, daemon=True)
    thread.start()
    st.session_state.ingestion_started = True
    logger.info("WebSocket ingestion thread started")
    time.sleep(2)  # Give thread time to start


def stop_websocket_ingestion():
    """Stop WebSocket ingestion"""
    try:
        if st.session_state.ws_client:
            logger.info("Stopping WebSocket client...")
            st.session_state.ws_client.stop()
        if st.session_state.ws_loop and st.session_state.ws_loop.is_running():
            st.session_state.ws_loop.call_soon_threadsafe(st.session_state.ws_loop.stop)
        st.session_state.ingestion_started = False
        logger.info("WebSocket ingestion stopped")
    except Exception as e:
        logger.error(f"Error stopping WebSocket: {e}")
        st.session_state.ingestion_started = False


# Main UI
st.title("📊 Quant Analytics Platform")
st.markdown("Real-time trading data analytics and visualization")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Symbol selection
    # Get symbols from database and combine with default symbols
    db_symbols = db.get_all_symbols()
    # Always include default symbols, plus any additional symbols from database
    available_symbols = list(set(config.DEFAULT_SYMBOLS + db_symbols))
    available_symbols.sort()  # Sort for consistent display
    
    selected_symbols = create_symbol_selector(
        available_symbols,
        default_symbols=config.DEFAULT_SYMBOLS[:2] if len(available_symbols) >= 2 else available_symbols[:1]
    )
    
    # Timeframe selection
    timeframe = create_timeframe_selector(default="1m")
    
    # Rolling window
    rolling_window = create_rolling_window_selector(default=100)
    
    # Regression type
    regression_type = create_regression_type_selector(default="ols")
    
    # Data ingestion control
    st.header("Data Ingestion")
    if st.button("Start Ingestion", disabled=st.session_state.ingestion_started):
        if selected_symbols:
            start_websocket_ingestion(selected_symbols)
            st.success("Data ingestion started!")
            st.rerun()
        else:
            st.error("Please select at least one symbol")
    
    if st.button("Stop Ingestion", disabled=not st.session_state.ingestion_started):
        stop_websocket_ingestion()
        st.success("Data ingestion stopped!")
        st.rerun()
    
    if st.session_state.ingestion_started:
        st.success("✅ Ingestion Active")
    
    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.rerun()
    
    # Alert management
    create_alert_widget(st.session_state.alert_manager)

# Main content area
if not selected_symbols:
    st.warning("Please select at least one symbol from the sidebar")
    st.stop()

# Tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Price Analytics",
    "🔗 Pair Analytics",
    "📊 Advanced Analytics",
    "📥 Data Export",
    "📤 OHLC Upload"
])

# Tab 1: Price Analytics
with tab1:
    st.header("Price Analytics")
    
    if len(selected_symbols) > 0:
        symbol = selected_symbols[0]
        
        # Get data
        df = db.get_latest_ticks(symbol, count=1000)
        
        if not df.empty:
            # Resample if needed
            if timeframe != "1s":
                df_resampled = resample_data(df, timeframe)
                if not df_resampled.empty:
                    # Create OHLC chart
                    fig_ohlc = go.Figure(data=go.Candlestick(
                        x=df_resampled['timestamp'],
                        open=df_resampled['open'],
                        high=df_resampled['high'],
                        low=df_resampled['low'],
                        close=df_resampled['close']
                    ))
                    fig_ohlc.update_layout(
                        title=f'{symbol} OHLC Chart ({timeframe})',
                        xaxis_title='Time',
                        yaxis_title='Price',
                        height=500
                    )
                    st.plotly_chart(fig_ohlc, use_container_width=True)
            
            # Price chart
            fig_price = create_price_chart(df, symbol, timeframe=timeframe)
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Price statistics
            col1, col2, col3, col4 = st.columns(4)
            
            stats = compute_price_stats(df, window=rolling_window)
            
            with col1:
                st.metric("Current Price", format_number(df['price'].iloc[-1] if len(df) > 0 else 0))
                st.metric("Mean", format_number(stats.get('mean', 0)))
            
            with col2:
                st.metric("Std Dev", format_number(stats.get('std', 0)))
                st.metric("Min", format_number(stats.get('min', 0)))
            
            with col3:
                st.metric("Max", format_number(stats.get('max', 0)))
                st.metric("Median", format_number(stats.get('median', 0)))
            
            with col4:
                st.metric("Q25", format_number(stats.get('q25', 0)))
                st.metric("Q75", format_number(stats.get('q75', 0)))
            
            # Time-series stats table
            st.subheader("Time-Series Statistics")
            if timeframe != "1s":
                df_resampled = resample_data(df, timeframe)
                if not df_resampled.empty:
                    # Compute stats for each period
                    stats_list = []
                    for idx, row in df_resampled.iterrows():
                        period_df = df[
                            (df['timestamp'] >= row['timestamp'] - pd.Timedelta(seconds=config.TIMEFRAMES[timeframe])) &
                            (df['timestamp'] <= row['timestamp'])
                        ]
                        if not period_df.empty:
                            period_stats = compute_price_stats(period_df)
                            stats_list.append({
                                'timestamp': row['timestamp'],
                                'open': row['open'],
                                'high': row['high'],
                                'low': row['low'],
                                'close': row['close'],
                                'volume': row.get('volume', 0),
                                'mean': period_stats.get('mean', 0),
                                'std': period_stats.get('std', 0),
                                'min': period_stats.get('min', 0),
                                'max': period_stats.get('max', 0)
                            })
                    
                    if stats_list:
                        stats_df = pd.DataFrame(stats_list)
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # Download button
                        csv = prepare_data_for_export(stats_df).to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"{symbol}_stats_{timeframe}.csv",
                            mime="text/csv"
                        )
        else:
            st.info(f"No data available for {symbol}. Data ingestion may be starting...")

# Tab 2: Pair Analytics
with tab2:
    st.header("Pair Analytics")
    
    if len(selected_symbols) >= 2:
        symbol1 = selected_symbols[0]
        symbol2 = selected_symbols[1]
        
        # Get data for both symbols
        df1 = db.get_latest_ticks(symbol1, count=1000)
        df2 = db.get_latest_ticks(symbol2, count=1000)
        
        if not df1.empty and not df2.empty:
            # Hedge ratio
            st.subheader("Hedge Ratio")
            
            if regression_type == "ols":
                hedge_result = compute_ols_hedge_ratio(df1, df2)
            else:
                hedge_result = compute_robust_regression(df1, df2, method=regression_type)
            
            if hedge_result and 'error' not in hedge_result:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Hedge Ratio", format_number(hedge_result.get('hedge_ratio', 0), 4))
                with col2:
                    st.metric("R²", format_number(hedge_result.get('r_squared', 0), 4))
                with col3:
                    st.metric("Intercept", format_number(hedge_result.get('intercept', 0), 4))
                
                hedge_ratio = hedge_result.get('hedge_ratio', 1.0)
            else:
                hedge_ratio = 1.0
                st.warning("Could not compute hedge ratio. Using 1.0")
            
            # Spread
            st.subheader("Spread Analysis")
            spread_df = compute_spread(df1, df2, hedge_ratio=hedge_ratio)
            
            if not spread_df.empty:
                fig_spread = create_spread_chart(spread_df, symbol1, symbol2)
                st.plotly_chart(fig_spread, use_container_width=True)
                
                spread_stats = compute_spread_stats(spread_df)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Spread", format_number(spread_stats.get('current', 0), 4))
                with col2:
                    st.metric("Mean", format_number(spread_stats.get('mean', 0), 4))
                with col3:
                    st.metric("Std Dev", format_number(spread_stats.get('std', 0), 4))
                with col4:
                    st.metric("Min / Max", 
                             f"{format_number(spread_stats.get('min', 0), 4)} / {format_number(spread_stats.get('max', 0), 4)}")
            
            # Z-Score
            st.subheader("Z-Score Analysis")
            zscore_df = compute_zscore(spread_df, window=rolling_window)
            
            if not zscore_df.empty:
                entry_threshold = st.slider("Entry Threshold", 0.0, 5.0, 2.0, 0.1)
                exit_threshold = st.slider("Exit Threshold", -2.0, 2.0, 0.0, 0.1)
                
                fig_zscore = create_zscore_chart(zscore_df, entry_threshold, exit_threshold)
                st.plotly_chart(fig_zscore, use_container_width=True)
                
                current_zscore = zscore_df['zscore'].iloc[-1] if len(zscore_df) > 0 else None
                if current_zscore is not None:
                    st.metric("Current Z-Score", format_number(current_zscore, 2))
                    
                    # Check alerts
                    metrics = {'zscore': current_zscore, 'spread': spread_stats.get('current', 0)}
                    triggered = st.session_state.alert_manager.check_alerts(metrics, symbol1)
                    if triggered:
                        for alert in triggered:
                            st.warning(f"🚨 Alert: {alert['rule'].name} triggered! Value: {alert['value']}")
            
            # Correlation
            st.subheader("Rolling Correlation")
            corr_df = compute_rolling_correlation(df1, df2, window=rolling_window)
            
            if not corr_df.empty:
                fig_corr = create_correlation_chart(corr_df, symbol1, symbol2)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                current_corr = corr_df['correlation'].iloc[-1] if len(corr_df) > 0 else None
                if current_corr is not None:
                    st.metric("Current Correlation", format_number(current_corr, 4))
            
            # ADF Test
            st.subheader("Stationarity Test (ADF)")
            if st.button("Run ADF Test"):
                adf_result = compute_adf_test_spread(spread_df)
                if adf_result and 'error' not in adf_result:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("ADF Statistic", format_number(adf_result.get('adf_statistic', 0), 4))
                    with col2:
                        st.metric("P-Value", format_number(adf_result.get('pvalue', 0), 4))
                    with col3:
                        is_stationary = adf_result.get('is_stationary', False)
                        st.metric("Result", "✅ Stationary" if is_stationary else "❌ Non-Stationary")
                    
                    st.json(adf_result)
        else:
            st.info("Insufficient data for pair analysis. Please wait for data ingestion...")
    else:
        st.warning("Please select at least 2 symbols for pair analysis")

# Tab 3: Advanced Analytics
with tab3:
    st.header("Advanced Analytics")
    
    if len(selected_symbols) >= 2:
        symbol1 = selected_symbols[0]
        symbol2 = selected_symbols[1]
        
        df1 = db.get_latest_ticks(symbol1, count=1000)
        df2 = db.get_latest_ticks(symbol2, count=1000)
        
        if not df1.empty and not df2.empty:
            # Kalman Filter
            st.subheader("Kalman Filter Hedge Ratio")
            kalman_df = kalman_filter_hedge(df1, df2)
            
            if not kalman_df.empty:
                fig_kalman = go.Figure()
                fig_kalman.add_trace(go.Scatter(
                    x=kalman_df['timestamp'],
                    y=kalman_df['hedge_ratio'],
                    mode='lines',
                    name='Dynamic Hedge Ratio'
                ))
                fig_kalman.update_layout(
                    title='Kalman Filter Hedge Ratio',
                    xaxis_title='Time',
                    yaxis_title='Hedge Ratio',
                    height=400
                )
                st.plotly_chart(fig_kalman, use_container_width=True)
                
                current_kalman_hedge = kalman_df['hedge_ratio'].iloc[-1] if len(kalman_df) > 0 else None
                if current_kalman_hedge:
                    st.metric("Current Dynamic Hedge Ratio", format_number(current_kalman_hedge, 4))
            
            # Mean Reversion Backtest
            st.subheader("Mean Reversion Backtest")
            spread_df = compute_spread(df1, df2)
            zscore_df = compute_zscore(spread_df, window=rolling_window)
            
            if not spread_df.empty and not zscore_df.empty:
                entry_threshold = st.slider("Backtest Entry Threshold", 0.0, 5.0, 2.0, 0.1, key="backtest_entry")
                exit_threshold = st.slider("Backtest Exit Threshold", -2.0, 2.0, 0.0, 0.1, key="backtest_exit")
                initial_capital = st.number_input("Initial Capital", 10000.0, 1000000.0, 100000.0, 10000.0)
                
                if st.button("Run Backtest"):
                    backtest_result = mean_reversion_backtest(
                        spread_df, zscore_df,
                        entry_threshold=entry_threshold,
                        exit_threshold=exit_threshold,
                        initial_capital=initial_capital
                    )
                    
                    if backtest_result:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Trades", backtest_result.get('total_trades', 0))
                        with col2:
                            st.metric("Win Rate", f"{backtest_result.get('win_rate', 0):.2f}%")
                        with col3:
                            st.metric("Total Return", f"{backtest_result.get('total_return', 0):.2f}%")
                        with col4:
                            st.metric("Final Capital", format_number(backtest_result.get('final_capital', 0)))
                        
                        # Equity curve
                        equity_curve = backtest_result.get('equity_curve', [])
                        if equity_curve:
                            fig_equity = create_equity_curve(equity_curve)
                            st.plotly_chart(fig_equity, use_container_width=True)
            
            # Correlation Heatmap
            if len(selected_symbols) >= 3:
                st.subheader("Correlation Heatmap")
                df_dict = {}
                for symbol in selected_symbols[:5]:  # Limit to 5 symbols
                    df = db.get_latest_ticks(symbol, count=500)
                    if not df.empty:
                        df_dict[symbol] = df
                
                if len(df_dict) >= 2:
                    corr_matrix = compute_correlation_heatmap(df_dict)
                    if not corr_matrix.empty:
                        fig_heatmap = create_heatmap(corr_matrix)
                        st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Insufficient data for advanced analytics...")
    else:
        st.warning("Please select at least 2 symbols for advanced analytics")

# Tab 4: Data Export
with tab4:
    st.header("Data Export")
    
    export_symbol = st.selectbox("Select Symbol", selected_symbols)
    export_timeframe = create_timeframe_selector(default=timeframe, key="export_timeframe")
    
    if st.button("Generate Export"):
        df = db.get_latest_ticks(export_symbol, count=10000)
        
        if not df.empty:
            if export_timeframe != "1s":
                df_export = resample_data(df, export_timeframe)
            else:
                df_export = df
            
            if not df_export.empty:
                df_export = prepare_data_for_export(df_export)
                csv = df_export.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{export_symbol}_{export_timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                st.dataframe(df_export.head(100), use_container_width=True)
            else:
                st.error("No data to export")
        else:
            st.error("No data available for export")

# Tab 5: OHLC Upload
with tab5:
    st.header("OHLC Data Upload")
    
    uploaded_file = st.file_uploader("Upload OHLC CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            
            # Expected columns: timestamp, symbol, open, high, low, close, volume, timeframe
            st.write("**Preview:**")
            st.dataframe(df_upload.head(), use_container_width=True)
            
            # Validate columns
            required_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close']
            if all(col in df_upload.columns for col in required_cols):
                upload_symbol = st.selectbox("Symbol", df_upload['symbol'].unique() if 'symbol' in df_upload.columns else [])
                upload_timeframe = st.selectbox("Timeframe", ["1s", "1m", "5m"])
                
                if st.button("Upload to Database"):
                    df_filtered = df_upload[df_upload['symbol'] == upload_symbol] if upload_symbol else df_upload
                    df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])
                    
                    count = 0
                    for _, row in df_filtered.iterrows():
                        db.insert_ohlc(
                            symbol=row['symbol'],
                            timeframe=upload_timeframe,
                            open=float(row['open']),
                            high=float(row['high']),
                            low=float(row['low']),
                            close=float(row['close']),
                            volume=float(row.get('volume', 0)),
                            timestamp=row['timestamp']
                        )
                        count += 1
                    
                    st.success(f"Uploaded {count} OHLC records!")
            else:
                st.error(f"Missing required columns. Expected: {', '.join(required_cols)}")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Auto-refresh (commented out to prevent infinite loop)
# Use manual refresh button or implement proper polling mechanism
# if st.session_state.ingestion_started:
#     time.sleep(0.5)
#     st.rerun()

