# Quant Analytics Platform

A real-time quantitative analytics platform for trading data ingestion, processing, and visualization.

## Overview

This application provides end-to-end analytics for trading data, from real-time WebSocket ingestion to interactive visualization. It's designed for traders and researchers at MFT firms involved in statistical arbitrage, risk-premia harvesting, and market-making.

## Features

### Core Analytics
- **Price Statistics**: Real-time price metrics (mean, std, min, max, etc.)
- **Hedge Ratio**: OLS regression-based hedge ratio calculation
- **Spread Analysis**: Price spread computation and visualization
- **Z-Score**: Statistical z-score for mean reversion strategies
- **ADF Test**: Augmented Dickey-Fuller test for stationarity
- **Rolling Correlation**: Dynamic correlation analysis between pairs

### Advanced Features
- **Kalman Filter**: Dynamic hedge estimation
- **Robust Regression**: Huber and Theil-Sen regression methods
- **Mean Reversion Backtest**: Z-score based entry/exit simulation
- **Cross-Correlation Heatmaps**: Multi-asset correlation visualization
- **Custom Alerts**: Rule-based alerting system
- **Data Export**: CSV download for processed data and analytics

### Data Handling
- Real-time WebSocket ingestion from Binance
- Multiple timeframe sampling (1s, 1m, 5m)
- OHLC data upload support
- Efficient data storage and retrieval

## Architecture

The system follows a modular architecture:

```
┌─────────────────┐
│  WebSocket      │
│  Data Source    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Ingestion │
│  Module         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Storage Layer  │
│  (SQLite)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Analytics      │
│  Engine         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Frontend       │
│  (Streamlit)    │
└─────────────────┘
```

### Design Decisions

1. **SQLite for Storage**: Chosen for simplicity and zero-configuration. The architecture allows easy migration to PostgreSQL or Redis for production scale.

2. **Streamlit for Frontend**: Python-based framework that integrates seamlessly with the backend, enabling rapid development and real-time updates.

3. **Modular Architecture**: Each component (ingestion, storage, analytics, frontend) is decoupled, allowing independent scaling and modification.

4. **FastAPI Backend**: Provides RESTful API for future extensions and clean separation of concerns.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd gemscap-project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application with a single command:

```bash
python run.py
```

Or directly with Streamlit:

```bash
streamlit run app.py
```

The application will:
1. Start the WebSocket data ingestion in the background (when you click "Start Ingestion")
2. Begin aggregating and storing tick data
3. Display the interactive dashboard
4. Enable analytics as sufficient data becomes available

### Sample WebSocket Tool

A sample HTML WebSocket tool is provided in `sample_websocket.html` for reference. You can open it in a browser to see raw WebSocket data streaming. The main application uses a Python WebSocket client for production use.

## Configuration

Default symbols: BTCUSDT, ETHUSDT
Default timeframes: 1s, 1m, 5m

You can modify these in `config.py` or through the UI.

## Project Structure

```
gemscap-project/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration settings
├── data_ingestion/        # WebSocket ingestion module
│   ├── __init__.py
│   └── websocket_client.py
├── storage/               # Data storage layer
│   ├── __init__.py
│   ├── database.py
│   └── models.py
├── analytics/             # Analytics engine
│   ├── __init__.py
│   ├── price_stats.py
│   ├── regression.py
│   ├── spread.py
│   ├── zscore.py
│   ├── adf_test.py
│   ├── correlation.py
│   └── advanced.py
├── alerts/                # Alerting system
│   ├── __init__.py
│   └── alert_manager.py
├── frontend/              # Frontend components
│   ├── __init__.py
│   ├── charts.py
│   └── widgets.py
├── utils/                 # Utility functions
│   ├── __init__.py
│   └── helpers.py
├── requirements.txt
├── README.md
├── run.py                 # Simple runner script
└── sample_websocket.html  # Sample HTML WebSocket tool
```

## Analytics Methodology

### Price Statistics
Computes basic statistical measures: mean, standard deviation, min, max, percentiles over rolling windows.

### Hedge Ratio (OLS)
Uses Ordinary Least Squares regression to estimate the hedge ratio between two assets:
```
Y = α + βX + ε
```
where β is the hedge ratio.

### Spread
Price difference between two assets: `Spread = Price_A - β * Price_B`

### Z-Score
Normalized spread: `Z = (Spread - μ) / σ`
Used for mean reversion strategies.

### ADF Test
Augmented Dickey-Fuller test for stationarity. Tests the null hypothesis that a unit root is present.

### Rolling Correlation
Pearson correlation coefficient computed over rolling windows.

## Extensibility

The architecture supports easy extension:

- **New Data Sources**: Implement the `DataIngestion` interface
- **New Analytics**: Add functions to the `analytics/` module
- **New Storage**: Swap SQLite for PostgreSQL/Redis by modifying `storage/database.py`
- **New Visualizations**: Add components to `frontend/charts.py`

## Notes

- Analytics requiring more than a day of data are not included
- The system works without dummy data uploads
- Real-time updates are optimized for different timeframes


