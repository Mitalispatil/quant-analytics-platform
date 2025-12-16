"""
Configuration settings for the Quant Analytics Platform
"""

# Default symbols to track
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Binance WebSocket endpoint
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/"

# Data storage
DATABASE_PATH = "trading_data.db"

# Sampling timeframes (in seconds)
TIMEFRAMES = {
    "1s": 1,
    "1m": 60,
    "5m": 300
}

# Default rolling window sizes
DEFAULT_ROLLING_WINDOW = 100  # ticks
DEFAULT_CORRELATION_WINDOW = 60  # seconds

# Update intervals (in milliseconds)
TICK_UPDATE_INTERVAL = 500  # 500ms for tick-based analytics
TIMEFRAME_UPDATE_INTERVAL = {
    "1s": 1000,
    "1m": 60000,
    "5m": 300000
}

# Alert thresholds
DEFAULT_ZSCORE_THRESHOLD = 2.0

# Database settings
DB_POOL_SIZE = 10
DB_MAX_OVERFLOW = 20

