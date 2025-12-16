# Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation Steps

1. **Clone or navigate to the project directory**
   ```bash
   cd gemscap-project
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the application**
   - The application will automatically open in your default browser
   - If not, navigate to `http://localhost:8501`

## First Steps

1. **Start Data Ingestion**
   - In the sidebar, select symbols (e.g., BTCUSDT, ETHUSDT)
   - Click "Start Ingestion" button
   - Wait a few seconds for data to accumulate

2. **View Price Analytics**
   - Go to the "Price Analytics" tab
   - Select a timeframe (1s, 1m, 5m)
   - View price charts and statistics

3. **Analyze Pairs**
   - Go to the "Pair Analytics" tab
   - Select at least 2 symbols
   - View hedge ratio, spread, z-score, and correlation

4. **Set Up Alerts**
   - In the sidebar, expand "Alert Management"
   - Create custom alerts (e.g., z-score > 2)
   - Alerts will trigger when conditions are met

5. **Export Data**
   - Go to the "Data Export" tab
   - Select symbol and timeframe
   - Click "Generate Export" and download CSV

## Troubleshooting

### WebSocket Connection Issues
- Check your internet connection
- Verify Binance WebSocket is accessible
- Try different symbols if one fails

### No Data Showing
- Ensure data ingestion is started
- Wait a few seconds for data to accumulate
- Check that symbols are valid (e.g., BTCUSDT, not BTC/USDT)

### Database Errors
- Delete `trading_data.db` and restart
- Check file permissions in the project directory

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify you're using the correct Python version (3.8+)

## Tips

- Use the "Refresh Data" button to update charts manually
- Start with 1-minute timeframe for faster data accumulation
- For pair analysis, ensure both symbols have sufficient data
- ADF test requires at least 10 data points

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
