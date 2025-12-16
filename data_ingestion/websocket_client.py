"""
WebSocket client for Binance tick data ingestion
"""

import asyncio
import websockets
import json
from datetime import datetime
from typing import List, Callable
import logging
import config
from storage.database import get_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketClient:
    """WebSocket client for Binance tick data"""
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or config.DEFAULT_SYMBOLS
        self.db = get_db()
        self.running = False
        self.callbacks = []
        
    def add_callback(self, callback: Callable):
        """Add a callback function to be called on each tick"""
        self.callbacks.append(callback)
    
    def _build_stream_url(self) -> str:
        """Build Binance WebSocket stream URL"""
        # Convert symbols to lowercase stream names
        streams = [f"{symbol.lower()}@trade" for symbol in self.symbols]
        
        if len(streams) == 1:
            # Single stream: use direct URL
            return f"{config.BINANCE_WS_URL}{streams[0]}"
        else:
            # Multiple streams: use combined stream
            # Format: wss://stream.binance.com:9443/stream?streams=stream1/stream2
            stream_names = "/".join(streams)
            # For combined streams, use /stream endpoint, not /ws/stream
            base_url = config.BINANCE_WS_URL.replace("/ws/", "/")
            return f"{base_url}stream?streams={stream_names}"
    
    async def _process_message(self, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            # Binance stream format can be:
            # 1. Combined stream: {"stream": "btcusdt@trade", "data": {...}}
            # 2. Single stream: direct trade data {...}
            trade_data = None
            stream_name = None
            
            if 'data' in data:
                # Combined stream format
                trade_data = data['data']
                stream_name = data.get('stream', 'unknown')
                logger.debug(f"Combined stream message for: {stream_name}")
            elif 'e' in data and data['e'] == 'trade':
                # Single stream format (direct trade data)
                trade_data = data
                logger.debug("Single stream message")
            else:
                # Unknown format, log and skip
                logger.warning(f"Unknown message format. Keys: {list(data.keys())[:5]}, Sample: {str(data)[:200]}")
                return
            
            if trade_data:
                symbol = trade_data.get('s')  # Symbol
                if not symbol:
                    logger.warning(f"Missing symbol in trade_data. Keys: {list(trade_data.keys())}")
                    return
                    
                price = float(trade_data['p'])  # Price
                quantity = float(trade_data['q'])  # Quantity
                timestamp_ms = trade_data['T']  # Trade time
                timestamp = datetime.utcfromtimestamp(timestamp_ms / 1000.0)
                
                # Log received symbol for debugging (first 20 messages per symbol)
                if not hasattr(self, '_symbol_counts'):
                    self._symbol_counts = {}
                if symbol not in self._symbol_counts:
                    self._symbol_counts[symbol] = 0
                    logger.info(f"First message received for {symbol} from stream: {stream_name or 'single'}")
                self._symbol_counts[symbol] += 1
                if self._symbol_counts[symbol] <= 20:
                    logger.info(f"Received tick #{self._symbol_counts[symbol]}: {symbol} @ {price}")
                
                # Store in database
                try:
                    self.db.insert_tick(symbol, price, quantity, timestamp)
                    if self._symbol_counts[symbol] <= 5:
                        logger.info(f"Successfully stored {symbol} tick #{self._symbol_counts[symbol]}")
                except Exception as db_error:
                    logger.error(f"Database error storing {symbol}: {db_error}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                # Call registered callbacks
                tick_info = {
                    'symbol': symbol,
                    'price': price,
                    'quantity': quantity,
                    'timestamp': timestamp
                }
                
                for callback in self.callbacks:
                    try:
                        callback(tick_info)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}, message: {message[:100] if len(message) > 100 else message}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def _connect_and_listen(self):
        """Connect to WebSocket and listen for messages"""
        url = self._build_stream_url()
        logger.info(f"Connecting to Binance WebSocket: {url}")
        
        while self.running:
            try:
                async with websockets.connect(url) as websocket:
                    logger.info("WebSocket connected")
                    async for message in websocket:
                        if not self.running:
                            break
                        await self._process_message(message)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed, reconnecting...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"WebSocket error: {e}, reconnecting...")
                await asyncio.sleep(5)
    
    async def start(self):
        """Start the WebSocket client"""
        self.running = True
        await self._connect_and_listen()
    
    def stop(self):
        """Stop the WebSocket client"""
        self.running = False


def start_ingestion(symbols: List[str] = None):
    """Start data ingestion in a background task"""
    client = WebSocketClient(symbols)
    
    async def run():
        await client.start()
    
    # Run in background
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    task = loop.create_task(run())
    
    return client, loop, task

