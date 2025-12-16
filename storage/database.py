"""
Database operations and connection management
"""

import sqlite3
from contextlib import contextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, NullPool
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Optional, Dict
import threading
import config
from .models import Base, TickData, OHLCData


class Database:
    """Database manager for tick and OHLC data"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DATABASE_PATH
        # Use NullPool for SQLite to avoid connection issues with multiple threads
        # Add timeout for busy connections
        self.engine = create_engine(
            f'sqlite:///{self.db_path}',
            connect_args={
                'check_same_thread': False,
                'timeout': 20.0  # 20 second timeout for locked database
            },
            poolclass=NullPool,  # Use NullPool to avoid connection pooling issues
            echo=False
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._lock = threading.Lock()  # Add lock for thread safety
    
    @contextmanager
    def get_session(self):
        """Context manager for database sessions with thread safety"""
        session = None
        try:
            with self._lock:  # Acquire lock for thread-safe access
                session = self.SessionLocal()
                yield session
                try:
                    session.commit()
                except Exception as commit_error:
                    session.rollback()
                    # Retry once if database is locked
                    if 'locked' in str(commit_error).lower() or 'database is locked' in str(commit_error):
                        import time
                        time.sleep(0.1)  # Wait 100ms
                        session.commit()
                    else:
                        raise
        except Exception as e:
            if session:
                session.rollback()
            raise
        finally:
            if session:
                session.close()
    
    def insert_tick(self, symbol: str, price: float, quantity: float, timestamp: datetime = None):
        """Insert a single tick data point"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        with self.get_session() as session:
            tick = TickData(
                symbol=symbol,
                price=price,
                quantity=quantity,
                timestamp=timestamp
            )
            session.add(tick)
    
    def insert_ticks_batch(self, ticks: List[Dict]):
        """Insert multiple ticks in a batch"""
        with self.get_session() as session:
            for tick in ticks:
                tick_obj = TickData(
                    symbol=tick['symbol'],
                    price=tick['price'],
                    quantity=tick['quantity'],
                    timestamp=tick.get('timestamp', datetime.utcnow())
                )
                session.add(tick_obj)
    
    def get_ticks(self, symbol: str, start_time: datetime = None, end_time: datetime = None, 
                  limit: int = None) -> pd.DataFrame:
        """Retrieve tick data as pandas DataFrame"""
        with self.get_session() as session:
            query = session.query(TickData).filter(TickData.symbol == symbol)
            
            if start_time:
                query = query.filter(TickData.timestamp >= start_time)
            if end_time:
                query = query.filter(TickData.timestamp <= end_time)
            
            query = query.order_by(TickData.timestamp)
            
            if limit:
                query = query.limit(limit)
            
            df = pd.read_sql(query.statement, session.bind, parse_dates=['timestamp'])
            return df
    
    def get_latest_ticks(self, symbol: str, count: int = 100) -> pd.DataFrame:
        """Get the most recent ticks for a symbol"""
        with self.get_session() as session:
            query = session.query(TickData).filter(
                TickData.symbol == symbol
            ).order_by(TickData.timestamp.desc()).limit(count)
            
            df = pd.read_sql(query.statement, session.bind, parse_dates=['timestamp'])
            return df.sort_values('timestamp')
    
    def insert_ohlc(self, symbol: str, timeframe: str, open: float, high: float, 
                    low: float, close: float, volume: float, timestamp: datetime):
        """Insert OHLC data"""
        with self.get_session() as session:
            ohlc = OHLCData(
                symbol=symbol,
                timeframe=timeframe,
                open=open,
                high=high,
                low=low,
                close=close,
                volume=volume,
                timestamp=timestamp
            )
            session.add(ohlc)
    
    def get_ohlc(self, symbol: str, timeframe: str, start_time: datetime = None, 
                 end_time: datetime = None) -> pd.DataFrame:
        """Retrieve OHLC data as pandas DataFrame"""
        with self.get_session() as session:
            query = session.query(OHLCData).filter(
                OHLCData.symbol == symbol,
                OHLCData.timeframe == timeframe
            )
            
            if start_time:
                query = query.filter(OHLCData.timestamp >= start_time)
            if end_time:
                query = query.filter(OHLCData.timestamp <= end_time)
            
            query = query.order_by(OHLCData.timestamp)
            
            df = pd.read_sql(query.statement, session.bind, parse_dates=['timestamp'])
            return df
    
    def get_latest_ohlc(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Get the most recent OHLC data"""
        with self.get_session() as session:
            query = session.query(OHLCData).filter(
                OHLCData.symbol == symbol,
                OHLCData.timeframe == timeframe
            ).order_by(OHLCData.timestamp.desc()).limit(count)
            
            df = pd.read_sql(query.statement, session.bind, parse_dates=['timestamp'])
            return df.sort_values('timestamp')
    
    def get_all_symbols(self) -> List[str]:
        """Get list of all symbols in the database"""
        with self.get_session() as session:
            symbols = session.query(TickData.symbol).distinct().all()
            return [s[0] for s in symbols]
    
    def get_data_range(self, symbol: str) -> tuple:
        """Get the time range of data for a symbol"""
        with self.get_session() as session:
            min_time = session.query(TickData.timestamp).filter(
                TickData.symbol == symbol
            ).order_by(TickData.timestamp).first()
            
            max_time = session.query(TickData.timestamp).filter(
                TickData.symbol == symbol
            ).order_by(TickData.timestamp.desc()).first()
            
            if min_time and max_time:
                return (min_time[0], max_time[0])
            return (None, None)


# Global database instance
_db_instance = None

def get_db() -> Database:
    """Get or create global database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance

