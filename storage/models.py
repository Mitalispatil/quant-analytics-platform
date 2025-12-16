"""
Database models for tick and OHLC data
"""

from sqlalchemy import Column, Float, String, Integer, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class TickData(Base):
    """Model for tick-level data"""
    __tablename__ = 'tick_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<TickData(symbol={self.symbol}, price={self.price}, timestamp={self.timestamp})>"


class OHLCData(Base):
    """Model for OHLC (Open, High, Low, Close) data"""
    __tablename__ = 'ohlc_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)  # '1s', '1m', '5m'
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    __table_args__ = (
        Index('idx_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<OHLCData(symbol={self.symbol}, timeframe={self.timeframe}, close={self.close}, timestamp={self.timestamp})>"

