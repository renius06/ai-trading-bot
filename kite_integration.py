import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import kiteconnect
from kiteconnect import KiteConnect, KiteTicker
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import threading
import time
import json
import os

Base = declarative_base()


class KiteTrade(Base):
    __tablename__ = 'kite_trades'
    
    id = Column(Integer, primary_key=True)
    order_id = Column(String(50), unique=True)
    symbol = Column(String(50))
    exchange = Column(String(20))
    transaction_type = Column(String(10))  # BUY/SELL
    quantity = Column(Integer)
    price = Column(Float)
    trigger_price = Column(Float)
    average_price = Column(Float)
    status = Column(String(20))
    order_timestamp = Column(DateTime)
    exchange_timestamp = Column(DateTime)
    created_at = Column(DateTime, default=datetime.now)


class KitePosition(Base):
    __tablename__ = 'kite_positions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50))
    exchange = Column(String(20))
    product_type = Column(String(20))  # NRML/MIS/CNC
    quantity = Column(Integer)
    overnight_quantity = Column(Integer)
    multiplier = Column(Float)
    price = Column(Float)
    buy_price = Column(Float)
    sell_price = Column(Float)
    pnl = Column(Float)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class KiteHolding(Base):
    __tablename__ = 'kite_holdings'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50))
    exchange = Column(String(20))
    isin = Column(String(20))
    quantity = Column(Integer)
    average_price = Column(Float)
    last_price = Column(Float)
    pnl = Column(Float)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class KiteMarketData(Base):
    __tablename__ = 'kite_market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50))
    exchange = Column(String(20))
    timestamp = Column(DateTime)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Integer)
    oi = Column(Integer)  # Open Interest for derivatives


class KiteIntegration:
    def __init__(self, api_key: str, api_secret: str, access_token: str = None, 
                 db_path: str = "data/kite_trading.db"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.db_path = db_path
        
        # Initialize Kite Connect
        self.kite = KiteConnect(api_key=api_key)
        if access_token:
            self.kite.set_access_token(access_token)
        
        # Initialize database
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Logger
        self.logger = logging.getLogger('KiteIntegration')
        
        # WebSocket for live data
        self.kws = None
        self.tickers = {}
        
        # Cache for instruments
        self.instruments_cache = None
        self.last_instruments_update = None
    
    def generate_session(self, request_token: str) -> str:
        """Generate access token from request token"""
        try:
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            
            # Save access token for future use
            self.save_access_token(self.access_token)
            
            self.logger.info("Successfully generated access token")
            return self.access_token
        
        except Exception as e:
            self.logger.error(f"Failed to generate session: {e}")
            raise
    
    def save_access_token(self, token: str):
        """Save access token to file"""
        try:
            with open('data/kite_access_token.txt', 'w') as f:
                f.write(token)
        except Exception as e:
            self.logger.error(f"Failed to save access token: {e}")
    
    def load_access_token(self) -> Optional[str]:
        """Load access token from file"""
        try:
            if os.path.exists('data/kite_access_token.txt'):
                with open('data/kite_access_token.txt', 'r') as f:
                    return f.read().strip()
        except Exception as e:
            self.logger.error(f"Failed to load access token: {e}")
        return None
    
    def get_instruments(self, exchange: str = None) -> List[Dict]:
        """Get instruments list with caching"""
        now = datetime.now()
        
        # Check cache validity (refresh every 24 hours)
        if (self.instruments_cache and self.last_instruments_update and 
            (now - self.last_instruments_update).hours < 24):
            instruments = self.instruments_cache
        else:
            try:
                instruments = self.kite.instruments(exchange)
                self.instruments_cache = instruments
                self.last_instruments_update = now
                self.logger.info(f"Updated instruments cache: {len(instruments)} instruments")
            except Exception as e:
                self.logger.error(f"Failed to fetch instruments: {e}")
                return []
        
        # Filter by exchange if specified
        if exchange:
            instruments = [inst for inst in instruments if inst['exchange'] == exchange]
        
        return instruments
    
    def search_instrument(self, symbol: str, exchange: str = 'NSE') -> Optional[Dict]:
        """Search for instrument by symbol"""
        instruments = self.get_instruments(exchange)
        
        for instrument in instruments:
            if instrument['tradingsymbol'] == symbol:
                return instrument
        
        return None
    
    def get_historical_data(self, instrument_token: int, from_date: datetime, 
                          to_date: datetime, interval: str = 'day') -> pd.DataFrame:
        """Get historical data for an instrument"""
        try:
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            return df
        
        except Exception as e:
            self.logger.error(f"Failed to fetch historical data: {e}")
            return pd.DataFrame()
    
    def get_quote(self, symbol: str, exchange: str = 'NSE') -> Optional[Dict]:
        """Get current quote for a symbol"""
        try:
            instrument = self.search_instrument(symbol, exchange)
            if not instrument:
                return None
            
            quote = self.kite.quote([instrument['instrument_token']])
            return quote.get(str(instrument['instrument_token']))
        
        except Exception as e:
            self.logger.error(f"Failed to fetch quote for {symbol}: {e}")
            return None
    
    def get_ohlc(self, symbols: List[str]) -> Dict:
        """Get OHLC data for multiple symbols"""
        try:
            # Get instrument tokens
            tokens = []
            for symbol in symbols:
                instrument = self.search_instrument(symbol)
                if instrument:
                    tokens.append(instrument['instrument_token'])
            
            if not tokens:
                return {}
            
            ohlc_data = self.kite.ohlc(tokens)
            return ohlc_data
        
        except Exception as e:
            self.logger.error(f"Failed to fetch OHLC data: {e}")
            return {}
    
    def place_order(self, symbol: str, transaction_type: str, quantity: int, 
                    order_type: str = 'MARKET', price: float = None, 
                    trigger_price: float = None, product: str = 'NRML',
                    exchange: str = 'NSE') -> Optional[str]:
        """Place an order"""
        try:
            instrument = self.search_instrument(symbol, exchange)
            if not instrument:
                self.logger.error(f"Instrument not found: {symbol}")
                return None
            
            order_params = {
                'exchange': exchange,
                'tradingsymbol': symbol,
                'transaction_type': transaction_type,
                'quantity': quantity,
                'order_type': order_type,
                'product': product,
                'variety': 'regular'
            }
            
            if order_type == 'LIMIT' and price:
                order_params['price'] = price
            
            if order_type == 'SL' and trigger_price:
                order_params['trigger_price'] = trigger_price
            
            order_id = self.kite.place_order(**order_params)
            
            # Log order to database
            self.log_order(order_id, order_params)
            
            self.logger.info(f"Order placed successfully: {order_id}")
            return order_id
        
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            return None
    
    def log_order(self, order_id: str, order_params: Dict):
        """Log order to database"""
        try:
            trade = KiteTrade(
                order_id=order_id,
                symbol=order_params['tradingsymbol'],
                exchange=order_params['exchange'],
                transaction_type=order_params['transaction_type'],
                quantity=order_params['quantity'],
                price=order_params.get('price'),
                trigger_price=order_params.get('trigger_price'),
                status='PLACED',
                order_timestamp=datetime.now()
            )
            
            self.session.add(trade)
            self.session.commit()
        
        except Exception as e:
            self.logger.error(f"Failed to log order: {e}")
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status"""
        try:
            order_status = self.kite.order_status(order_id)
            
            # Update database record
            trade = self.session.query(KiteTrade).filter_by(order_id=order_id).first()
            if trade:
                trade.status = order_status['status']
                trade.average_price = order_status.get('average_price')
                trade.exchange_timestamp = pd.to_datetime(order_status.get('exchange_timestamp'))
                self.session.commit()
            
            return order_status
        
        except Exception as e:
            self.logger.error(f"Failed to get order status: {e}")
            return None
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            positions = self.kite.positions()
            
            # Update database
            self.update_positions_in_db(positions)
            
            return positions
        
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return []
    
    def update_positions_in_db(self, positions: Dict):
        """Update positions in database"""
        try:
            # Clear existing positions
            self.session.query(KitePosition).delete()
            
            # Add current positions
            for pos_type in ['day', 'net']:
                for position in positions.get(pos_type, []):
                    db_position = KitePosition(
                        symbol=position['tradingsymbol'],
                        exchange=position['exchange'],
                        product_type=position['product'],
                        quantity=position['quantity'],
                        overnight_quantity=position['overnight_quantity'],
                        multiplier=position['multiplier'],
                        price=position['buy_price'] if position['quantity'] > 0 else position['sell_price'],
                        buy_price=position['buy_price'],
                        sell_price=position['sell_price'],
                        pnl=position['pnl']
                    )
                    self.session.add(db_position)
            
            self.session.commit()
        
        except Exception as e:
            self.logger.error(f"Failed to update positions in DB: {e}")
    
    def get_holdings(self) -> List[Dict]:
        """Get current holdings"""
        try:
            holdings = self.kite.holdings()
            
            # Update database
            self.update_holdings_in_db(holdings)
            
            return holdings
        
        except Exception as e:
            self.logger.error(f"Failed to get holdings: {e}")
            return []
    
    def update_holdings_in_db(self, holdings: List[Dict]):
        """Update holdings in database"""
        try:
            # Clear existing holdings
            self.session.query(KiteHolding).delete()
            
            # Add current holdings
            for holding in holdings:
                db_holding = KiteHolding(
                    symbol=holding['tradingsymbol'],
                    exchange=holding['exchange'],
                    isin=holding['isin'],
                    quantity=holding['quantity'],
                    average_price=holding['average_price'],
                    last_price=holding['last_price'],
                    pnl=holding['pnl']
                )
                self.session.add(db_holding)
            
            self.session.commit()
        
        except Exception as e:
            self.logger.error(f"Failed to update holdings in DB: {e}")
    
    def start_websocket(self, symbols: List[str], on_ticks_callback):
        """Start WebSocket for live data"""
        try:
            # Get instrument tokens
            tokens = []
            for symbol in symbols:
                instrument = self.search_instrument(symbol)
                if instrument:
                    tokens.append(instrument['instrument_token'])
            
            if not tokens:
                self.logger.error("No valid instruments found for WebSocket")
                return
            
            # Initialize KiteTicker
            self.kws = KiteTicker(self.api_key, self.access_token)
            
            # Assign callbacks
            self.kws.on_ticks = on_ticks_callback
            self.kws.on_connect = self._on_connect
            self.kws.on_close = self._on_close
            self.kws.on_error = self._on_error
            
            # Connect in a separate thread
            def connect_websocket():
                self.kws.connect(tokens)
            
            ws_thread = threading.Thread(target=connect_websocket, daemon=True)
            ws_thread.start()
            
            self.logger.info(f"WebSocket started for {len(tokens)} symbols")
        
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket: {e}")
    
    def _on_connect(self, ws, response):
        """WebSocket connect callback"""
        self.logger.info("WebSocket connected")
    
    def _on_close(self, ws, code, reason):
        """WebSocket close callback"""
        self.logger.warning(f"WebSocket closed: {code} - {reason}")
    
    def _on_error(self, ws, code, reason):
        """WebSocket error callback"""
        self.logger.error(f"WebSocket error: {code} - {reason}")
    
    def store_market_data(self, symbol: str, data: Dict):
        """Store market data in database"""
        try:
            instrument = self.search_instrument(symbol)
            if not instrument:
                return
            
            market_data = KiteMarketData(
                symbol=symbol,
                exchange=instrument['exchange'],
                timestamp=datetime.now(),
                open_price=data.get('open'),
                high_price=data.get('high'),
                low_price=data.get('low'),
                close_price=data.get('close'),
                volume=data.get('volume'),
                oi=data.get('oi')
            )
            
            self.session.add(market_data)
            self.session.commit()
        
        except Exception as e:
            self.logger.error(f"Failed to store market data: {e}")
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        try:
            # Get current positions and holdings
            positions = self.get_positions()
            holdings = self.get_holdings()
            
            # Calculate totals
            total_pnl = 0
            total_investment = 0
            total_current_value = 0
            
            # Process positions
            for position in positions.get('net', []):
                total_pnl += position.get('pnl', 0)
            
            # Process holdings
            for holding in holdings:
                investment = holding['quantity'] * holding['average_price']
                current_value = holding['quantity'] * holding['last_price']
                
                total_investment += investment
                total_current_value += current_value
                total_pnl += holding.get('pnl', 0)
            
            return {
                'total_pnl': total_pnl,
                'total_investment': total_investment,
                'total_current_value': total_current_value,
                'positions_count': len(positions.get('net', [])),
                'holdings_count': len(holdings),
                'margin_available': self.get_margins().get('equity', {}).get('net', 0)
            }
        
        except Exception as e:
            self.logger.error(f"Failed to get portfolio summary: {e}")
            return {}
    
    def get_margins(self) -> Dict:
        """Get margin details"""
        try:
            return self.kite.margins()
        except Exception as e:
            self.logger.error(f"Failed to get margins: {e}")
            return {}
    
    def close(self):
        """Clean up resources"""
        try:
            if self.kws:
                self.kws.close()
            
            self.session.close()
            self.logger.info("Kite integration closed successfully")
        
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# Utility functions for Indian market specifics
def get_nse_symbols() -> List[str]:
    """Get popular NSE symbols for trading"""
    return [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK', 'KOTAKBANK',
        'SBIN', 'BHARTIARTL', 'AXISBANK', 'BAJFINANCE', 'DMART', 'ASIANPAINT',
        'MARUTI', 'HCLTECH', 'SUNPHARMA', 'M&M', 'TITAN', 'ULTRACEMCO'
    ]


def get_bse_symbols() -> List[str]:
    """Get popular BSE symbols for trading"""
    return [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK', 'KOTAKBANK',
        'SBIN', 'BHARTIARTL', 'AXISBANK', 'BAJFINANCE', 'DMART', 'ASIANPAINT'
    ]


def get_mcx_symbols() -> List[str]:
    """Get popular MCX symbols for commodities"""
    return [
        'CRUDEOIL', 'NATURALGAS', 'GOLD', 'SILVER', 'COPPER', 'ZINC', 'LEAD',
        'ALUMINIUM', 'NICKEL', 'COTTON', 'MENTHAOIL'
    ]


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize with your credentials
    api_key = "your_api_key_here"
    api_secret = "your_api_secret_here"
    
    kite = KiteIntegration(api_key, api_secret)
    
    # Example: Get instruments
    instruments = kite.get_instruments('NSE')
    print(f"Found {len(instruments)} NSE instruments")
    
    # Example: Get quote for RELIANCE
    quote = kite.get_quote('RELIANCE', 'NSE')
    if quote:
        print(f"RELIANCE price: {quote['last_price']}")
    
    # Example: Get portfolio summary
    portfolio = kite.get_portfolio_summary()
    print(f"Portfolio P&L: {portfolio.get('total_pnl', 0)}")
