import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import ccxt
from dataclasses import dataclass
import ta


@dataclass
class TradingSignal:
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0-1
    price: float
    timestamp: datetime
    reason: str


@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float


class AITradingBot:
    def __init__(self, config: Dict):
        self.config = config
        self.positions = {}
        self.balance = config.get('initial_balance', 10000)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)
        self.max_positions = config.get('max_positions', 5)
        self.logger = self._setup_logging()
        
        # Initialize exchange connections
        self.exchanges = {}
        if 'binance' in config.get('exchanges', []):
            self.exchanges['binance'] = ccxt.binance({
                'apiKey': config.get('binance_api_key'),
                'secret': config.get('binance_secret'),
                'sandbox': config.get('sandbox', True)
            })
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('AITradingBot')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('logs/trading_bot.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Fetch market data for a given symbol"""
        try:
            if symbol.endswith('USD') or symbol.endswith('USDT'):
                # Crypto data from exchange
                exchange = self.exchanges.get('binance')
                if exchange:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    return df
            else:
                # Stock data from Yahoo Finance
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=f"{limit}h", interval=timeframe)
                return df
                
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for analysis"""
        if df.empty:
            return df
            
        # Moving averages
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # MACD
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        
        # Volume indicators
        df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'])
        
        return df
    
    def generate_trading_signals(self, symbol: str) -> TradingSignal:
        """Generate trading signals using AI/ML analysis"""
        df = self.get_market_data(symbol)
        if df.empty:
            return TradingSignal(symbol, 'HOLD', 0, 0, datetime.now(), "No data available")
        
        df = self.calculate_technical_indicators(df)
        
        # Simple rule-based strategy (can be replaced with ML model)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        confidence = 0
        
        # RSI signals
        if latest['rsi'] < 30:
            signals.append('BUY')
            confidence += 0.3
        elif latest['rsi'] > 70:
            signals.append('SELL')
            confidence += 0.3
        
        # MACD signals
        if prev['macd'] < prev['macd_signal'] and latest['macd'] > latest['macd_signal']:
            signals.append('BUY')
            confidence += 0.4
        elif prev['macd'] > prev['macd_signal'] and latest['macd'] < latest['macd_signal']:
            signals.append('SELL')
            confidence += 0.4
        
        # Moving average signals
        if latest['close'] > latest['sma_20'] and prev['close'] <= prev['sma_20']:
            signals.append('BUY')
            confidence += 0.3
        elif latest['close'] < latest['sma_20'] and prev['close'] >= prev['sma_20']:
            signals.append('SELL')
            confidence += 0.3
        
        # Determine final signal
        if signals.count('BUY') > signals.count('SELL'):
            action = 'BUY'
        elif signals.count('SELL') > signals.count('BUY'):
            action = 'SELL'
        else:
            action = 'HOLD'
        
        confidence = min(confidence, 1.0)
        
        reason = f"RSI: {latest['rsi']:.2f}, MACD: {latest['macd']:.4f}, Price: {latest['close']:.2f}"
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            price=latest['close'],
            timestamp=datetime.now(),
            reason=reason
        )
    
    def calculate_position_size(self, signal: TradingSignal) -> float:
        """Calculate position size based on risk management"""
        risk_amount = self.balance * self.risk_per_trade
        
        # Simple position sizing (can be enhanced with volatility-based sizing)
        position_size = risk_amount / signal.price
        
        # Limit position size to maximum allowed
        max_position_value = self.balance * 0.2  # 20% max per position
        max_shares = max_position_value / signal.price
        
        return min(position_size, max_shares)
    
    def execute_trade(self, signal: TradingSignal) -> bool:
        """Execute trading signal"""
        if signal.action == 'HOLD':
            return True
        
        if signal.action == 'BUY':
            if len(self.positions) >= self.max_positions:
                self.logger.warning(f"Max positions reached. Skipping {signal.symbol}")
                return False
            
            position_size = self.calculate_position_size(signal)
            cost = position_size * signal.price
            
            if cost > self.balance:
                self.logger.warning(f"Insufficient balance for {signal.symbol}")
                return False
            
            # Execute buy order
            self.positions[signal.symbol] = Position(
                symbol=signal.symbol,
                quantity=position_size,
                entry_price=signal.price,
                entry_time=signal.timestamp,
                current_price=signal.price,
                unrealized_pnl=0
            )
            
            self.balance -= cost
            self.logger.info(f"BOUGHT {position_size:.4f} {signal.symbol} at {signal.price:.2f}")
            
        elif signal.action == 'SELL':
            if signal.symbol not in self.positions:
                self.logger.warning(f"No position to sell for {signal.symbol}")
                return False
            
            position = self.positions[signal.symbol]
            proceeds = position.quantity * signal.price
            
            # Execute sell order
            self.balance += proceeds
            pnl = (signal.price - position.entry_price) * position.quantity
            
            self.logger.info(f"SOLD {position.quantity:.4f} {signal.symbol} at {signal.price:.2f}, PnL: {pnl:.2f}")
            del self.positions[signal.symbol]
        
        return True
    
    def update_positions(self):
        """Update current positions with latest prices"""
        for symbol, position in self.positions.items():
            df = self.get_market_data(symbol, limit=1)
            if not df.empty:
                current_price = df['close'].iloc[-1]
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        self.update_positions()
        
        total_value = self.balance
        unrealized_pnl = 0
        
        for position in self.positions.values():
            position_value = position.quantity * position.current_price
            total_value += position_value
            unrealized_pnl += position.unrealized_pnl
        
        return {
            'cash_balance': self.balance,
            'total_value': total_value,
            'unrealized_pnl': unrealized_pnl,
            'num_positions': len(self.positions),
            'positions': list(self.positions.values())
        }
    
    def run_strategy(self, symbols: List[str], interval_minutes: int = 60):
        """Run the trading strategy continuously"""
        self.logger.info("Starting AI Trading Bot")
        
        while True:
            try:
                for symbol in symbols:
                    signal = self.generate_trading_signals(symbol)
                    
                    if signal.confidence > 0.5:  # Only execute high-confidence signals
                        self.execute_trade(signal)
                    
                    self.logger.info(f"{symbol}: {signal.action} (confidence: {signal.confidence:.2f})")
                
                portfolio = self.get_portfolio_summary()
                self.logger.info(f"Portfolio Value: ${portfolio['total_value']:.2f}")
                
                # Wait for next interval
                import time
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                self.logger.info("Trading bot stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                import time
                time.sleep(60)  # Wait 1 minute before retrying


if __name__ == "__main__":
    # Example configuration
    config = {
        'initial_balance': 10000,
        'risk_per_trade': 0.02,
        'max_positions': 5,
        'exchanges': ['binance'],
        'sandbox': True,
        'binance_api_key': 'your_api_key_here',
        'binance_secret': 'your_secret_here'
    }
    
    bot = AITradingBot(config)
    
    # Example symbols to trade
    symbols = ['BTC/USDT', 'ETH/USDT', 'AAPL', 'GOOGL']
    
    # Run the bot (commented out for demo)
    # bot.run_strategy(symbols, interval_minutes=60)
    
    # Test single signal generation
    signal = bot.generate_trading_signals('BTC/USDT')
    print(f"Signal for BTC/USDT: {signal.action} with confidence {signal.confidence:.2f}")
    print(f"Reason: {signal.reason}")
