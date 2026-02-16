import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import ta
from dataclasses import dataclass

from src.trading_bot import TradingSignal, Position
from src.kite_integration import KiteIntegration
from src.ai_model import TradingAIModel
from src.risk_management import RiskManager, RiskLimits


@dataclass
class KiteTradingConfig:
    api_key: str
    api_secret: str
    access_token: str = None
    db_path: str = "data/kite_trading.db"
    default_exchange: str = "NSE"
    default_product: str = "NRML"
    initial_balance: float = 100000
    risk_per_trade: float = 0.02
    max_positions: int = 10
    min_confidence_threshold: float = 0.6


class KiteTradingBot:
    def __init__(self, config: KiteTradingConfig):
        self.config = config
        self.logger = logging.getLogger('KiteTradingBot')
        
        # Initialize Kite integration
        self.kite = KiteIntegration(
            api_key=config.api_key,
            api_secret=config.api_secret,
            access_token=config.access_token,
            db_path=config.db_path
        )
        
        # Initialize risk manager
        risk_limits = RiskLimits(
            max_position_size=config.risk_per_trade * 2,  # Allow some flexibility
            max_sector_exposure=0.3,
            max_daily_loss=0.05,
            max_drawdown=0.2,
            min_liquidity_ratio=0.1,
            max_leverage=2.0,
            var_confidence=0.95,
            var_timeframe=1
        )
        self.risk_manager = RiskManager(risk_limits)
        
        # Initialize AI model
        self.ai_model = TradingAIModel(model_type='random_forest')
        
        # Portfolio state
        self.positions = {}
        self.holdings = {}
        self.balance = config.initial_balance
        
        # Load existing positions and holdings
        self._load_portfolio_state()
    
    def _load_portfolio_state(self):
        """Load current portfolio state from Kite"""
        try:
            # Get positions
            positions_data = self.kite.get_positions()
            for position in positions_data.get('net', []):
                if position['quantity'] != 0:  # Only non-zero positions
                    self.positions[position['tradingsymbol']] = Position(
                        symbol=position['tradingsymbol'],
                        quantity=abs(position['quantity']),
                        entry_price=position['buy_price'] if position['quantity'] > 0 else position['sell_price'],
                        entry_time=datetime.now(),  # Kite doesn't provide entry time
                        current_price=position['buy_price'] if position['quantity'] > 0 else position['sell_price'],
                        unrealized_pnl=position.get('pnl', 0)
                    )
            
            # Get holdings
            holdings_data = self.kite.get_holdings()
            for holding in holdings_data:
                self.holdings[holding['tradingsymbol']] = {
                    'quantity': holding['quantity'],
                    'average_price': holding['average_price'],
                    'last_price': holding['last_price'],
                    'pnl': holding.get('pnl', 0)
                }
            
            # Get margins
            margins = self.kite.get_margins()
            self.balance = margins.get('equity', {}).get('net', self.config.initial_balance)
            
            self.logger.info(f"Loaded portfolio: {len(self.positions)} positions, {len(self.holdings)} holdings")
            
        except Exception as e:
            self.logger.error(f"Failed to load portfolio state: {e}")
    
    def get_market_data(self, symbol: str, exchange: str = None, 
                       days: int = 100) -> pd.DataFrame:
        """Get historical market data for a symbol"""
        try:
            exchange = exchange or self.config.default_exchange
            
            # Search for instrument
            instrument = self.kite.search_instrument(symbol, exchange)
            if not instrument:
                self.logger.error(f"Instrument not found: {symbol}")
                return pd.DataFrame()
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = self.kite.get_historical_data(
                instrument_token=instrument['instrument_token'],
                from_date=start_date,
                to_date=end_date,
                interval='day'
            )
            
            if not df.empty:
                # Rename columns to match expected format
                df.rename(columns={
                    'open': 'open',
                    'high': 'high', 
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                }, inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
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
    
    def generate_trading_signals(self, symbol: str, exchange: str = None) -> TradingSignal:
        """Generate trading signals using AI and technical analysis"""
        try:
            exchange = exchange or self.config.default_exchange
            
            # Get market data
            df = self.get_market_data(symbol, exchange)
            if df.empty:
                return TradingSignal(symbol, 'HOLD', 0, 0, datetime.now(), "No data available")
            
            df = self.calculate_technical_indicators(df)
            
            # Get current quote
            quote = self.kite.get_quote(symbol, exchange)
            if not quote:
                return TradingSignal(symbol, 'HOLD', 0, 0, datetime.now(), "No quote data")
            
            current_price = quote['last_price']
            
            # Generate AI signal if model is trained
            if self.ai_model.is_trained:
                prediction, confidence = self.ai_model.predict(df)
                actions = ['SELL', 'HOLD', 'BUY']
                ai_action = actions[prediction]
                ai_confidence = confidence
            else:
                ai_action = 'HOLD'
                ai_confidence = 0
            
            # Technical analysis signals
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            technical_signals = []
            tech_confidence = 0
            
            # RSI signals
            if latest['rsi'] < 30:
                technical_signals.append('BUY')
                tech_confidence += 0.3
            elif latest['rsi'] > 70:
                technical_signals.append('SELL')
                tech_confidence += 0.3
            
            # MACD signals
            if prev['macd'] < prev['macd_signal'] and latest['macd'] > latest['macd_signal']:
                technical_signals.append('BUY')
                tech_confidence += 0.4
            elif prev['macd'] > prev['macd_signal'] and latest['macd'] < latest['macd_signal']:
                technical_signals.append('SELL')
                tech_confidence += 0.4
            
            # Moving average signals
            if latest['close'] > latest['sma_20'] and prev['close'] <= prev['sma_20']:
                technical_signals.append('BUY')
                tech_confidence += 0.3
            elif latest['close'] < latest['sma_20'] and prev['close'] >= prev['sma_20']:
                technical_signals.append('SELL')
                tech_confidence += 0.3
            
            # Determine technical signal
            if technical_signals.count('BUY') > technical_signals.count('SELL'):
                technical_action = 'BUY'
            elif technical_signals.count('SELL') > technical_signals.count('BUY'):
                technical_action = 'SELL'
            else:
                technical_action = 'HOLD'
            
            # Combine AI and technical signals
            if ai_confidence > 0.7:
                final_action = ai_action
                final_confidence = ai_confidence
            else:
                final_action = technical_action
                final_confidence = tech_confidence
            
            reason = f"RSI: {latest['rsi']:.1f}, MACD: {latest['macd']:.3f}, AI: {ai_action} ({ai_confidence:.2f})"
            
            return TradingSignal(
                symbol=symbol,
                action=final_action,
                confidence=min(final_confidence, 1.0),
                price=current_price,
                timestamp=datetime.now(),
                reason=reason
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate signal for {symbol}: {e}")
            return TradingSignal(symbol, 'HOLD', 0, 0, datetime.now(), f"Error: {e}")
    
    def calculate_position_size(self, signal: TradingSignal) -> int:
        """Calculate position size in lots/quantity"""
        try:
            # Get instrument details
            instrument = self.kite.search_instrument(signal.symbol)
            if not instrument:
                return 0
            
            lot_size = instrument.get('lot_size', 1)
            
            # Calculate risk amount
            risk_amount = self.balance * self.config.risk_per_trade
            
            # Calculate position size based on risk
            if signal.action == 'BUY':
                # For buy orders, consider stop loss
                stop_loss_price = signal.price * 0.98  # 2% stop loss
                risk_per_unit = signal.price - stop_loss_price
                
                if risk_per_unit > 0:
                    max_units = int(risk_amount / risk_per_unit)
                else:
                    max_units = int(risk_amount / signal.price)
            else:
                # For sell orders
                max_units = int(risk_amount / signal.price)
            
            # Round to nearest lot size
            if lot_size > 1:
                max_units = (max_units // lot_size) * lot_size
            
            return max(max_units, lot_size)  # Minimum 1 lot
            
        except Exception as e:
            self.logger.error(f"Failed to calculate position size: {e}")
            return 0
    
    def execute_trade(self, signal: TradingSignal, exchange: str = None) -> bool:
        """Execute trading signal"""
        try:
            exchange = exchange or self.config.default_exchange
            
            if signal.action == 'HOLD':
                return True
            
            # Check confidence threshold
            if signal.confidence < self.config.min_confidence_threshold:
                self.logger.info(f"Signal confidence too low: {signal.confidence:.2f}")
                return True
            
            # Calculate position size
            quantity = self.calculate_position_size(signal)
            if quantity <= 0:
                self.logger.warning(f"Invalid position size for {signal.symbol}")
                return False
            
            # Check risk limits
            portfolio_value = self.get_portfolio_value()
            position_value = quantity * signal.price
            
            if not self.risk_manager.check_position_limits(
                self.positions, signal.symbol, position_value, portfolio_value
            )[0]:
                self.logger.warning(f"Risk limits exceeded for {signal.symbol}")
                return False
            
            # Place order
            order_id = self.kite.place_order(
                symbol=signal.symbol,
                transaction_type=signal.action,
                quantity=quantity,
                order_type='MARKET',
                product=self.config.default_product,
                exchange=exchange
            )
            
            if order_id:
                self.logger.info(f"Order placed: {signal.action} {quantity} {signal.symbol} at {signal.price:.2f}")
                
                # Update local position tracking
                if signal.action == 'BUY':
                    self.positions[signal.symbol] = Position(
                        symbol=signal.symbol,
                        quantity=quantity,
                        entry_price=signal.price,
                        entry_time=datetime.now(),
                        current_price=signal.price,
                        unrealized_pnl=0
                    )
                elif signal.action == 'SELL' and signal.symbol in self.positions:
                    del self.positions[signal.symbol]
                
                return True
            else:
                self.logger.error(f"Failed to place order for {signal.symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to execute trade for {signal.symbol}: {e}")
            return False
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        try:
            portfolio_summary = self.kite.get_portfolio_summary()
            return portfolio_summary.get('total_current_value', self.balance)
        except Exception as e:
            self.logger.error(f"Failed to get portfolio value: {e}")
            return self.balance
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        try:
            portfolio_summary = self.kite.get_portfolio_summary()
            
            # Add additional metrics
            portfolio_summary['cash_balance'] = self.balance
            portfolio_summary['positions'] = list(self.positions.values())
            portfolio_summary['holdings'] = self.holdings
            portfolio_summary['num_positions'] = len(self.positions)
            portfolio_summary['num_holdings'] = len(self.holdings)
            
            return portfolio_summary
            
        except Exception as e:
            self.logger.error(f"Failed to get portfolio summary: {e}")
            return {
                'total_value': self.balance,
                'cash_balance': self.balance,
                'positions': [],
                'holdings': {},
                'num_positions': 0,
                'num_holdings': 0,
                'total_pnl': 0
            }
    
    def train_ai_model(self, symbols: List[str], exchange: str = None):
        """Train AI model on historical data"""
        try:
            exchange = exchange or self.config.default_exchange
            all_data = []
            
            self.logger.info(f"Training AI model on {len(symbols)} symbols...")
            
            for symbol in symbols:
                df = self.get_market_data(symbol, exchange, days=365)
                if not df.empty:
                    df = self.calculate_technical_indicators(df)
                    all_data.append(df)
            
            if not all_data:
                self.logger.error("No data available for training")
                return
            
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Train model
            training_results = self.ai_model.train(combined_df)
            self.logger.info(f"Model trained with accuracy: {training_results['accuracy']:.3f}")
            
            # Save model
            model_path = f"models/kite_ai_model_{exchange}.joblib"
            self.ai_model.save_model(model_path)
            self.logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to train AI model: {e}")
    
    def run_strategy(self, symbols: List[str], exchange: str = None, 
                   interval_minutes: int = 15):
        """Run trading strategy continuously"""
        try:
            exchange = exchange or self.config.default_exchange
            
            self.logger.info(f"Starting Kite trading bot for {len(symbols)} symbols on {exchange}")
            
            while True:
                try:
                    # Update portfolio state
                    self._load_portfolio_state()
                    
                    # Check risk limits
                    portfolio_value = self.get_portfolio_value()
                    within_limits, violations = self.risk_manager.check_risk_limits(
                        portfolio_value, self.positions
                    )
                    
                    if not within_limits:
                        self.logger.warning(f"Risk limit violations: {violations}")
                        should_stop, reason = self.risk_manager.should_stop_trading(
                            portfolio_value, self.positions
                        )
                        if should_stop:
                            self.logger.error(f"Trading stopped: {reason}")
                            break
                    
                    # Generate and execute signals
                    for symbol in symbols:
                        signal = self.generate_trading_signals(symbol, exchange)
                        
                        if signal.action != 'HOLD':
                            success = self.execute_trade(signal, exchange)
                            if success:
                                self.logger.info(f"Executed: {signal.action} {signal.symbol} at {signal.price:.2f}")
                            else:
                                self.logger.error(f"Failed to execute: {signal.action} {signal.symbol}")
                        else:
                            self.logger.debug(f"No signal for {symbol}")
                    
                    # Portfolio summary
                    portfolio = self.get_portfolio_summary()
                    self.logger.info(f"Portfolio Value: ₹{portfolio['total_value']:.2f}, P&L: ₹{portfolio.get('total_pnl', 0):.2f}")
                    
                    # Wait for next interval
                    import time
                    time.sleep(interval_minutes * 60)
                    
                except KeyboardInterrupt:
                    self.logger.info("Trading stopped by user")
                    break
                except Exception as e:
                    self.logger.error(f"Error in trading loop: {e}")
                    import time
                    time.sleep(60)  # Wait 1 minute before retrying
                    
        except Exception as e:
            self.logger.error(f"Fatal error in strategy: {e}")
        finally:
            self.kite.close()
    
    def close(self):
        """Clean up resources"""
        try:
            self.kite.close()
            self.logger.info("Kite trading bot closed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    # Example usage
    from config import EXCHANGE_CONFIG, SYMBOLS_CONFIG
    
    # Configuration
    config = KiteTradingConfig(
        api_key=EXCHANGE_CONFIG['kite']['api_key'],
        api_secret=EXCHANGE_CONFIG['kite']['api_secret'],
        access_token=EXCHANGE_CONFIG['kite']['access_token'],
        initial_balance=100000
    )
    
    # Create bot
    bot = KiteTradingBot(config)
    
    # Test with NSE symbols
    symbols = SYMBOLS_CONFIG['nse'][:5]
    
    # Generate sample signals
    for symbol in symbols:
        signal = bot.generate_trading_signals(symbol, 'NSE')
        print(f"{symbol}: {signal.action} (confidence: {signal.confidence:.2f})")
        print(f"Reason: {signal.reason}")
    
    # Get portfolio summary
    portfolio = bot.get_portfolio_summary()
    print(f"\nPortfolio Summary:")
    print(f"Total Value: ₹{portfolio.get('total_value', 0):.2f}")
    print(f"Positions: {portfolio.get('num_positions', 0)}")
    print(f"Holdings: {portfolio.get('num_holdings', 0)}")
    
    bot.close()
