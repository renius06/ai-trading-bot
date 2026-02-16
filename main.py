#!/usr/bin/env python3

import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.trading_bot import AITradingBot, TradingSignal
from src.ai_model import TradingAIModel, EnsembleModel
from src.risk_management import RiskManager, RiskLimits
from src.kite_trading_bot import KiteTradingBot, KiteTradingConfig
from config import (
    TRADING_CONFIG, 
    EXCHANGE_CONFIG, 
    RISK_CONFIG, 
    AI_CONFIG, 
    SYMBOLS_CONFIG,
    LOGGING_CONFIG
)


def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOGGING_CONFIG['file_path']),
            logging.StreamHandler() if LOGGING_CONFIG['console_output'] else logging.NullHandler()
        ]
    )


def create_trading_bot() -> AITradingBot:
    """Create and configure the trading bot"""
    # Initialize risk manager
    risk_limits = RiskLimits(
        max_position_size=RISK_CONFIG['max_position_size'],
        max_sector_exposure=RISK_CONFIG['max_sector_exposure'],
        max_daily_loss=RISK_CONFIG['max_daily_loss'],
        max_drawdown=RISK_CONFIG['max_drawdown'],
        min_liquidity_ratio=RISK_CONFIG['min_liquidity_ratio'],
        max_leverage=RISK_CONFIG['max_leverage'],
        var_confidence=RISK_CONFIG['var_confidence'],
        var_timeframe=RISK_CONFIG['var_timeframe']
    )
    
    risk_manager = RiskManager(risk_limits)
    
    # Create trading bot configuration
    bot_config = {
        'initial_balance': TRADING_CONFIG['initial_balance'],
        'risk_per_trade': TRADING_CONFIG['risk_per_trade'],
        'max_positions': TRADING_CONFIG['max_positions'],
        'exchanges': [name for name, config in EXCHANGE_CONFIG.items() if config['enable']],
        'sandbox': EXCHANGE_CONFIG['binance']['sandbox'],
        'binance_api_key': EXCHANGE_CONFIG['binance']['api_key'],
        'binance_secret': EXCHANGE_CONFIG['binance']['secret'],
        'risk_manager': risk_manager
    }
    
    return AITradingBot(bot_config)


def create_ai_model() -> TradingAIModel:
    """Create and configure the AI model"""
    model = TradingAIModel(model_type=AI_CONFIG['model_type'])
    
    # Try to load existing model
    model_path = os.path.join(AI_CONFIG['model_save_path'], 'trading_model.joblib')
    if os.path.exists(model_path):
        try:
            model.load_model(model_path)
            print(f"Loaded existing AI model from {model_path}")
        except Exception as e:
            print(f"Failed to load existing model: {e}")
            print("Will train new model when sufficient data is available")
    
    return model


def run_backtest(bot: AITradingBot, symbols: list, start_date: str, end_date: str):
    """Run backtesting on historical data"""
    print(f"Running backtest from {start_date} to {end_date}")
    print(f"Symbols: {symbols}")
    
    # This is a simplified backtest - in production, you'd want more sophisticated backtesting
    total_return = 0
    trades = []
    
    for symbol in symbols:
        print(f"Backtesting {symbol}...")
        
        # Get historical data
        df = bot.get_market_data(symbol, limit=1000)
        if df.empty:
            continue
        
        df = bot.calculate_technical_indicators(df)
        
        # Simulate trading
        for i in range(50, len(df)):  # Start after indicators are calculated
            historical_df = df.iloc[:i+1]
            signal = bot.generate_trading_signals(symbol)
            
            if signal.action in ['BUY', 'SELL']:
                trades.append({
                    'symbol': symbol,
                    'action': signal.action,
                    'price': signal.price,
                    'timestamp': df.index[i],
                    'confidence': signal.confidence
                })
    
    print(f"Backtest completed. Total trades: {len(trades)}")
    return trades


def run_live_trading(bot: AITradingBot, ai_model: TradingAIModel, symbols: list):
    """Run live trading"""
    print("Starting live trading...")
    print(f"Trading symbols: {symbols}")
    print(f"Initial balance: ${bot.balance:.2f}")
    
    try:
        while True:
            print(f"\n{'='*50}")
            print(f"Trading cycle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Update portfolio
            portfolio = bot.get_portfolio_summary()
            print(f"Portfolio value: ${portfolio['total_value']:.2f}")
            print(f"Cash balance: ${portfolio['cash_balance']:.2f}")
            print(f"Open positions: {portfolio['num_positions']}")
            
            # Check risk limits
            within_limits, violations = bot.risk_manager.check_risk_limits(
                portfolio['total_value'], 
                {pos.symbol: {'quantity': pos.quantity, 'current_price': pos.current_price} 
                 for pos in portfolio['positions']}
            )
            
            if not within_limits:
                print(f"⚠️  Risk limit violations: {violations}")
                should_stop, reason = bot.risk_manager.should_stop_trading(
                    portfolio['total_value'], 
                    {pos.symbol: {'quantity': pos.quantity, 'current_price': pos.current_price} 
                     for pos in portfolio['positions']}
                )
                if should_stop:
                    print(f"🛑 Trading stopped: {reason}")
                    break
            
            # Analyze each symbol
            for symbol in symbols:
                try:
                    # Get market data
                    df = bot.get_market_data(symbol)
                    if df.empty:
                        continue
                    
                    df = bot.calculate_technical_indicators(df)
                    
                    # Generate AI signal if model is trained
                    if ai_model.is_trained:
                        prediction, confidence = ai_model.predict(df)
                        actions = ['SELL', 'HOLD', 'BUY']
                        ai_action = actions[prediction]
                        
                        # Combine AI signal with technical analysis
                        technical_signal = bot.generate_trading_signals(symbol)
                        
                        # Simple ensemble: trust AI more if confidence is high
                        if confidence > 0.7:
                            final_action = ai_action
                            final_confidence = confidence
                        else:
                            final_action = technical_signal.action
                            final_confidence = technical_signal.confidence
                    else:
                        # Use only technical signals if model not trained
                        signal = bot.generate_trading_signals(symbol)
                        final_action = signal.action
                        final_confidence = signal.confidence
                    
                    # Execute trade if confidence is high enough
                    if final_confidence > TRADING_CONFIG['min_confidence_threshold']:
                        enhanced_signal = TradingSignal(
                            symbol=symbol,
                            action=final_action,
                            confidence=final_confidence,
                            price=df['close'].iloc[-1],
                            timestamp=datetime.now(),
                            reason=f"AI+Technical Analysis (conf: {final_confidence:.2f})"
                        )
                        
                        success = bot.execute_trade(enhanced_signal)
                        if success:
                            print(f"✅ {symbol}: {final_action} {df['close'].iloc[-1]:.2f} (conf: {final_confidence:.2f})")
                        else:
                            print(f"❌ {symbol}: Failed to execute {final_action}")
                    else:
                        print(f"⏸️  {symbol}: HOLD (low confidence: {final_confidence:.2f})")
                
                except Exception as e:
                    print(f"❌ Error analyzing {symbol}: {e}")
            
            # Wait for next interval
            import time
            print(f"Waiting {TRADING_CONFIG['trading_interval_minutes']} minutes...")
            time.sleep(TRADING_CONFIG['trading_interval_minutes'] * 60)
    
    except KeyboardInterrupt:
        print("\n🛑 Trading stopped by user")
    
    # Final portfolio summary
    final_portfolio = bot.get_portfolio_summary()
    print(f"\n{'='*50}")
    print("FINAL PORTFOLIO SUMMARY")
    print(f"Total value: ${final_portfolio['total_value']:.2f}")
    print(f"Cash balance: ${final_portfolio['cash_balance']:.2f}")
    print(f"Unrealized P&L: ${final_portfolio['unrealized_pnl']:.2f}")
    print(f"Total return: {((final_portfolio['total_value'] - TRADING_CONFIG['initial_balance']) / TRADING_CONFIG['initial_balance'] * 100):.2f}%")


def train_ai_model(bot: AITradingBot, ai_model: TradingAIModel, symbols: list):
    """Train the AI model on historical data"""
    print("Training AI model...")
    
    all_data = []
    
    for symbol in symbols:
        print(f"Fetching training data for {symbol}...")
        df = bot.get_market_data(symbol, limit=2000)  # Get more data for training
        if not df.empty:
            df = bot.calculate_technical_indicators(df)
            all_data.append(df)
    
    if not all_data:
        print("No data available for training")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    try:
        training_results = ai_model.train(combined_df)
        print(f"✅ Model trained successfully!")
        print(f"Accuracy: {training_results['accuracy']:.3f}")
        
        # Save model
        os.makedirs(AI_CONFIG['model_save_path'], exist_ok=True)
        model_path = os.path.join(AI_CONFIG['model_save_path'], 'trading_model.joblib')
        ai_model.save_model(model_path)
        print(f"✅ Model saved to {model_path}")
        
    except Exception as e:
        print(f"❌ Failed to train model: {e}")


def create_kite_trading_bot() -> KiteTradingBot:
    """Create and configure the Kite trading bot"""
    if not EXCHANGE_CONFIG['kite']['enable']:
        raise ValueError("Kite API is not enabled in configuration")
    
    kite_config = KiteTradingConfig(
        api_key=EXCHANGE_CONFIG['kite']['api_key'],
        api_secret=EXCHANGE_CONFIG['kite']['api_secret'],
        access_token=EXCHANGE_CONFIG['kite']['access_token'],
        db_path=EXCHANGE_CONFIG['kite']['db_path'],
        default_exchange=EXCHANGE_CONFIG['kite']['default_exchange'],
        default_product=EXCHANGE_CONFIG['kite']['default_product'],
        initial_balance=TRADING_CONFIG['initial_balance'],
        risk_per_trade=TRADING_CONFIG['risk_per_trade'],
        max_positions=TRADING_CONFIG['max_positions'],
        min_confidence_threshold=TRADING_CONFIG['min_confidence_threshold']
    )
    
    return KiteTradingBot(kite_config)


def run_kite_strategy(bot: KiteTradingBot, symbols: list, exchange: str = None):
    """Run Kite trading strategy"""
    print(f"Starting Kite trading strategy for {len(symbols)} symbols")
    
    try:
        bot.run_strategy(symbols, exchange, TRADING_CONFIG['trading_interval_minutes'])
    except KeyboardInterrupt:
        print("\n🛑 Kite trading stopped by user")
    except Exception as e:
        print(f"❌ Error in Kite trading: {e}")
    finally:
        bot.close()


def train_kite_ai_model(bot: KiteTradingBot, symbols: list, exchange: str = None):
    """Train AI model for Kite trading"""
    print("Training AI model for Kite trading...")
    bot.train_ai_model(symbols, exchange)


def main():
    parser = argparse.ArgumentParser(description='AI Trading Bot')
    parser.add_argument('--mode', choices=['live', 'backtest', 'train', 'demo', 'kite-live', 'kite-train'], 
                       default='demo', help='Running mode')
    parser.add_argument('--symbols', nargs='+', 
                       default=SYMBOLS_CONFIG['crypto'][:3], 
                       help='Trading symbols')
    parser.add_argument('--exchange', default=None, 
                       help='Exchange (NSE, BSE, MCX for Kite)')
    parser.add_argument('--backtest-start', default='2023-01-01', 
                       help='Backtest start date')
    parser.add_argument('--backtest-end', default='2024-01-01', 
                       help='Backtest end date')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    print("🤖 AI Trading Bot Starting...")
    print(f"Mode: {args.mode}")
    print(f"Symbols: {args.symbols}")
    
    # Check if this is a Kite mode
    if args.mode.startswith('kite'):
        if not EXCHANGE_CONFIG['kite']['enable']:
            print("❌ Kite API is not enabled. Please set KITE_ENABLE=true in your .env file")
            return
        
        # Default to NSE symbols if no symbols provided
        if args.symbols == SYMBOLS_CONFIG['crypto'][:3]:
            args.symbols = SYMBOLS_CONFIG['nse'][:5]
        
        exchange = args.exchange or EXCHANGE_CONFIG['kite']['default_exchange']
        
        try:
            kite_bot = create_kite_trading_bot()
            
            if args.mode == 'kite-train':
                train_kite_ai_model(kite_bot, args.symbols, exchange)
            elif args.mode == 'kite-live':
                # Train model if not trained
                if not kite_bot.ai_model.is_trained:
                    print("AI model not trained. Training first...")
                    train_kite_ai_model(kite_bot, args.symbols, exchange)
                
                run_kite_strategy(kite_bot, args.symbols, exchange)
        
        except Exception as e:
            print(f"❌ Failed to initialize Kite trading bot: {e}")
            return
    
    else:
        # Original trading bot modes
        # Create trading bot
        bot = create_trading_bot()
        
        # Create AI model
        ai_model = create_ai_model()
        
        if args.mode == 'train':
            train_ai_model(bot, ai_model, args.symbols)
        
        elif args.mode == 'backtest':
            trades = run_backtest(bot, args.symbols, args.backtest_start, args.backtest_end)
            print(f"Backtest completed with {len(trades)} trades")
        
        elif args.mode == 'live':
            # Train model if not trained
            if not ai_model.is_trained:
                print("AI model not trained. Training first...")
                train_ai_model(bot, ai_model, args.symbols)
            
            run_live_trading(bot, ai_model, args.symbols)
        
        elif args.mode == 'demo':
            print("🎯 Demo Mode - Showing sample signals...")
            
            for symbol in args.symbols[:3]:  # Limit to 3 symbols for demo
                try:
                    print(f"\nAnalyzing {symbol}...")
                    
                    # Get market data
                    df = bot.get_market_data(symbol)
                    if df.empty:
                        print(f"No data available for {symbol}")
                        continue
                    
                    df = bot.calculate_technical_indicators(df)
                    
                    # Generate signals
                    technical_signal = bot.generate_trading_signals(symbol)
                    print(f"Technical Analysis: {technical_signal.action} (conf: {technical_signal.confidence:.2f})")
                    print(f"Reason: {technical_signal.reason}")
                    
                    if ai_model.is_trained:
                        prediction, confidence = ai_model.predict(df)
                        actions = ['SELL', 'HOLD', 'BUY']
                        print(f"AI Prediction: {actions[prediction]} (conf: {confidence:.2f})")
                    else:
                        print("AI Model: Not trained yet")
                    
                    print(f"Current Price: ${df['close'].iloc[-1]:.2f}")
                    
                except Exception as e:
                    print(f"Error analyzing {symbol}: {e}")
            
            print(f"\n💰 Current Portfolio: ${bot.get_portfolio_summary()['total_value']:.2f}")


if __name__ == "__main__":
    import pandas as pd  # Import here to avoid circular imports
    main()
