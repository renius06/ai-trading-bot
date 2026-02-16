import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Trading Configuration
TRADING_CONFIG = {
    'initial_balance': float(os.getenv('INITIAL_BALANCE', 10000)),
    'risk_per_trade': float(os.getenv('RISK_PER_TRADE', 0.02)),
    'max_positions': int(os.getenv('MAX_POSITIONS', 5)),
    'min_confidence_threshold': float(os.getenv('MIN_CONFIDENCE_THRESHOLD', 0.6)),
    'trading_interval_minutes': int(os.getenv('TRADING_INTERVAL_MINUTES', 60)),
    'lookback_period': int(os.getenv('LOOKBACK_PERIOD', 100)),
}

# Exchange Configuration
EXCHANGE_CONFIG = {
    'binance': {
        'api_key': os.getenv('BINANCE_API_KEY', ''),
        'secret': os.getenv('BINANCE_SECRET', ''),
        'sandbox': os.getenv('BINANCE_SANDBOX', 'true').lower() == 'true',
        'enable': os.getenv('BINANCE_ENABLE', 'false').lower() == 'true',
    },
    'coinbase': {
        'api_key': os.getenv('COINBASE_API_KEY', ''),
        'secret': os.getenv('COINBASE_SECRET', ''),
        'sandbox': os.getenv('COINBASE_SANDBOX', 'true').lower() == 'true',
        'enable': os.getenv('COINBASE_ENABLE', 'false').lower() == 'true',
    },
    'kite': {
        'api_key': os.getenv('KITE_API_KEY', ''),
        'api_secret': os.getenv('KITE_API_SECRET', ''),
        'access_token': os.getenv('KITE_ACCESS_TOKEN', ''),
        'request_token': os.getenv('KITE_REQUEST_TOKEN', ''),
        'enable': os.getenv('KITE_ENABLE', 'false').lower() == 'true',
        'db_path': os.getenv('KITE_DB_PATH', 'data/kite_trading.db'),
        'default_exchange': os.getenv('KITE_DEFAULT_EXCHANGE', 'NSE'),
        'default_product': os.getenv('KITE_DEFAULT_PRODUCT', 'NRML'),
    }
}

# Risk Management Configuration
RISK_CONFIG = {
    'max_position_size': float(os.getenv('MAX_POSITION_SIZE', 0.1)),  # 10% of portfolio
    'max_sector_exposure': float(os.getenv('MAX_SECTOR_EXPOSURE', 0.3)),  # 30% per sector
    'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', 0.05)),  # 5% daily loss
    'max_drawdown': float(os.getenv('MAX_DRAWDOWN', 0.2)),  # 20% max drawdown
    'min_liquidity_ratio': float(os.getenv('MIN_LIQUIDITY_RATIO', 0.1)),  # 10% cash minimum
    'max_leverage': float(os.getenv('MAX_LEVERAGE', 2.0)),  # 2x leverage max
    'var_confidence': float(os.getenv('VAR_CONFIDENCE', 0.95)),  # 95% VaR
    'var_timeframe': int(os.getenv('VAR_TIMEFRAME', 1)),  # 1-day VaR
    'stop_loss_pct': float(os.getenv('STOP_LOSS_PCT', 0.02)),  # 2% stop loss
    'take_profit_pct': float(os.getenv('TAKE_PROFIT_PCT', 0.05)),  # 5% take profit
}

# AI Model Configuration
AI_CONFIG = {
    'model_type': os.getenv('AI_MODEL_TYPE', 'random_forest'),  # 'random_forest', 'lstm', 'ensemble'
    'retrain_interval_hours': int(os.getenv('RETRAIN_INTERVAL_HOURS', 24)),
    'min_training_samples': int(os.getenv('MIN_TRAINING_SAMPLES', 1000)),
    'feature_window': int(os.getenv('FEATURE_WINDOW', 20)),
    'prediction_horizon': int(os.getenv('PREDICTION_HORIZON', 5)),
    'ensemble_models': ['random_forest', 'lstm'],
    'model_save_path': 'models/',
    'model_save_interval_hours': int(os.getenv('MODEL_SAVE_INTERVAL_HOURS', 6)),
}

# Data Configuration
DATA_CONFIG = {
    'default_timeframe': os.getenv('DEFAULT_TIMEFRAME', '1h'),
    'data_sources': ['yahoo_finance', 'binance'],
    'cache_enabled': os.getenv('DATA_CACHE_ENABLED', 'true').lower() == 'true',
    'cache_duration_hours': int(os.getenv('CACHE_DURATION_HOURS', 1)),
    'data_retention_days': int(os.getenv('DATA_RETENTION_DAYS', 365)),
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'file_path': 'logs/trading_bot.log',
    'max_file_size_mb': int(os.getenv('LOG_MAX_FILE_SIZE_MB', 10)),
    'backup_count': int(os.getenv('LOG_BACKUP_COUNT', 5)),
    'console_output': os.getenv('LOG_CONSOLE_OUTPUT', 'true').lower() == 'true',
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    'start_date': os.getenv('BACKTEST_START_DATE', '2023-01-01'),
    'end_date': os.getenv('BACKTEST_END_DATE', '2024-01-01'),
    'initial_capital': float(os.getenv('BACKTEST_INITIAL_CAPITAL', 10000)),
    'commission': float(os.getenv('BACKTEST_COMMISSION', 0.001)),  # 0.1% commission
    'slippage': float(os.getenv('BACKTEST_SLIPPAGE', 0.0005)),  # 0.05% slippage
}

# Trading Symbols Configuration
SYMBOLS_CONFIG = {
    'crypto': [
        'BTC/USDT',
        'ETH/USDT',
        'BNB/USDT',
        'ADA/USDT',
        'SOL/USDT',
        'XRP/USDT',
        'DOT/USDT',
        'LINK/USDT',
    ],
    'stocks': [
        'AAPL',
        'GOOGL',
        'MSFT',
        'AMZN',
        'TSLA',
        'META',
        'NVDA',
        'AMD',
    ],
    'nse': [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK', 'KOTAKBANK',
        'SBIN', 'BHARTIARTL', 'AXISBANK', 'BAJFINANCE', 'DMART', 'ASIANPAINT',
        'MARUTI', 'HCLTECH', 'SUNPHARMA', 'M&M', 'TITAN', 'ULTRACEMCO', 'WIPRO',
        'TECHM', 'GRASIM', 'NTPC', 'POWERGRID', 'ONGC', 'COALINDIA', 'BPCL',
        'HINDPETRO', 'IOC', 'GAIL', 'JSWSTEEL', 'TATASTEEL', 'HINDALCO', 'VEDL'
    ],
    'bse': [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK', 'KOTAKBANK',
        'SBIN', 'BHARTIARTL', 'AXISBANK', 'BAJFINANCE', 'DMART', 'ASIANPAINT'
    ],
    'mcx': [
        'CRUDEOIL', 'NATURALGAS', 'GOLD', 'SILVER', 'COPPER', 'ZINC', 'LEAD',
        'ALUMINIUM', 'NICKEL', 'COTTON', 'MENTHAOIL'
    ],
    'forex': [
        'EUR/USD',
        'GBP/USD',
        'USD/JPY',
        'USD/CHF',
    ],
    'commodities': [
        'GC=F',  # Gold
        'SI=F',  # Silver
        'CL=F',  # Crude Oil
    ]
}

# Notification Configuration
NOTIFICATION_CONFIG = {
    'email_enabled': os.getenv('EMAIL_ENABLED', 'false').lower() == 'true',
    'email_smtp_server': os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com'),
    'email_smtp_port': int(os.getenv('EMAIL_SMTP_PORT', 587)),
    'email_username': os.getenv('EMAIL_USERNAME', ''),
    'email_password': os.getenv('EMAIL_PASSWORD', ''),
    'email_recipients': os.getenv('EMAIL_RECIPIENTS', '').split(','),
    
    'telegram_enabled': os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true',
    'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
    'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
    
    'discord_enabled': os.getenv('DISCORD_ENABLED', 'false').lower() == 'true',
    'discord_webhook_url': os.getenv('DISCORD_WEBHOOK_URL', ''),
}

# Database Configuration
DATABASE_CONFIG = {
    'type': os.getenv('DB_TYPE', 'sqlite'),  # 'sqlite', 'postgresql', 'mysql'
    'sqlite_path': os.getenv('SQLITE_PATH', 'data/trading_bot.db'),
    'postgresql_host': os.getenv('POSTGRESQL_HOST', 'localhost'),
    'postgresql_port': int(os.getenv('POSTGRESQL_PORT', 5432)),
    'postgresql_database': os.getenv('POSTGRESQL_DATABASE', 'trading_bot'),
    'postgresql_username': os.getenv('POSTGRESQL_USERNAME', ''),
    'postgresql_password': os.getenv('POSTGRESQL_PASSWORD', ''),
}

# Web Interface Configuration
WEB_CONFIG = {
    'enabled': os.getenv('WEB_ENABLED', 'true').lower() == 'true',
    'host': os.getenv('WEB_HOST', '0.0.0.0'),
    'port': int(os.getenv('WEB_PORT', 8501)),
    'debug': os.getenv('WEB_DEBUG', 'false').lower() == 'true',
    'auth_enabled': os.getenv('WEB_AUTH_ENABLED', 'false').lower() == 'true',
    'username': os.getenv('WEB_USERNAME', 'admin'),
    'password': os.getenv('WEB_PASSWORD', 'password'),
}

# Performance Monitoring
MONITORING_CONFIG = {
    'enabled': os.getenv('MONITORING_ENABLED', 'true').lower() == 'true',
    'metrics_interval_seconds': int(os.getenv('METRICS_INTERVAL_SECONDS', 60)),
    'health_check_interval_seconds': int(os.getenv('HEALTH_CHECK_INTERVAL_SECONDS', 300)),
    'performance_log_interval_hours': int(os.getenv('PERFORMANCE_LOG_INTERVAL_HOURS', 24)),
}

# Strategy Configuration
STRATEGY_CONFIG = {
    'default_strategy': os.getenv('DEFAULT_STRATEGY', 'ai_enhanced'),
    'strategies': {
        'ai_enhanced': {
            'enabled': True,
            'min_confidence': 0.6,
            'use_ml_signals': True,
            'use_technical_signals': True,
        },
        'technical_only': {
            'enabled': True,
            'min_confidence': 0.7,
            'use_ml_signals': False,
            'use_technical_signals': True,
        },
        'mean_reversion': {
            'enabled': False,
            'bb_period': 20,
            'bb_std': 2,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
        },
        'momentum': {
            'enabled': False,
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
        },
    }
}

# Environment-specific settings
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

if ENVIRONMENT == 'production':
    LOGGING_CONFIG['level'] = 'WARNING'
    WEB_CONFIG['debug'] = False
    TRADING_CONFIG['risk_per_trade'] = 0.01  # Lower risk in production
elif ENVIRONMENT == 'testing':
    LOGGING_CONFIG['level'] = 'DEBUG'
    EXCHANGE_CONFIG['binance']['sandbox'] = True
    EXCHANGE_CONFIG['coinbase']['sandbox'] = True

# Validate critical configuration
def validate_config():
    """Validate critical configuration parameters"""
    errors = []
    
    if TRADING_CONFIG['initial_balance'] <= 0:
        errors.append("Initial balance must be positive")
    
    if not (0 < TRADING_CONFIG['risk_per_trade'] <= 0.1):
        errors.append("Risk per trade must be between 0 and 10%")
    
    if RISK_CONFIG['max_position_size'] <= 0 or RISK_CONFIG['max_position_size'] > 1:
        errors.append("Max position size must be between 0 and 100%")
    
    if EXCHANGE_CONFIG['binance']['enable'] and not EXCHANGE_CONFIG['binance']['api_key']:
        errors.append("Binance API key required when Binance is enabled")
    
    if errors:
        raise ValueError("Configuration validation failed: " + "; ".join(errors))

# Validate configuration on import
try:
    validate_config()
except ValueError as e:
    print(f"Configuration Error: {e}")
    print("Please check your .env file and configuration settings.")
