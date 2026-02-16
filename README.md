# AI Trading Bot

An intelligent trading bot that combines technical analysis with machine learning to make automated trading decisions across multiple asset classes including cryptocurrencies, stocks, forex, and commodities.

## Features

### 🤖 AI-Powered Trading
- **Machine Learning Models**: Random Forest and LSTM neural networks
- **Ensemble Methods**: Combines multiple models for better predictions
- **Technical Analysis**: RSI, MACD, Bollinger Bands, Moving Averages
- **Adaptive Learning**: Models retrain automatically with new data

### 🛡️ Risk Management
- **Position Sizing**: Kelly Criterion and risk-based position sizing
- **Stop Loss/Take Profit**: Automatic risk controls
- **Portfolio Monitoring**: Real-time risk metrics and alerts
- **Value at Risk (VaR)**: Statistical risk measurement

### 💱 Multi-Exchange Support
- **Binance**: Cryptocurrency trading
- **Yahoo Finance**: Stock and commodity data
- **Extensible**: Easy to add new exchanges

### 📊 Analytics & Monitoring
- **Real-time Dashboard**: Streamlit-based web interface
- **Performance Metrics**: Sharpe ratio, drawdown, returns
- **Backtesting**: Historical strategy testing
- **Logging**: Comprehensive trade and system logging

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai_trading_bot

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
```

### 2. Configuration

Edit the `.env` file with your settings:

```bash
# Basic trading settings
INITIAL_BALANCE=10000
RISK_PER_TRADE=0.02

# Exchange API keys (optional for demo mode)
BINANCE_API_KEY=your_api_key
BINANCE_SECRET=your_secret
```

### 3. Run the Bot

#### Demo Mode (No API keys required)
```bash
python main.py --mode demo --symbols BTC/USDT ETH/USDT AAPL
```

#### Train AI Model
```bash
python main.py --mode train --symbols BTC/USDT ETH/USDT
```

#### Live Trading
```bash
python main.py --mode live --symbols BTC/USDT ETH/USDT
```

#### Backtesting
```bash
python main.py --mode backtest --symbols BTC/USDT --backtest-start 2023-01-01 --backtest-end 2024-01-01
```

## Project Structure

```
ai_trading_bot/
├── src/
│   ├── trading_bot.py      # Main trading bot engine
│   ├── ai_model.py         # Machine learning models
│   └── risk_management.py  # Risk management system
├── data/                   # Data storage
├── models/                 # Trained AI models
├── logs/                   # Trading logs
├── tests/                  # Unit tests
├── config.py               # Configuration settings
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
└── .env.example           # Environment variables template
```

## Trading Strategies

### AI Enhanced Strategy
The default strategy combines:
- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **Machine Learning**: Pattern recognition and prediction
- **Risk Management**: Position sizing and stop-losses

### Technical Analysis Only
Uses traditional technical analysis without AI:
- Moving average crossovers
- RSI overbought/oversold conditions
- MACD signals

## Risk Management

### Built-in Safety Features
- **Maximum Position Size**: Limits exposure per trade
- **Portfolio Concentration**: Prevents over-concentration
- **Daily Loss Limits**: Stops trading if losses exceed threshold
- **Maximum Drawdown**: Protects against large losses
- **Leverage Limits**: Controls position sizing

### Risk Metrics
- Value at Risk (VaR)
- Sharpe Ratio
- Maximum Drawdown
- Portfolio Volatility
- Beta Coefficient

## AI Models

### Random Forest
- Fast training and prediction
- Handles non-linear relationships
- Feature importance analysis
- Good for classification tasks

### LSTM Neural Network
- Time series prediction
- Sequential pattern recognition
- Handles complex market dynamics
- Requires more training data

### Ensemble Model
- Combines multiple models
- Reduces overfitting
- Improves prediction accuracy
- Weighted voting system

## Supported Assets

### Cryptocurrencies
- Bitcoin (BTC/USDT)
- Ethereum (ETH/USDT)
- Binance Coin (BNB/USDT)
- And many more...

### Stocks
- Apple (AAPL)
- Google (GOOGL)
- Microsoft (MSFT)
- And major stocks...

### Forex
- EUR/USD, GBP/USD
- USD/JPY, USD/CHF
- Major currency pairs

### Commodities
- Gold (GC=F)
- Silver (SI=F)
- Crude Oil (CL=F)

## Configuration Options

### Trading Settings
- `INITIAL_BALANCE`: Starting capital
- `RISK_PER_TRADE`: Risk percentage per trade
- `MAX_POSITIONS`: Maximum concurrent positions
- `MIN_CONFIDENCE_THRESHOLD`: Minimum confidence for trades

### Risk Management
- `MAX_POSITION_SIZE`: Maximum position size (% of portfolio)
- `MAX_DAILY_LOSS`: Maximum daily loss (%)
- `MAX_DRAWDOWN`: Maximum allowed drawdown
- `STOP_LOSS_PCT`: Stop loss percentage

### AI Model Settings
- `AI_MODEL_TYPE`: Model type (random_forest, lstm, ensemble)
- `RETRAIN_INTERVAL_HOURS`: Model retraining frequency
- `MIN_TRAINING_SAMPLES`: Minimum samples for training

## Monitoring & Alerts

### Web Dashboard
- Real-time portfolio value
- Open positions and P&L
- Trading history
- Risk metrics

### Notifications
- Email alerts for important events
- Telegram bot notifications
- Discord webhook integration

### Logging
- Detailed trade logs
- System performance metrics
- Error tracking and debugging

## Backtesting

Test strategies on historical data:

```bash
python main.py --mode backtest \
  --symbols BTC/USDT ETH/USDT \
  --backtest-start 2023-01-01 \
  --backtest-end 2024-01-01
```

Backtesting features:
- Historical data simulation
- Commission and slippage modeling
- Performance metrics calculation
- Trade-by-trade analysis

## Security

### API Key Management
- Environment variable storage
- Sandbox mode for testing
- No hardcoded credentials
- Secure key handling

### Risk Controls
- Multiple safety layers
- Emergency stop mechanisms
- Real-time monitoring
- Automatic position closure

## Performance Optimization

### Data Handling
- Efficient data caching
- Minimal API calls
- Optimized indicators calculation
- Memory-efficient processing

### Model Performance
- Feature selection
- Hyperparameter tuning
- Model versioning
- Performance monitoring

## Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Check API keys in `.env` file
   - Verify network connectivity
   - Ensure exchange is enabled in config

2. **Model Training Errors**
   - Ensure sufficient historical data
   - Check data quality and completeness
   - Verify feature calculations

3. **Memory Issues**
   - Reduce lookback period
   - Clear data cache regularly
   - Optimize data processing

### Debug Mode
Enable debug logging:
```bash
LOG_LEVEL=DEBUG python main.py --mode demo
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Disclaimer

⚠️ **IMPORTANT**: This trading bot is for educational and research purposes only. Trading involves substantial risk of loss. Use at your own risk and never invest more than you can afford to lose.

- Past performance does not guarantee future results
- Cryptocurrency markets are highly volatile
- Always test thoroughly before using real funds
- Consider consulting with financial professionals

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**Happy Trading! 🚀**
# ai-trading-bot
# ai-trading-bot
