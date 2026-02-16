# Kite API Integration for AI Trading Bot

This document provides comprehensive information about the Kite API integration for trading in Indian markets through Zerodha.

## 🚀 Features

### **Kite API Integration**
- **Full API Support**: Complete Kite Connect API integration
- **Indian Markets**: NSE, BSE, and MCX exchange support
- **Real-time Data**: Live market data via WebSocket
- **Order Management**: Complete order lifecycle management
- **Portfolio Tracking**: Real-time positions and holdings monitoring

### **Database Backend**
- **SQLite Database**: Local database for trade history and analytics
- **SQLAlchemy ORM**: Structured data management
- **Trade History**: Complete order and position tracking
- **Market Data Storage**: Historical data caching

### **Advanced Features**
- **AI-Powered Signals**: Machine learning for Indian markets
- **Risk Management**: Indian market-specific risk controls
- **Multi-Exchange**: Trade across NSE, BSE, MCX
- **Technical Analysis**: Indicators optimized for Indian stocks

## 📋 Prerequisites

### **Kite API Credentials**
1. **Zerodha Account**: Active trading account with Zerodha
2. **Kite Connect API**: API key and secret from developers.zerodha.com
3. **Access Token**: Valid access token for API access

### **Installation**
```bash
# Install additional dependencies
pip install kiteconnect sqlalchemy psycopg2-binary mysql-connector-python

# Copy environment configuration
cp .env.example .env

# Configure Kite API credentials
```

## ⚙️ Configuration

### **Environment Variables (.env)**
```bash
# Kite API Configuration
KITE_API_KEY=your_kite_api_key_here
KITE_API_SECRET=your_kite_api_secret_here
KITE_ACCESS_TOKEN=your_kite_access_token_here
KITE_REQUEST_TOKEN=your_kite_request_token_here
KITE_ENABLE=true
KITE_DB_PATH=data/kite_trading.db
KITE_DEFAULT_EXCHANGE=NSE
KITE_DEFAULT_PRODUCT=NRML
```

### **Supported Exchanges**
- **NSE**: National Stock Exchange (Equities, F&O)
- **BSE**: Bombay Stock Exchange (Equities)
- **MCX**: Multi Commodity Exchange (Commodities)

### **Product Types**
- **NRML**: Normal (overnight positions)
- **MIS**: Margin Intraday Squareoff
- **CNC**: Cash and Carry (Delivery)

## 🎯 Usage Examples

### **Training AI Model**
```bash
# Train model on NSE stocks
python main.py --mode kite-train --symbols RELIANCE TCS INFY HDFCBANK --exchange NSE

# Train model on specific sectors
python main.py --mode kite-train --symbols SBIN AXISBANK KOTAKBANK --exchange NSE
```

### **Live Trading**
```bash
# Live trading on NSE
python main.py --mode kite-live --symbols RELIANCE TCS INFY --exchange NSE

# Live trading on BSE
python main.py --mode kite-live --symbols RELIANCE TCS --exchange BSE

# Commodity trading on MCX
python main.py --mode kite-live --symbols CRUDEOIL GOLD SILVER --exchange MCX
```

### **Demo Mode**
```bash
# Demo with default NSE symbols
python main.py --mode demo

# Demo with specific symbols
python main.py --mode demo --symbols RELIANCE TCS INFY
```

## 📊 Available Symbols

### **NSE Equities**
```python
# Large Cap Stocks
'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK', 'KOTAKBANK',
'SBIN', 'BHARTIARTL', 'AXISBANK', 'BAJFINANCE', 'DMART', 'ASIANPAINT'

# Mid Cap Stocks
'MARUTI', 'HCLTECH', 'SUNPHARMA', 'M&M', 'TITAN', 'ULTRACEMCO', 'WIPRO',
'TECHM', 'GRASIM', 'NTPC', 'POWERGRID', 'ONGC', 'COALINDIA'
```

### **MCX Commodities**
```python
# Energy
'CRUDEOIL', 'NATURALGAS'

# Precious Metals
'GOLD', 'SILVER'

# Base Metals
'COPPER', 'ZINC', 'LEAD', 'ALUMINIUM', 'NICKEL'

# Agricultural
'COTTON', 'MENTHAOIL'
```

## 🗄️ Database Schema

### **Tables Structure**

#### **kite_trades**
- Order tracking and execution history
- Complete order lifecycle management

#### **kite_positions**
- Current open positions tracking
- Real-time P&L calculation

#### **kite_holdings**
- Delivery holdings management
- Long-term portfolio tracking

#### **kite_market_data**
- Historical market data storage
- Technical analysis data caching

## 🔧 Advanced Configuration

### **Risk Management Settings**
```python
# Indian market specific risk limits
RISK_CONFIG = {
    'max_position_size': 0.1,        # 10% per position
    'max_sector_exposure': 0.25,      # 25% per sector (SEBI limit)
    'max_daily_loss': 0.05,           # 5% daily loss limit
    'max_drawdown': 0.20,             # 20% max drawdown
    'stop_loss_pct': 0.02,            # 2% stop loss
    'take_profit_pct': 0.05,          # 5% take profit
}
```

### **Trading Hours**
```python
# NSE/BSE Trading Hours
EQUITY_MARKET_HOURS = {
    'open': '09:15',
    'close': '15:30',
    'break_start': None,  # No lunch break now
}

# MCX Trading Hours
COMMODITY_MARKET_HOURS = {
    'open': '09:00',
    'close': '23:30',
    'break_start': '17:00',
    'break_end': '17:30'
}
```

## 📈 AI Model Training

### **Data Sources**
- **Historical Data**: 1+ year of daily data
- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **Market Features**: Volume, volatility, price patterns
- **Indian Market Factors**: Sector rotation, market sentiment

### **Model Types**
- **Random Forest**: Pattern recognition
- **LSTM**: Time series prediction
- **Ensemble**: Combined model approach

### **Training Process**
```python
# Automatic feature engineering
def prepare_features(df):
    # Price-based features
    features['returns'] = df['close'].pct_change()
    features['volatility'] = features['returns'].rolling(20).std()
    
    # Technical indicators
    features['rsi'] = ta.momentum.rsi(df['close'])
    features['macd'] = ta.trend.macd_diff(df['close'])
    
    # Market-specific features
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    return features
```

## 🛡️ Risk Management

### **Position Sizing**
- **Kelly Criterion**: Optimal position sizing
- **Volatility-Based**: Adjust for market volatility
- **Sector Limits**: SEBI compliance
- **Leverage Control**: Risk-based leverage

### **Stop Loss Management**
- **Fixed Stop Loss**: Percentage-based stops
- **Trailing Stops**: Dynamic profit protection
- **Time-based Exits**: Intraday square-off
- **Risk-Reward Ratio**: Minimum 1:2 ratio

### **Portfolio Monitoring**
- **Real-time P&L**: Live position tracking
- **Margin Requirements**: Continuous monitoring
- **Drawdown Control**: Automatic position reduction
- **Concentration Limits**: Sector and stock limits

## 📱 Monitoring & Alerts

### **Dashboard Features**
- **Portfolio Overview**: Real-time value and P&L
- **Position Tracking**: Live position monitoring
- **Order History**: Complete trade log
- **Risk Metrics**: VaR, drawdown, Sharpe ratio

### **Alert System**
- **Price Alerts**: Significant price movements
- **Risk Alerts**: Limit breaches
- **Order Alerts**: Execution confirmations
- **System Alerts**: Connection and API status

## 🔍 Market Data

### **Real-time Data**
- **WebSocket Integration**: Live price feeds
- **Tick Data**: High-frequency data capture
- **Multiple Symbols**: Concurrent data streams
- **Auto-reconnection**: Robust connection handling

### **Historical Data**
- **Daily Data**: Complete price history
- **Technical Indicators**: Pre-calculated indicators
- **Corporate Actions**: Splits, bonuses, dividends
- **Market Events**: Holiday and session data

## 🚨 Important Considerations

### **Regulatory Compliance**
- **SEBI Guidelines**: Follow all regulatory requirements
- **Margin Requirements**: Maintain adequate margins
- **Position Limits**: Adhere to exchange limits
- **Reporting Requirements**: Trade and position reporting

### **Market Risks**
- **Volatility**: Indian market can be highly volatile
- **Liquidity**: Some stocks may have low liquidity
- **Circuit Filters**: Price movement limits
- **Settlement**: T+1/T+2 settlement cycles

### **Technical Risks**
- **API Limits**: Rate limiting and usage limits
- **Network Issues**: Connectivity problems
- **Data Quality**: Incomplete or corrupted data
- **System Downtime**: Exchange maintenance

## 🛠️ Troubleshooting

### **Common Issues**

#### **API Connection Errors**
```bash
# Check API credentials
KITE_API_KEY=your_key
KITE_API_SECRET=your_secret
KITE_ACCESS_TOKEN=your_token

# Verify network connectivity
ping kite.zerodha.com
```

#### **Order Placement Errors**
```python
# Check market hours
if not is_market_open():
    print("Market is closed")

# Check margin requirements
margins = kite.get_margins()
if margins['equity']['net'] < required_margin:
    print("Insufficient margin")
```

#### **Data Issues**
```python
# Handle missing data
if df.empty:
    print("No data available")
    return

# Check data quality
if df['close'].isna().any():
    df = df.dropna()
```

### **Debug Mode**
```bash
# Enable debug logging
LOG_LEVEL=DEBUG python main.py --mode kite-demo

# Check database
sqlite3 data/kite_trading.db
.tables
```

## 📞 Support

### **Documentation**
- **Kite Connect API**: https://kite.trade/docs/connect/
- **Zerodha Developers**: https://developers.zerodha.com/
- **SEBI Guidelines**: https://www.sebi.gov.in/

### **Community**
- **Trading Forums**: Indian trading communities
- **Discord/Telegram**: Algorithmic trading groups
- **GitHub Issues**: Report bugs and feature requests

---

**⚠️ Disclaimer**: This is for educational purposes only. Trading involves substantial risk. Please consult with financial advisors before trading with real money.
