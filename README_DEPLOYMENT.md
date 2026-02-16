# 🚀 AI Trading Bot - Streamlit Cloud Deployment Guide

## 📋 Prerequisites

1. **GitHub Account**: Create a free account at [github.com](https://github.com)
2. **Git Installation**: Install Git on your system
3. **Streamlit Account**: Create account at [share.streamlit.io](https://share.streamlit.io)

## 🔧 Step 1: Install Git

### Windows:
```bash
# Download and install Git from: https://git-scm.com/download/win
# Or use winget (Windows 10/11)
winget install --id Git.Git -e --source winget
```

### macOS:
```bash
brew install git
```

### Linux:
```bash
sudo apt-get install git
# For Ubuntu/Debian
# or
sudo yum install git
# For CentOS/RHEL/Fedora
```

## 📁 Step 2: Initialize Git Repository

```bash
# Navigate to your project directory
cd C:\Users\bobcat\CascadeProjects\ai_trading_bot

# Initialize Git repository
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: AI Trading Bot with Dashboard"
```

## 🌐 Step 3: Create GitHub Repository

1. **Go to GitHub**: [github.com](https://github.com)
2. **Sign in** to your account
3. **Click "+"** → "New repository"
4. **Repository name**: `ai-trading-bot`
5. **Description**: "AI-powered trading bot with real-time dashboard"
6. **Visibility**: Choose Public or Private
7. **Don't initialize** with README (we already have files)
8. **Click "Create repository"**

## 🔗 Step 4: Connect Local to GitHub

```bash
# Add remote repository (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/ai-trading-bot.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## ☁️ Step 5: Deploy to Streamlit Community Cloud

### Method 1: Through Streamlit Website

1. **Go to**: [share.streamlit.io](https://share.streamlit.io)
2. **Click "New app"**
3. **Repository**: Select `ai-trading-bot`
4. **Branch**: `main`
5. **Main file path**: `src/ai_dashboard.py`
6. **Click "Deploy"**

### Method 2: Using Streamlit CLI

```bash
# Install streamlit CLI (if not already installed)
pip install streamlit

# Login to Streamlit
streamlit login

# Deploy
streamlit run src/ai_dashboard.py --server.headless true
```

## 🔐 Step 6: Configure Environment Variables

In Streamlit Cloud, you need to set up secrets:

1. **Go to your app** on Streamlit Cloud
2. **Click "⋮"** → "Settings"
3. **Go to "Secrets" section**
4. **Add your environment variables**:

```toml
# Kite API
KITE_API_KEY="your_kite_api_key"
KITE_API_SECRET="your_kite_api_secret"
KITE_ACCESS_TOKEN="your_kite_access_token"
KITE_REQUEST_TOKEN="your_kite_request_token"
KITE_ENABLE="true"
KITE_DB_PATH="data/kite_trading.db"
KITE_DEFAULT_EXCHANGE="NSE"
KITE_DEFAULT_PRODUCT="NRML"

# Swift API
SWIFT_API_KEY="your_swift_api_key"
SWIFT_API_SECRET="your_swift_api_secret"
SWIFT_ACCESS_TOKEN="your_swift_access_token"
SWIFT_BASE_URL="https://api.swift.com"
SWIFT_ENABLE="false"
SWIFT_ACCOUNT_ID="your_swift_account_id"
SWIFT_ENVIRONMENT="sandbox"
SWIFT_TIMEOUT="30"

# Trading Configuration
INITIAL_BALANCE="10000"
RISK_PER_TRADE="0.02"
MAX_POSITIONS="5"
MIN_CONFIDENCE_THRESHOLD="0.6"
TRADING_INTERVAL_MINUTES="60"
LOOKBACK_PERIOD="100"

# Risk Management
MAX_POSITION_SIZE="0.1"
MAX_SECTOR_EXPOSURE="0.3"
MAX_DAILY_LOSS="0.05"
MAX_DRAWDOWN="0.2"
MIN_LIQUIDITY_RATIO="0.1"
MAX_LEVERAGE="2.0"
VAR_CONFIDENCE="0.95"
VAR_TIMEFRAME="1"
STOP_LOSS_PCT="0.02"
TAKE_PROFIT_PCT="0.05"

# AI Model Configuration
AI_MODEL_TYPE="random_forest"
RETRAIN_INTERVAL_HOURS="24"
MIN_TRAINING_SAMPLES="1000"
FEATURE_WINDOW="20"
PREDICTION_HORIZON="5"
MODEL_SAVE_INTERVAL_HOURS="6"

# Database
DB_TYPE="sqlite"
SQLITE_PATH="data/trading_bot.db"

# Web Interface
WEB_ENABLED="true"
WEB_HOST="0.0.0.0"
WEB_PORT="8501"
WEB_DEBUG="false"
WEB_AUTH_ENABLED="false"
WEB_USERNAME="admin"
WEB_PASSWORD="password"

# Monitoring
MONITORING_ENABLED="true"
METRICS_INTERVAL_SECONDS="60"
HEALTH_CHECK_INTERVAL_SECONDS="300"
PERFORMANCE_LOG_INTERVAL_HOURS="24"

# Environment
ENVIRONMENT="production"
```

## 📦 Step 7: Update Requirements for Cloud

Create a `requirements.txt` file for Streamlit Cloud:

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
yfinance>=0.2.0
ccxt>=4.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
python-dotenv>=1.0.0
requests>=2.31.0
ta>=0.10.0
backtesting>=0.3.3
kiteconnect>=4.0.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
mysql-connector-python>=8.1.0
scipy>=1.10.0
joblib>=1.3.0
```

## 🚀 Step 8: Deploy and Test

1. **Push any changes** to GitHub:
```bash
git add .
git commit -m "Ready for Streamlit Cloud deployment"
git push origin main
```

2. **Monitor deployment** on Streamlit Cloud
3. **Test your app** at the provided URL
4. **Check logs** if any issues occur

## 🐛 Common Issues & Solutions

### Issue 1: Module Not Found
**Solution**: Ensure all dependencies are in `requirements.txt`

### Issue 2: Environment Variables Missing
**Solution**: Add all secrets to Streamlit Cloud settings

### Issue 3: File Not Found
**Solution**: Check file paths and ensure all files are committed

### Issue 4: Permission Denied
**Solution**: Check API keys and permissions

## 📱 Your App URL

Once deployed, your app will be available at:
```
https://YOUR_USERNAME-ai-trading-bot.streamlit.app
```

## 🔄 Updating Your App

To update your deployed app:

```bash
# Make changes locally
git add .
git commit -m "Update: description of changes"
git push origin main

# Streamlit Cloud will auto-redeploy
```

## 🎯 Best Practices

1. **Use environment variables** for all sensitive data
2. **Keep requirements.txt updated**
3. **Test locally** before deploying
4. **Use meaningful commit messages**
5. **Monitor app performance** regularly
6. **Set up logging** for debugging

## 📞 Support

- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **GitHub Docs**: [docs.github.com](https://docs.github.com)
- **Community Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)

---

**🎉 Your AI Trading Bot is now ready for cloud deployment!**
