# 📋 Manual GitHub Setup Instructions

Since Git is not installed on your system, here are the manual steps to deploy your AI Trading Bot to Streamlit Cloud:

## 🔧 Step 1: Install Git

### Windows:
1. **Download Git**: Go to https://git-scm.com/download/win
2. **Run Installer**: Download and run the Git for Windows installer
3. **Choose Options**: Use default settings during installation
4. **Restart**: Restart your computer after installation
5. **Verify**: Open Command Prompt and type `git --version`

### Alternative (Windows Package Manager):
If you have winget (Windows 10/11):
```cmd
winget install --id Git.Git -e --source winget
```

## 📁 Step 2: Initialize Git Repository

After Git is installed, open Command Prompt and navigate to your project:

```cmd
cd "C:\Users\bobcat\CascadeProjects\ai_trading_bot"
```

Then run these commands:

```cmd
git init
git add .
git commit -m "Initial commit: AI Trading Bot with Dashboard"
```

## 🌐 Step 3: Create GitHub Repository

1. **Go to GitHub**: https://github.com
2. **Sign in** to your account
3. **Click "+"** → "New repository"
4. **Repository settings**:
   - **Name**: `ai-trading-bot`
   - **Description**: AI-powered trading bot with real-time dashboard
   - **Visibility**: Public or Private (your choice)
   - **Don't initialize** with README (we already have files)
5. **Click "Create repository"**

## 🔗 Step 4: Connect Local to GitHub

```cmd
git remote add origin https://github.com/YOUR_USERNAME/ai-trading-bot.git
git branch -M main
git push -u origin main
```

**Replace** `YOUR_USERNAME` with your actual GitHub username.

## ☁️ Step 5: Deploy to Streamlit Cloud

### Method 1: Web Interface
1. **Go to**: https://share.streamlit.io
2. **Click "New app"**
3. **Repository**: Select `ai-trading-bot`
4. **Branch**: `main`
5. **Main file path**: `src/ai_dashboard.py`
6. **Click "Deploy"**

### Method 2: Streamlit CLI (if you have Streamlit installed)
```cmd
streamlit login
streamlit deploy
```

## 🔐 Step 6: Configure Environment Variables

In Streamlit Cloud, add these secrets:

### Kite API (Indian Markets):
```
KITE_API_KEY=nyj6rh8b0exlwh23
KITE_API_SECRET=qx662nkun2xes6tpghv4segsamu7swg9
KITE_ENABLE=true
KITE_DB_PATH=data/kite_trading.db
KITE_DEFAULT_EXCHANGE=NSE
KITE_DEFAULT_PRODUCT=NRML
```

### Swift API (Financial Services):
```
SWIFT_API_KEY=your_swift_api_key
SWIFT_API_SECRET=your_swift_api_secret
SWIFT_ENABLE=false
SWIFT_ACCOUNT_ID=your_swift_account_id
SWIFT_ENVIRONMENT=sandbox
```

### Trading Configuration:
```
INITIAL_BALANCE=10000
RISK_PER_TRADE=0.02
MAX_POSITIONS=5
MIN_CONFIDENCE_THRESHOLD=0.6
TRADING_INTERVAL_MINUTES=60
```

### Risk Management:
```
MAX_POSITION_SIZE=0.1
MAX_SECTOR_EXPOSURE=0.3
MAX_DAILY_LOSS=0.05
MAX_DRAWDOWN=0.2
MIN_LIQUIDITY_RATIO=0.1
MAX_LEVERAGE=2.0
VAR_CONFIDENCE=0.95
STOP_LOSS_PCT=0.02
TAKE_PROFIT_PCT=0.05
```

## 📱 Your App URL

Once deployed, your app will be available at:
```
https://YOUR_USERNAME-ai-trading-bot.streamlit.app
```

## 🔄 Updating Your App

To update your deployed app:

```cmd
git add .
git commit -m "Update: description of changes"
git push origin main
```

Streamlit Cloud will automatically redeploy your app.

## 📚 Additional Resources

- **Streamlit Documentation**: https://docs.streamlit.io
- **GitHub Documentation**: https://docs.github.com
- **Community Forum**: https://discuss.streamlit.io

## ✅ Checklist Before Deployment

- [ ] Git installed and working
- [ ] GitHub account created
- [ ] Repository created on GitHub
- [ ] All files committed to Git
- [ ] Code pushed to GitHub
- [ ] Streamlit account created
- [ ] Environment variables configured in Streamlit Cloud

## 🎯 Next Steps

1. **Install Git** following the instructions above
2. **Create GitHub repository** with the specified name
3. **Run the Git commands** to push your code
4. **Deploy to Streamlit Cloud** using the web interface
5. **Configure secrets** in Streamlit Cloud settings
6. **Test your deployed app**

---

**🚀 Your AI Trading Bot will be live on Streamlit Cloud!**
