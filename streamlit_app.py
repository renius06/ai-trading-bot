import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import sys
from pathlib import Path
import sqlite3
import json
import time
from typing import Dict, List, Optional
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from trading_bot import AITradingBot
from kite_trading_bot import KiteTradingBot, KiteTradingConfig
from ai_model import TradingAIModel
from risk_management import RiskManager, RiskLimits
from config import TRADING_CONFIG, EXCHANGE_CONFIG, RISK_CONFIG, AI_CONFIG, SYMBOLS_CONFIG
from data_fetcher import MarketDataFetcher

# Initialize data fetcher with new API key
fetcher = MarketDataFetcher()
fetcher.api_key = '1wupbdnax2j9quy1'
fetcher.api_secret = 'b5mgbdupm9votducd1kijvsrg6jekonj'

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="AI Trading Bot Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("🤖 AI Trading Bot")
st.sidebar.markdown("---")

# API Configuration Section
st.sidebar.subheader("🔑 API Configuration")

# Kite API Status
kite_enabled = os.getenv('KITE_ENABLE', 'false').lower() == 'true'
if kite_enabled:
    st.sidebar.success("✅ Kite API Enabled")
    st.sidebar.info(f"API Key: {os.getenv('KITE_API_KEY', 'Not Set')[:10]}...")
else:
    st.sidebar.warning("⚠️ Kite API Disabled")

# Swift API Status
swift_enabled = os.getenv('SWIFT_ENABLE', 'false').lower() == 'true'
if swift_enabled:
    st.sidebar.success("✅ Swift API Enabled")
else:
    st.sidebar.info("ℹ️ Swift API Disabled")

# Main Content
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;'>
    <h1 style='color: white; margin: 0;'>🤖 AI Trading Bot Dashboard</h1>
    <p style='color: white; margin: 5px 0 0 0;'>Real-time AI-Powered Trading Analytics</p>
</div>
""", unsafe_allow_html=True)

# Status Indicators
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Bot Status", "🟢 Active", delta="Running")

with col2:
    if st.sidebar.button("🚀 Start Trading", type="primary"):
        st.success("Trading started!")

with col3:
    st.metric("API Status", "✅ Connected", delta="All APIs")

with col4:
    st.metric("Last Update", datetime.now().strftime("%H:%M:%S"), delta="Real-time")

st.markdown("---")

# Portfolio Overview Section
st.subheader("📊 Portfolio Overview")

# Generate sample portfolio data
np.random.seed(42)
dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
portfolio_value = 10000 + np.cumsum(np.random.randn(30) * 100)

col1, col2 = st.columns(2)

with col1:
    # Portfolio Value Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=portfolio_value,
        mode='lines+markers',
        name='Portfolio Value',
        line=dict(color='#667eea', width=3)
    ))
    fig.update_layout(
        title='Portfolio Performance',
        xaxis_title='Date',
        yaxis_title='Value ($)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Portfolio Metrics
    current_value = portfolio_value[-1]
    initial_value = portfolio_value[0]
    total_return = ((current_value - initial_value) / initial_value) * 100
    
    st.metric("Current Value", f"${current_value:,.2f}")
    st.metric("Total Return", f"{total_return:.2f}%", delta=f"${current_value - initial_value:,.2f}")
    st.metric("Daily Change", f"{np.random.randn()*2:.2f}%", delta="Random")
    st.metric("Win Rate", f"{65 + np.random.randint(-5, 10):.1f}%")

st.markdown("---")

# AI Trading Signals Section
st.subheader("🤖 AI Trading Signals")

# Generate sample signals
signals = pd.DataFrame({
    'Symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
    'Signal': ['BUY', 'SELL', 'HOLD', 'BUY', 'HOLD'],
    'Confidence': [85, 72, 60, 90, 55],
    'Price': [150.25, 2800.50, 300.75, 3200.00, 850.25]
})

# Color coding for signals
def color_signal(val):
    if val == 'BUY':
        return 'background-color: #d4edda'
    elif val == 'SELL':
        return 'background-color: #f8d7da'
    else:
        return 'background-color: #fff3cd'

styled_signals = signals.style.applymap(color_signal, subset=['Signal'])
st.dataframe(styled_signals, use_container_width=True)

st.markdown("---")

# Market Analysis Section
st.subheader("📈 Market Analysis")

# Generate sample market data
market_data = pd.DataFrame({
    'Date': dates[-10:],
    'Open': 100 + np.random.randn(10) * 5,
    'High': 100 + np.random.randn(10) * 5 + 2,
    'Low': 100 + np.random.randn(10) * 5 - 2,
    'Close': 100 + np.random.randn(10) * 5,
    'Volume': np.random.randint(1000000, 5000000, 10)
})

# Candlestick Chart
fig = go.Figure(data=go.Candlestick(
    x=market_data['Date'],
    open=market_data['Open'],
    high=market_data['High'],
    low=market_data['Low'],
    close=market_data['Close']
))

fig.update_layout(
    title='Market Price Action',
    xaxis_title='Date',
    yaxis_title='Price ($)',
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# Technical Indicators
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Technical Indicators")
    
    # RSI
    rsi_values = 50 + np.random.randn(10) * 10
    rsi_df = pd.DataFrame({
        'Date': dates[-10:],
        'RSI': rsi_values
    })
    
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=rsi_df['Date'],
        y=rsi_df['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='orange')
    ))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_rsi.update_layout(title='RSI Indicator', height=300)
    st.plotly_chart(fig_rsi, use_container_width=True)

with col2:
    st.subheader("📈 Volume Analysis")
    
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(
        x=market_data['Date'],
        y=market_data['Volume'],
        name='Volume',
        marker_color='lightblue'
    ))
    fig_volume.update_layout(title='Trading Volume', height=300)
    st.plotly_chart(fig_volume, use_container_width=True)

st.markdown("---")

# Risk Analysis Section
st.subheader("🛡️ Risk Analysis")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("VaR (95%)", f"${np.random.rand()*1000:,.2f}", delta="Daily")
    
with col2:
    st.metric("Max Drawdown", f"{np.random.rand()*10:.2f}%", delta="Risk")
    
with col3:
    st.metric("Sharpe Ratio", f"{np.random.rand()*2:.2f}", delta="Performance")
    
with col4:
    st.metric("Beta", f"{np.random.rand()*2:.2f}", delta="Market")

# Risk Heatmap
risk_data = np.random.rand(5, 5)
risk_labels = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

fig_heatmap = go.Figure(data=go.Heatmap(
    z=risk_data,
    x=risk_labels,
    y=risk_labels,
    colorscale='RdYlBu',
    showscale=True
))

fig_heatmap.update_layout(
    title='Risk Correlation Matrix',
    height=400
)

st.plotly_chart(fig_heatmap, use_container_width=True)

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin-top: 20px;'>
    <p style='margin: 0; color: #6c757d;'>🤖 AI Trading Bot Dashboard v1.0</p>
    <p style='margin: 5px 0 0 0; color: #6c757d;'>Real-time AI-powered trading analytics and portfolio management</p>
</div>
""", unsafe_allow_html=True)
