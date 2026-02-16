import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="AI Trading Bot",
    page_icon="🤖",
    layout="wide"
)

# Header
st.title("🤖 AI Trading Bot Dashboard")
st.markdown("---")

# Generate sample data
np.random.seed(42)
dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
portfolio_value = 10000 + np.cumsum(np.random.randn(30) * 100)

# Portfolio Section
st.subheader("📊 Portfolio Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Current Value", f"${portfolio_value[-1]:,.2f}")
    
with col2:
    st.metric("Total Return", f"{((portfolio_value[-1] - portfolio_value[0]) / portfolio_value[0] * 100):.2f}%")
    
with col3:
    st.metric("Daily Change", f"{np.random.randn()*2:.2f}%")
    
with col4:
    st.metric("Win Rate", "65.5%")

# Chart
fig = px.line(
    x=dates, 
    y=portfolio_value,
    title="Portfolio Performance",
    labels={"x": "Date", "y": "Value ($)"}
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Trading Signals
st.subheader("🤖 AI Trading Signals")

signals = pd.DataFrame({
    'Symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
    'Signal': ['BUY', 'SELL', 'HOLD', 'BUY'],
    'Confidence': [85, 72, 60, 90],
    'Price': [150.25, 2800.50, 300.75, 3200.00]
})

st.dataframe(signals, use_container_width=True)

st.markdown("---")

# Risk Analysis
st.subheader("🛡️ Risk Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("VaR (95%)", f"${np.random.rand()*1000:,.2f}")
    
with col2:
    st.metric("Max Drawdown", f"{np.random.rand()*10:.2f}%")
    
with col3:
    st.metric("Sharpe Ratio", f"{np.random.rand()*2:.2f}")

# Footer
st.markdown("---")
st.markdown("🤖 AI Trading Bot Dashboard v1.0 - Real-time Trading Analytics")
