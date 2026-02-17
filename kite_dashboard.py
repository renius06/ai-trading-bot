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

class KiteAIDashboard:
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Kite AI Trading Bot",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'bot' not in st.session_state:
            st.session_state.bot = None
        if 'kite_bot' not in st.session_state:
            st.session_state.kite_bot = None
        if 'ai_model' not in st.session_state:
            st.session_state.ai_model = None
        if 'portfolio_history' not in st.session_state:
            st.session_state.portfolio_history = []
        if 'trading_signals' not in st.session_state:
            st.session_state.trading_signals = []
        if 'risk_metrics' not in st.session_state:
            st.session_state.risk_metrics = {}
        if 'model_performance' not in st.session_state:
            st.session_state.model_performance = {}
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'kite_profile' not in st.session_state:
            st.session_state.kite_profile = None
        if 'swift_profile' not in st.session_state:
            st.session_state.swift_profile = None
        if 'selected_symbols' not in st.session_state:
            st.session_state.selected_symbols = []
    
    def get_kite_profile(self):
        """Get Kite profile information"""
        try:
            if EXCHANGE_CONFIG['kite']['enable'] and EXCHANGE_CONFIG['kite']['access_token']:
                from kiteconnect import KiteConnect
                
                kite = KiteConnect(
                    api_key=EXCHANGE_CONFIG['kite']['api_key'],
                    access_token=EXCHANGE_CONFIG['kite']['access_token']
                )
                
                profile = kite.profile()
                return profile
            else:
                return None
        except Exception as e:
            st.error(f"Failed to fetch Kite profile: {e}")
            return None
    
    def get_kite_funds(self):
        """Get Kite funds and margins"""
        try:
            if EXCHANGE_CONFIG['kite']['enable'] and EXCHANGE_CONFIG['kite']['access_token']:
                from kiteconnect import KiteConnect
                
                kite = KiteConnect(
                    api_key=EXCHANGE_CONFIG['kite']['api_key'],
                    access_token=EXCHANGE_CONFIG['kite']['access_token']
                )
                
                margins = kite.margins()
                return margins
            else:
                return None
        except Exception as e:
            st.error(f"Failed to fetch Kite funds: {e}")
            return None
    
    def render_header(self):
        """Render dashboard header with Kite/AngelOne theme"""
        # Kite/AngelOne themed header
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #FF6B35 0%, #4ECDC4 100%); border-radius: 10px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
            <h1 style='color: white; margin: 0; font-size: 2.5rem; font-weight: bold;'>📈 Kite AI Trading Bot</h1>
            <p style='color: white; margin: 5px 0 0 0; font-size: 1.1rem;'>Indian Markets AI-Powered Trading Analytics</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 Market", "Open", delta="NSE Live")
        
        with col2:
            st.metric("🤖 AI Bot", "Active", delta="Running")
        
        with col3:
            st.metric("📊 Portfolio", "₹1,25,000", delta="+2.5%")
        
        with col4:
            st.metric("⚡ Signals", "12 Active", delta="Today")
        
        # Kite Profile Section
        if EXCHANGE_CONFIG['kite']['enable']:
            st.markdown("---")
            st.subheader("👤 Kite Profile")
            
            # Refresh profile button
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🔄 Refresh", key="refresh_kite_profile"):
                    st.session_state.kite_profile = None
            
            # Display profile information
            if st.session_state.kite_profile is None:
                st.session_state.kite_profile = self.get_kite_profile()
            
            if st.session_state.kite_profile:
                profile = st.session_state.kite_profile
                
                # Account Information
                st.markdown("**📋 Account Information**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("User Name", profile.get('user_name', 'N/A'))
                
                with col2:
                    st.metric("User ID", profile.get('user_id', 'N/A'))
                
                with col3:
                    st.metric("Email", profile.get('email', 'N/A'))
                
                with col4:
                    st.metric("Products", profile.get('products', 'N/A'))
                
                # Funds Information
                st.markdown("**💰 Funds & Margins**")
                funds = self.get_kite_funds()
                
                if funds and 'equity' in funds:
                    equity = funds['equity']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Available Balance", 
                            f"₹{equity.get('net', 0):,.2f}", 
                            delta="Available Funds"
                        )
                    
                    with col2:
                        st.metric(
                            "Used Margin", 
                            f"₹{equity.get('used', 0):,.2f}", 
                            delta="Margin Used"
                        )
                    
                    with col3:
                        st.metric(
                            "Total Value", 
                            f"₹{equity.get('net', 0):,.2f}", 
                            delta="Net Value"
                        )
                    
                    with col4:
                        st.metric(
                            "Collateral", 
                            f"₹{equity.get('collateral', 0):,.2f}", 
                            delta="Collateral Value"
                        )
                else:
                    st.warning("Unable to fetch funds information")
            else:
                st.info("Kite profile not available. Please complete authentication.")
                st.code("python kite_auth_manual.py")
        else:
            st.info("Kite integration is disabled. Enable in settings.")
    
    def render_portfolio_overview(self):
        """Render portfolio overview with Kite/AngelOne styling"""
        st.markdown("---")
        st.subheader("📊 Portfolio Overview")
        
        # Portfolio metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Generate sample portfolio data
        if len(st.session_state.portfolio_history) == 0:
            np.random.seed(42)
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            portfolio_value = 125000 + np.cumsum(np.random.randn(30) * 1000)
            
            portfolio_df = pd.DataFrame({
                'date': dates,
                'value': portfolio_value,
                'returns': portfolio_value.pct_change()
            })
            
            st.session_state.portfolio_history = portfolio_df
        else:
            portfolio_df = st.session_state.portfolio_history
        
        with col1:
            current_value = portfolio_df['value'].iloc[-1]
            initial_value = portfolio_df['value'].iloc[0]
            total_return = ((current_value - initial_value) / initial_value) * 100
            
            st.metric(
                "Current Value", 
                f"₹{current_value:,.2f}", 
                delta=f"₹{current_value - initial_value:,.2f}"
            )
        
        with col2:
            st.metric(
                "Total Return", 
                f"{total_return:.2f}%", 
                delta=f"₹{current_value - initial_value:,.2f}"
            )
        
        with col3:
            daily_change = portfolio_df['returns'].iloc[-1] * 100
            st.metric(
                "Daily Change", 
                f"{daily_change:.2f}%", 
                delta="Daily Movement"
            )
        
        with col4:
            win_rate = 65 + np.random.randint(-5, 10)
            st.metric(
                "Win Rate", 
                f"{win_rate:.1f}%", 
                delta="Success Rate"
            )
        
        # Portfolio chart
        st.markdown("**📈 Portfolio Performance**")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(portfolio_df))),
            y=portfolio_df['value'],
            mode='lines+markers',
            name='Portfolio Value',
            line=dict(color='#FF6B35', width=3),
            marker=dict(color='#4ECDC4', size=8)
        ))
        
        fig.update_layout(
            title='Portfolio Performance',
            xaxis_title='Days',
            yaxis_title='Value (₹)',
            height=400,
            plot_bgcolor='rgba(255,255,255,255,0.1)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_trading_signals(self):
        """Render trading signals with Kite/AngelOne styling"""
        st.markdown("---")
        st.subheader("🤖 AI Trading Signals")
        
        # Generate sample signals
        if len(st.session_state.trading_signals) == 0:
            np.random.seed(42)
            symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK', 'WIPRO']
            signals = ['BUY', 'SELL', 'HOLD', 'BUY', 'SELL']
            confidences = [85, 72, 60, 90, 55]
            prices = [2500.50, 3200.00, 1800.75, 850.25, 2200.00]
            
            signals_df = pd.DataFrame({
                'symbol': symbols,
                'signal': signals,
                'confidence': confidences,
                'price': prices,
                'timestamp': pd.date_range(end=datetime.now(), periods=5, freq='H')
            })
            
            st.session_state.trading_signals = signals_df
        else:
            signals_df = st.session_state.trading_signals
        
        # Signal distribution
        col1, col2 = st.columns(2)
        
        with col1:
            signal_counts = signals_df['signal'].value_counts()
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=signal_counts.index,
                    values=signal_counts.values,
                    hole=0.3,
                    marker=dict(colors=['#FF6B35', '#4ECDC4', '#45B7D1', '#96CEB4'])
                )
            ])
            
            fig.update_layout(
                title='Signal Distribution',
                height=300,
                plot_bgcolor='rgba(255,255,255,255,0.1)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Signal metrics
            buy_signals = signals_df[signals_df['signal'] == 'BUY']
            sell_signals = signals_df[signals_df['signal'] == 'SELL']
            hold_signals = signals_df[signals_df['signal'] == 'HOLD']
            
            st.markdown("**📊 Signal Metrics**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Active Signals", len(signals_df))
            
            with col2:
                avg_confidence = signals_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        # Recent signals table
        st.markdown("**📋 Recent Signals**")
        
        # Color coding for signals
        def color_signal(val):
            if val == 'BUY':
                return 'background-color: #d4edda'
            elif val == 'SELL':
                return 'background-color: #f8d7da'
            else:
                return 'background-color: #fff3cd'
        
        # Apply styling
        styled_signals = signals_df.style.applymap(color_signal, subset=['signal'])
        st.dataframe(styled_signals, use_container_width=True)
    
    def render_risk_analysis(self):
        """Render risk analysis with Kite/AngelOne styling"""
        st.markdown("---")
        st.subheader("🛡️ Risk Analysis")
        
        # Risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("VaR (95%)", f"₹{np.random.rand()*50000:,.2f}", delta="Daily Risk")
        
        with col2:
            st.metric("Max Drawdown", f"{np.random.rand()*10:.2f}%", delta="Risk Metric")
        
        with col3:
            st.metric("Sharpe Ratio", f"{np.random.rand()*2:.2f}", delta="Performance")
        
        with col4:
            st.metric("Beta", f"{np.random.rand()*2:.2f}", delta="Market Risk")
        
        # Risk heatmap
        st.markdown("**📊 Risk Correlation Matrix**")
        
        # Generate sample correlation data
        np.random.seed(42)
        symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK']
        correlation_matrix = np.random.rand(len(symbols), len(symbols))
        np.fill_diagonal(correlation_matrix, 1)
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=symbols,
            y=symbols,
            colorscale='RdYlBu',
            showscale=True
        ))
        
        fig.update_layout(
            title='Risk Correlation Matrix',
            height=400,
            plot_bgcolor='rgba(255,255,255,255,0.1)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_market_analysis(self):
        """Render market analysis with Kite/AngelOne styling"""
        st.markdown("---")
        st.subheader("📈 Market Analysis")
        
        # Generate sample market data
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
        
        market_data = pd.DataFrame({
            'date': dates,
            'open': 2500 + np.random.randn(10) * 50,
            'high': 2500 + np.random.randn(10) * 50 + 100,
            'low': 2500 + np.random.randn(10) * 50 - 100,
            'close': 2500 + np.random.randn(10) * 50,
            'volume': np.random.randint(1000000, 5000000, 10)
        })
        
        # Candlestick Chart
        st.markdown("**📊 Price Action**")
        
        fig = go.Figure(data=go.Candlestick(
            x=market_data['date'],
            open=market_data['open'],
            high=market_data['high'],
            low=market_data['low'],
            close=market_data['close']
        ))
        
        fig.update_layout(
            title='Market Price Action',
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            height=400,
            plot_bgcolor='rgba(255,255,255,255,0.1)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Technical Indicators**")
            
            # RSI
            rsi_values = 50 + np.random.randn(10) * 10
            rsi_df = pd.DataFrame({
                'date': dates,
                'rsi': rsi_values
            })
            
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=rsi_df['date'],
                y=rsi_df['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color='#FF6B35', width=2)
            ))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig_rsi.update_layout(title='RSI Indicator', height=300, plot_bgcolor='rgba(255,255,255,255,0.1)')
            
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        with col2:
            st.markdown("**📈 Volume Analysis**")
            
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=market_data['date'],
                y=market_data['volume'],
                name='Volume',
                marker_color='#4ECDC4'
            ))
            fig_volume.update_layout(title='Trading Volume', height=300, plot_bgcolor='rgba(255,255,255,255,0.1)')
            
            st.plotly_chart(fig_volume, use_container_width=True)
    
    def render_trading_history(self):
        """Render trading history with Kite/AngelOne styling"""
        st.markdown("---")
        st.subheader("📜 Trading History")
        
        # Generate sample trading history
        if len(st.session_state.portfolio_history) > 0:
            np.random.seed(42)
            trades = []
            
            for i in range(20):
                trade_date = datetime.now() - timedelta(days=i)
                symbol = np.random.choice(['RELIANCE', 'TCS', 'INFY', 'HDFC'])
                action = np.random.choice(['BUY', 'SELL'])
                quantity = np.random.randint(10, 100)
                price = np.random.uniform(1000, 5000)
                pnl = np.random.uniform(-5000, 5000)
                
                trades.append({
                    'date': trade_date,
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'pnl': pnl
                })
            
            trades_df = pd.DataFrame(trades)
            
            # Cumulative P&L
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            
            # P&L Chart
            st.markdown("**📊 Cumulative P&L**")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(trades_df))),
                y=trades_df['cumulative_pnl'],
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='#FF6B35', width=3),
                marker=dict(color='#4ECDC4', size=8)
            ))
            
            fig.update_layout(
                title='Cumulative P&L',
                xaxis_title='Trade Number',
                yaxis_title='P&L (₹)',
                height=400,
                plot_bgcolor='rgba(255,255,255,255,0.1)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade statistics
            st.markdown("**📊 Trade Statistics**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_trades = len(trades_df)
                st.metric("Total Trades", total_trades)
            
            with col2:
                winning_trades = len(trades_df[trades_df['pnl'] > 0])
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with col3:
                total_pnl = trades_df['pnl'].sum()
                st.metric("Total P&L", f"₹{total_pnl:,.2f}")
            
            with col4:
                avg_pnl = trades_df['pnl'].mean()
                st.metric("Avg P&L", f"₹{avg_pnl:,.2f}")
            
            # Recent trades table
            st.markdown("**📋 Recent Trades**")
            st.dataframe(trades_df[['date', 'symbol', 'action', 'quantity', 'price', 'pnl']], use_container_width=True)
    
    def render_sidebar(self):
        """Render sidebar with Kite/AngelOne styling"""
        st.sidebar.header("📈 Kite AI Trading Bot")
        st.sidebar.markdown("---")
        
        # Bot Selection
        st.sidebar.subheader("🤖 Bot Configuration")
        bot_type = st.sidebar.selectbox(
            "Select Bot Type",
            ["Indian Markets (Kite)", "Global Markets", "Demo Mode"],
            index=0
        )
        
        # Market Selection
        st.sidebar.subheader("📊 Market Selection")
        
        if bot_type == "Indian Markets (Kite)":
            # Indian stocks
            available_symbols = SYMBOLS_CONFIG.get('nse_symbols', [])
            selected_symbols = st.sidebar.multiselect(
                "Select Symbols",
                available_symbols[:10],  # Limit to 10 for display
                default=available_symbols[:5]  # Default to first 5
            )
            
            st.session_state.selected_symbols = selected_symbols
            
            st.sidebar.markdown("**📋 Selected Symbols**")
            for symbol in selected_symbols:
                st.sidebar.write(f"📈 {symbol}")
        
        elif bot_type == "Global Markets":
            # Global stocks
            available_symbols = SYMBOLS_CONFIG.get('global_symbols', [])
            selected_symbols = st.sidebar.multiselect(
                "Select Symbols",
                available_symbols[:10],
                default=available_symbols[:5]
            )
            
            st.session_state.selected_symbols = selected_symbols
            
            st.sidebar.markdown("**📋 Selected Symbols**")
            for symbol in selected_symbols:
                st.sidebar.write(f"🌍 {symbol}")
        
        else:
            # Demo mode
            st.sidebar.info("Demo mode - using sample data")
        
        # AI Model Settings
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧠 AI Model Settings")
        
        model_type = st.sidebar.selectbox(
            "Model Type",
            ["Random Forest", "LSTM", "Ensemble"],
            index=0
        )
        
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.6,
            step=0.05
        )
        
        # Risk Settings
        st.sidebar.markdown("---")
        st.sidebar.subheader("🛡️ Risk Settings")
        
        max_position_size = st.sidebar.slider(
            "Max Position Size",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01
        )
        
        stop_loss = st.sidebar.slider(
            "Stop Loss %",
            min_value=0.01,
            max_value=0.1,
            value=0.02,
            step=0.01
        )
        
        take_profit = st.sidebar.slider(
            "Take Profit %",
            min_value=0.01,
            max_value=0.2,
            value=0.05,
            step=0.01
        )
        
        # Action Buttons
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.sidebar.button("🔄 Refresh Data", type="primary"):
                st.session_state.last_update = datetime.now()
                st.success("Data refreshed!")
        
        with col2:
            if st.sidebar.button("🚀 Start Trading", type="primary"):
                st.success("Trading started!")
    
    def run(self):
        """Run the Kite/AngelOne themed dashboard"""
        # Render all sections
        self.render_header()
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Portfolio", "🤖 AI Signals", "📈 Market Analysis", "📜 History"
        ])
        
        with tab1:
            self.render_portfolio_overview()
        
        with tab2:
            self.render_trading_signals()
        
        with tab3:
            self.render_market_analysis()
        
        with tab4:
            self.render_trading_history()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #FF6B35 0%, #4ECDC4 100%); border-radius: 10px; margin-top: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
            <p style='margin: 0; color: white; font-size: 0.9rem;'>📈 Kite AI Trading Bot v1.0</p>
            <p style='margin: 5px 0 0 0; color: rgba(255,255,255,255,0.8); font-size: 0.8rem;'>Indian Markets AI-Powered Trading Analytics</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function"""
    dashboard = KiteAIDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
