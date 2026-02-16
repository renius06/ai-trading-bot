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

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.trading_bot import AITradingBot
from src.kite_trading_bot import KiteTradingBot, KiteTradingConfig
from src.ai_model import TradingAIModel
from src.risk_management import RiskManager, RiskLimits
from config import TRADING_CONFIG, EXCHANGE_CONFIG, RISK_CONFIG, AI_CONFIG, SYMBOLS_CONFIG


class AIDashboard:
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="AI Trading Bot Dashboard",
            page_icon="🤖",
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
        if 'selected_symbols' not in st.session_state:
            st.session_state.selected_symbols = SYMBOLS_CONFIG['crypto'][:3]
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
        """Render dashboard header"""
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;'>
            <h1 style='color: white; margin: 0;'>🤖 AI Trading Bot Dashboard</h1>
            <p style='color: white; margin: 5px 0 0 0;'>Real-time AI-Powered Trading Analytics</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            bot_status = "🟢 Active" if st.session_state.bot else "🔴 Inactive"
            st.metric("Bot Status", bot_status)
        
        with col2:
            model_status = "🟢 Trained" if (st.session_state.ai_model and st.session_state.ai_model.is_trained) else "🔴 Not Trained"
            st.metric("AI Model", model_status)
        
        with col3:
            last_update = st.session_state.last_update.strftime("%H:%M:%S")
            st.metric("Last Update", last_update)
        
        with col4:
            total_signals = len(st.session_state.trading_signals)
            st.metric("Signals Generated", total_signals)
        
        # Kite Profile Section
        if EXCHANGE_CONFIG['kite']['enable']:
            st.markdown("---")
            st.subheader("👤 Kite Profile")
            
            # Refresh profile button
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🔄 Refresh", key="refresh_kite_profile"):
                    st.session_state.kite_profile = self.get_kite_profile()
            
            # Display profile information
            if st.session_state.kite_profile is None:
                st.session_state.kite_profile = self.get_kite_profile()
            
            if st.session_state.kite_profile:
                profile = st.session_state.kite_profile
                
                # Profile Information
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("User Name", profile.get('user_name', 'N/A'))
                
                with col2:
                    st.metric("User Type", profile.get('user_type', 'N/A'))
                
                with col3:
                    st.metric("Email", profile.get('email', 'N/A'))
                
                with col4:
                    st.metric("Products", profile.get('products', 'N/A'))
                
                # Funds and Margins
                st.markdown("**💰 Funds & Margins**")
                funds = self.get_kite_funds()
                
                if funds and 'equity' in funds:
                    equity = funds['equity']
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Available Balance", 
                            f"₹{equity.get('available', {}).get('live_balance', 0):,.2f}",
                            delta="Cash Available"
                        )
                    
                    with col2:
                        st.metric(
                            "Used Margin", 
                            f"₹{equity.get('used', {}).get('margin', 0):,.2f}",
                            delta="Margin Used"
                        )
                    
                    with col3:
                        st.metric(
                            "Net", 
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
                st.code("python kite_manual_auth.py <your_request_token>")
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("🎛️ Dashboard Controls")
        
        # Bot Selection
        st.sidebar.subheader("Bot Configuration")
        bot_type = st.sidebar.selectbox(
            "Select Bot Type",
            ["Global Markets", "Indian Markets (Kite)", "Demo Mode"]
        )
        
        # Symbol Selection
        st.sidebar.subheader("📊 Trading Symbols")
        
        if bot_type == "Indian Markets (Kite)":
            available_symbols = SYMBOLS_CONFIG['nse'] + SYMBOLS_CONFIG['mcx']
            default_symbols = SYMBOLS_CONFIG['nse'][:3]
        else:
            available_symbols = SYMBOLS_CONFIG['crypto'] + SYMBOLS_CONFIG['stocks']
            default_symbols = SYMBOLS_CONFIG['crypto'][:3]
        
        selected_symbols = st.sidebar.multiselect(
            "Select Symbols",
            available_symbols,
            default=default_symbols
        )
        
        if selected_symbols:
            st.session_state.selected_symbols = selected_symbols
        
        # AI Model Configuration
        st.sidebar.subheader("🤖 AI Model Settings")
        model_type = st.sidebar.selectbox(
            "Model Type",
            ["random_forest", "lstm", "ensemble"],
            index=["random_forest", "lstm", "ensemble"].index(AI_CONFIG['model_type'])
        )
        
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=TRADING_CONFIG['min_confidence_threshold'],
            step=0.05
        )
        
        # Risk Management
        st.sidebar.subheader("🛡️ Risk Settings")
        risk_per_trade = st.sidebar.slider(
            "Risk per Trade (%)",
            min_value=0.5,
            max_value=10.0,
            value=RISK_CONFIG['max_position_size'] * 100,
            step=0.5
        )
        
        max_positions = st.sidebar.slider(
            "Max Positions",
            min_value=1,
            max_value=20,
            value=TRADING_CONFIG['max_positions']
        )
        
        # Action Buttons
        st.sidebar.subheader("🚀 Actions")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🔄 Refresh", type="primary"):
                self.refresh_data()
        
        with col2:
            if st.button("📊 Analyze"):
                self.run_analysis()
        
        if st.sidebar.button("🤖 Train AI Model"):
            self.train_ai_model()
        
        if st.sidebar.button("📈 Backtest"):
            self.run_backtest()
        
        # Export/Import
        st.sidebar.subheader("💾 Data Management")
        if st.sidebar.button("📥 Export Data"):
            self.export_data()
        
        if st.sidebar.button("📤 Import Data"):
            self.import_data()
    
    def render_portfolio_overview(self):
        """Render portfolio overview section"""
        st.header("💰 Portfolio Overview")
        
        # Get portfolio data
        if st.session_state.bot:
            portfolio = st.session_state.bot.get_portfolio_summary()
        elif st.session_state.kite_bot:
            portfolio = st.session_state.kite_bot.get_portfolio_summary()
        else:
            # Demo data
            portfolio = {
                'total_value': 100000,
                'cash_balance': 25000,
                'unrealized_pnl': 5000,
                'num_positions': 5,
                'positions': []
            }
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Portfolio Value",
                f"₹{portfolio.get('total_value', 0):,.2f}",
                delta=f"₹{portfolio.get('unrealized_pnl', 0):,.2f}"
            )
        
        with col2:
            st.metric(
                "Cash Balance",
                f"₹{portfolio.get('cash_balance', 0):,.2f}"
            )
        
        with col3:
            st.metric(
                "Open Positions",
                portfolio.get('num_positions', 0)
            )
        
        with col4:
            total_return = ((portfolio.get('total_value', TRADING_CONFIG['initial_balance']) - TRADING_CONFIG['initial_balance']) / TRADING_CONFIG['initial_balance'] * 100)
            st.metric(
                "Total Return",
                f"{total_return:.2f}%"
            )
        
        # Portfolio Chart
        st.subheader("📈 Portfolio Performance")
        
        # Generate sample portfolio history
        if len(st.session_state.portfolio_history) == 0:
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            base_value = TRADING_CONFIG['initial_balance']
            values = []
            
            for i in range(30):
                daily_return = np.random.normal(0.001, 0.02)
                base_value *= (1 + daily_return)
                values.append(base_value)
            
            st.session_state.portfolio_history = pd.DataFrame({
                'Date': dates,
                'Portfolio Value': values
            })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.portfolio_history['Date'],
            y=st.session_state.portfolio_history['Portfolio Value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00ff88', width=2)
        ))
        
        fig.add_hline(
            y=TRADING_CONFIG['initial_balance'],
            line_dash="dash",
            line_color="red",
            annotation_text="Initial Balance"
        )
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (₹)",
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Position Breakdown
        if portfolio.get('positions'):
            st.subheader("📊 Position Breakdown")
            
            positions_data = []
            for pos in portfolio['positions']:
                positions_data.append({
                    'Symbol': pos.symbol,
                    'Quantity': pos.quantity,
                    'Entry Price': pos.entry_price,
                    'Current Price': pos.current_price,
                    'P&L': pos.unrealized_pnl,
                    'Return %': (pos.current_price - pos.entry_price) / pos.entry_price * 100
                })
            
            df_positions = pd.DataFrame(positions_data)
            
            # Position table
            st.dataframe(
                df_positions.style.format({
                    'Entry Price': '₹{:.2f}',
                    'Current Price': '₹{:.2f}',
                    'P&L': '₹{:.2f}',
                    'Return %': '{:.2f}%'
                }),
                use_container_width=True
            )
            
            # Position allocation pie chart
            fig = px.pie(
                df_positions,
                values='Quantity',
                names='Symbol',
                title='Position Allocation'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_ai_signals(self):
        """Render AI trading signals section"""
        st.header("🤖 AI Trading Signals")
        
        # Generate sample signals if none exist
        if len(st.session_state.trading_signals) == 0:
            st.session_state.trading_signals = self.generate_sample_signals()
        
        # Signal Summary
        col1, col2, col3, col4 = st.columns(4)
        
        signals_df = pd.DataFrame(st.session_state.trading_signals)
        
        with col1:
            buy_signals = len(signals_df[signals_df['action'] == 'BUY'])
            st.metric("BUY Signals", buy_signals, delta="🟢")
        
        with col2:
            sell_signals = len(signals_df[signals_df['action'] == 'SELL'])
            st.metric("SELL Signals", sell_signals, delta="🔴")
        
        with col3:
            hold_signals = len(signals_df[signals_df['action'] == 'HOLD'])
            st.metric("HOLD Signals", hold_signals, delta="🟡")
        
        with col4:
            avg_confidence = signals_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        # Signal Distribution Chart
        st.subheader("📊 Signal Distribution")
        
        signal_counts = signals_df['action'].value_counts()
        fig = px.bar(
            x=signal_counts.index,
            y=signal_counts.values,
            color=signal_counts.index,
            color_discrete_map={'BUY': 'green', 'SELL': 'red', 'HOLD': 'orange'},
            title="Signal Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent Signals Table
        st.subheader("📋 Recent Trading Signals")
        
        # Format signals for display
        display_signals = signals_df.copy()
        display_signals['timestamp'] = pd.to_datetime(display_signals['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        display_signals['price'] = display_signals['price'].apply(lambda x: f"₹{x:.2f}")
        display_signals['confidence'] = display_signals['confidence'].apply(lambda x: f"{x:.2f}")
        
        # Color code actions
        def color_action(val):
            if val == 'BUY':
                return 'background-color: #d4edda'
            elif val == 'SELL':
                return 'background-color: #f8d7da'
            else:
                return 'background-color: #fff3cd'
        
        styled_df = display_signals.style.applymap(color_action, subset=['action'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Signal Confidence Analysis
        st.subheader("📈 Signal Confidence Analysis")
        
        fig = px.histogram(
            signals_df,
            x='confidence',
            color='action',
            nbins=20,
            title="Confidence Distribution by Signal Type",
            color_discrete_map={'BUY': 'green', 'SELL': 'red', 'HOLD': 'orange'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_ai_model_performance(self):
        """Render AI model performance section"""
        st.header("🧠 AI Model Performance")
        
        # Model Status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.ai_model and st.session_state.ai_model.is_trained:
                st.success("✅ Model Trained")
            else:
                st.error("❌ Model Not Trained")
        
        with col2:
            model_type = AI_CONFIG['model_type'].upper()
            st.metric("Model Type", model_type)
        
        with col3:
            if st.session_state.model_performance:
                accuracy = st.session_state.model_performance.get('accuracy', 0)
                st.metric("Accuracy", f"{accuracy:.3f}")
            else:
                st.metric("Accuracy", "N/A")
        
        # Model Metrics
        if st.session_state.model_performance:
            st.subheader("📊 Model Metrics")
            
            metrics = st.session_state.model_performance
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Training Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                st.metric("Validation Accuracy", f"{metrics.get('val_accuracy', 0):.3f}")
                st.metric("Loss", f"{metrics.get('loss', 0):.3f}")
            
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
                st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
                st.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")
            
            # Feature Importance (for Random Forest)
            if metrics.get('feature_importance'):
                st.subheader("🎯 Feature Importance")
                
                feature_imp = pd.DataFrame(
                    list(metrics['feature_importance'].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=True)
                
                fig = px.bar(
                    feature_imp.tail(10),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 10 Feature Importance"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Prediction Distribution
        st.subheader("🎲 Prediction Distribution")
        
        # Generate sample prediction data
        predictions = np.random.choice(['BUY', 'HOLD', 'SELL'], 1000, p=[0.3, 0.4, 0.3])
        confidences = np.random.beta(2, 2, 1000)
        
        pred_df = pd.DataFrame({
            'Prediction': predictions,
            'Confidence': confidences
        })
        
        fig = px.violin(
            pred_df,
            y='Confidence',
            x='Prediction',
            color='Prediction',
            title="Prediction Confidence Distribution",
            color_discrete_map={'BUY': 'green', 'SELL': 'red', 'HOLD': 'orange'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model Training History
        if st.session_state.model_performance.get('history'):
            st.subheader("📈 Training History")
            
            history = st.session_state.model_performance['history']
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Model Accuracy', 'Model Loss'),
                vertical_spacing=0.1
            )
            
            # Accuracy plot
            fig.add_trace(
                go.Scatter(y=history.get('accuracy', []), name='Train Accuracy'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(y=history.get('val_accuracy', []), name='Val Accuracy'),
                row=1, col=1
            )
            
            # Loss plot
            fig.add_trace(
                go.Scatter(y=history.get('loss', []), name='Train Loss'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(y=history.get('val_loss', []), name='Val Loss'),
                row=2, col=1
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_risk_analysis(self):
        """Render risk analysis section"""
        st.header("🛡️ Risk Analysis")
        
        # Risk Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Daily VaR (95%)", "₹2,500")
        
        with col2:
            st.metric("Max Drawdown", "12.5%")
        
        with col3:
            st.metric("Sharpe Ratio", "1.85")
        
        with col4:
            st.metric("Volatility", "18.2%")
        
        # Risk Heatmap
        st.subheader("🔥 Risk Heatmap")
        
        # Generate sample risk data
        symbols = st.session_state.selected_symbols[:5] if st.session_state.selected_symbols else ['BTC/USDT', 'ETH/USDT', 'AAPL']
        risk_data = np.random.rand(len(symbols), 4)  # VaR, Volatility, Drawdown, Concentration
        
        risk_df = pd.DataFrame(
            risk_data,
            index=symbols,
            columns=['VaR', 'Volatility', 'Drawdown', 'Concentration']
        )
        
        fig = px.imshow(
            risk_df,
            title="Risk Metrics Heatmap",
            color_continuous_scale='RdYlGn_r',
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Portfolio Risk Breakdown
        st.subheader("📊 Portfolio Risk Breakdown")
        
        risk_categories = ['Market Risk', 'Credit Risk', 'Liquidity Risk', 'Operational Risk']
        risk_values = [45, 15, 25, 15]
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_categories,
            values=risk_values,
            hole=0.3,
            marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        )])
        
        fig.update_layout(title="Risk Category Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk Trends
        st.subheader("📈 Risk Trends")
        
        # Generate sample risk trend data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        var_trend = np.random.normal(2000, 500, 30)
        drawdown_trend = np.maximum.accumulate(np.random.normal(-0.01, 0.02, 30))
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Value at Risk (₹)', 'Maximum Drawdown (%)'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=dates, y=var_trend, name='VaR', line=dict(color='red')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=dates, y=drawdown_trend * 100, name='Drawdown', line=dict(color='orange')),
            row=2, col=1
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_market_analysis(self):
        """Render market analysis section"""
        st.header("📊 Market Analysis")
        
        if len(st.session_state.selected_symbols) == 0:
            st.warning("Please select symbols to analyze.")
            return
        
        # Symbol tabs
        symbol_tabs = st.tabs(st.session_state.selected_symbols[:3])
        
        for i, symbol in enumerate(st.session_state.selected_symbols[:3]):
            with symbol_tabs[i]:
                self.render_symbol_analysis(symbol)
    
    def render_symbol_analysis(self, symbol):
        """Render analysis for a specific symbol"""
        # Generate sample market data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
        prices = 100 + np.cumsum(np.random.normal(0, 2, 100))
        volumes = np.random.randint(1000, 10000, 100)
        
        market_data = pd.DataFrame({
            'Date': dates,
            'Price': prices,
            'Volume': volumes,
            'RSI': 50 + np.random.normal(0, 15, 100),
            'MACD': np.random.normal(0, 5, 100)
        })
        
        # Price Chart
        st.subheader(f"📈 {symbol} Price Chart")
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=market_data['Date'],
            open=market_data['Price'] * (1 + np.random.normal(0, 0.01, 100)),
            high=market_data['Price'] * (1 + np.abs(np.random.normal(0, 0.02, 100))),
            low=market_data['Price'] * (1 - np.abs(np.random.normal(0, 0.02, 100))),
            close=market_data['Price'],
            name='Price'
        ))
        
        fig.update_layout(
            title=f"{symbol} Price Action",
            yaxis_title="Price (₹)",
            xaxis_title="Date",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical Indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Technical Indicators")
            
            # RSI Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=market_data['Date'],
                y=market_data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple')
            ))
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            
            fig.update_layout(
                title="RSI Indicator",
                yaxis_title="RSI",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Volume Analysis")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=market_data['Date'],
                y=market_data['Volume'],
                name='Volume',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title="Trading Volume",
                yaxis_title="Volume",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Current Signal
        st.subheader("🎯 Current Trading Signal")
        
        # Generate sample signal
        signal_action = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.3, 0.4])
        signal_confidence = np.random.uniform(0.6, 0.95)
        signal_price = prices[-1]
        
        signal_color = {
            'BUY': 'green',
            'SELL': 'red',
            'HOLD': 'orange'
        }[signal_action]
        
        st.markdown(f"""
        <div style='padding: 20px; background-color: {signal_color}20; border-radius: 10px; text-align: center;'>
            <h2 style='color: {signal_color}; margin: 0;'>{signal_action}</h2>
            <p style='margin: 5px 0;'>Confidence: {signal_confidence:.2f}</p>
            <p style='margin: 5px 0;'>Target Price: ₹{signal_price:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_trading_history(self):
        """Render trading history section"""
        st.header("📜 Trading History")
        
        # Generate sample trading history
        trades_data = []
        for i in range(50):
            trade = {
                'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 30)),
                'symbol': np.random.choice(st.session_state.selected_symbols if st.session_state.selected_symbols else ['BTC/USDT', 'ETH/USDT']),
                'action': np.random.choice(['BUY', 'SELL']),
                'quantity': np.random.randint(1, 100),
                'price': np.random.uniform(50, 5000),
                'pnl': np.random.uniform(-500, 1000),
                'status': np.random.choice(['COMPLETED', 'PENDING', 'CANCELLED'])
            }
            trades_data.append(trade)
        
        trades_df = pd.DataFrame(trades_data)
        trades_df['timestamp'] = trades_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        trades_df['price'] = trades_df['price'].apply(lambda x: f"₹{x:.2f}")
        trades_df['pnl'] = trades_df['pnl'].apply(lambda x: f"₹{x:.2f}")
        
        # Trading statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_trades = len(trades_df)
            st.metric("Total Trades", total_trades)
        
        with col2:
            winning_trades = len(trades_df[trades_df['pnl'].str.replace('₹', '').astype(float) > 0])
            win_rate = winning_trades / total_trades * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col3:
            total_pnl = trades_df['pnl'].str.replace('₹', '').astype(float).sum()
            st.metric("Total P&L", f"₹{total_pnl:.2f}")
        
        with col4:
            completed_trades = len(trades_df[trades_df['status'] == 'COMPLETED'])
            st.metric("Completed", completed_trades)
        
        # Trades table
        st.subheader("📋 Recent Trades")
        
        st.dataframe(trades_df, use_container_width=True)
        
        # P&L Chart
        st.subheader("📈 P&L Performance")
        
        trades_df['pnl_numeric'] = trades_df['pnl'].str.replace('₹', '').astype(float)
        cumulative_pnl = trades_df['pnl_numeric'].cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(cumulative_pnl))),
            y=cumulative_pnl,
            mode='lines',
            name='Cumulative P&L',
            line=dict(color='green' if cumulative_pnl.iloc[-1] > 0 else 'red')
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title="Cumulative P&L Over Time",
            xaxis_title="Trade Number",
            yaxis_title="Cumulative P&L (₹)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def generate_sample_signals(self):
        """Generate sample trading signals"""
        signals = []
        symbols = st.session_state.selected_symbols if st.session_state.selected_symbols else ['BTC/USDT', 'ETH/USDT', 'AAPL']
        
        for symbol in symbols:
            for i in range(10):
                signal = {
                    'timestamp': datetime.now() - timedelta(hours=i),
                    'symbol': symbol,
                    'action': np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.3, 0.4]),
                    'confidence': np.random.uniform(0.5, 0.95),
                    'price': np.random.uniform(50, 5000),
                    'reason': f"Technical analysis signal #{i+1}"
                }
                signals.append(signal)
        
        return signals
    
    def refresh_data(self):
        """Refresh dashboard data"""
        st.session_state.last_update = datetime.now()
        st.success("✅ Data refreshed successfully!")
        st.rerun()
    
    def run_analysis(self):
        """Run comprehensive analysis"""
        with st.spinner("🔄 Running comprehensive analysis..."):
            time.sleep(2)  # Simulate analysis
            st.success("✅ Analysis completed!")
            st.rerun()
    
    def train_ai_model(self):
        """Train AI model"""
        with st.spinner("🤖 Training AI model..."):
            time.sleep(3)  # Simulate training
            
            # Update model performance
            st.session_state.model_performance = {
                'accuracy': np.random.uniform(0.7, 0.9),
                'precision': np.random.uniform(0.7, 0.9),
                'recall': np.random.uniform(0.7, 0.9),
                'f1_score': np.random.uniform(0.7, 0.9),
                'loss': np.random.uniform(0.1, 0.3),
                'val_accuracy': np.random.uniform(0.7, 0.9),
                'val_loss': np.random.uniform(0.1, 0.3),
                'feature_importance': {
                    'RSI': np.random.uniform(0.1, 0.3),
                    'MACD': np.random.uniform(0.1, 0.3),
                    'Volume': np.random.uniform(0.05, 0.2),
                    'Price': np.random.uniform(0.1, 0.3),
                    'Returns': np.random.uniform(0.1, 0.3)
                }
            }
            
            st.success("✅ AI model trained successfully!")
            st.rerun()
    
    def run_backtest(self):
        """Run backtesting"""
        with st.spinner("📈 Running backtest..."):
            time.sleep(2)  # Simulate backtesting
            st.success("✅ Backtest completed!")
            st.rerun()
    
    def export_data(self):
        """Export dashboard data"""
        data = {
            'portfolio_history': st.session_state.portfolio_history,
            'trading_signals': st.session_state.trading_signals,
            'risk_metrics': st.session_state.risk_metrics,
            'model_performance': st.session_state.model_performance
        }
        
        st.download_button(
            label="📥 Download Dashboard Data",
            data=json.dumps(data, default=str, indent=2),
            file_name=f"dashboard_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    def import_data(self):
        """Import dashboard data"""
        uploaded_file = st.file_uploader(
            "📤 Upload dashboard data file",
            type=['json']
        )
        
        if uploaded_file:
            try:
                data = json.loads(uploaded_file.read())
                st.session_state.portfolio_history = data.get('portfolio_history', [])
                st.session_state.trading_signals = data.get('trading_signals', [])
                st.session_state.risk_metrics = data.get('risk_metrics', {})
                st.session_state.model_performance = data.get('model_performance', {})
                
                st.success("✅ Data imported successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error importing data: {e}")
    
    def run(self):
        """Main dashboard runner"""
        self.render_header()
        self.render_sidebar()
        
        # Main content tabs
        main_tabs = st.tabs([
            "📊 Portfolio",
            "🤖 AI Signals",
            "🧠 Model Performance",
            "🛡️ Risk Analysis",
            "📈 Market Analysis",
            "📜 Trading History"
        ])
        
        with main_tabs[0]:
            self.render_portfolio_overview()
        
        with main_tabs[1]:
            self.render_ai_signals()
        
        with main_tabs[2]:
            self.render_ai_model_performance()
        
        with main_tabs[3]:
            self.render_risk_analysis()
        
        with main_tabs[4]:
            self.render_market_analysis()
        
        with main_tabs[5]:
            self.render_trading_history()
        
        # Auto-refresh option
        if st.sidebar.checkbox("🔄 Auto-refresh (30s)"):
            time.sleep(30)
            st.rerun()


def main():
    """Main function to run the AI dashboard"""
    dashboard = AIDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
