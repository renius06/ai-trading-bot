import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.trading_bot import AITradingBot
from src.risk_management import RiskManager, RiskLimits
from config import TRADING_CONFIG, RISK_CONFIG, SYMBOLS_CONFIG


def create_dashboard():
    """Create Streamlit dashboard for trading bot monitoring"""
    st.set_page_config(
        page_title="AI Trading Bot Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 AI Trading Bot Dashboard")
    
    # Sidebar
    st.sidebar.header("Bot Configuration")
    
    # Initialize session state
    if 'bot' not in st.session_state:
        st.session_state.bot = None
    if 'selected_symbols' not in st.session_state:
        st.session_state.selected_symbols = SYMBOLS_CONFIG['crypto'][:3]
    
    # Symbol selection
    available_symbols = SYMBOLS_CONFIG['crypto'] + SYMBOLS_CONFIG['stocks']
    selected_symbols = st.sidebar.multiselect(
        "Select Trading Symbols",
        available_symbols,
        default=st.session_state.selected_symbols
    )
    
    if selected_symbols:
        st.session_state.selected_symbols = selected_symbols
    
    # Bot controls
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🚀 Start Bot", type="primary"):
            initialize_bot()
    
    with col2:
        if st.button("⏹️ Stop Bot"):
            st.session_state.bot = None
            st.rerun()
    
    # Main content
    if st.session_state.bot:
        display_bot_dashboard()
    else:
        display_welcome_screen()


def initialize_bot():
    """Initialize the trading bot"""
    try:
        # Create risk manager
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
        
        # Create bot configuration
        bot_config = {
            'initial_balance': TRADING_CONFIG['initial_balance'],
            'risk_per_trade': TRADING_CONFIG['risk_per_trade'],
            'max_positions': TRADING_CONFIG['max_positions'],
            'risk_manager': risk_manager
        }
        
        st.session_state.bot = AITradingBot(bot_config)
        st.success("✅ Trading Bot initialized successfully!")
        
    except Exception as e:
        st.error(f"❌ Failed to initialize bot: {e}")


def display_welcome_screen():
    """Display welcome screen when bot is not running"""
    st.markdown("""
    ## 🎯 Welcome to AI Trading Bot
    
    This intelligent trading bot combines technical analysis with machine learning to make automated trading decisions.
    
    ### Features:
    - 🤖 **AI-Powered Signals**: Machine learning models for market prediction
    - 🛡️ **Risk Management**: Advanced position sizing and risk controls
    - 💱 **Multi-Asset Support**: Crypto, stocks, forex, and commodities
    - 📊 **Real-time Monitoring**: Live portfolio tracking and analytics
    
    ### Getting Started:
    1. **Select symbols** from the sidebar
    2. **Click "Start Bot"** to initialize
    3. **Monitor performance** in real-time
    
    ### Configuration:
    - Edit `config.py` for advanced settings
    - Set up API keys in `.env` for live trading
    - Adjust risk parameters as needed
    """)
    
    # Sample market data display
    st.subheader("📈 Sample Market Data")
    
    if st.session_state.selected_symbols:
        cols = st.columns(len(st.session_state.selected_symbols))
        
        for i, symbol in enumerate(st.session_state.selected_symbols):
            with cols[i]:
                try:
                    # Create temporary bot for data fetching
                    temp_bot = AITradingBot({'initial_balance': 10000})
                    df = temp_bot.get_market_data(symbol, limit=50)
                    
                    if not df.empty:
                        df = temp_bot.calculate_technical_indicators(df)
                        
                        latest_price = df['close'].iloc[-1]
                        price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
                        latest_rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 0
                        
                        st.metric(
                            label=symbol,
                            value=f"${latest_price:.2f}",
                            delta=f"{price_change:.2f}%"
                        )
                        
                        st.caption(f"RSI: {latest_rsi:.1f}")
                except Exception as e:
                    st.error(f"Error loading {symbol}: {e}")


def display_bot_dashboard():
    """Display main dashboard when bot is running"""
    bot = st.session_state.bot
    
    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.rerun()
    
    # Portfolio Summary
    st.header("💰 Portfolio Summary")
    
    portfolio = bot.get_portfolio_summary()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Value",
            f"${portfolio['total_value']:.2f}",
            delta=f"${portfolio['total_value'] - TRADING_CONFIG['initial_balance']:.2f}"
        )
    
    with col2:
        st.metric(
            "Cash Balance",
            f"${portfolio['cash_balance']:.2f}"
        )
    
    with col3:
        st.metric(
            "Unrealized P&L",
            f"${portfolio['unrealized_pnl']:.2f}",
            delta=f"{(portfolio['unrealized_pnl'] / portfolio['total_value'] * 100):.2f}%"
        )
    
    with col4:
        st.metric(
            "Open Positions",
            portfolio['num_positions']
        )
    
    # Charts section
    st.header("📊 Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Performance", "Positions", "Market Analysis", "Risk Metrics"])
    
    with tab1:
        display_portfolio_performance(bot)
    
    with tab2:
        display_positions(bot, portfolio)
    
    with tab3:
        display_market_analysis(bot)
    
    with tab4:
        display_risk_metrics(bot, portfolio)


def display_portfolio_performance(bot):
    """Display portfolio performance chart"""
    # Generate sample performance data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    values = []
    base_value = TRADING_CONFIG['initial_balance']
    
    for i in range(30):
        # Simulate portfolio growth with volatility
        daily_return = np.random.normal(0.001, 0.02)  # 0.1% daily return, 2% volatility
        base_value *= (1 + daily_return)
        values.append(base_value)
    
    performance_df = pd.DataFrame({
        'Date': dates,
        'Portfolio Value': values
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=performance_df['Date'],
        y=performance_df['Portfolio Value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_hline(
        y=TRADING_CONFIG['initial_balance'],
        line_dash="dash",
        line_color="red",
        annotation_text="Initial Balance"
    )
    
    fig.update_layout(
        title="Portfolio Performance Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_return = (values[-1] - TRADING_CONFIG['initial_balance']) / TRADING_CONFIG['initial_balance'] * 100
        st.metric("Total Return", f"{total_return:.2f}%")
    
    with col2:
        daily_returns = np.diff(values) / values[:-1]
        volatility = np.std(daily_returns) * np.sqrt(252) * 100
        st.metric("Annual Volatility", f"{volatility:.2f}%")
    
    with col3:
        sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252) if np.std(daily_returns) != 0 else 0
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")


def display_positions(bot, portfolio):
    """Display current positions"""
    if not portfolio['positions']:
        st.info("No open positions")
        return
    
    positions_data = []
    for pos in portfolio['positions']:
        positions_data.append({
            'Symbol': pos.symbol,
            'Quantity': pos.quantity,
            'Entry Price': pos.entry_price,
            'Current Price': pos.current_price,
            'Unrealized P&L': pos.unrealized_pnl,
            'Return %': (pos.current_price - pos.entry_price) / pos.entry_price * 100,
            'Entry Time': pos.entry_time.strftime('%Y-%m-%d %H:%M')
        })
    
    df = pd.DataFrame(positions_data)
    
    # Display table
    st.dataframe(df, use_container_width=True)
    
    # Position breakdown chart
    fig = px.pie(
        df,
        values='Quantity',
        names='Symbol',
        title='Position Allocation'
    )
    st.plotly_chart(fig, use_container_width=True)


def display_market_analysis(bot):
    """Display market analysis for selected symbols"""
    if not st.session_state.selected_symbols:
        st.info("No symbols selected")
        return
    
    for symbol in st.session_state.selected_symbols:
        st.subheader(f"📈 {symbol} Analysis")
        
        try:
            # Get market data
            df = bot.get_market_data(symbol, limit=100)
            if df.empty:
                st.warning(f"No data available for {symbol}")
                continue
            
            df = bot.calculate_technical_indicators(df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Price chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'] if 'Open' in df.columns else df['open'],
                    high=df['High'] if 'High' in df.columns else df['high'],
                    low=df['Low'] if 'Low' in df.columns else df['low'],
                    close=df['Close'] if 'Close' in df.columns else df['close'],
                    name='Price'
                ))
                
                if 'sma_20' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['sma_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='orange')
                    ))
                
                fig.update_layout(
                    title=f"{symbol} Price Chart",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Technical indicators
                latest = df.iloc[-1]
                
                st.metric("Current Price", f"${latest['Close'] if 'Close' in latest else latest['close']:.2f}")
                
                if 'rsi' in df.columns:
                    st.metric("RSI", f"{latest['rsi']:.1f}")
                
                if 'macd' in df.columns:
                    st.metric("MACD", f"{latest['macd']:.4f}")
                
                # Generate signal
                signal = bot.generate_trading_signals(symbol)
                
                signal_color = {
                    'BUY': 'green',
                    'SELL': 'red',
                    'HOLD': 'orange'
                }.get(signal.action, 'gray')
                
                st.markdown(f"### Signal: <span style='color: {signal_color}'>{signal.action}</span>", 
                           unsafe_allow_html=True)
                st.caption(f"Confidence: {signal.confidence:.2f}")
                st.caption(signal.reason)
        
        except Exception as e:
            st.error(f"Error analyzing {symbol}: {e}")


def display_risk_metrics(bot, portfolio):
    """Display risk management metrics"""
    risk_manager = bot.risk_manager
    
    # Update portfolio history
    risk_manager.update_portfolio_history(portfolio['total_value'])
    
    # Calculate risk metrics
    try:
        risk_metrics = risk_manager.calculate_risk_metrics(
            portfolio['total_value'],
            {pos.symbol: {'quantity': pos.quantity, 'current_price': pos.current_price} 
             for pos in portfolio['positions']}
        )
        
        # Risk metrics display
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Daily VaR (95%)", f"{risk_metrics.daily_var:.2%}")
            st.metric("Max Drawdown", f"{risk_metrics.max_drawdown:.2%}")
            st.metric("Sharpe Ratio", f"{risk_metrics.sharpe_ratio:.2f}")
        
        with col2:
            st.metric("Annual Volatility", f"{risk_metrics.volatility:.2%}")
            st.metric("Beta", f"{risk_metrics.beta:.2f}")
            st.metric("Max Concentration", f"{risk_metrics.position_concentration:.2%}")
        
        # Risk limits check
        within_limits, violations = risk_manager.check_risk_limits(
            portfolio['total_value'],
            {pos.symbol: {'quantity': pos.quantity, 'current_price': pos.current_price} 
             for pos in portfolio['positions']}
        )
        
        if within_limits:
            st.success("✅ All risk limits within bounds")
        else:
            st.error(f"⚠️ Risk limit violations: {', '.join(violations)}")
        
        # Risk report
        if st.button("📋 Generate Risk Report"):
            report = risk_manager.generate_risk_report(
                portfolio['total_value'],
                {pos.symbol: {'quantity': pos.quantity, 'current_price': pos.current_price} 
                 for pos in portfolio['positions']}
            )
            
            st.json(report)
    
    except Exception as e:
        st.error(f"Error calculating risk metrics: {e}")


if __name__ == "__main__":
    create_dashboard()
