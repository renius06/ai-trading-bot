#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MarketDataFetcher:
    """Comprehensive market data fetcher for multiple APIs"""
    
    def __init__(self):
        self.api_key = '1wupbdnax2j9quy1'
        self.api_secret = 'b5mgbdupm9votducd1kijvsrg6jekonj'
        self.access_token = os.getenv('KITE_ACCESS_TOKEN', '')
        
    def fetch_kite_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from Kite API"""
        try:
            if not self.access_token:
                st.warning("Kite API not authenticated. Please complete authentication first.")
                return None
            
            from kiteconnect import KiteConnect
            kite = KiteConnect(
                api_key=self.api_key,
                access_token=self.access_token
            )
            
            # Fetch historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            historical_data = kite.historical_data(
                instrument_token=symbol,
                from_date=start_date,
                to_date=end_date,
                interval="day"
            )
            
            # Fetch current quote
            quote = kite.quote([symbol])
            
            # Fetch margins
            margins = kite.margins()
            
            return {
                'historical': historical_data,
                'quote': quote,
                'margins': margins,
                'source': 'kite',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            st.error(f"Kite API error: {e}")
            return None
    
    def fetch_angelone_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from AngelOne API"""
        try:
            # AngelOne API implementation (placeholder)
            # Note: You'll need to implement AngelOne API integration
            
            # Sample data for demonstration
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # Generate sample data
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            prices = np.random.uniform(1000, 5000, len(dates))
            
            return {
                'historical': pd.DataFrame({
                    'date': dates,
                    'open': prices + np.random.randn(len(dates)) * 10,
                    'high': prices + np.random.randn(len(dates)) * 20 + 100,
                    'low': prices + np.random.randn(len(dates)) * 20 - 100,
                    'close': prices + np.random.randn(len(dates)) * 10
                }),
                'quote': {
                    'price': prices[-1] if len(prices) > 0 else 1000,
                    'change': np.random.uniform(-5, 5)
                },
                'source': 'angelone',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            st.error(f"AngelOne API error: {e}")
            return None
    
    def fetch_yahoo_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from Yahoo Finance API"""
        try:
            # Fetch historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo", interval="1d")
            
            # Fetch current quote
            current = ticker.info()
            
            return {
                'historical': hist.reset_index().rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'Volume'
                }),
                'quote': {
                    'price': current.get('regularMarketPrice', 0),
                    'change': current.get('regularMarketChangePercent', 0),
                    'volume': current.get('averageVolume', 0)
                },
                'source': 'yahoo',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            st.error(f"Yahoo Finance API error: {e}")
            return None
    
    def fetch_alpha_vantage_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from Alpha Vantage API"""
        try:
            # Alpha Vantage API implementation (placeholder)
            api_key = os.getenv('ALPHA_VANTAGE_API_KEY', '')
            
            if not api_key:
                st.warning("Alpha Vantage API key not configured.")
                return None
            
            # Sample implementation
            return {
                'historical': pd.DataFrame(),
                'quote': {'price': 0, 'change': 0},
                'source': 'alpha_vantage',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            st.error(f"Alpha Vantage API error: {e}")
            return None
    
    def fetch_polygon_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from Polygon.io API"""
        try:
            # Polygon.io API implementation (placeholder)
            api_key = os.getenv('POLYGON_API_KEY', '')
            
            if not api_key:
                st.warning("Polygon.io API key not configured.")
                return None
            
            # Sample implementation
            return {
                'historical': pd.DataFrame(),
                'quote': {'price': 0, 'change': 0},
                'source': 'polygon',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            st.error(f"Polygon.io API error: {e}")
            return None
    
    def fetch_iex_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from IEX Cloud API"""
        try:
            # IEX Cloud API implementation (placeholder)
            api_key = os.getenv('IEX_API_KEY', '')
            
            if not api_key:
                st.warning("IEX Cloud API key not configured.")
                return None
            
            # Sample implementation
            return {
                'historical': pd.DataFrame(),
                'quote': {'price': 0, 'change': 0},
                'source': 'iex',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            st.error(f"IEX Cloud API error: {e}")
            return None
    
    def fetch_financial_model_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from financial modeling APIs"""
        try:
            # Financial modeling API implementation (placeholder)
            return {
                'historical': pd.DataFrame(),
                'quote': {'price': 0, 'change': 0},
                'source': 'financial_model',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            st.error(f"Financial model API error: {e}")
            return None
    
    def fetch_news_data(self, symbol: str) -> Optional[Dict]:
        """Fetch news data for the symbol"""
        try:
            # News API implementation (placeholder)
            return {
                'news': pd.DataFrame({
                    'title': ['Sample News 1', 'Sample News 2'],
                    'source': ['NewsAPI', 'NewsAPI'],
                    'date': [datetime.now(), datetime.now() - timedelta(days=1)],
                    'sentiment': ['positive', 'neutral']
                }),
                'source': 'news_api',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            st.error(f"News API error: {e}")
            return None
    
    def fetch_social_sentiment_data(self, symbol: str) -> Optional[Dict]:
        """Fetch social sentiment data for the symbol"""
        try:
            # Social sentiment API implementation (placeholder)
            return {
                'sentiment': {
                    'twitter': 0.5,
                    'reddit': 0.3,
                    'stocktwits': 0.7
                },
                'source': 'social_sentiment',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            st.error(f"Social sentiment API error: {e}")
            return None
    
    def fetch_economic_data(self) -> Optional[Dict]:
        """Fetch economic data"""
        try:
            # Economic data API implementation (placeholder)
            return {
                'indicators': {
                    'gdp': 2.5,
                    'inflation': 1.8,
                    'unemployment': 3.2,
                    'interest_rates': 0.25
                },
                'source': 'economic_api',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            st.error(f"Economic data API error: {e}")
            return None
    
    def fetch_technical_indicators(self, symbol: str, data: Dict) -> Dict:
        """Calculate technical indicators from price data"""
        try:
            if not data or 'historical' not in data:
                return {}
            
            df = data['historical']
            if len(df) < 20:
                return {}
            
            # Calculate technical indicators
            close_prices = df['close'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            volumes = df['volume'].values
            
            # RSI (Relative Strength Index)
            deltas = np.diff(close_prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            rs = 100 - (100 / (1 + (avg_loss / avg_gain))) if avg_gain != 0 else 0
            rsi_values = [rs] * len(close_prices)
            
            # MACD (Moving Average Convergence Divergence)
            exp1 = close_prices.ewm(span=12, adjust=False).mean()
            exp2 = close_prices.ewm(span=26, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            sma20 = close_prices.rolling(window=20).mean()
            std20 = close_prices.rolling(window=20).std()
            upper_band = sma20 + (std20 * 2)
            lower_band = sma20 - (std20 * 2)
            
            # Moving Averages
            sma5 = close_prices.rolling(window=5).mean()
            sma10 = close_prices.rolling(window=10).mean()
            sma20 = close_prices.rolling(window=20).mean()
            sma50 = close_prices.rolling(window=50).mean()
            
            return {
                'rsi': rsi_values[-1] if len(rsi_values) > 0 else 50,
                'macd': {
                    'macd': macd_line[-1] if len(macd_line) > 0 else 0,
                    'signal': signal_line[-1] if len(signal_line) > 0 else 0
                },
                'bollinger': {
                    'upper': upper_band[-1] if len(upper_band) > 0 else 0,
                    'middle': sma20[-1] if len(sma20) > 0 else 0,
                    'lower': lower_band[-1] if len(lower_band) > 0 else 0
                },
                'moving_averages': {
                    'sma5': sma5[-1] if len(sma5) > 0 else 0,
                    'sma10': sma10[-1] if len(sma10) > 0 else 0,
                    'sma20': sma20[-1] if len(sma20) > 0 else 0,
                    'sma50': sma50[-1] if len(sma50) > 0 else 0
                },
                'source': 'technical_analysis',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            st.error(f"Technical analysis error: {e}")
            return {}
    
    def fetch_all_data(self, symbol: str) -> Dict:
        """Fetch data from all available sources"""
        all_data = {}
        
        # Fetch from Kite API
        kite_data = self.fetch_kite_data(symbol)
        if kite_data:
            all_data['kite'] = kite_data
        
        # Fetch from AngelOne API
        angelone_data = self.fetch_angelone_data(symbol)
        if angelone_data:
            all_data['angelone'] = angelone_data
        
        # Fetch from Yahoo Finance
        yahoo_data = self.fetch_yahoo_data(symbol)
        if yahoo_data:
            all_data['yahoo'] = yahoo_data
        
        # Fetch from Alpha Vantage
        alpha_data = self.fetch_alpha_vantage_data(symbol)
        if alpha_data:
            all_data['alpha_vantage'] = alpha_data
        
        # Fetch from Polygon.io
        polygon_data = self.fetch_polygon_data(symbol)
        if polygon_data:
            all_data['polygon'] = polygon_data
        
        # Fetch from IEX Cloud
        iex_data = self.fetch_iex_data(symbol)
        if iex_data:
            all_data['iex'] = iex_data
        
        # Fetch news data
        news_data = self.fetch_news_data(symbol)
        if news_data:
            all_data['news'] = news_data
        
        # Fetch social sentiment
        sentiment_data = self.fetch_social_sentiment_data(symbol)
        if sentiment_data:
            all_data['social_sentiment'] = sentiment_data
        
        # Fetch economic data
        economic_data = self.fetch_economic_data()
        if economic_data:
            all_data['economic'] = economic_data
        
        # Calculate technical indicators
        if all_data:
            technical_data = self.fetch_technical_indicators(symbol, all_data)
            all_data['technical'] = technical_data
        
        return all_data

def create_streamlit_interface():
    """Create Streamlit interface for data fetching"""
    st.set_page_config(
        page_title="📊 AI Data Fetcher",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📊 Comprehensive Market Data Fetcher")
    st.markdown("---")
    
    # Initialize fetcher
    fetcher = MarketDataFetcher()
    
    # Sidebar configuration
    st.sidebar.header("🔧 Data Source Configuration")
    
    # API Keys
    st.sidebar.subheader("🔑 API Keys")
    
    kite_api_key = st.sidebar.text_input("Kite API Key", value=fetcher.api_key, type="password")
    kite_api_secret = st.sidebar.text_input("Kite API Secret", value=fetcher.api_secret, type="password")
    
    alpha_vantage_key = st.sidebar.text_input("Alpha Vantage Key", value=os.getenv('ALPHA_VANTAGE_API_KEY', ''), type="password")
    polygon_key = st.sidebar.text_input("Polygon.io Key", value=os.getenv('POLYGON_API_KEY', ''), type="password")
    iex_key = st.sidebar.text_input("IEX Cloud Key", value=os.getenv('IEX_API_KEY', ''), type="password")
    
    # Data sources selection
    st.sidebar.subheader("📊 Data Sources")
    
    data_sources = st.sidebar.multiselect(
        "Select Data Sources",
        ["Kite API", "AngelOne API", "Yahoo Finance", "Alpha Vantage", "Polygon.io", "IEX Cloud", "News API", "Social Sentiment", "Economic Data"],
        default=["Kite API", "Yahoo Finance"]
    )
    
    # Symbol input
    st.sidebar.subheader("📈 Symbol Selection")
    
    symbols = st.sidebar.text_input("Enter Symbols (comma-separated)", value="RELIANCE,TCS,INFY")
    
    # Time period selection
    st.sidebar.subheader("📅 Time Period")
    
    period_options = {
        "1 Week": 7,
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365
    }
    
    selected_period = st.sidebar.selectbox(
        "Select Period",
        options=list(period_options.keys()),
        format_func=lambda x: period_options[x]
    )
    
    # Fetch data button
    if st.sidebar.button("🚀 Fetch Data", type="primary"):
        if symbols:
            symbol_list = [s.strip() for s in symbols.split(',')]
            
            with st.spinner("Fetching data..."):
                all_results = {}
                
                for symbol in symbol_list:
                    st.write(f"📊 Fetching data for {symbol}...")
                    
                    symbol_data = {}
                    
                    for source in data_sources:
                        if source == "Kite API":
                            data = fetcher.fetch_kite_data(symbol)
                        elif source == "AngelOne API":
                            data = fetcher.fetch_angelone_data(symbol)
                        elif source == "Yahoo Finance":
                            data = fetcher.fetch_yahoo_data(symbol)
                        elif source == "Alpha Vantage":
                            data = fetcher.fetch_alpha_vantage_data(symbol)
                        elif source == "Polygon.io":
                            data = fetcher.fetch_polygon_data(symbol)
                        elif source == "IEX Cloud":
                            data = fetcher.fetch_iex_data(symbol)
                        elif source == "News API":
                            data = fetcher.fetch_news_data(symbol)
                        elif source == "Social Sentiment":
                            data = fetcher.fetch_social_sentiment_data(symbol)
                        elif source == "Economic Data":
                            data = fetcher.fetch_economic_data()
                        
                        if data:
                            symbol_data[source] = data
                    
                    all_results[symbol] = symbol_data
                
                # Display results
                st.success(f"✅ Data fetched for {len(symbol_list)} symbols!")
                
                for symbol, symbol_data in all_results.items():
                    st.markdown(f"---")
                    st.markdown(f"## 📈 {symbol}")
                    
                    # Display data from each source
                    for source, data in symbol_data.items():
                        if data:
                            st.markdown(f"### 🔑 {source}")
                            
                            if 'historical' in data:
                                st.dataframe(data['historical'].tail(10), use_container_width=True)
                            
                            if 'quote' in data:
                                quote = data['quote']
                                st.metric("Current Price", f"₹{quote.get('price', 0):,.2f}")
                                if 'change' in quote:
                                    st.metric("Change", f"{quote.get('change', 0):.2f}%")
                            
                            if 'technical' in data:
                                tech = data['technical']
                                
                                st.markdown("**📊 Technical Indicators**")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("RSI", f"{tech.get('rsi', 50):.1f}")
                                
                                with col2:
                                    macd = tech.get('macd', {})
                                    st.metric("MACD", f"{macd.get('macd', 0):.4f}")
                                
                                with col3:
                                    bollinger = tech.get('bollinger', {})
                                    st.metric("BB Position", f"{(bollinger.get('middle', 0) / (bollinger.get('upper', 0) - (bollinger.get('lower', 0)) * 100):.1f}")
                            
                            if 'news' in data:
                                news = data['news']
                                st.dataframe(news, use_container_width=True)
                            
                            if 'margins' in data:
                                margins = data['margins']
                                if 'equity' in margins:
                                    equity = margins['equity']
                                    st.metric("Available Balance", f"₹{equity.get('net', 0):,.2f}")
                    
                    st.markdown("---")
                else:
                    st.error(f"❌ No data available for {symbol}")
        else:
            st.warning("Please enter at least one symbol")

def main():
    """Main function"""
    create_streamlit_interface()

if __name__ == "__main__":
    main()
