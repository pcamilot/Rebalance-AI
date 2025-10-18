import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Import required libraries
import yfinance as yf
import praw
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import google.generativeai as genai
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json

# Page configuration
st.set_page_config(
    page_title="RebalanceAI - Investment Co-Pilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'reddit_client' not in st.session_state:
    st.session_state.reddit_client = None
if 'news_client' not in st.session_state:
    st.session_state.news_client = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'multi_sentiment' not in st.session_state:
    st.session_state.multi_sentiment = None

# ========================================
# ALL BACKEND FUNCTIONS (from your notebook)
# ========================================

def load_market_data(tickers, period="6mo", interval="1d"):
    """Load market data for given tickers using yfinance"""
    try:
        data = yf.download(tickers, period=period, interval=interval, group_by='ticker', auto_adjust=True, progress=False)
        price_data = pd.DataFrame()

        for ticker in tickers:
            if isinstance(data.columns, pd.MultiIndex):
                if (ticker, 'Close') in data.columns:
                    price_data[ticker] = data[(ticker, 'Close')]
                elif (ticker, 'Adj Close') in data.columns:
                    price_data[ticker] = data[(ticker, 'Adj Close')]
            else:
                if 'Close' in data.columns:
                    price_data[ticker] = data['Close']
                elif 'Adj Close' in data.columns:
                    price_data[ticker] = data['Adj Close']

        price_data.dropna(how='all', inplace=True)
        return price_data

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()


def compute_portfolio_metrics(price_df, risk_level="moderate", rf_rate=0.04):
    """Compute portfolio metrics: returns, volatility, Sharpe ratio"""
    if price_df.empty:
        return pd.DataFrame()

    returns = price_df.pct_change().dropna()
    mean_daily = returns.mean()
    vol_daily = returns.std()
    mean_annual = mean_daily * 252
    vol_annual = vol_daily * np.sqrt(252)
    sharpe = (mean_annual - rf_rate) / vol_annual

    metrics_df = pd.DataFrame({
        "Annual Return": mean_annual,
        "Volatility": vol_annual,
        "Sharpe Ratio": sharpe
    }).sort_values(by="Sharpe Ratio", ascending=False)

    if risk_level == "conservative":
        metrics_df["Suggested Weight"] = np.where(
            metrics_df["Volatility"] < metrics_df["Volatility"].median(), 0.25, 0.05
        )
    elif risk_level == "moderate":
        metrics_df["Suggested Weight"] = np.where(
            metrics_df["Sharpe Ratio"] > metrics_df["Sharpe Ratio"].median(), 0.3, 0.15
        )
    elif risk_level == "aggressive":
        metrics_df["Suggested Weight"] = np.where(
            metrics_df["Sharpe Ratio"] > metrics_df["Sharpe Ratio"].median(), 0.4, 0.2
        )
    else:
        metrics_df["Suggested Weight"] = 0.25

    metrics_df["Suggested Weight"] = metrics_df["Suggested Weight"] / metrics_df["Suggested Weight"].sum()
    return metrics_df.round(4)


def fetch_reddit_sentiment(ticker, reddit_client, limit=20):
    """Fetch sentiment from Reddit"""
    if reddit_client is None:
        return None
    try:
        subreddits = ["wallstreetbets", "stocks", "investing"]
        posts = []
        for sub in subreddits:
            subreddit = reddit_client.subreddit(sub)
            for post in subreddit.search(ticker, sort="new", limit=limit):
                posts.append(post.title + " " + post.selftext)
        if not posts:
            return 0
        sentiments = [st.session_state.analyzer.polarity_scores(p)["compound"] for p in posts if p]
        return np.mean(sentiments)
    except Exception as e:
        return None


def fetch_news_sentiment(ticker, news_client, days_back=5):
    """Fetch sentiment from NewsAPI"""
    if news_client is None:
        return None
    try:
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        articles = news_client.get_everything(
            q=ticker,
            from_param=from_date,
            language="en",
            sort_by="relevancy",
            page_size=30
        )
        if not articles or not articles.get("articles"):
            return 0
        headlines = [a["title"] + " " + (a.get("description") or "") for a in articles["articles"]]
        sentiments = [st.session_state.analyzer.polarity_scores(h)["compound"] for h in headlines if h]
        return np.mean(sentiments)
    except Exception as e:
        return None


def compute_combined_sentiment(tickers, reddit_client, news_client):
    """Combine Reddit and News sentiment"""
    sentiment_summary = []
    for t in tickers:
        reddit_score = fetch_reddit_sentiment(t, reddit_client)
        news_score = fetch_news_sentiment(t, news_client)
        combined = None
        if reddit_score is not None and news_score is not None:
            combined = (0.6 * news_score) + (0.4 * reddit_score)
        elif news_score is not None:
            combined = news_score
        elif reddit_score is not None:
            combined = reddit_score
        else:
            combined = 0
        sentiment_summary.append({
            "Ticker": t,
            "Reddit_Sentiment": reddit_score,
            "News_Sentiment": news_score,
            "Combined_Sentiment": combined
        })
    return pd.DataFrame(sentiment_summary)


class MultiSourceSentiment:
    """Aggregate sentiment from multiple sources"""
    def __init__(self, reddit_client, news_client, analyzer):
        self.analyzer = analyzer
        self.reddit_client = reddit_client
        self.news_client = news_client

    def analyze_complete_sentiment(self, symbol: str, company_name: str = None) -> dict:
        reddit_score = fetch_reddit_sentiment(symbol, self.reddit_client)
        news_score = fetch_news_sentiment(symbol, self.news_client)
        scores = [s for s in [reddit_score, news_score] if s is not None]
        overall = np.mean(scores) if scores else 0

        if overall > 0.15:
            signal = "STRONG BUY"
        elif overall > 0.05:
            signal = "BUY"
        elif overall > -0.05:
            signal = "HOLD"
        elif overall > -0.15:
            signal = "SELL"
        else:
            signal = "STRONG SELL"

        return {
            'overall_sentiment': overall,
            'signal': signal,
            'confidence': 0.75,
            'sources': {
                'reddit': {'sentiment_score': reddit_score or 0},
                'news': {'sentiment_score': news_score or 0}
            }
        }


def complete_stock_analysis(symbol: str, multi_sentiment) -> dict:
    """Complete analysis combining all data sources"""
    ticker = yf.Ticker(symbol)
    info = ticker.info
    hist = ticker.history(period="1mo")

    financial_data = {
        'price': hist['Close'].iloc[-1] if not hist.empty else 0,
        'volume': hist['Volume'].iloc[-1] if not hist.empty else 0,
        'market_cap': info.get('marketCap', 0),
        'pe_ratio': info.get('trailingPE', 0),
        'company_name': info.get('longName', symbol)
    }

    sentiment = multi_sentiment.analyze_complete_sentiment(symbol, financial_data['company_name'])

    financial_score = 0
    if 0 < financial_data['pe_ratio'] < 15:
        financial_score += 2
    elif 15 <= financial_data['pe_ratio'] < 25:
        financial_score += 1
    else:
        financial_score -= 1

    sentiment_score = sentiment['overall_sentiment'] * 10
    total_score = financial_score + sentiment_score

    if total_score > 2:
        recommendation = "STRONG BUY"
    elif total_score > 0:
        recommendation = "BUY"
    elif total_score > -2:
        recommendation = "HOLD"
    else:
        recommendation = "SELL"

    return {
        'symbol': symbol,
        'financial': financial_data,
        'sentiment': sentiment,
        'recommendation': recommendation,
        'total_score': total_score
    }


class RebalanceAIAgent:
    """AI Investment Co-Pilot powered by Gemini"""
    def __init__(self, gemini_api_key: str, reddit_client, news_client, analyzer):
        genai.configure(api_key=gemini_api_key)
        self.reddit_client = reddit_client
        self.news_client = news_client
        self.analyzer = analyzer
        
        model_names = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-exp"]
        self.model = None
        self.model_name = None

        for model_name in model_names:
            try:
                self.model = genai.GenerativeModel(model_name)
                test_response = self.model.generate_content("test")
                self.model_name = model_name
                break
            except:
                continue

        if self.model is None:
            raise Exception("Could not initialize any Gemini model")

        self.system_prompt = """You are RebalanceAI, an expert investment co-pilot assistant.
Your role is to analyze investment queries and provide clear, defensible recommendations."""

    def analyze_query(self, user_query: str) -> dict:
        analysis_plan = self._create_analysis_plan(user_query)
        params = self._extract_parameters(user_query, analysis_plan)

        if 'portfolio' in user_query.lower() or 'invest' in user_query.lower():
            result = self._analyze_portfolio_request(params)
        elif 'compare' in user_query.lower():
            result = self._compare_stocks(params)
        else:
            result = self._general_analysis(params)

        recommendation = self._generate_recommendation(result, params)

        return {
            'query': user_query,
            'analysis_plan': analysis_plan,
            'parameters': params,
            'analysis_result': result,
            'recommendation': recommendation
        }

    def _create_analysis_plan(self, query: str) -> dict:
        prompt = f"""{self.system_prompt}
User Query: "{query}"
Create an analysis plan in JSON format with reasoning, query_type, required_data, key_metrics, and risk_profile."""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                text = text.split('```')[1].split('```')[0]
            return json.loads(text.strip())
        except:
            return {
                "reasoning": "Analyzing investment query for portfolio optimization.",
                "query_type": "portfolio_creation",
                "required_data": ["financial", "sentiment"],
                "key_metrics": ["return", "risk", "sharpe"],
                "risk_profile": "moderate"
            }

    def _extract_parameters(self, query: str, plan: dict) -> dict:
        import re
        params = {
            'symbols': [],
            'amount': 10000,
            'risk_profile': plan.get('risk_profile', 'moderate'),
            'emphasis': None
        }
        symbols = re.findall(r'\b[A-Z]{2,5}\b', query)
        if symbols:
            params['symbols'] = symbols
        amounts = re.findall(r'\$?([\d,]+)', query)
        if amounts:
            try:
                params['amount'] = float(amounts[0].replace(',', ''))
            except:
                pass
        if 'tech' in query.lower():
            params['emphasis'] = 'technology'
        if 'conservative' in query.lower():
            params['risk_profile'] = 'conservative'
        elif 'aggressive' in query.lower():
            params['risk_profile'] = 'aggressive'
        return params

    def _analyze_portfolio_request(self, params: dict) -> dict:
        if params['emphasis'] == 'technology':
            symbols = params['symbols'] if params['symbols'] else ['AAPL', 'MSFT', 'NVDA', 'GOOGL']
        else:
            symbols = params['symbols'] if params['symbols'] else ['AAPL', 'MSFT', 'GOOGL', 'JPM']

        price_df = load_market_data(symbols, period="6mo")
        metrics_df = compute_portfolio_metrics(price_df, risk_level=params['risk_profile'])
        sentiment_df = compute_combined_sentiment(symbols, self.reddit_client, self.news_client)

        return {
            'symbols': symbols,
            'portfolio_metrics': metrics_df.to_dict(),
            'sentiment_data': sentiment_df.to_dict(),
            'total_amount': params['amount']
        }

    def _compare_stocks(self, params: dict) -> dict:
        symbols = params['symbols']
        comparison_data = {}
        multi_sentiment = MultiSourceSentiment(self.reddit_client, self.news_client, self.analyzer)

        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")
            info = ticker.info
            returns = hist['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252)
            annual_return = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 4
            sharpe = (annual_return - 0.04) / volatility if volatility > 0 else 0
            sentiment = multi_sentiment.analyze_complete_sentiment(symbol)

            comparison_data[symbol] = {
                'current_price': hist['Close'].iloc[-1],
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'sentiment': sentiment['overall_sentiment'],
                'signal': sentiment['signal']
            }
        return comparison_data

    def _general_analysis(self, params: dict) -> dict:
        symbols = params['symbols'] if params['symbols'] else ['SPY']
        return self._compare_stocks({'symbols': symbols})

    def _generate_recommendation(self, result: dict, params: dict) -> str:
        prompt = f"""{self.system_prompt}
ANALYSIS RESULTS: {json.dumps(result, indent=2, default=str)}
USER PARAMETERS:
- Investment Amount: ${params['amount']:,.2f}
- Risk Profile: {params['risk_profile']}
- Emphasis: {params.get('emphasis', 'None')}

Generate a clear, actionable recommendation."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating recommendation: {e}"


# ========================================
# STREAMLIT UI
# ========================================

# Sidebar - API Configuration
with st.sidebar:
    st.header("🔑 API Configuration")
    
    with st.expander("Configure API Keys", expanded=not st.session_state.initialized):
        gemini_key = st.text_input("Gemini API Key", type="password", value="")
        
        st.markdown("---")
        st.caption("Optional APIs")
        reddit_id = st.text_input("Reddit Client ID", type="password", value="")
        reddit_secret = st.text_input("Reddit Client Secret", type="password", value="")
        news_key = st.text_input("NewsAPI Key", type="password", value="")
        
        if st.button("🚀 Initialize System", use_container_width=True):
            with st.spinner("Initializing RebalanceAI..."):
                try:
                    # Initialize sentiment analyzer
                    st.session_state.analyzer = SentimentIntensityAnalyzer()
                    
                    # Initialize Reddit
                    if reddit_id and reddit_secret:
                        st.session_state.reddit_client = praw.Reddit(
                            client_id=reddit_id,
                            client_secret=reddit_secret,
                            user_agent="rebalance_ai"
                        )
                    
                    # Initialize News
                    if news_key:
                        st.session_state.news_client = NewsApiClient(api_key=news_key)
                    
                    # Initialize MultiSourceSentiment
                    st.session_state.multi_sentiment = MultiSourceSentiment(
                        st.session_state.reddit_client,
                        st.session_state.news_client,
                        st.session_state.analyzer
                    )
                    
                    # Initialize Agent
                    st.session_state.agent = RebalanceAIAgent(
                        gemini_key,
                        st.session_state.reddit_client,
                        st.session_state.news_client,
                        st.session_state.analyzer
                    )
                    
                    st.session_state.initialized = True
                    st.success("✅ System initialized!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    if st.session_state.initialized:
        st.success("✅ System Ready")
        st.info(f"🤖 Model: {st.session_state.agent.model_name}")

# Main content
st.markdown("<h1 class='main-header'>🤖 RebalanceAI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>AI-Powered Investment Co-Pilot</p>", unsafe_allow_html=True)

if not st.session_state.initialized:
    st.warning("⚠️ Please configure API keys in the sidebar to get started.")
    st.info("""
    ### Welcome to RebalanceAI
    
    This AI-powered investment co-pilot helps you:
    - 📊 Analyze portfolio metrics and risk
    - 📰 Assess market sentiment from news and social media
    - 🎯 Get personalized investment recommendations
    - ⚖️ Compare stocks across multiple dimensions
    
    **To begin:** Enter your Gemini API key in the sidebar and click "Initialize System"
    """)
else:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 AI Chat", "📊 Portfolio Analysis", "🔍 Stock Analysis", "⚖️ Compare Stocks"])
    
    with tab1:
        st.header("Chat with RebalanceAI")
        
        with st.expander("💡 Example Questions"):
            st.markdown("""
            - "Analyze a $10,000 portfolio for moderate risk with tech emphasis"
            - "Compare TSLA vs. NVDA on volatility and Sharpe ratio"
            - "What are the best tech stocks for aggressive growth?"
            """)
        
        user_query = st.text_area(
            "Your Question:",
            placeholder="e.g., Analyze a $10,000 portfolio for moderate risk with tech emphasis",
            height=100
        )
        
        if st.button("🚀 Analyze"):
            if user_query:
                with st.spinner("🤖 AI is analyzing..."):
                    try:
                        result = st.session_state.agent.analyze_query(user_query)
                        
                        st.success("✅ Analysis Complete!")
                        
                        with st.expander("🧠 AI Reasoning", expanded=True):
                            st.write(result['analysis_plan']['reasoning'])
                        
                        st.markdown("### 📋 Recommendation")
                        st.markdown(result['recommendation'])
                        
                        with st.expander("⚙️ Parameters"):
                            st.json(result['parameters'])
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("Please enter a question")
    
    with tab2:
        st.header("Portfolio Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            investment_amount = st.number_input("Investment ($)", min_value=100, value=10000, step=100)
        with col2:
            risk_level = st.selectbox("Risk Profile", ["conservative", "moderate", "aggressive"])
        with col3:
            period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"])
        
        ticker_input = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,NVDA")
        
        if st.button("📊 Analyze Portfolio"):
            with st.spinner("Analyzing..."):
                try:
                    tickers = [t.strip().upper() for t in ticker_input.split(",")]
                    price_df = load_market_data(tickers, period=period)
                    
                    if not price_df.empty:
                        metrics_df = compute_portfolio_metrics(price_df, risk_level=risk_level)
                        
                        st.success("✅ Complete!")
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = go.Figure()
                            fig.add_trace(go.Bar(x=metrics_df.index, y=metrics_df['Sharpe Ratio'], marker_color='lightblue'))
                            fig.update_layout(title="Sharpe Ratio", height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig2 = go.Figure()
                            fig2.add_trace(go.Bar(x=metrics_df.index, y=metrics_df['Volatility'], marker_color='lightcoral'))
                            fig2.update_layout(title="Volatility", height=400)
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        fig3 = go.Figure()
                        for ticker in tickers:
                            if ticker in price_df.columns:
                                normalized = (price_df[ticker] / price_df[ticker].iloc[0]) * 100
                                fig3.add_trace(go.Scatter(x=price_df.index, y=normalized, name=ticker, mode='lines'))
                        fig3.update_layout(title="Normalized Performance", height=500)
                        st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.error("Could not load data")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with tab3:
        st.header("Stock Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            stock_symbol = st.text_input("Stock Symbol", "AAPL").upper()
        with col2:
            st.write("")
            st.write("")
            analyze_btn = st.button("🔍 Analyze")
        
        if analyze_btn:
            with st.spinner(f"Analyzing {stock_symbol}..."):
                try:
                    analysis = complete_stock_analysis(stock_symbol, st.session_state.multi_sentiment)
                    
                    st.success("✅ Complete!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Price", f"${analysis['financial']['price']:.2f}")
                    with col2:
                        st.metric("P/E", f"{analysis['financial']['pe_ratio']:.2f}")
                    with col3:
                        st.metric("Sentiment", f"{analysis['sentiment']['overall_sentiment']:.3f}")
                    with col4:
                        st.metric("Signal", analysis['recommendation'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📊 Financial")
                        st.write(f"**Company:** {analysis['financial']['company_name']}")
                        st.write(f"**Market Cap:** ${analysis['financial']['market_cap']/1e9:.2f}B")
                    
                    with col2:
                        st.subheader("📰 Sentiment")
                        st.write(f"**Reddit:** {analysis['sentiment']['sources']['reddit']['sentiment_score']:.3f}")
                        st.write(f"**News:** {analysis['sentiment']['sources']['news']['sentiment_score']:.3f}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with tab4:
        st.header("Compare Stocks")
        
        col1, col2 = st.columns(2)
        with col1:
            stock1 = st.text_input("First Stock", "AAPL").upper()
        with col2:
            stock2 = st.text_input("Second Stock", "MSFT").upper()
        
        if st.button("⚖️ Compare"):
            with st.spinner("Comparing..."):
                try:
                    query = f"Compare {stock1} vs. {stock2} on volatility, Sharpe ratio, and sentiment."
                    result = st.session_state.agent.analyze_query(query)
                    
                    st.success("✅ Complete!")
                    st.markdown("### 📊 Results")
                    st.markdown(result['recommendation'])
                    
                    if 'analysis_result' in result:
                        df = pd.DataFrame(result['analysis_result']).T
                        st.dataframe(df, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🤖 RebalanceAI - Powered by Gemini AI & Market Data APIs</p>
    <p>⚠️ For educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
