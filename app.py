import streamlit as st
import pandas as pd
import plotly.express as px
from pycoingecko import CoinGeckoAPI
from pytrends.request import TrendReq
from datetime import datetime, timedelta
import time
import json
import os
from pathlib import Path

# Initialize APIs
cg = CoinGeckoAPI()

# Configure Streamlit page
st.set_page_config(layout="wide")

# List of known stablecoins to filter out
STABLECOINS = {'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'USDD'}

# Known search terms for popular coins
COIN_SEARCH_TERMS = {
    'BTC': 'Bitcoin',
    'ETH': 'Ethereum',
    'SOL': 'Solana',
    'XRP': 'Ripple',
    'BNB': 'Binance',
    'ADA': 'Cardano',
    'DOGE': 'Dogecoin',
    'AVAX': 'Avalanche',
    'DOT': 'Polkadot',
    'MATIC': 'Polygon',
    'LINK': 'Chainlink',
    'SHIB': 'Shiba Inu',
    'LTC': 'Litecoin',
    'TRX': 'Tron',
    'ATOM': 'Cosmos',
    'UNI': 'Uniswap',
    'ETC': 'Ethereum Classic',
    'XLM': 'Stellar',
    'OP': 'Optimism',
    'NEAR': 'Near Protocol'
}

# Cache directory
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_top_coins():
    """Get top coins from CoinGecko with caching"""
    coins = cg.get_coins_markets(
        vs_currency='usd',
        order='market_cap_desc',
        per_page=50,  # Get more to filter stablecoins
        sparkline=False
    )
    return [coin for coin in coins if coin['symbol'].upper() not in STABLECOINS][:20]

def get_cache_file(coin_symbol, timeframe):
    """Get cache file path for a coin"""
    return CACHE_DIR / f"{coin_symbol}_{timeframe}.json"

def get_cached_data(coin_symbol, timeframe):
    """Get cached data for a coin if available and fresh"""
    cache_file = get_cache_file(coin_symbol, timeframe)
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        cache_time = datetime.fromtimestamp(data['timestamp'])
        if datetime.now() - cache_time < timedelta(hours=1):  # Cache valid for 1 hour
            return data['trends_data']
    return None

def save_to_cache(coin_symbol, timeframe, trends_data):
    """Save trend data to cache"""
    cache_file = get_cache_file(coin_symbol, timeframe)
    data = {
        'timestamp': datetime.now().timestamp(),
        'trends_data': trends_data
    }
    cache_file.write_text(json.dumps(data))

def fetch_google_trends(coin):
    """Fetch Google Trends data for a given coin with caching"""
    symbol = coin['symbol'].upper()
    # Use exact timeframe format that Google expects
    end_time = datetime.now()
    start_time = end_time - timedelta(days=3)
    timeframe = f"{start_time.strftime('%Y-%m-%d')} {end_time.strftime('%Y-%m-%d')}"
    
    # Check cache first
    cached_data = get_cached_data(symbol, timeframe)
    if cached_data:
        return {
            'coin': coin['name'],
            'symbol': symbol,
            'market_cap_rank': coin['market_cap_rank'],
            'trends_data': cached_data,
            'search_term': COIN_SEARCH_TERMS.get(symbol, coin['name'])
        }
    
    try:
        search_term = COIN_SEARCH_TERMS.get(symbol, coin['name'])
        pytrends = TrendReq(hl='en-US', 
                           tz=420,
                           timeout=(10,25),
                           requests_args={'verify': True})
        
        # Build payload with minimal parameters
        pytrends.build_payload(
            kw_list=[search_term],
            timeframe=timeframe,
            geo='TH',
            gprop=''
        )
        
        # Add small delay
        time.sleep(1)
        
        interest_df = pytrends.interest_over_time()
        
        if not interest_df.empty:
            trends_data = [{
                'date': index.strftime('%Y-%m-%d %H:%M'),
                'value': row[search_term]
            } for index, row in interest_df.iterrows()]
            
            save_to_cache(symbol, timeframe, trends_data)
            
            return {
                'coin': coin['name'],
                'symbol': symbol,
                'market_cap_rank': coin['market_cap_rank'],
                'trends_data': trends_data,
                'search_term': search_term
            }
    except Exception as e:
        st.error(f"Error fetching trends for {symbol}: {str(e)}")
    
    return None

def get_recent_trends(results, hours=6):
    """Get the most recent trend data and calculate average interest"""
    recent_trends = []
    
    for result in results:
        trends = result['trends_data']
        if trends:
            df = pd.DataFrame(trends)
            df['date'] = pd.to_datetime(df['date'])
            recent_data = df.nlargest(hours, 'date')
            
            if not recent_data.empty:
                avg_interest = recent_data['value'].mean()
                recent_trends.append({
                    'coin': result['coin'],
                    'symbol': result['symbol'],
                    'search_term': result['search_term'],
                    'avg_interest': avg_interest
                })
    
    return sorted(recent_trends, key=lambda x: x['avg_interest'], reverse=True)[:5]

def main():
    st.title("ThailandCryptocurrency Trends Analysis ")
    st.write("Analyzing Google Trends data for cryptocurrencies in Thailand")

    # Add hours selector
    col1, col2 = st.columns([2, 1])
    with col1:
        hours = st.slider("Select hours to analyze", min_value=1, max_value=72, value=6)

    # Get top coins (cached)
    with st.spinner("Fetching cryptocurrency data..."):
        filtered_coins = get_top_coins()

    # Process coins sequentially
    results = []
    with st.spinner("Fetching Google Trends data..."):
        progress_bar = st.progress(0)
        
        for i, coin in enumerate(filtered_coins):
            result = fetch_google_trends(coin)
            if result:
                results.append(result)
            progress_bar.progress((i + 1) / len(filtered_coins))
            time.sleep(1)  # Add delay between requests
        
        progress_bar.empty()
        
        if not results:
            st.error("No trend data available. Please try again later.")
            return

        # Get top 5 trending coins in the last N hours
        top_trending = get_recent_trends(results, hours)
        
        # Display top trending coins
        st.header(" Top 5 Trending Coins")
        st.caption(f"Based on search interest in the last {hours} hours")
        
        # Create columns for trending coins
        cols = st.columns(5)
        for i, (col, coin) in enumerate(zip(cols, top_trending)):
            with col:
                st.metric(
                    label=f"#{i+1} {coin['symbol']}", 
                    value=f"{coin['coin']}", 
                    delta=f"Interest: {coin['avg_interest']:.1f}"
                )
        
        st.markdown("---")
        st.subheader(" Detailed Trends")

        # Prepare data for visualization
        trend_data = []
        for result in results:
            for point in result['trends_data']:
                trend_data.append({
                    'Date': point['date'],
                    'Interest': point['value'],
                    'Cryptocurrency': f"{result['symbol']}"
                })

        df = pd.DataFrame(trend_data)
        
        # Create the line chart using Plotly
        fig = px.line(
            df,
            x='Date',
            y='Interest',
            color='Cryptocurrency',
            title='Cryptocurrency Search Interest (Last 3 Days)',
            labels={'Interest': 'Search Interest', 'Date': 'Date'}
        )
        
        # Update layout for better readability
        fig.update_layout(
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            ),
            xaxis=dict(
                tickmode='auto',
                nticks=12
            )
        )

        # Display the chart with full width
        st.plotly_chart(fig, use_container_width=True)
        
        # Add expander at the bottom for processing details
        with st.expander("Show Processing Details"):
            st.write("Processed the following coins:")
            for coin in filtered_coins:
                st.write(f"- {coin['name']} ({coin['symbol'].upper()})")

if __name__ == "__main__":
    main()
