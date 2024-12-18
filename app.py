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
import numpy as np
from scipy import stats

# Initialize APIs
cg = CoinGeckoAPI()

# Configure Streamlit page
st.set_page_config(layout="wide")

# List of known stablecoins and derivatives to filter out
STABLECOINS = {'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'USDD'}
LIQUID_STAKING = {'STETH', 'WSTETH', 'RETH', 'FRXETH', 'ANKR', 'LDO', 'CRV', 'SD', 'SWISE', 'SDN', 'MATICX', 'STMX', 'cbETH', 'rETH2', 'cbETH', 'sETH2', 'ETH2X-FLI', 'ETH2', 'ETH2-FLI', 'sETH', 'ETH2X', 'rETH', 'cbETH-FLI', 'sETH-FLI', 'ETH2-FLI-P', 'ETH2-P', 'ETH2X-FLI-P', 'cbETH-P', 'sETH-FLI-P', 'sETH-P', 'ETH2X-P', 'rETH-P', 'cbETH-FLI-P', 'rETH-FLI', 'rETH-FLI-P', 'sETH2-FLI', 'sETH2-FLI-P', 'sETH2-P', 'stETH', 'stETH-FLI', 'stETH-FLI-P', 'stETH-P', 'wstETH', 'wstETH-FLI', 'wstETH-FLI-P', 'wstETH-P'}

# Known search terms for popular coins
COIN_SEARCH_TERMS = {
    'BTC': 'Bitcoin',
    'ETH': 'Ethereum',
    'SOL': 'Solana',
    'XRP': 'Ripple',
    'ADA': 'Cardano',
    'DOGE': 'Dogecoin',
    'AVAX': 'Avalanche',
    'DOT': 'Polkadot',
    'MATIC': 'Polygon',
    'LINK': 'Chainlink',
    'SHIB': 'Shiba',
    'LTC': 'Litecoin',
    'TRX': 'Tron',
    'ATOM': 'Cosmos',
    'UNI': 'Uniswap',
    'ETC': 'Ethereum Classic',
    'BCH': 'Bitcoin Cash',
    'XLM': 'Stellar',
    'NEAR': 'NEAR Protocol',
    'ALGO': 'Algorand',
    'FIL': 'Filecoin',
    'VET': 'VeChain',
    'HBAR': 'Hedera',
    'APE': 'ApeCoin',
    'SAND': 'Sandbox',
    'MANA': 'Decentraland',
    'AAVE': 'Aave',
    'GRT': 'Graph',
    'THETA': 'Theta',
    'FTM': 'Fantom',
    'XMR': 'Monero',
    'CAKE': 'PancakeSwap',
    'KCS': 'KuCoin Token',
    'CRO': 'Cronos',
    'ZEC': 'Zcash',
    'FLOW': 'Flow',
    'CHZ': 'Chiliz',
    'BAT': 'Basic Attention Token',
    'ENJ': 'Enjin',
    'ONE': 'Harmony',
    'HOT': 'Holochain',
    'KLAY': 'Klaytn',
    'DASH': 'Dash',
    'WAVES': 'Waves',
    'COMP': 'Compound',
    'EGLD': 'MultiversX',
    'XTZ': 'Tezos',
    'RENDER': 'RNDR',
    'WIF': 'dogwifhat',
    'LEO': 'UNUS SED LEO',
    'BNB': 'BNB'  # Special case, only use ticker
}

# Cache directory
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

@st.cache_data(ttl=3600)
def get_top_coins():
    """Get top coins from CoinGecko"""
    try:
        # Get top 50 coins
        coins = cg.get_coins_markets(
            vs_currency='usd',
            order='market_cap_desc',
            per_page=50,
            sparkline=False
        )
        
        # Filter out stablecoins and liquid staking tokens
        filtered_coins = [
            coin for coin in coins 
            if coin['symbol'].upper() not in STABLECOINS 
            and coin['symbol'].upper() not in LIQUID_STAKING
        ]
        
        return filtered_coins  # Return all filtered coins from top 50
        
    except Exception as e:
        st.error(f"Error fetching top coins: {str(e)}")
        return []

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
        # For BNB, just use the ticker
        if symbol == 'BNB':
            search_terms = [symbol]
        else:
            # For other coins, use both ticker and full name if available
            full_name = COIN_SEARCH_TERMS.get(symbol, coin['name'])
            search_terms = [symbol, full_name] if symbol != full_name else [symbol]
        
        pytrends = TrendReq(hl='en-US', 
                           tz=420,
                           timeout=(10,25),
                           requests_args={'verify': True})
        
        # Build payload with all search terms
        pytrends.build_payload(
            kw_list=search_terms,
            timeframe=timeframe,
            geo='TH',
            gprop=''
        )
        
        # Add longer delay to avoid rate limiting
        time.sleep(5)
        
        interest_df = pytrends.interest_over_time()
        
        # Debug information
        st.sidebar.write(f"Debug - {symbol}:")
        st.sidebar.write(f"Search terms: {', '.join(search_terms)}")
        
        if interest_df.empty:
            st.sidebar.error(f"No data returned for {symbol}")
            return None
        
        st.sidebar.write(f"Data points: {len(interest_df)}")
        
        # First pass: calculate raw sums for each timestamp
        raw_trends = []
        max_value = 0
        for index, row in interest_df.iterrows():
            total_value = sum(row[term] for term in search_terms)
            max_value = max(max_value, total_value)
            raw_trends.append({
                'date': index.strftime('%Y-%m-%d %H:%M'),
                'value': total_value
            })
        
        # Second pass: normalize values relative to the maximum
        trends_data = []
        for trend in raw_trends:
            normalized_value = (trend['value'] / max_value * 100) if max_value > 0 else 0
            trends_data.append({
                'date': trend['date'],
                'value': normalized_value
            })
        
        # Show value ranges for debugging
        if trends_data:
            values = [d['value'] for d in trends_data]
            st.sidebar.write(f"Raw value range: {min(raw_trends, key=lambda x: x['value'])['value']:.1f} - {max_value:.1f}")
            st.sidebar.write(f"Normalized range: {min(values):.1f} - {max(values):.1f}")
        
        save_to_cache(symbol, timeframe, trends_data)
        
        return {
            'coin': coin['name'],
            'symbol': symbol,
            'market_cap_rank': coin['market_cap_rank'],
            'trends_data': trends_data,
            'search_term': ' + '.join(search_terms)
        }
        
    except Exception as e:
        st.sidebar.error(f"Error for {symbol}: {str(e)}")
        if "429" in str(e):
            # Add extra delay on rate limit
            time.sleep(10)
        return None
    
    return None

def get_recent_trends(results, hours=6):
    """Get the most recent trend data and calculate average interest"""
    recent_trends = []
    
    for result in results:
        trends = result['trends_data']
        if trends:
            df = pd.DataFrame(trends)
            df['date'] = pd.to_datetime(df['date'])
            
            # Get just the last 6 hours of data
            recent_data = df.nlargest(hours, 'date')
            
            if not recent_data.empty:
                avg_interest = recent_data['value'].mean()
                recent_trends.append({
                    'coin': result['coin'],
                    'symbol': result['symbol'],
                    'search_term': result['search_term'],
                    'avg_interest': avg_interest
                })
    
    # Sort by average interest and return top 5
    sorted_trends = sorted(recent_trends, key=lambda x: x['avg_interest'], reverse=True)
    
    # Add trend indicators
    for trend in sorted_trends:
        if trend['avg_interest'] > 0:
            trend['direction'] = '↗️'
        else:
            trend['direction'] = '➡️'
    
    # Show debug information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Search Interest (Last 6 Hours)")
    for trend in sorted_trends:
        st.sidebar.write(f"{trend['symbol']}:")
        st.sidebar.write(f"- Search term: {trend['search_term']}")
        st.sidebar.write(f"- Interest: {trend['avg_interest']:.1f}")
    
    return sorted_trends[:5]

def main():
    st.title("Trending Cryptocurrency in Thailand")
    
    # Get top coins (cached)
    with st.spinner("Fetching top 20 cryptocurrencies by market cap..."):
        filtered_coins = get_top_coins()
        st.caption(f"Data sourced from CoinGecko - Top {len(filtered_coins)} coins by market cap. Trends data is fetched from Google Trends.")
    
    # Add custom coin input
    custom_coins = st.text_input(
        "Add custom coins (optional)",
        placeholder="Enter coin symbols of your choice separated by commas (e.g., UNI, NEAR, TAO, Pepe etc.)",
        help="Enter additional coin symbols to track, separated by commas"
    )
    
    # Process custom coins if provided
    if custom_coins:
        custom_symbols = [s.strip().upper() for s in custom_coins.split(',')]
        # Add custom coins to the filtered list if they're not already there
        existing_symbols = {coin['symbol'].upper() for coin in filtered_coins}
        for symbol in custom_symbols:
            if symbol and symbol not in existing_symbols:
                try:
                    # Try to get coin info from CoinGecko
                    cg = CoinGeckoAPI()
                    coins_list = cg.get_coins_list()
                    coin_info = next((coin for coin in coins_list if coin['symbol'].upper() == symbol), None)
                    if coin_info:
                        coin_data = cg.get_coin_by_id(coin_info['id'])
                        filtered_coins.append({
                            'id': coin_info['id'],
                            'symbol': symbol,
                            'name': coin_info['name'],
                            'market_cap_rank': coin_data.get('market_cap_rank', 9999)
                        })
                except Exception as e:
                    st.warning(f"Could not fetch data for {symbol}: {str(e)}")

    # Process coins sequentially
    results = []
    errors = []
    
    with st.spinner("Fetching Google Trends data..."):
        progress_bar = st.progress(0)
        
        for i, coin in enumerate(filtered_coins):
            result = fetch_google_trends(coin)
            if result:
                if 'error' in result:
                    errors.append(result)
                else:
                    results.append(result)
            progress_bar.progress((i + 1) / len(filtered_coins))
            time.sleep(1)  # Add delay between requests
        
        progress_bar.empty()
        
        if not results:
            st.error("No trend data available. Please try again later.")
            return

        # Get top 5 trending coins in the last 6 hours
        top_trending = get_recent_trends(results, hours=6)
        
        # Display top trending coins
        st.header(" Top 5 Trending Coins in Thailand")
        st.caption("Based on search interest in the last 6 hours")
        
        # Create columns for trending coins
        cols = st.columns(5)
        for i, (col, coin) in enumerate(zip(cols, top_trending)):
            with col:
                st.metric(
                    label=f"#{i+1} {coin['symbol']}", 
                    value=f"{coin['coin']}", 
                    delta=f"Interest: {coin['avg_interest']:.1f} {coin['direction']}"
                )
        
        st.markdown("---")

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
                
        # Add expander for errors at the bottom
        if errors:
            with st.expander("Show Error Details"):
                st.write("The following coins had errors during processing:")
                for error in errors:
                    st.write(f"- {error['coin']} ({error['symbol']}): {error['error']}")

if __name__ == "__main__":
    main()
