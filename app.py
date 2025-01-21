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
from scipy.stats import zscore

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

def fetch_google_trends(coin, retries=3):
    """Fetch Google Trends data for a given coin with caching and retries"""
    symbol = coin['symbol'].upper()
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
    
    for attempt in range(retries):
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
                               timeout=(10, 25),
                               requests_args={'verify': True})
            
            # Build payload with all search terms
            pytrends.build_payload(
                kw_list=search_terms,
                timeframe=timeframe,
                geo='TH',
                gprop=''
            )
            
            # Add delay to avoid rate limiting
            time.sleep(10)  # Increased delay to 10 seconds
            
            interest_df = pytrends.interest_over_time()
            
            if interest_df.empty:
                st.sidebar.warning(f"No data returned for {symbol}")
                return None
            
            # Calculate Z-scores for normalization
            raw_values = []
            for index, row in interest_df.iterrows():
                total_value = sum(row[term] for term in search_terms)
                raw_values.append(total_value)
            
            # Z-score normalization
            z_scores = zscore(raw_values)
            
            trends_data = []
            for index, z in zip(interest_df.index, z_scores):
                trends_data.append({
                    'date': index.strftime('%Y-%m-%d %H:%M'),
                    'value': z
                })
            
            # Calculate spike ratio
            df_coin = pd.DataFrame(trends_data)
            df_coin['value'] = pd.to_numeric(df_coin['value'])
            df_coin['ma_6h'] = df_coin['value'].rolling(window=6).mean()
            spike_ratio = (df_coin['value'].iloc[-1] / df_coin['ma_6h'].iloc[-6]) if len(df_coin) >= 6 else 1
            
            save_to_cache(symbol, timeframe, trends_data)
            
            return {
                'coin': coin['name'],
                'symbol': symbol,
                'market_cap_rank': coin['market_cap_rank'],
                'trends_data': trends_data,
                'search_term': ' + '.join(search_terms),
                'spike_ratio': spike_ratio
            }
            
        except Exception as e:
            if "429" in str(e):
                st.sidebar.warning(f"Rate limit hit for {symbol}. Retrying ({attempt + 1}/{retries})...")
                time.sleep(30)  # Longer delay on rate limit
                continue
            st.sidebar.error(f"Error for {symbol}: {str(e)}")
            return None
    
    st.sidebar.error(f"Failed to fetch data for {symbol} after {retries} attempts.")
    return None

def get_recent_trends(results, hours=6):
    """Improved trending detection using multiple factors"""
    recent_trends = []
    
    for result in results:
        trends = result['trends_data']
        if trends:
            df = pd.DataFrame(trends)
            df['value'] = pd.to_numeric(df['value'])
            df['date'] = pd.to_datetime(df['date'])  # Ensure date is datetime
            
            # Get time windows
            recent = df.sort_values('date', ascending=False).head(hours)
            previous = df.sort_values('date', ascending=True).head(hours)
            
            if len(recent) < hours or len(previous) < hours:
                continue
                
            # Calculate metrics
            avg_recent = recent['value'].mean()
            avg_previous = previous['value'].mean()
            
            # Trending score combines multiple factors
            trending_score = (
                0.4 * avg_recent +
                0.4 * (avg_recent - avg_previous) +
                0.2 * result.get('spike_ratio', 1)
            )
            
            recent_trends.append({
                'coin': result['coin'],
                'symbol': result['symbol'],
                'search_term': result['search_term'],
                'trending_score': trending_score,
                'change_pct': ((avg_recent - avg_previous)/avg_previous)*100 if avg_previous != 0 else 0,
                'avg_interest': avg_recent
            })
    
    # Sort by trending score
    sorted_trends = sorted(recent_trends, key=lambda x: x['trending_score'], reverse=True)
    
    # Add visual indicators
    for trend in sorted_trends:
        if trend['change_pct'] > 50:
            trend['direction'] = '↑↑'
        elif trend['change_pct'] > 25:
            trend['direction'] = '↑'
        else:
            trend['direction'] = '→'
    
    return sorted_trends[:5]

def main():
    st.title("🚀 Trending Cryptocurrency in Thailand")
    
    # Add time window selector
    time_window = st.slider(
        "Select trending time window (hours):",
        min_value=3,
        max_value=24,
        value=6,
        step=3
    )
    
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
        existing_symbols = {coin['symbol'].upper() for coin in filtered_coins}
        for symbol in custom_symbols:
            if symbol and symbol not in existing_symbols:
                try:
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
    
    # In the main() function, update the processing loop:
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
            time.sleep(15)  # Add a 15-second delay between requests
        
        progress_bar.empty()

        
        if not results:
            st.error("No trend data available. Please try again later.")
            return

        # Get top 5 trending coins
        top_trending = get_recent_trends(results, hours=time_window)
        
        # Display top trending coins
        st.header("🔥 Top 5 Trending Coins")
        st.caption(f"Based on composite trending score (Last {time_window} hours)")
        
        # Create columns with color coding
        cols = st.columns(5)
        colors = ['#4CAF50', '#8BC34A', '#CDDC39', '#FFEB3B', '#FFC107']
        
        for i, (col, coin) in enumerate(zip(cols, top_trending)):
            with col:
                delta_color = "normal" if i < 2 else "off" if i < 4 else "inverse"
                st.metric(
                    label=f"**#{i+1}** {coin['symbol']}",
                    value=coin['coin'],
                    delta=f"{coin['change_pct']:.1f}% {coin['direction']}",
                    delta_color=delta_color
                )
                
                # Add mini sparkline
                spark_data = pd.DataFrame([p['value'] for p in next(r for r in results if r['symbol'] == coin['symbol'])['trends_data']][-24:])
                col.line_chart(spark_data, height=50, use_container_width=True)
        
        st.markdown("---")

        # Prepare data for visualization
        trend_data = []
        for result in results:
            for point in result['trends_data']:
                trend_data.append({
                    'Date': point['date'],
                    'Z-Score': point['value'],
                    'Cryptocurrency': f"{result['symbol']}"
                })

        df = pd.DataFrame(trend_data)

        # Add Data Formatting:
        if len(df) > 0:
            # Ensure 'Cryptocurrency' column has unique values
            unique_coins = df['Cryptocurrency'].unique()
            if len(unique_coins) == 0:
                st.warning("No valid cryptocurrencies found in the data.")
                return
            
            # Ensure 'Z-Score' column is numeric
            if not pd.api.types.is_numeric_dtype(df['Z-Score']):
                st.error("Invalid data format: Z-Score must be numeric.")
                st.write("Debug Data:")
                st.write(df)
                return

        #Handle Edge Cases in Data Fetching
        if len(results) == 0:
        st.error("No trend data available for any coin. Please try again later.")
        return
        
        #Debugging output
        st.write("Debug: Trend Data")
        st.write(df)

        # Create faceted line chart
        if len(df) > 0:
            try:
                # Create faceted line chart
                fig = px.line(
                    df,
                    x='Date',
                    y='Z-Score',
                    color='Cryptocurrency',
                    facet_row='Cryptocurrency',
                    height=200 * len(df['Cryptocurrency'].unique()),
                    title='Normalized Search Interest (Z-Scores)'
                )
                
                # Update layout
                fig.update_layout(
                    showlegend=False,
                    hovermode='x unified',
                    margin=dict(t=100),
                    yaxis_title="Z-Score"
                )
                
                # Configure subplots
                fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                fig.update_yaxes(matches=None, showticklabels=True)
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating plot: {str(e)}")
                st.write("Debug Data:")
                st.write(df)
        else:
            st.warning("No valid trend data available to plot.")
        
        # Add expanders for details
        with st.expander("Technical Details"):
            st.markdown("""
            **Methodology:**
            - Trends normalized using Z-scores (standard deviations from mean)
            - Trending score combines:
              * Recent interest (40%)
              * Momentum vs previous period (40%)
              * Recent spikes (20%)
            - Data cached for 1 hour to prevent API abuse
            """)
            
        if errors:
            with st.expander("Error Log"):
                st.write("Failed to process these coins:")
                for error in errors:
                    st.write(f"- {error.get('symbol', 'Unknown')}: {error.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()