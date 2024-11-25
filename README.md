# Cryptocurrency Trends Analysis 🚀

This Streamlit application analyzes Google Trends data for the top cryptocurrencies in Thailand. It combines data from CoinGecko and Google Trends to create an interactive visualization of cryptocurrency search interest over time.

## Features 🌟

- Tracks top 20 non-stablecoin cryptocurrencies by market cap
- Real-time Google Trends data for Thailand
- Interactive 3-day trend analysis with hourly granularity
- Top 5 trending coins display
- Beautiful Plotly visualization with interactive legend
- Smart caching system for API responses
- Comprehensive error handling

## Demo 📊

The app shows:
- Top 5 trending cryptocurrencies based on recent search interest
- Detailed line chart showing search trends over time
- Customizable analysis timeframe (1-72 hours)
- Processing details and error logs in collapsible sections

## Setup 🛠️

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/trending-coins.git
   cd trending-coins
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Requirements 📋

- Python 3.11+
- Internet connection for API access

## Dependencies 📦

- streamlit==1.24.0
- pandas==2.0.3
- plotly==5.15.0
- pycoingecko==3.1.0
- pytrends==4.9.0

## Features in Detail 🔍

### Data Sources
- CoinGecko API for cryptocurrency market data
- Google Trends API for search interest data
- Geolocation targeting for Thailand (geo='TH')

### Cryptocurrency Filtering
- Excludes stablecoins (USDT, USDC, etc.)
- Dynamic filtering based on market cap
- Custom search term mapping for better results

### Performance Optimizations
- Local file-based caching system
- Rate limiting protection
- Sequential processing to avoid API errors

### Error Handling
- Graceful failure modes
- Detailed error logging
- Collapsible error display
- Progress indicators

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 💡

- CoinGecko for cryptocurrency market data
- Google Trends for search interest data
- Streamlit for the amazing web framework
