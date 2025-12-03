"""Test script to verify earliest data fetching."""

from datetime import datetime
from src.data.data_downloader import DataDownloader

# This demonstrates that when start_date=None, 
# the downloader will fetch from 2017-01-01 (earliest available)

downloader = DataDownloader(exchange_name="binance", market_type="future", data_dir="data")

print("Testing earliest data fetch...")
print("When start_date=None, the downloader will fetch from 2017-01-01")
print("This ensures we get all available historical data for the coin.")
print("\nNote: The actual earliest data depends on when the coin was listed on the exchange.")
print("For BTC, data goes back to 2017. For newer coins, it will start from their listing date.")
