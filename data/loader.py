import yfinance as yf
import pandas as pd
import numpy as np
import time

def get_data(tickers, start_date, end_date):
    print(f"Attempting to download: {tickers}")
    
    # We'll try a few times in case the 'database is locked' error is transient
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # download returns a MultiIndex DataFrame for multiple tickers
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                print(f"Attempt {attempt+1}: No data returned. Retrying...")
                time.sleep(1)
                continue

            # Extract Adjusted Close
            # yfinance sometimes returns different structures based on version
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            else:
                prices = data['Close']

            # Basic cleaning
            prices = prices.ffill().dropna()

            # Logic Check: If we asked for 4 stocks, do we have 4 columns?
            if isinstance(prices, pd.DataFrame) and prices.shape[1] < len(tickers):
                print(f"Missing some tickers. Found: {list(prices.columns)}")

            # Calculate Log Returns
            returns = np.log(prices / prices.shift(1)).dropna()
            
            return prices, returns

        except Exception as e:
            print(f" Attempt {attempt+1} failed: {e}")
            time.sleep(1)

    return pd.DataFrame(), pd.DataFrame()

if __name__ == "__main__":
    # Test with these specific tickers
    stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS']
    prices, rets = get_data(stocks, '2023-01-01', '2026-05-01')
    
    if not rets.empty:
        print("\nSuccess! Correlation Matrix:")
        print(rets.corr())
        print("\nSample Returns:")
        print(rets.head())
    else:
        print("\nTotal failure. Check your internet connection or ticker names.")