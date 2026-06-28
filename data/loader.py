import yfinance as yf
import pandas as pd
import numpy as np
import time

def get_data(tickers, start_date, end_date, benchmark="^BSESN"):
    all_symbols = tickers + [benchmark]
    print(f"Attempting to download assets and benchmark: {all_symbols}")
    
    for attempt in range(3):
        try:
            data = yf.download(all_symbols, start=start_date, end=end_date, progress=False)
            if data.empty:
                time.sleep(1)
                continue

            prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
            prices = prices.ffill().dropna()

            # Separate asset universe from market benchmark
            asset_prices = prices[tickers]
            bench_prices = prices[[benchmark]]

            asset_returns = np.log(asset_prices / asset_prices.shift(1)).dropna()
            bench_returns = np.log(bench_prices / bench_prices.shift(1)).dropna()

            # Align indices perfectly via inner join
            aligned = pd.concat([asset_returns, bench_returns], axis=1, join="inner").dropna()
            
            return prices[tickers], aligned[tickers], aligned[benchmark]

        except Exception as e:
            print(f"Download failed: {e}")
            time.sleep(1)

    return pd.DataFrame(), pd.DataFrame(), pd.Series()