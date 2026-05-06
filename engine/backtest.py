import pandas as pd
import numpy as np
from engine.optimizer import solve_markowitz

def run_backtest(returns_df, window=126, gamma=2.0, lambda_cost=0.01):
    n_days, n_assets = returns_df.shape
    portfolio_returns = []
    benchmark_returns = []
    dates = []
    
    # Start with an equal-weight allocation
    w_prev = np.ones(n_assets) / n_assets

    for i in range(window, n_days - 1):
        lookback_data = returns_df.iloc[i-window : i]
        next_day_returns = returns_df.iloc[i + 1]

        # 1. Optimize for today (passing in yesterday's weights)
        weights = solve_markowitz(lookback_data, gamma=gamma, w_prev=w_prev, lambda_cost=lambda_cost)
        
        # 2. Strategy Return
        port_ret = np.dot(weights.values, next_day_returns.values)
        
        # 3. Benchmark Return (Equal Weight 1/N)
        bench_ret = np.dot(np.ones(n_assets)/n_assets, next_day_returns.values)

        portfolio_returns.append(port_ret)
        benchmark_returns.append(bench_ret)
        dates.append(returns_df.index[i + 1])
        
        # 4. Update w_prev for the next loop iteration
        w_prev = weights.values

    # Return a DataFrame with both curves
    return pd.DataFrame({
        'Strategy': portfolio_returns,
        'Equal_Weight': benchmark_returns
    }, index=dates)