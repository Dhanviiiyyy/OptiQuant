import numpy as np
import pandas as pd
from engine.optimizer import solve_markowitz


def run_backtest(
    returns_df,
    bench_returns=None,
    window=126,
    method="mvo",
    gamma=2.0,
    lambda_cost=0.01,
    lambda_ridge=0.05,
):
    """
    Walk-Forward Portfolio Backtesting Engine.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Historical daily log returns of the asset universe.
    bench_returns : pd.Series, optional
        Historical daily log returns of the market benchmark (e.g., Sensex).
    window : int
        Rolling look-back window size in trading days.
    method : str
        Optimization method ('mvo', 'gmv', 'ridge').
    gamma : float
        Risk aversion parameter.
    lambda_cost : float
        L1 transaction cost penalty.
    lambda_ridge : float
        L2 ridge regularization strength.

    Returns
    -------
    pd.DataFrame
        Out-of-sample daily returns for Strategy, Equal Weight (1/N), and optional Benchmark.
    """
    n_days, n_assets = returns_df.shape
    portfolio_returns = []
    equal_weight_returns = []
    benchmark_returns = []
    dates = []

    # Start with an equal-weight allocation prior to Day 1 of backtest
    w_prev = np.ones(n_assets) / n_assets

    # ------------------------------------------------------------------
    # Walk-Forward Loop
    # ------------------------------------------------------------------
    # NOTE: Fixed off-by-one indexing bug. We loop from `window` to `n_days`.
    # At step `i`, lookback uses rows [i-window : i] (strictly past data),
    # and realized PnL is evaluated on row [i] (today's actual market return).
    for i in range(window, n_days):
        lookback_data = returns_df.iloc[i - window : i]
        current_day_returns = returns_df.iloc[i]

        # 1. Optimize portfolio for today using yesterday's closing weights
        weights = solve_markowitz(
            lookback_data,
            method=method,
            gamma=gamma,
            w_prev=w_prev,
            lambda_cost=lambda_cost,
            lambda_ridge=lambda_ridge,
        )

        # 2. Realized Strategy Return (w^T * r_today)
        port_ret = np.dot(weights.values, current_day_returns.values)

        # 3. Realized Equal-Weight Benchmark Return (1/N * r_today)
        eq_ret = np.dot(np.ones(n_assets) / n_assets, current_day_returns.values)

        portfolio_returns.append(port_ret)
        equal_weight_returns.append(eq_ret)
        dates.append(returns_df.index[i])

        # 4. Track optional external market benchmark (e.g., BSE Sensex)
        if bench_returns is not None and len(bench_returns) == n_days:
            benchmark_returns.append(bench_returns.iloc[i])

        # 5. Update w_prev for the next day's turnover calculation
        w_prev = weights.values

    # ------------------------------------------------------------------
    # Assemble Performance Output
    # ------------------------------------------------------------------
    output_dict = {
        "Strategy": portfolio_returns,
        "Equal_Weight": equal_weight_returns,
    }

    if benchmark_returns:
        output_dict["Sensex_Benchmark"] = benchmark_returns

    return pd.DataFrame(output_dict, index=dates)