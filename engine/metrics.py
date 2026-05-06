import numpy as np
import pandas as pd

def calculate_metrics(returns_series):
    """
    Computes Quant metrics for a given series of daily returns.
    """
    # 1. Annualized Return (assuming 252 trading days)
    # Using log returns property: total return = exp(sum)
    total_return = np.exp(returns_series.sum()) - 1
    days = len(returns_series)
    if days == 0:
        return {
            "Annualized Return": "0.00%",
            "Annualized Vol": "0.00%",
            "Sharpe Ratio": "0.00",
            "Max Drawdown": "0.00%"
        }
    ann_return = (1 + total_return) ** (252 / days) - 1
    
    # 2. Annualized Volatility
    ann_vol = returns_series.std() * np.sqrt(252)
    
    # 3. Sharpe Ratio (Risk-Free Rate assumed at 0 for simplicity)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    
    # 4. Max Drawdown
    cumulative = np.exp(returns_series.cumsum())
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        "Annualized Return": f"{ann_return:.2%}",
        "Annualized Vol": f"{ann_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}"
    }