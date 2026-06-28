import numpy as np
import pandas as pd


def calculate_metrics(strategy_returns, benchmark_returns=None):
    """
    Calculate portfolio performance and risk metrics.

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily log returns of the strategy.

    benchmark_returns : pd.Series, optional
        Daily log returns of the benchmark (e.g. Sensex).

    Returns
    -------
    dict
        Dictionary of formatted performance metrics.
    """

    if strategy_returns.empty:
        return {}

    # -----------------------------
    # Annualized Return
    # -----------------------------
    total_return = np.exp(strategy_returns.sum()) - 1
    days = len(strategy_returns)

    ann_return = (1 + total_return) ** (252 / days) - 1

    # -----------------------------
    # Annualized Volatility
    # -----------------------------
    ann_vol = strategy_returns.std() * np.sqrt(252)

    # -----------------------------
    # Sharpe Ratio
    # -----------------------------
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # -----------------------------
    # Sortino Ratio
    # -----------------------------
    downside = strategy_returns[strategy_returns < 0]

    if len(downside) > 0:
        downside_dev = downside.std() * np.sqrt(252)
        sortino = ann_return / downside_dev if downside_dev > 0 else 0
    else:
        sortino = 0

    # -----------------------------
    # Equity Curve & Drawdown
    # -----------------------------
    equity = np.exp(strategy_returns.cumsum())

    running_max = equity.cummax()

    drawdown = (equity - running_max) / running_max

    max_drawdown = drawdown.min()

    # -----------------------------
    # Calmar Ratio
    # -----------------------------
    calmar = (
        ann_return / abs(max_drawdown)
        if max_drawdown != 0
        else 0
    )

    # -----------------------------
    # Historical VaR & CVaR
    # -----------------------------
    var95 = np.percentile(strategy_returns, 5)

    cvar95 = strategy_returns[
        strategy_returns <= var95
    ].mean()

    # -----------------------------
    # Portfolio Beta
    # -----------------------------
    beta = None

    if benchmark_returns is not None:

        aligned = pd.concat(
            [strategy_returns, benchmark_returns],
            axis=1
        ).dropna()

        if len(aligned) > 2:

            cov = np.cov(
                aligned.iloc[:, 0],
                aligned.iloc[:, 1]
            )[0, 1]

            bench_var = np.var(aligned.iloc[:, 1])

            if bench_var > 0:
                beta = cov / bench_var

    # -----------------------------
    # Output
    # -----------------------------
    metrics = {
        "Ann. Return": f"{ann_return:.2%}",
        "Ann. Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Sortino Ratio": f"{sortino:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Calmar Ratio": f"{calmar:.2f}",
        "Daily VaR (95%)": f"{var95:.2%}",
        "Daily CVaR (95%)": f"{cvar95:.2%}",
    }

    if beta is not None:
        metrics["Sensex Beta"] = f"{beta:.2f}"

    return metrics