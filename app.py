import streamlit as st
import pandas as pd
import numpy as np
from data.loader import get_data
from engine.backtest import run_backtest
from engine.metrics import calculate_metrics

# --- UI Setup ---
st.set_page_config(page_title="OptiQuant Dashboard", layout="wide")
st.title("🧭 OptiQuant: Multi-Period Portfolio Optimizer")
st.sidebar.header("Strategy Parameters")

# --- Inputs ---
tickers = st.sidebar.text_input("Enter Tickers", "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS")
gamma = st.sidebar.slider("Risk Aversion (Gamma)", 0.1, 10.0, 2.0)
window = st.sidebar.selectbox("Look-back Window (Days)", [63, 126, 252], index=1)

# 💥 New Slider for Transaction Costs
# Change the lambda slider to have a smaller default and step size
lambda_cost = st.sidebar.slider("Transaction Cost Penalty (Lambda)", 0.0000, 0.0050, 0.0005, step=0.0001, format="%.4f")
if st.sidebar.button("Run Backtest"):
    ticker_list = [t.strip() for t in tickers.split(",")]
    
    with st.spinner("Fetching data and optimizing..."):
        prices, rets = get_data(ticker_list, '2023-01-01', '2026-05-01')
        
        # Checking if yfinance actually gave us enough data
        if rets.empty or len(rets) <= window:
            st.error(f"Not enough data fetched from Yahoo Finance. We need at least {window + 1} days of data, but got {len(rets)}. Try again in a minute or try different tickers.")
        else:
            # Run Strategy
            returns_df = run_backtest(rets, window=window, gamma=gamma, lambda_cost=lambda_cost)
            
            # Calculate Metrics
            metrics = calculate_metrics(returns_df['Strategy'])
            
            # --- Display Results ---
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sharpe Ratio", metrics["Sharpe Ratio"])
            col2.metric("Ann. Return", metrics["Annualized Return"])
            col3.metric("Max Drawdown", metrics["Max Drawdown"])
            col4.metric("Volatility", metrics["Annualized Vol"])

            # Plot Equity Curve
            st.subheader("Cumulative Wealth: Strategy vs. Equal Weight")
            equity_curves = np.exp(returns_df.cumsum())
            st.line_chart(equity_curves)

            # --- Educational Section ---
            st.markdown("---")
            st.subheader("📚 Under the Hood: How it Works")
            
            with st.expander("What do these metrics mean?"):
                st.markdown("""
                * **Sharpe Ratio:** The ultimate "Quant" metric. It measures how much excess return you are getting for every unit of risk (volatility) you take. A higher number is better. 
                * **Ann. Return:** The average percentage the portfolio is expected to grow per year.
                * **Max Drawdown:** The "pain factor." It measures the largest single drop from a previous peak to the lowest trough. If you invested at the absolute worst time, this is the maximum percentage you would have lost.
                * **Volatility:** The annualized standard deviation of daily returns. Higher volatility means a bumpier, more unpredictable ride.
                """)
                
            with st.expander("The Methodology (Convex Optimization)"):
                st.markdown("""
                This engine uses **Disciplined Convex Programming (DCP)** to dynamically optimize asset allocation, simulating a real-world quantitative trading strategy.

                1. **The Walk-Forward Loop:** The model doesn't peak into the future. At each time step, it only looks at the previous `N` days (the look-back window) to estimate expected returns ($\mu$) and asset covariance ($\Sigma$).
                2. **Mean-Variance Optimization:** Using the `CVXPY` solver, it finds the exact portfolio weights ($w$) that maximize expected return while penalizing variance (controlled by the **Risk Aversion** $\gamma$ slider).
                3. **Transaction Costs:** Real markets aren't free. We apply an $L_1$-norm penalty to the objective function: $\lambda \|w_t - w_{t-1}\|_1$. This forces the optimizer to be "lazy," only executing a trade if the expected alpha strictly outweighs the broker fees (controlled by the **Lambda** slider).
                """)