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
    
    with st.spinner("Optimizing Portfolio..."):
        prices, rets = get_data(ticker_list, '2023-01-01', '2026-05-01')
        
        # Update function call to include lambda
        returns_df = run_backtest(rets, window=window, gamma=gamma, lambda_cost=lambda_cost)
        
        # Calculate metrics ONLY for the Strategy column
        metrics = calculate_metrics(returns_df['Strategy'])
        
        # --- Display Results ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sharpe Ratio", metrics["Sharpe Ratio"])
        col2.metric("Ann. Return", metrics["Annualized Return"])
        col3.metric("Max Drawdown", metrics["Max Drawdown"])
        col4.metric("Volatility", metrics["Annualized Vol"])

        # 💥 Plot both curves!
        st.subheader("Cumulative Wealth: Strategy vs. Equal Weight")
        equity_curves = np.exp(returns_df.cumsum())
        st.line_chart(equity_curves)