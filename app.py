import streamlit as st
import pandas as pd
import numpy as np
from data.loader import get_data
from engine.backtest import run_backtest
from engine.metrics import calculate_metrics
from engine.simulation import project_monte_carlo

# --- UI Setup ---
st.set_page_config(page_title="OptiQuant Dashboard", layout="wide")
st.title("OptiQuant: Multi-Period Portfolio Optimizer")

# --- Sidebar Inputs ---
st.sidebar.header("Strategy Parameters")
tickers = st.sidebar.text_input("Enter Tickers", "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS")

st.sidebar.subheader("Optimization Engine")
method_options = {
    "mvo": "Mean-Variance (MVO)", 
    "gmv": "Global Min Variance (GMV)", 
    "ridge": "Ridge Regularized (L2)"
}
method = st.sidebar.selectbox(
    "Select Paradigm", 
    list(method_options.keys()),
    format_func=lambda x: method_options[x]
)

gamma = st.sidebar.slider("Risk Aversion (Gamma)", 0.1, 10.0, 2.0)
window = st.sidebar.selectbox("Look-back Window (Days)", [63, 126, 252], index=1)

st.sidebar.subheader("Regularization & Constraints")
lambda_cost = st.sidebar.slider("L1 Transaction Cost (Lambda)", 0.0000, 0.0100, 0.0005, step=0.0001, format="%.4f")
lambda_ridge = st.sidebar.slider("L2 Ridge Penalty (Delta)", 0.00, 0.20, 0.05, step=0.01)

# Sidebar info box
st.sidebar.markdown("---")
st.sidebar.info("This dashboard compares multiple convex portfolio optimization strategies using rolling-window backtesting, transaction-cost modeling, and Monte Carlo stress testing.")

if st.sidebar.button("Run Backtest"):
    ticker_list = [t.strip() for t in tickers.split(",")]
    
    with st.spinner("Fetching market data and running convex solver..."):
        # 1. Fetch Data
        prices, asset_rets, bench_rets = get_data(ticker_list, '2023-01-01', '2026-05-01')
        
        if asset_rets.empty or len(asset_rets) <= window:
            st.error(f"Not enough data fetched. Need at least {window + 1} days, got {len(asset_rets)}.")
        else:
            # 2. Run Walk-Forward Engine
            returns_df = run_backtest(
                asset_rets, 
                bench_returns=bench_rets, 
                window=window, 
                method=method, 
                gamma=gamma, 
                lambda_cost=lambda_cost,
                lambda_ridge=lambda_ridge
            )
            
            # 3. Calculate Institutional Metrics
            metrics = calculate_metrics(returns_df['Strategy'], returns_df.get('Sensex_Benchmark'))
            
            # --- Optimization Summary Card ---
            st.subheader("⚙️ Optimization Summary")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Optimization Method", method_options[method])
            s2.metric("Risk Aversion (γ)", f"{gamma:.1f}")
            s3.metric("Rolling Window", f"{window} Days")
            s4.metric("Number of Assets", len(ticker_list))

            s5, s6, s7, s8 = st.columns(4)
            s5.metric("Transaction Cost λ", f"{lambda_cost:.4f}")
            s6.metric("Ridge Penalty λ", f"{lambda_ridge:.2f}" if method == "ridge" else "N/A")
            s7.metric("Benchmark Used", "Sensex")
            s8.empty() # Placeholder for alignment

            st.markdown("---")
            
            # --- Display Risk & Performance Card ---
            st.subheader("📊 Institutional Risk & Performance Card")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Sharpe / Sortino", f"{metrics.get('Sharpe Ratio','0')} / {metrics.get('Sortino Ratio','0')}")
            m2.metric("Ann. Return", metrics.get("Ann. Return", "0%"))
            m3.metric("Max Drawdown", metrics.get("Max Drawdown", "0%"))
            m4.metric("Market Beta (Sensex)", metrics.get("Sensex Beta", "N/A"))

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Calmar Ratio", metrics.get("Calmar Ratio", "0"))
            m6.metric("Annualized Volatility", metrics.get("Ann. Volatility", "0%"))
            m7.metric("Daily VaR (95%)", metrics.get("Daily VaR (95%)", "0%"))
            m8.metric("Daily CVaR (95%)", metrics.get("Daily CVaR (95%)", "0%"))

            # --- Visualizations ---
            st.markdown("---")
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader("📈 Portfolio Growth vs Benchmark")
                # Convert log returns back to cumulative price percentage
                equity_curves = np.exp(returns_df.cumsum())
                st.line_chart(equity_curves)

            with col_chart2:
                st.subheader("🌊 Portfolio Drawdown")
                roll_max = equity_curves["Strategy"].cummax()
                drawdown = (equity_curves["Strategy"] - roll_max) / roll_max
                st.area_chart(drawdown)

            st.subheader("🎲 Monte Carlo Forward Scenarios")
            mc_df = project_monte_carlo(returns_df["Strategy"])
            st.line_chart(mc_df)

            # --- Educational Section ---
            st.markdown("---")
            
            with st.expander("📖 Optimization Methods"):
                st.markdown("""
                **Mean-Variance Optimization (MVO)**
                * Maximizes expected return while penalizing portfolio variance.
                * Uses the risk-aversion parameter γ to balance return and risk.

                **Global Minimum Variance (GMV)**
                * Ignores expected returns.
                * Finds the portfolio with the minimum possible variance.
                * Useful when expected return estimates are noisy.

                **Ridge-Regularized MVO**
                * Extends classical Markowitz by adding an L2 penalty on portfolio weights.
                * Encourages smoother, less concentrated allocations.
                * Improves robustness against noisy covariance and return estimates.
                """)
                
            with st.expander("📖 Regularization & Transaction Costs"):
                st.markdown("""
                **L1 Transaction Cost Regularization**
                * Penalizes changes in portfolio weights between consecutive rebalancing periods.
                * Models transaction costs and discourages excessive trading.
                * Helps produce more realistic portfolio turnover.

                **L2 Ridge Regularization**
                * Penalizes large portfolio weights.
                * Produces more diversified and stable portfolios.
                * Similar in spirit to ridge regression in machine learning.
                """)

            with st.expander("📖 Risk Metrics"):
                st.markdown("""
                * **Sharpe Ratio**: Measures the excess return per unit of total risk (volatility).
                * **Sortino Ratio**: Measures the excess return per unit of downside risk, ignoring positive volatility.
                * **Maximum Drawdown**: Represents the maximum observed loss from a peak to a trough before a new peak is attained.
                * **Calmar Ratio**: Evaluates the annualized return relative to the maximum drawdown risk.
                * **Value at Risk (VaR)**: Estimates the maximum expected loss over a specific timeframe at a given confidence level (e.g., 95%).
                * **Conditional Value at Risk (CVaR)**: Calculates the expected average loss given that the VaR threshold has been breached.
                """)

            with st.expander("📖 Monte Carlo Simulation"):
                st.markdown("""
                * Historical strategy returns are used to estimate mean and volatility.
                * Future return paths are generated using Monte Carlo simulation.
                * The plotted median, optimistic, and pessimistic paths illustrate possible future portfolio trajectories rather than predictions.
                """)