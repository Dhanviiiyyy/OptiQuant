# OptiQuant

OptiQuant is a quantitative portfolio optimization platform that implements classical mean-variance portfolio optimization using convex optimization techniques. The application enables users to construct, backtest, and analyze investment portfolios using historical market data through an interactive web interface.

The project combines Modern Portfolio Theory, convex optimization, and quantitative risk analysis to evaluate portfolio performance under different optimization strategies.

---

## Features

- Mean-Variance (Markowitz) Portfolio Optimization
- Global Minimum Variance (GMV) Portfolio
- Ridge-Regularized Portfolio Optimization (L2 Regularization)
- L1 Transaction Cost Penalty for turnover control
- Rolling-window walk-forward backtesting
- Benchmark comparison against an Equal-Weight Portfolio and the BSE Sensex
- Monte Carlo simulation for one-year portfolio projections
- Interactive Streamlit dashboard
- Institutional portfolio risk and performance metrics

---

## Optimization Methods

### Mean-Variance Optimization

Implements Harry Markowitz's Modern Portfolio Theory by maximizing expected return while penalizing portfolio variance.

Objective:

```
maximize

μᵀw − γwᵀΣw
```

where:

- μ denotes expected asset returns
- Σ denotes the covariance matrix
- γ denotes the investor's risk aversion parameter

---

### Global Minimum Variance

Constructs the portfolio with minimum variance while ignoring expected return estimates.

This approach is often preferred when expected returns are considered noisy or unreliable.

---

### Ridge-Regularized Portfolio

Applies L2 regularization to discourage extreme portfolio allocations and improve out-of-sample stability.

Objective includes

```
δ ||w||²
```

where δ controls the strength of the regularization.

---

### Transaction Cost Penalty

Models portfolio turnover using an L1 penalty

```
λ ||wₜ − wₜ₋₁||₁
```

which discourages unnecessary rebalancing and approximates trading costs.

---

## Backtesting Framework

The strategy is evaluated using a rolling-window walk-forward backtest.

For every trading day:

1. Estimate expected returns and covariance using historical data.
2. Solve the portfolio optimization problem.
3. Allocate portfolio weights.
4. Evaluate performance on unseen market data.
5. Repeat throughout the investment horizon.

This methodology avoids look-ahead bias and better reflects practical portfolio management.

---

## Risk Metrics

The dashboard reports:

- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
- Calmar Ratio
- Value-at-Risk (VaR)
- Conditional Value-at-Risk (CVaR)
- Portfolio Beta relative to the BSE Sensex

---

## Monte Carlo Simulation

Following the historical backtest, the application generates one-year forward portfolio projections using Monte Carlo simulation based on the empirical distribution of portfolio returns.

The dashboard visualizes:

- 5th percentile (Pessimistic Scenario)
- Median Scenario
- 95th percentile (Optimistic Scenario)

---

## Technology Stack

### Backend

- Python
- NumPy
- Pandas
- CVXPY
- yfinance

### Optimization

- Convex Optimization
- Quadratic Programming
- Markowitz Portfolio Theory
- L1 Regularization
- L2 Regularization

### Frontend

- Streamlit

---

## Project Structure

```
OptiQuant/
│
├── app.py
│
├── data/
│   └── loader.py
│
├── engine/
│   ├── optimizer.py
│   ├── backtest.py
│   ├── metrics.py
│   └── simulation.py
│
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd OptiQuant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Future Work

- Black-Litterman Portfolio Optimization
- Hierarchical Risk Parity
- Efficient Frontier Visualization
- Multi-period Portfolio Optimization
- Factor-based Risk Models
- Transaction Cost Calibration
- Risk Attribution Analysis

---

## Disclaimer

This project was developed for educational purposes to demonstrate concepts in quantitative finance and convex optimization. It should not be interpreted as financial or investment advice.
