import cvxpy as cp
import numpy as np
import pandas as pd

def solve_markowitz(returns_df, gamma=0.5, w_prev=None, lambda_cost=0.01):
    """
    Mean-Variance Optimization with L1 Transaction Costs.
    """
    mu = returns_df.mean().values
    Sigma = returns_df.cov().values
    n = len(mu)

    w = cp.Variable(n)
    
    expected_return = mu @ w
    risk = cp.quad_form(w, cp.psd_wrap(Sigma))
    
    # 💥 The "Boyd" Signal: L1 Transaction Cost Penalty
    if w_prev is not None and lambda_cost > 0:
        transaction_cost = lambda_cost * cp.norm(w - w_prev, 1)
    else:
        transaction_cost = 0

    # Objective: Maximize Return, Minimize Risk, Minimize Turnover
    objective = cp.Maximize(expected_return - gamma * risk - transaction_cost)

    constraints = [
        cp.sum(w) == 1, 
        w >= 0,
        w <= 0.4  # Max 40% in one stock to enforce diversification
    ]

    prob = cp.Problem(objective, constraints)
    
    try:
        # OSQP is a fast solver for Quadratic Programs
        prob.solve()
        if w.value is None:
            raise ValueError("Solver failed to find optimal weights")
        return pd.Series(w.value, index=returns_df.columns)
    except Exception as e:
        # Fallback to equal weight if solver fails for a day
        print(f"Solver Error: {e}")
        return pd.Series(np.ones(n)/n, index=returns_df.columns)