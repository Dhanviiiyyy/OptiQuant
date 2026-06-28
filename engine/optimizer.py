import cvxpy as cp
import numpy as np
import pandas as pd


def solve_markowitz(
    returns_df,
    method="mvo",
    gamma=2.0,
    w_prev=None,
    lambda_cost=0.01,
    lambda_ridge=0.05,
):
    """
    Portfolio Optimization Engine

    Methods
    -------
    mvo    : Classical Mean-Variance Optimization
    gmv    : Global Minimum Variance Portfolio
    ridge  : Ridge-Regularized Mean-Variance Optimization

    Parameters
    ----------
    gamma : Risk aversion parameter
    lambda_cost : L1 transaction cost penalty
    lambda_ridge : L2 ridge regularization strength
    """

    # ------------------------------------------------------------------
    # Estimate expected returns and covariance matrix
    # ------------------------------------------------------------------
    mu = returns_df.mean().values
    Sigma = returns_df.cov().values
    n = len(mu)

    # Portfolio weights (decision variable)
    w = cp.Variable(n)

    # Portfolio expected return
    expected_return = mu @ w

    # Portfolio variance
    risk = cp.quad_form(w, cp.psd_wrap(Sigma))

    # ------------------------------------------------------------------
    # L1 Transaction Cost Penalty
    # ------------------------------------------------------------------
    if w_prev is not None and lambda_cost > 0:
        transaction_cost = lambda_cost * cp.norm(w - w_prev, 1)
    else:
        transaction_cost = 0

    # ------------------------------------------------------------------
    # Optimization Objective
    # ------------------------------------------------------------------
    if method == "gmv":

        # Global Minimum Variance
        objective = cp.Minimize(
            risk + transaction_cost
        )

    elif method == "ridge":

        # Ridge (L2) Regularization
        ridge_penalty = lambda_ridge * cp.sum_squares(w)

        objective = cp.Maximize(
            expected_return
            - gamma * risk
            - transaction_cost
            - ridge_penalty
        )

    else:

        # Classical Markowitz Mean-Variance
        objective = cp.Maximize(
            expected_return
            - gamma * risk
            - transaction_cost
        )

    # ------------------------------------------------------------------
    # Portfolio Constraints
    # ------------------------------------------------------------------
    constraints = [
        cp.sum(w) == 1,   # Fully invested
        w >= 0,           # Long only
        w <= 0.40         # Max 40% allocation to one asset
    ]

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.OSQP)

        if w.value is None:
            raise ValueError("No feasible solution found.")

        return pd.Series(w.value, index=returns_df.columns)

    except Exception as e:

        print(f"Optimization Warning: {e}")
        print("Falling back to equal-weight portfolio.")

        return pd.Series(
            np.ones(n) / n,
            index=returns_df.columns
        )