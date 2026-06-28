import numpy as np
import pandas as pd

def project_monte_carlo(daily_returns, days_forward=252, n_simulations=500):
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    
    # Generate random daily shocks from normal distribution
    simulated_shocks = np.random.normal(mu, sigma, size=(days_forward, n_simulations))
    
    # Compound returns starting from final backtest equity value
    paths = np.exp(np.cumsum(simulated_shocks, axis=0))
    
    # Extract 5th (Worst Case), 50th (Median), and 95th (Best Case) percentiles
    p5 = np.percentile(paths, 5, axis=1)
    p50 = np.percentile(paths, 50, axis=1)
    p95 = np.percentile(paths, 95, axis=1)
    
    return pd.DataFrame({"Pessimistic (VaR 95%)": p5, "Expected Median": p50, "Optimistic (95%)": p95})