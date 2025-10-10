import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Read CSV files:

EU = pd.read_csv("csv_files/EXPORT EU EUR.csv")

US = pd.read_csv("csv_files/EXPORT US EUR.csv")

# Select columns with assets, and prefix EU/US to each asset.

portfolios = list(EU.columns[5:36])

EU = EU[portfolios]

EU.columns = ["EU_" + p for p in portfolios]

US = US[portfolios]

US.columns = ["US_" + p for p in portfolios]

# Make a combined datafrme with all assets.

all_assets = pd.concat([EU,US],axis = 1)

def markowitz_min_obj(weights, cov_matrix):
    return(weights.T@cov_matrix@weights)

def markowitz_min_constraint(weights,mean_return,mu):
    return(weights.T@mean_return-mu)

def markowitz_max_obj(weights, mean_return):
    return(-weights.T@mean_return)

# if return >= 0 then variance less than sigma
def markowitz_max_constraint(weights, cov_matrix, sigma):
    return(sigma - np.matmul(np.matmul(np.transpose(weights),cov_matrix),weights))

def weights_sum_constr(weights):
    return(sum(weights)-1)




def markowitz(data, target, direction = "min"):
    cov_matrix = data.cov().to_numpy()
    mean_return = data.mean().to_numpy()

    w0 = np.zeros(len(mean_return)) + 1/len(mean_return)

    bound = [(0, 1)] * len(mean_return)

    if direction == "min":
        cons = [{'type': 'eq', 'fun': weights_sum_constr},
                {'type': 'ineq', 'fun': markowitz_min_constraint, 'args': (mean_return, target)}]
        
        res = minimize(markowitz_min_obj, w0, args = (cov_matrix,), constraints = cons, bounds = bound, method = "SLSQP")

    if direction == "max":
        cons = [{'type': 'eq', 'fun': weights_sum_constr},
                {'type': 'ineq', 'fun': markowitz_max_constraint, 'args': (cov_matrix, target)}]
        
        res = minimize(markowitz_max_obj, w0, args = (mean_return,), constraints = cons, bounds = bound, method = "SLSQP")
    
    if not res.success:
        print("Optimization failed:", res.message)
    
    return res




# --- Efficient Frontier Calculation ---

def efficient_frontier(data, n_points=150):

    mean_return = data.mean().to_numpy()
    cov_matrix = data.cov().to_numpy()

    mu_min, mu_max = mean_return.min(), mean_return.max()
    target_returns = np.linspace(mu_min, mu_max, n_points)

    frontier_returns = []
    frontier_risks = []
    frontier_weights = []

    for mu in target_returns:
        res = markowitz(data, mu, direction="min")
        if res.success:
            w = res.x
            port_return = w @ mean_return
            port_var = w.T @ cov_matrix @ w
            frontier_returns.append(port_return)
            frontier_risks.append(np.sqrt(port_var))
            frontier_weights.append(w)
        else:
            print(f"Optimization failed for target return {mu:.4f}: {res.message}")

    return np.array(frontier_risks), np.array(frontier_returns), np.array(frontier_weights)

test_data = all_assets.tail(240)

# --- Generate and plot the frontier ---
risks, returns, weights = efficient_frontier(test_data)

plt.figure(figsize=(10, 6))
plt.plot(risks, returns, 'b-', lw=2, label='Efficient Frontier')
plt.xlabel('Portfolio Risk (σ)')
plt.ylabel('Expected Return (μ)')
plt.title('Mean–Variance Efficient Frontier')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()



