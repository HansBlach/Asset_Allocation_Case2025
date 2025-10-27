import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Read CSV files:

EU_data = pd.read_csv("csv_files/long_EXPORT EU EUR.csv")

US_data = pd.read_csv("csv_files/long_EXPORT US EUR.csv")


# Define the objective and constraint functions for the minimization problem

def markowitz_min_obj(weights, cov_matrix):
    return(weights.T@cov_matrix@weights)

def markowitz_min_constraint(weights,mean_return,mu):
    return(weights.T@mean_return-mu)

# Define objective and constraint functions fo the maximization problem (it can only minimize so we max -obj

def markowitz_max_obj(weights, mean_return):
    return(-weights.T@mean_return)

def markowitz_max_constraint(weights, cov_matrix, sigma):
    return(sigma - np.matmul(np.matmul(np.transpose(weights),cov_matrix),weights))

# Define constraint that the weights sum to 1
def weights_sum_constr(weights):
    return(sum(weights)-1)

# Function that takes data and a target (either variance or mean depending on direction) and returns the scipy.optimize result object

def markowitz_optimizer(mean_return, cov_matrix, target, direction = "min", allow_short = False):
    # Add regularization to avoid singular covariance matrix
    cov_matrix_reg = cov_matrix + np.eye(len(mean_return)) * 1e-6
    
    w0 = np.zeros(len(mean_return)) + 1/len(mean_return)

    bound = [(0, 1)] * len(mean_return) # Ensures no shorting is allowed

    if allow_short:
        bound = [(-5, 5)] * len(mean_return)

    if direction == "min":
        cons = [{'type': 'eq', 'fun': weights_sum_constr},
                {'type': 'eq', 'fun': markowitz_min_constraint, 'args': (mean_return, target)}]
        
        res = minimize(markowitz_min_obj, w0, args = (cov_matrix_reg,), constraints = cons, bounds = bound, method = "SLSQP")

    if direction == "max":
        cons = [{'type': 'eq', 'fun': weights_sum_constr},
                {'type': 'ineq', 'fun': markowitz_max_constraint, 'args': (cov_matrix_reg, target)}]
        
        res = minimize(markowitz_max_obj, w0, args = (mean_return,), constraints = cons, bounds = bound, method = "SLSQP")
    
    # if not res.success:
    #     print("Optimization failed:", res.message)
    
    return res

# function that finds the empirical minimum variance frontier:

def efficient_frontier(mean_return, cov_matrix, n_points=150, allow_short = False):
    mu_min, mu_max = mean_return.min(), mean_return.max()

    target_returns = np.linspace(mu_min, mu_max, n_points)

    frontier_returns = []
    frontier_variance = []
    frontier_weights = []

    if allow_short:
        for mu in target_returns:
            res = markowitz_optimizer(mean_return, cov_matrix, mu, allow_short= True, direction="min")
            if res.success:
                w = res.x
                port_return = w @ mean_return
                port_var = w.T @ cov_matrix @ w
                frontier_returns.append(port_return)
                frontier_variance.append(np.sqrt(port_var))
                frontier_weights.append(w)
            # else:
            #     print(f"Optimization failed for target return {mu:.4f}: {res.message}")
    if not allow_short:
        for mu in target_returns:
            res = markowitz_optimizer(mean_return, cov_matrix, mu, direction="min")
            if res.success:
                w = res.x
                port_return = w @ mean_return
                port_var = w.T @ cov_matrix @ w
                frontier_returns.append(port_return)
                frontier_variance.append(np.sqrt(port_var))
                frontier_weights.append(w)
            # else:
            #     print(f"Optimization failed for target return {mu:.4f}: {res.message}")

    return np.array(frontier_variance), np.array(frontier_returns), np.array(frontier_weights)

# Function that plots the frontier:


def frontier_plotter(risks, returns):
    plt.figure(figsize=(10, 6))
    plt.plot(risks, returns, 'b-', lw=2)
    plt.xlabel('Portfolio Risk (Standard Deviation)')
    plt.ylabel('Expected Excess Return')
    plt.title('Mean–Variance Frontier')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(lambda v,pos: f"{v:.1f}%")
    plt.gca().yaxis.set_major_formatter(lambda v,pos: f"{v:.1f}%")
    # saves plot in high resolution
    plt.savefig("efficient_frontier.png", dpi=300)
    #plt.show()


us_factors, eu_factors = ["RM_RF", "MOM"], ["RM_RF", "MOM", "SMB"]

eu = EU_data[eu_factors]
# prefix EU_ to each column name to avoid duplicates
eu.columns = ["EU_" + col for col in eu.columns]
us = US_data[us_factors]
# prefix US_ to each column name to avoid duplicates
us.columns = ["US_" + col for col in us.columns]

data = pd.concat([eu, us], axis = 1)

mean_return = data.mean().to_numpy()
cov_matrix = data.cov().to_numpy()

variance, returns, weights = efficient_frontier(mean_return, cov_matrix, n_points=150)

# frontier_plotter(variance, returns)

# We want to invest in the portfolio with the highest sharpe ratio, that is the tangent portfolio
# but we can't use theoretical results because we have a no-shorting constraint
# Therefore we use the efficient frontier function to find (a portfolio close to) the tangent portfolio by taking the one with the highest sharpe ratio

def markowitz_strategy(eu_data, us_data, eu_factors,us_factors, mu_target = 0.5, n_points = 50, strategy = "tangent", allow_short = False):
    eu = eu_data[eu_factors]
    # prefix EU_ to each column name to avoid duplicates
    eu.columns = ["EU_" + col for col in eu.columns]
    us = us_data[us_factors]
    # prefix US_ to each column name to avoid duplicates
    us.columns = ["US_" + col for col in us.columns]

    data = pd.concat([eu, us], axis = 1)

    mean_return = data.mean().to_numpy()
    cov_matrix = data.cov().to_numpy()

    if strategy == "tangent":
        variance, returns, weights = efficient_frontier(mean_return, cov_matrix, n_points, allow_short= allow_short)
        
        # Check if we got valid results
        if len(variance) == 0:
            print("Warning: No valid frontier points found, using equal weights")
            return np.ones(len(mean_return)) / len(mean_return)

        # Avoid division by zero
        sharpe = np.divide(returns, variance, out=np.zeros_like(returns), where=variance!=0)

        idx_tangent = np.argmax(sharpe)

        return(weights[idx_tangent])
    
    if strategy == "min_variance":
        variance, returns, weights = efficient_frontier(mean_return, cov_matrix, n_points, allow_short= allow_short)
        
        # Check if we got valid results
        if len(variance) == 0:
            print("Warning: No valid frontier points found, using equal weights")
            return np.ones(len(mean_return)) / len(mean_return)

        idx = np.argmin(variance)

        return(weights[idx])

    if strategy == "max_return":
        max_return = max(mean_return)
        result = markowitz_optimizer(mean_return, cov_matrix, mu_target, direction = "min")
        if result.success:
            return(result.x)
        else:
            #print("Warning: Fixed strategy optimization failed, using equal weights")
            return np.ones(len(mean_return)) / len(mean_return)   

    
    if strategy == "fixed":
        result = markowitz_optimizer(mean_return, cov_matrix, mu_target, direction = "min")
        if result.success:
            return(result.x)
        else:
            #print("Warning: Fixed strategy optimization failed, using equal weights")
            return np.ones(len(mean_return)) / len(mean_return)


# Test how many points are needed to get a "tight enough" grid

# test_list = [5, 20, 50, 100]

# for n in test_list:
#     weights, sharpe, returns = markowitz_strategy(all_assets.tail(36), n_points = n)
#     print(n, sharpe, returns)

# risks, returns, weights = efficient_frontier(all_assets)

# frontier_plotter(risks, returns)


def rolling_dataframes(data, window_size):
    return [data.iloc[i:i+window_size] for i in range(data.shape[0] - window_size + 1)]

def markowitz_historical(eu_data, us_data, eu_factors, us_factors, window, strategy = "tangent", mu_target = 0.05, n_points = 50, allow_short = False):
    rolling_data_eu = rolling_dataframes(eu_data, window)
    rolling_data_us = rolling_dataframes(us_data, window)

    eu = eu_data[eu_factors]
    # prefix EU_ to each column name to avoid duplicates
    eu.columns = ["EU_" + col for col in eu.columns]
    us = us_data[us_factors]
    # prefix US_ to each column name to avoid duplicates
    us.columns = ["US_" + col for col in us.columns]

    data = pd.concat([eu, us], axis = 1)

    N = eu_data.shape[0]

    actual_return = np.zeros(N - window)

    for i in range(N - window):
        weights = markowitz_strategy(rolling_data_eu[i], rolling_data_us[i], eu_factors, us_factors, mu_target, n_points, strategy, allow_short)
        next_per_return = np.array(data.iloc[window + i, :])
        actual_return[i] = weights.T @ next_per_return
    
    return actual_return

# Function to convert returns to value development

def returns_to_value(returns, start_value=1.0):
    value_data = np.zeros(returns.shape)
    value_data[0] = start_value

    for i in range(1, returns.shape[0]):
        value_data[i] = value_data[i-1] * (1 + returns[i]/100)
    
    return value_data




def markowitz_result_plot(data, window, n_points):
    # plots the value development of the strategy

    plt.figure(figsize=(10, 6))
    for i in n_points:
            result = markowitz_historical(data, window, n_points = i)
            value_development = returns_to_value(result)
            plt.plot(value_development, label=f'Markowitz Strategy, points = {i}')
    plt.xlabel('Time (Months)')
    plt.ylabel('Portfolio Value')
    plt.title('Value Development of Markowitz Strategy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


# n_points = [10, 20, 50]

# markowitz_result_plot(EU, 36, n_points)



