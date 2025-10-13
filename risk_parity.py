import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load the data from CSV files
df_EU = pd.read_csv("csv_files/EXPORT EU EUR.csv")
df_USEUR = pd.read_csv("csv_files/EXPORT US EUR.csv")
df_US = pd.read_csv("csv_files/EXPORT US USD.csv")
def return_variance(returns: np.ndarray) -> float:
    """Calculate variance of a vector of returns."""
    return np.var(returns, ddof=1)  # sample variance (ddof=1)

def return_variance_period(returns: np.ndarray, start_index: int, window: int) -> float:
    """Return variance of a window of returns starting at `start_index` with length `window`."""
    sub_vector = returns[start_index : start_index + window]
    return return_variance(sub_vector)

def rolling_variance(returns: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling variance of returns over a moving window.
    Returns an array of variances with length len(returns) - window + 1.
    """
    variances = np.empty(len(returns) - window + 1)
    for i in range(len(variances)):
        variances[i] = return_variance_period(returns, i, window)
    return variances

def risk_parity_weights(variances_matrix: np.ndarray) -> np.ndarray:
    """
    Compute risk parity weights given a matrix of variances.
    Each row = time step, each column = asset type.
    Returns a matrix of same shape with weights.
    """
    inv_var = 1 / variances_matrix  # inverse variance
    weights = inv_var / np.sum(inv_var, axis=1, keepdims=True)
    return weights

def apply_weights_to_next_month_returns(weights: np.ndarray, returns: np.ndarray, window: int) -> np.ndarray:
    # Make sure shapes match
    M = min(weights.shape[0], len(returns) - window)
    next_month_returns = returns[window : window + M, :]
    strat_returns = np.einsum('ij,ij->i', next_month_returns, weights[:M, :])
    return strat_returns

def risk_parity(csv, window,has_MOM,has_SMB,has_RM_RF,use_covariance = False):

    date = csv['Date'].to_numpy()
    RF = csv['RF'].to_numpy()

    variance_list = []   # ← start with an empty list
    names_included = []  # optional, to track which ones you added
    return_list = []

    if has_MOM:
        MOM = csv['MOM'].to_numpy()
        sigma_MOM = rolling_variance(MOM, window)
        variance_list.append(sigma_MOM)
        return_list.append(MOM)
        names_included.append("MOM")

    if has_RM_RF:
        RM_RF = csv['RM_RF'].to_numpy()
        sigma_RM_RF = rolling_variance(RM_RF, window)
        variance_list.append(sigma_RM_RF)
        return_list.append(RM_RF)
        names_included.append("RM_RF")

    if has_SMB:
        SMB = csv['SMB'].to_numpy()
        sigma_SMB = rolling_variance(SMB, window)
        variance_list.append(sigma_SMB)
        return_list.append(SMB)
        names_included.append("SMB")

    # ... add others (RF, HML, etc.) similarly if relevant

    # Finally, make the matrix
    if variance_list:
        sigma_matrix = np.column_stack(variance_list)

    
    if return_list:
        r_matrix = np.column_stack(return_list)
    
    weights = risk_parity_weights(sigma_matrix)
    return apply_weights_to_next_month_returns(weights, r_matrix, window)

print(risk_parity(df_EU,36,True,True,True))



