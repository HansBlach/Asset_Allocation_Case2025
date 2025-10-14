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

def data_generator(csv,market,window, has_MOM, has_SMB, has_RM_RF):
    variance_list = []   # ← start with an empty list
    names_included = []  # track added factors
    return_list = []

    if has_MOM:
        MOM = csv['MOM'].to_numpy()
        sigma_MOM = rolling_variance(MOM, window)
        variance_list.append(sigma_MOM)
        return_list.append(MOM)
        names_included.append("MOM_" + market)

    if has_RM_RF:
        RM_RF = csv['RM_RF'].to_numpy()
        sigma_RM_RF = rolling_variance(RM_RF, window)
        variance_list.append(sigma_RM_RF)
        return_list.append(RM_RF)
        names_included.append("RM_RF_" + market)

    if has_SMB:
        SMB = csv['SMB'].to_numpy()
        sigma_SMB = rolling_variance(SMB, window)
        variance_list.append(sigma_SMB)
        return_list.append(SMB)
        names_included.append("SMB_" + market)
    
    return return_list, variance_list, names_included


def risk_parity(csv1,csv2, market1, market2, include_market2, window,has_MOM1,has_SMB1,has_RM_RF1, has_MOM2, has_SMB2, has_RM_RF2):
    date = csv1['Date'].to_numpy()
    date2 = csv1['Date'].to_numpy()

    names_included = []  # track added factors
    return_list = []
    r_list1, v_list1, n_list1 = data_generator(csv1, market1, window, has_MOM1, has_SMB1,has_RM_RF1)
    r_list2, v_list2, n_list2 = data_generator(csv2, market2, window, has_MOM2, has_SMB2,has_RM_RF2)
    # Combine lists from both markets
    if(include_market2):
        sigma_matrix = np.column_stack(v_list1 + v_list2)
        r_matrix = np.column_stack(r_list1 + r_list2)
        names_included = n_list1 + n_list2
    else:
        sigma_matrix = np.column_stack(v_list1)
        r_matrix = np.column_stack(r_list1)
        names_included = n_list1
    
    weights = risk_parity_weights(sigma_matrix)
    # we can always print the weight and variance aswell
    #region data_transport
    asset_names = names_included
    aligned_dates = date[window - 1:].astype(str)
    view_weight = pd.DataFrame(weights, columns=asset_names)
    view_weight.insert(0, "Date", aligned_dates)

    view_Var = pd.DataFrame(sigma_matrix, columns = asset_names)
    view_Var.insert(0, "Date", aligned_dates)
    view_return = pd.DataFrame(apply_weights_to_next_month_returns(weights, r_matrix, window), columns = ["return"])
    aligned_dates_return = date[window :].astype(str)
    view_return.insert(0,"Date", aligned_dates_return)

    #endregion
    #this will return a dataframe with dates and returns of the risk parity portfolio
    return view_weight,view_return

EU_risk_parity = risk_parity(df_EU,df_USEUR,"EU","US",True,36,True,True,True,True,True,True)
# print("weights")
#print(weights, len(weights))
# print("variance")
# print( EU_sigma_matrix, len(EU_sigma_matrix))
#The length of the parity should be one shorter than the weights and variance, since it is the previous months weights and variance that is used for this months return
# print("EU_risk_Parity_return")
print(EU_risk_parity, len(EU_risk_parity))
# """
# Small test to ckeck that the first and last return is as expected
# """
# var_a = np.array([2.50966659e+00,  4.95679603e+00, -3.16383356e+00])
# var_b = np.array([0.40768592, 0.06749699, 0.52481709])
# print("first return",sum(var_a*var_b), "actual return", EU_risk_parity["return"][0], "difference", sum(var_a*var_b) - EU_risk_parity["return"][0])
# var_e = np.array([-5.14611285e-01, -5.12105144e+00,  1.12418060e+00])
# var_f = np.array([0.23781546, 0.0105637,  0.75162085])
# print("last return",sum(var_e*var_f), "actual return", EU_risk_parity["return"].iloc[-1], "difference", sum(var_e*var_f) - EU_risk_parity["return"].iloc[-1])



