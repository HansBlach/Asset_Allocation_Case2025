import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load the data from CSV files
df_EU = pd.read_csv("csv_files/EXPORT EU EUR.csv")
df_USEUR = pd.read_csv("csv_files/EXPORT US EUR.csv")
df_US = pd.read_csv("csv_files/EXPORT US USD.csv")

# -----------------------------
# Rolling standard deviation
# -----------------------------
def rolling_std(returns: np.ndarray, window: int) -> np.ndarray:
    stds = np.empty(len(returns) - window + 1)
    for i in range(len(stds)):
        stds[i] = np.std(returns[i : i + window], ddof=1)
    return stds

# -----------------------------
# Portfolio volatility helper
# -----------------------------
def portfolio_std(returns: np.ndarray, weights: np.ndarray, window: int) -> np.ndarray:
    port_std = np.empty(len(returns) - window + 1)
    for i in range(len(port_std)):
        window_returns = returns[i:i+window, :]
        cov = np.cov(window_returns.T)
        w = weights[i, :].reshape(-1, 1)
        port_std[i] = np.sqrt(float(w.T @ cov @ w))
    return port_std

# -----------------------------
# Apply weights to next-period returns
# -----------------------------
def apply_weights_to_next_month_returns(weights: np.ndarray, returns: np.ndarray, window: int) -> np.ndarray:
    M = min(weights.shape[0], len(returns) - window)
    next_month_returns = returns[window : window + M, :]
    strat_returns = np.einsum('ij,ij->i', next_month_returns, weights[:M, :])
    return strat_returns

# -----------------------------
# Data generator for one market
# -----------------------------
def data_generator(csv, market, window, has_MOM, has_SMB, has_RM_RF):
    std_list, return_list, names = [], [], []

    if has_RM_RF:
        RM_RF = csv['RM_RF'].to_numpy()
        std_list.append(rolling_std(RM_RF, window))
        return_list.append(RM_RF)
        names.append("RM_RF_" + market)

    if has_MOM:
        MOM = csv['MOM'].to_numpy()
        std_list.append(rolling_std(MOM, window))
        return_list.append(MOM)
        names.append("MOM_" + market)

    if has_SMB:
        SMB = csv['SMB'].to_numpy()
        std_list.append(rolling_std(SMB, window))
        return_list.append(SMB)
        names.append("SMB_" + market)

    return return_list, std_list, names

# -----------------------------
# Main Risk Parity Function (two markets)
# -----------------------------
def risk_parity(csv1, csv2, market1, market2, include_market2, window,
                has_MOM1, has_SMB1, has_RM_RF1,
                has_MOM2, has_SMB2, has_RM_RF2,
                use_covariance=False, allow_short=False, target_std=None):
    """
    Two-market Risk Parity with non-negative MOM/SMB, RM caps at 1, shared RF residual.
    """

    pd.set_option('display.precision', 6)
    date = csv1['Date'].to_numpy().astype(str)

    # --- gather factors ---
    r_list1, v_list1, n_list1 = data_generator(csv1, market1, window, has_MOM1, has_SMB1, has_RM_RF1)
    r_list2, v_list2, n_list2 = data_generator(csv2, market2, window, has_MOM2, has_SMB2, has_RM_RF2)

    if include_market2:
        std_matrix = np.column_stack(v_list1 + v_list2)
        r_matrix = np.column_stack(r_list1 + r_list2)
        names = n_list1 + n_list2
    else:
        std_matrix = np.column_stack(v_list1)
        r_matrix = np.column_stack(r_list1)
        names = n_list1

    # --- Base equal-risk (1/std)
    inv_std = 1 / std_matrix
    risky_weights = inv_std / np.sum(inv_std, axis=1, keepdims=True)
    risky_weights = np.clip(risky_weights, 0, None)   # ensure no negatives

    # --- Identify market columns for cap enforcement
    market_cols = [i for i, n in enumerate(names) if "RM_RF" in n]

    # --- Scaling
    if not allow_short:
        # Long-only: normalize to sum=1 across risky assets (fully invested)
        risky_weights = risky_weights / np.sum(risky_weights, axis=1, keepdims=True)
        # Enforce market cap ≤1
        for i in market_cols:
            risky_weights[:, i] = np.minimum(risky_weights[:, i], 1.0)
        w_RF = np.zeros(len(risky_weights))  # RF not used (fully invested)
    else:
        if target_std is None:
            raise ValueError("allow_short=True requires target_std to be set.")

        base_std = portfolio_std(r_matrix, risky_weights, window)
        alpha_target = target_std / base_std

        # Market-cap limit scaling per window
        alpha_cap = np.min(1.0 / risky_weights[:, market_cols], axis=1)
        alpha = np.minimum(alpha_target, alpha_cap)

        risky_weights = risky_weights * alpha[:, None]

        # RF residual = 1 - sum of market weights (shared across markets)
        w_RF = 1 - np.sum(risky_weights[:, market_cols], axis=1)
        w_RF = np.clip(w_RF, 0, None)

    # --- Build final outputs
    aligned_dates = date[window - 1:]
    aligned_dates_return = date[window:]

    # Combine RF + risky weights
    view_weight = pd.DataFrame({"Date": aligned_dates, "RF": w_RF})
    for i, name in enumerate(names):
        view_weight[name] = risky_weights[:, i]
    view_weight["sum_risky"] = risky_weights.sum(axis=1)
    view_weight["sum_market"] = risky_weights[:, market_cols].sum(axis=1)
    view_weight["market_capped"] = (view_weight["sum_market"] >= 0.999)

    # Variance table: only risky factors
    view_Var = pd.DataFrame(std_matrix, columns=names)
    view_Var.insert(0, "Date", aligned_dates)

    # Portfolio returns (RF adds no excess return)
    strat_returns = apply_weights_to_next_month_returns(risky_weights, r_matrix, window)
    view_return = pd.DataFrame({"Date": aligned_dates_return, "return": strat_returns})

    return view_weight, view_Var, view_return



# Example usage:
weights, variances, returns = risk_parity(df_EU, df_USEUR, "EU", "US", True, 12,
                                          True, True, True,
                                          True, True, True,
                                          use_covariance=False, allow_short=False, target_std=2)
print(weights.head())
print(variances.head())
print(returns.head())