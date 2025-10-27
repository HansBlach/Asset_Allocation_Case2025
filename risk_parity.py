import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load the data from CSV files
df_EU = pd.read_csv("csv_files/EXPORT EU EUR.csv")
df_USEUR = pd.read_csv("csv_files/EXPORT US EUR.csv")
df_US = pd.read_csv("csv_files/EXPORT US USD.csv")

import numpy as np
import pandas as pd


# -----------------------------
# Rolling standard deviation
# -----------------------------
def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    out = np.empty(len(x) - window + 1)
    for i in range(len(out)):
        out[i] = np.std(x[i:i+window], ddof=1)
    return out


# -----------------------------
# Rolling covariance (T x N x N)
# -----------------------------
def rolling_covariance(R: np.ndarray, window: int, shrink = 0.1) -> np.ndarray:
    T = len(R) - window + 1
    N = R.shape[1]
    covs = np.empty((T, N, N))
    for i in range(T):
        covs[i] = (1 - shrink) * np.cov(R[i:i+window].T) + shrink * np.eye(N)
    return covs


# -----------------------------
# Equal-risk weights (variance-only): 1/std normalize
# (non-negative by construction)
# -----------------------------
def inv_vol_weights(std_matrix: np.ndarray) -> np.ndarray:
    invs = 1.0 / std_matrix
    return invs / invs.sum(axis=1, keepdims=True)


# -----------------------------
# ERC via iterative equalization of risk contributions on Σ
# Constraints enforced each iteration:
# - w >= 0 for all
# - w_RM per market <= 1
# - sum(w) == 1 (long-only normalization inside; scaling handled later)
# This is a practical heuristic that converges well for ERC.
# -----------------------------
def erc_weights_from_cov(cov: np.ndarray,
                         market_cols: list[int],
                         max_iter: int = 200,
                         tol: float = 1e-8) -> np.ndarray:
    N = cov.shape[0]
    # Start from inverse-vol long-only
    stds = np.sqrt(np.diag(cov))
    w = 1.0 / stds
    w = np.clip(w, 0, None)
    s = w.sum()
    if s == 0:
        w = np.full(N, 1.0 / N)
    else:
        w /= s

    # Iterate to equalize risk contributions
    for _ in range(max_iter):
        # Risk contributions RC_i = w_i * (Σ w)_i
        Sigma_w = cov @ w
        RC = w * Sigma_w
        RC_mean = RC.mean()
        # Multiplicative update: push each RC towards the mean
        # (avoid div-by-zero)
        adj = np.where(RC > 0, RC_mean / RC, 0.0)
        w = w * adj

        # Enforce constraints each iteration
        w = np.clip(w, 0, None)                   # non-negative for all
        if market_cols:
            w[market_cols] = np.minimum(w[market_cols], 1.0)  # cap each RM at 1
        s = w.sum()
        if s == 0:
            w = np.full(N, 1.0 / N)
        else:
            w /= s  # long-only normalization here

        # Check convergence on risk contributions
        Sigma_w = cov @ w
        RC = w * Sigma_w
        if np.max(np.abs(RC - RC.mean())) < tol:
            break

    return w


# -----------------------------
# Portfolio rolling std given weights (aligned to window end)
# -----------------------------
def portfolio_std(R: np.ndarray, W: np.ndarray, window: int) -> np.ndarray:
    T = len(R) - window + 1
    out = np.empty(T)
    for i in range(T):
        cov = np.cov(R[i:i+window].T)
        w = W[i].reshape(-1, 1)
        out[i] = np.sqrt(float(w.T @ cov @ w))
    return out


# -----------------------------
# Apply weights to next-period returns (alignment +1)
# -----------------------------
def apply_weights_to_next_month_returns(W: np.ndarray, R: np.ndarray, window: int) -> np.ndarray:
    M = min(W.shape[0], len(R) - window)
    nxt = R[window:window+M, :]
    return np.einsum('ij,ij->i', nxt, W[:M, :])


# -----------------------------
# Factor extraction per market (respect includes)
# Returns (R_list, std_list, names) where order is RM_RF, MOM, SMB if present
# -----------------------------
def gather_market(csv, market, window, has_MOM, has_SMB, has_RM_RF):
    R_list, std_list, names = [], [], []

    if has_RM_RF:
        x = csv['RM_RF'].to_numpy()
        R_list.append(x)
        std_list.append(rolling_std(x, window))
        names.append(f"RM_RF_{market}")

    if has_MOM:
        x = csv['MOM'].to_numpy()
        R_list.append(x)
        std_list.append(rolling_std(x, window))
        names.append(f"MOM_{market}")

    if has_SMB:
        x = csv['SMB'].to_numpy()
        R_list.append(x)
        std_list.append(rolling_std(x, window))
        names.append(f"SMB_{market}")

    return R_list, std_list, names


# -----------------------------
# Main: two-market risk parity with optional ERC (covariance)
# -----------------------------
def risk_parity(csv1, csv2, market1, market2, include_market2, window,
                has_MOM1, has_SMB1, has_RM_RF1,
                has_MOM2, has_SMB2, has_RM_RF2,
                use_covariance=False, allow_short=False, target_std=None):

    pd.set_option('display.precision', 6)
    date = csv1['Date'].to_numpy().astype(str)

    # --- Extract RF return (same for both markets)
    rf_return = csv1['RF'].to_numpy()
    rf_std = rolling_std(rf_return, window)  # For view_std only

    # --- Gather risky factors (order preserved per market)
    R1, S1, N1 = gather_market(csv1, market1, window, has_MOM1, has_SMB1, has_RM_RF1)
    R2, S2, N2 = gather_market(csv2, market2, window, has_MOM2, has_SMB2, has_RM_RF2)

    if include_market2:
        R_risky = np.column_stack(R1 + R2)
        std_matrix = np.column_stack(S1 + S2)
        names = N1 + N2
    else:
        R_risky = np.column_stack(R1)
        std_matrix = np.column_stack(S1)
        names = N1

    T = len(R_risky) - window + 1
    N = len(names)

    # Identify market columns for caps (each "RM_RF_*")
    market_cols = [i for i, nm in enumerate(names) if nm.startswith("RM_RF_")]

    # --- Compute base risky weights
    W = np.zeros((T, N))

    if use_covariance:
        Covs = rolling_covariance(R_risky, window)
        for t in range(T):
            W[t] = erc_weights_from_cov(Covs[t], market_cols)
    else:
        W = inv_vol_weights(std_matrix)
        for i in market_cols:
            W[:, i] = np.minimum(W[:, i], 1.0)
        W = W / W.sum(axis=1, keepdims=True)
        

    # --- Mode handling
    if not allow_short:
        # Long-only: fully invested in risky factors
        for i in market_cols:
            W[:, i] = np.minimum(W[:, i], 1.0)
        W_scaled = W.copy()
        w_RF = np.zeros(T)
    else:
        if target_std is None:
            raise ValueError("allow_short=True requires target_std.")

        # Compute base portfolio vol (full covariance)
        base_std = portfolio_std(R_risky, W, window)
        alpha_target = target_std / base_std

        # Cap each RM ≤ 1
        alpha_cap_each = np.full_like(alpha_target, np.inf, dtype=float)
        for i in market_cols:
            wi = W[:, i]
            cap_i = np.where(wi > 0, 1.0 / wi, np.inf)
            alpha_cap_each = np.minimum(alpha_cap_each, cap_i)

        # Global market cap: Σ RM ≤ 1
        market_base = W[:, market_cols].sum(axis=1)
        alpha_cap_total = np.where(market_base > 0, 1.0 / market_base, np.inf)

        # Choose tightest cap
        alpha = np.minimum(alpha_target, np.minimum(alpha_cap_each, alpha_cap_total))

        # Scale risky weights
        W_scaled = W * alpha[:, None]

        # Clamp individual & total markets
        for i in market_cols:
            W_scaled[:, i] = np.minimum(W_scaled[:, i], 1.0)
        sum_market = W_scaled[:, market_cols].sum(axis=1)
        over = sum_market > 1.0 + 1e-12
        if np.any(over):
            idx = np.where(over)[0]
            for t in idx:
                W_scaled[t, market_cols] *= (1.0 / sum_market[t])

        # Enforce non-negativity
        W_scaled = np.clip(W_scaled, 0, None)

        # Residual RF allocation (unused capital)
        w_RF = 1.0 - W_scaled[:, market_cols].sum(axis=1)
        w_RF = np.maximum(w_RF, 0.0)

    # --- Add RF to weights and returns (as first column)
    # Align RF to rolling window end
    aligned_rf = rf_return[window - 1:]
    aligned_rf_std = rf_std

    # Add to weight matrix
    W_full = np.column_stack([w_RF, W_scaled])
    R_full = np.column_stack([rf_return, R_risky])

    all_names = ["RF"] + names

    # --- Outputs
    aligned_dates = date[window - 1:]
    aligned_dates_return = date[window:]

    # Weights (RF + all risky)
    view_weight = pd.DataFrame(W_full, columns=["RF"] + names)
    view_weight.insert(0, "Date", aligned_dates)
    view_weight["sum_risky"] = W_scaled.sum(axis=1)
    view_weight["sum_market"] = W_scaled[:, market_cols].sum(axis=1)
    view_weight["market_capped"] = view_weight["sum_market"] >= 1.0 - 1e-12

    # Variance / std table
    view_std = pd.DataFrame(np.column_stack([aligned_rf_std, std_matrix]),
                            columns=["RF"] + names)
    view_std.insert(0, "Date", aligned_dates)

    # --- Compute portfolio returns (includes RF contribution)
    strat_returns = apply_weights_to_next_month_returns(W_full, R_full, window)
    view_return = pd.DataFrame({"Date": aligned_dates_return, "return": strat_returns})

    return view_return




#You can now chose to short the model by chosing the portfolios with allow_short = True, and set a target standard deviation for the portfolio with target_std.
#The max short is when the market rate is fully invested so it equals 1. if less then one and shorting is allowed the rest is invested in risk free rate.
#The target std is hit by using the std matrix computed from the rolling std function. The target std represents the portfolios desired risk level, and not the markets risk level.
# Example usage:
returns = risk_parity(df_EU, df_USEUR, "EU", "US", True, 36,
                                          True, True, True,
                                          True, True, True,
                                          use_covariance=False, allow_short=True, target_std=3)
# # everything in percentages not decimals
# print(weights.head())
# print(std.head())
#print(returns.head())
# print(np.mean(returns['return']))