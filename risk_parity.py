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
def rolling_covariance(R: np.ndarray, window: int) -> np.ndarray:
    T = len(R) - window + 1
    N = R.shape[1]
    covs = np.empty((T, N, N))
    for i in range(T):
        covs[i] = np.cov(R[i:i+window].T)
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

    # Gather factors (order preserved per market: RM_RF, MOM, SMB if included)
    R1, S1, N1 = gather_market(csv1, market1, window, has_MOM1, has_SMB1, has_RM_RF1)
    R2, S2, N2 = gather_market(csv2, market2, window, has_MOM2, has_SMB2, has_RM_RF2)

    if include_market2:
        R = np.column_stack(R1 + R2)   # full returns matrix (T_full x N)
        std_matrix = np.column_stack(S1 + S2)  # (T x N)
        names = N1 + N2
    else:
        R = np.column_stack(R1)
        std_matrix = np.column_stack(S1)
        names = N1

    T = len(R) - window + 1
    N = len(names)

    # Identify market columns for caps (each "RM_RF_*")
    market_cols = [i for i, nm in enumerate(names) if nm.startswith("RM_RF_")]

    # --- Compute base risky weights per window
    W = np.zeros((T, N))

    if use_covariance:
        # ERC per window using rolling Σ, with constraints
        Covs = rolling_covariance(R, window)
        for t in range(T):
            W[t] = erc_weights_from_cov(Covs[t], market_cols)
    else:
        # Inverse-vol parity (variance-only)
        W = inv_vol_weights(std_matrix)
        # Non-negativity already guaranteed; still cap RM at 1 defensively and renormalize (no shorting baseline)
        for i in market_cols:
            W[:, i] = np.minimum(W[:, i], 1.0)
        # Normalize to sum 1 (long-only baseline)
        W = W / W.sum(axis=1, keepdims=True)

    # --- Mode handling
    if not allow_short:
        # Long-only mode: fully invested in risky factors
        # (already normalized to sum 1 from both branches above)
        # Re-enforce RM caps (should already hold)
        for i in market_cols:
            W[:, i] = np.minimum(W[:, i], 1.0)
        # RF is not used; report 0 for convenience
        w_RF = np.zeros(T)
        W_scaled = W.copy()
    else:
       # --- Shorting allowed + target_std scaling (BUT: SMB/MOM >= 0; RM capped) ---
        if target_std is None:
            raise ValueError("allow_short=True requires target_std.")

        # base portfolio std using full covariance Σ_t
        base_std = portfolio_std(R, W, window)            # shape (T,)
        alpha_target = target_std / base_std               # desired scaling to hit target

        # per-market cap (keep each RM_RF_* <= 1)
        alpha_cap_each = np.full_like(alpha_target, np.inf, dtype=float)
        for i in market_cols:
            wi = W[:, i]
            # if wi==0, that asset doesn't constrain scaling
            cap_i = np.where(wi > 0, 1.0 / wi, np.inf)
            alpha_cap_each = np.minimum(alpha_cap_each, cap_i)

        # NEW: global market cap (keep total RM exposure across BOTH markets <= 1)
        market_base = W[:, market_cols].sum(axis=1)               # sum of all RM columns before scaling
        alpha_cap_total = np.where(market_base > 0, 1.0 / market_base, np.inf)

        # choose the tightest cap
        alpha = np.minimum(alpha_target, np.minimum(alpha_cap_each, alpha_cap_total))

        # scale
        W_scaled = W * alpha[:, None]

        # numeric safety: enforce per-asset and total market caps exactly
        # (1) hard per-asset clamp
        for i in market_cols:
            W_scaled[:, i] = np.minimum(W_scaled[:, i], 1.0)

        # (2) hard total clamp (preserve proportions among market legs)
        sum_market = W_scaled[:, market_cols].sum(axis=1)
        over = sum_market > 1.0 + 1e-12
        if np.any(over):
            idx = np.where(over)[0]
            for t in idx:
                W_scaled[t, market_cols] *= (1.0 / sum_market[t])  # scale down both markets proportionally

        # keep SMB/MOM non-negative
        W_scaled = np.clip(W_scaled, 0, None)

        # RF reported as residual against TOTAL market exposure
        w_RF = 1.0 - W_scaled[:, market_cols].sum(axis=1)
        w_RF = np.maximum(w_RF, 0.0)

    # --- Outputs
    aligned_dates = date[window - 1:]
    aligned_dates_return = date[window:]

    # Weights (RF + all risky)
    view_weight = pd.DataFrame({"Date": aligned_dates, "RF": w_RF})
    for j, nm in enumerate(names):
        view_weight[nm] = W_scaled[:, j]
    view_weight["sum_risky"] = W_scaled.sum(axis=1)
    view_weight["sum_market"] = W_scaled[:, market_cols].sum(axis=1)
    view_weight["market_capped"] = (view_weight["sum_market"] >= 1.0 - 1e-12)

    # Variance view = rolling stds of risky factors (no RF)
    view_std = pd.DataFrame(std_matrix, columns=names)
    view_std.insert(0, "Date", aligned_dates)

    # Returns use risky weights only (RF has zero excess return)
    strat_returns = apply_weights_to_next_month_returns(W_scaled, R, window)
    view_return = pd.DataFrame({"Date": aligned_dates_return, "return": strat_returns})

    return view_weight, view_std, view_return



#You can now chose to short the model by chosing the portfolios with allow_short = True, and set a target standard deviation for the portfolio with target_std.
#The max short is when the market rate is fully invested so it equals 1. if less then one and shorting is allowed the rest is invested in risk free rate.
#The target std is hit by using the std matrix computed from the rolling std function. The target std represents the portfolios desired risk level, and not the markets risk level.
# Example usage:
weights, std, returns = risk_parity(df_EU, df_USEUR, "EU", "US", True, 36,
                                          True, True, True,
                                          True, True, True,
                                          use_covariance=False, allow_short=True, target_std=3)
# everything in percentages not decimals
print(weights.head())
print(std.head())
print(returns.head())
print(np.mean(returns['return']))