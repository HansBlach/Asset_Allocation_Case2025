# Create CPPI
# Rule: target risky exposure = min(m * cushion, b * wealth)

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from RF_EUR import zcb_price_generator
from risk_parity import risk_parity
from markowitz import markowitz_historical


@dataclass
class CPPIParams:
    m: float = 1             # CPPI multiplier (>=1)
    b: float = 1.0           # max leverage ratio (keep at 1.0 for "no leverage")
    W0: float = 100.0        # initial wealth
    F0: float = 80.0         # initial floor in currency (e.g., 80 for 80/20 split)
    F_target: float = 1.25   # target funded ratio
    F_trigger: float = 1.30  # trigger funded ratio for tie-in


def CPPI(
    active_returns: np.ndarray,       # has dimension (T) (monthly discrete returns of Active)
    zcb_prices: np.ndarray,           # starts at 0 so has dimension (T+1) (rolling ZCB prices, including month 0 and maturity)
    contributions: np.ndarray = None, # has dimension (T) (monthly contributions, if any) - VÆR OPMÆRKSOM PÅ INITIAL WEALTH ER 100 SÅ C0 ER 0.
    cppip: CPPIParams | None = None,   # if we want to change parameters in TieInFonfig we can: simulate_tie_in_path(active, zcb, TieInConfig(L_target=1.20))
) -> pd.DataFrame:
    if cppip is None:
        cppip = CPPIParams()

    T = len(active_returns)                             # number of months to simulate
    assert len(zcb_prices) == T + 1, "zcb_prices must have length T+1 (including month 0 and maturity)."
    
    if contributions is None:
        contributions = np.zeros(T + 1, dtype=float)
    contributions = np.asarray(contributions, dtype=float)
    assert len(contributions) == T + 1, "contributions must have length T+1 (including month 0)."
    

    # ---- Initial parameters and calculations ----
    m = cppip.m
    b = cppip.b
    W0 = cppip.W0
    F0 = cppip.F0

    # Starting values for price processes
    R = zcb_prices[0]                                  
    A = 1     
    cushion = max(W0 - F0, 0.0)
    E0 = min(m * cushion, b * W0)    # Exposure
    
    # Initial allocation proportions
    eta_A = E0 / A                               # units of active
    eta_R = (W0 - E0)/R             # units of ZCB

    # Initial guarantee units N0 from floor and current ZCB price
    N = F0/ R
    F = N * R                          

    # ---- Store results ----

    rows = []
    rows.append({
        "month": 0,
        "A": A,
        "MV_A": eta_A*A,
        "MV_R": eta_R*R,
        "W": eta_A*A + eta_R*R,
        "Floor": F,
        "Cushion": cushion,
        "N": N,
        "P": R,
        "C": contributions[0],
        "c_A": 0, 
        "c_R": 0,
        "L": (eta_A * A + eta_R * R) / F,           # Her genbruger vi konceptet med funded ratio
        "tie_in": False,                            # Vi kalder den tie-in hvis vi ændrer floor
    })

    # ---- Iterate months 1 to T ----
    for t in range(1, T + 1):
        # New market values
        A = (1+ active_returns[t-1]) * A
        R = zcb_prices[t]
        # Evolve holdings to new market values before contributions
        W_pre = eta_A * A + eta_R * R
        F_pre = N * R
        cushion = max(W_pre - F_pre, 0)

        # Exposure
        E = min(m * cushion, b * W_pre)

        eta_A = E / A
        eta_R = (W_pre - E) / R

        # funded ratio before contributions for tie-in logic
        L_pre = (W_pre / F_pre)
        tie_in = (L_pre > cppip.F_trigger)
        if tie_in:
            # enforce target funded ratio by lifting the floor (i.e., moving more into reserve)
            # W_pre / (F_target) = reserve after tie-in
            MV_R_target = W_pre / cppip.F_target
            eta_R = MV_R_target / R
            eta_A = (W_pre - MV_R_target) / A
        
        # HER SKAL VI ALTSÅ LIGE OVERVEJE MICHAELS MAIL:
        # FLOOR ÆNDRER SIG EFTER AKTIVER, MEN VI INDBETALER EFTER TARGTET????
        C = contributions[t]        
        c_R = C / cppip.F_target
        c_A = C - c_R

        # 5) Update active and reserve with contributions
        eta_A += c_A / A
        eta_R += c_R / R
        
        # Keep N == eta_R so floor follows reserve and doesn’t “run away”
        N = eta_R

        # 6) Post-update diagnostics
        MV_A = eta_A * A
        MV_R = eta_R * R
        W = MV_A + MV_R
        F = N * R
        L = W / F

        # Save
        rows.append({
            "month": t,
            "A": A,
            "MV_A": MV_A,
            "MV_R": MV_R,
            "W": W,
            "Floor": F,
            "Cushion": max(W - F, 0.0),
            "N": N,
            "P": R,
            "C": C,
            "c_A": c_A,
            "c_R": c_R,
            "L": L,
            "tie_in": tie_in,
        })

    history = pd.DataFrame(rows, columns=[
        'month', 'A', 'MV_A', 'MV_R', 'c_A', 'c_R', 'N', 'P', 'W', 'Floor', 'L', 'tie_in'
    ])
    return history




# GENBRUGT FRA TIE-IN.PY
EU_data = pd.read_csv("csv_files/long_EXPORT EU EUR.csv")
US_data = pd.read_csv("csv_files/long_EXPORT US EUR.csv")

# Modify data for use in markowitz

portfolios = list(EU_data.columns[5:36])
EU = EU_data[portfolios]
EU.columns = ["EU_" + p for p in portfolios]
US = US_data[portfolios]
US.columns = ["US_" + p for p in portfolios]

# Make a combined dataframe with all assets.
all_assets = pd.concat([EU,US],axis = 1)

# Data for ZCB
zcb_data = pd.read_csv('csv_files/yield_curve_data.csv', parse_dates=['TIME_PERIOD'])


# --- Set parameters ---
T = 120
years = 10
contributions = np.zeros(T+1) + 100.0

allow_short = False
portfolio_strategy = "markowitz"  # "markowitz" or "risk_parity"
market = "EU"
window = 36

if allow_short:
    EU_data = pd.read_csv("csv_files/EXPORT EU EUR.csv")
    US_data = pd.read_csv("csv_files/EXPORT US EUR.csv")

## Markowitz
n_points = 10
strategy = "tangent"
mu_target = 0.01

## Risk parity
if market == "both": include_market2 = True
if market == "EU": include_market2 = False

has_MOM1, has_SMB1, has_RM_RF1 = True, True, True
has_MOM2, has_SMB2, has_RM_RF2 = include_market2, include_market2, include_market2
use_covariance = False
target_std = 4


# --- Active portfolio strategies --- 
# Markowitz
if portfolio_strategy == "markowitz":
    if market == "both":
        active_returns_full = np.array(markowitz_historical(all_assets, window, strategy, mu_target, n_points, allow_short))/100
    if market == "EU":
        active_returns_full = np.array(markowitz_historical(EU, window, strategy, mu_target, n_points, allow_short))/100


# Risk Parity
if portfolio_strategy == "risk_parity":
    active_returns_full = risk_parity(EU_data, US_data, "EU", "US", include_market2, window,
                                        has_MOM1, has_SMB1, has_RM_RF1,
                                        has_MOM2, has_SMB2, has_RM_RF2,
                                        use_covariance, allow_short, target_std)
    active_returns_full = np.array(active_returns_full['return'])/100



# --- Simulate multiple paths and collect results ---


MVA_120, MVR_120, W_120, Return_120, num_tie_in, num_guarantee = [], [], [], [], [], []

for i in range(len(active_returns_full)-120):
    
    zcb_prices = zcb_price_generator(years, T + 1, start = i, data = zcb_data)
    active_returns = active_returns_full[i:i+T]
    # active_returns = np.zeros(T) + 0.10  # Comment out or remove this line to use actual strategy returns

    # Sanity checks
    assert active_returns.shape == (T,)
    assert zcb_prices.shape == (T + 1,)

    summary = CPPI(active_returns, zcb_prices, contributions)

    summary_120 = summary.iloc[120,:]

    # Add observations to results:
    MVA_120.append(summary_120['MV_A'])
    MVR_120.append(summary_120['MV_R'])
    W_120.append(summary_120['W'])
    Return_120.append((summary_120['W']- sum(contributions))/sum(contributions))
    num_guarantee.append(summary_120['W'] - sum(contributions) < 0)
    num_tie_in.append(sum(summary['tie_in']))

# Prints final results

print("Final Results after 10 years:")
print("Average MV Active: ", np.mean(MVA_120))
print("Average MV Reserve: ", np.mean(MVR_120))
print("Average Total Wealth: ", np.mean(W_120))
print("Average Return: ", np.mean(Return_120))
print("Average number of tie-ins: ", np.mean(num_tie_in))
print("Number of paths with guarantee shortfall: ", sum(num_guarantee)/len(num_guarantee))
print(sum(contributions))