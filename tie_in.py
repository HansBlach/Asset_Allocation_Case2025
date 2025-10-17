import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
from RF_EUR import zcb_price_generator
from risk_parity import risk_parity
from markowitz import markowitz_historical


# ---- Load Data ----


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



# ---- Tie-in function ----

@dataclass
# Define parameters for the tie-in strategy
class TieInConfig:
    initial_wealth: float = 0
    L_target: float = 1.2
    L_trigger: float = 1.3


@dataclass
# Zero coupon bond parameters
class PathResult:
    final_N: float
    initial_N: float
    # two different ways to measure accrual
    accrual_abs: float
    accrual_pct: float
    history: pd.DataFrame


# Simulate a single path of the tie-in strategy
# Important to note cfg is not needed
def simulate_tie_in_path(
    active_returns: np.ndarray,       # has dimension (T) (monthly discrete returns of Active)
    zcb_prices: np.ndarray,           # starts at 0 so has dimension (T+1) (rolling ZCB prices, including month 0 and maturity)
    contributions: np.ndarray = None, # has dimension (T) (monthly contributions, if any) - VÆR OPMÆRKSOM PÅ INITIAL WEALTH ER 100 SÅ C0 ER 0.
    cfg: TieInConfig | None = None,   # if we want to change parameters in TieInFonfig we can: simulate_tie_in_path(active, zcb, TieInConfig(L_target=1.20))
) -> pd.DataFrame:
    if cfg is None:
        cfg = TieInConfig()
    
    T = len(active_returns)
    assert len(zcb_prices) == T + 1, "zcb_prices must have length T+1 (including month 0 and maturity)."

    if contributions is None:
        contributions = np.zeros(T + 1, dtype=float)
    contributions = np.asarray(contributions, dtype=float)
    assert len(contributions) == T + 1, "contributions must have length T+1 (including month 0)."

    W0 = cfg.initial_wealth + contributions[0]            # We can both set initial wealt and contributions
    MV_R = W0 / cfg.L_target                              # start exactly at target funded ratio
    MV_A = W0 - MV_R                                      # Rest in active
    N = MV_R / zcb_prices[0]                              # initial face (units of the ZCB)

    rows = []
    rows.append({
        "month": 0,
        "MV_A": MV_A,
        "MV_R": MV_R,
        "N": N,
        "P": zcb_prices[0],
        "W": MV_A + MV_R,
        "C": contributions[0],
        "c_A": contributions[0]*MV_A/W0, 
        "c_R": contributions[0]*MV_R/W0,
        "L": (MV_A + MV_R) / MV_R,
        "tie_in": False,
    })
    
    # Do the calculations month by month
    for i in range(1, T+1):
        # 1) Active evolves                                     - VI SKAL OGSÅ LIGE VÆRE OPMÆRKSOMME PÅ HVORDAN ACTIVE RETURNS ER GIVET I VALGT STRATEGI
        MV_A = MV_A * (1.0 + active_returns[i-1])              #- HVIS ACTIVE RETURNS ER AGGREGEREDE SKAL DET VÆRE SÅDAN, ELLERS SKAL ÆNDRES
        # 2) Reserve repriced from rolling ZCB
        P_i = zcb_prices[i]
        MV_R = N * P_i
        # Total wealth
        W = MV_A + MV_R

        # Calculate funded ratio
        L = (MV_A + MV_R) / MV_R
        tie_in = L > cfg.L_trigger
        
        # Make relevant adjustments if the funded ratio exceeds the trigger level
        if tie_in:
            # Reset to target: W = L_target * N_new * P_i
            N_new = W / (cfg.L_target * P_i)       # new face (units of the ZCB)
            # new reserve and active
            MV_R = N_new * P_i
            MV_A = W - MV_R
        
        # Calculate contributions based on the value of active/reserve portfolio
        C = contributions[i]        
        c_R = C * MV_R / (W)
        c_A = C * MV_A / (W)
        # 4) Update active and reserve with contributions
        MV_R += c_R
        MV_A += c_A
        # 5) Add contribution of ZCB to guarentee
        N += c_R/P_i
            

        rows.append(dict(month=i, MV_A=MV_A, MV_R=MV_R, c_A = c_A, c_R = c_R, N=N, P=P_i, W=MV_A+MV_R, L=(MV_A+MV_R)/MV_R, tie_in=tie_in))

    history = pd.DataFrame(rows, columns=['month','MV_A','MV_R', "c_A", "c_R",'N','P','W','L','tie_in'])
    return history



# --- Set parameters ---
T = 120
years = 10
contributions = np.zeros(T+1) + 100

allow_short = False
portfolio_strategy = "risk_parity"
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

    summary = simulate_tie_in_path(active_returns, zcb_prices, contributions)

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