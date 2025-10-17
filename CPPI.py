# Create CPPI
# Rule: target risky exposure = min(m * cushion, b * wealth)

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional
import numpy as np
import pandas as pd


@dataclass
class CPPIParams:
    m: float = 3             # CPPI multiplier (>=1)
    b: float = 1.0           # max leverage ratio (keep at 1.0 for "no leverage")
    W0: float = 100.0        # initial wealth
    F0: float = 80.0         # initial floor in currency (e.g., 80 for 80/20 split)


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
    cushion = max(W0 - F0, 0.0)
    E = min(m * cushion, b * W0)    # Exposure
    
    # Initial allocation proportions
    eta_A = E / 1                               # units of active
    eta_R = (W0 - E)/zcb_prices[0]             # units of ZCB

    # Initial guarantee units N0 from floor and current ZCB price
    N = F0 / zcb_prices[0]
    F = N * zcb_prices[0]
    
    # Starting values for price processes
    R = zcb_prices[0]                                  # start exactly at target funded ratio
    A = 1                                # Rest in active

    # initial allocations
    MV_A = eta_A * A
    MV_R = eta_R * R

    # ---- Store results ----

    rows = []
    rows.append({
        "month": 0,
        "MV_A": MV_A,
        "MV_R": MV_R,
        "W": MV_A + MV_R,
        "Floor": F0,
        "Cushion": cushion,
        "N": N,
        "P": zcb_prices[0],
        "C": contributions[0],
        "c_A": contributions[0]*MV_A/W0, 
        "c_R": contributions[0]*MV_R/W0,
    })

    # ---- Iterate months 1 to T ----
    for t in range(1, T + 1):
        # New market values
        A = (1+ active_returns[t]) * A
        R = zcb_prices[t]
        # Evolve holdings to new market values
        W = eta_A * A + eta_R * R
        F = N * zcb_prices[t]

        cushion = max(W - F, 0)
        E = min(m * cushion, b * W)


        eta_A = E / A
        eta_R = (W - E) / R

        # Add the new contribution in the same proportion as the eta's

        weight_A = E/W
        weight_R = (W-E)/W

        c_A = contributions[t]*weight_A
        c_R = contributions[t]*weight_R
        

        MV_A = eta_A * A + c_A
        MV_R = eta_R * R + c_R


        # Save
        rows.append({
        "month": 0,
        "MV_A": MV_A,
        "MV_R": MV_R,
        "W": W,
        "Floor": F,
        "Cushion": cushion,
        "N": N,
        "P": zcb_prices[t],
        "C": contributions[t],
        "c_A": c_A*contributions[t], 
        "c_R": c_R*contributions[t],
        "L": W/F,
        "tie_in": False,
    })

    history = pd.DataFrame(rows, columns=['month','MV_A','MV_R', "c_A", "c_R",'N','P','W','L','tie_in'])
    return history
