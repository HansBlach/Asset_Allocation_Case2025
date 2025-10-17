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
    
    m = cppip.m
    b = cppip.b
    W0 = cppip.W0
    F0 = cppip.F0
    
    # ---- t = 0 (post-initial allocation)
    # Initial guarantee units N0 from floor and current ZCB price
    N = F0 / zcb_prices[0]
    F = N * zcb_prices[0]
    W = W0
    cousin = max(W - F, 0.0)

    # CPPI exposure (no leverage)
    E = min(m * cousin, b * W)                                 # Exposure
    eta_A = E / active_returns[0]                         # units of active
    eta_R = (W - E) / zcb_prices[0]                       # units of ZCB
    MV_R = eta_R * W                                      # start exactly at target funded ratio
    MV_A = eta_A * W                                      # Rest in active


    rows = []
    rows.append({
        "month": 0,
        "MV_A": MV_A,
        "MV_R": MV_R,
        "W": MV_A + MV_R,
        "Floor": F0,
        "Cousin": cousin,
        "N": N,
        "P": zcb_prices[0],
        "C": contributions[0],
        "c_A": contributions[0]*MV_A/W0, 
        "c_R": contributions[0]*MV_R/W0,
    })

    # ---- Iterate months 1 to T
    for t in range(1, T + 1):
        # Evolve holdings to new market values
        W = eta_A * MV_A[t] + eta_R * MV_R[t]
        F = N * zcb_prices[t]

        # Recompute CPPI exposure for next period based on NEW W,F at time t
        C = max(W - F, 0.0)
        E = min(m * C, b * W)
        eta_A = 0.0 if active_returns[t] == 0 else E / active_returns[t]
        eta_R = 0.0 if zcb_prices[t] == 0 else (W - E) / zcb_prices[t]

        # Save
        rows.append({
        "month": 0,
        "MV_A": MV_A,
        "MV_R": MV_R,
        "W": MV_A + MV_R,
        "Floor": b * MV_R,
        "Cousin": cousin,
        "N": N,
        "P": zcb_prices[0],
        "C": contributions[t],
        "c_A": contributions[t]*MV_A/(MV_A + MV_R), 
        "c_R": contributions[t]*MV_R/(MV_A + MV_R),
        "L": (MV_A + MV_R) / MV_R,
        "tie_in": False,
    })

    history = pd.DataFrame(rows, columns=['month','MV_A','MV_R', "c_A", "c_R",'N','P','W','L','tie_in'])
    return history

