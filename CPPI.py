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
    m: float = 3             # CPPI multiplier (>=1)
    b: float = 1.0           # max leverage ratio (keep at 1.0 for "no leverage")
    W0: float = 100        # initial wealth
    L_target: float = 1.5   # target funded ratio
    L_trigger: float = 2  # trigger funded ratio for tie-in
    # F0: float = 80 (W0 / L_target)         # initial floor in currency (e.g., 80 for 80/20 split)


def CPPI(
    active_returns: np.ndarray,       # has dimension (T) (monthly discrete returns of Active)
    zcb_prices: np.ndarray,           # starts at 0 so has dimension (T+1) (rolling ZCB prices, including month 0 and maturity)
    # contributions: np.ndarray = None, # has dimension (T) (monthly contributions, if any) - VÆR OPMÆRKSOM PÅ INITIAL WEALTH ER 100 SÅ C0 ER 0.
    cppip: CPPIParams | None = None,   # if we want to change parameters in TieInFonfig we can: simulate_tie_in_path(active, zcb, TieInConfig(L_target=1.20))
) -> pd.DataFrame:
    if cppip is None:
        cppip = CPPIParams()

    T = len(active_returns)                             # number of months to simulate
    assert len(zcb_prices) == T + 1, "zcb_prices must have length T+1 (including month 0 and maturity)."
    
    '''if contributions is None:
        contributions = np.zeros(T + 1, dtype=float)
    contributions = np.asarray(contributions, dtype=float)
    assert len(contributions) == T + 1, "contributions must have length T+1 (including month 0)."
    '''

    # ---- Initial parameters and calculations ----
    m = cppip.m
    b = cppip.b
    W0 = cppip.W0
    F0 = W0 / cppip.L_target          # initial floor in currency


    # Starting values for price processes
    R = zcb_prices[0]                                  
    A = 1     
    cushion = max(W0 - F0, 0.0)
    E0 = min(m * cushion, b * W0)    # Exposure
    
    # Initial allocation proportions
    eta_A = E0 / A                               # units of active
    eta_R = (W0 - E0)/R                          # units of ZCB
    F = F0

    # Initial guarantee units N0 from floor and current ZCB price
    # N = F0 * R
    # F = F0

    # Floor_ratio = W0/F0                          

    # ---- Store results ----  HAR FJERNET CONTRIBUTIONS

    rows = []
    rows.append({
        "month": 0,
        "A": A,
        "MV_A": eta_A*A,
        "MV_R": eta_R*R,
        "W": eta_A*A + eta_R*R,
        "Floor": F0,
        "Cushion": cushion,
        "Guarantee": F0 / R,
        "P": R,
        "E": E0,
        "L": (eta_A * A + eta_R * R) / F0,           # Her genbruger vi konceptet med funded ratio
        "tie_in": False,                             # Vi kalder den tie-in hvis vi ændrer floor
    })

    # ---- Iterate months 1 to T ----
    for t in range(1, T + 1):
        # New market values
        A = (1+ active_returns[t-1]) * A
        R = zcb_prices[t]
        # F = F * R
        # Evolve holdings to new market values before contributions
        # W_pre = eta_A * A + eta_R * R
        # L_pre = N * R

        # cushion_pre = max(W_pre - L_pre, 0)

        # Exposure
        # E_pre = min(m * cushion_pre, b * W_pre)

        # eta_A_pre = E_pre / A
        # eta_R_pre = (W_pre - E_pre) / R

        # funded ratio before contributions for tie-in logic
        # L_pre = (W_pre / L_pre)

        # HER SKAL VI ALTSÅ LIGE OVERVEJE MICHAELS MAIL:
        # FLOOR ÆNDRER SIG EFTER AKTIVER, MEN VI INDBETALER EFTER TARGTET????
        # C = contributions[t]        
        # c_R = C / cppip.L_target
        # c_A = C - c_R

        # 5) Update active and reserve with contributions
        W = eta_A * A + eta_R * R
        F = F * zcb_prices[t] / zcb_prices[t-1]
        cushion = max(W - F, 0)
        E = max(0.0, min(m * cushion, b * W))
        eta_A = E / A
        eta_R = (W - E) / R

        # Keep N == eta_R so floor follows reserve and doesn’t “run away”
        #N = eta_R

        # 6) Post-update diagnostics
        MV_A = eta_A * A
        MV_R = eta_R * R
        W = MV_A + MV_R

        L = W / F         # Floor ratio


        tie_in = (L > cppip.L_trigger)

        if tie_in:
            # enforce target funded ratio by lifting the floor (i.e., moving more into reserve)
            # W_pre / (L_target) = reserve after tie-in
            # MV_R_target = W / cppip.L_target
            # eta_R = MV_R_target / R
            # eta_A = (W - MV_R_target) / A
            # N = eta_R
            F = W / cppip.L_target
            L = cppip.L_target
            cushion = max(W - F, 0)
            E = max(0.0, min(m * cushion, b * W))
            eta_A = E / A
            eta_R = (W - E) / R
            MV_A = eta_A * A
            MV_R = eta_R * R


        # Save ------ HAR SLETTET CONTRIBUTIONS OG L_PRE FRA OUTPUT
        rows.append({
            "month": t,
            "A": A,
            "MV_A": MV_A,
            "MV_R": MV_R,
            "W": W,
            "Floor": F,
            "Cushion": max(W - F, 0.0),
            "Guarantee":F / R,
            "P": R,
            "E": E,
            "L": L,
            "tie_in": tie_in,
        })

    history = pd.DataFrame(rows, columns=[
        'month', 'A', 'MV_A', 'MV_R', 'Guarantee', 'P', 'W', 'Floor', 'E', 'L', 'tie_in'
    ])
    return history




# GENBRUGT FRA TIE-IN.PY
EU_data = pd.read_csv("csv_files/long_EXPORT EU EUR.csv")
US_data = pd.read_csv("csv_files/long_EXPORT US EUR.csv")

columns_to_add = list(EU_data.columns[2:36])

# Add the risk free rate to the portfolios such that they are no longer excess returns

EU_data[columns_to_add] = EU_data[columns_to_add].add(EU_data['RF'], axis=0)
US_data[columns_to_add] = US_data[columns_to_add].add(US_data['RF'], axis=0)

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

allow_short = False
portfolio_strategy = "markowitz"  # "markowitz" or "risk_parity"
market = "both"
window = 36

if allow_short:
    EU_data = pd.read_csv("csv_files/EXPORT EU EUR.csv")
    US_data = pd.read_csv("csv_files/EXPORT US EUR.csv")

## Markowitz
n_points = 50
strategy = "tangent"
mu_target = 0.01

eu_factors = ["SMB", "MOM"]
us_factors = ["RM_RF", "SMB", "MOM"]

## Risk parity
if market == "both": include_market2 = True
if market == "EU": include_market2 = False

has_MOM1, has_SMB1, has_RM_RF1 = True, True, False
has_MOM2, has_SMB2, has_RM_RF2 = True, True, True
has_MOM2, has_SMB2, has_RM_RF2 = include_market2, include_market2, include_market2
use_covariance = False
target_std = 4


# --- Active portfolio strategies --- 
# Markowitz
if portfolio_strategy == "markowitz":
    active_returns_full = np.array(markowitz_historical(EU_data, US_data, eu_factors, us_factors, window, strategy = "tangent"))/100


# Risk Parity
if portfolio_strategy == "risk_parity":
    active_returns_full = risk_parity(EU_data, US_data, "EU", "US", include_market2, window,
                                        has_MOM1, has_SMB1, has_RM_RF1,
                                        has_MOM2, has_SMB2, has_RM_RF2,
                                        use_covariance, allow_short, target_std)
    active_returns_full = np.array(active_returns_full['return'])/100


def path_simulator(active_returns_full, zcb_data, years, T, cppip=CPPIParams(), summary_path = 0):
    # --- Simulate multiple paths and collect results ---
    MVA_120, MVR_120, W_120, Return_120, num_tie_in, guarantee = [], [], [], [], [], []

    # forskellige perioder i linjerne under (første og sidste):
    # for i in range(0, 120):
    for i in range(len(active_returns_full)-156):
        
        zcb_prices = zcb_price_generator(years, T + 1, start = i, data = zcb_data)
        active_returns = active_returns_full[i:T+i]

        # Sanity checks
        assert active_returns.shape == (T ,)
        assert zcb_prices.shape == (T + 1,)

        summary = CPPI(active_returns, zcb_prices, cppip)

        summary_120 = summary.iloc[T,:]

        # Add observations to results:
        MVA_120.append(summary_120['MV_A'])
        MVR_120.append(summary_120['MV_R'])
        W_120.append(summary_120['W'])
        Return_120.append((summary_120['W'])/100 - 1.0)
        guarantee.append(summary_120['Guarantee'])
        num_tie_in.append(sum(summary['tie_in']))
        if i == summary_path:
            active_returns_saved = active_returns
            zcb_prices_saved = zcb_prices

    summary_print = CPPI(active_returns_saved, zcb_prices_saved, cppip)

    return MVA_120, MVR_120, W_120, Return_120, num_tie_in, guarantee, summary_print


MVA_120, MVR_120, W_120, Return_120, num_tie_in, guarantee, summary_print = path_simulator(active_returns_full, zcb_data, years, T, cppip=CPPIParams(), summary_path=0)

print("Final Results after 10 years:")
print("Average MV Active: ", np.mean(MVA_120))
print("Average MV Reserve: ", np.mean(MVR_120))
print("Average Total Wealth: ", np.mean(W_120))
print("Average Return: ", np.mean(Return_120))
print("Average number of tie-ins: ", np.mean(num_tie_in))
#print("Number of paths with guarantee shortfall: ", sum(num_guarantee)/len(num_guarantee))


# active_returns = active_returns = active_returns_full[0:T]

# zcb_prices = zcb_price_generator(years, T + 1, start = 0, data = zcb_data)

# contributions = np.zeros(T+1) + 100



Guarantee = summary_print['Guarantee']
Floor = summary_print['Floor']
Wealth = summary_print['W']

#print(summary_print)

plt.plot(summary_print['month'], Wealth, label='Wealth', color = 'red')
plt.plot(summary_print['month'], Floor, label='Floor', color = 'blue')
plt.plot(summary_print['month'], Guarantee, label='Guarantee', color = 'green')
plt.xlabel('Month')
plt.ylabel('Amount')
plt.title('CPPI Wealth and Guarantee over Time')
plt.legend()
#plt.show()


# ---- Loop for different m with initial L_target and L_trigger ----
L_target = [1.25]
L_trigger = [1.3]
m_values = [1, 2, 3, 4, 5, 6]

rows = []

for m_val in m_values:
    for L_t in L_target:
        for L_tr in L_trigger:
                cppip = CPPIParams(m = m_val, L_target = L_t, L_trigger = L_tr)
                MVA_120, MVR_120, W_120, Return_120, num_tie_in, guarantee, summary_print = path_simulator(active_returns_full, zcb_data, years, T, cppip=cppip, summary_path=0)
                rows.append({
                    "m": m_val,
                    # "L_target": L_t,
                    # "L_trigger": L_tr,
                    "Avg Guarantee": np.mean(guarantee),
                    "Minimum guarantee": np.min(guarantee),
                    "Minimum Wealth": np.min(W_120),
                    "Avg Return": np.mean(Return_120),
                    # "Min Return": np.min(Return_120)
                })

table = pd.DataFrame(rows, columns=[
    'm', 'Avg Guarantee', 'Minimum guarantee', 'Minimum Wealth', 'Avg Return'
])
print(table)



# ---- Another loop with different L_target and L_trigger ----
print("--------------------------------")
L_target = [1.25, 1.35, 1.5]
L_trigger = [[1.3, 1.5, 2], [1.5, 2], [2]]
m_values = [2,3]

rows = []

for m_val in m_values:
    for i, L_t in enumerate(L_target):
        for L_tr in L_trigger[i]:
                cppip = CPPIParams(m = m_val, L_target = L_t, L_trigger = L_tr)
                MVA_120, MVR_120, W_120, Return_120, num_tie_in, guarantee, summary_print = path_simulator(active_returns_full, zcb_data, years, T, cppip=cppip, summary_path=0)
                rows.append({
                    "m": m_val,
                    "L_target": L_t,
                    "L_trigger": L_tr,
                    "Avg Guarantee": np.mean(guarantee),
                    "Minimum guarantee": np.min(guarantee),
                    "Minimum Wealth": np.min(W_120),
                    "Avg Return": np.mean(Return_120),
                    # "Min Return": np.min(Return_120)
                })

table = pd.DataFrame(rows, columns=[
    'm', 'L_target', 'L_trigger', 'Avg Guarantee', 'Minimum guarantee', 'Minimum Wealth', 'Avg Return'
])
print(table)





# # ------- Loop for different starting times -------
# T = 120  # horizon in months
# n_paths = len(active_returns_full) - T + 1
# if n_paths <= 0:
#     raise ValueError("Not enough data to form at least one 120-month path.")

# # Collectors
# MVA_120, MVR_120, W_120, G_120 = [], [], [], []
# RET_120 = []            # total return over the 120-month path (relative to 100 initial wealth)
# NUM_GUARANTEE_BREACH = []  # whether terminal wealth < 100
# NUM_TIE_IN = []            # number of months flagged as tie_in along each path

# for start in range(n_paths):
#     # Slice a 120-month window starting at `start`
#     active_returns = active_returns_full[start : start + T]          # shape (T,)
#     zcb_prices = zcb_price_generator(years, T + 1, start=start, data=zcb_data)  # shape (T+1,)

#     # Sanity checks
#     assert active_returns.shape == (T,)
#     assert zcb_prices.shape == (T + 1,)

#     # Run CPPI over this 120-month window
#     summary = CPPI(active_returns, zcb_prices)

#     # Terminal row is index T because you have T steps and T+1 rows (including month 0)
#     summary_120 = summary.iloc[T, :]

#     # Collect terminal metrics
#     MVA_120.append(summary_120['MV_A'])
#     MVR_120.append(summary_120['MV_R'])
#     W_120.append(summary_120['W'])
#     G_120.append(summary_120['Guarantee'])

#     # Return over the path relative to initial (assumed 100). Your old code had W/W == 1.0.
#     RET_120.append(summary_120['W'] / 100.0 - 1.0)

#     # guarantee breach (terminal wealth below initial 100)
#     NUM_GUARANTEE_BREACH.append(summary_120['W'] < 100)

#     # count of tie-in months in this path
#     NUM_TIE_IN.append(int(summary['tie_in'].sum()))

# # ---- Make histograms of terminal values across starting months ----
# df_terminal = pd.DataFrame({
#     "terminal_wealth": W_120,
#     "terminal_guarantee": G_120,
#     "terminal_return": RET_120,
#     "mv_a": MVA_120,
#     "mv_r": MVR_120,
#     "tie_in_months": NUM_TIE_IN,
#     "breach": NUM_GUARANTEE_BREACH,
# })

# # 1) Terminal Wealth
# plt.figure()
# df_terminal["terminal_wealth"].hist(bins=30)
# plt.xlabel("Terminal Wealth")
# plt.ylabel("Frequency")
# plt.title("Terminal Wealth (120m) across starting months")
# w_mean = df_terminal["terminal_wealth"].mean()
# w_med  = df_terminal["terminal_wealth"].median()
# plt.axvline(w_mean, linestyle="--", linewidth=1, label=f"Mean={w_mean:,.2f}")
# plt.axvline(w_med,  linestyle=":",  linewidth=1, label=f"Median={w_med:,.2f}")
# plt.legend()
# plt.show()

# # 2) Terminal Guarantee
# plt.figure()
# df_terminal["terminal_guarantee"].hist(bins=30)
# plt.xlabel("Terminal Guarantee")
# plt.ylabel("Frequency")
# plt.title("Terminal Guarantee (120m) across starting months")
# g_mean = df_terminal["terminal_guarantee"].mean()
# g_med  = df_terminal["terminal_guarantee"].median()
# plt.axvline(g_mean, linestyle="--", linewidth=1, label=f"Mean={g_mean:,.2f}")
# plt.axvline(g_med,  linestyle=":",  linewidth=1, label=f"Median={g_med:,.2f}")
# plt.legend()
# plt.show()

# # 3) (Optional) Terminal Return or Funded Ratio
# plt.figure()
# df_terminal["terminal_return"].hist(bins=30)
# plt.xlabel("Terminal Return over 120m (relative to 100)")
# plt.ylabel("Frequency")
# plt.title("Terminal Return (120m) across starting months")
# r_mean = df_terminal["terminal_return"].mean()
# r_med  = df_terminal["terminal_return"].median()
# plt.axvline(r_mean, linestyle="--", linewidth=1, label=f"Mean={r_mean:.2%}")
# plt.axvline(r_med,  linestyle=":",  linewidth=1, label=f"Median={r_med:.2%}")
# plt.legend()
# plt.show()
import matplotlib.dates as mdates

def plot_cppi_path(
    active_returns_full: np.ndarray,
    zcb_data: pd.DataFrame,
    years: int,
    T: int,
    cppip: CPPIParams,
    start_index: int,
    save_path: str = "cppi_path_visualization.png",
):
    """
    Plots Wealth, Floor, Guarantee, Active (MV_A), and Reserve (MV_R)
    over time for a single CPPI path.
    X-axis shows calendar years (e.g., 2007, 2008, ...).
    """
    # Generate the corresponding ZCB prices and active returns for this window
    zcb_prices = zcb_price_generator(years, T + 1, start=start_index, data=zcb_data)
    active_returns = active_returns_full[start_index:start_index + T]

    # Run CPPI simulation
    summary = CPPI(active_returns, zcb_prices, cppip)

    # --- Create date index ---
    start_date = pd.Timestamp("2007-08-01") + pd.DateOffset(months=start_index)
    dates = pd.date_range(start=start_date, periods=len(summary), freq="M")

    # Extract series
    wealth = summary["W"].to_numpy()
    floor = summary["Floor"].to_numpy()
    guarantee = summary["Guarantee"].to_numpy()
    mv_a = summary["MV_A"].to_numpy()
    mv_r = summary["MV_R"].to_numpy()
    tie_idx = summary.index[summary["tie_in"]]

    # Plot
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(dates, wealth, color="red", label="Wealth (W)", linewidth=2)
    ax.plot(dates, mv_a, color="orange", label="Active Portfolio (MV_A)", linestyle="--", linewidth=1.8)
    ax.plot(dates, mv_r, color="deepskyblue", label="Reserve Portfolio (MV_R)", linestyle="--", linewidth=1.8)
    ax.plot(dates, floor, color="blue", label="Floor (F)")
    ax.plot(dates, guarantee, color="green", label="Guarantee", linewidth=2)
    ax.scatter(dates[tie_idx], guarantee[tie_idx],
               color="green", edgecolor="black", zorder=5, s=60, label="Tie-in events")

    # --- Format x-axis to show only years ---
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    ax.set_xlabel("Year")
    ax.set_ylabel("Amount")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ CPPI path visualization saved to '{save_path}' (start index = {start_index})")
    return summary

best_i = 88  # December 2014
cppip = CPPIParams(m=2, L_target=1.5, L_trigger=1.65)

plot_cppi_path(
    active_returns_full=active_returns_full,
    zcb_data=zcb_data,
    years=10,
    T=120,
    cppip=cppip,
    start_index=best_i,
)

def compare_cppi_paths(
    active_returns_full: np.ndarray,
    zcb_data: pd.DataFrame,
    years: int,
    T: int,
    cppip: CPPIParams,
    start_index_1: int,
    start_index_2: int,
    save_path: str = "cppi_comparison.png",
):
    """
    Plots two CPPI paths (with different start indices) side-by-side
    using the same y-axis scale for direct comparison.
    """
    # --- Run first path ---
    zcb_prices_1 = zcb_price_generator(years, T + 1, start=start_index_1, data=zcb_data)
    active_returns_1 = active_returns_full[start_index_1:start_index_1 + T]
    summary_1 = CPPI(active_returns_1, zcb_prices_1, cppip)

    # --- Run second path ---
    zcb_prices_2 = zcb_price_generator(years, T + 1, start=start_index_2, data=zcb_data)
    active_returns_2 = active_returns_full[start_index_2:start_index_2 + T]
    summary_2 = CPPI(active_returns_2, zcb_prices_2, cppip)

    # --- Create date axes ---
    base_date = pd.Timestamp("2007-08-01")
    dates_1 = pd.date_range(base_date + pd.DateOffset(months=start_index_1), periods=len(summary_1), freq="M")
    dates_2 = pd.date_range(base_date + pd.DateOffset(months=start_index_2), periods=len(summary_2), freq="M")

    # --- Extract data ---
    W1, F1, G1 = summary_1["W"].to_numpy(), summary_1["Floor"].to_numpy(), summary_1["Guarantee"].to_numpy()
    W2, F2, G2 = summary_2["W"].to_numpy(), summary_2["Floor"].to_numpy(), summary_2["Guarantee"].to_numpy()

    # --- Determine common y-limits ---
    ymin = min(W1.min(), W2.min(), F1.min(), F2.min(), G1.min(), G2.min())
    ymax = max(W1.max(), W2.max(), F1.max(), F2.max(), G1.max(), G2.max())

    # --- Plot side-by-side with same y-limits ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Left panel (Path 1)
    ax = axes[0]
    ax.plot(dates_1, W1, color="red", label="Wealth")
    ax.plot(dates_1, F1, color="blue", label="Floor")
    ax.plot(dates_1, G1, color="green", label="Guarantee", linewidth=2)
    ax.set_title(f"CPPI Path — With start before QE")
    ax.set_xlabel("Year")
    ax.set_ylabel("Amount")
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    # Right panel (Path 2)
    ax = axes[1]
    ax.plot(dates_2, W2, color="red", label="Wealth")
    ax.plot(dates_2, F2, color="blue", label="Floor")
    ax.plot(dates_2, G2, color="green", label="Guarantee", linewidth=2)
    ax.set_title(f"CPPI Path — With start during QE")
    ax.set_xlabel("Year")
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ CPPI comparison saved to '{save_path}'\n"
          f"   (start indices {start_index_1} and {start_index_2}, common y-axis scale)")
    
cppip = CPPIParams(m=2, L_target=1.25, L_trigger=1.3)
compare_cppi_paths(
    active_returns_full=active_returns_full,
    zcb_data=zcb_data,
    years=10,
    T=120,
    cppip=cppip,
    start_index_1=14,   # e.g., Oct 2008
    start_index_2=88,   # e.g., Dec 2014
)