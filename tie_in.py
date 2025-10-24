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



# # --- Set parameters ---
# T = 120
# years = 10
# contributions = np.zeros(T+1)
# contributions[0] = 100  # Initial contribution at month 0

# allow_short = False
# portfolio_strategy = "european_equity" # "markowitz" or "risk_parity", european_equity
# market = "EU"
# window = 36

# if allow_short:
#     EU_data = pd.read_csv("csv_files/EXPORT EU EUR.csv")
#     US_data = pd.read_csv("csv_files/EXPORT US EUR.csv")


# columns_to_add = list(EU_data.columns[2:36])

# # Add the risk free rate to the portfolios such that they are no longer excess returns

# EU_data[columns_to_add] = EU_data[columns_to_add].add(EU_data['RF'], axis=0)
# US_data[columns_to_add] = US_data[columns_to_add].add(US_data['RF'], axis=0)

# ## Markowitz
# n_points = 10
# strategy = "tangent"
# mu_target = 0.01

# ## Risk parity
# if market == "both": include_market2 = True
# if market == "EU": include_market2 = False

# has_MOM1, has_SMB1, has_RM_RF1 = True, True, True
# has_MOM2, has_SMB2, has_RM_RF2 = include_market2, include_market2, include_market2
# use_covariance = False
# target_std = 4


# # --- Active portfolio strategies --- 

# # Markowitz
# if portfolio_strategy == "markowitz":
#     if market == "both":
#         active_returns_full = np.array(markowitz_historical(all_assets, window, strategy, mu_target, n_points, allow_short))/100
#     if market == "EU":
#         active_returns_full = np.array(markowitz_historical(EU, window, strategy, mu_target, n_points, allow_short))/100


# # Risk Parity
# if portfolio_strategy == "risk_parity":
#     active_returns_full = risk_parity(EU_data, US_data, "EU", "US", include_market2, window,
#                                         has_MOM1, has_SMB1, has_RM_RF1,
#                                         has_MOM2, has_SMB2, has_RM_RF2,
#                                         use_covariance, allow_short, target_std)
#     active_returns_full = np.array(active_returns_full['return'])/100
# if portfolio_strategy =="european_equity":
#     active_returns_full = EU_data['RM_RF']/100
#     active_returns_full = active_returns_full[window:].to_numpy()

# # --- Simulate multiple paths and collect results ---


# MVA_120, MVR_120, W_120, Return_120, num_tie_in, num_guarantee = [], [], [], [], [], []

# for i in range(len(active_returns_full)-120):
    
#     zcb_prices = zcb_price_generator(years, T + 1, start = i, data = zcb_data)
#     active_returns = active_returns_full[i:i+T]
#     # active_returns = np.zeros(T) + 0.10  # Comment out or remove this line to use actual strategy returns

#     # Sanity checks
#     assert active_returns.shape == (T,)
#     assert zcb_prices.shape == (T + 1,)

#     summary = simulate_tie_in_path(active_returns, zcb_prices, contributions)

#     summary_120 = summary.iloc[120,:]

#     # Add observations to results:
#     MVA_120.append(summary_120['MV_A'])
#     MVR_120.append(summary_120['MV_R'])
#     W_120.append(summary_120['W'])
#     Return_120.append((summary_120['W']- sum(contributions))/sum(contributions))
#     num_guarantee.append(summary_120['W'] - sum(contributions) < 0)
#     num_tie_in.append(sum(summary['tie_in']))

# # Prints final results

# print("Final Results after 10 years:")
# print("Average MV Active: ", np.mean(MVA_120))
# print("Average MV Reserve: ", np.mean(MVR_120))
# print("Average Total Wealth: ", np.mean(W_120))
# print("Average Return: ", np.mean(Return_120))
# print("Average number of tie-ins: ", np.mean(num_tie_in))
# print("Number of paths with guarantee shortfall: ", sum(num_guarantee)/len(num_guarantee))
# print(sum(contributions))





# # --- Grid search over (L_trigger, L_target) ---

# # 1) A helper that runs your existing rolling-window loop for one (Ltr, Ltgt)
# def evaluate_params(Ltr: float, Ltgt: float) -> dict:
#     assert Ltgt < Ltr, "Require L_target < L_trigger"
#     # use your existing contributions, T, years, zcb_data, active_returns_full
#     MVA_120, MVR_120, W_120, Ret_120, tieins, shortfall, final_N = [], [], [], [], [], [], []
    
#     for i in range(len(active_returns_full) - T):
#         zcb_prices = zcb_price_generator(years, T + 1, start=i, data=zcb_data)
#         active_returns = active_returns_full[i:i+T]

#         # pass a custom cfg with target/trigger
#         cfg = TieInConfig(L_target=Ltgt, L_trigger=Ltr, initial_wealth=0)  # adjust initial_wealth if you want

#         summary = simulate_tie_in_path(active_returns, zcb_prices, contributions, cfg)
#         s120 = summary.iloc[T]  # month 120

#         MVA_120.append(s120['MV_A'])
#         MVR_120.append(s120['MV_R'])
#         W_120.append(s120['W'])
#         Ret_120.append((s120['W'] - sum(contributions)) / sum(contributions))
#         shortfall.append(s120['W'] - sum(contributions) < 0)
#         tieins.append(int(summary['tie_in'].sum()))
#         final_N.append(float(s120['N']))

#     return dict(
#         L_trigger=Ltr,
#         L_target=Ltgt,
#         avg_MV_A=np.mean(MVA_120),
#         avg_MV_R=np.mean(MVR_120),
#         avg_W=np.mean(W_120),
#         avg_return=np.mean(Ret_120),
#         avg_tieins=np.mean(tieins),
#         p_shortfall=np.mean(shortfall),
#         median_final_N=np.median(final_N),
#         p10_final_N=np.percentile(final_N, 10),
#         p90_final_N=np.percentile(final_N, 90),
#         paths=len(W_120),
#     )

# # 2) Define a grid (feel free to tweak)
# triggers = np.round(np.arange(1.10, 2.05, 0.1), 2)
# deltas   = np.round(np.arange(0.02, 0.23, 0.04), 2)  # delta = L_trigger - L_target

# records = []
# for Ltr in triggers:
#     for d in deltas:
#         Ltgt = round(Ltr - d, 2)
#         if Ltgt <= 1.00 or Ltgt >= Ltr:  # feasibility
#             continue
#         rec = evaluate_params(Ltr, Ltgt)
#         records.append(rec)

# grid_df = pd.DataFrame(records)
# print(grid_df.head())


# # 3) Heatmap of average return across (L_trigger, L_target)
# pivot_ret = grid_df.pivot(index="L_target", columns="L_trigger", values="avg_return").sort_index(ascending=False)
# pivot_ti  = grid_df.pivot(index="L_target", columns="L_trigger", values="avg_tieins").sort_index(ascending=False)
# pivot_ps  = grid_df.pivot(index="L_target", columns="L_trigger", values="p_shortfall").sort_index(ascending=False)

# def show_heatmap(pivot, title, fmt="{:.2%}", good_is_high=True, tick_step=2):
#     """
#     good_is_high=True  -> higher values are better, use RdYlGn
#     good_is_high=False -> lower values are better, use RdYlGn_r
#     tick_step: show every Nth y tick to reduce clutter
#     """
#     cmap = "RdYlGn" if good_is_high else "RdYlGn_r"

#     fig, ax = plt.subplots(figsize=(10, 6))
#     im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)

#     ax.set_title(title)
#     ax.set_xlabel("L_trigger")
#     ax.set_ylabel("L_target")

#     # x ticks (keep them all, but you can thin them too if needed)
#     ax.set_xticks(range(pivot.shape[1]))
#     ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns], rotation=45, ha="right")

#     # y ticks: thin them and bump font size for readability
#     y_idx = np.arange(pivot.shape[0])
#     y_idx = y_idx[::tick_step] if tick_step > 1 else y_idx
#     ax.set_yticks(y_idx)
#     ax.set_yticklabels([f"{pivot.index[i]:.2f}" for i in y_idx], fontsize=10)

#     # annotate cells
#     vals = pivot.values
#     # choose annotation color based on luminance so text stays readable
#     vmin, vmax = np.nanmin(vals), np.nanmax(vals)
#     for i in range(pivot.shape[0]):
#         for j in range(pivot.shape[1]):
#             val = vals[i, j]
#             if np.isfinite(val):
#                 # normalized brightness: 0 (dark) .. 1 (bright)
#                 norm = (val - vmin) / (vmax - vmin + 1e-12)
#                 text_color = "black" if norm > 0.5 else "white"
#                 ax.text(j, i,
#                         (fmt.format(val) if ("return" in title.lower() or "shortfall" in title.lower())
#                          else f"{val:.1f}"),
#                         ha="center", va="center", fontsize=8, color=text_color)

#     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     plt.tight_layout()
#     plt.show()

# show_heatmap(pivot_ret, "Avg terminal return", fmt="{:.2%}", good_is_high=True,  tick_step=2)
# show_heatmap(pivot_ti,  "Avg # tie-ins",      fmt="{:.1f}",  good_is_high=False, tick_step=2)
# show_heatmap(pivot_ps,  "Probability of shortfall", fmt="{:.1%}", good_is_high=False, tick_step=2)


# # 4) Scatter: avg return vs avg tie-ins; color by L_trigger, marker by delta
# grid_df["delta"] = (grid_df["L_trigger"] - grid_df["L_target"]).round(2)

# fig, ax = plt.subplots(figsize=(7,5))
# for d in sorted(grid_df["delta"].unique()):
#     sub = grid_df[grid_df["delta"] == d]
#     sc = ax.scatter(sub["avg_tieins"], sub["avg_return"], label=f"Δ={d:.2f}", alpha=0.8)
# ax.set_xlabel("Average tie-ins per path")
# ax.set_ylabel("Average terminal return")
# ax.set_title("Return vs. tie-in activity")
# ax.legend(title="Spacing (Ltr−Ltgt)")
# plt.tight_layout(); plt.show()




# # ----------------- summarize against current settings -----------------
# baseline = evaluate_params(1.30, 1.25)  # adjust to your current
# print("Baseline:", baseline)

# # For the scatter, add a crosshair
# plt.figure(figsize=(7,5))
# for d in sorted(grid_df["delta"].unique()):
#     sub = grid_df[grid_df["delta"] == d]
#     plt.scatter(sub["avg_tieins"], sub["avg_return"], label=f"Δ={d:.2f}", alpha=0.8)
# plt.axvline(baseline["avg_tieins"], linestyle="--")
# plt.axhline(baseline["avg_return"], linestyle="--")
# plt.text(baseline["avg_tieins"], baseline["avg_return"], "  baseline", va="bottom")
# plt.xlabel("Average tie-ins per path"); plt.ylabel("Average terminal return")
# plt.title("Return vs. tie-in activity (with baseline)")
# plt.legend(title="Spacing (Ltr−Ltgt)")
# plt.tight_layout(); plt.show()

def simulate_strategy(portfolio_strategy: str):
    """Simulate tie-in strategy and return final reserve (MV_R) and total wealth (W)."""
    T = 120
    years = 10
    contributions = np.zeros(T+1)
    contributions[0] = 100
    window = 36
    allow_short = False
    use_covariance = False
    target_std = 4

    # --- Choose active strategy ---
    if portfolio_strategy == "markowitz":
        active_returns_full = np.array(markowitz_historical(EU, window, "tangent", 0.01, 10, allow_short)) / 100

    elif portfolio_strategy == "risk_parity":
        active_returns_full = risk_parity(EU_data, US_data, "EU", "US", False, window,
                                          True, True, True, False, False, False,
                                          use_covariance, allow_short, target_std)
        active_returns_full = np.array(active_returns_full['return']) / 100

    elif portfolio_strategy == "european_equity":
        active_returns_full = EU_data['RM_RF'][window:].to_numpy() / 100

    else:
        raise ValueError("Unknown strategy")

    # --- Simulate rolling 10-year paths ---
    MVR_120, W_120 = [], []
    for i in range(len(active_returns_full) - T):
        zcb_prices = zcb_price_generator(years, T + 1, start=i, data=zcb_data)
        active_returns = active_returns_full[i:i+T]
        summary = simulate_tie_in_path(active_returns, zcb_prices, contributions)
        MVR_120.append(summary.iloc[T, :]['MV_R'])
        W_120.append(summary.iloc[T, :]['W'])
    
    return np.array(MVR_120), np.array(W_120)


# --- Select which strategy to compare ---
chosen_strategy = "markowitz"   # change to "risk_parity" if you want

# --- Run both chosen strategy and benchmark ---
print(f"Running {chosen_strategy.capitalize()} strategy...")
MVR_active, W_active = simulate_strategy(chosen_strategy)
print("Running European Equity benchmark...")
MVR_benchmark, W_benchmark = simulate_strategy("european_equity")

results = {
    chosen_strategy.capitalize(): {"MV_R": MVR_active, "W": W_active},
    "European Equity": {"MV_R": MVR_benchmark, "W": W_benchmark}
}


# --- Plot side-by-side histograms ---
fig, axes = plt.subplots(1, 2, figsize=(13,6), sharey=True)
bins = 40
colors = {
    chosen_strategy.capitalize(): "#1f77b4",
    "European Equity": "#ff7f0e"
}

# Left panel: Reserve portfolio (MV_R)
ax = axes[0]
for name, res in results.items():
    ax.hist(res["MV_R"], bins=bins, alpha=0.45, density=True, label=name,
            color=colors[name], edgecolor='black')
    ax.axvline(np.mean(res["MV_R"]), color=colors[name], linestyle="--", linewidth=2)
ax.set_title("Final Reserve Value (MV_R)")
ax.set_xlabel("Value after 10 years")
ax.set_ylabel("Density")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()

# Right panel: Total wealth (W)
ax = axes[1]
for name, res in results.items():
    ax.hist(res["W"], bins=bins, alpha=0.45, density=True, label=name,
            color=colors[name], edgecolor='black')
    ax.axvline(np.mean(res["W"]), color=colors[name], linestyle="--", linewidth=2)
ax.set_title("Final Total Wealth (W)")
ax.set_xlabel("Value after 10 years")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()

plt.suptitle(f"Distribution of Final Outcomes — {chosen_strategy.capitalize()} vs. European Equity", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("tie_in_histograms.png", dpi=300)
plt.close()

# --- Print summary statistics ---
summary_df = pd.DataFrame({
    name: {
        "MV_R mean": np.mean(res["MV_R"]),
        "MV_R std": np.std(res["MV_R"]),
        "W mean": np.mean(res["W"]),
        "W std": np.std(res["W"])
    }
    for name, res in results.items()
}).T.round(2)

print("\nSummary of outcomes:")
print(summary_df)


def simulate_tie_in_effects(portfolio_strategy: str):
    """Simulate tie-ins: return number of tie-ins and total reserve added from tie-ins per path."""
    T = 120
    years = 10
    contributions = np.zeros(T+1)
    contributions[0] = 100
    window = 36
    allow_short = False
    use_covariance = False
    target_std = 4

    # --- Select active strategy ---
    if portfolio_strategy == "markowitz":
        active_returns_full = np.array(markowitz_historical(EU, window, "tangent", 0.01, 10, allow_short)) / 100
    elif portfolio_strategy == "risk_parity":
        active_returns_full = risk_parity(EU_data, US_data, "EU", "US", False, window,
                                          True, True, True, False, False, False,
                                          use_covariance, allow_short, target_std)
        active_returns_full = np.array(active_returns_full['return']) / 100
    elif portfolio_strategy == "european_equity":
        active_returns_full = EU_data['RM_RF'][window:].to_numpy() / 100
    else:
        raise ValueError("Unknown strategy")

    num_tie_ins, added_to_reserve = [], []

    # --- Rolling simulation ---
    for i in range(len(active_returns_full) - T):
        zcb_prices = zcb_price_generator(years, T + 1, start=i, data=zcb_data)
        active_returns = active_returns_full[i:i+T]
        summary = simulate_tie_in_path(active_returns, zcb_prices, contributions)

        # count tie-ins
        num_tie_ins.append(summary["tie_in"].sum())

        # calculate reserve added *only during tie-ins*
        reserve_changes = summary["MV_R"].diff()
        added_amount = reserve_changes[summary["tie_in"]].sum()
        added_to_reserve.append(added_amount)

    return np.array(num_tie_ins), np.array(added_to_reserve)


# --- Choose strategy to compare ---

# --- Run both chosen and benchmark ---
print(f"Running {chosen_strategy.capitalize()} strategy...")
tieins_active, reserve_added_active = simulate_tie_in_effects(chosen_strategy)
print("Running European Equity benchmark...")
tieins_benchmark, reserve_added_benchmark = simulate_tie_in_effects("european_equity")

results = {
    chosen_strategy.capitalize(): {"tieins": tieins_active, "reserve_added": reserve_added_active},
    "European Equity": {"tieins": tieins_benchmark, "reserve_added": reserve_added_benchmark}
}

# --- Plot results side by side ---
fig, axes = plt.subplots(1, 2, figsize=(13,6))

colors = {
    chosen_strategy.capitalize(): "#1f77b4",
    "European Equity": "#ff7f0e"
}

# Left: number of tie-ins
ax = axes[0]
for name, res in results.items():
    ax.hist(res["tieins"], bins=range(0, int(max(res["tieins"].max(), res["tieins"].max())) + 2),
            alpha=0.45, label=name, color=colors[name], edgecolor='black')
ax.set_title("Distribution of Number of Tie-Ins (per 10-year path)")
ax.set_xlabel("# Tie-ins")
ax.set_ylabel("Frequency")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.6)

# Right: total reserve added from tie-ins
ax = axes[1]
for name, res in results.items():
    ax.hist(res["reserve_added"], bins=40, alpha=0.45, density=True,
            label=name, color=colors[name], edgecolor='black')
    ax.axvline(np.mean(res["reserve_added"]), color=colors[name], linestyle="--", linewidth=2)
ax.set_title("Total Reserve Added by Tie-Ins (per 10-year path)")
ax.set_xlabel("Reserve Added (currency units)")
ax.set_ylabel("Density")
ax.legend() 
ax.grid(True, linestyle="--", alpha=0.6)

plt.suptitle(f"Tie-In Frequency and Impact — {chosen_strategy.capitalize()} vs. European Equity", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("tie_in_effects.png", dpi=300)
plt.close()

# --- Print summary statistics ---
summary = pd.DataFrame({
    name: {
        "Avg # Tie-ins": np.mean(res["tieins"]),
        "Avg Reserve Added": np.mean(res["reserve_added"]),
        "Std Reserve Added": np.std(res["reserve_added"])
    }
    for name, res in results.items()
}).T.round(2)

print("\nSummary of tie-in behavior:")
print(summary)

