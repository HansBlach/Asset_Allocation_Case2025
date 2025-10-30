# ==========================
#   HISTOGRAM COMPARISON
# ==========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
from RF_EUR import zcb_price_generator
from risk_parity import risk_parity
from markowitz import markowitz_historical
from tie_in import simulate_tie_in_path

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
plt.show()

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
plt.show()

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

