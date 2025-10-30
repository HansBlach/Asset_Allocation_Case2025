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
    active_returns: np.ndarray,       # (T,) monthly discrete returns of the active portfolio
    zcb_prices: np.ndarray,           # (T+1,) rolling ZCB prices incl. month 0 and maturity
    contributions: np.ndarray = None, # (T+1,) monthly contributions, first element is initial deposit
    cfg: TieInConfig | None = None,   # tie-in configuration (targets, trigger, etc.)
) -> pd.DataFrame:
    if cfg is None:
        cfg = TieInConfig()

    T = len(active_returns)
    assert len(zcb_prices) == T + 1, "zcb_prices must have length T+1 (including month 0 and maturity)."

    if contributions is None:
        contributions = np.zeros(T + 1)
    contributions = np.asarray(contributions, dtype=float)
    assert len(contributions) == T + 1, "contributions must have length T+1 (including month 0)."

    # --- Initial setup ---
    W0 = cfg.initial_wealth + contributions[0]
    MV_R = W0 / cfg.L_target        # reserve value so L = L_target initially
    MV_A = W0 - MV_R
    N = MV_R / zcb_prices[0]        # face value of the ZCB (guarantee units)

    rows = [{
        "month": 0,
        "MV_A": MV_A,
        "MV_R": MV_R,
        "N": N,
        "P": zcb_prices[0],
        "W": W0,
        "C": contributions[0],
        "c_A": contributions[0] * MV_A / W0,
        "c_R": contributions[0] * MV_R / W0,
        "L": W0 / MV_R,
        "tie_in": False,
        "transfer_to_reserve": 0.0,
    }]

    # --- Monthly simulation loop ---
    for i in range(1, T + 1):
        # 1) Active portfolio evolves
        MV_A *= (1.0 + active_returns[i - 1])

        # 2) Reserve repriced from rolling ZCB
        P_i = zcb_prices[i]
        MV_R = N * P_i
        W = MV_A + MV_R

        # 3) Compute funded ratio and check trigger
        L = W / MV_R
        tie_in = L > cfg.L_trigger
        transfer = 0.0

        # 4) If tie-in: lock in gains by raising guarantee and rebalancing
        if tie_in:
            MV_R_old = MV_R

            # target reserve value to restore funded ratio to L_target
            MV_R_target = W / cfg.L_target
            transfer = MV_R_target - MV_R

            # ensure we only transfer positive amount
            if transfer > 0:
                N_added = transfer / P_i
                N += N_added
                MV_R = N * P_i          # new reserve market value
                MV_A = W - MV_R         # reduce active by same amount

        # 5) Apply new contributions
        C = contributions[i]
        if W > 0:
            c_R = C * MV_R / W
            c_A = C * MV_A / W
        else:
            c_R = c_A = 0.0
        MV_R += c_R
        MV_A += c_A
        N += c_R / P_i                         # contributions increase ZCB face value

        # Record month
        rows.append({
            "month": i,
            "MV_A": MV_A,
            "MV_R": MV_R,
            "c_A": c_A,
            "c_R": c_R,
            "N": N,
            "P": P_i,
            "W": MV_A + MV_R,
            "L": (MV_A + MV_R) / MV_R,
            "tie_in": tie_in,
            "transfer_to_reserve": transfer,
        })

    # --- Output history ---
    cols = ["month", "MV_A", "MV_R", "c_A", "c_R", "N", "P",
            "W", "L", "tie_in", "transfer_to_reserve"]
    return pd.DataFrame(rows, columns=cols)




# # --- Set parameters ---
# T = 120
# years = 10
# contributions = np.zeros(T+1)
# contributions[0] = 100  # Initial contribution at month 0

allow_short = False
# portfolio_strategy = "european_equity" # "markowitz" or "risk_parity", european_equity
window = 36

# if allow_short:
#     EU_data = pd.read_csv("csv_files/EXPORT EU EUR.csv")
#     US_data = pd.read_csv("csv_files/EXPORT US EUR.csv")


columns_to_add = list(EU_data.columns[2:36])

# # Add the risk free rate to the portfolios such that they are no longer excess returns

EU_data[columns_to_add] = EU_data[columns_to_add].add(EU_data['RF'], axis=0)
US_data[columns_to_add] = US_data[columns_to_add].add(US_data['RF'], axis=0)

# ## Markowitz
n_points = 50
strategy = "tangent"
mu_target = 0.05

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

# eu_factors, us_factors = ["RM_RF", "MOM"], ["RM_RF", "MOM", "SMB"]

def analyze_tie_in_strategies(strategy_1: str,
                              strategy_2: str | None,
                              zcb_data: pd.DataFrame,
                              EU_data: pd.DataFrame,
                              US_data: pd.DataFrame,
                              eu_factors: list,
                              us_factors: list,
                              contributions: np.ndarray = None,
                              years: int = 10,
                              window: int = 36,
                              allow_short: bool = False,
                              use_covariance: bool = False,
                              target_std: float = 4,
                              cfg: TieInConfig | None = None):
    """
    Analyze one or two investment strategies (e.g. 'markowitz', 'european_equity') 
    under the tie-in mechanism:
      - Runs rolling 10-year tie-in simulations
      - Generates histograms for Wealth & Reserve
      - Visualizes path with most tie-ins
      - Outputs LaTeX summary table (saved to file)

    If `strategy_2` is None, only one strategy is analyzed.
    """

    from RF_EUR import zcb_price_generator
    from markowitz import markowitz_historical
    from risk_parity import risk_parity
    from tie_in import simulate_tie_in_path  # adjust path if needed

    T = 12 * years
    if contributions is None:
        contributions = np.zeros(T + 1)
        contributions[0] = 100

    # ---------------------------------------------------------
    # BUILD ACTIVE RETURNS
    # ---------------------------------------------------------
    def get_active_returns(name: str) -> np.ndarray:
        if name == "markowitz":
            return np.array(
                markowitz_historical(EU_data, US_data, eu_factors, us_factors,
                                     window, "tangent", 0.05, 50, allow_short)
            ) / 100
        elif name == "risk_parity":
            rp = risk_parity(EU_data, US_data, "EU", "US", False, window,
                             True, True, True, False, False, False,
                             use_covariance, allow_short, target_std)
            return np.array(rp["return"]) / 100
        elif name == "european_equity":
            return EU_data["RM_RF"][window:].to_numpy() / 100
        else:
            raise ValueError(f"Unknown strategy '{name}'")

    # ---------------------------------------------------------
    # RUN A SINGLE STRATEGY
    # ---------------------------------------------------------
    def run_strategy(name: str):
        active_returns_full = get_active_returns(name)
        MVR_120, W_120, TIE_120 = [], [], []

        for i in range(len(active_returns_full) - T):
            zcb_prices = zcb_price_generator(years, T + 1, start=i, data=zcb_data)
            active_returns = active_returns_full[i:i + T]
            summary = simulate_tie_in_path(active_returns, zcb_prices, contributions, cfg)
            MVR_120.append(summary.iloc[T]["MV_R"])
            W_120.append(summary.iloc[T]["W"])
            TIE_120.append(summary["tie_in"].sum())

        return np.array(MVR_120), np.array(W_120), np.array(TIE_120)

    # ---------------------------------------------------------
    # RUN ONE OR TWO STRATEGIES
    # ---------------------------------------------------------
    results = {}

    print(f"Running {strategy_1.capitalize()} strategy...")
    MVR_1, W_1, TIE_1 = run_strategy(strategy_1)
    results[strategy_1.capitalize()] = {"MV_R": MVR_1, "W": W_1, "tieins": TIE_1}

    if strategy_2:
        print(f"Running {strategy_2.capitalize()} strategy...")
        MVR_2, W_2, TIE_2 = run_strategy(strategy_2)
        results[strategy_2.capitalize()] = {"MV_R": MVR_2, "W": W_2, "tieins": TIE_2}

    # ---------------------------------------------------------
    # HISTOGRAMS
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13,6), sharey=True)
    bins = 40

    colors = ["#1f77b4", "#ff7f0e"]
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(results.keys())}

    # Reserve histogram
    ax = axes[0]
    for name, res in results.items():
        ax.hist(res["MV_R"], bins=bins, alpha=0.45, density=True,
                label=name, color=color_map[name], edgecolor='black')
        ax.axvline(np.mean(res["MV_R"]), color=color_map[name], linestyle="--", linewidth=2)
    ax.set_title("Guarantee")
    ax.set_xlabel("Value after 10 years")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Wealth histogram
    ax = axes[1]
    for name, res in results.items():
        ax.hist(res["W"], bins=bins, alpha=0.45, density=True,
                label=name, color=color_map[name], edgecolor='black')
        ax.axvline(np.mean(res["W"]), color=color_map[name], linestyle="--", linewidth=2)
    ax.set_title("Wealth")
    ax.set_xlabel("Value after 10 years")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("tie_in_histograms.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # MOST ACTIVE PATH (from strategy_1)
    # ---------------------------------------------------------
    active_returns_full = get_active_returns(strategy_1)
    best_i, best_tieins, best_summary = None, -1, None
    for i in range(len(active_returns_full) - T):
        zcb_prices = zcb_price_generator(years, T + 1, start=i, data=zcb_data)
        active_returns = active_returns_full[i:i+T]
        summary = simulate_tie_in_path(active_returns, zcb_prices, contributions, cfg)
        tieins = summary["tie_in"].sum()
        if tieins > best_tieins:
            best_i, best_tieins, best_summary = i, tieins, summary

    wealth = best_summary["W"].to_numpy()
    reserve = best_summary["MV_R"].to_numpy()
    guarantee = best_summary["N"].to_numpy()
    months = best_summary["month"].to_numpy()
    tie_idx = best_summary.index[best_summary["tie_in"]]

    plt.figure(figsize=(8,6))
    plt.plot(months, wealth, color="red", label="Wealth")
    plt.plot(months, reserve, color="blue", label="Reserve")
    plt.plot(months, guarantee, color="green", label="Guarantee", linewidth=2)
    plt.scatter(months[tie_idx], guarantee[tie_idx],
                color="green", edgecolor="black", zorder=5, s=60, label="Tie-in events")
    plt.xlabel("Month")
    plt.ylabel("Amount")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tie_in_path_visualization.png", dpi=300)
    plt.close()
    print(f"Path with most tie-ins for {strategy_1.capitalize()} starts at index {best_i} with {best_tieins} tie-ins.")
    # ---------------------------------------------------------
    # SUMMARY TABLE (same format as LaTeX version)
    # ---------------------------------------------------------
    rows = []
    for name, res in results.items():
        MV_R, W, tieins = res["MV_R"], res["W"], res["tieins"]
        avg_G = np.mean(MV_R)
        min_G = np.min(MV_R)
        std_G = np.std(MV_R)
        avg_W = np.mean(W)
        std_W = np.std(W)
        avg_T = np.mean(tieins)

        rows.append({
            "Strategy": name,
            "Avg(G)": round(avg_G, 4),
            "Min(G)": round(min_G, 4),
            "Std(G)": round(std_G, 4),
            "Avg(W)": round(avg_W, 4),
            "Std(W)": round(std_W, 4),
            "Avg(Tie-ins)": round(avg_T, 4)
        })

    summary_df = pd.DataFrame(rows)

    latex_caption = (f"Results by leverage multiplier $m$ for {strategy_1.capitalize()}"
                    if not strategy_2
                    else f"Results by leverage multiplier $m$ ({strategy_1.capitalize()} vs. {strategy_2.capitalize()})")

    latex_table = summary_df.to_latex(
        index=False,
        caption=latex_caption,
        label="tab:guarantee_results_L",
        column_format="lccccccc",
        escape=False
    )

    filename = f"tie_in_statistics_{strategy_1.lower()}" + (f"_vs_{strategy_2.lower()}" if strategy_2 else "") + ".tex"
    with open(filename, "w") as f:
        f.write(latex_table)

    print("\n✅ Outputs saved:")
    print(f" - {filename}")
    print(summary_df)
    print("\nLaTeX Table:\n", latex_table)

    return summary_df, latex_table

cfg = TieInConfig(L_target=1.2, L_trigger=1.3)

# analyze_tie_in_strategies(
#     strategy_1="markowitz",
#     strategy_2="european_equity",
#     zcb_data=zcb_data,
#     EU_data=EU_data,
#     US_data=US_data,
#     eu_factors=["RM_RF", "MOM"],
#     us_factors=["RM_RF", "MOM", "SMB"],
#     cfg=cfg
# )

analyze_tie_in_strategies(
    strategy_1="european_equity",
    strategy_2=None,
    zcb_data=zcb_data,
    EU_data=EU_data,
    US_data=US_data,
    eu_factors=["RM_RF", "MOM"],
    us_factors=["RM_RF", "MOM", "SMB"],
    cfg=cfg
)

import matplotlib.dates as mdates
import pandas as pd
import matplotlib.pyplot as plt

# ---- Tie-in simulator (parallel to CPPI path_simulator) ----
def tie_in_path_simulator(active_returns_full, zcb_data, years, T, cfg=TieInConfig(), summary_path=0):
    """Simulate rolling 10-year tie-in paths and return final stats."""
    MV_A_120, MV_R_120, W_120, Return_120, guarantee, num_tie_in = [], [], [], [], [], []

    for i in range(len(active_returns_full) - T):
        zcb_prices = zcb_price_generator(years, T + 1, start=i, data=zcb_data)
        active_returns = active_returns_full[i:T+i]

        assert active_returns.shape == (T,), "active_returns wrong length"
        assert zcb_prices.shape == (T + 1,), "zcb_prices wrong length"

        summary = simulate_tie_in_path(active_returns, zcb_prices, cfg=cfg)
        summary_120 = summary.iloc[T, :]

        MV_A_120.append(summary_120["MV_A"])
        MV_R_120.append(summary_120["MV_R"])
        W_120.append(summary_120["W"])
        Return_120.append(summary_120["W"] / 100 - 1.0)
        guarantee.append(summary_120["N"])  # final guarantee = face value N
        num_tie_in.append(sum(summary["tie_in"]))

        if i == summary_path:
            active_returns_saved = active_returns
            zcb_prices_saved = zcb_prices

    summary_print = simulate_tie_in_path(active_returns_saved, zcb_prices_saved, cfg=cfg)
    return MV_A_120, MV_R_120, W_120, Return_120, num_tie_in, guarantee, summary_print


# ---- Tie-in path visualization (same start-index logic as CPPI) ----
def plot_tie_in_path(
    active_returns_full: np.ndarray,
    zcb_data: pd.DataFrame,
    years: int,
    T: int,
    cfg: TieInConfig,
    start_index: int,
    save_path: str = "tie_in_path_visualization.png",
):
    """Plots Wealth, Reserve, and Guarantee over time for a single tie-in path with calendar year x-axis."""
    zcb_prices = zcb_price_generator(years, T + 1, start=start_index, data=zcb_data)
    active_returns = active_returns_full[start_index:start_index + T]

    summary = simulate_tie_in_path(active_returns, zcb_prices, cfg=cfg)

    # --- Create date index ---
    start_date = pd.Timestamp("2007-08-01") + pd.DateOffset(months=start_index)
    dates = pd.date_range(start=start_date, periods=len(summary), freq="M")

    wealth = summary["W"].to_numpy()
    reserve = summary["MV_R"].to_numpy()
    guarantee = summary["N"].to_numpy()
    tie_idx = summary.index[summary["tie_in"]]

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(dates, wealth, color="red", label="Wealth")
    ax.plot(dates, reserve, color="blue", label="Reserve")
    ax.plot(dates, guarantee, color="green", label="Guarantee", linewidth=2)
    ax.scatter(dates[tie_idx], guarantee[tie_idx],
               color="green", edgecolor="black", zorder=5, s=60, label="Tie-in events")

    # --- Format x-axis to show years ---
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

    print(f"✅ Tie-in path visualization saved to '{save_path}' (start index = {start_index})")
    return summary

active_returns_full = EU_data["RM_RF"][window:].to_numpy() / 100
years, T = 10, 120
cfg = TieInConfig(initial_wealth=100, L_target=1.25, L_trigger=1.3)

# Example: use index 14 (≈ October 2008)
plot_tie_in_path(
    active_returns_full=active_returns_full,
    zcb_data=zcb_data,
    years=years,
    T=T,
    cfg=cfg,
    start_index=14,
)
