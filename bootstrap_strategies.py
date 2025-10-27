import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from arch.bootstrap import CircularBlockBootstrap

from risk_parity import risk_parity

from markowitz import markowitz_historical, returns_to_value

# -------- Set parameters -------

n_bootstrap = 100

window = 36
n_points = 10
include_US = True

allow_short = True

has_MOM1, has_SMB1, has_RM_RF1 = True, False, True 
has_MOM2, has_SMB2, has_RM_RF2 = False, False, False

# ------------------------------

EU_data = pd.read_csv("csv_files/long_EXPORT EU EUR.csv")

US_data = pd.read_csv("csv_files/long_EXPORT US EUR.csv")

if allow_short:
    EU_data = pd.read_csv("csv_files/EXPORT EU EUR.csv")

    US_data = pd.read_csv("csv_files/EXPORT US EUR.csv")

def annualizer(mean_return, type, periods_per_year = 12):
    if type == "return":
        return ((1 + mean_return/100) ** periods_per_year - 1)*100
    if type == "volatility":
        return (mean_return/100 * np.sqrt(periods_per_year))*100


def block_bootstrap(df, n_bootstrap=1000, block_length=12, random_state=None):
    rng = np.random.default_rng(random_state)
    n = len(df)
    n_blocks = int(np.ceil(n / block_length))
    
    bootstraps = []
    for _ in range(n_bootstrap):
        # Randomly choose starting indices for blocks
        start_indices = rng.integers(0, n - block_length + 1, size=n_blocks)
        # Concatenate the selected blocks
        sample = pd.concat(
            [df.iloc[i:i + block_length] for i in start_indices],
            ignore_index=True
        ).iloc[:n]  # Trim to original length
        bootstraps.append(sample)
    return bootstraps

# Assuming your previous setup & functions (block_bootstrap, markowitz_historical, risk_parity) are defined



# --------------------------------

boot_EU = block_bootstrap(EU_data, n_bootstrap=n_bootstrap, block_length=12, random_state=42)

boot_US = block_bootstrap(US_data, n_bootstrap=n_bootstrap, block_length=12, random_state=42)

mean_markowitz, std_markowitz, sharpe_markowitz = [], [], []
mean_risk_parity, std_risk_parity, sharpe_risk_parity = [], [], []
mean_market, std_market, sharpe_market = [], [], []

annualized_mean_markowitz, annualized_std_markowitz, annualized_sharpe_markowitz = [], [], []
annualized_mean_risk_parity, annualized_std_risk_parity, annualized_sharpe_risk_parity = [], [], []
annualized_mean_market, annualized_std_market, annualized_sharpe_market = [], [], []


for i in range(n_bootstrap):
    print(f"Processing bootstrap sample {i+1}/{n_bootstrap}", end='\r')

    portfolios = list(boot_EU[i].columns[5:36])
    EU_marko = boot_EU[i][portfolios]   
    US_marko = boot_US[i][portfolios] 

    if include_US:
        marko_data = pd.concat([EU_marko,US_marko],axis = 1)
    
    if not include_US:
        marko_data = EU_marko

    marko_returns = markowitz_historical(marko_data, window, n_points=n_points, allow_short= allow_short)

    rp_returns = risk_parity(boot_EU[i],boot_US[i], "EU", "US", window = window, include_market2 = include_US,
                             has_MOM1 = has_MOM1,has_SMB1 = has_SMB1,has_RM_RF1 = has_RM_RF1, has_MOM2 = has_MOM2, has_SMB2 = has_SMB2, has_RM_RF2 = has_RM_RF2)['return']


    market_returns = boot_EU[i]['RM_RF']

    # --- Markowitz ---
    annualized_mean_markowitz.append(annualizer(np.mean(marko_returns), type="return"))
    annualized_std_markowitz.append(annualizer(np.std(marko_returns), type="volatility"))

    # --- Risk Parity ---
    annualized_mean_risk_parity.append(annualizer(np.mean(rp_returns), type="return"))
    annualized_std_risk_parity.append(annualizer(np.std(rp_returns), type="volatility"))

    # --- Market ---
    annualized_mean_market.append(annualizer(np.mean(market_returns), type="return"))
    annualized_std_market.append(annualizer(np.std(market_returns), type="volatility"))

    # Not annualized

    # --- Markowitz ---
    mean_markowitz.append(np.mean(marko_returns))
    std_markowitz.append(np.std(marko_returns))

    # --- Risk Parity ---
    mean_risk_parity.append(np.mean(rp_returns))
    std_risk_parity.append(np.std(rp_returns))

    # --- Market ---
    mean_market.append(np.mean(market_returns))
    std_market.append(np.std(market_returns))


sharpe_markowitz = np.array(mean_markowitz)/np.array(std_markowitz)

sharpe_risk_parity = np.array(mean_risk_parity)/np.array(std_risk_parity)

sharpe_market = np.array(mean_market)/np.array(std_market)

annualized_sharpe_markowitz = np.array(annualized_mean_markowitz)/np.array(annualized_std_markowitz)
annualized_sharpe_risk_parity = np.array(annualized_mean_risk_parity)/np.array(annualized_std_risk_parity)
annualized_sharpe_market = np.array(annualized_mean_market)/np.array(annualized_std_market)

if allow_short:
    print("Shorting was allowed")

if not allow_short:
    print("Shorting was not allowed")


# --- Summary Table not annualized ---
summary_df = pd.DataFrame({
    "Strategy": ["Markowitz", "Risk Parity", "European Market"],
    "Mean Excess Return": [np.mean(mean_markowitz), np.mean(mean_risk_parity), np.mean(mean_market)],
    "Standard deviation": [np.mean(std_markowitz), np.mean(std_risk_parity), np.mean(std_market)],
    "Sharpe Ratio": [np.mean(sharpe_markowitz), np.mean(sharpe_risk_parity), np.mean(sharpe_market)]
})

# --- Pretty Formatting ---
summary_df["Mean Excess Return"] = summary_df["Mean Excess Return"].map("{:.6f}".format)
summary_df["Standard deviation"] = summary_df["Standard deviation"].map("{:.6f}".format)
summary_df["Sharpe Ratio"] = summary_df["Sharpe Ratio"].map("{:.3f}".format)

print("\nBootstrap Portfolio Performance Summary (not annualized)")
print(summary_df.to_string(index=False))

# --- Summary Table (annualized) ---

annualized_summary_df = pd.DataFrame({
    "Strategy": ["Markowitz", "Risk Parity", "European Market"],
    "Mean Excess Return": [np.mean(annualized_mean_markowitz), np.mean(annualized_mean_risk_parity), np.mean(annualized_mean_market)],
    "Standard deviation": [np.mean(annualized_std_markowitz), np.mean(annualized_std_risk_parity), np.mean(annualized_std_market)],
    "Sharpe Ratio": [np.mean(annualized_sharpe_markowitz), np.mean(annualized_sharpe_risk_parity), np.mean(annualized_sharpe_market)]
})

# --- Pretty Formatting ---
annualized_summary_df["Mean Excess Return"] = annualized_summary_df["Mean Excess Return"].map("{:.6f}".format)
annualized_summary_df["Standard deviation"] = annualized_summary_df["Standard deviation"].map("{:.6f}".format)
annualized_summary_df["Sharpe Ratio"] = annualized_summary_df["Sharpe Ratio"].map("{:.3f}".format)
print("\nBootstrap Portfolio Performance Summary (annualized)")
print(annualized_summary_df.to_string(index=False))



# --- Bootstrap distribution charts for Mean and Sharpe ---

bins = 10

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Mean returns histogram
axes[0].hist(mean_markowitz, bins=bins, alpha=0.7, label="Markowitz")
axes[0].hist(mean_risk_parity, bins=bins, alpha=0.7, label="Risk Parity")
axes[0].set_title("Bootstrap Distribution of Mean Excess Returns")
axes[0].set_xlabel("Mean ExcessReturn")
axes[0].set_ylabel("Frequency")
axes[0].legend()

# Sharpe ratio histogram

axes[1].hist(sharpe_markowitz, bins=bins, alpha=0.7, label="Markowitz")
axes[1].hist(sharpe_risk_parity, bins=bins, alpha=0.7, label="Risk Parity")
axes[1].set_title("Bootstrap Distribution of Sharpe Ratios")
axes[1].set_xlabel("Sharpe Ratio")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.suptitle("Bootstrap Distributions of Portfolio Performance", fontsize=14)
plt.tight_layout()
plt.savefig("bootstrap_histograms.png")
#plt.show()





# --- Boxplots for Mean and Sharpe ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Mean returns boxplot
boxplot_kwargs = {}
try:
    # Matplotlib 3.9+ uses 'tick_labels'
    axes[0].boxplot.__call__
    boxplot_kwargs['tick_labels'] = ["Markowitz", "Risk Parity"]
except Exception:
    # Fallback for older Matplotlib versions
    boxplot_kwargs['labels'] = ["Markowitz", "Risk Parity"]
axes[0].boxplot([mean_markowitz, mean_risk_parity], **boxplot_kwargs)
axes[0].set_title("Distribution of Mean Excess Returns")
axes[0].set_ylabel("Mean ExcessReturn")

# Variance boxplot
boxplot_kwargs = {}
try:
    boxplot_kwargs['tick_labels'] = ["Markowitz", "Risk Parity"]
except Exception:
    boxplot_kwargs['labels'] = ["Markowitz", "Risk Parity"]
axes[1].boxplot([sharpe_markowitz, sharpe_risk_parity], **boxplot_kwargs)
axes[1].set_title("Distribution of Sharpe Ratios")
axes[1].set_ylabel("Sharpe Ratio")

plt.suptitle("Bootstrap Boxplots of Portfolio Performance", fontsize=14)
plt.tight_layout()
plt.savefig("bootstrap_boxplots.png")
#plt.show()