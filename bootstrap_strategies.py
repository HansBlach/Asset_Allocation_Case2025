import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from arch.bootstrap import CircularBlockBootstrap

from risk_parity import risk_parity

from markowitz import markowitz_historical, returns_to_value

EU_data = pd.read_csv("csv_files/EXPORT EU EUR.csv")

def annualizer(mean_return, type, periods_per_year = 12):
    if type == "return":
        return (1 + mean_return/100) ** periods_per_year - 1
    if type == "volatility":
        return mean_return * periods_per_year


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

n_bootstrap = 1000
boot_samples = block_bootstrap(EU_data, n_bootstrap=n_bootstrap, block_length=12, random_state=42)

mean_markowitz, variance_markowitz, sharpe_markowitz = [], [], []
mean_risk_parity, variance_risk_parity, sharpe_risk_parity = [], [], []

window = 36
n_points = 3
factors = True, True, True

for i in range(n_bootstrap):
    print(f"Processing bootstrap sample {i+1}/{n_bootstrap}", end='\r')

    portfolios = list(boot_samples[i].columns[5:36])
    marko_data = boot_samples[i][portfolios]    
    marko_returns = markowitz_historical(marko_data, window, n_points=n_points)

    rp_returns = risk_parity(boot_samples[i], window, *factors)['return']

    # --- Markowitz ---
    mean_markowitz.append(annualizer(np.mean(marko_returns), type="return"))
    variance_markowitz.append(annualizer(np.var(marko_returns), type="volatility"))
    sharpe_markowitz.append(np.mean(marko_returns) / np.std(marko_returns))

    # --- Risk Parity ---
    mean_risk_parity.append(annualizer(np.mean(rp_returns), type="return"))
    variance_risk_parity.append(annualizer(np.var(rp_returns), type="volatility"))
    sharpe_risk_parity.append(np.mean(rp_returns) / np.std(rp_returns))

# --- Summary Table ---
summary_df = pd.DataFrame({
    "Strategy": ["Markowitz", "Risk Parity"],
    "Mean Excess Return (annualized)": [np.mean(mean_markowitz), np.mean(mean_risk_parity)],
    "Variance": [np.mean(variance_markowitz), np.mean(variance_risk_parity)],
    "Sharpe Ratio": [np.mean(sharpe_markowitz), np.mean(sharpe_risk_parity)]
})

# --- Pretty Formatting ---
summary_df["Mean Return"] = summary_df["Mean Excess Return (annualized)"].map("{:.6f}".format)
summary_df["Variance"] = summary_df["Variance"].map("{:.6f}".format)
summary_df["Sharpe Ratio"] = summary_df["Sharpe Ratio"].map("{:.3f}".format)

print("\nBootstrap Portfolio Performance Summary")
print(summary_df.to_string(index=False))




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
plt.show()





# --- Boxplots for Mean and Sharpe ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Mean returns boxplot
axes[0].boxplot([mean_markowitz, mean_risk_parity], labels=["Markowitz", "Risk Parity"])
axes[0].set_title("Distribution of Mean Excess Returns")
axes[0].set_ylabel("Mean ExcessReturn")

# Variance boxplot
axes[1].boxplot([sharpe_markowitz, sharpe_risk_parity], labels=["Markowitz", "Risk Parity"])
axes[1].set_title("Distribution of Sharpe Ratios")
axes[1].set_ylabel("Sharpe Ratio")

plt.suptitle("Bootstrap Distributions of Portfolio Performance", fontsize=14)
plt.tight_layout()
plt.savefig("bootstrap_distributions.png")
plt.show()