import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from arch.bootstrap import CircularBlockBootstrap

from risk_parity import risk_parity

from markowitz import markowitz_historical, returns_to_value

EU_data = pd.read_csv("csv_files/EXPORT EU EUR.csv")

def block_bootstrap(df, n_bootstrap=1000, block_length=10, random_state=None):
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
n_points = 10
factors = True, True, True

for i in range(n_bootstrap):
    portfolios = list(boot_samples[i].columns[5:36])
    marko_data = boot_samples[i][portfolios]    
    marko_returns = markowitz_historical(marko_data, window, n_points=n_points)

    rp_returns = risk_parity(boot_samples[i], window, *factors)['return']

    # --- Markowitz ---
    mean_markowitz.append(np.mean(marko_returns))
    variance_markowitz.append(np.var(marko_returns))
    sharpe_markowitz.append(np.mean(marko_returns) / np.std(marko_returns))

    # --- Risk Parity ---
    mean_risk_parity.append(np.mean(rp_returns))
    variance_risk_parity.append(np.var(rp_returns))
    sharpe_risk_parity.append(np.mean(rp_returns) / np.std(rp_returns))

# --- Summary Table ---
summary_df = pd.DataFrame({
    "Strategy": ["Markowitz", "Risk Parity"],
    "Mean Return": [np.mean(mean_markowitz), np.mean(mean_risk_parity)],
    "Variance": [np.mean(variance_markowitz), np.mean(variance_risk_parity)],
    "Sharpe Ratio": [np.mean(sharpe_markowitz), np.mean(sharpe_risk_parity)]
})

# --- Pretty Formatting ---
summary_df["Mean Return"] = summary_df["Mean Return"].map("{:.6f}".format)
summary_df["Variance"] = summary_df["Variance"].map("{:.6f}".format)
summary_df["Sharpe Ratio"] = summary_df["Sharpe Ratio"].map("{:.3f}".format)

print("\nBootstrap Portfolio Performance Summary")
print(summary_df.to_string(index=False))







# --- Boxplots for Mean and Variance ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Mean returns boxplot
axes[0].boxplot([mean_markowitz, mean_risk_parity], labels=["Markowitz", "Risk Parity"])
axes[0].set_title("Distribution of Mean Returns")
axes[0].set_ylabel("Mean Return")

# Variance boxplot
axes[1].boxplot([variance_markowitz, variance_risk_parity], labels=["Markowitz", "Risk Parity"])
axes[1].set_title("Distribution of Return Variances")
axes[1].set_ylabel("Variance")

plt.suptitle("Bootstrap Distributions of Portfolio Performance", fontsize=14)
plt.tight_layout()
plt.savefig("bootstrap_distributions.png")
plt.show()