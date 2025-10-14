# Calculate Surplus and Covariance Matrix
import pandas as pd
import numpy as np

# Load the data from CSV files
df_EU = pd.read_csv("csv_files/EXPORT EU EUR.csv")
df_USEUR = pd.read_csv("csv_files/EXPORT US EUR.csv")
df_US = pd.read_csv("csv_files/EXPORT US USD.csv")
df_long_EU = pd.read_csv("csv_files/long_EXPORT EU EUR.csv")
df_long_USEUR = pd.read_csv("csv_files/long_EXPORT US EUR.csv")
df_long_US = pd.read_csv("csv_files/long_EXPORT US USD.csv")

# string to year:
def parse_yyyymm(x):
    s = str(int(x))
    year, month = int(s[:4]), int(s[4:6])
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)

for df in (df_EU, df_USEUR, df_US, df_long_EU, df_long_USEUR, df_long_US):
    df["Date"] = df["Date"].apply(parse_yyyymm)


# function to calculate mean (expected surplus) and the covariance matrix between surplus
def mu_cov_func(df, start, n_months, cols, date_col="Date"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

    # when we start including data
    start = pd.Timestamp(start).to_period("M")
    end = start - 1

    # begin = first day of the month n_months ago
    begin = end - (n_months - 1)  

    # make sure we have defined interval:
    mask = (df[date_col].dt.to_period("M") >= begin) & \
           (df[date_col].dt.to_period("M") <= end)
    X = df.loc[mask, cols].dropna()

    # values
    mu_m = X.mean()
    Sigma = X.cov()
    return mu_m, Sigma



# Loop over 6 relevant data set:
datasets = {
    "EU (EUR)": df_EU,
    "US (EUR)": df_USEUR,
    "US (USD)": df_US,
    "EU Long (EUR)": df_long_EU,
    "US Long (EUR)": df_long_USEUR,
    "US Long (USD)": df_long_US,
}

# Relevant factors - may change after we have decided what is relevant
cols = ["RM_RF", "SMB", "MOM"]
# Start
start = "2020-09"
# We do 36 now for 3 years (can do lenght of dataset if we want all)
n_months = 36

# empty results
results = {}

# run example
for name, df in datasets.items():
    mu, Sigma = mu_cov_func(df, start, n_months, cols)
    results[name] = {"mu": mu, "Sigma": Sigma}

    print(f"\n=== {name} | window: {n_months} months ending before {start} ===")
    print("Monthly expected surplus (mu):")
    print(mu.to_string())
    print("\nMonthly covariance matrix (Sigma):")
    print(Sigma)