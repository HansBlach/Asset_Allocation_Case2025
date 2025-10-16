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
# for one data set
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


# we change to two inputs so we can compare data sets
def mu_cov_pair(df_left, df_right, start, n_months, cols, suffixes=("EU", "US"), date_col="Date"):
    L = df_left.copy()
    R = df_right.copy()
    L[date_col] = pd.to_datetime(L[date_col], errors="coerce")
    R[date_col] = pd.to_datetime(R[date_col], errors="coerce")
    for c in cols:
        L[c] = pd.to_numeric(L[c], errors="coerce")
        R[c] = pd.to_numeric(R[c], errors="coerce")
    
    # need to rename since we have same column names:
    L = L.rename(columns={c: f"{c}_{suffixes[0]}" for c in cols})
    R = R.rename(columns={c: f"{c}_{suffixes[1]}" for c in cols})

    left_cols  = [f"{c}_{suffixes[0]}" for c in cols]
    right_cols = [f"{c}_{suffixes[1]}" for c in cols]
    wanted_cols = [date_col] + left_cols + right_cols
    
    # merge dataset and align on Date (inner join keeps common months only)
    X = pd.merge(L[[date_col] + left_cols], 
                 R[[date_col] + right_cols], 
                 on=date_col, 
                 how="inner"
                 )
    
    # when we start including data
    start = pd.Timestamp(start).to_period("M")  # convert start correctly
    end = start - 1
    
    # begin = first day of the month n_months ago
    begin = end - (n_months - 1)

    # make sure we have defined interval:
    mask = (X[date_col].dt.to_period("M") >= begin) & \
           (X[date_col].dt.to_period("M") <= end)
    
    Xw = X.loc[mask, wanted_cols].dropna()
    if Xw.empty:
        raise ValueError(f"No overlapping data in window {begin}..{end}. Check dates and overlap.")

    # values
    Xw_vals = Xw.drop(columns=[date_col])
    mu_m = Xw_vals.mean()
    Sigma = Xw_vals.cov()

    # in case we need ordering
    ordered = left_cols + right_cols
    mu_m = mu_m[ordered]
    Sigma = Sigma.loc[ordered, ordered]
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
# Start december 24
start = "2025-01"
# We do 36 now for 3 years (can do lenght of dataset if we want all)
n_months = len(df_EU)  

# empty results
results = {}

# run example
# Pair 1: df_EU with df_USEUR  -> 6x6
mu, Sigma = mu_cov_pair(df_EU, df_USEUR, start, n_months, cols, suffixes=("EU","US_EUR"))
print(f"\n=== EU vs US(EUR) | window: {n_months} months ending before {start} ===")
print("Mean vector (mu):")
print(mu.to_string())
print("\nCovariance matrix (Sigma):")
print(Sigma)

# Pair 2: df_long_EU with df_long_USEUR  -> 6x6
mu_long, Sigma_long = mu_cov_pair(df_long_EU, df_long_USEUR, start, n_months, cols, suffixes=("EU_long","US_long(EUR)"))
print(f"\n=== EU Long vs US Long (EUR) | window: {n_months} months ending before {start} ===")
print("Mean vector (mu):")
print(mu_long.to_string())
print("\nCovariance matrix (Sigma):")
print(Sigma_long)