# Make regression
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load the data from CSV files
df_EU = pd.read_csv("csv_files/EXPORT EU EUR.csv")
df_USEUR = pd.read_csv("csv_files/EXPORT US EUR.csv")
df_US = pd.read_csv("csv_files/EXPORT US USD.csv")
df_long_EU = pd.read_csv("csv_files/long_EXPORT EU EUR.csv")
df_long_USEUR = pd.read_csv("csv_files/long_EXPORT US EUR.csv")
df_long_US = pd.read_csv("csv_files/long_EXPORT US USD.csv")

# Define the target and explanatory variables
explanatory_cols = ["RM_RF", "SMB", "MOM"]

def run_regression(df, explanatory_cols):
    # Fra chat: Coerce to numeric where possible and drop impossible rows per portfolio ---
    # Do for all portfolios in sorted data set
    portfolio_cols = list(df.columns[5:36])
    # (Factor columns must exist)
    missing = [c for c in explanatory_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing factor columns: {missing}. Found columns: {df.columns.tolist()}")
    # ikke chat

    # store results
    rows = []   # Metrics and intercepts
    coef_rows = [] # Coefficients

    # Loop over portfolios
    for portfolio in portfolio_cols:
        # Skip non-numeric portfolio columns
        if not pd.api.types.is_numeric_dtype(df[portfolio]):
            try:
                df[portfolio] = pd.to_numeric(df[portfolio], errors="coerce")
            except Exception:
                continue
        # Build clean dataset for this portfolio
        data = pd.concat([df[explanatory_cols], df[portfolio].rename("y")], axis=1)
        data = data.apply(pd.to_numeric, errors="coerce").dropna()

        X = data[explanatory_cols].copy()
        y = data["y"].copy()
        
        # Create and fit the model and t-statistic
       # === Use statsmodels so we get standard errors and t-stats ===
        X_const = sm.add_constant(X)                     # adds intercept term 'const'
        ols_res = sm.OLS(y, X_const).fit()               # OLS fit

        # Map names for clarity
        a      = ols_res.params.get("const", np.nan)
        t_a    = ols_res.tvalues.get("const", np.nan)

        beta   = ols_res.params.get("RM_RF", np.nan)
        t_beta = ols_res.tvalues.get("RM_RF", np.nan)

        s      = ols_res.params.get("SMB", np.nan)
        t_s    = ols_res.tvalues.get("SMB", np.nan)

        m      = ols_res.params.get("MOM", np.nan)
        t_m    = ols_res.tvalues.get("MOM", np.nan)

        # Predictions for metrics you already record
        y_pred = ols_res.predict(X_const)
        
        # Store results
        rows.append({
            "portfolio": portfolio,
            "a": a,
            "t(a)": t_a,
            "beta": beta,
            "t(beta)": t_beta,
            "s": s,
            "t(s)": t_s,
            "m": m,
            "t(m)": t_m,
            "R2": ols_res.rsquared,
            #"MSE": np.sqrt((y-y_pred)^2), KOM EVT. TILBAGE TIL FEJLEN
        })

        # Collect coefficients in tidy form
        for name in ["RM_RF", "SMB", "MOM"]:
            coef_rows.append({
                "portfolio": portfolio,
                "feature": name,
                "coef": ols_res.params.get(name, np.nan),
                "t": ols_res.tvalues.get(name, np.nan)
            })
        
    # Results tables
    results_df = pd.DataFrame(rows).set_index("portfolio").sort_values("R2", ascending=False)
    return results_df

# Run regression for EU EUR
results_df_EU = run_regression(df_EU, explanatory_cols)

# Run regression for US EUR
results_df_USEUR = run_regression(df_USEUR, explanatory_cols)

# Run regression for US USD
results_df_US = run_regression(df_US, explanatory_cols)

# Run regression for EU EUR
results_df_long_EU = run_regression(df_long_EU, explanatory_cols)

# Run regression for US EUR
results_df_long_USEUR = run_regression(df_long_USEUR, explanatory_cols)

# Run regression for US USD
results_df_long_US = run_regression(df_long_US, explanatory_cols)



###################################
# Print data i nyt format:
# 1:
results_df_EU.to_csv("csv_files/results_df_EU.csv")        # metrics per portfolio
results_df_USEUR.to_csv("csv_files/results_df_USEUR.csv")
results_df_US.to_csv("csv_files/results_df_US.csv")
results_df_long_EU.to_csv("csv_files/results_df_long_EU.csv")
results_df_long_USEUR.to_csv("csv_files/results_df_long_USEUR.csv")
results_df_long_US.to_csv("csv_files/results_df_long_US.csv")

