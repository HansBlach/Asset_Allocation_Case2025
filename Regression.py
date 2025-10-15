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

def data_interpreter(df):
    # Get some aggregate statistics from the regression results
    # with all factors
    explanatory_cols = ["RM_RF", "SMB", "MOM"]
    res = run_regression(df, explanatory_cols)
    m_R2 = np.mean(res["R2"])
    R2_5_worst = res["R2"].nsmallest(5)
    m_tb = np.mean(abs(res["t(beta)"]))
    m_ts = np.mean(abs(res["t(s)"]))
    m_tm = np.mean(abs(res["t(m)"]))
    t_reject_count_RF_RM = (abs(res["t(beta)"]) < 2.).sum()
    t_reject_names_RF_RM = list(res[abs(res["t(beta)"]) < 2.].index)
    t_reject_count_SMB = (abs(res["t(s)"]) < 2.).sum()
    t_reject_names_SMB = list(res[abs(res["t(s)"]) < 2.].index)
    t_reject_count_MOM = (abs(res["t(m)"]) < 2.).sum()
    t_reject_names_MOM = list(res[abs(res["t(m)"]) < 2.].index)

    # Now run regressions with one factor removed to see explanatory power loss
    explanatory_cols_RM_RF = ["SMB", "MOM"]
    res_RM_RF = run_regression(df, explanatory_cols_RM_RF)
    m_R2_RM_RF = np.mean(res_RM_RF["R2"])
    explanatory_power_loss= m_R2 - np.mean(res_RM_RF["R2"])
    R2_5_worst_RM_RF = res_RM_RF["R2"].nsmallest(5)

    explanatory_cols_SMB = ["RM_RF", "MOM"]
    res_SMB = run_regression(df, explanatory_cols_SMB)
    m_R2_SMB = np.mean(res_SMB["R2"])
    explanatory_power_loss_SMB= m_R2 - np.mean(res_SMB["R2"])
    R2_5_worst_SMB = res_SMB["R2"].nsmallest(5)

    explanatory_cols_MOM = ["RM_RF", "SMB"]
    res_MOM = run_regression(df, explanatory_cols_MOM)
    m_R2_MOM = np.mean(res_MOM["R2"])
    explanatory_power_loss_MOM= m_R2 - np.mean(res_MOM["R2"])
    R2_5_worst_MOM = res_MOM["R2"].nsmallest(5)

    # Create a DataFrame with only portfolios present in the 5 worst R2 tables
    worst_portfolios = set(R2_5_worst.index) | set(R2_5_worst_RM_RF.index) | set(R2_5_worst_SMB.index) | set(R2_5_worst_MOM.index)
    summary_df = pd.DataFrame(0, index=sorted(worst_portfolios), columns=["RM_RF", "SMB", "MOM"])

    # Fill in the 5 worst portfolios for each regression with their R2 values
    for name, worst, result in [
        ("RM_RF", R2_5_worst_RM_RF, res_RM_RF),
        ("SMB", R2_5_worst_SMB, res_SMB),
        ("MOM", R2_5_worst_MOM, res_MOM),
    ]:
        for portfolio in worst.index:
            summary_df.loc[portfolio, name] = result.loc[portfolio, "R2"]

    # For the full model, fill in the 5 worst portfolios in the "none" column
    summary_df["none"] = 0
    for portfolio in R2_5_worst.index:
        summary_df.loc[portfolio, "none"] = res.loc[portfolio, "R2"]

    # Reorder columns
    summary_df = summary_df[["none", "RM_RF", "SMB", "MOM"]]

    # Format values: 2 decimals for nonzero, 0 for zeros
    def format_value(x):
        if x == 0:
            return 0
        return f"{x:.3f}"

    summary_df = summary_df.applymap(format_value)
    # Add m_R2 and explanatory_loss as rows
    summary_df.loc["m_R2"] = [f"{m_R2:.3f}", f"{m_R2_RM_RF:.3f}", f"{m_R2_SMB:.3f}", f"{m_R2_MOM:.3f}"]
    summary_df.loc["explanatory_loss"] = [
        "0.000",  # No loss for full model
        f"{explanatory_power_loss:.3f}",
        f"{explanatory_power_loss_SMB:.3f}",
        f"{explanatory_power_loss_MOM:.3f}"
    ]

    # Add "total" column: count of nonzero R2 values per row
    summary_df["total"] = summary_df[["none", "RM_RF", "SMB", "MOM"]].apply(lambda row: sum([v != 0 for v in row]), axis=1)

    # Rename index label
    summary_df.index.name = "without factor"
    

    # Calculate summary statistics for each factor
    summary = {
        "beta": {
            "t_mean": m_tb,
            "reject_count": t_reject_count_RF_RM,
            "reject_names": ", ".join(t_reject_names_RF_RM)
        },
        "s": {
            "t_mean": m_ts,
            "reject_count": t_reject_count_SMB,
            "reject_names": ", ".join(t_reject_names_SMB)
        },
        "m": {
            "t_mean": m_tm,
            "reject_count": t_reject_count_MOM,
            "reject_names": ", ".join(t_reject_names_MOM)
        }
    }

    # Create the summary table
    summary_table = pd.DataFrame({
        "beta": [summary["beta"]["t_mean"], summary["beta"]["reject_count"], summary["beta"]["reject_names"]],
        "s": [summary["s"]["t_mean"], summary["s"]["reject_count"], summary["s"]["reject_names"]],
        "m": [summary["m"]["t_mean"], summary["m"]["reject_count"], summary["m"]["reject_names"]],
    }, index=["t_mean", "reject_count", "reject_names"])
    return summary_df, summary_table

data_interpreter(df_EU)

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

