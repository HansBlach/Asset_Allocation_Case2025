# Make regression
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load the data from CSV files
df_EU = pd.read_csv("csv_files/EXPORT EU EUR.csv")
print(df_EU)
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
    m_ts_no_RM_RF = np.mean(abs(res_RM_RF["t(s)"]))
    m_tm_no_RM_RF = np.mean(abs(res_RM_RF["t(m)"]))
    t_reject_count_SMB_no_RM_RF = (abs(res_RM_RF["t(s)"]) < 2.).sum()
    t_reject_names_SMB_no_RM_RF = list(res_RM_RF[abs(res_RM_RF["t(s)"]) < 2.].index)
    t_reject_count_MOM_no_RM_RF = (abs(res_RM_RF["t(m)"]) < 2.).sum()
    t_reject_names_MOM_no_RM_RF = list(res_RM_RF[abs(res_RM_RF["t(m)"]) < 2.].index)

    explanatory_cols_SMB = ["RM_RF", "MOM"]
    res_SMB = run_regression(df, explanatory_cols_SMB)
    m_R2_SMB = np.mean(res_SMB["R2"])
    explanatory_power_loss_SMB= m_R2 - np.mean(res_SMB["R2"])
    R2_5_worst_SMB = res_SMB["R2"].nsmallest(5)
    m_tb_no_SMB = np.mean(abs(res_SMB["t(beta)"]))
    m_tm_no_SMB = np.mean(abs(res_SMB["t(m)"]))
    t_reject_count_RF_RM_no_SMB = (abs(res_SMB["t(beta)"]) < 2.).sum()
    t_reject_names_RF_RM_no_SMB = list(res_SMB[abs(res_SMB["t(beta)"]) < 2.].index)
    t_reject_count_MOM_no_SMB = (abs(res_SMB["t(m)"]) < 2.).sum()
    t_reject_names_MOM_no_SMB = list(res_SMB[abs(res_SMB["t(m)"]) < 2.].index)

    explanatory_cols_MOM = ["RM_RF", "SMB"]
    res_MOM = run_regression(df, explanatory_cols_MOM)
    m_R2_MOM = np.mean(res_MOM["R2"])
    explanatory_power_loss_MOM= m_R2 - np.mean(res_MOM["R2"])
    R2_5_worst_MOM = res_MOM["R2"].nsmallest(5)
    m_tb_no_MOM = np.mean(abs(res_MOM["t(beta)"]))
    m_ts_no_MOM = np.mean(abs(res_MOM["t(s)"]))
    t_reject_count_RF_RM_no_MOM = (abs(res_MOM["t(beta)"]) < 2.).sum()
    t_reject_names_RF_RM_no_MOM = list(res_MOM[abs(res_MOM["t(beta)"]) < 2.].index)
    t_reject_count_SMB_no_MOM = (abs(res_MOM["t(s)"]) < 2.).sum()
    t_reject_names_SMB_no_MOM = list(res_MOM[abs(res_MOM["t(s)"]) < 2.].index)

    # Create a DataFrame with only portfolios present in the 5 worst R2 tables
    worst_portfolios = set(R2_5_worst.index) | set(R2_5_worst_RM_RF.index) | set(R2_5_worst_SMB.index) | set(R2_5_worst_MOM.index)
    summary_df = pd.DataFrame(0, index=sorted(worst_portfolios), columns=["RM_RF", "SMB", "MOM"])

    for name, worst, result in [
        ("RM_RF", R2_5_worst_RM_RF, res_RM_RF),
        ("SMB", R2_5_worst_SMB, res_SMB),
        ("MOM", R2_5_worst_MOM, res_MOM),
    ]:
        for portfolio in worst.index:
            summary_df.loc[portfolio, name] = result.loc[portfolio, "R2"]

    summary_df["none"] = 0
    for portfolio in R2_5_worst.index:
        summary_df.loc[portfolio, "none"] = res.loc[portfolio, "R2"]

    summary_df = summary_df[["none", "RM_RF", "SMB", "MOM"]]

    def format_value(x):
        if x == 0:
            return 0
        return f"{x:.3f}"

    summary_df = summary_df.applymap(format_value)
    summary_df.loc["m_R2"] = [f"{m_R2:.3f}", f"{m_R2_RM_RF:.3f}", f"{m_R2_SMB:.3f}", f"{m_R2_MOM:.3f}"]
    summary_df.loc["explanatory_loss"] = [
        "0.000",
        f"{explanatory_power_loss:.3f}",
        f"{explanatory_power_loss_SMB:.3f}",
        f"{explanatory_power_loss_MOM:.3f}"
    ]
    summary_df["total"] = summary_df[["none", "RM_RF", "SMB", "MOM"]].apply(lambda row: sum([v != 0 for v in row]), axis=1)
    summary_df.index.name = "without factor"

    # ------------------------------------------------------------------------
    # ADDITION: Summary tables for t_mean, reject_count, and rejected portfolios
    # ------------------------------------------------------------------------
    def make_summary_table(factors, t_means, reject_counts, reject_names):
        max_rejects = max(len(reject_names[f]) for f in factors)
        data = {
            f: [t_means[f], reject_counts[f]] + reject_names[f] + [""] * (max_rejects - len(reject_names[f]))
            for f in factors
        }
        index = ["t_mean", "reject_count"] + [f"rejected_{i+1}" for i in range(max_rejects)]
        return pd.DataFrame(data, index=index)

    # --- All factors ---
    summary_all = make_summary_table(
        ["RM_RF", "SMB", "MOM"],
        {"RM_RF": m_tb, "SMB": m_ts, "MOM": m_tm},
        {"RM_RF": t_reject_count_RF_RM, "SMB": t_reject_count_SMB, "MOM": t_reject_count_MOM},
        {"RM_RF": t_reject_names_RF_RM, "SMB": t_reject_names_SMB, "MOM": t_reject_names_MOM}
    )

    # --- Without Market Rate ---
    summary_no_RM = make_summary_table(
        ["SMB", "MOM"],
        {"SMB": m_ts_no_RM_RF, "MOM": m_tm_no_RM_RF},
        {"SMB": t_reject_count_SMB_no_RM_RF, "MOM": t_reject_count_MOM_no_RM_RF},
        {"SMB": t_reject_names_SMB_no_RM_RF, "MOM": t_reject_names_MOM_no_RM_RF}
    )

    # --- Without SMB ---
    summary_no_SMB = make_summary_table(
        ["RM_RF", "MOM"],
        {"RM_RF": m_tb_no_SMB, "MOM": m_tm_no_SMB},
        {"RM_RF": t_reject_count_RF_RM_no_SMB, "MOM": t_reject_count_MOM_no_SMB},
        {"RM_RF": t_reject_names_RF_RM_no_SMB, "MOM": t_reject_names_MOM_no_SMB}
    )

    # --- Without MOM ---
    summary_no_MOM = make_summary_table(
        ["RM_RF", "SMB"],
        {"RM_RF": m_tb_no_MOM, "SMB": m_ts_no_MOM},
        {"RM_RF": t_reject_count_RF_RM_no_MOM, "SMB": t_reject_count_SMB_no_MOM},
        {"RM_RF": t_reject_names_RF_RM_no_MOM, "SMB": t_reject_names_SMB_no_MOM}
    )

    print("\n=== All Factors ===")
    print(summary_all)
    print("\n=== Without Market Rate (RM_RF) ===")
    print(summary_no_RM)
    print("\n=== Without SMB ===")
    print(summary_no_SMB)
    print("\n=== Without MOM ===")
    print(summary_no_MOM)

    # return main R² summary (tables are printed)
    return summary_df


def data_interpreter_new(df):
    """
    Same as before, but ensures portfolio names render correctly in LaTeX
    (escapes underscores and removes unwanted text parts).
    """

    import numpy as np
    import pandas as pd
    import re

    FACTOR_SHORT = {"RM_RF": "b", "SMB": "s", "MOM": "m"}

    # ==============================================================
    # Helper: Clean and escape portfolio names for LaTeX
    # ==============================================================
    def clean_name(name: str) -> str:
        if not isinstance(name, str):
            return name
        name = re.sub(r"PRIOR", "PRI", name)
        name = re.sub(r"_RF$", "", name)
        # Escape LaTeX-sensitive chars (mainly underscores)
        name = name.replace("_", "\\_")
        return name

    # ==============================================================
    # Helper to run regression for a given set of factors
    # ==============================================================
    def analyze_model(df, cols, reference_R2=None):
        res = run_regression(df, cols)
        m_R2 = float(np.mean(res["R2"]))
        loss = 0.0 if reference_R2 is None else (reference_R2 - m_R2)
        R2_5_worst = res["R2"].nsmallest(5)

        # Collect t-stats and clean names
        t_means, reject_counts, reject_names = {}, {}, {}
        mapping = [("RM_RF", "t(beta)"), ("SMB", "t(s)"), ("MOM", "t(m)")]
        for fac, tcol in mapping:
            if tcol in res.columns:
                tvals = res[tcol].astype(float)
                t_means[fac] = float(np.mean(np.abs(tvals)))
                mask = np.abs(tvals) < 2.0
                reject_counts[fac] = int(mask.sum())
                reject_names[fac] = [clean_name(i) for i in res.index[mask]]
            else:
                t_means[fac] = 0.0
                reject_counts[fac] = 0
                reject_names[fac] = []

        return {
            "res": res,
            "m_R2": m_R2,
            "loss": loss,
            "R2_5_worst": R2_5_worst,
            "t_means": t_means,
            "reject_counts": reject_counts,
            "reject_names": reject_names,
        }

    # ==============================================================
    # Helper to make t-statistics tables
    # ==============================================================
    def make_summary_table(factors, model):
        max_rejects = max((len(model["reject_names"][f]) for f in factors), default=0)
        data = {}
        for f in factors:
            data[f] = [
                model["t_means"].get(f, 0.0),
                model["reject_counts"].get(f, 0),
            ] + model["reject_names"].get(f, []) + [""] * (max_rejects - len(model["reject_names"].get(f, [])))
        idx = ["t_mean", "reject_count"] + [f"rejected_{i+1}" for i in range(max_rejects)]
        return pd.DataFrame(data, index=idx)

    def t_summary_to_latex(title, summary_df):
        """Format one t-table as LaTeX with headers b, s, m."""
        factors = list(summary_df.columns)
        short = [FACTOR_SHORT.get(f, f.lower()) for f in factors]
        latex = f"\\textbf{{{title}}} \\\\[0.5em]\n"
        latex += "\\begin{tabular}{l" + "c" * len(factors) + "}\n"
        latex += "\\toprule\n"
        latex += " & " + " & ".join(short) + " \\\\\n"
        latex += "\\midrule\n"
        latex += "$t$ mean & " + " & ".join(f"{summary_df.loc['t_mean', f]:.2f}" for f in factors) + " \\\\\n"
        latex += "$\\lvert t \\rvert<2$ & " + " & ".join(f"{int(summary_df.loc['reject_count', f])}" for f in factors) + " \\\\\n"
        for i in range(2, len(summary_df)):
            row = [summary_df.iloc[i][f] if summary_df.iloc[i][f] else " " for f in factors]
            latex += "& " + " & ".join(row) + " \\\\\n"
        latex += "\\bottomrule\n\\end{tabular}\n"
        return latex

    def combine_t_summaries_to_latex(summary_all, summary_no_RM, summary_no_SMB, summary_no_MOM):
        """Combine four t-tables into a 2x2 LaTeX layout."""
        def block(left_title, left_tbl, right_title, right_tbl):
            return (
                "\\begin{minipage}[t]{0.47\\textwidth}\n\\centering\n"
                + t_summary_to_latex(left_title, left_tbl)
                + "\\end{minipage}\n\\hfill\n"
                "\\begin{minipage}[t]{0.47\\textwidth}\n\\centering\n"
                + t_summary_to_latex(right_title, right_tbl)
                + "\\end{minipage}\n"
            )
        return (
            "\\begin{table}[H]\n\\centering\n\n"
            + block("All Factors", summary_all, "Without Market Rate (RM)", summary_no_RM)
            + "\n\n\\vspace{1em}\n\n"
            + block("Without SMB", summary_no_SMB, "Without MOM", summary_no_MOM)
            + "\n\n\\end{table}"
        )

    def fmt3(x):
        if isinstance(x, (float, int)):
            return f"{x:.3f}" if x != 0 else 0
        return x

    # ==============================================================
    # Run regressions
    # ==============================================================
    model_all = analyze_model(df, ["RM_RF", "SMB", "MOM"])
    model_no_RM = analyze_model(df, ["SMB", "MOM"], model_all["m_R2"])
    model_no_SMB = analyze_model(df, ["RM_RF", "MOM"], model_all["m_R2"])
    model_no_MOM = analyze_model(df, ["RM_RF", "SMB"], model_all["m_R2"])

    # ==============================================================
    # Build R² summary
    # ==============================================================
    worst = (
        set(model_all["R2_5_worst"].index)
        | set(model_no_RM["R2_5_worst"].index)
        | set(model_no_SMB["R2_5_worst"].index)
        | set(model_no_MOM["R2_5_worst"].index)
    )

    summary_df = pd.DataFrame(0.0, index=sorted([clean_name(w) for w in worst]), columns=["none", "RM-RF", "SMB", "MOM"])

    # Fill R² values for worst portfolios
    for label, model in [("none", model_all), ("RM-RF", model_no_RM), ("SMB", model_no_SMB), ("MOM", model_no_MOM)]:
        for p in model["R2_5_worst"].index:
            summary_df.loc[clean_name(p), label] = float(model["res"].loc[p, "R2"])

    # Add summary rows
    summary_df.loc["mean $R^2$"] = [model_all["m_R2"], model_no_RM["m_R2"], model_no_SMB["m_R2"], model_no_MOM["m_R2"]]
    summary_df.loc["explanatory loss"] = [0.0, model_no_RM["loss"], model_no_SMB["loss"], model_no_MOM["loss"]]

    # Totals and format
    totals = summary_df.iloc[:-2][["none", "RM-RF", "SMB", "MOM"]].apply(lambda r: int(sum(v != 0 for v in r)), axis=1)
    summary_df["total"] = ""
    summary_df.loc[totals.index, "total"] = totals
    summary_df = summary_df.map(fmt3)

    # ==============================================================
    # LaTeX R² table (escaped names)
    # ==============================================================
    latex_r2 = (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{5 Portfolios with lowest $R^2$ for each model and mean $R^2$ across 31 portfolios}\n"
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "\\textbf{Without Factor} & \\textbf{none} & \\textbf{RM-RF} & \\textbf{SMB} & \\textbf{MOM} & \\textbf{total} \\\\\n"
        "\\midrule\n"
    )
    for idx, row in summary_df.iloc[:-2].iterrows():
        latex_r2 += f"{idx} & {row['none']} & {row['RM-RF']} & {row['SMB']} & {row['MOM']} & {row['total']} \\\\\n"
    latex_r2 += "\\midrule\n"
    latex_r2 += f"mean $R^2$ & {summary_df.loc['mean $R^2$','none']} & {summary_df.loc['mean $R^2$','RM-RF']} & {summary_df.loc['mean $R^2$','SMB']} & {summary_df.loc['mean $R^2$','MOM']} &  \\\\\n"
    latex_r2 += f"explanatory loss & {summary_df.loc['explanatory loss','none']} & {summary_df.loc['explanatory loss','RM-RF']} & {summary_df.loc['explanatory loss','SMB']} & {summary_df.loc['explanatory loss','MOM']} &  \\\\\n"
    latex_r2 += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    # ==============================================================
    # t-tables
    # ==============================================================
    summary_all = make_summary_table(["RM_RF", "SMB", "MOM"], model_all)
    summary_no_RM = make_summary_table(["SMB", "MOM"], model_no_RM)
    summary_no_SMB = make_summary_table(["RM_RF", "MOM"], model_no_SMB)
    summary_no_MOM = make_summary_table(["RM_RF", "SMB"], model_no_MOM)
    latex_t_tables = combine_t_summaries_to_latex(summary_all, summary_no_RM, summary_no_SMB, summary_no_MOM)

    return {
        "R2_summary": summary_df,
        "latex_r2_table": latex_r2,
        "latex_t_tables": latex_t_tables,
    }
def single_factor_market_regression(df):
    """
    Runs regression using only RM-RF as explanatory variable.
    Returns:
        - dict with results DataFrame and LaTeX summary table
    """

    import numpy as np
    import pandas as pd
    import re

    # --- Helper: clean and escape names for LaTeX ---
    def clean_name(name: str) -> str:
        if not isinstance(name, str):
            return name
        name = re.sub(r"PRIOR", "PRI", name)
        name = re.sub(r"_RF$", "", name)
        return name.replace("_", "\\_")

    # --- Run regression ---
    results = run_regression(df, ["RM_RF"])

    # Compute metrics
    mean_R2 = float(np.mean(results["R2"]))
    worst_R2 = results["R2"].nsmallest(5)
    t_mean = float(np.mean(np.abs(results["t(beta)"])))
    reject_mask = np.abs(results["t(beta)"]) < 2.0
    reject_count = int(reject_mask.sum())
    reject_names = [clean_name(i) for i in results.index[reject_mask]]

    # Build display DataFrame
    summary_df = pd.DataFrame({
        "R2": results["R2"],
        "t(beta)": results["t(beta)"],
    }).copy()

    # Add info rows
    summary_df.loc["mean $R^2$"] = [mean_R2, np.nan]
    summary_df.loc["mean |t|"] = [np.nan, t_mean]
    summary_df.loc["|t|<2 (count)"] = [np.nan, reject_count]

    # --- Format LaTeX table ---
    latex = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Single-Factor Regression using $RM - RF$}\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "\\textbf{Portfolio} & $R^2$ & $t(\\beta)$ \\\\\n"
        "\\midrule\n"
    )

    # Add 5 worst R2 portfolios
    for p in worst_R2.index:
        latex += f"{clean_name(p)} & {worst_R2.loc[p]:.3f} & {results.loc[p, 't(beta)']:.2f} \\\\\n"

    latex += "\\midrule\n"
    latex += f"mean $R^2$ & {mean_R2:.3f} &  \\\\\n"
    latex += f"$t$ mean &  & {t_mean:.2f} \\\\\n"
    latex += f"$|t|<2$ (count) &  & {reject_count} \\\\\n"

    # If there are rejected portfolios, add them below
    if reject_names:
        latex += "\\midrule\n"
        latex += "Rejected Portfolios & \\multicolumn{2}{l}{"
        latex += ", ".join(reject_names)
        latex += "} \\\\\n"

    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    return {
        "results_df": results,
        "latex_table": latex,
        "mean_R2": mean_R2,
        "worst_R2": worst_R2,
        "t_mean": t_mean,
        "reject_count": reject_count,
        "reject_names": reject_names,
    }
def mean_t_mom_high_prior_latex(df):
    """
    Compute mean t(m) for 'HiPRIOR' and 'PRIOR5' portfolios
    and mean R² across all portfolios, for three regression models:
    (All Factors, Without RM_RF, Without SMB).
    Returns a LaTeX-formatted table.
    """
    models = {
        "All Factors": ["RM_RF", "SMB", "MOM"],
        "Without RM_RF": ["SMB", "MOM"],
        "Without SMB": ["RM_RF", "MOM"],
    }

    results = {}

    for name, cols in models.items():
        # Run regression
        res = run_regression(df, cols)

        # Mean R² across all portfolios
        mean_r2 = res["R2"].mean() if "R2" in res.columns else float('nan')

        # Mean t(m) across HiPRIOR and PRIOR5 portfolios
        if "t(m)" in res.columns:
            mask = res.index.str.contains("HiPRIOR", case=False) | res.index.str.contains("PRIOR5", case=False)
            selected_t = res.loc[mask, "t(m)"].dropna()
            mean_t_m = selected_t.mean() if not selected_t.empty else float('nan')
        else:
            mean_t_m = float('nan')

        results[name] = {"Mean t(m)": mean_t_m, "Mean R2": mean_r2}

        print(f"{name}: mean t(m) = {mean_t_m:.2f}, mean R² = {mean_r2:.3f}")

    # Build DataFrame
    summary_df = pd.DataFrame(results).T
    summary_df.index.name = "Model"

    # Generate LaTeX table
    latex_table = "\\begin{table}[H]\n\\centering\n"
    latex_table += "\\caption{Mean $t(m)$ for High- and Prior5-Portfolios and Mean $R^2$ across Models}\n"
    latex_table += "\\begin{tabular}{lrr}\n\\toprule\n"
    latex_table += "Model & Mean $t(m)$ & Mean $R^2$ \\\\\n\\midrule\n"

    for model, row in summary_df.iterrows():
        mean_t = f"{row['Mean t(m)']:.2f}" if pd.notna(row['Mean t(m)']) else "-"
        mean_r2 = f"{row['Mean R2']:.3f}" if pd.notna(row['Mean R2']) else "-"
        latex_table += f"{model} & {mean_t} & {mean_r2} \\\\\n"

    latex_table += "\\bottomrule\n\\end{tabular}\n\\end{table}"

    return latex_table
def compare_models_high_prior_diff_table(df1, df2, label1="Long-short", label2="Long-only"):
    """
    Computes the difference in mean t(m) for 'HiPRIOR' and 'PRIOR5' portfolios
    and mean R² across models (Long-only minus Long-short).
    Returns a LaTeX-formatted table with only differences (Δ).
    """

    def compute_stats(df, factors):
        res = run_regression(df, factors)
        mean_r2 = res["R2"].mean()
        if "t(m)" in res.columns:
            mask = res.index.str.contains("HiPRIOR", case=False) | res.index.str.contains("PRIOR5", case=False)
            mean_t_m = res.loc[mask, "t(m)"].dropna().mean()
        else:
            mean_t_m = float("nan")
        return mean_t_m, mean_r2

    models = {
        "All Factors": ["RM_RF", "SMB", "MOM"],
        "Without RM_RF": ["SMB", "MOM"],
        "Without SMB": ["RM_RF", "MOM"],
    }

    rows = []
    for model_name, factors in models.items():
        t1, r21 = compute_stats(df1, factors)  # long-short
        t2, r22 = compute_stats(df2, factors)  # long-only
        rows.append({
            "Model": model_name,
            "Δ Mean $t(m)$": t2 - t1 if pd.notna(t1) and pd.notna(t2) else float("nan"),
            "Δ Mean $R^2$": r22 - r21 if pd.notna(r21) and pd.notna(r22) else float("nan"),
        })

    summary = pd.DataFrame(rows).set_index("Model")

    # --- Build LaTeX ---
    latex = "\\begin{table}[H]\n\\centering\n"
    latex += "\\caption{Difference (Long-only $-$ Long-short) in Mean $t(m)$ for High- and Prior5-Portfolios and Mean $R^2$}\n"
    latex += "\\begin{tabular}{lrr}\n\\toprule\n"
    latex += "Model & $\\Delta$ Mean $t(m)$ & $\\Delta$ Mean $R^2$ \\\\\n"
    latex += "\\midrule\n"

    for idx, row in summary.iterrows():
        delta_t = f"{row['Δ Mean $t(m)$']:.2f}" if pd.notna(row['Δ Mean $t(m)$']) else "-"
        delta_r2 = f"{row['Δ Mean $R^2$']:.3f}" if pd.notna(row['Δ Mean $R^2$']) else "-"
        latex += f"{idx} & {delta_t} & {delta_r2} \\\\\n"

    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}"
    return latex





# summary = data_interpreter_new(df_USEUR)

# print(summary["latex_r2_table"])
# print(summary["latex_t_tables"])

# market_reg = single_factor_market_regression(df_USEUR)

# print(market_reg["latex_table"])
m = mean_t_mom_high_prior_latex(df_USEUR)
print(m)

# m_dif = compare_models_high_prior_diff_table(df_USEUR, df_long_USEUR, label1="long-short", label2="Long-only")
# print(m_dif)

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

