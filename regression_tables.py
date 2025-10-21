import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from Regression import run_regression

# Load the data from CSV files
df_EU = pd.read_csv("csv_files/EXPORT EU EUR.csv")
df_USEUR = pd.read_csv("csv_files/EXPORT US EUR.csv")
df_US = pd.read_csv("csv_files/EXPORT US USD.csv")

df_long_EU = pd.read_csv("csv_files/long_EXPORT EU EUR.csv")
df_long_USEUR = pd.read_csv("csv_files/long_EXPORT US EUR.csv")
df_long_US = pd.read_csv("csv_files/long_EXPORT US USD.csv")

# Define the target and explanatory variables
explanatory_cols = ["RM_RF", "SMB", "MOM"]

# print(f"full: {run_regression(df_US, explanatory_cols)['R2'].mean()}")

# explanatory_cols = ["RM_RF", "MOM"]

# print(f"drop smb: {run_regression(df_US, explanatory_cols)['R2'].mean()}")

# explanatory_cols = ["RM_RF", "SMB"]

# print(f"drop mom: {run_regression(df_US, explanatory_cols)['R2'].mean()}")

# print(df_EU['MOM'].mean())

# print(df_EU['MOM'].std())


def table_generator(df):
    explanatory_cols = [["RM_RF"], ["RM_RF", "SMB", "MOM"], ["RM_RF", "MOM"], ["RM_RF", "SMB"], ["SMB", "MOM"]]

    names = ["CAPM", "Full factor", "-SMB", "-MOM", "-MKT"]

    rows = []
    for exp_col in explanatory_cols:
        res = run_regression(df, exp_col)

        mean_R2 = res['R2'].mean()
        mean_a = res['a'].mean()

        perc_a = (abs(res["t(a)"]) < 2.).sum()/25
        perc_beta = (abs(res["t(beta)"]) < 2.).sum()/25
        perc_m = (abs(res["t(m)"]) < 2.).sum()/25
        perc_s = (abs(res["t(s)"]) < 2.).sum()/25

        rows.append({
            "mean R^2": mean_R2,
            "mean alpha": mean_a,
            "t(a)>2": perc_a,
        })

    table = pd.DataFrame(rows, index=names)

    return table.T

print("EU")
print(table_generator(df_long_EU))
print("USEUR")
print(table_generator(df_long_USEUR))
print("US")
print(table_generator(df_long_US))

def factor_mean_table(df):
    cols = ["RM_RF", "MOM", "SMB"]
    n = len(df["MOM"])

    rows = []
    for col in cols:
        series = np.array(df[col])
        mean = np.mean(series)
        std = np.std(series)
        sharpe = mean/std
        t_stat = mean*np.sqrt(n)/std
        rows.append({
            "Monthly Excess Return": mean,
            "Standard deviation": std,
            "Sharpe": sharpe,
            "t-stat for mean = 0": t_stat
        })
    table = pd.DataFrame(rows, index= ["MKT", "MOM", "SMB"])
    return table

# print("EU")
# print(factor_mean_table(df_EU))
# print("US EUR")
# print(factor_mean_table(df_USEUR))
# print("----------------")
print("EU long")
print(factor_mean_table(df_long_EU))
print("US EUR")
print(factor_mean_table(df_long_USEUR))

def covariance_matrix(EU, US, EU_factors, US_factors):
    EU_selected = EU[EU_factors]
    US_selected = US[US_factors]
    factor_universe = pd.concat([EU_selected, US_selected], axis = 1)
    return factor_universe.cov()

print("Covariance matrix of factor universe")
print(covariance_matrix(df_long_EU, df_long_US, ["MOM"], ["MOM", "SMB"]))




