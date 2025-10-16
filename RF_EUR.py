import pandas as pd
import numpy as np

# ---- Reading and manipulating the big file ----

# This was only done for the first time for each application, since it shortens the workflow with the big file a lot

# # Read the file
# df = pd.read_csv("data.csv")

# # Convert to datetime format
# df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'], errors='coerce')

# # Change the September 6th observation to August 30

# df.loc[df['TIME_PERIOD'] == '2004-09-06', 'TIME_PERIOD'] = pd.Timestamp('2004-08-30')

# # Select only relevant columns and rows

# df = df[['DATA_TYPE_FM', 'TIME_PERIOD', 'OBS_VALUE']]

# df = df[df['DATA_TYPE_FM'].isin(['BETA0','BETA1','BETA2','BETA3','TAU1','TAU2'])]


# # Select the relevant period
# start = '2004-08-30'
# start = '2007-07-25'
# end = '2024-12-31'

# df = df[(df['TIME_PERIOD'] >= start) & (df['TIME_PERIOD'] <= end)]

# # Select only the last observation in each month
# df = (
#     df.sort_values('TIME_PERIOD')
#       .groupby([df['DATA_TYPE_FM'], df['TIME_PERIOD'].dt.to_period('M')])
#       .tail(1)
#       .reset_index(drop=True)
# )

# df.to_csv("yield_curve_data.csv")

df = pd.read_csv("yield_curve_data.csv")


# ---- Risk free rate ----

# Save each of the parameters as an np.array for calculations.

df_pivot = df.pivot(index='TIME_PERIOD', columns='DATA_TYPE_FM', values='OBS_VALUE').sort_index()

BETA_0 = np.array(df_pivot['BETA0'].values)/100
BETA_1 = np.array(df_pivot['BETA1'].values)/100
BETA_2 = np.array(df_pivot['BETA2'].values)/100
BETA_3 = np.array(df_pivot['BETA3'].values)/100
TAU_1  = np.array(df_pivot['TAU1'].values)/100
TAU_2  = np.array(df_pivot['TAU2'].values)/100

TTM = 1/12

# Calculated the zero coupon rate using Svensson yield curve formula

z = BETA_0 + BETA_1*(1-np.exp(-TTM/TAU_1))/(TTM/TAU_1) + BETA_2*((1-np.exp(-TTM/TAU_1))/(TTM/TAU_1)-np.exp(-TTM/TAU_1)) + BETA_3*((1-np.exp(-TTM/TAU_2))/(TTM/TAU_2)-np.exp(-TTM/TAU_2))

# Using continuous compounding fo discount factors

RF = (np.exp(z/12)-1)*100

discount_factor = np.exp(-TTM*z)

rf_df = pd.DataFrame({
    'TIME_PERIOD': df_pivot.index,
    'RISK_FREE_RATE': RF
})

rf_df.to_csv('risk_free_rates_eur.csv', index=False)

# ---- ZCB prices ----


# Periods untill maturity
N = 120
# Time to maturity
TTM = 10

start = 0

def zcb_price_generator(TTM, N, start, data):
    # Extract parameters for svensson
    # with start set to 0 the first date is august 2007

    df_pivot = data.pivot(index='TIME_PERIOD', columns='DATA_TYPE_FM', values='OBS_VALUE').sort_index()

    BETA_0 = np.array(df_pivot['BETA0'].values)/100
    BETA_1 = np.array(df_pivot['BETA1'].values)/100
    BETA_2 = np.array(df_pivot['BETA2'].values)/100
    BETA_3 = np.array(df_pivot['BETA3'].values)/100
    TAU_1  = np.array(df_pivot['TAU1'].values)/100
    TAU_2  = np.array(df_pivot['TAU2'].values)/100


    if len(BETA_0) < N + start:
        print("Start should be 10 or more years from final observation")
        return ""

    # Calculate zcb prices
    zcb_prices = np.zeros(N)
    period_length = TTM/N

    for i in range(N):
        TTM = TTM - period_length

        z = BETA_0[i + start] + BETA_1[i + start]*(1-np.exp(-TTM/TAU_1[i + start]))/(TTM/TAU_1[i + start]) 
        + BETA_2[i + start]*((1-np.exp(-TTM/TAU_1[i + start]))/(TTM/TAU_1[i + start])-np.exp(-TTM/TAU_1[i + start])) 
        + BETA_3[i + start]*((1-np.exp(-TTM/TAU_2[i + start]))/(TTM/TAU_2[i + start])-np.exp(-TTM/TAU_2[i + start]))

        zcb_prices[i] = np.exp(-TTM*z)

    return zcb_prices



