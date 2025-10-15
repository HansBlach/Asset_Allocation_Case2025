import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from risk_parity import risk_parity

from markowitz import markowitz_historical, returns_to_value

# Read CSV files:

EU_data = pd.read_csv("csv_files/long_EXPORT EU EUR.csv")
US_data = pd.read_csv("csv_files/long_EXPORT US EUR.csv")

print(EU_data['Date'])

def data_prep(EU_data, US_data):
    portfolios = list(EU_data.columns[5:36])

    EU = EU_data[portfolios]

    EU.columns = ["EU_" + p for p in portfolios]

    US = US_data[portfolios]

    US.columns = ["US_" + p for p in portfolios]

    # Make a combined datafrme with all assets.

    return pd.concat([EU,US],axis = 1)

columns_to_add = list(EU_data.columns[2:36])

# Add the risk free rate to the portfolios such that they are no longer excess returns

EU_data[columns_to_add] = EU_data[columns_to_add].add(EU_data['RF'], axis=0)
US_data[columns_to_add] = US_data[columns_to_add].add(US_data['RF'], axis=0)

def result_plot(EU_data, US_data, window, n_points, name = "", data = "both", baseline_market = "EU", has_MOM1=True,has_MOM2=True,has_SMB1=True,has_SMB2=True,has_RM_RF1=True,has_RM_RF2=True):
    if data == "both":
        include_market2 = True
    if data == "EU" or data == "US":
        include_market2 = False

    # Convert date columns - handle the YYYYMM integer format properly
    EU_data['Date'] = pd.to_datetime(EU_data['Date'], format='%Y%m')
    US_data['Date'] = pd.to_datetime(US_data['Date'], format='%Y%m')

    # Make data such that it fits with markowitz functions
    if data == "both":
        markowitz_data = data_prep(EU_data, US_data)
    if data == "EU":
        markowitz_data = EU_data[list(EU_data.columns[5:36])]
    if data == "US":
        markowitz_data = US_data[list(US_data.columns[5:36])]
    
    # Use markowitz function to get markowitz return and value development of tangent strategy
    markowitz_returns = markowitz_historical(markowitz_data, window, n_points = n_points)
    markowitz_value = returns_to_value(markowitz_returns)

    # Use risk parity funciton to get risk parity return and value development
    risk_parity_returns = np.array(risk_parity(EU_data,US_data,"EU","US",include_market2 , 36, has_MOM1, has_SMB1, has_RM_RF1, has_MOM2, has_SMB2, has_RM_RF2)['return'])
    risk_parity_value = returns_to_value(risk_parity_returns)

    # # Use markowitz function to get markowitz return of strategy with mu_target = average return from risk parity
    # markowitz_returns_fixed = markowitz_historical(markowitz_data, window, strategy = "fixed", mu_target= np.mean(risk_parity_returns))
    # markowitz_value_fixed = returns_to_value(markowitz_returns_fixed)

    # Take market returns and make value development
    if baseline_market == "EU":
        market_returns = np.array(EU_data['RM_RF'].iloc[window:])
    if baseline_market == "US":
        market_returns = np.array(US_data['RM_RF'].iloc[window:])
    
    market_value = returns_to_value(market_returns)

    # Plot the value development of the strategies

    plt.figure(figsize=(10, 6))
    plt.plot(EU_data['Date'][-len(markowitz_value):], markowitz_value, label='Markowitz Tangent Strategy')
    plt.plot(EU_data['Date'][-len(risk_parity_value):], risk_parity_value, label='Risk Parity Strategy')
    # plt.plot(data['Date'][-len(markowitz_value_fixed):], markowitz_value_fixed, label='Markowitz Fixed Strategy')
    plt.plot(EU_data['Date'][-len(market_value):], market_value, label='Market', linestyle='--', color='black')
    plt.xlabel('Time (Months)')
    plt.ylabel('Portfolio Value')
    plt.title(f'Value Development of Strategies: {name}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(f"result_plot_{name}.png")
    #plt.show()
    

window = 36

n_points = 15

result_plot(EU_data, US_data, window, n_points, data = "both", baseline_market= "EU", name = "Both Markets")

result_plot(EU_data, US_data, window, n_points, data = "EU", baseline_market= "EU", name = "EU Only")

# result_plot(EU_data, US_data, window, n_points, data = "both", baseline_market= "EU", has_SMB1=False, has_SMB2= False, name = "noSMB")

# result_plot(EU_data, US_data, window, n_points, data = "both", baseline_market= "EU", has_SMB1=False, has_SMB2= False, has_MOM2= False, name = "noSMB_noMOMUS")
