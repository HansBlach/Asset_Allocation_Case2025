import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from risk_parity import risk_parity

from markowitz import markowitz_historical, returns_to_value

# Read CSV files:

EU_data = pd.read_csv("csv_files/long_EXPORT EU EUR.csv")

columns_to_add = list(EU_data.columns[2:36])

# Add the risk free rate to the portfolios such that they are no longer excess returns

EU_data[columns_to_add] = EU_data[columns_to_add].add(EU_data['RF'], axis=0)

def result_plot(data, window, n_points):
    data['Date'] = pd.to_datetime(data['Date'].astype(str), format='%Y%m')
    # Make data such that it fits with markowitz functions
    portfolios = list(data.columns[5:36])
    markowitz_data = data[portfolios]
    # Use markowitz function to get markowitz return and value development of tangent strategy
    markowitz_returns = markowitz_historical(markowitz_data, window, n_points = n_points)
    markowitz_value = returns_to_value(markowitz_returns)

    # Use risk parity funciton to get risk parity return and value development
    risk_parity_returns = np.array(risk_parity(EU_data,36,True,True,True)['return'])
    risk_parity_value = returns_to_value(risk_parity_returns)

    # # Use markowitz function to get markowitz return of strategy with mu_target = average return from risk parity
    # markowitz_returns_fixed = markowitz_historical(markowitz_data, window, strategy = "fixed", mu_target= np.mean(risk_parity_returns))
    # markowitz_value_fixed = returns_to_value(markowitz_returns_fixed)

    # Take market returns and make value development
    market_returns = np.array(data['RM_RF'].iloc[window:])
    market_value = returns_to_value(market_returns)

    # Plot the value development of the strategies

    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'][-len(markowitz_value):], markowitz_value, label='Markowitz Tangent Strategy')
    plt.plot(data['Date'][-len(risk_parity_value):], risk_parity_value, label='Risk Parity Strategy')
    # plt.plot(data['Date'][-len(markowitz_value_fixed):], markowitz_value_fixed, label='Markowitz Fixed Strategy')
    plt.plot(data['Date'][-len(market_value):], market_value, label='Market', linestyle='--', color='black')
    plt.xlabel('Time (Months)')
    plt.ylabel('Portfolio Value')
    plt.title('Value Development of Markowitz Strategy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig("result_plot.png")
    plt.show()
    

window = 36

n_points = 20

result_plot(EU_data, window, n_points)