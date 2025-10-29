import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from risk_parity import risk_parity

from markowitz import markowitz_historical, returns_to_value

# Read CSV files:

EU_data = pd.read_csv("csv_files/EXPORT EU EUR.csv")
US_data = pd.read_csv("csv_files/EXPORT US EUR.csv")

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

def result_plot(EU_data, US_data, window, n_points, eu_factors, us_factors, markowitz_short = False, volatility_targets = "", name = "", data = "both", baseline_market = "EU", has_MOM1=True,has_MOM2=True,has_SMB1=True,has_SMB2=True,has_RM_RF1=True,has_RM_RF2=True):
    if data == "both":
        include_market2 = True
    if data == "EU" or data == "US":
        include_market2 = False

    # Convert date columns - handle the YYYYMM integer format properly
    EU_data['Date'] = pd.to_datetime(EU_data['Date'], format='%Y%m')
    US_data['Date'] = pd.to_datetime(US_data['Date'], format='%Y%m')
    
    # Use markowitz function to get markowitz return and value development of tangent strategy
    markowitz_returns_tan = markowitz_historical(EU_data, US_data, eu_factors, us_factors, window, strategy = "tangent", n_points = n_points, allow_short=markowitz_short)
    markowitz_value_tan = returns_to_value(markowitz_returns_tan)

    if name == "Markowitz strategy comparison":     
        # Use markowitz function to get markowitz return and value development of tangent strategy
        markowitz_returns_tan = markowitz_historical(EU_data, US_data, eu_factors, us_factors, window, strategy = "tangent", n_points = n_points, allow_short=markowitz_short)
        markowitz_value_tan = returns_to_value(markowitz_returns_tan)   
        # Use markowitz function to get markowitz return and value development of min_var strategy
        markowitz_returns_minvar = markowitz_historical(EU_data, US_data, eu_factors, us_factors, window, strategy = "min_variance", n_points = n_points, allow_short=markowitz_short)
        markowitz_value_minvar = returns_to_value(markowitz_returns_minvar)

        # Use markowitz function to get markowitz return and value development of max_return strategy
        markowitz_returns_max = markowitz_historical(EU_data, US_data, eu_factors, us_factors, window, strategy = "max_return", n_points = n_points, allow_short=markowitz_short)
        markowitz_value_max = returns_to_value(markowitz_returns_max)

    if name == "Markowitz vs Risk Parity":
        # Use markowitz function to get markowitz return and value development of tangent strategy
        markowitz_returns_tan = markowitz_historical(EU_data, US_data, eu_factors, us_factors, window, strategy = "tangent", n_points = n_points, allow_short=markowitz_short)
        markowitz_value_tan = returns_to_value(markowitz_returns_tan)           
        # Use risk parity function to get risk parity return and value development
        risk_parity_returns = np.array(risk_parity(EU_data,US_data,"EU","US",include_market2 , 36, has_MOM1, has_SMB1, has_RM_RF1, has_MOM2, has_SMB2, has_RM_RF2, use_covariance= False)['return'])
        risk_parity_value = returns_to_value(risk_parity_returns)

    if name == "Markowitz vs Risk Parity strategies (with shorting)":
        # Use markowitz function to get markowitz return and value development of tangent strategy
        markowitz_returns_tan = markowitz_historical(EU_data, US_data, eu_factors, us_factors, window, strategy = "tangent", n_points = n_points, allow_short=markowitz_short)
        markowitz_value_tan = returns_to_value(markowitz_returns_tan)
        risk_parity_returns = []
        risk_parity_value = []   
        for vol_target in volatility_targets:
            rp_return = np.array(risk_parity(EU_data,US_data,"EU","US",include_market2 , 36, has_MOM1, has_SMB1, has_RM_RF1, has_MOM2, has_SMB2, has_RM_RF2, use_covariance= False, allow_short= markowitz_short, target_std = vol_target)['return'])
            risk_parity_returns.append(rp_return)
            risk_parity_value.append(returns_to_value(rp_return))

    # Take market returns and make value development
    if baseline_market == "EU":
        market_returns = np.array(EU_data['RM_RF'].iloc[window:])
    if baseline_market == "US":
        market_returns = np.array(US_data['RM_RF'].iloc[window:])
    
    market_value = returns_to_value(market_returns)

    # Plot the value development of the strategies

    plt.figure(figsize=(10, 6))
    if name == "Markowitz strategy comparison":
        plt.plot(EU_data['Date'][-len(markowitz_value_tan):], markowitz_value_tan, label='Tangent Strategy')
        plt.plot(EU_data['Date'][-len(markowitz_value_minvar):], markowitz_value_minvar, label='Min Variance Strategy')
        plt.plot(EU_data['Date'][-len(markowitz_value_max):], markowitz_value_max, label='Max Return Strategy')

    if name == "Markowitz vs Risk Parity":
        plt.plot(EU_data['Date'][-len(markowitz_value_tan):], markowitz_value_tan, label='Tangent Strategy')
        plt.plot(EU_data['Date'][-len(risk_parity_value):], risk_parity_value, label='Risk Parity Strategy')
    
    if name == "Markowitz vs Risk Parity strategies (with shorting)":
        plt.plot(EU_data['Date'][-len(markowitz_value_tan):], markowitz_value_tan, label='Tangent Strategy')
        for i, vol_target in enumerate(volatility_targets):
            plt.plot(EU_data['Date'][-len(risk_parity_value[i]):], risk_parity_value[i], label=f'Risk Parity Strategy (Vol Target: {vol_target}%)')

    plt.plot(EU_data['Date'][-len(market_value):], market_value, label='Market', linestyle='--', color='black')
    plt.xlabel('Time (Months)')
    plt.ylabel('Portfolio Value')
    plt.title(f'Value Development of Strategies: {name}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(f"result_plot_{name}.png", dpi = 300)
    #plt.show()
    if name == "Markowitz vs Risk Parity":
        return markowitz_returns_tan, risk_parity_returns, market_returns
    if name == "Markowitz strategy comparison":
       return markowitz_returns_tan, markowitz_returns_max, markowitz_returns_minvar, market_returns
    if name == "Markowitz vs Risk Parity strategies (with shorting)":
         return markowitz_returns_tan, risk_parity_returns, market_returns



EU_data = pd.read_csv("csv_files/EXPORT EU EUR.csv")
US_data = pd.read_csv("csv_files/EXPORT US EUR.csv")

def annualizer(mean_return, type, periods_per_year = 12):
    if type == "return":
        return (1 + mean_return/100) ** periods_per_year - 1
    if type == "volatility":
        return mean_return/100 * np.sqrt(periods_per_year)



# ---- Run the code ---- 
long_only = False
markowitz_short = True

name = "Markowitz vs Risk Parity strategies (with shorting)"

volatility_targets = [1.5, 3, 5]

window = 36

n_points = 50

if long_only:
    EU_data = pd.read_csv("csv_files/long_EXPORT EU EUR.csv")
    US_data = pd.read_csv("csv_files/long_EXPORT US EUR.csv")

has_MOM1, has_SMB1, has_RM_RF1 = True, True, True
has_MOM2, has_SMB2, has_RM_RF2 = True, True, True

eu_factors, us_factors = ["MOM", "SMB", "RM_RF"], ["RM_RF", "MOM", "SMB"]

markowitz_returns_tan, risk_parity_returns, market_returns = result_plot(name = name, volatility_targets= volatility_targets, EU_data = EU_data, US_data = US_data, eu_factors = eu_factors, us_factors = us_factors, n_points = n_points, window = window, markowitz_short= markowitz_short, data = "both", baseline_market= "EU", has_SMB1=has_SMB1, has_SMB2=has_SMB2, has_MOM1=has_MOM1, has_MOM2=has_MOM2, has_RM_RF1=has_RM_RF1, has_RM_RF2=has_RM_RF2)

#markowitz_returns_tan, markowitz_returns_max, markowitz_returns_minvar, market_returns = result_plot(name = name, volatility_targets= volatility_targets, EU_data = EU_data, US_data = US_data, eu_factors = eu_factors, us_factors = us_factors, n_points = n_points, window = window, markowitz_short= markowitz_short, data = "both", baseline_market= "EU", has_SMB1=has_SMB1, has_SMB2=has_SMB2, has_MOM1=has_MOM1, has_MOM2=has_MOM2, has_RM_RF1=has_RM_RF1, has_RM_RF2=has_RM_RF2) 



# --- Summary statistics ---

if name == "Markowitz vs Risk Parity":
    strategies = {
        "Tangent": markowitz_returns_tan,
        "Risk Parity": risk_parity_returns,
        "Market": market_returns
    }

if name == "Markowitz strategy comparison":
    strategies = {
        "Tangent": markowitz_returns_tan,
        "Max Return": markowitz_returns_max,
        "Min Variance": markowitz_returns_minvar,
        "Market": market_returns
    }

if name == "Markowitz vs Risk Parity strategies (with shorting)":
    strategies = {
        "Tangent": markowitz_returns_tan,
    }
    for i, vol_target in enumerate(volatility_targets):
        strategies[f'Risk Parity (Vol Target: {vol_target}%)'] = risk_parity_returns[i]
    strategies["Market"] = market_returns


data = []

n = len(EU_data["MOM"]) - window

for name, returns in strategies.items():
    mean = np.mean(returns)
    std = np.std(returns)
    sharpe = mean / std 
    t_stat = sharpe * np.sqrt(n)
    data.append([mean, std, sharpe, t_stat])

summary = pd.DataFrame(data, columns=["Mean", "Std", "Sharpe", "t-stat"], index=strategies.keys())
print(summary)