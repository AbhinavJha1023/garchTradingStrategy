import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import matplotlib.pyplot as plt

# Load the data
eurusd = pd.read_csv("C:/Users/ABHINAV/Desktop/TimeSeries/TradingStrategy/SP500.csv")
eurusd['Date'] = pd.to_datetime(eurusd['Date'], format='%d/%m/%Y')
eurusd.set_index('Date', inplace=True)

# Calculate log returns using the closing price
eurusd['Log_Returns'] = np.log(eurusd['C']).diff()

# Set parameters for the rolling window
window_length = 100
returns = eurusd['Log_Returns'].dropna().values  # drop NA values
forecasts_length = len(returns) - window_length

# Initialize arrays to store forecasts and directions
forecasts = np.zeros(forecasts_length)
directions = np.zeros(forecasts_length)

# Loop through each window to fit the ARIMA-GARCH model
for i in range(forecasts_length):
    roll_returns = returns[i:i+window_length]
    
    # Find optimal ARIMA model (p,q < 4) based on AIC
    best_aic = np.inf
    best_order = None
    best_model = None
    
    for p in range(1, 5):
        for q in range(1, 5):
            try:
                model = ARIMA(roll_returns, order=(p, 0, q))
                fitted_model = model.fit()
                aic = fitted_model.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, 0, q)
                    best_model = fitted_model
            except:
                continue
    
    # Fit GARCH(1,1) model if best ARIMA order was found
    if best_model is not None:
        try:
            # Specify and fit the GARCH model with best ARIMA mean model
            garch_model = arch_model(best_model.resid, vol='Garch', p=1, q=1, dist='skewt')
            garch_fit = garch_model.fit(disp='off')
            # Forecast next day’s return
            forecast = garch_fit.forecast(horizon=1)
            next_day_return = forecast.mean.values[-1, 0]
            forecasts[i] = next_day_return
            directions[i] = 1 if next_day_return > 0 else -1
        except:
            forecasts[i] = 0  # If model fails to converge, set forecast to 0
    else:
        forecasts[i] = 0

# Create cumulative returns based on forecasted directions
strategy_returns = directions * returns[window_length:]
cumulative_strategy = np.cumsum(strategy_returns)
long_term_returns = np.cumsum(returns[window_length:])

# Plot the results
dates = eurusd.index[window_length:len(returns)]
plt.figure(figsize=(12, 8))
plt.plot(dates, cumulative_strategy, label='ARIMA&GARCH Strategy', color='green')
plt.plot(dates, long_term_returns, label='Long Term Investing', color='red')
plt.title('Cumulative Returns')
plt.xlabel('Time')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()
