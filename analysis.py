import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load dataset
data = pd.read_csv("electricity.csv", thousands=",")

# convert month column
data["month"] = pd.to_datetime(data["month"], dayfirst=True)

# sort by time
data = data.sort_values("month")

print(data.head())
print("\nTotal observations:", len(data))

# plot
plt.figure(figsize=(10,5))
plt.plot(data["month"], data["kwh"], marker="o")

plt.title("Monthly Electricity Supply")
plt.xlabel("Month")
plt.ylabel("Electricity (kWh)")

plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()

#test for stationarity
from statsmodels.tsa.stattools import adfuller

result = adfuller(data["kwh"])

print("ADF Statistic:", result[0])
print("p-value:", result[1])

for key, value in result[4].items():
    print(f"Critical Value ({key}): {value}")


#inspecting autocorrelation function and partial autocorrelation function
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(10,4))
plot_acf(data["kwh"], lags=12)
plt.show()

plt.figure(figsize=(10,4))
plot_pacf(data["kwh"], lags=12)
plt.show()