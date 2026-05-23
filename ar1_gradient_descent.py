import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# -----------------------------
# LOAD AND PREPARE THE DATA
# -----------------------------
data = pd.read_csv("electricity.csv", thousands=",")

data["month"] = pd.to_datetime(data["month"], dayfirst=True)
data = data.sort_values("month").reset_index(drop=True)

y = data["kwh"].values

# Create lagged data
y_t = y[1:]      # current values
y_lag = y[:-1]   # previous values

# -----------------------------
# GRADIENT DESCENT SETTINGS
# -----------------------------
c = 0.0
phi = 0.0

learning_rate = 1e-16
max_iter = 5000
tolerance = 1e-3

loss_history = []
param_history = []

n = len(y_t)

# -----------------------------
# GRADIENT DESCENT LOOP
# -----------------------------
for i in range(max_iter):
    # Predictions
    y_pred = c + phi * y_lag

    # Errors
    error = y_t - y_pred

    # Loss function: Mean Squared Error
    loss = np.mean(error**2)
    loss_history.append(loss)
    param_history.append((c, phi))

    # Gradients
    dc = (-2 / n) * np.sum(error)
    dphi = (-2 / n) * np.sum(error * y_lag)

    # Parameter update
    new_c = c - learning_rate * dc
    new_phi = phi - learning_rate * dphi

    # Check convergence
    if np.sqrt((new_c - c)**2 + (new_phi - phi)**2) < tolerance:
        c, phi = new_c, new_phi
        print(f"Converged after {i+1} iterations")
        break

    c, phi = new_c, new_phi

else:
    print("Reached maximum iterations without full convergence")

# -----------------------------
# RESULTS
# -----------------------------
print(f"Estimated c: {c}")
print(f"Estimated phi: {phi}")
print(f"Final loss: {loss_history[-1]}")

# Final predictions
final_pred = c + phi * y_lag

# -----------------------------
# PLOT 1: ACTUAL VS PREDICTED
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(data["month"][1:], y_t, marker="o", label="Actual")
plt.plot(data["month"][1:], final_pred, marker="o", label="Predicted")

plt.title("AR(1) Model Fit using Gradient Descent")
plt.xlabel("Month")
plt.ylabel("Electricity (kWh)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.tight_layout()
plt.show()

# -----------------------------
# PLOT 2: LOSS CONVERGENCE
# -----------------------------
plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.title("Gradient Descent Convergence")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_t, final_pred)
rmse = np.sqrt(mean_squared_error(y_t, final_pred))

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")