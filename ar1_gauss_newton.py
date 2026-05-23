import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# LOAD AND PREPARE DATA
# -----------------------------
data = pd.read_csv("electricity.csv", thousands=",")
data["month"] = pd.to_datetime(data["month"], dayfirst=True)
data = data.sort_values("month").reset_index(drop=True)

y = data["kwh"].values
y_t = y[1:]      # current values
y_lag = y[:-1]   # lagged values
n = len(y_t)

# -----------------------------
# INITIAL PARAMETERS
# -----------------------------
c = 0.0
phi = 0.5

max_iter = 100
tolerance = 1e-6

loss_history = []
param_history = []

# -----------------------------
# GAUSS-NEWTON LOOP
# -----------------------------
for i in range(max_iter):
    # Residuals: r = y_t - (c + phi*y_lag)
    r = y_t - (c + phi * y_lag)

    # Jacobian J of residuals wrt parameters [c, phi]
    # dr/dc = -1
    # dr/dphi = -y_lag
    J = np.column_stack([
        -np.ones(n),
        -y_lag
    ])

    # Gauss-Newton step:
    # delta = (J^T J)^(-1) J^T r
    JTJ = J.T @ J
    JTr = J.T @ r

    try:
        delta = np.linalg.solve(JTJ, JTr)
    except np.linalg.LinAlgError:
        print("Matrix inversion failed. J^T J may be singular.")
        break

    # Parameter update
    new_c = c - delta[0]
    new_phi = phi - delta[1]

    # Track loss
    loss = np.mean(r**2)
    loss_history.append(loss)
    param_history.append((c, phi))

    # Check convergence
    step_size = np.sqrt((new_c - c)**2 + (new_phi - phi)**2)
    if step_size < tolerance:
        c, phi = new_c, new_phi
        print(f"Converged after {i+1} iterations")
        break

    c, phi = new_c, new_phi

else:
    print("Reached maximum iterations without full convergence")

# -----------------------------
# FINAL RESULTS
# -----------------------------
final_pred = c + phi * y_lag
mae = mean_absolute_error(y_t, final_pred)
rmse = np.sqrt(mean_squared_error(y_t, final_pred))
final_loss = np.mean((y_t - final_pred)**2)

print(f"Estimated c: {c}")
print(f"Estimated phi: {phi}")
print(f"Final loss: {final_loss}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# -----------------------------
# PLOT 1: ACTUAL VS PREDICTED
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(data["month"][1:], y_t, marker="o", label="Actual")
plt.plot(data["month"][1:], final_pred, marker="o", label="Predicted")

plt.title("AR(1) Model Fit using Gauss-Newton")
plt.xlabel("Month")
plt.ylabel("Electricity (kWh)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.tight_layout()
plt.show()

# -----------------------------
# PLOT 2: CONVERGENCE
# -----------------------------
plt.figure(figsize=(8, 4))
plt.plot(loss_history, marker="o")
plt.title("Gauss-Newton Convergence")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()