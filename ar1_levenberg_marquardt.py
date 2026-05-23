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
y_t = y[1:]
y_lag = y[:-1]
n = len(y_t)

# -----------------------------
# INITIAL PARAMETERS
# -----------------------------
c = 0.0
phi = 0.5

max_iter = 100
tolerance = 1e-6
lam = 1e6
lam_increase = 10
lam_decrease = 0.1

loss_history = []
lambda_history = []
param_history = []

# -----------------------------
# HELPER: compute loss
# -----------------------------
def compute_loss(c, phi):
    pred = c + phi * y_lag
    residual = y_t - pred
    return np.mean(residual**2)

# initial loss
current_loss = compute_loss(c, phi)

# -----------------------------
# LEVENBERG-MARQUARDT LOOP
# -----------------------------
for i in range(max_iter):
    # residuals
    r = y_t - (c + phi * y_lag)

    # Jacobian
    J = np.column_stack([
        -np.ones(n),
        -y_lag
    ])

    JTJ = J.T @ J
    JTr = J.T @ r

    # damped system
    A = JTJ + lam * np.eye(2)

    try:
        delta = np.linalg.solve(A, JTr)
    except np.linalg.LinAlgError:
        print("Matrix solve failed.")
        break

    # candidate update
    new_c = c - delta[0]
    new_phi = phi - delta[1]

    new_loss = compute_loss(new_c, new_phi)

    loss_history.append(current_loss)
    lambda_history.append(lam)
    param_history.append((c, phi))

    # accept or reject step
    if new_loss < current_loss:
        step_size = np.sqrt((new_c - c)**2 + (new_phi - phi)**2)

        # update parameters
        c, phi = new_c, new_phi

        # check relative improvement in loss
        rel_improvement = abs(new_loss - current_loss) / current_loss
        current_loss = new_loss

        # reduce lambda after successful step
        lam *= lam_decrease

        # stopping conditions
        if step_size < tolerance or rel_improvement < 1e-10:
            print(f"Converged after {i+1} iterations")
            break
    else:
        # reject step and increase damping
        lam *= lam_increase

        if lam > 1e20:
            print("Lambda became too large; stopping.")
            break

# -----------------------------
# FINAL RESULTS
# -----------------------------
final_pred = c + phi * y_lag
final_loss = np.mean((y_t - final_pred)**2)
mae = mean_absolute_error(y_t, final_pred)
rmse = np.sqrt(mean_squared_error(y_t, final_pred))

print(f"Estimated c: {c}")
print(f"Estimated phi: {phi}")
print(f"Final loss: {final_loss}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"Final lambda: {lam}")

# -----------------------------
# PLOT 1: ACTUAL VS PREDICTED
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(data["month"][1:], y_t, marker="o", label="Actual")
plt.plot(data["month"][1:], final_pred, marker="o", label="Predicted")

plt.title("AR(1) Model Fit using Levenberg-Marquardt")
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
plt.plot(loss_history, marker="o")
plt.title("Levenberg-Marquardt Convergence")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# PLOT 3: LAMBDA EVOLUTION
# -----------------------------
plt.figure(figsize=(8, 4))
plt.plot(lambda_history, marker="o")
plt.title("Levenberg-Marquardt Damping Parameter")
plt.xlabel("Iteration")
plt.ylabel("Lambda")
plt.yscale("log")
plt.grid(True)
plt.tight_layout()
plt.show()