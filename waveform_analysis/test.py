import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Your time and signal data
time = np.linspace(0, 10, 1000)  # replace with your actual time points
signal = np.random.normal(0, 1, 1000)  # replace with your actual signal data

# The model function
def model(t, A1, tau1, A2, tau2):
    return A1 * np.exp(-t/tau1) + A2 * np.exp(-t/tau2)

# Initial guess for the parameters
initial_guess = [1, 1, 1, 1]

# Fit the model to the data
popt, pcov = curve_fit(model, time, signal, p0=initial_guess)

# Extracting the fitted parameters
A1_fitted, tau1_fitted, A2_fitted, tau2_fitted = popt

print(f"Fitted Parameters: A1={A1_fitted}, tau1={tau1_fitted}, A2={A2_fitted}, tau2={tau2_fitted}")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(time, signal, label='Original Signal')
plt.plot(time, model(time, *popt), label='Fitted Model', color='red')
plt.legend()
plt.show()

