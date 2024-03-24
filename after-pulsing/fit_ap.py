import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from time_data import time_data


def compute_integral(t, tau_recharge, tau_decay, max_charge, A, beta, omega, phi, window_size=45):
    # This function computes the integrated value over 45 time points
    integrated_values = []
    for i in range(len(t)):
        sum_val = 0
        for j in range(window_size):
            idx = i - j
            if idx < 0:
                break
            charge = max_charge * (1 - np.exp(-t[idx] / tau_recharge)) * np.exp(-(t[idx]) / tau_decay)
            oscillation = A * np.exp(-beta * t[idx]) * np.sin(omega * t[idx] + phi)
            sum_val += charge + oscillation
        integrated_values.append(sum_val / min(i + 1, window_size))
    return np.array(integrated_values)

def compute_integral_old(t, tau_recharge, tau_decay, max_charge, window_size=45):
    # This function computes the integrated value over 45 time points
    integrated_values = []
    for i in range(len(t)):
        sum_val = 0
        for j in range(window_size):
            idx = i - j
            if idx < 0:
                break
            sum_val += max_charge * (1 - np.exp(-(t[idx])/ tau_recharge)) * np.exp(-(t[idx]) / tau_decay)
        integrated_values.append(sum_val / min(i + 1, window_size))
    return np.array(integrated_values)

def model_function(t, tau_recharge, tau_decay, max_charge, A, beta, omega, phi):
    return compute_integral(t, tau_recharge, tau_decay, max_charge, A, beta, omega, phi)

def model_function_old(t, tau_recharge, tau_decay, max_charge):
    return compute_integral(t, tau_recharge, tau_decay, max_charge)

#def model_function(t, tau_recharge, tau_decay, max_charge):
#    return max_charge * (1 - np.exp(-t / tau_recharge)) * np.exp(-t) / tau_decay

charge_data = np.array(time_data)
print(len(time_data))
time_data = np.arange(0, len(time_data), 1)
print(len(charge_data))
## Initial guesses for the parameters
#initial_guesses = [20, 20, 0.05]  # Example: [tau_recharge, tau_decay, max_charge]
#
## Fit the model to your data
#popt, pcov = curve_fit(model_function, time_data, charge_data, p0=initial_guesses, maxfev=100000)
##
### Extract the best fitting parameters
##tau_recharge_fit, tau_decay_fit, max_charge_fit = popt
##
### Plot the data and the fit
##plt.scatter(time_data, charge_data, label='Data')
##plt.plot(time_data, model_function(time_data, *popt), label='Fit', color='red')
##plt.xlabel('Time')
##plt.ylabel('Charge Amplitude')
##plt.title('Charge Amplitude Fit')
##plt.legend()
##plt.show()
##
### Print the optimized parameters
##print(f"Optimized tau_recharge: {tau_recharge_fit}")
##print(f"Optimized tau_decay: {tau_decay_fit}")
##print(f"Optimized max_charge: {max_charge_fit}")
##
## Compute the fitted values and residuals
#fitted_values = model_function(time_data, *popt)
#residuals = charge_data - fitted_values
#
## Create subplots
#fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
#
## Plot the data and the fit on the first subplot
#ax1.scatter(time_data, charge_data, label='Data')
#ax1.plot(time_data, fitted_values, label='Fit', color='red')
#ax1.set_ylabel('Integrated Charge Amplitude')
#ax1.set_title('Integrated Charge Amplitude Fit')
#ax1.legend()
#
## Plot the residuals on the second subplot
#ax2.plot(time_data, residuals, label='Residuals', color='green')
#ax2.axhline(0, color='black', linewidth=0.8)  # Horizontal line at 0 for reference
#ax2.set_xlabel('Time')
#ax2.set_ylabel('Residuals')
#ax2.legend()
#
#plt.tight_layout()
#plt.show()
#
## Print the optimized parameters
#print(f"Optimized tau_recharge: {tau_recharge_fit}")
#print(f"Optimized tau_decay: {tau_decay_fit}")
#print(f"Optimized max_charge: {max_charge_fit}")
# Initial guesses for the parameters
initial_guesses = [4, 100, 0.05, 0.5, 0.1, 2*np.pi, 0]  # Example: [tau_recharge, tau_decay, max_charge, A, beta, omega, phi]

# Fit the model to your data
popt, pcov = curve_fit(model_function, time_data, charge_data, p0=initial_guesses)

# Extract the best fitting parameters
tau_recharge_fit, tau_decay_fit, max_charge_fit, A_fit, beta_fit, omega_fit, phi_fit = popt

# Compute the fitted values and residuals
fitted_values = model_function(time_data, *popt)
residuals = charge_data - fitted_values

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

# Plot the data and the fit on the first subplot
ax1.scatter(time_data, charge_data, label='Data')
ax1.plot(time_data, fitted_values, label='Fit', color='red')
ax1.set_ylabel('Integrated Charge Amplitude')
ax1.set_title('Integrated Charge Amplitude Fit with Oscillation')
ax1.legend()

# Plot the residuals on the second subplot
ax2.plot(time_data, residuals, label='Residuals', color='green')
ax2.axhline(0, color='black', linewidth=0.8)  # Horizontal line at 0 for reference
ax2.set_xlabel('Time')
ax2.set_ylabel('Residuals')
ax2.legend()

plt.tight_layout()
plt.show()

# Print the optimized parameters
print(f"Optimized tau_recharge: {tau_recharge_fit}")
print(f"Optimized tau_decay: {tau_decay_fit}")
print(f"Optimized max_charge: {max_charge_fit}")
print(f"Optimized A (Amplitude): {A_fit}")
print(f"Optimized beta (Decay rate): {beta_fit}")
print(f"Optimized omega (Frequency): {omega_fit}")
print(f"Optimized phi (Phase): {phi_fit}")
