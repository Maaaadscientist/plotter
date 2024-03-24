import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Set parameters for visual aesthetics
labelsize = 28
titlesize = 40
textsize = 24
plt.rcParams.update({
    'figure.figsize': (25, 15),
    'font.size': textsize,
    'axes.labelsize': labelsize,
    'axes.titlesize': titlesize,
    'xtick.labelsize': labelsize,
    'ytick.labelsize': labelsize,
    'legend.fontsize': labelsize,
    'errorbar.capsize': 3,
    'lines.markersize': 5,
    'lines.linewidth': 2,
    'axes.titlepad': 20,
    'axes.labelpad': 15
})

# Import your spectrum parameters
from spectrum_parameters import time_bins, gains

time_bins = time_bins[0:500]
gains = gains[0:500]
# Define the fit function with the integral result and t_0
def fit_function(t, amp_max, tau_0, tau_1, t_0):
    shifted_t = t - t_0  # Shift time by t_0
    # Define the integral of the function over the range [0, shifted_t]
    return amp_max * (
        (tau_1**2) / (tau_0 + tau_1)
        - tau_0 * tau_1 * np.exp(shifted_t/tau_0) / (tau_0 * np.exp(shifted_t/tau_0) * np.exp(shifted_t/tau_1) + tau_1 * np.exp(shifted_t/tau_0) * np.exp(shifted_t/tau_1))
        + tau_0 * tau_1 / (tau_0 * np.exp(shifted_t/tau_0) * np.exp(shifted_t/tau_1) + tau_1 * np.exp(shifted_t/tau_0) * np.exp(shifted_t/tau_1))
        - tau_1**2 * np.exp(shifted_t/tau_0) / (tau_0 * np.exp(shifted_t/tau_0) * np.exp(shifted_t/tau_1) + tau_1 * np.exp(shifted_t/tau_0) * np.exp(shifted_t/tau_1))
    )

# Perform the curve fitting with the improved function
initial_guess = [max(gains), 1, 1, 1265]  # Initial guess for amp_max, tau_0, tau_1, and t_0
params, covariance = curve_fit(fit_function, time_bins, gains, p0=initial_guess)

# Extract the parameters from the fit
amp_max, tau_0, tau_1, t_0 = params

time_bins = np.array(time_bins)
# Calculate the fitted Y values using the improved function
fitted_y = fit_function(time_bins, amp_max, tau_0, tau_1, t_0)

# Plotting the original data and the fitted curve
plt.figure()
plt.plot(time_bins, gains, '-o', color='blue', label='Original Data')
plt.plot(time_bins, fitted_y, color='red', linewidth=2.5, label=f'Fit with Integral Function, $amp\_max$={amp_max:.2f}, $tau_0$={tau_0:.2f}, $tau_1$={tau_1:.2f}, $t_0$={t_0}')

plt.title('Gains Over Time with Improved Fit')
plt.xlabel('Time (arbitrary units)')
plt.ylabel('Gains')
plt.legend()
plt.grid(True)

# Save the plot with the improved fit
plt.savefig("gains_with_improved_fit_integral.pdf")

