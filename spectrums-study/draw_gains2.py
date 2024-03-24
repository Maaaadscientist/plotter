import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import scienceplots  # Assumed to be a custom style script
plt.style.use('science')
plt.style.use('nature')

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
# Define the improved fit function with t_0 parameter
def improved_fit_function(t, amp_max, tau, t_0):
    shifted_t = t - t_0  # Shift time by t_0
    return amp_max * tau * (0.5 - np.exp(-shifted_t / tau) + 0.5 * np.exp(-2 * shifted_t / tau))

# Perform the curve fitting with the improved function
initial_guess = [max(gains[0:300]), 1, 1265]  # Initial guess for amp_max, tau, and t_0
params, covariance = curve_fit(improved_fit_function, time_bins[0:300], gains[0:300], p0=initial_guess)

# Extract the parameters from the fit
amp_max, tau, t_0 = params

# Calculate the fitted Y values using the improved function
fitted_y = improved_fit_function(time_bins[0:300], amp_max, tau, t_0)

# Plotting the original data and the fitted curve
plt.figure()
plt.plot(time_bins[0:300], gains[0:300], '-o', color='blue', label='Original Data')
plt.plot(time_bins[0:300], fitted_y, color='red', linewidth=2.5, label=f'Improved Fit: $amp\_max \cdot \\tau \cdot (0.5 - \exp(-t/\\tau) + 0.5 \exp(-2t/\\tau))$, $amp\_max$={amp_max:.2f}, $tau$={tau:.2f}')

plt.title('Gains Over Time with Improved Fit')
plt.xlabel('Time (arbitrary units)')
plt.ylabel('Gains')
plt.legend()
plt.grid(True)

# Save the plot with the improved fit
plt.savefig("gains_with_improved_fit.pdf")

