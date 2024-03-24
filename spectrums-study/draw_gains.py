import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, gaussian
from scipy.ndimage import gaussian_filter1d
from spectrum_parameters import time_bins, means, variances, ratios, gains

import scienceplots
plt.style.use('science')
plt.style.use('nature')

labelsize=28
titlesize=40
textsize=24
size_marker = 100
timeEnd = 2009

# Set global font sizes
plt.rcParams['figure.figsize'] = (25, 15)
plt.rcParams['font.size'] = textsize  # Sets default font size
plt.rcParams['axes.labelsize'] = labelsize
plt.rcParams['axes.titlesize'] = titlesize
plt.rcParams['xtick.labelsize'] = labelsize
plt.rcParams['ytick.labelsize'] = labelsize
plt.rcParams['legend.fontsize'] = labelsize
plt.rcParams['errorbar.capsize'] = 3
plt.rcParams['lines.markersize'] = 5  # For example, 8 points
plt.rcParams['lines.linewidth'] = 2 # For example, 2 points
# Set global parameters using rcParams
plt.rcParams['axes.titlepad'] = 20  # Padding above the title
plt.rcParams['axes.labelpad'] = 15  # Padding for both x and y axis labels


# Define the function to fit
def fit_function(t, max_gain, tau):
    return max_gain * (1 - np.exp(-(t - 1265) / tau))

# Perform the curve fitting
params, covariance = curve_fit(fit_function, time_bins[0:300], gains[0:300], p0=[max(gains[0:300]), 1])

# Extract the parameters
max_gain, tau = params

# Create a smooth set of X values for plotting the fit
#x_smooth = np.linspace(min(time_bins), max(time_bins), 400)

# Calculate the fitted Y values
time_bins = np.array(time_bins)
fitted_y = fit_function(time_bins[0:300], max_gain , tau)

# Plotting the original data
plt.figure()
plt.plot(time_bins[0:300], gains[0:300], '-o', color='blue', label='Original Data')

# Plotting the fitted curve
plt.plot(time_bins[0:300], fitted_y, color='red', linewidth=2.5, label=f'Fit: max_gain$(1 - \exp(-t/\tau))$, max_gain={max_gain:.2f}, tau={tau:.2f}')

plt.title('Gains Over Time with Fit')  # Title of the plot
plt.xlabel('Time (arbitrary units)')  # X-axis label
plt.ylabel('Gains')  # Y-axis label
plt.legend()
plt.grid(True)  # Adding grid for better readability

# Save the plot with the fit
plt.savefig("gains_with_fit.pdf")
