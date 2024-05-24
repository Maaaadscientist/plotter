import matplotlib.pyplot as plt
import numpy as np
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
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(20, 15), gridspec_kw={'height_ratios': [5, 2]})
# Plot for Mean

# Plot for Mean
color = 'tab:red'
ax1.set_ylabel('Mean', color=color)
ax1.plot(time_bins, means, label='Mean', color=color, linestyle=":")
ax1.tick_params(axis='y', labelcolor=color)
# ... other code for ax1 ...
ax1.legend(loc='best')  # Add this line for ax1's legend

#ax1.grid(True, which='both', linestyle='--', linewidth=0.5)  # Enable grid

# Create a second y-axis for the same subplot to plot Variance
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Var.', color=color)
ax2.plot(time_bins, variances, label='Variance', color=color, linestyle="--")
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='center right')  # Add this line for ax1's legend
#ax2.grid(True, which='both', linestyle='--', linewidth=0.5)  # Enable grid for the second y-axis

# Plot for Ratio of Variance to Mean Squared
color = 'tab:green'
ax3.set_xlabel('Time (8 ns)')
ax3.set_ylabel('$\\mathrm{Res.}^2$', color=color)
ax3.plot(time_bins, ratios, color=color)
ax3.tick_params(axis='y', labelcolor=color)
ax3.set_ylim(0.97, 1.12)
ax3.grid(True, which='both', linestyle='--', linewidth=0.5)  # Enable grid

# Improve layout to accommodate for the second subplot and titles
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#plt.legend()
plt.title("Mean, Variance, and Resolution of the Charge Spectrum", pad=20)
plt.savefig("timewindow_check.pdf")
