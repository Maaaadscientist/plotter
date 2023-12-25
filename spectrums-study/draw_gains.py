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


# Creating a simple line plot
plt.figure()  # Set the figure size (optional)
plt.plot(time_bins, gains, '-o', color='blue')  # Plot gains as a blue line with circles at the data points
plt.title('Gains Over Time')  # Title of the plot
plt.xlabel('Time (arbitrary units)')  # X-axis label
plt.ylabel('Gains')  # Y-axis label
plt.grid(True)  # Adding grid for better readability

# Display the plot
plt.savefig("gains.pdf")

