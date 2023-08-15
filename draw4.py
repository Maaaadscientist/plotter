import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

# Read the coordinates from the text file
file_path = 'coordinates.txt'  # Update this with your file path
data = np.loadtxt(file_path, delimiter=',', unpack=True)

x = data[0]
y = data[1]

# Sort coordinates by x-values
sorted_indices = np.argsort(x)
x_sorted = x[sorted_indices]
y_sorted = y[sorted_indices]

# Find the index where the split should occur
split_index = np.searchsorted(x_sorted, 5.8e15)

# Split the data into two segments
x_segment1 = x_sorted[:split_index]
y_segment1 = y_sorted[:split_index]
x_segment2 = x_sorted[split_index:]
y_segment2 = y_sorted[split_index:]

# Create PCHIP interpolation functions for each segment
pchip_segment1 = PchipInterpolator(x_segment1, y_segment1)
pchip_segment2 = PchipInterpolator(x_segment2, y_segment2)

# Generate points for interpolation
x_interp1 = np.logspace(np.log10(min(x_segment1)), np.log10(max(x_segment1)), 500)
x_interp2 = np.logspace(np.log10(min(x_segment2)), np.log10(max(x_segment2)), 500)

# Calculate interpolated y-values using the PCHIP functions for each segment
y_interp1 = pchip_segment1(x_interp1)
y_interp2 = pchip_segment2(x_interp2)

# Combine the two segments for the final interpolated curve
x_combined = np.concatenate((x_interp1, x_interp2))
y_combined = np.concatenate((y_interp1, y_interp2))

# Create a logarithmic plot for both x and y axes with the combined interpolated curve
plt.figure()
plt.loglog(x_combined, y_combined, linestyle='-', color='b', label = "energy spectrum")
plt.xlabel('Neutrino Energy (eV)')
# Add LaTeX text in Y-axis title
plt.ylabel(r'Cross Section $\left(\bar{v}_{\mathrm{e}} \mathrm{e}^{-} \rightarrow \bar{v}_{\mathrm{e}} \mathrm{e}^{-}\right.$ in $mb$', fontsize=12)
plt.title('Combined Interpolated Logarithmic X-Y Plot')
plt.legend()

# Set x and y axis limits
plt.xlim(1e-5, 1e18)
plt.ylim(1e-32, 1e-1)

# Add shaded region and text annotation
plt.axvspan(1e-5, 1e-3, color='gray', alpha=0.3)
plt.text(2e-4, 5e-2, 'Big Bang', fontsize=10, color='r')

# Customize plot appearance
plt.grid(True)
plt.show()

