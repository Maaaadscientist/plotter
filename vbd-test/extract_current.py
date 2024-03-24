import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import re

def gaussian(x, mean, amplitude, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

#directory = 'path/to/your/directory'  # Update this path
#directory = '/Users/wanghanwen/IHEPBox/TAO/TEST_0304'
directory_root = '/Users/wanghanwen/codes/plotter/vbd-test/ivtest'
voltages = {}
mean_currents = {}

# Loop through each file in the directory
for ch in [14,16]:
    voltages[ch] = []
    mean_currents[ch] = []
    directory = directory_root + f'/ch{ch}-cold'
    print(directory)
    for filename in os.listdir(directory):
        print(filename)
        if filename.endswith('.TXT'):
            # Extract voltage from the filename
            voltage = float(re.search(r'(\d+\.\d+)V', filename).group(1))
            file_path = os.path.join(directory, filename)
            
            # Read the current values from the file
            currents = np.loadtxt(file_path, delimiter=';', usecols=[2], skiprows=4)
            
            # Histogram of the currents (without plotting it)
            counts, bin_edges = np.histogram(currents, bins=30, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Fit the histogram with a Gaussian
            popt, _ = curve_fit(gaussian, bin_centers, counts, p0=[np.mean(currents), np.max(counts), np.std(currents)])
            
            voltages[ch].append(voltage)
            mean_currents[ch].append(popt[0])  # mean current from Gaussian fit

# Plotting the current vs voltage
plt.figure(figsize=(10, 6))
plt.scatter(voltages[14], mean_currents[14], color='green', label='ch14')
plt.scatter(voltages[16], mean_currents[16], color='red', label='ch16')
plt.title('Mean Current vs. Voltage')
plt.xlabel('Voltage (V)')
plt.ylabel('Mean Current (A)')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.show()

