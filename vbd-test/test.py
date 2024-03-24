import matplotlib.pyplot as plt
import numpy as np
import scienceplots
plt.style.use('science')
plt.style.use('nature')

labelsize=28
titlesize=40
textsize=24
size_marker = 100

# Set global font sizes
#plt.rcParams['text.usetex'] = False
plt.rcParams['figure.figsize'] = (20, 15)
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

# Load the data
ch10_path = "/Users/wanghanwen/IHEPBox/TAO/IV_test_2024.02.20/ch10_-50degrees.TXT"
ch11_path = "/Users/wanghanwen/IHEPBox/TAO/IV_test_2024.02.20/ch11_-50degrees.TXT"
ch12_path = "/Users/wanghanwen/IHEPBox/TAO/IV_test_2024.02.20/ch12_-50degrees.TXT"
ch14_path = "/Users/wanghanwen/IHEPBox/TAO/IV_test_2024.02.20/ch14_-50degrees.TXT"
plt.figure()
for ch in [10,11,12,14]:
    data = np.loadtxt(globals()[f'ch{ch}'+'_path'], delimiter=';')
    
    # Extract voltage and current
    voltage = data[:, 1]  # Second column
    current = data[:, 2]  # Third column
    mask = voltage <= 52.9
    filtered_voltage = voltage[mask]
    filtered_current = current[mask]
    
    
    # Plot the voltage-current curve
    plt.plot(filtered_voltage, filtered_current, label=f'ch {ch}')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.title('Current v.s. Voltage')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig("IVcurve.pdf")

