import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')
plt.style.use('nature')


labelsize=28
titlesize=42
textsize=21
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
plt.rcParams['errorbar.capsize'] = 5
plt.rcParams['lines.markersize'] = 5  # For example, 8 points
plt.rcParams['lines.linewidth'] = 2 # For example, 2 points
# Set global parameters using rcParams
plt.rcParams['axes.titlepad'] = 20  # Padding above the title
plt.rcParams['axes.labelpad'] = 15  # Padding for both x and y axis labels

# Read the CSV files
measured_df = pd.read_csv('measured_dcr.csv')
crosscheck_df = pd.read_csv('crosscheck_dcr.csv')

## Plotting
#plt.errorbar(measured_df['voltage'], measured_df['dcr']/144, yerr=measured_df['dcr_err'], fmt='-o', label='Measured DCR')
#plt.errorbar(crosscheck_df['voltage'], crosscheck_df['dcr']/144, yerr=crosscheck_df['dcr_err'], fmt='-x', label='Crosscheck DCR#')
# Scale DCR and DCR error by 1/144
scale_factor = 1/144
measured_df['dcr_scaled'] = measured_df['dcr'] * scale_factor
measured_df['dcr_err_scaled'] = measured_df['dcr_err'] * scale_factor
crosscheck_df['dcr_scaled'] = crosscheck_df['dcr'] * scale_factor
crosscheck_df['dcr_err_scaled'] = crosscheck_df['dcr_err'] * scale_factor

# Plotting
plt.errorbar(measured_df['voltage'], measured_df['dcr_scaled'], yerr=measured_df['dcr_err_scaled'], fmt='-o', label='Measured DCR')
plt.errorbar(crosscheck_df['voltage'], crosscheck_df['dcr_scaled'], yerr=crosscheck_df['dcr_err_scaled'], fmt='-x', label='Crosscheck DCR')

plt.ylim(0,100)

# Customization
plt.xlabel('Over-voltage (V)')
plt.ylabel('DCR' + ' ($\\mathrm{Hz}/\\mathrm{mm}^2$)')
plt.title('DCR vs Over Voltage')
plt.legend(loc='upper left')
plt.grid(True)

# Show the plot
plt.savefig("dcr_crosscheck.pdf")

