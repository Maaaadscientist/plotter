import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
plt.style.use('nature')

labelsize=28
titlesize=40
textsize=24
size_marker = 100

# Set global font sizes
plt.rcParams['figure.figsize'] = (25, 15)
plt.rcParams['font.size'] = textsize  # Sets default font size
plt.rcParams['axes.labelsize'] = labelsize
plt.rcParams['axes.titlesize'] = titlesize
plt.rcParams['xtick.labelsize'] = labelsize
plt.rcParams['ytick.labelsize'] = labelsize
plt.rcParams['legend.fontsize'] = textsize
plt.rcParams['errorbar.capsize'] = 4
plt.rcParams['lines.markersize'] = 5  # For example, 8 points
plt.rcParams['lines.linewidth'] = 2 # For example, 2 points
# Set global parameters using rcParams
plt.rcParams['axes.titlepad'] = 20  # Padding above the title
plt.rcParams['axes.labelpad'] = 15  # Padding for both x and y axis labels

# Load the data
file_path = sys.argv[1]#'path/to/your/file.csv'  # Update this to the path of your CSV file
data = pd.read_csv(file_path)

# Select the specific 'tsn'
selected_tsn = int(sys.argv[2])  # Replace with the tsn value you're interested in
data = data[data['tsn'] == selected_tsn]
# Adjust 'vol' by adding 48
data['adjusted_vol'] = data['vol'] + 48
plt.figure()

# Define the number of channels
channels = range(16)  # Adjust if your channel numbering is different

for ch in channels:
    # Filter data for the specific channel
    channel_data = data[data['ch'] == ch]
    
    # Plotting
    plt.errorbar(channel_data['adjusted_vol'], channel_data['prefit_gain'],
                 yerr=channel_data['prefit_gain_err'], fmt='o', capsize=5,
                 label=f'Channel {ch}')

# Adding labels and title
plt.title(f'Gain vs. Voltage for: {selected_tsn}')
plt.xlabel('Voltage (V)')
plt.ylabel('Gain')
plt.legend()  # Add a legend to distinguish channels

# Show the plot
plt.savefig("fit_gain.pdf")


