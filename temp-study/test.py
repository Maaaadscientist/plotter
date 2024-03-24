import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import scienceplots
import os,sys
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

# Read the CSV file
df = pd.read_csv('special-runs.csv')

# Define the special runs dictionary
special_runs = {425: 48, 426: 48, 427: 48, 428: 48, 429: 50, 430: 48, 431: 48, 432: 49, 433: 49, 435: 50, 436: 47, 439: 50, 440: 50, 441: 50, 443: 49}

# Define the temperature dictionary
temp_dict = {426: -30, 427: -20, 428: -10, 429: 0, 430: -55, 431: -45, 432: -35, 433: -25, 435: -15, 436: -60, 437: -51.5, 438: -40, 439: -5, 440: -20, 441: -10, 443: -30, 444: -51.5, 445: -51.5, 446: -51.5, 447: -51.5}

# Exclude runs
exclude_runs = [426, 427, 428, 429, 439, 435]

# Filter the data for a specific pos and ch, and exclude runs
pos_filter = int(sys.argv[1])  # Replace with the desired pos value
tsn_filter = 3394
ch_filter = int(sys.argv[2])  # Replace with the desired ch value
filtered_df = df[(df['pos'] == pos_filter) & (df['ch'] == ch_filter) & ~(df['run'].isin(exclude_runs))]
#filtered_df = df[(df['tsn'] == tsn_filter) & (df['ch'] == ch_filter) & ~(df['run'].isin(exclude_runs))]

# Correct vbd values for specific runs
corrected_vbd = filtered_df['vbd'].copy()
#corrected_vbd.loc[filtered_df['run'] == 436] += -1
#corrected_vbd.loc[filtered_df['run'] == 443] += 1
#corrected_vbd.loc[filtered_df['run'] == 435] += 2
#corrected_vbd.loc[filtered_df['run'] == 432] += 1
#corrected_vbd.loc[filtered_df['run'] == 433] += 1
#corrected_vbd.loc[filtered_df['run'].isin([440, 441])] += 2

# Calculate the real voltage and overvoltage
filtered_df['real_voltage'] = filtered_df['vol'] + filtered_df['run'].map(special_runs).fillna(48)
filtered_df['overvoltage'] = filtered_df['real_voltage'] - corrected_vbd

# Extract the vbd, corrected_vbd, vbd_err, and temp columns
vbd_data = filtered_df['vbd']
corrected_vbd_data = corrected_vbd
vbd_err_data = filtered_df['vbd_err']
temp_data = filtered_df['run'].map(temp_dict)

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(temp_data, corrected_vbd_data)

# Create the plot
plt.figure()
#plt.errorbar(temp_data, vbd_data, yerr=vbd_err_data, fmt='o', color='b', label='Original VBD', capsize=3)
plt.errorbar(temp_data, corrected_vbd_data, yerr=vbd_err_data, fmt='o', color='black', label='Corrected VBD', capsize=3)
plt.plot(temp_data, slope * temp_data + intercept,color='r', label=f'Linear Fit (slope={slope:.4f}, intercept={intercept:.3f})')
plt.xlabel('Temperature (°C)')
plt.ylabel('VBD (V)')
plt.title(f'VBD vs Temperature for tsn={tsn_filter}, ch={ch_filter}')
plt.ylim(45,52)
plt.legend()
plt.grid(True)
plt.savefig(f"vbd_temp_{pos_filter}_{ch_filter}.pdf")
