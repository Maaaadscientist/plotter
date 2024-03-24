import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
import scienceplots
plt.style.use('science')
plt.style.use('nature')

labelsize=28
titlesize=40
textsize=24
size_marker = 100

# Set global font sizes
#plt.rcParams['text.usetex'] = False
plt.rcParams['figure.figsize'] = (20, 12)
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


param_name = 'enf_data'
ov = 6
# Read the CSV file
df = pd.read_csv('special-runs.csv')

# Define the special runs dictionary
special_runs = {425: 48, 426: 48, 427: 48, 428: 48, 429: 50, 430: 48, 431: 48, 432: 49, 433: 49, 435: 50, 436: 47, 439: 50, 440: 50, 441: 50, 443: 49}

# Define the temperature dictionary
temp_dict = {426: -30, 427: -20, 428: -10, 429: 0, 430: -55, 431: -45, 432: -35, 433: -25, 435: -15, 436: -60, 437: -51.5, 438: -40, 439: -5, 440: -20, 441: -10, 443: -30, 444: -51.5, 445: -51.5, 446: -51.5, 447: -51.5}

# Exclude runs
exclude_runs = [426, 427, 428, 429, 439, 441]
#exclude_runs = [426, 427, 428,]

# Set the pos and ch values
pos_filter = 14  # Replace with the desired pos value
ch_filter =  3 # Replace with the desired ch value

# Create the plot
fig, ax = plt.subplots()

# Filter the data for the desired pos and ch
filtered_df = df[(df['pos'] == pos_filter) & (df['ch'] == ch_filter) & ~(df['run'].isin(exclude_runs))]

# Loop over runs
legend_entries = []
for run_filter, run_group in filtered_df.groupby('run'):
    # Correct vbd values for specific runs
    corrected_vbd = run_group['vbd'].copy()
    #corrected_vbd.loc[run_group['run'] == 443] += 1
    #corrected_vbd.loc[run_group['run'].isin([440, 441])] += 2

    # Calculate the real voltage and overvoltage
    run_group['real_voltage'] = run_group['vol'] + run_group['run'].map(special_runs).fillna(48)
    run_group['overvoltage'] = run_group['real_voltage'] - corrected_vbd

    # Interpolate alpha and alpha_err at ov=3
    ov_data = run_group['overvoltage']
    alpha_data = run_group[param_name]
    alpha_err_data = run_group[f'{param_name}_err']
    interp_func_alpha = interp1d(ov_data, alpha_data, kind='linear', fill_value='extrapolate')
    interp_func_alpha_err = interp1d(ov_data, alpha_err_data, kind='nearest', fill_value='extrapolate')
    alpha_at_ov3 = interp_func_alpha(ov)
    alpha_err_at_ov3 = interp_func_alpha_err(ov)

    # Get the temperature for the current run
    temp_data = temp_dict[run_filter]

    # Plot the interpolated alpha value with error bars
    ax.errorbar(temp_data, alpha_at_ov3, yerr=alpha_err_at_ov3, fmt='o', label=f'run={run_filter}', capsize=3)

ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Crosstalk')
ax.set_title(f'Crosstalk vs Temperature for pos={pos_filter}, ch={ch_filter}')

# Create a separate subplot for the legend
#legend_ax = fig.add_axes([0.8, 0.1, 0.2, 0.8])  # [left, bottom, width, height]
#legend_ax.axis('off')  # Turn off the axes for the legend subplot
#legend_ax.legend()

plt.subplots_adjust(right=0.75)  # Adjust the spacing between subplots
plt.grid(True)
plt.legend()
plt.savefig(f"{param_name}_pos{pos_filter}_ch{ch_filter}.pdf")
