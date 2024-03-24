import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import scienceplots
plt.style.use('science')
plt.style.use('nature')

labelsize=28
titlesize=40
textsize=24
size_marker = 100

# Set global font sizes
#plt.rcParams['text.usetex'] = False
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


# Read the CSV file
df = pd.read_csv('special-runs.csv')

# Define the special runs dictionary
special_runs = {425: 48, 426: 48, 427: 48, 428: 48, 429: 50, 430: 48, 431: 48, 432: 49, 433: 49, 435: 50, 436: 47, 439: 50, 440: 50, 441: 50, 443: 49}

# Define the temperature dictionary
temp_dict = {426: -30, 427: -20, 428: -10, 429: 0, 430: -55, 431: -45, 432: -35, 433: -25, 435: -15, 436: -60, 437: -51.5, 438: -40, 439: -5, 440: -20, 441: -10, 443: -30, 444: -51.5, 445: -51.5, 446: -51.5, 447: -51.5}

# Exclude runs
exclude_runs = [426, 427, 428, 429,433, 439, 441]

# Set the pos value
pos_filter = 10  # Replace with the desired pos value

# Create the plot
fig, ax = plt.subplots()

# List of colors for channels
colors = list(mcolors.TABLEAU_COLORS.values())[:8]
# Loop over channels
legend_entries = []
for ch_filter in range(1, 17):
    # Filter the data for the current pos and ch, and exclude runs
    filtered_df = df[(df['pos'] == pos_filter) & (df['ch'] == ch_filter) & ~(df['run'].isin(exclude_runs))]

    # Correct vbd values for specific runs
    corrected_vbd = filtered_df['vbd'].copy()
    #corrected_vbd.loc[filtered_df['run'] == 443] += 1
    #corrected_vbd.loc[filtered_df['run'].isin([440, 441])] += 2

    # Calculate the real voltage and overvoltage
    filtered_df['real_voltage'] = filtered_df['vol'] + filtered_df['run'].map(special_runs).fillna(48)
    filtered_df['overvoltage'] = filtered_df['real_voltage'] - corrected_vbd

    # Extract the corrected_vbd, vbd_err, and temp columns
    corrected_vbd_data = corrected_vbd
    vbd_err_data = filtered_df['vbd_err']
    temp_data = filtered_df['run'].map(temp_dict)

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(temp_data, corrected_vbd_data)

    # Get the color and line style for the current channel
    color = colors[(ch_filter - 1) % len(colors)]
    linestyle = '-' if ch_filter <= 8 else '--'

    # Plot the data and linear fit
    data_line, = ax.plot(temp_data, corrected_vbd_data, marker='o', color=color, linestyle='')
    fit_line, = ax.plot(temp_data, slope * temp_data + intercept, linestyle=linestyle, color=color, label=f'Linear Fit (ch={ch_filter}, slope={slope:.3f})')
    legend_entries.append((data_line, fit_line))



ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('VBD (V)')
ax.set_title(f'VBD vs Temperature for pos={pos_filter}')

#Create a separate subplot for the legend
legend_ax = fig.add_axes([0.8, 0.1, 0.2, 0.8])  # [left, bottom, width, height]
legend_ax.axis('off')  # Turn off the axes for the legend subplot
# Create custom legend handles and labels

legend_lines, legend_labels = ax.get_legend_handles_labels()
legend_ax.legend(legend_lines, legend_labels, loc='center')

plt.subplots_adjust(right=0.75)  # Adjust the spacing between subplots
plt.grid(True)
plt.savefig("output2.pdf")
