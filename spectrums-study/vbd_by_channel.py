import os
import sys
import numpy as np
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
#plt.rcParams['text.usetex'] = False
plt.rcParams['figure.figsize'] = (20, 15)
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
selected_tsn = float(sys.argv[2])  # Replace with the tsn value you're interested in
data = data[data['tsn'] == selected_tsn]
runs = np.unique(data['run'].to_numpy())
print(runs)
# Adjust 'vol' by adding 48
data['adjusted_vol'] = data['vol'] + 48

# Define the number of channels
channels = np.arange(1,17,1)  # Adjust if your channel numbering is different


temp_norm = False
# Define a list of marker styles and line styles
markers = ['o', 's', 'v', 'D', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
linestyles = ['-', '--', '-.', ':']

for run in runs:
    plt.figure()
    # Initialize variables to identify the channel with the maximum vbd
    max_vbd = -np.inf
    max_vbd_channel = None
    min_vbd = np.inf
    min_vbd_channel = None
    # Create combinations of markers and line styles
    marker_iter = iter(markers)
    line_iter = iter(linestyles)
    count = 0
    # Create empty lists to store handles for lines and markers
    line_handles = []
    errorbar_containers = []
    vbd_dict = {}
    slope_dict = {}
    vbd_err_dict = {}
    voltage_change_per_degree = 0.054  # V/°C
    temp_dict = {}
    for ch in channels:
        # Filter data for the specific channel
        channel_data = data[(data['ch'] == ch) & (data['run'] == run)]
        pos =  np.unique(channel_data['pos'].to_numpy())[0]
    
        if channel_data.empty:
            continue  # Skip if no data for this channel
    
        # Get slope and vbd for the linear fit (ensure you have valid data)
        slope = channel_data['slope'].iloc[0]  # Assuming slope is constant per channel
        vbd = channel_data['vbd'].iloc[0]      # Assuming vbd is constant per channel
        nPoints =np.unique(channel_data['ndf'].to_numpy())[0] # int(channel_data['ndf'].iloc[0]) + 2
        # Update the channel with the maximum vbd
        if vbd > max_vbd:
            max_vbd = vbd
            max_vbd_channel = ch
        # Update the channel with the maximum vbd
        if vbd < min_vbd:
            min_vbd = vbd
            min_vbd_channel = ch
        vbd_err = channel_data['vbd_err'].iloc[0]      # Assuming vbd is constant per channel
        # Filter out zero values from the temperature data
        temp = channel_data['temp'].to_numpy()
        temp = temp[temp != 0]
        temp_mean = temp.mean()
        temp_dict[ch] = temp_mean
        vbd_50 = vbd + (-50 - temp_mean) * voltage_change_per_degree
        
    
        if len(temp) > 1:  # Ensure there's enough data for a standard error calculation
            # Calculate the standard error of the temperature
            temp_std_err = np.sqrt(temp.var())#np.std(temp, ddof=1) #/ np.sqrt(len(temp))
            
            # Calculate the additional vbd_err due to temperature fluctuations
            temp_induced_vbd_err = temp_std_err * voltage_change_per_degree
            
            # Combine the inherent vbd_err with the temperature-induced error
            total_vbd_err = np.sqrt(vbd_err**2 + temp_induced_vbd_err**2)
        else:
            # If there's not enough temperature data, use the inherent vbd_err
            total_vbd_err = vbd_err
    
        # Calculate the y-values for the fit line
        fit_x = channel_data['adjusted_vol']
        
        # Calculate x-intercept where y=0 (x = (0 - intercept) / slope)
        x_intercept = vbd  # As y = slope * (x - vbd), y=0 when x=vbd for your specific case
        
        if temp_norm:
            vbd_dict[ch] = vbd_50
        else:
            vbd_dict[ch] = vbd
        vbd_err_dict[ch] = total_vbd_err
        slope_dict[ch] = slope
        # Extend the fit_x to include the x_intercept if it's not already included
        if x_intercept < fit_x.min():
            extended_fit_x = np.linspace(x_intercept, fit_x.max(), 1000)  # Change 1000 to a higher number for a smoother line if needed
        else:
            extended_fit_x = fit_x
        fit_y = slope * (fit_x - vbd)  # y = slope * (x - x_intercept)
        
        # Calculate extended fit_y with the extended_fit_x
        extended_fit_y = slope * (extended_fit_x - vbd)
        if count % 7 == 0:
            marker = next(marker_iter)
            linestyle = next(line_iter)
        count += 1
     
        # Plotting the channel data (markers)
        marker_plot = plt.errorbar(fit_x, channel_data['prefit_gain'],
                                    yerr=channel_data['prefit_gain_err'], capsize=5,
                                    color=f"C{ch}", fmt=f'{marker}')
        errorbar_containers.append(marker_plot)  # Store the marker handle
    
        # Plotting the linear fit (line)
        line_plot, = plt.plot(extended_fit_x, extended_fit_y, linestyle=f'{linestyle}', color=f"C{ch}")
        line_handles.append(line_plot)  # Store the line handle   plt.plot(fit_x, fit_y, label=f'Channel {ch}',linestyle=f'{linestyle}', color=f"C{ch}")
    
    # Create custom legend entries that combine line and marker for each channel
    #custom_legend = [plt.Line2D([0], [0], color=handle.get_color(), marker=errorbar_containers[idx].get_marker(),
    #                            linestyle=handle.get_linestyle()) for idx, handle in enumerate(line_handles)]
    #
    #
    #plt.legend(custom_legend, [f'Channel {ch}' for ch in channels if not data[data['ch'] == ch].empty], loc='best')
    #custom_legend = [plt.Line2D([0], [0], color=line.get_color(), 
    #                            marker=errorbar_containers[idx].lines[0].get_marker(), 
    #                            linestyle=line.get_linestyle()) 
    #                 for idx, line in enumerate(line_handles)]
    #plt.legend(custom_legend, [f'Channel {ch}' +" $\\mathrm{V}_\\mathrm{bd}$: " + f" {vbd_dict[ch]:.2f} (V)"  for ch in channels if not data[data['ch'] == ch].empty], loc='best')
    
    # Split the handles and labels for two legends
    first_half_handles = []
    second_half_handles = []
    first_half_labels = []
    second_half_labels = []
    
    total_channels = len(line_handles)  # Adjust this if you're using a different criterion
    
    for idx, (line, errorbar_container) in enumerate(zip(line_handles, errorbar_containers)):
        custom_handle = plt.Line2D([0], [0], color=line.get_color(), 
                                   marker=errorbar_container.lines[0].get_marker(), 
                                   linestyle=line.get_linestyle())
        
        #label = f'Ch. {idx+1}' +" $\\mathrm{V}_\\mathrm{bd}$: " + f" {vbd_dict[idx + 1]:.3f}"+" $\\pm$ "+f"{vbd_err_dict[idx + 1]:.3f} (V)"  # Adjust this label as per your requirement
        if temp_norm:
            label = f'Ch.{idx+1},' +" $\\mathrm{V}^{-50^\\circ\\mathrm{C}}_\\mathrm{breakdown}$: " + f" {vbd_dict[idx + 1]:.3f}"+" $\\pm$ "+f"{vbd_err_dict[idx + 1]:.3f}, " + " Slope: " +f"{slope_dict[idx + 1]:.2f}" #+ " $\\mathrm{N}_\\mathrm{point}$: " +f" {nPoints}"
        else:
            label = f'Ch.{idx+1},' +" $\\mathrm{V}_\\mathrm{breakdown}$: " + f" {vbd_dict[idx + 1]:.3f}"+" $\\pm$ "+f"{vbd_err_dict[idx + 1]:.3f}, " + " Slope: " +f"{slope_dict[idx + 1]:.2f}" #+ " $\\mathrm{N}_\\mathrm{point}$: " +f" {nPoints}"
        
        if idx < total_channels / 2:  # Adjust this condition to split as you like
            first_half_handles.append(custom_handle)
            first_half_labels.append(label)
        else:
            second_half_handles.append(custom_handle)
            second_half_labels.append(label)
    
    # Create the first legend and add it to the plot
    first_legend = plt.legend(first_half_handles, first_half_labels, loc='upper left')
    plt.gca().add_artist(first_legend)  # Important to keep the first legend when adding the second
    
    # Create the second legend and add it to the plot
    second_legend = plt.legend(second_half_handles, second_half_labels, loc='lower right')
    # Change the text color for the maximum vbd channel
    for text, ch in zip(first_legend.get_texts(), np.arange(1,9,1)):
        if ch == max_vbd_channel:
            text.set_color('red')  # Change 'red' to your preferred color
        if ch == min_vbd_channel:
            text.set_color('blue')  # Change 'red' to your preferred color
    
    for text, ch in zip(second_legend.get_texts(), np.arange(9,17,1)):
        if ch == max_vbd_channel:
            text.set_color('red')  # Change 'red' to your preferred color
        if ch == min_vbd_channel:
            text.set_color('blue')  # Change 'red' to your preferred color
    temp_list = [temp_dict[ch] for ch in channels ]
    vbd_list = [vbd_dict[ch] for ch in channels ]
    vbd_err_list = [vbd_err_dict[ch] for ch in channels ]
    print(vbd_list)
    print(vbd_err_list)
    # Initialize variables to keep track of the maximum difference and its associated error
    max_diff = 0
    max_diff_err = 0
    
    # Calculate pairwise differences and their errors
    for i in range(1,17):
        for j in range(i+1, 17):  # Avoid repeating comparisons and comparing the same channels
            # Calculate the difference between vbd values
            diff = abs(vbd_dict[i] - vbd_dict[j])
            
            # Calculate the error associated with this difference
            # Assuming the errors are independent, you'd add them in quadrature (square root of the sum of the squares)
            err = np.sqrt(vbd_err_dict[i]**2 + vbd_err_dict[j]**2)
            
            # Update the maximum difference and its error if this one is larger
            if diff > max_diff:
                max_diff = diff
                max_diff_err = err
    params_text = (
                  f"Temperature: {np.mean(temp_list):.2f}" + "$^\\circ\\mathrm{C}$\n"
                  "Max $\\mathrm{V}_\\mathrm{breakdown}$ diff. :" + f"{max_diff:.3f}" + " $\\pm$ " + f"{max_diff_err:.3f} (V)\n"
                  f"Max channel {max_vbd_channel} :"  + f"{max_vbd:.3f}" + " $\\pm$ " + f"{vbd_err_dict[max_vbd_channel]:.3f} (V)\n"
                  f"Min channel {min_vbd_channel} :"  + f"{min_vbd:.3f}" + " $\\pm$ " + f"{vbd_err_dict[min_vbd_channel]:.3f} (V)"
                  )
    plt.text(0.04, 0.6, params_text, transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=textsize)
    
    # Adding labels and title
    plt.title(f'Gain vs. Voltage of tsn {int(selected_tsn)} (Run {run} PCBid {pos})')
    plt.ylim(0,55)
    plt.xlabel('Voltage (V)')
    plt.ylabel('Charge Gain (pC)')
    #plt.legend()  # Add a legend to distinguish channels
    
    # Save the plot
    plt.savefig(f"fit_gain_tsn_{int(selected_tsn)}_run_{run}.pdf")
    plt.clf()
