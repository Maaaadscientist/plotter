import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os, sys
import scienceplots
plt.style.use('science')
plt.style.use('nature')

labelsize=28
titlesize=40
textsize=24
size_marker = 100

# Set global font sizes
#plt.rcParams['text.usetex'] = False
plt.rcParams['figure.figsize'] = (24, 15)
plt.rcParams['font.size'] = textsize  # Sets default font size
plt.rcParams['axes.labelsize'] = labelsize
plt.rcParams['axes.titlesize'] = titlesize
plt.rcParams['xtick.labelsize'] = labelsize
plt.rcParams['ytick.labelsize'] = labelsize
plt.rcParams['legend.fontsize'] = labelsize
plt.rcParams['errorbar.capsize'] = 4
plt.rcParams['lines.markersize'] = 8  # For example, 8 points
plt.rcParams['lines.linewidth'] = 2 # For example, 2 points
# Set global parameters using rcParams
plt.rcParams['axes.titlepad'] = 20  # Padding above the title
plt.rcParams['axes.labelpad'] = 15  # Padding for both x and y axis labels

# Step 1: Define the ADC thresholds
adc_thresholds = range(0, 1126, 25)  # Adjust the range if necessary

#dir_path = os.path.abspath(sys.argv[1])
root_name = "dcr"
vol_list = [48.5, 49.5, 50.5, 51.5, 52.5, 53.5]
cmap = plt.cm.get_cmap('viridis', len(vol_list))  # Get a colormap with distinct colors

for idx, vol in enumerate(vol_list):
    all_data_list = []  # List to hold individual DataFrames
    errors_list = []  # List to hold the errors
    # Step 2: Load the data from each file and append it to the list
    for threshold in adc_thresholds:
        filename = f'darkcounts_{vol}V_-50/data_{threshold}.csv'
        temp_df = pd.read_csv(filename, header=None)
        #temp_df['ADC_Threshold'] = threshold  # Add a column for the ADC Threshold
        #all_data_list.append(temp_df)
        # Calculate the mean for each channel across the 6 measurements
        mean_data = temp_df.mean().to_frame().T  # Transpose to get the means as a row
        mean_data['ADC_Threshold'] = threshold  # Add a column for the ADC Threshold
        all_data_list.append(mean_data)
        # Calculate the error for each channel
        errors = np.sqrt(temp_df.sum(axis=0)) / 6
        errors_df = errors.to_frame().T  # Convert Series to DataFrame
        errors_list.append(errors_df) 
    
    # Use pd.concat instead of DataFrame.append
    all_data = pd.concat(all_data_list, ignore_index=True)
    errors = pd.concat(errors_list, ignore_index=True)
    
    # Reshape the data to have a separate column for each channel
    all_data.columns = ['Channel_' + str(i) for i in range(16)] + ['ADC_Threshold']
    errors.columns = ['Channel_' + str(i) for i in range(16)]  # Errors don't need the ADC_Threshold column
    all_data = all_data.melt(id_vars='ADC_Threshold', var_name='Channel', value_name='Frequency')
    errors = errors.melt(var_name='Channel', value_name='Error')
    
    #channel_data = all_data[all_data['Channel'] == 'Channel_0']
    
    # Step 4: Plotting
    for i in range(1):
        channel_data = all_data[all_data['Channel'] == f'Channel_{i}']
        channel_errors = errors[errors['Channel'] == f'Channel_{i}']
        #plt.errorbar(channel_data['ADC_Threshold'], channel_data['Frequency'],yerr=channel_errors['Error'], fmt='o', label=f'Channel {i}')
        print(i, cmap(idx))
        plt.errorbar(channel_data['ADC_Threshold'], channel_data['Frequency'],label=f'o.v. = {vol-46.5}V', fmt='o-', color=cmap(idx))
        #plt.scatter(channel_data['ADC_Threshold'], channel_data['Frequency'], label=f'{vol}V')

plt.ylim(1e-2,1e7)
plt.xlabel('ADC Threshold')
plt.ylabel('Rate (Hz / 144mm$^2$)')
#plt.title('Frequency over Different ADC Threshold')
plt.title(' ')
plt.legend()
plt.yscale('log')
plt.grid(axis='y')
plt.savefig(f"{root_name}.pdf")
