import pandas as pd
import matplotlib.pyplot as plt
import glob

# Step 1: Define the ADC thresholds
adc_thresholds = range(0, 1126, 25)  # Adjust the range if necessary
all_data_list = []  # List to hold individual DataFrames

# Step 2: Load the data from each file and append it to the list
for threshold in adc_thresholds:
    filename = f'data_{threshold}.csv'
    temp_df = pd.read_csv(filename, header=None)
    #temp_df['ADC_Threshold'] = threshold  # Add a column for the ADC Threshold
    #all_data_list.append(temp_df)
    # Calculate the mean for each channel across the 6 measurements
    mean_data = temp_df.mean().to_frame().T  # Transpose to get the means as a row
    mean_data['ADC_Threshold'] = threshold  # Add a column for the ADC Threshold
    all_data_list.append(mean_data)

# Use pd.concat instead of DataFrame.append
all_data = pd.concat(all_data_list, ignore_index=True)

# Step 3: Reshape the data to have a separate column for each channel
all_data.columns = ['Channel_' + str(i) for i in range(16)] + ['ADC_Threshold']
all_data = all_data.melt(id_vars='ADC_Threshold', var_name='Channel', value_name='Frequency')
#channel_data = all_data[all_data['Channel'] == 'Channel_0']

# Step 4: Plotting
plt.figure(figsize=(15, 10))
for i in range(4):
    channel_data = all_data[all_data['Channel'] == f'Channel_{i}']
    plt.scatter(channel_data['ADC_Threshold'], channel_data['Frequency'], label=f'Channel {i}')

plt.xlabel('ADC Threshold')
plt.ylabel('Frequency')
plt.title('ADC Threshold vs Frequency for 16 Channels')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()
