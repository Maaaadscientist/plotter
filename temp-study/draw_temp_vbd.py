import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('special-runs.csv')

# Filter the data for a specific pos and ch
pos_filter = 10  # Replace with the desired pos value
ch_filter = 5  # Replace with the desired ch value
filtered_df = df[(df['pos'] == pos_filter) & (df['ch'] == ch_filter)]

# Extract the vbd and temp columns
vbd_data = filtered_df['vbd']
temp_data = filtered_df['temp']
print(temp_data)

# Create the plot
plt.figure(figsize=(8, 6))
plt.scatter(temp_data, vbd_data)
plt.xlabel('Temperature (°C)')
plt.ylabel('VBD (V)')
plt.title(f'VBD vs Temperature for pos={pos_filter}, ch={ch_filter}')
plt.grid(True)
plt.show()
