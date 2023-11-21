import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Read the CSV file
led_df = pd.read_csv('led_width.csv', header=None, names=["Wavelength", "Intensity"])

# Step 1: Create an interpolation function
interpolation_func = interp1d(led_df['Wavelength'], led_df['Intensity'], kind='linear')

# Step 2: Create a regular interval for wavelength
min_wavelength = led_df['Wavelength'].min()
max_wavelength = led_df['Wavelength'].max()
print(min_wavelength, max_wavelength)
regular_wavelength = np.linspace(min_wavelength, max_wavelength, num=100)  # Adjust the number as needed

# Step 3: Apply the interpolation
interpolated_intensity = interpolation_func(regular_wavelength)

# Step 4: Normalize the interpolated data
normalized_intensity = interpolated_intensity / interpolated_intensity.sum()

# Plotting the normalized PDF
plt.plot(regular_wavelength, normalized_intensity)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Intensity')
plt.title('Interpolated and Normalized LED Intensity Distribution PDF')
plt.show()

# Function to read and process PDE data (adapted from your script)
def read_and_process_pde_csv(file_path):
    df = pd.read_csv(file_path, header=None, names=["Wavelength", "PDE"])
    df['Wavelength'] = (df['Wavelength'] / 2).round() * 2
    return df

# Read and process the PDE data
pde_df = read_and_process_pde_csv('s13360-6075CS.csv')  # Replace with your PDE data file
# Print the average PDE
# Interpolate PDE data over the same wavelength range as LED data
pde_interpolation_func = interp1d(pde_df['Wavelength'], pde_df['PDE'], kind='linear', bounds_error=False, fill_value=0)
interpolated_pde = pde_interpolation_func(regular_wavelength)

# Weighted average of PDE
weighted_average_pde = np.sum(normalized_intensity * interpolated_pde) / np.sum(normalized_intensity)

# Print the weighted average PDE
print("single PDE:",  pde_interpolation_func(404))
print("Weighted Average PDE:", weighted_average_pde)

