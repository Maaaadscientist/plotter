import os
import sys
import ROOT
import matplotlib.pyplot as plt

# Function to list all ROOT files in a given directory
def list_root_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.root')]

# Function to extract over-voltage from filename
def extract_over_voltage(filename):
    return float(filename.split('_')[1].replace('.root', ''))
# Function to fit histogram and calculate energy resolution
def fit_histogram_and_calculate_resolution(hist):
    hist.Fit("gaus")
    fit = hist.GetFunction("gaus")
    mean = fit.GetParameter(1)
    sigma = fit.GetParameter(2)
    return sigma / mean


# Specify the directory path here
directory_path = sys.argv[1]

# List all ROOT files in the directory
root_files = list_root_files(directory_path)
# Initialize lists to store over-voltages and energy resolutions
over_voltages = []
energy_resolutions_charge = []
energy_resolutions_dcr = []

# Loop through each file
for file in root_files:
    full_path = os.path.join(directory_path, file)

    # Open the ROOT file
    f = ROOT.TFile(full_path, "READ")

    # Extract the "hist_charge" histogram and calculate its energy resolution
    hist_charge = f.Get("hist_charge")
    resolution_charge = fit_histogram_and_calculate_resolution(hist_charge)

    # Extract the "hist_dcr" histogram and calculate its energy resolution
    hist_dcr = f.Get("hist_dcr")
    resolution_dcr = fit_histogram_and_calculate_resolution(hist_dcr)

    # Store the over-voltage and energy resolutions
    over_voltage = extract_over_voltage(file)
    over_voltages.append(over_voltage)
    energy_resolutions_charge.append(resolution_charge)
    energy_resolutions_dcr.append(resolution_dcr)

    # Close the file
    f.Close()

# Plotting
plt.scatter(over_voltages, energy_resolutions_charge, label='Charge Histogram')
plt.scatter(over_voltages, energy_resolutions_dcr, label='DCR Histogram')
plt.xlabel("Over Voltage (V)")
plt.ylabel("Energy Resolution")
plt.title("Energy Resolution vs. Over Voltage")
plt.legend()
plt.grid(True)
plt.show()
