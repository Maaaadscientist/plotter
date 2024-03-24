import os
import ROOT
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial
import scienceplots
plt.style.use('science')
plt.style.use('nature')

labelsize=28
titlesize=40
textsize=24
size_marker = 30
# Function to list all ROOT files in a given directory
def extract_overvoltage_and_pct(data_str):
    rows = data_str.strip().split("\n")
    overvoltage_ct_list = []
    for row in rows:
        values = row.split()
        overvoltage = float(values[0])
        P_ct = float(values[1])
        overvoltage_ct_list.append((overvoltage, P_ct))
    return overvoltage_ct_list


def borel_expected_value(lambda_):
    """Calculate the expected value of the Borel distribution."""
    # Since calculating the expected value analytically is complex,
    # we use a numerical approximation.
    k_values = np.arange(1, 100)
    pmf_values = (lambda_ * k_values)**(k_values - 1) * np.exp(-k_values * lambda_) / factorial(k_values)
    expected_value = np.sum(k_values * pmf_values)
    return expected_value

def borel_variance(lambda_):
    """Calculate the variance of the Borel distribution."""
    k_values = np.arange(1, 100)
    pmf_values = (lambda_ * k_values)**(k_values - 1) * np.exp(-k_values * lambda_) / factorial(k_values)
    mean = borel_expected_value(lambda_)
    variance = np.sum((k_values**2) * pmf_values) - mean**2
    return variance

def calculate_resolution_single(overvoltage, P_ct, N):
    """Calculate the resolution for N Borel events for a single overvoltage point."""
    lambda_ = np.log(1 / (1 - P_ct))
    EX = borel_expected_value(lambda_)
    DX = borel_variance(lambda_)
    mean = N * EX
    sigma = np.sqrt(DX * N)
    resolution = sigma / mean
    return resolution

data_str = """
2.5 0.08956
2.6 0.09249
2.7 0.09582
2.8 0.09954
2.9 0.10365
3.0 0.10819
3.1 0.11314
3.2 0.11843
3.3 0.12397
3.4 0.12970
3.5 0.13556
3.6 0.14149
3.7 0.14740
3.8 0.15326
3.9 0.15904
4.0 0.16471
4.1 0.17025
4.2 0.17566
4.3 0.18093
4.4 0.18607
4.5 0.19108
4.6 0.19599
4.7 0.20082
4.8 0.20559
4.9 0.21032
5.0 0.21505
5.1 0.21979
5.2 0.22457
5.3 0.22941
5.4 0.23431
5.5 0.23928
5.6 0.24432
5.7 0.24943
5.8 0.25460
5.9 0.25982
6.0 0.26508
6.1 0.27038
6.2 0.27569
6.3 0.28099
6.4 0.28628
6.5 0.29150
6.6 0.29661
6.7 0.30160
6.8 0.30640
6.9 0.31099
7.0 0.31528
"""

overvoltage_ct_list = extract_overvoltage_and_pct(data_str)
print(overvoltage_ct_list)

def list_root_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.root')]

# Function to extract over-voltage from filename
def extract_over_voltage(filename):
    return round(float(filename.split('_')[1].replace('.root', '')),1)

# Function to fit histogram and get mean and sigma
def fit_histogram(hist):
    hist.Fit("gaus")
    fit = hist.GetFunction("gaus")
    mean = fit.GetParameter(1)
    sigma = fit.GetParameter(2)
    return mean, sigma
# Function to calculate energy resolution
def calculate_energy_resolution(hist_target, key, hist_prev=None, breakdown=False):
    mean, sigma = fit_histogram(hist_target)
    if key == "max":
        mean /= 0.44
    else:
        mean /= 0.47
    if not breakdown:
        return sigma / mean 
    elif not hist_prev is None:
        mean_prev, sigma_prev = fit_histogram(hist_prev)
        if key == "max":
            mean_prev /= 0.44
        else:
            mean_prev /= 0.47
        return np.sqrt(sigma**2 - sigma_prev**2) / mean_prev
    else:
        raise ValueError("hist_prev is not provided")

# Specify the directory path here
directory_path = 'merged'
#directory_path = 'old_pde_merged'

# Process 'merged_max.root' and 'merged_typical.root'
special_files = ['merged_max.root', 'merged_typical.root']
special_resolutions = {'max': {}, 'typical': {}}

for s_file in special_files:
    full_path = os.path.join(directory_path, s_file)
    f = ROOT.TFile(full_path, "READ")

    key = 'max' if 'max' in s_file else 'typical'
    resolutions = calculate_energy_resolution(f.Get("hist_ct"), key, f.Get("hist_PDE"), True)
    special_resolutions[key] = {'ct': resolutions, 'charge': resolutions}

    f.Close()

# List all ROOT files in the directory
root_files = list_root_files(directory_path)

# Initialize lists to store over-voltages and energy resolutions
over_voltages = []
energy_resolutions_ct = []

# Initialize lists to store the data
ov_list = []
pde_list = []
pde_err_list = []

# Open the file and read line by line
with open('pde.txt', 'r') as file:
    for line in file:
        # Split each line into parts
        parts = line.split()

        # Assuming each line has three space-separated values
        if len(parts) == 3:
            # Append the values to the respective lists
            ov_list.append(float(parts[0]))
            pde_list.append(float(parts[1]))
            pde_err_list.append(float(parts[2]))

for i,file in enumerate(root_files):
    if "max" in file:
        continue
    if "typical" in file:
        continue
    full_path = os.path.join(directory_path, file)

    # Open the ROOT file
    f = ROOT.TFile(full_path, "READ")

    # Process "hist_ap" and "hist_ct" to calculate correction factor
    hist_ct = f.Get("hist_ct")
    mean_ct, sigma_ct = fit_histogram(hist_ct)
    hist_pde = f.Get("hist_PDE")
    mean_pde, sigma_pde = fit_histogram(hist_pde)


    over_voltage = extract_over_voltage(file)
    index = ov_list.index(over_voltage)
    pde = pde_list[index]
    mean_pde /= pde
    mean_ct /= pde
    energy_resolution_ct = np.sqrt(sigma_ct**2 - sigma_pde**2) / mean_ct
    # Process "hist_ct" and "hist_charge" using the correction factor for their means


    # Store the over-voltage and energy resolutions
    over_voltages.append(over_voltage)
    energy_resolutions_ct.append(energy_resolution_ct)

    # Close the file
    f.Close()


# Add horizontal lines for 'merged_max.root' and 'merged_typical.root'
plt.figure(figsize=(20, 15))

# Plotting
plt.scatter(over_voltages, energy_resolutions_ct, label='Crosstalk',s=size_marker, marker='o', color='mediumblue')
#plt.scatter(over_voltages, energy_resolutions_ct, label='DCR',s=size_marker)
plt.axhline(y=special_resolutions['max']['charge'], color='r', linestyle='--', label='Max', linewidth=2)

plt.axhline(y=special_resolutions['typical']['charge'], color='darkviolet', linestyle='--', label='Typical', linewidth=2)
#plt.scatter(over_voltages, energy_resolutions_pte, label='PTE',s=size_marker)
#plt.scatter(over_voltages, energy_resolutions_LS, label='LS',s=size_marker)
plt.ylim(0.0,0.01)
plt.xticks(fontsize=labelsize)
plt.yticks(fontsize=labelsize)
plt.xlabel("Over Voltage (V)", fontsize=labelsize, labelpad=15)
plt.ylabel("Energy Resolution", fontsize=labelsize, labelpad=15)
plt.title("Energy Resolution vs. Over Voltage", fontsize=titlesize, pad=20)
plt.legend(loc='upper right',ncol=2,fontsize=labelsize )
plt.grid(True)
plt.savefig("energy_res_ct.pdf")
