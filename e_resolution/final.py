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
size_marker = 50
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
def calculate_energy_resolution(hist_pde, hist_dcr, hist_charge):
    mean_pde, sigma_pde = fit_histogram(hist_pde)
    mean_dcr, sigma_dcr = fit_histogram(hist_dcr)
    mean_charge, sigma_charge = fit_histogram(hist_charge)

    correction_factor = mean_pde / mean_dcr if mean_dcr != 0 else 1
    energy_resolution_charge = sigma_charge  / (mean_charge * correction_factor)
    energy_resolution_pde = sigma_pde / mean_pde

    return energy_resolution_pde, energy_resolution_charge


# Specify the directory path here
directory_path = 'merged'
#directory_path = 'old_pde_merged'

# Process 'merged_max.root' and 'merged_typical.root'
special_files = ['merged_max.root', 'merged_typical.root']
special_resolutions = {'max': {}, 'typical': {}}

for s_file in special_files:
    full_path = os.path.join(directory_path, s_file)
    f = ROOT.TFile(full_path, "READ")

    resolutions = calculate_energy_resolution(f.Get("hist_PDE"), f.Get("hist_dcr"), f.Get("hist_charge"))
    key = 'max' if 'max' in s_file else 'typical'
    special_resolutions[key] = {'ct': resolutions[0], 'charge': resolutions[1]}

    f.Close()

# List all ROOT files in the directory
root_files = list_root_files(directory_path)

# Initialize lists to store over-voltages and energy resolutions
over_voltages = []
energy_resolutions_charge = []
energy_resolutions_dcr = []
energy_resolutions_ap = []
energy_resolutions_ct = []
energy_resolutions_ct_analytical = []
energy_resolutions_pde = []
energy_resolutions_pde_sqrtN = []
energy_resolutions_pde_analytical = []
energy_resolutions_pte = []
energy_resolutions_LS = []
energy_resolutions_ct_dec = []
energy_resolutions_gain_nuf3_uni = []
energy_resolutions_gain_nuf5_uni = []
energy_resolutions_gain_nuf10_uni = []
energy_resolutions_gain_nuf15_uni = []
energy_resolutions_gain_nuf3_gas = []
energy_resolutions_gain_nuf5_gas = []
energy_resolutions_gain_nuf10_gas = []
energy_resolutions_gain_nuf15_gas = []
energy_resolutions_final = []

N_pe = []
overvoltage_ct_list = extract_overvoltage_and_pct(data_str)
# Loop through each file
# Sorting the list in ascending order by overvoltage
overvoltage_ct_list = sorted(overvoltage_ct_list, key=lambda x: x[0])

for i,file in enumerate(root_files):
    if "max" in file:
        continue
    if "typical" in file:
        continue
    full_path = os.path.join(directory_path, file)

    # Open the ROOT file
    f = ROOT.TFile(full_path, "READ")

    # Process "hist_ap" and "hist_dcr" to calculate correction factor
    hist_ap = f.Get("hist_ap")
    mean_ap, sigma_ap = fit_histogram(hist_ap)

    hist_charge = f.Get("hist_charge")
    mean_charge, sigma_charge = fit_histogram(hist_charge)
    hist_dcr = f.Get("hist_dcr")
    mean_dcr, sigma_dcr = fit_histogram(hist_dcr)
    hist_ct = f.Get("hist_ct")
    mean_ct, sigma_ct = fit_histogram(hist_ct)
    hist_pte = f.Get("hist_PTE")
    mean_pte, sigma_pte = fit_histogram(hist_pte)
    hist_pde = f.Get("hist_PDE")
    mean_pde, sigma_pde = fit_histogram(hist_pde)
    hist_ls = f.Get("hist_LS")
    mean_ls, sigma_ls = fit_histogram(hist_ls)
    hist_gain_nuf3_uni = f.Get("hist_charge_nuf3_uni")
    mean_gain_nuf3_uni, sigma_gain_nuf3_uni = fit_histogram(hist_gain_nuf3_uni)
    hist_gain_nuf5_uni = f.Get("hist_charge_nuf5_uni")
    mean_gain_nuf5_uni, sigma_gain_nuf5_uni = fit_histogram(hist_gain_nuf5_uni)
    hist_gain_nuf10_uni = f.Get("hist_charge_nuf10_uni")
    mean_gain_nuf10_uni, sigma_gain_nuf10_uni = fit_histogram(hist_gain_nuf10_uni)
    hist_gain_nuf15_uni = f.Get("hist_charge_nuf15_uni")
    mean_gain_nuf15_uni, sigma_gain_nuf15_uni = fit_histogram(hist_gain_nuf15_uni)
    hist_gain_nuf3_gas = f.Get("hist_charge_nuf3_gas")
    mean_gain_nuf3_gas, sigma_gain_nuf3_gas = fit_histogram(hist_gain_nuf3_gas)
    hist_gain_nuf5_gas = f.Get("hist_charge_nuf5_gas")
    mean_gain_nuf5_gas, sigma_gain_nuf5_gas = fit_histogram(hist_gain_nuf5_gas)
    hist_gain_nuf10_gas = f.Get("hist_charge_nuf10_gas")
    mean_gain_nuf10_gas, sigma_gain_nuf10_gas = fit_histogram(hist_gain_nuf10_gas)
    hist_gain_nuf15_gas = f.Get("hist_charge_nuf15_gas")
    mean_gain_nuf15_gas, sigma_gain_nuf15_gas = fit_histogram(hist_gain_nuf15_gas)


    over_voltage = extract_over_voltage(file)
    pct = 0
    for i in range(len(overvoltage_ct_list)):
        if round(overvoltage_ct_list[i][0],1) == round(over_voltage,1):
            pct = overvoltage_ct_list[i][1]
    energy_resolution_ct_analytical = calculate_resolution_single(over_voltage, pct, mean_pde)
    N_pe.append(mean_pde)
    print(over_voltage, energy_resolution_ct_analytical)
    energy_resolution_ls = sigma_ls / mean_ls
    energy_resolution_pte = sigma_pte / mean_pte
    energy_resolution_pde = sigma_pde / mean_pde
    energy_resolution_pde_sqrtN = np.sqrt(mean_pde) / mean_pde
    energy_resolution_pde_analytical = np.sqrt(mean_pde * (1 - mean_pde/ mean_pte)) / mean_pde
    energy_resolution_ct = sigma_ct / mean_ct
    energy_resolution_ct_dec = np.sqrt(sigma_ct**2 - sigma_pde**2) / mean_ct
    energy_resolution_ap = sigma_ap / mean_ap
    energy_resolution_dcr = sigma_dcr / mean_dcr
    # Process "hist_dcr" and "hist_charge" using the correction factor for their means

    energy_resolution_charge = sigma_charge  / mean_charge
    energy_resolution_gain_nuf3_uni = np.sqrt(sigma_gain_nuf3_uni**2 - sigma_charge**2)  / mean_gain_nuf3_uni
    energy_resolution_gain_nuf5_uni = np.sqrt(sigma_gain_nuf5_uni**2 - sigma_charge**2)  / mean_gain_nuf5_uni
    energy_resolution_gain_nuf10_uni = np.sqrt(sigma_gain_nuf10_uni**2 - sigma_charge**2)  / mean_gain_nuf10_uni
    energy_resolution_gain_nuf15_uni = np.sqrt(sigma_gain_nuf15_uni**2 - sigma_charge**2)  / mean_gain_nuf15_uni
    energy_resolution_gain_nuf3_gas = np.sqrt(sigma_gain_nuf3_gas**2 - sigma_charge**2)  / mean_gain_nuf3_gas
    energy_resolution_gain_nuf5_gas = np.sqrt(sigma_gain_nuf5_gas**2 - sigma_charge**2)  / mean_gain_nuf5_gas
    energy_resolution_gain_nuf10_gas = np.sqrt(sigma_gain_nuf10_gas**2 - sigma_charge**2)  / mean_gain_nuf10_gas
    energy_resolution_gain_nuf15_gas = np.sqrt(sigma_gain_nuf15_gas**2 - sigma_charge**2)  / mean_gain_nuf15_gas
    energy_resolution_final = sigma_gain_nuf15_gas  / mean_gain_nuf15_gas
    print(energy_resolution_gain_nuf3_uni)
    energy_resolutions_gain_nuf3_uni.append(energy_resolution_gain_nuf3_uni)
    energy_resolutions_gain_nuf5_uni.append(energy_resolution_gain_nuf5_uni)
    energy_resolutions_gain_nuf10_uni.append(energy_resolution_gain_nuf10_uni)
    energy_resolutions_gain_nuf15_uni.append(energy_resolution_gain_nuf15_uni)
    energy_resolutions_gain_nuf3_gas.append(energy_resolution_gain_nuf3_gas)
    energy_resolutions_gain_nuf5_gas.append(energy_resolution_gain_nuf5_gas)
    energy_resolutions_gain_nuf10_gas.append(energy_resolution_gain_nuf10_gas)
    energy_resolutions_gain_nuf15_gas.append(energy_resolution_gain_nuf15_gas)
    #energy_resolution_charge = sigma_charge  / mean_charge 

    energy_resolutions_final.append(energy_resolution_final)

    # Store the over-voltage and energy resolutions
    over_voltages.append(over_voltage)
    energy_resolutions_charge.append(energy_resolution_charge)
    energy_resolutions_dcr.append(energy_resolution_dcr)
    energy_resolutions_ap.append(energy_resolution_ap)
    energy_resolutions_ct.append(energy_resolution_ct)
    energy_resolutions_ct_dec.append(energy_resolution_ct_dec)
    energy_resolutions_ct_analytical.append(energy_resolution_ct_analytical)
    energy_resolutions_pte.append(energy_resolution_pte)
    energy_resolutions_pde.append(energy_resolution_pde)
    energy_resolutions_pde_sqrtN.append(energy_resolution_pde_sqrtN)
    energy_resolutions_pde_analytical.append(energy_resolution_pde_analytical)
    energy_resolutions_LS.append(energy_resolution_ls)

    # Close the file
    f.Close()


# Add horizontal lines for 'merged_max.root' and 'merged_typical.root'
plt.figure(figsize=(20, 15))

# Plotting
plt.scatter(over_voltages, energy_resolutions_LS, label='LS',s=size_marker, marker='o', color='magenta')
plt.scatter(over_voltages, energy_resolutions_pte, label='LS+PTE',s=size_marker, marker='o', color='darkslateblue')
plt.scatter(over_voltages, energy_resolutions_pde, label='LS+PTE+PDE',s=size_marker, marker='o', color='firebrick')
plt.scatter(over_voltages, energy_resolutions_dcr, label='LS+PTE+PDE+DCR',s=size_marker, marker='o', color='mediumblue')
plt.scatter(over_voltages, energy_resolutions_ct, label='LS+PTE+PDE+DCR+CT',s=size_marker, marker='o',color='orange')
plt.scatter(over_voltages, energy_resolutions_ap, label='LS+PTE+PDE+DCR+CT+AP',s=size_marker, marker='o',color='lawngreen')
plt.scatter(over_voltages, energy_resolutions_charge, label='LS+PTE+PDE+DCR+CT+AP+Q_RES',s=size_marker, marker='o', color='dimgrey')
plt.scatter(over_voltages, energy_resolutions_final, label='LS+PTE+PDE+DCR+CT+AP+Q_RES+NUF0.3V',s=size_marker, marker='*', color='gold')
#plt.scatter(over_voltages, energy_resolutions_dcr, label='DCR',s=size_marker)
plt.axhline(y=special_resolutions['max']['charge'], color='r', linestyle='--', label='Max', linewidth=2)

plt.axhline(y=special_resolutions['typical']['charge'], color='darkviolet', linestyle='--', label='Typical', linewidth=2)
plt.ylim(0.005,0.024)
plt.xticks(fontsize=labelsize)
plt.yticks(fontsize=labelsize)
plt.xlabel("Over Voltage (V)", fontsize=labelsize, labelpad=15)
plt.ylabel("Energy Resolution ($\\sigma \\mathrm{E} / \\mathrm{E}$)", fontsize=labelsize, labelpad=15)
plt.title("Energy Resolution vs. Over Voltage", fontsize=titlesize, pad=20)
plt.legend(loc='upper right',ncol=2,fontsize=textsize )
plt.grid(True)
plt.savefig("energy_res_final.pdf")
