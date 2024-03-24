import ROOT
import matplotlib.pyplot as plt
import re
import os
import numpy as np

def list_root_files(directory, pattern):
    return [f for f in os.listdir(directory) if f.endswith(f'{pattern}.root')]

def extract_energy(filename):
    """ Extract energy value from the file name. """
    match = re.search(r'e_(\d+\.\d+)_', filename)
    if match:
        return float(match.group(1))
    return None
def fit_histogram(file, hist_name):
    """ Fit the specified histogram in the given file and return the mean and sigma. """
    hist = file.Get(hist_name)
    print(hist)
    if not hist:
        print(f"Histogram {hist_name} not found in file {file.GetName()}")
        return None, None

    # Perform Gaussian fit
    hist.Fit("gaus")
    fit_result = hist.GetFunction("gaus")
    
    mean = fit_result.GetParameter(1)
    sigma = fit_result.GetParameter(2)

    return mean, sigma

def calculate_resolution(means, sigmas):
    """ Calculate the decomposed energy resolutions. """
    resolutions = {}
    for key in means.keys():
        if key != "hist_charge":
            next_key = list(means.keys())[list(means.keys()).index(key)+1]
            resolution = np.sqrt(sigmas[key]**2 - sigmas[next_key]**2) / means[next_key]
            resolutions[key] = resolution
    return resolutions

def main():

    directory_path = "energy_merged"
    root_files = list_root_files(directory_path, "3.8")

    hist_names = ["hist_LS", "hist_PTE", "hist_PDE", "hist_dcr", "hist_ct", "hist_ap", "hist_charge"]

    energy_resolutions = {name: [] for name in hist_names}
    energies = []

    for file_name in root_files:
        file = ROOT.TFile(f"{directory_path}/{file_name}", "READ")
        energy = extract_energy(file_name)
        if energy is None:
            continue

        means, sigmas = {}, {}
        for hist_name in hist_names:
            print(file_name,hist_name)
            mean, sigma = fit_histogram(file, hist_name)
            if mean is not None and sigma is not None:
                means[hist_name] = mean
                sigmas[hist_name] = sigma

        # Calculate decomposed resolutions
        resolutions = calculate_resolution(means, sigmas)
        for key, res in resolutions.items():
            energy_resolutions[key].append(res)

        energies.append(energy)

    # Plotting
    plt.figure(figsize=(10, 6))
    for key, res_values in energy_resolutions.items():
        if key != "hist_charge":  # Exclude the total resolution from individual plots
            plt.scatter(energies, res_values, label=key.replace("hist_", ""))

    plt.xlabel('Energy (GeV)')
    plt.ylabel('Resolution (sigma/mean)')
    plt.title('Energy Resolution Decompositions vs Energy')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
