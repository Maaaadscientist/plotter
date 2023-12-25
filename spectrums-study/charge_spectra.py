import os
import sys
import re
import ROOT
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, gaussian
from scipy.ndimage import gaussian_filter1d
import scienceplots
plt.style.use('science')
plt.style.use('nature')

labelsize=28
titlesize=40
textsize=24
size_marker = 100

# Set global font sizes
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
def fetch_file_info(filename):
    pattern_name = r'(\w+)_run_(\w+)_ov_(\d+).00_sipmgr_(\d+)_(\w+)'
    name_match = re.match(pattern_name, filename)
    if name_match:
        return {
            'run': str(name_match.group(2)),
            'ov': int(name_match.group(3)),
            'channel': int(name_match.group(4)),
            'sipm_type': name_match.group(5)
        }
    else:
        return {}
# Open the ROOT file
input_path = os.path.abspath(sys.argv[1])
csv_path = os.path.abspath(sys.argv[2])

file_info = fetch_file_info(input_path.split("/")[-1])
run = int(file_info['run'])
vol = int(file_info['ov'])
channel = int(file_info['channel'])

pos = 5
df = pd.read_csv(csv_path)
channel_data = df[(df['pos'] == pos) & (df['run'] == run) & (df['vol'] == vol) & (df['ch'] == channel)]

file = ROOT.TFile(input_path, "READ")
hist_name = f"waveform_ch{pos}"
hist = file.Get(hist_name)
# Find the corresponding bin numbers
min_bin = hist.GetXaxis().FindBin(0)
max_bin = hist.GetXaxis().FindBin(2009)

# Loop over the specified range of x-bins (overvoltages)
value_list = []
err_list = []
x_list = []
for i in range(min_bin, max_bin + 1):
    # Get the overvoltage value for the current bin
    x = hist.GetXaxis().GetBinLowEdge(i)
    x_list.append(round(x,1))
    # Project the y-values for this x-bin (overvoltage) to a 1D histogram
    hist_name = f"hist1D_{i}"
    hist1D = hist.ProjectionY(hist_name, i, i, "e")

    # Fit the 1D histogram with a Gaussian
    fit_result = hist1D.Fit("gaus", "S")

    # Extract the parameters (mean and standard deviation)
    mean = fit_result.Parameter(1)
    std_dev = fit_result.Parameter(2)

    value_list.append(mean)
    err_list.append(std_dev)

# Remember to close the ROOT file
baseline = np.array(value_list[200:900]).mean()
baseline_fl = np.array(value_list[200:900]).var()
baseline_sigma = (np.array(err_list[200:900])**2).mean()
baseline_err = np.sqrt(baseline_fl + baseline_sigma)

# Load the TTree
tree = uproot.open(f"{input_path}:signal")
# Convert branch to numpy array
for branch in ['sigQ_ch', 'baselineQ_ch']:
    data = tree[f"{branch}{pos}"].array(library="np")
    amp_max = data.max()
    amp_min = data.min()
    #nBins = 800
    amp_step = 0.5#(amp_max - amp_min) / nBins
    nBins = int((amp_max - amp_min) / amp_step)
    print(nBins)
    amp_gain = 42/ 2 / 5.28 * (vol + 2)
    
    # Creating a histogram from the 'data' array
    hist_counts, bin_edges = np.histogram(data, bins=nBins)
    
    # The 'hist_counts' represents the frequency of data in each bin
    # 'bin_edges' represents the edges of the bins
    
    # For peak finding, we'll work with the 'hist_counts' as our y-values (frequencies)
    # and the bin centers as our x-values.
    
    # Calculating bin centers from bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Now 'bin_centers' and 'hist_counts' together form the spectrum of the histogram
    spectrum_x = bin_centers
    spectrum_y = hist_counts
    
    # Returning the x and y values of the histogram spectrum for further analysis
    # Now 'array' is a NumPy array
    
    # Smooth the data
    smoothed_data = gaussian_filter1d(spectrum_y, sigma=2)
    
    # Find peaks
    peaks, properties = find_peaks(smoothed_data, prominence=0.4, distance=amp_gain/amp_step, height=10)
    
    # Returning the peak positions for further analysis
    peak_positions = spectrum_x[peaks]
    
    print("Detected peak positions:", peak_positions)
    peak_diff = []
    for i,peak in enumerate(peak_positions):
        if i < len(peak_positions) - 1:
            single_gain = peak_positions[i+1] - peak
            peak_diff.append(single_gain)
    gains = np.array(peak_diff)
    if branch != 'baselineQ_ch':
        avg_gain = gains.mean()
        avg_gain_err = np.sqrt(gains.var())
    else:
        avg_gain = 0 #gains.mean()
        avg_gain_err = 0 # np.sqrt(gains.var())
    
    # Assuming there's only one row that matches, otherwise you might need to handle multiple rows
    if not channel_data.empty:
        #baseline_position = channel_data['bl'].iloc[0]  # .iloc[0] accesses the first row of the filtered data
        baseline_rms = channel_data['bl_rms'].iloc[0] * 45
        baseline_position = baseline * 45#channel_data['bl'].iloc[0]  # .iloc[0] accesses the first row of the filtered data
        mu =  channel_data['mu'].iloc[0]
        ref_mu =  channel_data['ref_mu'].iloc[0]
        ref_mu_err =  channel_data['ref_mu_err'].iloc[0]
        mu_err =  channel_data['mu_err'].iloc[0]
        lambda_ = channel_data['lambda'].iloc[0]
        lambda_err = channel_data['lambda_err'].iloc[0]
        gain = channel_data['gain'].iloc[0]
        n_peaks = channel_data['n_peaks'].iloc[0]
        events = channel_data['events'].iloc[0]
        dcr = channel_data['dcr'].iloc[0]
        dcr_err = channel_data['dcr_err'].iloc[0]
        
        batch = channel_data['batch'].iloc[0]
        box = channel_data['box'].iloc[0]
        tsn = channel_data['tsn'].iloc[0]
    
        vbd = channel_data['vbd'].iloc[0]
        vbd_err = channel_data['vbd_err'].iloc[0]
        over_vol = channel_data['ov'].iloc[0]
    
    else:
        print(f"No data found for Channel {selected_channel}")
    # Plotting the results
    #plt.plot(spectrum_x, spectrum_y, label='Original Data', alpha=0.5)
    param_text = (f"SN: {batch}-{box}-{int(tsn)}-{channel}\n"
                  "Events:"+f" {events}\n"
                  "$\\mathrm{Q}_\\mathrm{\\,baseline}$:"+f" {baseline_position:.2f} (pC)\n"
                  "$\\sigma^\\mathrm{Q}_\\mathrm{\\,baseline}$:"+f" {baseline_rms:.2f} (pC)\n"
                  "Recognised Peaks :"+f" {len(peak_positions)}\n"
                  "\n"
                  #"$\\mathrm{V}_\\mathrm{bd}$ :"+f" {vbd:.2f}"+" $\\pm$ "+f"{vbd_err:.3f} (V)\n"
                  "$\\mathrm{V}_\\mathrm{preset}$ :" +f" {vol} (V)\n"
                  "$\\mathrm{V}_\\mathrm{bias}$ :"+f" {over_vol:.2f}"+" $\\pm$ "+f"{vbd_err:.2f} (V)")

                  #"$\\mu$ :"+f" {mu:.3f}"+" $\\pm$ "+f"{mu_err:.3f}\n"
                  #"$\\mu_\\mathrm{ref.}$ :"+f" {ref_mu:.3f}"+" $\\pm$ "+f"{ref_mu_err:.3f}\n"
                  #"$\\lambda$ :"+f" {lambda_:.3f}"+" $\\pm$ "+f"{lambda_err:.3f}\n"
                  #"DCR:"+f" {dcr:.1f}"+" $\\pm$ "+f"{dcr_err:.1f}"+" ($\\mathrm{Hz}/\\mathrm{mm}^2$)\n"
                  #"Charge Gain :"+f" {avg_gain:.2f}"+" $\\pm$ "+f"{avg_gain_err:.2f} (pC)\n"
    if branch != 'baselineQ_ch':
        param_text += "\nCharge Gain :"+f" {avg_gain:.2f} (pC)"
    
    # Calculate the statistical error (standard deviation) for each data point
    errors = np.sqrt(spectrum_y)
    
    # Create an error bar plot
    plt.errorbar(spectrum_x, spectrum_y, yerr=errors, fmt='o', color='black', ecolor='black', label='Original Data', alpha = 0.7, zorder = 1)
    
    plt.text(0.78, 0.75, param_text, transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=textsize)
    plt.plot(spectrum_x, smoothed_data, label='Smoothed Data', color='orange', zorder=2)
    plt.scatter(spectrum_x[peaks], smoothed_data[peaks], color='red', marker='o', label='Detected Peaks', s=size_marker, zorder=3)
    if branch == "baselineQ_ch":
        title_suffix = "Baseline Range"
    else:
        title_suffix = "LED Signal Range"
    plt.title(f'Charge Spectrum ({title_suffix})')
    plt.legend()
    # Set the yscale to 'log'
    plt.yscale('log')
    plt.xlabel('Accumulated Charge (pC)')
    plt.ylabel('Number of Events')
    plt.savefig(f"run{run}_sipm{channel}_{branch}{pos}.pdf")
    plt.clf()

