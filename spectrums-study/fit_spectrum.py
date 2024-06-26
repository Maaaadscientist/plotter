#! /opt/homebrew/bin/python3.12
import os
import sys
import re
import ROOT
from math import factorial
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import binom
#from scipy.signal import find_peaks, gaussian
from scipy.signal import find_peaks
from scipy.signal.windows import gaussian
from scipy.ndimage import gaussian_filter1d
import warnings

sigmaAp=0
# Ignore divide by zero warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import scienceplots
plt.style.use('science')
plt.style.use('nature')

scale = 1
labelsize=28
titlesize=40
textsize=24
size_marker = 100

labelsize *= scale
titlesize*= scale
textsize*=scale
size_marker*=scale
# Set global font sizes
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (24,15)
plt.rcParams['font.size'] = textsize  # Sets default font size
plt.rcParams['axes.labelsize'] = labelsize
plt.rcParams['axes.titlesize'] = titlesize
plt.rcParams['xtick.labelsize'] = labelsize
plt.rcParams['ytick.labelsize'] = labelsize
plt.rcParams['legend.fontsize'] = labelsize
plt.rcParams['errorbar.capsize'] = 4
plt.rcParams['lines.markersize'] = 6  # For example, 8 points
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
# Define the functions
def gauss(Q, n, sigma_n, Ped, Gain):
    return np.exp(-((Q - (Ped + n * Gain)) ** 2) / (2 * sigma_n ** 2)) / (np.sqrt(2 * np.pi) * sigma_n)

def gauss_ap(Q, n, i, sigma_n, Ped, Gain, ap_gain):
    return np.exp(-((Q - (Ped + n * Gain + i * ap_gain)) ** 2) / (2 * sigma_n ** 2)) / (np.sqrt(2 * np.pi) * sigma_n)
def poisson(n, mu, lambda_):
    return (mu * (mu + n * lambda_) ** (n - 1) * np.exp(-mu - n * lambda_)) / factorial(n)

def sigma_n(n, sigma0, sigmak):
    return np.sqrt(n * sigmak**2 + sigma0**2)
def sigma_n_i(n, i, sigma0, sigmak, sigmaAp):
    return np.sqrt(n * sigmak**2 + sigma0**2)
def bino(n, i, alpha):
    return 
# Define the PDF
def compound_pdf(Q, Ped, Gain, mu, lambda_, sigma0, sigmak, n_max):
    total_pdf = np.zeros_like(Q)
    for n in range(n_max + 1):
        total_pdf += poisson(n, mu, lambda_) * gauss(Q, n, sigma_n(n, sigma0, sigmak), Ped, Gain)
    return total_pdf

def compound_pdf_ap(Q, Ped, Gain, AP_Gain,  mu, lambda_, alpha, sigma0, sigmak, sigmaAp, n_max):
    total_pdf = np.zeros_like(Q)
    for n in range(n_max + 1):
        for i in range(n+1):
            total_pdf += poisson(n, mu, lambda_) * binom.pmf(i, n, alpha) * gauss_ap(Q, n, i, sigma_n_i(n, i, sigma0, sigmak, sigmaAp), Ped, Gain, AP_Gain)
    ap_pdf = np.zeros_like(Q)
    for n in range(1, n_max + 1):
        for i in range(1,n+1):
            ap_pdf += poisson(n, mu, lambda_) * binom.pmf(i, n, alpha) * gauss_ap(Q, n, i, sigma_n_i(n, i, sigma0, sigmak, sigmaAp), Ped, Gain,  AP_Gain)
    gp_pdf = np.zeros_like(Q)
    for n in range(0, n_max + 1):
        gp_pdf += poisson(n, mu, lambda_) * binom.pmf(0, n, alpha) * gauss_ap(Q, n, 0, sigma_n_i(n, 0, sigma0, sigmak, sigmaAp), Ped, Gain , AP_Gain)
    return total_pdf, ap_pdf, gp_pdf

def geometric( i,n, alpha):
    return (1-alpha*n)*n**i  * alpha**i
# Define the PDF
def geometric_pdf(Q, Ped, Gain, mu, lambda_, sigma0, sigmak, n_max):
    total_pdf = np.zeros_like(Q)
    for n in range(n_max + 1):
        total_pdf += poisson(n, mu, lambda_) * gauss(Q, n, sigma_n(n, sigma0, sigmak), Ped, Gain)
    return total_pdf

def geometric_pdf_ap(Q, Ped, Gain, AP_Gain,  mu, lambda_, alpha, sigma0, sigmak, sigmaAp, n_max):
    total_pdf = np.zeros_like(Q)
    for n in range(n_max + 1):
        for i in range(20):
            total_pdf += poisson(n, mu, lambda_) * geometric(i, n, alpha) * gauss_ap(Q, n, i, sigma_n_i(n, i, sigma0, sigmak, sigmaAp), Ped, Gain, AP_Gain)
    ap_pdf = np.zeros_like(Q)
    for n in range(1, n_max + 1):
        for i in range(1,20):
            ap_pdf += poisson(n, mu, lambda_) * geometric(i, n, alpha) * gauss_ap(Q, n, i, sigma_n_i(n, i, sigma0, sigmak, sigmaAp), Ped, Gain,  AP_Gain)
    gp_pdf = np.zeros_like(Q)
    for n in range(0, n_max + 1):
        gp_pdf += poisson(n, mu, lambda_) * geometric(0, n, alpha) * gauss_ap(Q, n, 0, sigma_n_i(n, 0, sigma0, sigmak, sigmaAp), Ped, Gain , AP_Gain)
    return total_pdf, ap_pdf, gp_pdf
# Open the ROOT file
input_path = os.path.abspath(sys.argv[1])
csv_path = os.path.abspath(sys.argv[2])

file_info = fetch_file_info(input_path.split("/")[-1])
run = int(file_info['run'])
vol = int(file_info['ov'])
channel = int(file_info['channel'])

pos = int(sys.argv[3])
output_path = os.getcwd()
if len(sys.argv) > 4:
    output_path = os.path.abspath(sys.argv[4])
run_path = "run_" + str(run).zfill(4)
df = pd.read_csv(csv_path)
channel_data = df[(df['pos'] == pos) & (df['run'] == run) & (df['vol'] == vol) & (df['ch'] == channel)]

file = ROOT.TFile(input_path, "READ")
#hist_name = f"waveform_ch{pos}"
#hist = file.Get(hist_name)
## Find the corresponding bin numbers
#min_bin = hist.GetXaxis().FindBin(0)
#max_bin = hist.GetXaxis().FindBin(2009)
#
## Loop over the specified range of x-bins (overvoltages)
#value_list = []
#err_list = []
#x_list = []
#for i in range(min_bin, max_bin + 1):
#    # Get the overvoltage value for the current bin
#    x = hist.GetXaxis().GetBinLowEdge(i)
#    x_list.append(round(x,1))
#    # Project the y-values for this x-bin (overvoltage) to a 1D histogram
#    hist_name = f"hist1D_{i}"
#    hist1D = hist.ProjectionY(hist_name, i, i, "e")
#
#    # Fit the 1D histogram with a Gaussian
#    fit_result = hist1D.Fit("gaus", "S")
#
#    # Extract the parameters (mean and standard deviation)
#    mean = fit_result.Parameter(1)
#    std_dev = fit_result.Parameter(2)
#
#    value_list.append(mean)
#    err_list.append(std_dev)

# Remember to close the ROOT file
#baseline = np.array(value_list[200:900]).mean()
#baseline_fl = np.array(value_list[200:900]).var()
#baseline_sigma = (np.array(err_list[200:900])**2).mean()
#baseline_err = np.sqrt(baseline_fl + baseline_sigma)

# Load the TTree
tree = uproot.open(f"{input_path}:signal")
# Convert branch to numpy array
for branch in ['sigQ_ch']:
    if not channel_data.empty:
        #baseline_position = channel_data['bl'].iloc[0]  # .iloc[0] accesses the first row of the filtered data
        mu =  channel_data['mu'].iloc[0]
        mu_err =  channel_data['mu_err'].iloc[0]
        lambda_ = channel_data['lambda'].iloc[0]
        lambda_err = channel_data['lambda_err'].iloc[0]
        sigma0 = channel_data['sigma0'].iloc[0]
        sigmak = channel_data['sigmak'].iloc[0]
        sigmaAp = 0
        sigma0_err = channel_data['sigma0_err'].iloc[0]
        sigmak_err = sigma0_err#channel_data['sigmak_err'].iloc[0]
        gain = channel_data['gain'].iloc[0]
        gain_err = channel_data['gain_err'].iloc[0]
        overvol =  channel_data['ov'].iloc[0] 
        vol =  channel_data['vol'].iloc[0] 
        vbd_err =  channel_data['vbd_err'].iloc[0]  
        batch =  channel_data['batch'].iloc[0] 
        box =  channel_data['box'].iloc[0] 
        tsn =  int(channel_data['tsn'].iloc[0]) 
        alpha = channel_data['alpha'].iloc[0]
        alpha_err = channel_data['alpha_err'].iloc[0]
        ap_gain = channel_data['ap_gain'].iloc[0]
        ap_gain_err = channel_data['ap_gain_err'].iloc[0]
        
        lower_edge = channel_data['lower_edge'].iloc[0]
        upper_edge = channel_data['upper_edge'].iloc[0]
        nbins = int(channel_data['nbins'].iloc[0])
        n_peaks = channel_data['n_peaks'].iloc[0]
        events = channel_data['events'].iloc[0]
        mean = channel_data['mean'].iloc[0] - channel_data['ped'].iloc[0]
        stderr = channel_data['var'].iloc[0]
        ndf = channel_data['charge_fit_ndf'].iloc[0]
        chi2 = channel_data['norm_chi2'].iloc[0]
        FD_bin_width = channel_data['fd_bin_width'].iloc[0]
    
        Ped = channel_data['ped'].iloc[0]    # Replace with your specific value
        Ped_err = channel_data['ped_err'].iloc[0]    # Replace with your specific value
        Gain = gain   # Replace with your specific value
        sigma0 = channel_data['sigma0'].iloc[0]  # Replace with your specific value
        sigmak =  channel_data['sigmak'].iloc[0]  # Replace with your specific value
        n_max = 25   # Replace with your specific value for the maximum n
    else:
        print(f"No data found for Channel {selected_channel}")
    data = tree[f"{branch}{pos}"].array(library="np")
    data = np.sort(data)

    print(len(data), int(len(data) * 0.05), data[int(len(data) * 0.01)])
    upper_edge = data[int(len(data) * 0.998)]
    amp_max = upper_edge
    amp_min = lower_edge
    #nBins = 800
    nBins = nbins #int((amp_max - amp_min) / amp_step)
    # Creating a histogram from the 'data' array
    # Create bin edges array
    bins = np.linspace(lower_edge, upper_edge, nbins + 1)
    # Calculate the width of each bin
    hist_counts, bin_edges = np.histogram(data, bins=bins)
    
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
    
    
    # Assuming there's only one row that matches, otherwise you might need to handle multiple rows
    # Plotting the results
    #plt.plot(spectrum_x, spectrum_y, label='Original Data', alpha=0.5)
    param_text = (#f"SN: {batch}-{box}-{int(tsn)}-{channel}\n"
                  #"Events:"+f" {events}\n"
                  #"$\\mathrm{Q}_\\mathrm{\\,baseline}$:"+f" {baseline_position:.2f} (pC)\n"
                  #"$\\sigma^\\mathrm{Q}_\\mathrm{\\,baseline}$:"+f" {baseline_rms:.2f} (pC)\n"
                  #"Recognised Peaks :"+f" {len(peak_positions)}\n"
                  #"$\\mathrm{V}_\\mathrm{bd}$ :"+f" {vbd:.2f}"+" $\\pm$ "+f"{vbd_err:.3f} (V)\n"
                  #"$\\mathrm{V}_\\mathrm{preset}$ :" +f" {vol} (V)\n"
                  #"$\\mathrm{V}_\\mathrm{bias}$ :"+f" {over_vol:.2f}"+" $\\pm$ "+f"{vbd_err:.2f} (V)")

                  "$\\mu$ :"+f" {mu:.3f}"+" $\\pm$ "+f"{mu_err:.3f}\n"
                  #"$\\mu_\\mathrm{ref.}$ :"+f" {ref_mu:.3f}"+" $\\pm$ "+f"{ref_mu_err:.3f}\n"
                  "$\\lambda$ :"+f" {lambda_:.3f}"+" $\\pm$ "+f"{lambda_err:.3f}\n"
                  "$\\alpha$ :" +f" {alpha:.3f}" + " $\\pm$ "+f"{alpha_err:.3f}\n"
                  #"DCR:"+f" {dcr:.1f}"+" $\\pm$ "+f"{dcr_err:.1f}"+" ($\\mathrm{Hz}/\\mathrm{mm}^2$)\n"
<<<<<<< HEAD
=======
                  "Gain :"+f" {gain:.2f}"+" $\\pm$ "+f"{gain_err:.2f}\n"
                  "AP pe:"+f" {ap_gain/gain:.3f}"+" $\\pm$ "+f"{ap_gain_err/gain:.3f}\n"
                  
>>>>>>> server
                  "Ped. :"+f" {Ped:.2f}"+" $\\pm$ "+f"{Ped_err:.2f}\n"
                  "Gain :"+f" {gain:.2f}"+" $\\pm$ "+f"{gain_err:.2f}\n"
                  "AP p.e.:"+f" {ap_gain/gain:.2f}"+" $\\pm$ "+f"{ap_gain_err/gain:.2f}\n"
                  "$\\sigma_0$ :"+f" {sigma0:.2f}"+" $\\pm$ "+f"{sigma0_err:.2f}\n"
                  "$\\sigma_k$ :"+f" {sigmak:.2f}"+" $\\pm$ "+f"{sigmak_err:.2f}"
                  )
    param_top_text = (f"SN: {batch}-{box}-{int(tsn)}-{channel}\n"
                  "Events: "+f"{events}\n"
                  "$\\mathrm{V}_\\mathrm{preset}$ :" +f" {vol} (V)\n"
                  "$\\mathrm{V}_\\mathrm{bias}$ : " +f"{overvol:0.2f}" +" $\\pm$ "+f"{vbd_err:.3f}\n"
                  "Mean: " +f"{mean:0.2f}\n"
                  "Std. Dev.: " +f"{stderr:0.2f}\n"
                   "FD bin width: " +f"{FD_bin_width:.3f}\n"
                  "DoF: " +f"{ndf:0.0f}\n"
                  "$\\chi^2$ / DoF: " +f"{chi2:0.3f}"
                  )
    # Calculate the statistical error (standard deviation) for each data point
    errors = np.sqrt(spectrum_y)
    
    # Create an error bar plot
    
    # Create a range of Q values
    scale = 1
    Q_values = np.linspace(lower_edge, upper_edge, int((upper_edge - lower_edge) * scale))  # Adjust the range and number of points as needed
    
    # Calculate the PDF
    #pdf_values_norm = compound_pdf(bins, Ped, Gain, mu, lambda_, sigma0, sigmak, n_max)
    #pdf_values = compound_pdf(Q_values, Ped, Gain, mu, lambda_, sigma0, sigmak, n_max)
    # Calculate the PDF
    if vol <= 2:
        pdf_values_norm, ap_pdf_values_norm, gp_pdf_values_norm = compound_pdf_ap(bins, Ped, Gain, ap_gain, mu, lambda_, alpha,  sigma0, sigmak,sigmaAp, n_max)
        pdf_values, ap_pdf_values, gp_pdf_values = compound_pdf_ap(Q_values, Ped, Gain, ap_gain, mu, lambda_, alpha,  sigma0, sigmak,sigmaAp, n_max)
    else:
        pdf_values_norm, ap_pdf_values_norm, gp_pdf_values_norm = geometric_pdf_ap(bins, Ped, Gain, ap_gain, mu, lambda_, alpha,  sigma0, sigmak,sigmaAp, n_max)
        pdf_values, ap_pdf_values, gp_pdf_values = geometric_pdf_ap(Q_values, Ped, Gain, ap_gain, mu, lambda_, alpha,  sigma0, sigmak,sigmaAp, n_max)
    #pdf_values = compound_pdf_ap(Q_values, Ped, Gain, mu, lambda_, sigma0, sigmak, n_max)
    # Normalize the PDF to a probability
    bin_widths = np.diff(Q_values)
    data_bin_widths = np.diff(bins)
    pdf_area = np.sum(pdf_values[:-1] * bin_widths)
    normalized_pdf = pdf_values / pdf_area
    # Scale the PDF to the number of events
    scaled_pdf = normalized_pdf * np.sum(hist_counts) / np.sum(pdf_values_norm /  pdf_area)#* (data_bin_widths.mean() / bin_widths.mean())
    #scale_factor = np.sum(hist_counts) / ((upper_edge - lower_edge) / len(Q_values)) /pdf_area 
    scale_factor = np.sum(hist_counts) / np.sum(pdf_values_norm) #np.sum(hist_counts) / ((upper_edge - lower_edge) / len(Q_values)) /pdf_area 
    scaled_pdf = pdf_values * scale_factor
    scaled_ap_pdf = ap_pdf_values * scale_factor
    scaled_gp_pdf = gp_pdf_values * scale_factor
    print("pdf_area:", pdf_area, "sum pdf:", np.sum(normalized_pdf),"pdf values norm:", np.sum(pdf_values_norm), "pdf sum:", np.sum(scaled_pdf), " hist counts: ", np.sum(hist_counts))
    
    # Calculate the residuals
    residuals = spectrum_y - np.interp(spectrum_x, Q_values, scaled_pdf)

    
    # Calculate the errors for the residuals
    # Assuming Poisson statistics, the error is the square root of the observed counts.
    residual_errors = np.sqrt(spectrum_y)
    pulls = residuals / residual_errors 

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [6, 2]})
    
     # Get the current color cycle
    default_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Get the first color
    first_color = default_color_cycle[0]
    
    #print("The default first color is:", first_color)   
    # First plot (Original Data and PDF)
    axs[0].plot(Q_values, scaled_pdf, label="Fit", alpha=0.7, zorder=3) # default color = #0C5DA5
    axs[0].plot(Q_values, scaled_ap_pdf, label="AP", alpha=0.7, linestyle=':', zorder=2) # default color = #0C5DA5
    axs[0].plot(Q_values, scaled_gp_pdf, label="GP", alpha=0.7, linestyle='--',zorder=1) # default color = #0C5DA5
    #axs[0].plot(Q_values, scaled_pdf, label="Fit", color='mediumblue', alpha=0.8, zorder=1, linewidth=3)
    axs[0].errorbar(spectrum_x, spectrum_y, yerr=errors, fmt='o', color='black', ecolor='black', label='Data', alpha=0.9, zorder=2)#, markerfacecolor='none')
    axs[0].set_yscale('log')
    axs[0].set_ylim(0.1, 1e4)
    #axs[0].set_title(f'Charge Spectrum Fit of SiPM {pos}-{channel} at {vol}V')
    #axs[0].text(0.5, 0.25, param_text, transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=textsize)
    axs[0].text(0.65, 0.95, param_text, transform=axs[0].transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=textsize)
    axs[0].text(0.85, 0.95, param_top_text, transform=axs[0].transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='paleturquoise', alpha=0.5), fontsize=textsize)
    #axs[0].text(0.5,0.5,param_text,fontsize=textsize, color='blue', verticalalignment='center', horizontalalignment='center')
    common_x_min = min(spectrum_x)  # Adjust as necessary
    common_x_max = max(spectrum_x)  # Adjust as necessary
    axs[0].legend(loc=(0.45,0.72))
    
    axs[0].set_xlim(common_x_min, common_x_max)
    axs[1].set_xlim(common_x_min, common_x_max)
    # ... (rest of your plotting code for the first plot)
    
    # Second plot (Pulls)
    axs[1].errorbar(spectrum_x, pulls, fmt='o', color='red', label='Pulls', alpha=0.7)
    #axs[1].axhline(0, color='black', linestyle='dashed', linewidth=1)  # Add a horizontal line at 0
    axs[1].axhline(1, color='gray', linestyle='dashed', linewidth=1)  # Add a horizontal line at 0
    axs[1].axhline(-1, color='gray', linestyle='dashed', linewidth=1)  # Add a horizontal line at 0
    axs[1].yaxis.set_major_locator(ticker.MultipleLocator(base=1))  # More granular y-ticks
    
    axs[1].set_ylim(-4, 4)
    # Adjust the second plot
    axs[1].set_xlabel('Accumulated Charge (pC)')
    axs[0].set_ylabel('Events')
    axs[1].set_ylabel('Pulls')
    axs[0].set_ylabel('Events')
    #axs[1].legend()

    # Adjust layout
    plt.tight_layout()
    #plt.show()

    ## Plotting
    #plt.plot(Q_values, scaled_pdf, label="Compound PDF")
    #plt.text(0.78, 0.75, param_text, transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=textsize)
    #if branch == "baselineQ_ch":
    #    title_suffix = "Baseline Range"
    #else:
    #    title_suffix = "LED Signal Range"
    #plt.legend()
    ## Set the yscale to 'log'
    #plt.yscale('log')
    #plt.ylim(0.1,1e4)
    #plt.xlabel('Accumulated Charge (pC)')
    #plt.ylabel('Number of Events')
    plt.savefig(f"charge_fit_{tsn}_{channel}_{vol}V.pdf")
    #plt.show()
    #plt.clf()

