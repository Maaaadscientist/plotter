import os, sys
import re
import ROOT
import pandas as pd
from math import factorial
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

root_file = os.path.abspath(sys.argv[1])
csv_file = os.path.abspath(sys.argv[2])
if len(sys.argv) > 3: 
    tsn = str(sys.argv[3])
else:
    tsn = 1463
plt.style.use('science')
plt.style.use('nature')


labelsize=28
titlesize=42
textsize=21
size_marker = 100

# Set global font sizes
#plt.rcParams['text.usetex'] = False
plt.rcParams['figure.figsize'] = (20, 15)
plt.rcParams['font.size'] = textsize  # Sets default font size
plt.rcParams['axes.labelsize'] = labelsize
plt.rcParams['axes.titlesize'] = titlesize
plt.rcParams['xtick.labelsize'] = labelsize
plt.rcParams['ytick.labelsize'] = labelsize
plt.rcParams['legend.fontsize'] = labelsize
plt.rcParams['errorbar.capsize'] = 4
plt.rcParams['lines.markersize'] = 5  # For example, 8 points
plt.rcParams['lines.linewidth'] = 1 # For example, 2 points
# Set global parameters using rcParams
plt.rcParams['axes.titlepad'] = 20  # Padding above the title
plt.rcParams['axes.labelpad'] = 15  # Padding for both x and y axis labels


def gaussian(x, mean, amplitude, std_dev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2)) / (std_dev * np.sqrt(2*np.pi))

# Function to extract data from TH1F
#def get_data_from_TH1F(th1f):
#    bin_contents = np.array([th1f.GetBinContent(i) for i in range(1, th1f.GetNbinsX() + 1)])
#    bin_edges = np.array([th1f.GetBinLowEdge(i) for i in range(1, th1f.GetNbinsX() + 2)])
#    return bin_contents, bin_edges
# Function to extract data from TH1F
def get_data_from_TH1F(th1f, rebin_factor=10):
    # Rebin the histogram if necessary
    if rebin_factor > 1:
        th1f.Rebin(rebin_factor)
    bin_contents = np.array([th1f.GetBinContent(i) for i in range(1, th1f.GetNbinsX() + 1)])
    bin_edges = np.array([th1f.GetBinLowEdge(i) * 45  for i in range(1, th1f.GetNbinsX() + 2)])
    return bin_contents, bin_edges

def borel_pmf(k, lambda_):
    return (lambda_ * k)**(k - 1) * np.exp(-k * lambda_) / factorial(k)

def fetch_file_info(filename):
    ### example ### main_run_0149_ov_1.00_sipmgr_01_tile.root
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
file_info = fetch_file_info(root_file.split("/")[-1])
run = int(file_info['run'])
vol = int(file_info['ov'])
channel = int(file_info['channel'])
df = pd.read_csv(csv_file)
if 'p' in tsn:
    pos = int(tsn.replace("p",""))
    channel_data = df[(df['pos'] == pos) & (df['run'] == run) & (df['vol'] == vol) & (df['ch'] == channel)]
else:
    channel_data = df[(df['tsn'] == int(tsn)) & (df['run'] == run) & (df['vol'] == vol) & (df['ch'] == channel)]
    pos = channel_data['pos'].iloc[0]


# Step 1: Open the ROOT file and get the TH1F
lambda_ = channel_data['lambda'].iloc[0]
lambda_err = channel_data['lambda_err'].iloc[0]
sigma0 = channel_data['sigma0'].iloc[0]
sigmak = channel_data['sigmak'].iloc[0]
gain = channel_data['gain'].iloc[0]
gain_err = channel_data['gain_err'].iloc[0]
ov = channel_data['ov'].iloc[0]
dcr = channel_data['dcr'].iloc[0]
dcr_err = channel_data['dcr_err'].iloc[0]
batch = channel_data['batch'].iloc[0]
box = channel_data['box'].iloc[0]
sn = channel_data['tsn'].iloc[0]
print(sn)
vbd_err = channel_data['vbd_err'].iloc[0]
temp = channel_data['temp'].iloc[0]
if temp == 0:
    if channel != 16:
        bkp_data = df[(df['pos'] == pos) & (df['run'] == run) & (df['vol'] == vol) & (df['ch'] == channel+1)]
        temp = bkp_data['temp'].iloc[0]
    else:
        bkp_data = df[(df['pos'] == pos) & (df['run'] == run) & (df['vol'] == vol) & (df['ch'] == channel-1)]
        temp = bkp_data['temp'].iloc[0]

events = channel_data['events'].iloc[0]
#file = ROOT.TFile('root/main_run_0318_ov_2.00_sipmgr_11_tile.root ', 'READ')
file = ROOT.TFile(root_file, 'READ')
# Step 2: Get the TH1F histograms
dcrQ = file.Get(f'dcrQ_ch{pos}')
dcrQ_neg = file.Get(f'dcrQ_neg_ch{pos}')

# Step 3: Extract bin contents and edges from the histograms
bin_contents, bin_edges = get_data_from_TH1F(dcrQ)
bin_contents_neg, bin_edges_neg = get_data_from_TH1F(dcrQ_neg)
positive_events = np.sum(bin_contents)
positive_events_err = np.sqrt(positive_events)
negative_events = np.sum(bin_contents_neg)
negative_events_err = np.sqrt(negative_events)
net_events = positive_events - negative_events if (positive_events - negative_events) > 0 else 0
net_events_err = np.sqrt(positive_events + negative_events)
time_length = events * 1155 * 8 * 1e-9

# Ensure the bin edges are the same for both histograms (they should be if they're comparable)
assert np.array_equal(bin_edges, bin_edges_neg), "Bin edges are not equal!"

# Step 4: Calculate the difference
difference = bin_contents - bin_contents_neg
errors = np.sqrt(bin_contents + bin_contents_neg)

# Calculate the bin centers
bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
# Plot dcrQ and dcrQ_neg together
plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
norm_factor = 2/ (2-np.exp(-lambda_))
#norm_factor_up = 2/ (2-np.exp(-lambda_ + lambda_err)) - norm_factor
#norm_factor_down = norm_factor - 2/ (2-np.exp(-lambda_ - lambda_err))
#norm_factor_err = np.sqrt(norm_factor_up**2 + norm_factor_down**2)/2
dcr_ch_err = dcr_err * 144

param_text = (f"SN: {batch}-{box}-{int(sn)}-{channel}\n"
              #"Events:"+f" {events}\n"
              #"$\\mathrm{Q}_\\mathrm{\\,baseline}$:"+f" {baseline_position:.2f} (pC)\n"
              #"$\\sigma^\\mathrm{Q}_\\mathrm{\\,baseline}$:"+f" {baseline_rms:.2f} (pC)\n"
              #"Recognised Peaks :"+f" {len(peak_positions)}\n"
              #"$\\mathrm{V}_\\mathrm{bd}$ :"+f" {vbd:.2f}"+" $\\pm$ "+f"{vbd_err:.3f} (V)\n"
              #"$\\mathrm{V}_\\mathrm{preset}$ :" +f" {vol} (V)\n"
              #"$\\mathrm{V}_\\mathrm{bias}$ :"+f" {over_vol:.2f}"+" $\\pm$ "+f"{vbd_err:.2f} (V)")

              #"$\\mu_\\mathrm{ref.}$ :"+f" {ref_mu:.3f}"+" $\\pm$ "+f"{ref_mu_err:.3f}\n"
              "$\\mathrm{V}_\\mathrm{preset}$ :" +f" {vol} (V)\n"
              "$\\mathrm{V}_\\mathrm{bias}$ :"+f" {ov:.2f}"+" $\\pm$ "+f"{vbd_err:.2f} (V)\n"
              f"Temperature: {temp:.2f}" + "$^\\circ\\mathrm{C}$\n"
              "$\\lambda$ :"+f" {lambda_:.3f}"+" $\\pm$ "+f"{lambda_err:.3f}\n"
              #"Norm.:" + f" {norm_factor:.3f}"+ " $\\pm$ "+f"{norm_factor_err:.3f}"+"\n"
              f"O. Thres. Events: {positive_events}" + "$\\pm$" +f" {positive_events_err:.1f}" + "$_\\mathrm{(stat.)}$"+"\n"
              f"E. Noise Events: {negative_events}" + "$\\pm$" +f" {negative_events_err:.1f}" + "$_\\mathrm{(stat.)}$"+"\n"
              f"Net Events: {net_events}" + "$\\pm$" +f" {net_events_err:.1f}" + "$_\\mathrm{(stat.)}$"+"\n"
              f"Time: {time_length:.2f} s\n"
              "DCR (ch): " + f"{net_events * norm_factor / time_length:.0f}" + "$\\pm$" +f" {net_events_err * norm_factor/time_length:.0f}" + "$_\\mathrm{(stat.)}$"+ "($\\mathrm{Hz}$)"
              #"\n"
              #"DCR:"+f" {dcr:.1f}"+" $\\pm$ "+f" {net_events_err*norm_factor/time_length/144:.1f}" + "$_\\mathrm{(stat.)}$"+ " $\\pm$ "+f"{np.sqrt(dcr_ch_err**2 - (net_events_err*norm_factor/time_length)**2)/144:.1f}" + "$_\\mathrm{(syst.)}$"+" ($\\mathrm{Hz}/\\mathrm{mm}^2$)"
              )
plt.text(0.68, 0.75, param_text, transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=textsize)


plt.bar(bin_centers, bin_contents, width=np.diff(bin_edges), edgecolor='black', alpha=0.7, label='Above Thres.', color='mediumblue', hatch='--')
plt.bar(bin_centers, bin_contents_neg, width=np.diff(bin_edges_neg), edgecolor='black', alpha=0.8, label='Electronic Noise', color='salmon', hatch='**')
plt.title('DCR Charge Distribution')
plt.xlabel('Accumulated Charge (pC)')
plt.ylabel('Events')
#plt.ylim(0,200)
plt.ylim(1,max(bin_contents)*10)
plt.xlim(0, 100)
plt.yscale('log')
plt.legend()

# Plot the difference
plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
plt.bar(bin_centers, difference, width=np.diff(bin_edges), edgecolor='black', alpha=0.7, color='red', label='Net')
plt.errorbar(bin_centers, difference, yerr=errors, fmt='none', ecolor='black', elinewidth=2, capsize=4, label='Statistical Unc.')
#plt.title('Difference (dcrQ - dcrQ_neg)')
plt.xlabel('Accumulated Charge (pC)')
plt.ylabel('Events')
#plt.ylim(0,100)
plt.ylim(1,(max(bin_contents)*10)/2)
plt.xlim(0, 100)
plt.yscale('log')
# Define Gaussian parameters
bin_width = bin_edges[1] - bin_edges[0]
print(bin_width)
sigma1 = np.sqrt(sigma0**2 + sigmak**2)
sigma2 = np.sqrt(sigma0**2 + 2*sigmak**2)
sigma3 = np.sqrt(sigma0**2 + 3*sigmak**2)
mean1, amp1, std1 = gain, net_events / (1 - 0.5*borel_pmf(1, lambda_)) * borel_pmf(1, lambda_) * bin_width, sigma1   # for the first Gaussian
mean2, amp2, std2 = gain*2, net_events / (1 - 0.5*borel_pmf(1, lambda_)) * borel_pmf(2, lambda_) * bin_width , sigma2  # for the second Gaussian
mean3, amp3, std3 = gain*3, net_events / (1 - 0.5*borel_pmf(1, lambda_)) * borel_pmf(3, lambda_) * bin_width , sigma3  # for the second Gaussian

# Generate x values
x_values = np.linspace(0, 100, 5000)  # Adjust the range and number of points as needed

# Calculate y values for the Gaussians
y_gauss1 = gaussian(x_values, mean1, amp1, std1)
y_gauss2 = gaussian(x_values, mean2, amp2, std2)
y_gauss3 = gaussian(x_values, mean3, amp3, std3)
# Plot the Gaussians
plt.plot(x_values, y_gauss1, color='orange',linestyle='--', linewidth=3)  # red dashed line for the first Gaussian
plt.plot(x_values, y_gauss2, color='deepskyblue',linestyle='--', linewidth=3)   # green solid line for the second Gaussian
plt.plot(x_values, y_gauss3, color='purple',linestyle='--', linewidth=3)   # green solid line for the second Gaussian

# Example: Adding two vertical lines at x=5 and x=10
plt.axvline(x=gain, color='orange', linestyle=':', linewidth=3, label='1 p.e.')  # red dashed line at x=5
plt.axvline(x=gain*2, color='deepskyblue', linestyle=':', linewidth=3, label='2 p.e.')   # green solid line at x=10
plt.axvline(x=gain*3, color='purple', linestyle=':', linewidth=3, label='3 p.e.')   # green solid line at x=10


plt.legend()

plt.tight_layout()  # Adjust subplots to fit into figure area.
plt.savefig(f"dcr_hist_run{run}_sn{int(sn)}_ov{vol}_ch{channel}.pdf")
