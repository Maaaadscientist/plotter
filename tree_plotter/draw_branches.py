import os
import sys
import yaml
import numpy as np
import ROOT
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
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
config_path = os.path.abspath(sys.argv[1])
with open(config_path, 'r') as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)
root_file = ROOT.TFile(yaml_data['tree_path'])
tree = root_file.Get(yaml_data['tree_name'])
# Assuming 'branchName' is the name of your branch

for branch in yaml_data['branch_to_draw']:
    branch_name = yaml_data['branch_to_draw'][branch]['name']
    # Example bin edges in a NumPy array
    if not 'bins' in yaml_data['branch_to_draw'][branch]:
        bin_edges = np.arange(yaml_data['branch_to_draw'][branch]['lower_edge'], yaml_data['branch_to_draw'][branch]['upper_edge'] + yaml_data['branch_to_draw'][branch]['step'], yaml_data['branch_to_draw'][branch]['step'])
    else:
        bin_edges = [float(key) for key in yaml_data['branch_to_draw'][branch]['bins'].split(',')]
    # Convert NumPy array to a ROOT array, as ROOT expects an array of doubles
    root_bin_edges = ROOT.std.vector('double')(len(bin_edges))
    for i, edge in enumerate(bin_edges):
        root_bin_edges[i] = edge
    
    # Create a TH1F histogram with variable bin widths
    hist = ROOT.TH1F(f"hist_{branch}", "Variable Bins Histogram;Value;Frequency", len(bin_edges)-1, root_bin_edges.data())
    if 'selections' in yaml_data['branch_to_draw'][branch]:
        selections = yaml_data['branch_to_draw'][branch]['selections']
    else:
        selections = ''
    tree.Draw(f"{branch_name}>>hist_{branch}",selections)
        
    if 'print_stat' in yaml_data['branch_to_draw'][branch]:
        print(f"{branch_name},{hist.GetMean()},{hist.GetRMS()}",end='')
    if 'print_text' in yaml_data['branch_to_draw'][branch]:
        print(','+str(yaml_data['branch_to_draw'][branch]['print_text'])+'\n',end='')
    # Prepare arrays for bin contents and bin edges
    n_bins = hist.GetNbinsX()
    bin_contents = np.zeros(n_bins)
    bin_edges = np.zeros(n_bins + 1)
    
    # Extract the bin contents and edges
    for i in range(1, n_bins + 1):
        bin_contents[i - 1] = hist.GetBinContent(i)
        bin_edges[i - 1] = hist.GetBinLowEdge(i)
    bin_edges[-1] = hist.GetBinLowEdge(n_bins + 1)
    
    if 'histtype' in yaml_data['branch_to_draw'][branch]:
        hist_type = yaml_data['branch_to_draw'][branch]['histtype']
    else:
        hist_type = 'stepfilled'

    if 'color' in yaml_data['branch_to_draw'][branch]:
        color = yaml_data['branch_to_draw'][branch]['color']
    else:
        color = 'deepskyblue'

    plt.figure(figsize=(20, 15))
    # Calculate statistics
    n_entries = hist.GetEntries()
    mean = hist.GetMean()
    RMS = hist.GetRMS()

    # ... [existing code for plotting histogram] ...

    # Add a statistical pad with text
    if 'scientific' in yaml_data['branch_to_draw'][branch]:
        stats_text = f'Entries: {n_entries:.0f}\nMean: {mean:.3e}\nRMS: {RMS:.3e}'
    else:
        stats_text = f'Entries: {n_entries:.0f}\nMean: {mean:.3f}\nRMS: {RMS:.3f}'
    plt.gca().text(0.8, 0.95, stats_text, transform=plt.gca().transAxes, 
                   fontsize=textsize, verticalalignment='top')

    _, bins, _ = plt.hist(bin_edges[:-1], bins=bin_edges, weights=bin_contents, edgecolor='black', color=color,linewidth=1, histtype='stepfilled')
    if 'bin_text' in yaml_data['branch_to_draw'][branch]:
        # Add text on top of each bar
        for i in range(len(bin_contents)):
            bin_center = (bins[i] + bins[i+1]) / 2  # Calculate the center of the bin
            plt.text(bin_center, bin_contents[i], f'{bin_contents[i]:.0f}', ha='center', va='bottom', fontsize=textsize)

    # Set axes to use scientific notation
    sci_formatter = ScalarFormatter(useMathText=True)  # Enables LaTeX-style notation
    sci_formatter.set_powerlimits((-10,10))

    plt.xlabel(yaml_data['branch_to_draw'][branch]['xlabel'])
    plt.ylabel(yaml_data['branch_to_draw'][branch]['ylabel'])
    #plt.xticks()
    #plt.yticks(fontsize=labelsize)
    if 'log' in yaml_data['branch_to_draw'][branch]:
        plt.yscale('log')
    if 'axis_sci_format' in yaml_data['branch_to_draw'][branch]:
        plt.rcParams['axes.formatter.use_mathtext'] = True
        plt.rcParams['axes.formatter.limits'] = (-3, 3)

    plt.title(yaml_data['branch_to_draw'][branch]['title'])
    plt.savefig(f"{branch}.pdf")
    plt.clf()
    plt.close()
#
