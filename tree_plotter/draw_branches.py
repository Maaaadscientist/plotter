import os
import sys
import yaml
import numpy as np
import ROOT
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
plt.style.use('nature')

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
    hist = ROOT.TH1F("hist", "Variable Bins Histogram;Value;Frequency", len(bin_edges)-1, root_bin_edges.data())
    if 'selections' in yaml_data['branch_to_draw'][branch]:
        selections = yaml_data['branch_to_draw'][branch]['selections']
    else:
        selections = ''
    tree.Draw(f"{branch_name}>>hist",selections)
        
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
    plt.figure(figsize=(4, 3))
    plt.hist(bin_edges[:-1], bins=bin_edges, weights=bin_contents, edgecolor='black', color=color,linewidth=0.5, histtype='stepfilled')
    plt.xlabel(yaml_data['branch_to_draw'][branch]['xlabel'])
    plt.ylabel(yaml_data['branch_to_draw'][branch]['ylabel'])
    plt.title(yaml_data['branch_to_draw'][branch]['title'])
    plt.savefig(f"{branch}.pdf")
    plt.clf()
#
