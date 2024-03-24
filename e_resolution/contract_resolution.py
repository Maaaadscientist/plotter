import numpy as np
import pandas as pd
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import ROOT
import argparse  # Import argparse module

# Constants
CHANNELS_PER_TILE = 16  # Number of channels per SiPM tile
#N = 20  # Number of tests
#ov = 3.0
PHOTONS_PER_TEST = 9846  # Number of photons simulated in each test

# Example parameters
#dcr = 50.
PTE = 0.9  # Probability of photon hitting SiPM active area 815074 / 975251
keys = [round(2.5 + 0.1 * i, 1) for i in range(46)]

pde_dict = {"max": 0.44, "typical":0.47}
pct_dict = {"max": 0.15, "typical":0.12}
dcr_dict = {"max": 41.7, "typical":13.9}
pap_dict = {"max": 0.08, "typical":0.04}
gain_dict = {"max": 1e6, "typical":4e6}
#optical_crosstalk_param = 0.2  # Parameter for generalized Poisson distribution

def parse_arguments():
    parser = argparse.ArgumentParser(description='Photon Simulation Parameters')
    parser.add_argument('--N', type=int, default=20, help='Number of tests')
    parser.add_argument('--ov', type=str, default="typical", help='Overvoltage')
    parser.add_argument('--input', type=str, default="bychannel.csv", help='Input CSV file path')
    parser.add_argument('--output', type=str, default='all_hist.root', help='Output ROOT file path')
    parser.add_argument('--seed', type=int,default=123456, help='Random seed for simulation', required=False)
    args = parser.parse_args()
    return args.N, args.ov, args.input, args.output, args.seed

# Parse command line arguments
N, ov, input_file, output_file, seed = parse_arguments()

# Initialize random seed
if seed is not None:
    np.random.seed(seed)

def generalized_poisson_pmf(k, mu, lambda_):
    exp_term = np.exp(-(mu + k * lambda_))
    main_term = mu * ((mu + k * lambda_) ** (k - 1))
    factorial_term = math.factorial(k)
    return (main_term * exp_term) / factorial_term

def generate_random_generalized_poisson(mu, lambda_, max_k=20):
    # Generate PMF values
    pmf_values = [generalized_poisson_pmf(k, mu, lambda_) for k in range(max_k)]

    # Normalize the PMF
    total = sum(pmf_values)
    normalized_pmf = [value / total for value in pmf_values]

    # Random sampling based on the PMF
    return np.random.choice(range(max_k), p=normalized_pmf)

def simulate_photon(ov):
    """ Simulate the journey of a single photon. """
    # Check if photon hits the active area
    if np.random.rand() > PTE:
        return 0, 0, 0, 0

    pde = pde_dict[ov]

    # Simulate optical crosstalk
    
    pct = pct_dict[ov]
    lambda_ = np.log(1 / (1 - pct))
    electrons = generate_random_generalized_poisson(pde, lambda_)

    # Simulate afterpulsing
    p_ap = pap_dict[ov]
    afterpulses = np.sum(np.random.rand(electrons) < p_ap)
    
    # 1V = 572,722.73
    sigma = 572722.73 * 0.19 / 6 
    gain = (gain_dict[ov] + np.random.normal(0, sigma)) / 1e6

    return 1, electrons, electrons + afterpulses, gain * (electrons + afterpulses)

def simulate_dcr(ov):
    dcr = dcr_dict[ov]#average_value_from_csv(df, dcr_name)
    # 1V = 572,722.73
    sigma = 572722.73 * 0.19 / 6 
    gain = (gain_dict[ov] + np.random.normal(0, sigma)) / 1e6
    init_pe = dcr * 10 * 1e6 * 300 * 1e-9
    pe_ct = 0
    pct = pct_dict[ov]
    for _ in range(int(init_pe)):
        pe_ct += generate_random_generalized_poisson(1, pct)

    # Simulate afterpulsing
    pap = pap_dict[ov]
    pe_ap = np.sum(np.random.rand(pe_ct) < pap)
    return pe_ct + pe_ap, (pe_ct + pe_ap) * gain
        
def main():
    # Data collection arrays
    initial_photons_list = []
    detected_photons_list = []  # List to collect detected photons
    detected_electrons_list = []
    total_electrons_list = []
    dcr_added_electrons_list = []
    # Running N tests
    
    h_init = ROOT.TH1F("hist_init", "hist_init", 20000,0,20000)
    h_LS = ROOT.TH1F("hist_LS", "hist_LS", 20000,0,20000)
    h_pte = ROOT.TH1F("hist_PTE", "hist_PTE", 20000,0,20000)
    h_ct = ROOT.TH1F("hist_ct", "hist_ct", 20000,0,20000)
    h_ap = ROOT.TH1F("hist_ap", "hist_ap", 20000,0,20000)
    h_dcr = ROOT.TH1F("hist_dcr", "hist_dcr", 20000,0,20000)
    h_charge = ROOT.TH1F("hist_charge", "hist_charge", 200000,0,200000)
    for _ in range(N):
        LS_photons = np.random.poisson(PHOTONS_PER_TEST)
        test_results = [simulate_photon(ov) for _ in range(LS_photons)]
        detected_photons = sum(r[0] for r in test_results)
        detected_electrons = sum(r[1] for r in test_results)
        total_electrons = sum(r[2] for r in test_results)
        total_charge = sum(r[3] for r in test_results)
        dcr_electrons = simulate_dcr(ov)[0]
        dcr_charge = simulate_dcr(ov)[1]
        
        h_init.Fill(PHOTONS_PER_TEST)
        h_LS.Fill(LS_photons)
        h_pte.Fill(detected_photons)
        h_ct.Fill(detected_electrons)
        h_ap.Fill(total_electrons)
        h_dcr.Fill(total_electrons + dcr_electrons)
        h_charge.Fill(total_charge + dcr_charge)
    
        detected_photons_list.append(detected_photons)
        detected_electrons_list.append(detected_electrons)
        total_electrons_list.append(total_electrons)
        dcr_added_electrons_list.append(total_electrons + dcr_electrons)
    
    # At the end, where the ROOT file is saved
    f1 = ROOT.TFile(output_file, "recreate")
    h_init.Write()
    h_LS.Write()
    h_pte.Write()
    h_ct.Write()
    h_ap.Write()
    h_dcr.Write()
    h_charge.Write()
    f1.Close()

if __name__ == "__main__":
    main()
## Plotting histograms
#plt.figure(figsize=(20, 5))
#
#plt.subplot(1, 5, 1)
#plt.hist(initial_photons_list, bins=50, color='blue', alpha=0.7)
#plt.title('Initial Photons Detected')
#plt.xlabel('Number of Photons')
#plt.ylabel('Frequency')
#
#plt.subplot(1, 5, 2)
#plt.hist(detected_photons_list, bins=50, color='orange', alpha=0.7)
#plt.title('Detected Photons')
#plt.xlabel('Number of Photons')
#plt.ylabel('Frequency')
#
#plt.subplot(1, 5, 3)
#plt.hist(detected_electrons_list, bins=50, color='green', alpha=0.7)
#plt.title('Detected Electrons After Crosstalk')
#plt.xlabel('Number of Electrons')
#plt.ylabel('Frequency')
#
#plt.subplot(1, 5, 4)
#plt.hist(total_electrons_list, bins=50, color='red', alpha=0.7)
#plt.title('Total Electrons Including Afterpulsing')
#plt.xlabel('Number of Electrons')
#plt.ylabel('Frequency')
#
#plt.subplot(1, 5, 5)
#plt.hist(dcr_added_electrons_list, bins=50, color='deepskyblue', alpha=0.7)
#plt.title('Total Electrons Including DCR')
#plt.xlabel('Number of Electrons')
#plt.ylabel('Frequency')
#plt.tight_layout()
#plt.show()
#
