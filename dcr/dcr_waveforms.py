import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ROOT

import scienceplots
plt.style.use('science')
plt.style.use('nature')

labelsize=28
titlesize=36
textsize=24
size_marker = 100

# Set global font sizes
#plt.rcParams['text.usetex'] = False
plt.rcParams['figure.figsize'] = (30, 9)
plt.rcParams['font.size'] = textsize  # Sets default font size
plt.rcParams['axes.labelsize'] = labelsize
plt.rcParams['axes.titlesize'] = titlesize
plt.rcParams['xtick.labelsize'] = labelsize
plt.rcParams['ytick.labelsize'] = labelsize
plt.rcParams['legend.fontsize'] = textsize
plt.rcParams['errorbar.capsize'] = 3
plt.rcParams['lines.markersize'] = 2  # For example, 8 points
plt.rcParams['lines.linewidth'] = 1.5 # For example, 2 points
# Set global parameters using rcParams
plt.rcParams['axes.titlepad'] = 20  # Padding above the title
plt.rcParams['axes.labelpad'] = 15  # Padding for both x and y axis labels

#def plot_waveform(amplitudes, charges, filtered_amp, filtered_charge, baseline, gain, sigma):
def plot_waveform(amplitudes, charges, baseline, gain, sigma, idx):
    if idx == 7:
        title = '1 p.e.'
    elif idx == 8:
        title = '0 p.e.'
    else:
        title = '2 p.e.'
    tmin = 954
    tmax = 1354
    points = np.arange(45,1964,1)
    
    hist = ROOT.TH1F("hist", "hist", 4000,  -200, 200)
    for i in range(len(amplitudes[0:tmax])):
        hist.Fill(amplitudes[i])
    mean_value = hist.GetMean()
    expect_value = hist.GetBinCenter(hist.GetMaximumBin())
    baseline = mean_value#expect_value
    plt.figure()
    plt.errorbar(points[tmin:tmax], amplitudes[tmin:tmax], label='Amplitude', fmt= '-o', color='black')
    plt.errorbar(points[tmin:tmax], charges[tmin:tmax], label='Average', color='orange',fmt='-s')
    #plt.plot(filtered_amp[tmin:tmax], label='filterd amp')
    #plt.plot(filtered_charge[tmin:tmax], label='filtered charge')
    plt.title(f'Typical Waveform ({title})')
    plt.xlabel('Time (8 ns)')
    #plt.ylim(80, 110)
    plt.ylabel('Amplitude (mV)')
    
    # Draw the baseline
    plt.axhline(y=baseline, color='r', linestyle='-', label='Baseline')
    
    # Draw the 1 p.e. line
    one_pe = baseline + gain
    plt.axhline(y=one_pe, color='g', linestyle='-', label='1 p.e.')
    
    # Draw the 1 p.e. - 1 * sigma line
    plt.axhline(y=one_pe - sigma, color='b', linestyle='--', label='1 p.e. - 1 $\\sigma$')
    
    # Draw the 1 p.e. - 2 * sigma line
    plt.axhline(y=one_pe - 2*sigma, color='c', linestyle='--', label='1 p.e. - 2 $\\sigma$')
    
    # Draw the 1 p.e. - 3 * sigma line
    plt.axhline(y=one_pe - 3*sigma, color='m', linestyle='--', label='1 p.e. - 3 $\\sigma$')
    plt.ylim(83,86)
    #plt.legend(loc='upper right')  # Add a legend to the plot
    #plt.savefig("dcr_waveform.pdf")
    #plt.show()
    # Place the legend outside of the figure/plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Adjust location as needed

    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust the rect parameter as needed for your figure

    plt.savefig(f"dcr_waveform_{idx}.pdf", bbox_inches='tight')  # Save with a tight bounding box
    #plt.show()

def main():
    file_path = sys.argv[1]
    file_path2 = sys.argv[2]
    csv_path = sys.argv[3]
    #file_path3 = sys.argv[3]
    #file_path4 = sys.argv[4]
    
    ch = 1
    pos = 11
    vol = 1
    run = 149
    df = pd.read_csv(csv_path, usecols=['pos', 'ch', 'vol', 'bl','bl_rms','gain','sigma0','sigmak', 'run'])
    df = df.loc[(df['ch'] == ch) & (df['pos'] == pos) & (df['vol'] == vol) & (df['run'] == run)]
    baseline = df.head(1)['bl'].values[0] / 45
    #baseline = 3698.7454/45 # for original
    gain = df.head(1)['gain'].values[0] / 45
    #gain = 14.266638738693526/45
    sigma = np.sqrt(df.head(1)['sigma0'].values[0]**2 + df.head(1)['sigmak'].values[0]**2) /45
    #sigma = 5.091490512172288 /45#4.6507185/45
    #baseline = 103.07520002357398
    #gain = 18.814526268814543 / 45
    #sigma = 5/45

    with open(file_path, 'r') as file:
        waveforms = [list(map(float, line.strip().split())) for line in file]

    with open(file_path2, 'r') as file:
        charge_waveforms = [list(map(float, line.strip().split())) for line in file]
    #with open(file_path3, 'r') as file:
    #    filtered_amp = [list(map(float, line.strip().split())) for line in file]
    #with open(file_path4, 'r') as file:
    #    filtered_charge = [list(map(float, line.strip().split())) for line in file]
    for i in range(len(waveforms)):
        #plot_waveform(waveforms[i], charge_waveforms[i], filtered_amp[i], filtered_charge[i], baseline, gain, sigma)
        plot_waveform(waveforms[i],charge_waveforms[i], baseline, gain, sigma, i)

if __name__ == "__main__":
    main()

