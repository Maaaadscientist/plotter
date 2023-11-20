import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


labelsize=18
titlesize=24
textsize=14

color_3V = 'blue'
color_5V = 'orangered'
def read_and_process_csv(file_path):
    # Read CSV file
    df = pd.read_csv(file_path, header=None, names=["X", "Y"])

    # Round X values to nearest multiple of 2
    df['X'] = (df['X'] / 2).round() * 2

    return df

def interpolate_pde(df, wavelength):
    # Interpolating the PDE values
    f = interp1d(df['X'], df['Y'], kind='linear')
    return f(wavelength)

def calculate_normalized_difference(pde_404, pde_415):
    return (pde_415 - pde_404) / pde_404


# Read and process the CSV files
df_ov3V = read_and_process_csv('ov3V.csv')
df_ov5V = read_and_process_csv('ov5V.csv')

# Interpolate PDE values at 404 nm and 415 nm for both datasets
pde_404_ov3V = interpolate_pde(df_ov3V, 404)
pde_415_ov3V = interpolate_pde(df_ov3V, 415)
pde_404_ov5V = interpolate_pde(df_ov5V, 404)
pde_415_ov5V = interpolate_pde(df_ov5V, 415)

# Calculate normalized differences
norm_diff_ov3V = calculate_normalized_difference(pde_404_ov3V, pde_415_ov3V)
norm_diff_ov5V = calculate_normalized_difference(pde_404_ov5V, pde_415_ov5V)

# Plotting
plt.figure(figsize=(12, 6))

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    #"font.family": "helvet",
    "text.latex.preamble": r"\usepackage{courier}",
})

# Line plot for ov3V
plt.plot(df_ov3V['X'], df_ov3V['Y'], color=color_3V, label='Overvoltage 3V', linestyle='--', marker='x', markersize=4, alpha=0.7, linewidth=1)

# Line plot for ov5V
plt.plot(df_ov5V['X'], df_ov5V['Y'], color=color_5V, label='Overvoltage 5V', linestyle='-', marker='^', markersize=4, alpha=0.7, linewidth=1)

# Draw horizontal lines for interpolated PDE values
plt.hlines(y=pde_404_ov3V, xmin=350, xmax=404, color=color_3V, linestyle=':', alpha=0.5)
plt.hlines(y=pde_415_ov3V, xmin=350, xmax=415, color=color_3V, linestyle=':', alpha=0.5)
plt.hlines(y=pde_404_ov5V, xmin=350, xmax=404, color=color_5V, linestyle='-', alpha=0.5)
plt.hlines(y=pde_415_ov5V, xmin=350, xmax=415, color=color_5V, linestyle='-', alpha=0.5)
plt.text(490, pde_415_ov3V - 0.06, '$\\frac{\\mathrm{PDE}^{\\,415\\mathrm{nm}}_{\\,3\\mathrm{V}}-\\mathrm{PDE}^{\\,404\\mathrm{nm}}_{\\,3\\mathrm{V}}}{\\mathrm{PDE}^{\\,404\\mathrm{nm}}_{\\,3\\mathrm{V}}} =\\, $'+f'{norm_diff_ov3V:.4f}', verticalalignment='top', horizontalalignment='right', color=color_3V,fontsize=textsize)
plt.text(435, pde_415_ov5V + 0.06, '$\\frac{\\mathrm{PDE}^{\\,415\\mathrm{nm}}_{\\,5\\mathrm{V}}-\\mathrm{PDE}^{\\,404\\mathrm{nm}}_{\\,5\\mathrm{V}}}{\\mathrm{PDE}^{\\,404\\mathrm{nm}}_{\\,5\\mathrm{V}}} =\\, $'+f'{norm_diff_ov5V:.4f}', verticalalignment='bottom', horizontalalignment='right', color=color_5V,fontsize=textsize)

# Adding arrows to indicate the difference
arrow_offset = 0.005
plt.annotate('', xy=(390, pde_404_ov3V - arrow_offset), xytext=(390, pde_415_ov3V + arrow_offset),
             arrowprops=dict(arrowstyle="<->", lw=1, color=color_3V))
plt.annotate('', xy=(380, pde_404_ov5V - arrow_offset), xytext=(380, pde_415_ov5V + arrow_offset),
             arrowprops=dict(arrowstyle="<->", lw=1, color=color_5V))

plt.xlim(350, max(df_ov3V['X']) + 10)
plt.xticks(fontsize=labelsize)  # Adjust font size for x-axis ticks
plt.yticks(fontsize=labelsize)  # Adjust font size for y-axis ticks
plt.xlabel('Wavelength (nm)', fontsize=labelsize)
plt.ylabel('Relative PDE (Photon Detection Efficiency)', fontsize=labelsize)
plt.title('PDE vs Wavelength for Different Overvoltages', fontsize=titlesize)
plt.legend(fontsize=labelsize)
plt.show()

