import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


labelsize=18
titlesize=24
textsize=20

color_404 ="blue"
color_415 ="red"
# Function to read data from a file
def read_data(file_name):
    pde_values = []
    errors = []
    with open(file_name, 'r') as file:
        for line in file:
            pde, error = map(float, line.split())
            pde_values.append(pde)
            errors.append(error)
    return pde_values, errors

# Read data for both wavelengths
pde_404, error_404 = read_data('pde_404nm.txt')
pde_415, error_415 = read_data('pde_415nm.txt')

# Assuming the over voltages are the same for both and are in order
#over_voltages = [1, 2, 3, 4, 5, 6]
over_voltages = [vol - 47.17 for vol in range(48,54)]  # Replace with actual over voltage values

# Interpolating the PDE values
interp_404 = interp1d(over_voltages, pde_404, kind='cubic')
interp_415 = interp1d(over_voltages, pde_415, kind='cubic')

# Evaluate the interpolation at OV = 3V
pde_404_at_5V = interp_404(5)
pde_415_at_5V = interp_415(5)

# Print the PDE values at OV = 3V for comparison
print(f"PDE at OV = 5V for 404 nm wavelength: {pde_404_at_5V}")
print(f"PDE at OV = 5V for 415 nm wavelength: {pde_415_at_5V}")

# Creating the plot
plt.figure(figsize=(10, 6))

#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "serif",
#    #"font.family": "helvet",
#    "text.latex.preamble": r"\usepackage{courier}",
#})


# Plotting data and interpolation for 404 nm
plt.errorbar(over_voltages, pde_404, yerr=error_404, linestyle='-', label='404 nm Data', capsize=3, markersize=4, linewidth=1.5,marker="s", color=color_404,markerfacecolor='none')
#plt.plot(over_voltages, interp_404(over_voltages), label='404 nm Interpolation')

# Plotting data and interpolation for 415 nm
plt.errorbar(over_voltages, pde_415, yerr=error_415, linestyle='--', label='415 nm Data', capsize=3,markersize=4, linewidth=1.5, marker="o", color=color_415,markerfacecolor='none')
#plt.plot(over_voltages, interp_415(over_voltages), label='415 nm Interpolation')

# Draw horizontal lines for interpolated PDE values
plt.hlines(y=pde_404_at_5V, xmin=0, xmax=5, color=color_404, linestyle=':', alpha=0.5)
plt.hlines(y=pde_415_at_5V, xmin=0, xmax=5, color=color_415, linestyle=':', alpha=0.5)
#plt.text(4, pde_404_at_5V - 0.06, f'{pde_404_at_5V:.2f}', verticalalignment='top', horizontalalignment='right', color=color_404,fontsize=textsize)
plt.text(5.5, pde_415_at_5V - 0.2, f'PDE difference = \n ({pde_415_at_5V:.3f} - {pde_404_at_5V:.3f})/{pde_404_at_5V:.3f} = {(pde_415_at_5V - pde_404_at_5V)/pde_404_at_5V * 100:.1f}%', verticalalignment='top', horizontalalignment='right', color="black",fontsize=textsize)

# Adding arrows to indicate the difference
arrow_offset = 0.005
#plt.annotate('', xy=(3, pde_404_at_5V - arrow_offset), xytext=(3, pde_415_at_5V + arrow_offset),
#             arrowprops=dict(arrowstyle="<->", lw=1, color="black"))


# Adding labels and title

plt.ylim(0,0.6)
plt.xlim(0,7)
plt.xlabel('Over Voltage (V)',fontsize=labelsize)
plt.ylabel('PDE',fontsize=labelsize)
plt.title('PDE vs Over Voltage', fontsize=titlesize)
plt.legend(fontsize=labelsize, loc='lower right')

plt.xticks(fontsize=labelsize)  # Adjust font size for x-axis ticks
plt.yticks(fontsize=labelsize)  # Adjust font size for y-axis ticks

# Show plot
#plt.grid(True)
plt.show()

