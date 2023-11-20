import matplotlib.pyplot as plt

labelsize=18
titlesize=24
textsize=20

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
over_voltages = [vol - 47.17 for vol in range(48,54)]  # Replace with actual over voltage values

# Creating the plot
plt.figure(figsize=(10, 6))

#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "serif",
#    #"font.family": "helvet",
#    "text.latex.preamble": r"\usepackage{courier}",
#})

# Plotting data for 404 nm
plt.errorbar(over_voltages, pde_404, yerr=error_404, fmt='-o', label='404 nm', capsize=5)

# Plotting data for 415 nm
plt.errorbar(over_voltages, pde_415, yerr=error_415, fmt='-o', label='415 nm', capsize=5)

# Adding labels and title
plt.xlabel('Over Voltage (V)',fontsize=labelsize)
plt.ylabel('PDE',fontsize=labelsize)
plt.title('Photodetection Efficiency (PDE) vs Over Voltage', fontsize=titlesize)
plt.legend(fontsize=labelsize)

plt.xticks(fontsize=labelsize)  # Adjust font size for x-axis ticks
plt.yticks(fontsize=labelsize)  # Adjust font size for y-axis ticks

# Show plot
plt.grid(True)
plt.show()

