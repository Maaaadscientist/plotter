import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  # Import colormap module
from matplotlib.colors import Normalize  # Import the Normalize class

import scienceplots
plt.style.use('science')
plt.style.use('nature')

labelsize=28
titlesize=40
textsize=24

# Assuming your CSV data is in a string for demonstration; replace this with file reading if needed

# Convert the string to a file-like object and read it into a DataFrame
from io import StringIO
#df = pd.read_csv(StringIO(csv_data), header=None, names=['mu', 'lambda', 'ap', 'enf_residual'])
df = pd.read_csv("ap_table.csv")

# Define the sets of mu and lambda you're interested in
mu_values = [0.5, 1, 1.5, 2, 2.5, 3]
lambda_values = np.arange(0.05, 0.45, 0.05)#[0.1, 0.2, 0.3, 0.4, 0.5]
lambda_values_map = np.arange(-0.2, 0.5, 0.02)#[0.1, 0.2, 0.3, 0.4, 0.5]
# Create a colormap and a normalization instance
#cmap = cm.viridis  # Choose the colormap
cmap = cm.magma
norm = Normalize(vmin=min(lambda_values_map), vmax=max(lambda_values_map))  # Normalize lambda values


# Function to plot for a specific mu and lambda
def plot_for_parameters(df, mu, lambda_val, color):
    subset = df[(df['lambda'] == lambda_val)]
    #plt.plot( subset['enf_residual'].to_numpy(), subset['ap'].to_numpy(), label='$\\lambda$'+f'={lambda_val}',linewidth=3)
    plt.plot(subset['enf_residual'].to_numpy(), subset['ap'].to_numpy(),
             label='$\\lambda$='+f'{lambda_val}', color=color, linewidth=3)



# Now fix mu (e.g., 1) and vary lambda
plt.figure(figsize=(20, 15))
print(lambda_values)
for lambda_val in lambda_values:
    color = cmap(norm(lambda_val))
    plot_for_parameters(df, 1, round(lambda_val,3), color)
plt.axhline(y=1/3, color='red', linestyle='--', label='$P_\\mathrm{ap}$ = 1/3', linewidth=2)

plt.xticks(fontsize=labelsize)
plt.yticks(fontsize=labelsize)
plt.ylabel('AP Probability', fontsize=labelsize, labelpad=15)
plt.xlabel('ENF residual',fontsize=labelsize, labelpad=15)
plt.title('After-pulsing vs. ENF residuals',fontsize=titlesize, pad=20)
plt.legend(fontsize=labelsize, loc='upper right')
plt.grid(True)
plt.savefig("enf_residual_vs_ap.pdf")

