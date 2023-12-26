import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

# Define the parameter values
Ped = 0    # Replace with your specific value
Gain = 42   # Replace with your specific value
mu = 2    # Replace with your specific value
lambda_ = 0.3 # Replace with your specific value
sigma0 = 5  # Replace with your specific value
sigmak = 2  # Replace with your specific value
n_max = 15   # Replace with your specific value for the maximum n

# Define the functions
def gauss(Q, n, sigma_n, Ped, Gain):
    return np.exp(-((Q - (Ped + n * Gain)) ** 2) / (2 * sigma_n ** 2)) / (np.sqrt(2 * np.pi) * sigma_n)

def poisson(n, mu, lambda_):
    return (mu * (mu + n * lambda_) ** (n - 1) * np.exp(-mu - n * lambda_)) / factorial(n)

def sigma_n(n, sigmak, sigma0):
    return np.sqrt(n * sigmak**2 + sigma0**2)

# Define the PDF
def compound_pdf(Q, Ped, Gain, mu, lambda_, sigma0, sigmak, n_max):
    total_pdf = np.zeros_like(Q)
    for n in range(n_max + 1):
        total_pdf += poisson(n, mu, lambda_) * gauss(Q, n, sigma_n(n, sigmak, sigma0), Ped, Gain)
    return total_pdf

# Create a range of Q values
Q_values = np.linspace(-10, 990, 1000)  # Adjust the range and number of points as needed

# Calculate the PDF
pdf_values = compound_pdf(Q_values, Ped, Gain, mu, lambda_, sigma0, sigmak, n_max)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(Q_values, pdf_values, label="Compound PDF")
plt.title("Compound Probability Density Function")
plt.xlabel("Q")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()

