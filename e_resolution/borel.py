import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
import scienceplots
#plt.style.use('science')

plt.rcParams.update({'font.size': 14})  # Sets default font size
plt.rcParams['axes.titlesize'] = 14  # Title size
plt.rcParams['axes.labelsize'] = 12  # Axis labels
plt.rcParams['xtick.labelsize'] = 10 # X-tick labels
plt.rcParams['ytick.labelsize'] = 10 # Y-tick labels
plt.style.use('nature')
def borel_pmf(k, lambda_):
    return (lambda_ * k)**(k - 1) * np.exp(-k * lambda_) / factorial(k)

lambdas = [0.1, 0.2, 0.3]
colors = ['blue', 'green', 'red']  # Different color for each lambda
k_values_plot = np.arange(1, 6)  # Points to plot
k_values_calc = np.arange(0, 51)  # Points for calculation

plt.figure(figsize=(8, 6))

for lambda_, color in zip(lambdas, colors):
    pmf_values_plot = borel_pmf(k_values_plot, lambda_)
    print(pmf_values_plot)
    plt.plot(k_values_plot - 1, pmf_values_plot, label=f'λ = {lambda_}', color=color, marker='o')

    # Calculating approximate expected value and standard deviation using more points
    pmf_values_calc = borel_pmf(k_values_calc[1:], lambda_)
    expected_value = np.sum(k_values_calc[:-1] * pmf_values_calc)
    variance = np.sum((k_values_calc[:-1]**2) * pmf_values_calc) - expected_value**2
    #std_deviation = np.sqrt(variance)
    std_deviation = variance

    # Adjusting annotation position
    y_pos = lambda_*1.2 + 0.05  # Slight vertical adjustment for each lambda
    plt.text(3.5, y_pos, f'λ = {lambda_}\nE(X) ≈ {expected_value:.2f}\nD(X) ≈ {std_deviation:.2f}', color=color)

plt.title('Borel Distribution PMF for Various λ')
plt.xlabel('k')
plt.ylabel('Probability')
plt.legend()
plt.grid("True")
plt.show()

