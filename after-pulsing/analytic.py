import numpy as np
import matplotlib.pyplot as plt

def after_pulsing_probability(t, tau_recharge, tau_decay):
    return (1 - np.exp(-t / tau_recharge)) * np.exp(-t) / tau_decay


# Parameters
tau_recharge = 4  # Example value
tau_decay = 15.0     # Example value
t_recharge = 1.0    # Time at which recharge ends and decay starts

# Time points
t_values = np.linspace(0, 10, 500)  # Adjust the range and number of points as needed

# Probability values
p_values = np.array([after_pulsing_probability(t, tau_recharge, tau_decay) for t in t_values])
# Inserting forty-five zeros at the beginning
p_values = np.concatenate((np.zeros(45), p_values))
t_values = np.linspace(0, 10* 545/500, 545)

# Moving integral window size
N = 45

# Moving integral using convolution
window = np.ones(N) / N
integrated_p_values = np.convolve(p_values, window, mode='same')

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t_values, p_values)
plt.xlabel('Time')
plt.ylabel('Probability')
plt.title('After Pulsing Probability over Time')

plt.subplot(1, 2, 2)
plt.plot(t_values, integrated_p_values)
plt.xlabel('Time')
plt.ylabel('Integrated Probability')
plt.title(f'Moving Integral of Probability (Window = {N} points)')
plt.tight_layout()
plt.show()

