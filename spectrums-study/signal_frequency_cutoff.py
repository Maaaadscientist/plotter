import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# Define the time parameters
t0 = 0
tau0 = 10e-9  # 10 ns
tau1 = 40e-9  # 40 ns
amp_max = 1.0  # You can adjust this as needed

# Define the time vector
t = np.linspace(0, 500e-9, 10000)  # Adjust the time span as needed

# Define the original signal
original_signal = amp_max * (1 - np.exp(-(t - t0) / tau0)) * np.exp(-(t - t0) / tau1)

# Perform Fourier transform
signal_fft = fft(original_signal)

# Define ADC frequency cutoff (several hundred MHz)
adc_cutoff_frequency = 50e6  # 500 MHz

# Calculate the corresponding cutoff index
cutoff_index = int(len(signal_fft) * adc_cutoff_frequency / (1 / (t[1] - t[0])))

# Set high-frequency components to zero
signal_fft[cutoff_index:] = 0

# Perform the inverse Fourier transform to get the filtered signal
filtered_signal = ifft(signal_fft)

# Plot the original and filtered signals
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, original_signal)
plt.title("Original Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(t, filtered_signal)
plt.title("Filtered Signal (After ADC Cutoff)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

