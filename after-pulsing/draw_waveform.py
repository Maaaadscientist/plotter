import numpy as np
import matplotlib.pyplot as plt

# Constants and Parameters
R_s = 1  # Series resistance, Ohm
C_J = 1e-9  # Junction capacitance, Farads
R_Q = 10  # Quenching resistance, Ohm
V_BIAS = 5  # Bias voltage, Volts
V_BD = 1  # Breakdown voltage, Volts
Delta_V = V_BIAS - V_BD  # Voltage difference, Volts
e = 1.6e-19  # Charge of an electron, Coulombs

# Time Constants
tau_rec = R_Q * C_J  # Pixel recharge time constant
tau_ap = 2 * tau_rec  # Afterpulsing time constant, example value

# Time array
t = np.linspace(0, 10 * tau_rec, 1000)  # Time from 0 to 10*tau_rec

# Primary signal
leading_edge = 1 - np.exp(-t / (R_s * C_J))
i_max = Delta_V / (R_Q + R_s)
pulse = i_max * leading_edge * np.exp(-t / (R_Q * C_J))

# Afterpulsing signal
amplitude_afterpulse = 1 - np.exp(-t / tau_rec)
probability_afterpulse = np.exp(-t / tau_ap)  # Normalization can be adjusted if needed
afterpulse = amplitude_afterpulse * probability_afterpulse

# Combined signal
combined_signal = pulse + afterpulse  # This is a simplification

# Plotting the signals
plt.figure(figsize=(12, 8))
plt.plot(t, pulse, label='Primary Pulse')
plt.plot(t, afterpulse, label='Afterpulse')
plt.plot(t, combined_signal, label='Combined Signal')
plt.title('SiPM Signal and Afterpulsing Effect')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

