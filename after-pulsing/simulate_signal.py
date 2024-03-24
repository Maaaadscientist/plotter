import numpy as np
import random
import matplotlib.pyplot as plt

# Given parameters
R_s = 1  # Series resistance, Ohm
C_J = 1e-9  # Junction capacitance, Farads
R_Q = 10  # Quenching resistance, Ohm
V_BIAS = 5  # Bias voltage, Volts
V_BD = 1  # Breakdown voltage, Volts
Delta_V = V_BIAS - V_BD  # Voltage difference, Volts
e = 1.6e-19  # Charge of an electron, Coulombs

# AC coupling time constant
tau_ac = 3 * R_Q * C_J

# Function to apply high-pass filter (AC coupling effect)
def apply_ac_coupling(signal, t, tau_ac):
    # Differential equation: Vout' = (Vin - Vout) / tau_ac
    dt = np.diff(t)[0]  # Time step
    filtered_signal = np.zeros_like(signal)
    for i in range(1, len(signal)):
        filtered_signal[i] = filtered_signal[i-1] + dt/tau_ac * (signal[i] - filtered_signal[i-1])
    return filtered_signal

# Function to generate a pulse
def generate_pulse(t, i_max):
    leading_edge = 1 - np.exp(-t / (R_s * C_J))
    return i_max * leading_edge * np.exp(-t / (R_Q * C_J))

# Function to calculate the maximum current of an afterpulse at time t
def calculate_i_max_ap(t, i_max_primary):
    amp_ap = (1 - np.exp(-t / (R_Q * C_J))) * i_max_primary
    return amp_ap

# Function to pick a random time for an afterpulse based on its probability distribution
def pick_afterpulse_time(t, tau_ap):
    probabilities = np.exp(-t / tau_ap) / tau_ap  # Probability distribution of afterpulses over time
    probabilities /= np.sum(probabilities)  # Normalize probabilities
    return np.random.choice(t, p=probabilities)

# Time array
t = np.linspace(0, 10 * R_Q * C_J, 1000)  # Time from 0 to 10*tau_rec

# Calculate maximum current for the primary pulse
i_max_primary = Delta_V / (R_Q + R_s)
# Generate the primary pulse
primary_pulse = generate_pulse(t, i_max_primary)

# Parameters for the afterpulse
tau_ap = 2 * R_Q * C_J  # Afterpulsing time constant, example value

while True:
    # Pick a random time for the afterpulse
    t_ap = pick_afterpulse_time(t, tau_ap)
     
    # Calculate the i_max for the afterpulse at the picked time
    i_max_ap = calculate_i_max_ap(t_ap, i_max_primary)
    
    # Generate the afterpulse waveform
    afterpulse_waveform = generate_pulse(t - t_ap, i_max_ap) 
    
    # Change negative values of the afterpulse waveform to zero
    afterpulse_waveform[afterpulse_waveform < 0] = 0
    
    random_waveform = primary_pulse + afterpulse_waveform
    # Plotting the random waveform
    plt.figure(figsize=(8, 6))
    plt.plot(t, random_waveform, label='Primary + Afterpulse', zorder = 3)
    plt.plot(t, afterpulse_waveform, label='Afterpulse',zorder=2)
    plt.plot(t, primary_pulse, label='Primary)',zorder=1)
    plt.title('Simulated SiPM Random Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Apply AC coupling to the primary and afterpulse waveforms
    primary_pulse_ac = apply_ac_coupling(primary_pulse, t, tau_ac)
    afterpulse_waveform_ac = apply_ac_coupling(afterpulse_waveform, t, tau_ac)
    
    # Recalculate the random waveform with AC coupling
    random_waveform_ac = primary_pulse_ac + afterpulse_waveform_ac
    
    # Plotting the random waveform with AC coupling
    plt.figure(figsize=(8, 6))
    plt.plot(t, random_waveform_ac, label='Primary + Afterpulse', zorder = 3)
    plt.plot(t, afterpulse_waveform_ac, label='Afterpulse',zorder=2)
    plt.plot(t, primary_pulse_ac, label='Primary)',zorder=1)
    plt.title('Simulated SiPM Random Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()
