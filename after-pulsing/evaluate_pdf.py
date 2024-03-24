import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
# Defining the function f(PH) with all necessary parameters
def f_PH_function(PH, tau, tau_Ap, tau_rec, t_gate):
    # Defining the inner part of the log function for readability
    inner_log = (2 * np.exp(t_gate/tau) / (-PH * np.exp(t_gate/tau) +
                np.sqrt(PH**2 * np.exp(2*t_gate/tau) - 2*PH * np.exp(2*t_gate/tau) -
                2*PH * np.exp(t_gate/tau) + np.exp(2*t_gate/tau) - 2*np.exp(t_gate/tau) + 1) +
                np.exp(t_gate/tau) + 1))
    
    # Calculating t from PH
    t_from_PH = tau * np.log(inner_log)
    
    # The original function f(t) substituted with t from PH
    return (tau * tau_Ap**2 * (1 - np.exp(-t_from_PH/tau_rec)) * (tau_Ap + tau_rec)**2 *
            np.exp(-t_from_PH/tau_Ap) * np.exp(t_gate/(2*tau)) / 
            (2 * (tau_Ap - (tau_Ap + tau_rec*(1 - np.exp(-t_gate/tau_rec))) * 
            np.exp(-t_gate/tau_Ap))**2 * np.sinh((-t_from_PH + t_gate/2)/tau)))

def integral_f_PH(tau, tau_Ap, tau_rec, t_gate, PH_min, PH_max):
    result, _ = quad(lambda PH: f_PH_function(PH, tau, tau_Ap, tau_rec, t_gate), PH_min, PH_max)
    return result

# Example usage with a specific PH value and the given parameters
#f_PH_value = f_PH_function(PH_example, tau, tau_Ap, tau_rec, t_gate)
t_gate = 45
tau_rec = 5
tau_Ap = 15
tau = 0

# Adjusting the range of PH for plotting
PH_min_plot = 0
PH_max_plot = 0.999 #(1 - np.exp(-t_gate/(2*tau)))**2 - 0.001
PH_values = np.linspace(PH_min_plot, PH_max_plot, 300)

# Calculating the function values for the range of PH
f_PH_values = np.array([f_PH_function(PH, tau, tau_Ap, tau_rec, t_gate) for PH in PH_values])

# Computing the integral of the function over the range of PH
integral_f = integral_f_PH(tau, tau_Ap, tau_rec, t_gate, PH_min_plot, PH_max_plot)

# Normalizing the function values by the integral
f_PH_values_normalized_by_integral = f_PH_values / integral_f

# Plotting the normalized function f versus PH by the integral
plt.figure(figsize=(12, 6))
plt.plot(PH_values, f_PH_values_normalized_by_integral, label="Function vs. PH (Normalized by Integral)")
plt.title("Normalized Plot of the Function vs. PH (by Integral)")
plt.xlabel("PH")
plt.ylabel("Normalized Function Value")
plt.legend()
plt.grid(True)
plt.show()

