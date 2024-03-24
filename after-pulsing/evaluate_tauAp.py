import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def f_PH_function_old(PH, tau, tau_Ap, tau_rec, t_gate):
    inner_log = (2 * np.exp(t_gate/tau) / (-PH * np.exp(t_gate/tau) +
                np.sqrt(PH**2 * np.exp(2*t_gate/tau) - 2*PH * np.exp(2*t_gate/tau) -
                2*PH * np.exp(t_gate/tau) + np.exp(2*t_gate/tau) - 2*np.exp(t_gate/tau) + 1) +
                np.exp(t_gate/tau) + 1))
    t_from_PH = tau * np.log(inner_log)
    return (tau * tau_Ap**2 * (1 - np.exp(-t_from_PH/tau_rec)) * (tau_Ap + tau_rec)**2 *
            np.exp(-t_from_PH/tau_Ap) * np.exp(t_gate/(2*tau)) / 
            (2 * (tau_Ap - (tau_Ap + tau_rec*(1 - np.exp(-t_gate/tau_rec))) * 
            np.exp(-t_gate/tau_Ap))**2 * np.sinh((-t_from_PH + t_gate/2)/tau)))

def f_PH_function(PH, tau, tau_Ap, tau_rec, t_gate, gain=10):
    # Calculate the original upper limit
    original_upper_limit = (1 - np.exp(-t_gate / (2 * tau))) ** 2
    
    # Scale PH to the new range [0, gain)
    scaled_PH = PH / gain * original_upper_limit
    
    # Ensure that the scaled_PH does not exceed the new upper limit, which is 'gain'
    #scaled_PH = min(scaled_PH, gain * 0.999)  # To prevent it from reaching the singularity
    
    inner_log = (2 * np.exp(t_gate / tau) / (-scaled_PH * np.exp(t_gate / tau) +
                np.sqrt(scaled_PH ** 2 * np.exp(2 * t_gate / tau) - 2 * scaled_PH * np.exp(2 * t_gate / tau) -
                2 * scaled_PH * np.exp(t_gate / tau) + np.exp(2 * t_gate / tau) - 2 * np.exp(t_gate / tau) + 1) +
                np.exp(t_gate / tau) + 1))
    t_from_PH = tau * np.log(inner_log)
    
    # Calculate the charge Q
    Q = (tau * tau_Ap ** 2 * (1 - np.exp(-t_from_PH / tau_rec)) * (tau_Ap + tau_rec) ** 2 *
         np.exp(-t_from_PH / tau_Ap) * np.exp(t_gate / (2 * tau)) / 
         (2 * (tau_Ap - (tau_Ap + tau_rec * (1 - np.exp(-t_gate / tau_rec))) * 
         np.exp(-t_gate / tau_Ap)) ** 2 * np.sinh((-t_from_PH + t_gate / 2) / tau)))
    return Q


def integral_f_PH(tau, tau_Ap, tau_rec, t_gate, PH_min, PH_max):
    result, _ = quad(lambda PH: f_PH_function(PH, tau, tau_Ap, tau_rec, t_gate), PH_min, PH_max)
    return result

# Fixed parameters
t_gate = 45
tau_rec = 2
tau = 8.336#15#0.001

# Adjusting the range of PH for plotting
PH_min_plot = 0
PH_max_plot = 9.99#(1 - np.exp(-t_gate/(2*tau)))**2 - 0.001
PH_values = np.linspace(PH_min_plot, PH_max_plot, 500)

# Values of tau_Ap to evaluate
tau_Ap_values = [1,2, 5, 10, 15, 20, 50]
tau_rec_values = [2,3,4, 5, 10]
tau_values = [5,10,15,20,25,30]
plt.figure(figsize=(12, 6))
#tau_Ap = 20
#for tau in tau_values:
#    f_PH_values = np.array([f_PH_function(PH, tau, tau_Ap, tau_rec, t_gate) for PH in PH_values])
#    integral_f = integral_f_PH(tau, tau_Ap, tau_rec, t_gate, PH_min_plot, PH_max_plot)
#    f_PH_values_normalized_by_integral = f_PH_values / integral_f
#
#    # Plotting the normalized function f versus PH by the integral for each tau_Ap
#    plt.plot(PH_values, f_PH_values_normalized_by_integral, label=f"$\\tau = {tau}$")
#
for tau_Ap in tau_Ap_values:
    f_PH_values = np.array([f_PH_function(PH, tau, tau_Ap, tau_rec, t_gate) for PH in PH_values])
    integral_f = integral_f_PH(tau, tau_Ap, tau_rec, t_gate, PH_min_plot, PH_max_plot)
    f_PH_values_normalized_by_integral = f_PH_values / integral_f

#    # Plotting the normalized function f versus PH by the integral for each tau_Ap
    plt.plot(PH_values, f_PH_values_normalized_by_integral, label=f"tau_Ap = {tau_Ap}")
#tau_Ap = 20
#for tau_rec in tau_rec_values:
#    f_PH_values = np.array([f_PH_function(PH, tau, tau_Ap, tau_rec, t_gate) for PH in PH_values])
#    integral_f = integral_f_PH(tau, tau_Ap, tau_rec, t_gate, PH_min_plot, PH_max_plot)
#    f_PH_values_normalized_by_integral = f_PH_values / integral_f
#
#    # Plotting the normalized function f versus PH by the integral for each tau_Ap
#    plt.plot(PH_values, f_PH_values_normalized_by_integral, label=f"tau_rec = {tau_rec}")

plt.title("Normalized Plot of the Function vs. PH (by Integral) for Different tau_Ap")
plt.xlabel("PH")
plt.ylabel("Normalized Function Value")
#plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

