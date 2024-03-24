import ROOT
import numpy as np
import matplotlib.pyplot as plt
# Create a RooPlot object for PH
from scipy.integrate import quad

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

epsilon = 1e-3
ph_min = 0
ph_max = 10 - epsilon
tau_initial_value = 8.336
tau_Ap_initial_value = 20
tau_rec_initial_value = 5
t_gate_initial_value = 45
gain_initial_value = 10
# Create RooRealVar objects for each of your parameters and the variable
PH = ROOT.RooRealVar("PH", "PH", ph_min, ph_max)
sigQ = ROOT.RooRealVar("sigQ", "sigQ", ph_min, ph_max*2)
tau = ROOT.RooRealVar("tau", "tau", tau_initial_value, tau_initial_value -2, tau_initial_value + 2)
tau_Ap = ROOT.RooRealVar("tau_Ap", "tau_Ap", tau_Ap_initial_value, tau_Ap_initial_value - 2, tau_Ap_initial_value + 2)
tau_rec = ROOT.RooRealVar("tau_rec", "tau_rec", tau_rec_initial_value)
t_gate = ROOT.RooRealVar("t_gate", "t_gate", t_gate_initial_value)
gain = ROOT.RooRealVar("gain", "gain", gain_initial_value)

stepFunctionLow = ROOT.RooFormulaVar("stepFunctionLow", "TMath::Sign(1., @0 - 0) ", ROOT.RooArgList(sigQ));
stepFunctionHigh = ROOT.RooFormulaVar("stepFunctionHigh", f"TMath::Sign(1., @1 - {epsilon} -  @0)", ROOT.RooArgList(sigQ, gain))
original_upper_limit = ROOT.RooFormulaVar("singularity"," (1 - TMath::Exp(-@0 / (2 * @1))) ** 2", ROOT.RooArgList(t_gate, tau))
scaled_PH = ROOT.RooFormulaVar("scaledQ","@0 / @1 * @2 *@3 * @4", ROOT.RooArgList(sigQ,gain,original_upper_limit, stepFunctionLow, stepFunctionHigh))
inner_log = ROOT.RooFormulaVar("innerLog", "(2 * TMath::Exp(@0 / @1) / (-@2 * TMath::Exp(@0 / @1) + TMath::Sqrt(@2^2 * TMath::Exp(2 * @0 / @1) - 2 * @2 * TMath::Exp(2 * @0 / @1) - 2 * @2 * TMath::Exp(@0 / @1) + TMath::Exp(2 * @0 / @1) - 2 * TMath::Exp(@0 / @1) + 1) + TMath::Exp(@0 / @1) + 1))", ROOT.RooArgList(t_gate, tau, scaled_PH))
t_from_PH = ROOT.RooFormulaVar("t_from_PH", "@0 * TMath::Log(@1)", ROOT.RooArgList(tau, inner_log))
fq = ROOT.RooGenericPdf("fq", "(@0 * @1^2 * (1 - TMath::Exp(-@2 / @3)) * (@1 + @3)^2 * TMath::Exp(-@2 / @1) * TMath::Exp(@4 / (2 * @0)) / (2 * (@1 - (@1 + @3 * (1 - TMath::Exp(-@4 / @3))) * TMath::Exp(-@4 / @1))^2 * 0.5 * (TMath::Exp((-@2 + @4 / 2) / @0)-TMath::Exp(-(-@2 + @4 / 2) / @0))))", ROOT.RooArgList(tau, tau_Ap, t_from_PH, tau_rec, t_gate))

# Define the Heaviside functions
#stepFunctionLow = ROOT.RooFormulaVar("stepFunctionLow", "TMath::Heaviside(@0 - 0)", ROOT.RooArgList(PH));
#stepFunctionLow = ROOT.RooFormulaVar("stepFunctionLow", "(1 + TMath::Sign(1., @0 - 0)) / 2", ROOT.RooArgList(PH));
#stepFunctionHigh = ROOT.RooFormulaVar("stepFunctionHigh", "(1 + TMath::Sign(1., @1 - @0)) / 2", ROOT.RooArgList(PH, gain))

# Modify the PDF to be zero outside the range [0, gain)
#modifiedPDF = ROOT.RooGenericPdf("modifiedPDF", "(@0 * @1 * @2) * fq", ROOT.RooArgList(stepFunctionLow, stepFunctionHigh, fq));
modifiedPDF = ROOT.RooGenericPdf("modifiedPDF", "fq", ROOT.RooArgList(fq));

# Create a convolution PDF (self-convolution)

num_events = 30000
# Generate a dataset
#data = fq.generate(ROOT.RooArgSet(PH), num_events)  # Replace num_events with the desired number of events
data = modifiedPDF.generate(ROOT.RooArgSet(sigQ), num_events)  # Replace num_events with the desired number of events

#fq.fitTo(data)
data_array = data.to_numpy()
print(data_array)
# Optionally, fit the data to your PDF to see how well it describes the generated data
#fit_result = fq.fitTo(data)

# Plotting the data as a histogram
nBins = 50
plt.hist(data_array['sigQ'], bins=nBins, alpha=0.7, label='Data')

# Adjusting the range of PH for plotting
PH_min_plot = ph_min 
PH_max_plot = ph_max  - epsilon#(1 - np.exp(-t_gate/(2*tau)))**2 - 0.001
PH_values = np.linspace(PH_min_plot, PH_max_plot, 500)
f_PH_values = np.array([f_PH_function(PH, tau_initial_value, tau_Ap_initial_value, tau_rec_initial_value, t_gate_initial_value, gain_initial_value) for PH in PH_values])
integral_f = integral_f_PH(tau_initial_value, tau_Ap_initial_value, tau_rec_initial_value, t_gate_initial_value, PH_min_plot, PH_max_plot)
# norm factor = integral_f ^ -1 * num_events * bin_width
f_PH_values_normalized_by_integral = f_PH_values / integral_f * num_events * ((PH_max_plot - PH_min_plot) / nBins)
plt.plot(PH_values, f_PH_values_normalized_by_integral, label=f"tau_Ap = {tau_Ap_initial_value}")

# Evaluate the PDF across a range of values
#x_values = np.linspace(ph_min, ph_max, 500)
#pdf_values = [fq.evaluate(ROOT.RooArgSet(PH)) for PH in x_values]

# Plotting the PDF as a line
#plt.plot(x_values, pdf_values, label='PDF', color='red')

# Adding labels and legend
plt.xlabel('PH')
plt.ylabel('Frequency')
plt.title('Data and PDF')
plt.legend()

# Show the plot
plt.show()
