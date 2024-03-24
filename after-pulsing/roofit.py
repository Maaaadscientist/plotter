import ROOT
from ROOT import RooRealVar, RooArgList, RooGenericPdf, RooArgSet, RooDataSet, RooFit, TCanvas, RooPlot

# Define the PH function (as a Python callable)
def PH_python(PH, params):
    tau, tau_Ap, tau_rec, t_gate, gain = params
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

# Define a C++ wrapper for the PH function (as ROOT expects C++ functions)
PH_cpp_code = """
double PH_cpp(double* x, double* params) {
    return PH_python(x[0], params);
}
"""

# Compile the C++ code
ROOT.gInterpreter.ProcessLine(".L libRooFit")  # Ensure RooFit libraries are loaded
ROOT.gInterpreter.Declare(PH_cpp_code)

#t_gate = 45
#tau_rec = 5
#tau = 8.336#15#0.001
#tau_Ap = 20
# Set up the RooFit environment
gain = 10
x = RooRealVar("x", "PH", 0, gain - 0.001)  # PH range
tau = RooRealVar("tau", "tau", 8.336)  # Example parameter ranges
tau_Ap = RooRealVar("tau_Ap", "tau_Ap", 20)
tau_rec = RooRealVar("tau_rec", "tau_rec",5)
t_gate = RooRealVar("t_gate", "t_gate", 45)
params = RooArgList(tau, tau_Ap, tau_rec, t_gate)

# Create the RooFit PDF from the C++ function
#PH_pdf = RooCustomPdf("PH_pdf", "PH_cpp(x, {tau, tau_Ap, tau_rec, t_gate})", RooArgList(x, params))
PH_pdf = RooGenericPdf("PH_pdf", "PH_cpp(x, {tau, tau_Ap, tau_rec, t_gate})", RooArgList(x, params))

# Generate random data from the PDF
data = PH_pdf.generate(RooArgSet(x), 1000)  # Generate 1000 events

# Plot the generated data
canvas = TCanvas("canvas", "PH Function", 800, 600)
frame = x.frame(RooFit.Title("Generated Photoelectron Data"))
data.plotOn(frame)
frame.Draw()
canvas.Draw()

