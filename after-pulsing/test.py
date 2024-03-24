import ROOT

def generate_and_fit():
    # Importing necessary RooFit components
    from ROOT import RooRealVar, RooConstVar, RooGenericPdf, RooDataSet, RooArgList, RooFit

    # Define the observable
    PH = RooRealVar("PH", "PH", 0, 1)  # Adjust the range as needed

    # Define parameters
    tau_rec = RooRealVar("tau_rec", "tau_rec", 5, 0, 10)  # Adjust the range as needed
    tau_Ap = RooRealVar("tau_Ap", "tau_Ap", 15, 10, 20)  # Adjust the range as needed

    # Fixed parameters (You can change these values as needed)
    t_gate = RooConstVar("t_gate", "t_gate", 45)
    tau = RooConstVar("tau", "tau", 5)

    # Define the formula for your PDF (translated from the C++ version)
    formula = ("@1 * @2^2 * (1 - exp(-@3*log((2 * exp(@4/@0) / (-@5 * exp(@4/@0) + "
              "sqrt(pow(@5, 2) * exp(2*@4/@0) - 2*@5*exp(2*@4/@0) - 2*@5*exp(@4/@0) + exp(2*@4/@0) - 2*exp(@4/@0) + 1) + exp(@4/@0) + 1))/@2)) * "
              "(@2 + @1)^2 * exp(-@3*log((2 * exp(@4/@0) / (-@5 * exp(@4/@0) + "
              "sqrt(pow(@5, 2) * exp(2*@4/@0) - 2*@5*exp(2*@4/@0) - 2*@5*exp(@4/@0) + exp(2*@4/@0) - 2*exp(@4/@0) + 1) + exp(@4/@0) + 1))/@2) * "
              "exp(@4/(2*@0)) / (2 * (@2 - (@2 + @1*(1 - exp(-@4/@1))) * exp(-@4/@2))^2 * "
              "sinh((-@3*log((2 * exp(@4/@0) / (-@5 * exp(@4/@0) + "
              "sqrt(pow(@5, 2) * exp(2*@4/@0) - 2*@5*exp(2*@4/@0) - 2*@5*exp(@4/@0) + exp(2*@4/@0) - 2*exp(@4/@0) + 1) + exp(@4/@0) + 1)) + @4/2)/@0))")

    # Create the PDF from the formula
    f_PH_pdf = RooGenericPdf("f_PH_pdf", "PDF of the function", formula,
                             RooArgList(tau, tau_rec, tau_Ap, PH, t_gate))

    # Generate a dataset
    data = f_PH_pdf.generate(RooArgSet(PH), 10000)  # 10000 events

    # Fit the PDF to the generated data
    f_PH_pdf.fitTo(data)

    # Create a frame and plot the data and the PDF
    frame = PH.frame(ROOT.RooFit.Title("Function vs. PH (Normalized by Integral)"))
    data.plotOn(frame)
    f_PH_pdf.plotOn(frame, RooFit.LineColor(ROOT.kRed))

    # Create a canvas to show the plot
    c = ROOT.TCanvas("c", "Function Plot", 800, 400)
    frame.Draw()
    c.SaveAs("FunctionPlotWithGeneratedData.pdf")  # Save the plot as a PDF file

if __name__ == "__main__":
    generate_and_fit()

