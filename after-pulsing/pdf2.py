from sympy import exp, cosh, simplify, diff, symbols, Piecewise, acosh

# Define the new variables and parameters
t, t_gate, tau, t_Ap, tau_Ap, tau_rec, Norm, PH = symbols('t t_gate tau t_Ap tau_Ap tau_rec Norm PH')

# Given expression for PH(t)
PH_expr = 1 + exp(-t_gate / tau) - 2 * exp(-t_gate / (2 * tau)) * cosh((t_gate / 2 - t) / tau)
dPH_dt = diff(PH_expr, t)

# Attempt to simplify the expression
simplified_dPHdt = simplify(dPH_dt)

print("dPH/dt:", simplified_dPHdt)

# Define the function f_Ap(t_Ap)
f_Ap = Piecewise(
    ((1 - exp(-t_Ap/tau_rec)) * exp(-t_Ap/tau_Ap) / Norm, (t_Ap > 0) & (t_Ap < t_gate)),
    (0, True)
)

# Define the Norm term
Norm_expr = (tau_Ap - exp(-t_gate/tau_Ap)*(tau_Ap + tau_rec*(1 - exp(-t_gate/tau_rec)))) / (tau_Ap * (tau_Ap + tau_rec))

# Calculate d(P_ap)/dt
dP_ap_dt = f_Ap.subs(Norm, Norm_expr)

print("dP_ap/dt:", dP_ap_dt)

# Define t_Ap(PH)
t_Ap_PH_expr = t_gate / 2 - tau * acosh((exp(t_gate / (2 * tau)) * (1 - PH) + exp(-t_gate / (2 * tau))) / 2)

# Substitute t_Ap with t_Ap(PH) in f_Ap
f_Ap_PH = f_Ap.subs(t_Ap, t_Ap_PH_expr)

# Define Norm with t_Ap(PH)
Norm_PH_expr = Norm_expr.subs(t_Ap, t_Ap_PH_expr)

# Calculate dP_ap/dt with t_Ap(PH)
dP_ap_dt_PH = f_Ap_PH.subs(Norm, Norm_PH_expr)

print("f_Ap as a function of PH:", f_Ap_PH)
print("dP_ap/dt as a function of PH:", dP_ap_dt_PH)

