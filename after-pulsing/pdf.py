from sympy import exp, cosh, simplify, diff,symbols, Piecewise

# Define the new variables and parameters
t, t_gate, tau, t_Ap, tau_Ap, tau_rec, Norm = symbols('t t_gate tau t_Ap tau_Ap tau_rec Norm')


# Given expression for PH(t)
PH_expr = 1 + exp(-t_gate / tau) - 2 * exp(-t_gate / (2 * tau)) * cosh((t_gate / 2 - t) / tau)
dPH_dt = diff(PH_expr, t)

# Attempt to simplify the expression
simplified_dPHdt = simplify(dPH_dt)

print(simplified_dPHdt)
# Define the function f_Ap(t_Ap)
f_Ap = Piecewise(
    ((1 - exp(-t_Ap/tau_rec)) * exp(-t_Ap/tau_Ap) / Norm, (t_Ap > 0) & (t_Ap < t_gate)),
    (0, True)
)

# Define the Norm term
Norm_expr = (tau_Ap - exp(-t_gate/tau_Ap)*(tau_Ap + tau_rec*(1 - exp(-t_gate/tau_rec)))) / (tau_Ap * (tau_Ap + tau_rec))

# Calculate d(P_ap)/dt
dP_ap_dt = f_Ap.subs(Norm, Norm_expr)

print(dP_ap_dt)
# Calculate d(P_ap)/d(PH)
#dP_ap_dPH = dP_ap_dt / dPH_dt

#dP_ap_dPH.simplify()

