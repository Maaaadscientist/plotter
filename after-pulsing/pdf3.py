from sympy import exp, cosh, simplify, diff, symbols, Piecewise, acosh, Eq, solve

# Define the new variables and parameters
t, t_gate, tau, t_Ap, tau_Ap, tau_rec, Norm, PH = symbols('t t_gate tau t_Ap tau_Ap tau_rec Norm PH')

# Given expression for PH(t)
PH_expr = 1 + exp(-t_gate / tau) - 2 * exp(-t_gate / (2 * tau)) * cosh((t_gate / 2 - t) / tau)
dPH_dt = diff(PH_expr, t)

print(dPH_dt)
inverse_solution = solve(Eq(PH_expr, PH), t)

print(inverse_solution)
# Attempt to simplify the expression for dPH/dt
#simplified_dPHdt = simplify(dPH_dt)

# Define t_Ap(PH)
#t_Ap_PH_expr = t_gate / 2 - tau * acosh((exp(t_gate / (2 * tau)) * (1 - PH) + exp(-t_gate / (2 * tau))) / 2)

# Define the function f_Ap(t_Ap) and calculate dP_ap/dt
#f_Ap = Piecewise(
#    ((1 - exp(-t_Ap/tau_rec)) * exp(-t_Ap/tau_Ap) / Norm, (t_Ap > 0) & (t_Ap < t_gate)),
#    (0, True)
#)
Norm = (tau_Ap - exp(-t_gate/tau_Ap)*(tau_Ap + tau_rec*(1 - exp(-t_gate/tau_rec)))) / (tau_Ap * (tau_Ap + tau_rec))
f_Ap = (1 - exp(-t/tau_rec)) * exp(-t/tau_Ap) / Norm #, (t_Ap > 0) & (t_Ap < t_gate)

dPap_dt = f_Ap / Norm / dPH_dt
print(dPap_dt)
