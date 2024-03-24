from math import exp, factorial
import numpy as np
import math
from scipy.special import comb
import time

def generalized_poisson_pmf(n, mu, lam):
    """Calculate Generalized Poisson PMF."""
    return mu * (mu + lam * n)**(n - 1) * exp(-mu - n * lam) / factorial(n)

def binomial_pmf(n, k, p):
    """Calculate Binomial PMF."""
    return comb(n, k) * (p**k) * ((1 - p)**(n - k))

def compound_pmf(i, j, mu, lam, p):
    """Calculate the compound PMF for given j, mu, lambda, and p."""
    if i == 0:
        return generalized_poisson_pmf(0, mu, lam)
    elif i >= 1 and j <= i:
        return generalized_poisson_pmf(i, mu, lam) * binomial_pmf(i, j, p)
    else:
        print("j should equal to or be less than i")
        return 0.

# Start measuring time
start_time = time.time()

# Calculate the total number of iterations
ct_step = 0.01
ct_max = 1 - ct_step
ap_step = 0.005
ap_max = 1 - ap_step
gain_step = 0.01
gain_max = 1 - gain_step
total_iterations = len(np.arange(ct_step, ct_max, ct_step)) * len(np.arange(ap_step, ap_max, ap_step)) * len(np.arange(gain_step, gain_max, gain_step))
current_iteration = 0

# Example usage
mu = 1
for single_gain in np.arange(gain_step, gain_max, gain_step):
    for lambda_ in np.arange(ct_step, ct_max, ct_step):
        for pap in np.arange(ap_step,ap_max,ap_step):
            current_iteration += 1
            mean = 0
            var = 0
            for i in range(100):
                if i == 0:
                    mean += compound_pmf(0,0, mu, lambda_, pap)
                for j in range(i+1):
                    pmf = compound_pmf(i,j, mu, lambda_, pap)
                    mean += pmf * (i + single_gain * j)
            
            for i in range(100):
                if i == 0:
                    var += compound_pmf(0,0, mu, lambda_, pap) * (0 - mean)**2
                for j in range(i+1):
                    pmf = compound_pmf(i,j, mu, lambda_, pap)
                    var += pmf * (i + single_gain *j -mean)**2
    
            enf_diff = round(mu * var / mean**2 - 1 / (1 - lambda_), 7)
            pap = round(pap, 6)
            lambda_ = round(lambda_, 5)
            mu = round(mu, 5)
            single_gain = round(single_gain, 2)
            #print(f"mu: {mu}, lambda:{lambda_}, pap: {pap}, enf_diff: {enf_diff}")
            #print(f"{mu},{lambda_},{pap},{enf_diff}")
            with open('gain_ap_table.csv', 'a') as file:
                file.write(f"{single_gain},{lambda_},{pap},{enf_diff}\n")
                        # Periodically check time and estimate total time
            if current_iteration % 100 == 0:  # Adjust the frequency of time checks here
                elapsed_time = time.time() - start_time
                estimated_total_time = (elapsed_time / current_iteration) * total_iterations
                remaining_time = estimated_total_time - elapsed_time
                print(f"Completed {current_iteration}/{total_iterations} iterations. Estimated remaining time: {remaining_time:.2f} seconds")



