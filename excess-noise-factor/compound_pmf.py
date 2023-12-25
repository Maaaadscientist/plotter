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

def compound_pmf(j, mu, lam, p):
    """Calculate the compound PMF for given j, mu, lambda, and p."""
    if j == 0:
        return generalized_poisson_pmf(0, mu, lam)
    if j == 1:
        return generalized_poisson_pmf(1, mu, lam) * binomial_pmf(1, 0, p)
    if j % 2 == 0:  # For even j
        start, end = int(j // 2), int(j + 1)
    else:           # For odd j
        start, end = int((j + 1) // 2), int(j + 1)

    total = 0
    for i in range(start, end):
        #print(f"C_{j} += P_{i} + B({i},{j-i})")
        total += generalized_poisson_pmf(i, mu, lam) * binomial_pmf(i, j - i, p)
    return total

# Start measuring time
start_time = time.time()

# Calculate the total number of iterations
ct_step = 0.002
ct_max = 1 - ct_step
ap_step = 0.001
ap_max = 1 - ap_step
total_iterations = len(np.arange(ct_step, ct_max, ct_step)) * len(np.arange(ap_step, ap_max, ap_step))
current_iteration = 0

# Example usage
mu = 1
for lambda_ in np.arange(ct_step, ct_max, ct_step):
    for pap in np.arange(ap_step,ap_max,ap_step):
        current_iteration += 1
        mean = 0
        var = 0
        for i in range(100):
            pmf = compound_pmf(i, mu, lambda_, pap)
            mean += pmf * i
        
        for i in range(100):
            pmf = compound_pmf(i, mu, lambda_, pap)
            var += pmf * (i-mean)**2

        enf_diff = round(mu * var / mean**2 - 1 / (1 - lambda_), 7)
        pap = round(pap, 6)
        lambda_ = round(lambda_, 5)
        mu = round(mu, 5)
        #print(f"mu: {mu}, lambda:{lambda_}, pap: {pap}, enf_diff: {enf_diff}")
        #print(f"{mu},{lambda_},{pap},{enf_diff}")
        with open('ap_table.csv', 'a') as file:
            file.write(f"{lambda_},{pap},{enf_diff}\n")
                    # Periodically check time and estimate total time
        if current_iteration % 100 == 0:  # Adjust the frequency of time checks here
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / current_iteration) * total_iterations
            remaining_time = estimated_total_time - elapsed_time
            print(f"Completed {current_iteration}/{total_iterations} iterations. Estimated remaining time: {remaining_time:.2f} seconds")



