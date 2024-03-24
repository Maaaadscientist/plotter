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

pmf = 0.
for i in range(100):
    for j in range(i+1):
        pmf += compound_pmf(i, j, 1, 0.5, 0.8)
print(pmf)
