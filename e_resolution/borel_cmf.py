import numpy as np
from math import factorial

def borel_pmf(k, lambda_):
    return (lambda_ * k)**(k - 1) * np.exp(-k * lambda_) / factorial(k)

#lambda_values = np.arange(0.05,1,0.05)
lambda_values = [0.08398, 0.1411, 0.1998, 0.2304, 0.3179, 0.3817]
for lambda_ in lambda_values:
    borel_cmf = 0
    borel_cmf += borel_pmf(1, lambda_) * 0.5
    for i in range(2,100):
        borel_cmf += borel_pmf(i, lambda_)
    print(round(lambda_, 2), (1/ borel_cmf), borel_cmf ,(2 / (2- np.exp(-lambda_))))

        
        
