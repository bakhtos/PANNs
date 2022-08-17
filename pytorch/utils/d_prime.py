import numpy as np
import scipy.stats

def d_prime(auc):
    d_prime = scipy.stats.norm().ppf(auc) * np.sqrt(2.0)
    return d_prime
