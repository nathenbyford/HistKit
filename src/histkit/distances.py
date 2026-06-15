import numpy as np
from .hist import Histogram

def wasserstein_distance(h1: Histogram, h2: Histogram, p: int = 2) -> float:
    """
    Computes the Wasserstein-p distance between two histograms.
    
    This implementation treats the histograms as quantile functions (inverse CDFs) 
    defined on the range [0,1].
    
    The Wasserstein-p distance is defined as the integral of the 
    difference between the quantile functions to the power p of the 
    two distributions:
    W_p = ( integral_0^1 |F_1^-1(t) - F_2^-1(t)|^p dt )^(1/p)
    
    Args:
        h1 (Histogram): First histogram.
        h2 (Histogram): Second histogram.
        p: power of the integrand.
        
    Returns:
        float: The squared Wasserstein-p distance.
    """
    # Get all unique CDF breaks
    all_breaks = np.union1d(h1.breaks, h2.breaks)
    
    # Compute quantile values at all breaks
    q1 = np.interp(all_breaks, h1.cdf, h1.breaks[1:])
    q2 = np.interp(all_breaks, h2.cdf, h2.breaks[1:])
    
    # Differences
    dt = np.diff(all_breaks)

    # Calculate value with special case for p=2 using np.sqrt
    integral = np.dot(np.abs(q1[:-1] - q2[:-1]) ** p, dt)
    return float(np.power(integral, 1/p))

def kl_divergence(h1: Histogram, h2: Histogram, epsilon: float = 1e-12):
    """
    Computes the Kullback Leibler (KL) Divergence of two histograms.

    This implementation treats the histograms as probability density.

    The KL divergence is defined as the sum of the scaled log ratio
    of the two distribution functions across the range of values:
    KL = sum_1^n p_i log( p_i / q_i )
    """
    # Get all unique CDF breaks
    all_breaks = np.union1d(h1.breaks, h2.breaks)

    # Compute all CDF values
    cdf1 = np.interp(all_breaks, h1.breaks[1:], h1.cdf, left=0.0, right=1.0)
    cdf2 = np.interp(all_breaks, h2.breaks[1:], h2.cdf, left=0.0, right=1.0)

    p = np.diff(cdf1)
    q = np.diff(cdf2)

    # Mask to remove 0 probarbilities and prevent log(0) or dividing by 0.
    mask = p > 0
    p = p[mask]
    q = np.clip(q[mask], epsilon, 1.0)

    # normalize
    p /= np.sum(p)
    q /= np.sum(q)

    return float(np.sum(p * np.log(p / q)))
