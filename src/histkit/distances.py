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
    
    # Compute CDF values at all breaks
    cdf1 = np.interp(all_breaks, h1.breaks[1:], h1.cdf, left=0.0, right=1.0)
    cdf2 = np.interp(all_breaks, h2.breaks[1:], h2.cdf, left=0.0, right=1.0)
    
    # Differences
    widths = np.diff(all_breaks)

    # Calculate value with special case for p=2 using np.sqrt
    integral = np.dot(np.abs(cdf1[:-1] - cdf2[:-1]) ** p, widths)
    return float(np.sqrt(integral) if p == 2 else integral ** (1 / p))
