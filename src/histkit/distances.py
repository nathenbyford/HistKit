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
    W_p = sqrt( integral_0^1 |F_1^-1(t) - F_2^-1(t)|^p dt )
    
    Args:
        h1 (Histogram): First histogram.
        h2 (Histogram): Second histogram.
        p: power of the integrand.
        
    Returns:
        float: The squared Wasserstein-2 distance.
    """
    # Get all unique CDF values
    cdf1 = np.concatenate([[0], h1.cdf])
    cdf2 = np.concatenate([[0], h2.cdf])

    # create grid of histogram breaks
    all_cdf_breaks = np.unique(np.concatenate([cdf1, cdf2]))
    
    # Compute quantile functions (inverse CDFs)
    quantile1 = np.interp(all_cdf_breaks, cdf1, h1.breaks)
    quantile2 = np.interp(all_cdf_breaks, cdf2, h2.breaks)
    
    # Compute exact iintegral over each peice
    dist = 0.0
    for i in range(len(all_cdf_breaks) - 1):
        delta_u = all_cdf_breaks[i+1] - all_cdf_breaks[i]
        diff = (quantile1[i] - quantile2[i]) ** p
        dist += diff * delta_u
    
    return np.sqrt(dist)
