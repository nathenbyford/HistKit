import numpy as np
from .hist import Histogram
from .matrix import HistogramMatrix

def wasserstein_1D_barycenter(
    histograms: HistogramMatrix,
    weights: np.ndarray | None = None,
    n_quantiles: int = 1000,
) -> Histogram:
    """
    Compute the Wasserstein barycenter from a histogram matrix.

    Uses the closed-form solution: the barycenter's quantile function
    is the (weighted) average of the individual quantile functions.

    Args:
        histograms (HistogramMatrix): First histogram.
        weights (arrray): Optional weights array of shape (len(histograms),). Defaults to uniform weighting.
        n_quantiles (int): Resolution of the quantile grid.
        
    Returns:
        histogram: The barycenter mean histogram.
    """
    if weights is None:
        weights = np.ones(histograms.shape[0]) / histograms.shape[0]
    else:
        weights = np.asarray(weights)
        weights = weights / weights.sum() # normalize just in case

    # Common probability grid: length of n_quantiles (avoid 0.0 and 1.0)
    prob_grid = np.linspace(0, 1, n_quantiles + 2)[1:-1]

    quantile_sum = np.zeros(n_quantiles)
    for h, w in zip(histograms, weights):
        quantile_sum += w * np.interp(prob_grid, [0.0, *h.cdf], h.breaks)

    counts = np.ones(n_quantiles - 1)
    return Histogram(breaks=quantile_sum, counts=counts)

    