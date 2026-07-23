import numpy as np
import ot
from .hist import Histogram
from .hist2d import Histogram2D

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
        float: The Wasserstein-p distance.
    """
    # Get all unique CDF values, starting at 0.0
    xp1 = np.concatenate([[0.0], h1.cdf])
    xp2 = np.concatenate([[0.0], h2.cdf])
    all_cdf = np.union1d(xp1, xp2)
    
    # Compute quantile values at all CDF values
    q1 = np.interp(all_cdf, xp1, h1.breaks)
    q2 = np.interp(all_cdf, xp2, h2.breaks)
    
    # Differences in CDF space
    dt = np.diff(all_cdf)

    # Calculate integral using midpoint approximation
    mid_diff = (q1[:-1] + q1[1:]) / 2 - (q2[:-1] + q2[1:]) / 2
    integral = np.sum(np.abs(mid_diff) ** p * dt)
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

def energy_distance(h1: Histogram, h2: Histogram) -> float:
    """
    Computes the 1D Energy Distance between two histrograms.

    In the 1D case the energy distance results in 2 times the Cramer distance.

    Args:
        h1 (Histogram): First 1D Histogram
        h2 (histogram): Second 1D Histogram

    Returns:
        float: The Energy Distance
    """
    # Get all unique CDF breaks
    all_breaks = np.union1d(h1.breaks, h2.breaks)

    # Compute all CDF values
    cdf1 = np.interp(all_breaks, h1.breaks[1:], h1.cdf, left=0.0, right=1.0)
    cdf2 = np.interp(all_breaks, h2.breaks[1:], h2.cdf, left=0.0, right=1.0)

    # Comute integral as sum of squared differences
    val = np.sum(cdf1 - cdf2)**2

    return float(2 * val)


def sliced_wasserstein_distance(
    h2d_1: Histogram2D, 
    h2d_2: Histogram2D, 
    n_projections: int = 50, 
    n_bins_1d: int = 30,
    p: int = 2
) -> float:
    """
    Computes the Sliced Wasserstein-p distance between two 2D spatial histograms.
    
    Project both 2D spatial distributions onto 1D lines along uniformly spaced projection
    angles and compute the average 1D Wasserstein-p distance between them.
    
    Args:
        h2d_1 (Histogram2D): First 2D spatial histogram.
        h2d_2 (Histogram2D): Second 2D spatial histogram.
        n_projections (int): Number of projection directions (default: 50).
        n_bins_1d (int): Number of bins to use for the projected 1D histograms (default: 30).
        p (int): The power parameter of the Wasserstein distance (default: 2).
        
    Returns:
        float: The Sliced Wasserstein-p distance.
    """
    # Get midpoints of the 2D grid cells
    mid_x1 = (h2d_1.x_breaks[:-1] + h2d_1.x_breaks[1:]) / 2
    mid_y1 = (h2d_1.y_breaks[:-1] + h2d_1.y_breaks[1:]) / 2
    X, Y = np.meshgrid(mid_x1, mid_y1)
    coords = np.column_stack([X.flatten(), Y.flatten()])
    
    # Normalized counts (densities)
    p1 = h2d_1.p.flatten()
    p2 = h2d_2.p.flatten()
    
    # Generate uniform projection angles from 0 to pi
    angles = np.linspace(0, np.pi, n_projections, endpoint=False)
    vectors = np.column_stack([np.cos(angles), np.sin(angles)])
    
    sum_dist_p = 0.0
    
    for v in vectors:
        # Project 2D cell coordinates to 1D
        projected_coords = coords @ v
        
        z_min = projected_coords.min()
        z_max = projected_coords.max()
        
        # Define 1D breaks along the projected line
        breaks = np.linspace(z_min, z_max, n_bins_1d + 1)
        
        # Bin the projected densities using weights
        counts1, _ = np.histogram(projected_coords, bins=breaks, weights=p1)
        counts2, _ = np.histogram(projected_coords, bins=breaks, weights=p2)
        
        # Construct 1D Histogram objects
        proj_h1 = Histogram(breaks=breaks, counts=counts1)
        proj_h2 = Histogram(breaks=breaks, counts=counts2)
        
        # Compute 1D Wasserstein distance
        w_dist = wasserstein_distance(proj_h1, proj_h2, p=p)
        sum_dist_p += w_dist ** p
        
    return float(np.power(sum_dist_p / n_projections, 1/p))


def wasserstein2d(
    hist1: Histogram2D,
    hist2: Histogram2D,
    p: int = 2,
    return_plan: bool = False,
):
    """
    Exact p-Wasserstein distance between two 2D histograms.

    Parameters
    ----------
    hist1, hist2 : Histogram2D

    p : int
        Wasserstein order (1 or 2).

    return_plan : bool
        If True, also return the optimal transport plan.

    Returns
    -------
    distance : float

    plan : ndarray, optional
    """

    X = hist1.bin_centers
    Y = hist2.bin_centers

    a = hist1.histogram_mass
    b = hist2.histogram_mass

    # cost matrix
    M = ot.dist(X, Y, metric="euclidean") ** p

    plan = ot.emd(a, b, M)

    cost = np.sum(plan * M)

    distance = cost ** (1 / p)

    if return_plan:
        return distance, plan

    return distance


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

    # Mask to remove 0 probabilities and prevent log(0) or dividing by 0.
    mask = p > 0
    p = p[mask]
    q = np.clip(q[mask], epsilon, 1.0)

    # normalize
    p /= np.sum(p)
    q /= np.sum(q)

    return float(np.sum(p * np.log(p / q)))


def energy_distance_2d(h1: Histogram2D, h2: Histogram2D) -> float:
    """
    Computes the geometric Energy Distance between two 2D spatial histograms.

    Args:
        h1 (Histogram2D): First 2D histogram.
        h2 (Histogram2D): Second 2D histogram.

    Returns:
        float: The Energy Distance.
    """
    # Verify breaks match
    if not (np.array_equal(h1.x_breaks, h2.x_breaks) and np.array_equal(h1.y_breaks, h2.y_breaks)):
        raise ValueError("Histograms must have matching grids (x_breaks and y_breaks).")

    # Get cell centroids
    mid_x = (h1.x_breaks[:-1] + h1.x_breaks[1:]) / 2
    mid_y = (h1.y_breaks[:-1] + h1.y_breaks[1:]) / 2
    X, Y = np.meshgrid(mid_x, mid_y)
    coords = np.column_stack([X.flatten(), Y.flatten()])

    # Compute cell distance matrix
    D = cdist(coords, coords, metric="euclidean")

    # Extract probability densities
    p = h1.p.flatten()
    q = h2.p.flatten()

    # D_E^2 = 2 * E[|X - Y|] - E[|X - X'|] - E[|Y - Y'|]
    dist_sq = 2 * np.dot(p, np.dot(D, q)) - np.dot(p, np.dot(D, p)) - np.dot(q, np.dot(D, q))
    return float(np.sqrt(max(0.0, dist_sq)))

