import numpy as np
from .hist import Histogram
from .distances import wasserstein_distance

def permutation_test(h1: Histogram, h2: Histogram, n_permutations: int = 1000, 
	p: int = 2):
	"""
	Computes the permutation test for two samples of histogram matrices.
	"""
	# Recreate sample from histograms using midpoints and counts
	mids1 = (h1.breaks[:-1] + h1.breaks[1:]) / 2
	mids2 = (h2.breaks[:-1] + h2.breaks[1:]) / 2
	samp1 = np.repeat(mids1, np.round(h1.counts).astype(int))
	samp2 = np.repeat(mids2, np.round(h2.counts).astype(int))

	n1 = len(h1)
	n2 = len(h2)
	pooled = np.concatenate([samp1, samp2])

	# Calculate observed W2 distance
	obs_dist = wasserstein_distance(h1, h2, p = p)

	# Permutation loop
	count = 0
	for _ in range(n_permutations):
		np.random.shuffle(pooled)

		perm_s1 = pooled[:n1]
		perm_s2 = pooled[n1:]

		perm_h1 = Histogram(perm_s1, breaks=h1.breaks)
