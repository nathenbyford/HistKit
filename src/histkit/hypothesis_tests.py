import numpy as np
from .hist import Histogram
from .distances import wasserstein_distance

def permutation_test(h1: Histogram, h2: Histogram, n_permutations: int = 1000, 
	p: int = 2) -> float:
	"""
	Computes the permutation test for two samples of 1D histograms.
	
	Args:
		h1 (Histogram): First 1D histogram.
		h2 (Histogram): Second 1D histogram.
		n_permutations (int): Number of permutations to run (default: 1000).
		p (int): Power parameter for Wasserstein distance.
		
	Returns:
		float: The p-value of the test.
	"""
	# Recreate sample from histograms using midpoints and counts
	mids1 = (h1.breaks[:-1] + h1.breaks[1:]) / 2
	mids2 = (h2.breaks[:-1] + h2.breaks[1:]) / 2
	
	samp1 = np.repeat(mids1, np.round(h1.counts).astype(int))
	samp2 = np.repeat(mids2, np.round(h2.counts).astype(int))

	n1 = len(samp1)
	n2 = len(samp2)
	pooled = np.concatenate([samp1, samp2])

	# Calculate observed W distance
	obs_dist = wasserstein_distance(h1, h2, p = p)

	# Permutation loop
	count = 0
	for _ in range(n_permutations):
		np.random.shuffle(pooled)

		perm_s1 = pooled[:n1]
		perm_s2 = pooled[n1:]

		# Bin the permuted samples using the original breaks
		perm_counts1, _ = np.histogram(perm_s1, bins=h1.breaks)
		perm_counts2, _ = np.histogram(perm_s2, bins=h2.breaks)

		perm_h1 = Histogram(breaks=h1.breaks, counts=perm_counts1)
		perm_h2 = Histogram(breaks=h2.breaks, counts=perm_counts2)

		perm_dist = wasserstein_distance(perm_h1, perm_h2, p = p)
		if perm_dist >= obs_dist:
			count += 1

	return count / n_permutations


def permutation_test_2d(
	group1: list[Histogram2D], 
	group2: list[Histogram2D], 
	n_permutations: int = 500, 
	n_projections: int = 30,
	p: int = 2
) -> float:
	"""
	Performs a 2D spatial permutation test to determine if the spatial distributions 
	of defects differ significantly between two groups of wafers.
	
	Hypothesis:
	  Null (H0): Group 1 and Group 2 have the same spatial distribution of defects.
	  Alternative (H1): Group 1 and Group 2 have different spatial distributions of defects.
	  
	Test Statistic:
	  Sliced Wasserstein distance between the average (mean) 2D histogram of Group 1 
	  and the average (mean) 2D histogram of Group 2.
	
	Args:
		group1 (list[Histogram2D]): First sample of spatial 2D histograms.
		group2 (list[Histogram2D]): Second sample of spatial 2D histograms.
		n_permutations (int): Number of permutations to run (default: 500).
		n_projections (int): Number of projections for Sliced Wasserstein (default: 30).
		p (int): Power parameter for Sliced Wasserstein distance.
		
	Returns:
		float: The p-value of the test.
	"""
	n1 = len(group1)
	n2 = len(group2)
	if n1 == 0 or n2 == 0:
		raise ValueError("Groups must contain at least one Histogram2D object.")
		
	# Verify grid compatibility (breaks must match)
	g1_ref = group1[0]
	g2_ref = group2[0]
	if not (np.array_equal(g1_ref.x_breaks, g2_ref.x_breaks) and np.array_equal(g1_ref.y_breaks, g2_ref.y_breaks)):
		raise ValueError("All histograms must have matching x_breaks and y_breaks grids.")

	# Helper function to compute the mean/average Histogram2D from a list of Histogram2D
	def mean_histogram2d(h_list: list[Histogram2D]) -> Histogram2D:
		ref = h_list[0]
		avg_counts = np.mean([h.counts for h in h_list], axis=0)
		return Histogram2D(
			x_breaks=ref.x_breaks,
			y_breaks=ref.y_breaks,
			counts=avg_counts,
			metric=ref.metric
		)

	# Calculate observed test statistic
	mean_h1 = mean_histogram2d(group1)
	mean_h2 = mean_histogram2d(group2)
	obs_stat = wasserstein2d(mean_h1, mean_h2, p=p)

	# Pool groups
	combined = list(group1) + list(group2)
	
	# Permutation loop
	greater_count = 0
	for _ in range(n_permutations):
		# Shuffle combined list using indices
		indices = np.arange(len(combined))
		np.random.shuffle(indices)
		
		# Split into two random permuted groups
		perm_group1 = [combined[i] for i in indices[:n1]]
		perm_group2 = [combined[i] for i in indices[n1:]]
		
		# Compute mean histograms for permuted groups
		perm_mean_h1 = mean_histogram2d(perm_group1)
		perm_mean_h2 = mean_histogram2d(perm_group2)
		
		# Compute 2D Wasserstein between permuted means
		perm_stat = wasserstein2d(perm_mean_h1, perm_mean_h2, p=p)
		
		if perm_stat >= obs_stat:
			greater_count += 1
			
	p_value = greater_count / n_permutations
	return p_value


def permutation_test_energy(
	group1: list[Histogram2D], 
	group2: list[Histogram2D], 
	n_permutations: int = 500
) -> float:
	"""
	Performs a 2D spatial permutation test using the Energy Distance metric.
	
	Hypothesis:
	  Null (H0): Group 1 and Group 2 have the same spatial distribution of defects.
	  Alternative (H1): Group 1 and Group 2 have different spatial distributions of defects.
	  
	Test Statistic:
	  Energy Distance between the average (mean) 2D histogram of Group 1 
	  and the average (mean) 2D histogram of Group 2.
	
	Args:
		group1 (list[Histogram2D]): First sample of spatial 2D histograms.
		group2 (list[Histogram2D]): Second sample of spatial 2D histograms.
		n_permutations (int): Number of permutations to run (default: 500).
		
	Returns:
		float: The p-value of the test.
	"""
	n1 = len(group1)
	n2 = len(group2)
	if n1 == 0 or n2 == 0:
		raise ValueError("Groups must contain at least one Histogram2D object.")

	# Helper function to compute the mean/average Histogram2D from a list of Histogram2D
	def mean_histogram2d(h_list: list[Histogram2D]) -> Histogram2D:
		ref = h_list[0]
		avg_counts = np.mean([h.counts for h in h_list], axis=0)
		return Histogram2D(
			x_breaks=ref.x_breaks,
			y_breaks=ref.y_breaks,
			counts=avg_counts,
			metric=ref.metric
		)

	# Calculate observed test statistic
	mean_h1 = mean_histogram2d(group1)
	mean_h2 = mean_histogram2d(group2)
	obs_stat = energy_distance_2d(mean_h1, mean_h2)

	# Pool groups
	combined = list(group1) + list(group2)
	
	# Permutation loop
	greater_count = 0
	for _ in range(n_permutations):
		# Shuffle combined list using indices
		indices = np.arange(len(combined))
		np.random.shuffle(indices)
		
		# Split into two random permuted groups
		perm_group1 = [combined[i] for i in indices[:n1]]
		perm_group2 = [combined[i] for i in indices[n1:]]
		
		# Compute mean histograms for permuted groups
		perm_mean_h1 = mean_histogram2d(perm_group1)
		perm_mean_h2 = mean_histogram2d(perm_group2)
		
		# Compute Energy Distance between permuted means
		perm_stat = energy_distance_2d(perm_mean_h1, perm_mean_h2)
		
		if perm_stat >= obs_stat:
			greater_count += 1
			
	p_value = greater_count / n_permutations
	return p_value

