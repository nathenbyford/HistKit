from dataclasses import dataclass
from collections.abc import Sequence
from typing import Union
import numpy as np
from .hist import Histogram

@dataclass(frozen=True, slots=True)
class HistogramMatrix:
	rows: list[list[Histogram]]

	@property
	def shape(self) -> tuple[int, int]:
		return (len(self.rows), len(self.rows[0]))

	@classmethod
	def from_rows(cls, rows: Sequence[Sequence[Histogram]]) -> "HistogramMatrix":
		return cls(rows)

	@classmethod
	def from_cols(cls, cols: Sequence[Sequence[Histogram]]) -> "HistogramMatrix":
		cols_list = [list(c) for c in cols]
		# Data checks
		n_rows = len(cols[0])
		# Check n_rows for empty rows and equal number of rows

		rows = [[cols_list[j][i] for j in range(len(cols_list))] for i in range(n_rows)]
		return cls(rows)

	@property		
	def T(self) -> "HistogramMatrix":
		return self.transpose()
		

def hmat(
	hists: Union[
		Sequence[Histogram],
		Sequence[Sequence[Histogram]]
	],
	*,
	axis: int = 0,
) -> HistogramMatrix:
	"""
	Constructor

	axis=0 (Default): treat a 1D list as rows -> shape (n, 1)
	axis=1: treat a 1D list as cols -> shape (1, n)

	If 'hists' is already 2D (nested sequences), interpreted as rows when axis=0
	and as cols when axis=1.
	"""
	if axis not in (0, 1):
		raise ValueError('axis must be 0 (rows) or 1 (cols).')

	hists_list = list(hists)
	# Add check for empty list

	first = hists_list[0]

	if isinstance(first, Histogram):
		one_d = list(hists_list)
		if axis == 0:
			return HistogramMatrix.from_rows([[h] for h in one_d]) # (n, 1)
		else:
			return HistogramMatrix.from_cols([one_d])

	two_d = [list(row) for row in hists_list]
	if axis == 0:
		return HistogramMatrix.from_rows(two_d)
	else:
		return HistogramMatrix.from_cols(two_d)

# Currently still in development 
def plot_hists(hists: HistogramMatrix):
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    
    dists = hists.rows[:]
    y_offsets = np.arange(len(dists))
    
    for y, g in zip(y_offsets, dists):
        
        hist = g[0].counts
        edges = g[0].breaks
        
        venters = 0.5 * (edges[:-1] + edges[1:])
        
        ax.hist(
            edges[:-1],
            bins=edges,
            weights=hist,
            width=np.diff(edges)
        )
    
    ax.set_yticks(y_offsets)
    ax.set_yticklabels(dists)
    ax.set_xlabel('Vals')
    ax.set_ylabel('group')
    plt.show()
    
    