from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Histogram:
    """Object representing data as a histogram with breaks and counts."""
    breaks: np.ndarray
    counts: np.ndarray

    def __post_init__(self):
        """Logic to run after the generated __init__."""
        if len(self.breaks) != len(self.counts) + 1:
            raise ValueError("Breaks must have length len(counts) + 1")

    @property
    def p(self):
        """Percentage of data in each bin of Histogram object."""
        return self.counts / np.sum(self.counts)

    @property
    def mean(self):
        """Average value of Histogram object."""
        midpoints = (self.breaks[:-1] + self.breaks[1:]) / 2
        return np.sum(midpoints * self.p)

    @property
    def std(self):
        """Standard deviation of Histogram object."""
        midpoints = (self.breaks[:-1] + self.breaks[1:]) / 2
        widths = np.diff(self.breaks)
        m2 = np.sum(self.p * (midpoints**2 + (widths**2 / 12)))
        return np.sqrt(m2 - self.mean**2)

    @classmethod
    def from_raw_data(cls, data, bins=10):
        """Creates a Histogram object directly from a list of numbers."""
        counts, breaks = np.histogram(data, bins=bins)
        return cls(breaks=breaks, counts=counts)

    def plot(self, **kwargs):
        """Plots the histogram using matplotlib."""
        from .plot import plot_hist
        plot_hist(self, **kwargs)

    def __repr__(self):
        """Technical representation for debugging."""
        return f"Histogram(bins={len(self.counts)}, mean={self.mean:.2f})"

    def __str__(self):
        """Table style output for Histogram object."""
        header = f"{'':>10} {'X':^20} {'p':^10}\n"
        rows = []
        for i in range(len(self.counts)):
            bracket = "]" if i == len(self.counts) - 1 else ")"
            bin_label = f"[ {self.breaks[i]:g} ; {self.breaks[i+1]:g} {bracket}"
            rows.append(f"Bin_{i+1:<5} {bin_label:<20} {self.p[i]:^10.4g}")
        footer = f"\nmean = {self.mean:<18.10f} std = {self.std:<18.10f}"
        return header + "\n".join(rows) + footer