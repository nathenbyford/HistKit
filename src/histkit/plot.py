import matplotlib.pyplot as plt
from .hist import Histogram

def plot_hist(hist: Histogram, **kwargs):
    """
    Plots a Histogram object using matplotlib.

    Args:
        hist: The Histogram object to plot.
        **kwargs: Additional keyword arguments to pass to matplotlib.pyplot.hist.
    """
    plt.hist(hist.breaks[:-1], bins=hist.breaks, weights=hist.counts, **kwargs)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram")
    plt.show()