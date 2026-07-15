from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class Histogram2D:
    """Object representing a 2D spatial histogram with breaks and counts/rates."""
    x_breaks: np.ndarray  # Shape: (nx + 1,)
    y_breaks: np.ndarray  # Shape: (ny + 1,)
    counts: np.ndarray    # Shape: (ny, nx)
    metric: str = "rate"  # Metric type: "rate" or "count"

    def __post_init__(self):
        """Validate input arrays."""
        ny, nx = self.counts.shape
        if len(self.y_breaks) != ny + 1:
            raise ValueError(f"y_breaks (len={len(self.y_breaks)}) must have length len(counts.shape[0]) + 1 (expected {ny + 1})")
        if len(self.x_breaks) != nx + 1:
            raise ValueError(f"x_breaks (len={len(self.x_breaks)}) must have length len(counts.shape[1]) + 1 (expected {nx + 1})")

    @property
    def p(self) -> np.ndarray:
        """Normalized probability distribution of defects over the bins."""
        total = np.sum(self.counts)
        if total == 0:
            return np.zeros_like(self.counts)
        return self.counts / total

    @property
    def mean_x(self) -> float:
        """Expected X coordinate of defects (center of mass)."""
        midpoints_x = (self.x_breaks[:-1] + self.x_breaks[1:]) / 2
        p_x = np.sum(self.p, axis=0)  # Marginal probability over X
        return float(np.sum(midpoints_x * p_x))

    @property
    def mean_y(self) -> float:
        """Expected Y coordinate of defects (center of mass)."""
        midpoints_y = (self.y_breaks[:-1] + self.y_breaks[1:]) / 2
        p_y = np.sum(self.p, axis=1)  # Marginal probability over Y
        return float(np.sum(midpoints_y * p_y))

    @classmethod
    def from_coordinates(
        cls, 
        xs, 
        ys, 
        x_range, 
        y_range, 
        bins=(8, 8), 
        metric: str = "count"
    ):
        """
        Creates a Histogram2D directly from list of continuous x and y coordinates.

        Args:
            xs (array-like): X coordinates.
            ys (array-like): Y coordinates.
            x_range (tuple): (xmin, xmax) range of X coordinate space.
            y_range (tuple): (ymin, ymax) range of Y coordinate space.
            bins (tuple): (ny, nx) number of spatial bins.
            metric (str): 'count' or 'rate' (default: 'count').
        """
        ny, nx = bins
        x_breaks = np.linspace(x_range[0], x_range[1], nx + 1)
        y_breaks = np.linspace(y_range[0], y_range[1], ny + 1)
        
        # np.histogram2d bins xs as first dimension and ys as second dimension, returning (nx, ny)
        # We pass bins as [x_breaks, y_breaks] and transpose the result to shape (ny, nx)
        counts, _, _ = np.histogram2d(xs, ys, bins=[x_breaks, y_breaks])
        return cls(x_breaks=x_breaks, y_breaks=y_breaks, counts=counts.T, metric=metric)

    def as_dict(self):
        return asdict(self)

    def plot(self, ax=None, cmap="YlOrRd", show=False, **kwargs):
        """
        Plots the 2D spatial histogram as a heatmap using matplotlib.
        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # meshgrid for bin boundaries
        X, Y = np.meshgrid(self.x_breaks, self.y_breaks)
        
        # pcolormesh is ideal for drawing rectangular grid bins with physical edges
        mesh = ax.pcolormesh(X, Y, self.counts, cmap=cmap, edgecolors="#ffffff", linewidths=0.5, **kwargs)
        
        # Invert Y axis to match matrix view (row 0 at top)
        ax.invert_yaxis()
        
        # Labels and formatting
        is_normalized = self.x_breaks[-1] == 1.0
        ax.set_xlabel("X Coordinate (Normalized)" if is_normalized else "X Coordinate (Dies)")
        ax.set_ylabel("Y Coordinate (Normalized)" if is_normalized else "Y Coordinate (Dies)")
        
        label = "Defect Rate (%)" if self.metric == "rate" else "Defect Count"
        cbar = plt.colorbar(mesh, ax=ax)
        if self.metric == "rate":
            cbar.set_label("Defect Rate")
            # Format colorbar to show percentages if appropriate
            cbar.ax.yaxis.set_major_formatter(lambda x, pos: f"{x*100:.1f}%")
        else:
            cbar.set_label(label)

        # Plot center of mass
        if np.sum(self.counts) > 0:
            ax.plot(self.mean_x, self.mean_y, "bx", markersize=10, markeredgewidth=2, label="Defect Centroid")
            ax.legend(loc="upper right")

        if show:
            plt.show()
        return ax

    def __repr__(self):
        """Technical representation for debugging."""
        ny, nx = self.counts.shape
        return f"Histogram2D(bins=({ny}x{nx}), metric='{self.metric}', mean_x={self.mean_x:.2f}, mean_y={self.mean_y:.2f})"

    def __str__(self):
        """Grid-like representation of the 2D spatial histogram."""
        ny, nx = self.counts.shape
        
        # Title & Metadata
        output = [
            f"2D Spatial Histogram ({ny}x{nx} bins, metric='{self.metric}')",
            f"X range: [{self.x_breaks[0]:g}, {self.x_breaks[-1]:g}]",
            f"Y range: [{self.y_breaks[0]:g}, {self.y_breaks[-1]:g}]",
            ""
        ]

        # Column Header (X bin labels)
        col_header = " " * 15
        for j in range(nx):
            col_header += f"Col_{j+1:<5}"
        output.append(col_header)
        
        # Separator line
        output.append(" " * 12 + "-" * (nx * 9))

        # Grid rows (Y bins)
        for i in range(ny):
            row_label = f"Row_{i+1:<5}"
            row_str = f"{row_label:<10} | "
            for j in range(nx):
                val = self.counts[i, j]
                if self.metric == "rate":
                    # Display as percentage
                    row_str += f"{val*100:6.2f}%  "
                else:
                    row_str += f"{val:7.1f}  "
            output.append(row_str)

        # Summary statistics
        output.append(" " * 12 + "-" * (nx * 9))
        output.append(f"Defect Centroid (X, Y) = ({self.mean_x:.2f}, {self.mean_y:.2f})")
        return "\n".join(output)
