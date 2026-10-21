"""Edge detection and line fitting for slanted-edge targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EdgeFit:
    """Straight-line fit of a near-vertical edge: x = slope * y + intercept."""

    slope: float
    intercept: float
    centroids: np.ndarray  # per-row edge positions used for the fit

    @property
    def angle_deg(self) -> float:
        """Edge tilt from vertical, in degrees."""
        return float(np.degrees(np.arctan(self.slope)))


def _orient_vertical(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return the image transposed if the edge looks horizontal.

    The edge is treated as vertical when the horizontal gradient energy is
    larger than the vertical gradient energy.
    """
    gx = np.abs(np.diff(image, axis=1)).sum()
    gy = np.abs(np.diff(image, axis=0)).sum()
    if gy > gx:
        return image.T, True
    return image, False


def find_edge(image: np.ndarray, window: int = 8) -> EdgeFit:
    """Locate a near-vertical edge by row-wise derivative centroids.

    Each row is differentiated, and the centroid of the absolute derivative
    gives a sub-pixel edge location. A least-squares line through those
    centroids gives the edge geometry.

    Parameters
    ----------
    image:
        2-D grayscale image containing one dominant near-vertical edge.
    window:
        Half-width, in pixels, of the region around each row's derivative
        peak used for the centroid. Restricting the centroid to this window
        keeps noise in flat regions from biasing the edge position.

    Returns
    -------
    EdgeFit
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError("find_edge expects a 2-D grayscale image")

    deriv = np.abs(np.diff(img, axis=1))

    # Zero out everything outside a window around each row's peak.
    peaks = deriv.argmax(axis=1)
    col_idx = np.arange(deriv.shape[1])
    mask = np.abs(col_idx[None, :] - peaks[:, None]) <= window
    deriv = deriv * mask

    weights = deriv.sum(axis=1)
    if np.any(weights == 0):
        raise ValueError("Found rows with no edge contrast")

    # Derivative sample k sits halfway between pixels k and k + 1.
    cols = np.arange(deriv.shape[1]) + 0.5
    centroids = (deriv * cols).sum(axis=1) / weights
    rows = np.arange(img.shape[0])

    slope, intercept = np.polyfit(rows, centroids, 1)
    return EdgeFit(slope=float(slope), intercept=float(intercept), centroids=centroids)
