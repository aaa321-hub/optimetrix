"""Synthetic test targets with known, analytic MTF."""

from __future__ import annotations

from math import erf, sqrt

import numpy as np

_erf = np.vectorize(erf)


def make_slanted_edge(
    size: tuple[int, int] = (100, 100),
    angle_deg: float = 5.0,
    blur_sigma: float = 1.0,
    low: float = 0.2,
    high: float = 0.8,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Make a Gaussian-blurred slanted edge.

    A Gaussian blur of standard deviation `blur_sigma` pixels has the
    analytic MTF exp(-2 * pi^2 * sigma^2 * f^2), which makes this target
    useful for checking the analysis pipeline.
    """
    h, w = size
    rows, cols = np.indices((h, w), dtype=float)
    theta = np.radians(angle_deg)
    x0 = w / 2.0
    y0 = h / 2.0
    # Signed distance to a line through the image centre tilted by theta.
    dist = (cols - x0) * np.cos(theta) - (rows - y0) * np.sin(theta)

    profile = 0.5 * (1.0 + _erf(dist / (blur_sigma * sqrt(2.0))))
    image = low + (high - low) * profile

    if noise_std > 0:
        rng = np.random.default_rng(seed)
        image = image + rng.normal(0.0, noise_std, image.shape)
    return image


def gaussian_mtf(freqs: np.ndarray, sigma: float) -> np.ndarray:
    """Analytic MTF of a Gaussian blur with standard deviation `sigma`."""
    return np.exp(-2.0 * np.pi**2 * sigma**2 * np.asarray(freqs) ** 2)
