"""Slanted-edge MTF computation.

The method follows the general approach described publicly in ISO 12233:
1. Fit the edge location in every row.
2. Project each pixel onto the edge normal to build an oversampled
   edge spread function (ESF).
3. Differentiate the ESF to get the line spread function (LSF).
4. Window the LSF and take the magnitude of its Fourier transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimetrix.edge import _orient_vertical, find_edge


@dataclass
class MTFResult:
    frequencies: np.ndarray  # cycles per pixel
    mtf: np.ndarray  # normalized so that mtf[0] == 1
    esf: np.ndarray
    lsf: np.ndarray
    edge_angle_deg: float
    oversampling: int

    def mtf_at(self, fraction: float) -> float:
        """Frequency (cycles/pixel) where MTF first drops to `fraction`."""
        below = np.nonzero(self.mtf < fraction)[0]
        if below.size == 0:
            return float("nan")
        i = below[0]
        if i == 0:
            return 0.0
        f0, f1 = self.frequencies[i - 1], self.frequencies[i]
        m0, m1 = self.mtf[i - 1], self.mtf[i]
        return float(f0 + (fraction - m0) * (f1 - f0) / (m1 - m0))

    @property
    def mtf50(self) -> float:
        return self.mtf_at(0.5)

    @property
    def mtf10(self) -> float:
        return self.mtf_at(0.1)


def _build_esf(img: np.ndarray, slope: float, intercept: float, oversampling: int) -> np.ndarray:
    rows, cols = np.indices(img.shape)
    # Signed horizontal distance from each pixel to the fitted edge, scaled
    # onto the edge normal.
    dist = (cols - (slope * rows + intercept)) * np.cos(np.arctan(slope))

    bins = np.floor(dist * oversampling).astype(int)
    bins -= bins.min()
    counts = np.bincount(bins.ravel())
    sums = np.bincount(bins.ravel(), weights=img.ravel())

    # Fill empty bins by interpolating neighbours so the ESF stays smooth.
    filled = counts > 0
    esf = np.empty(counts.size)
    esf[filled] = sums[filled] / counts[filled]
    if not filled.all():
        idx = np.arange(counts.size)
        esf[~filled] = np.interp(idx[~filled], idx[filled], esf[filled])
    return esf


def slanted_edge_mtf(image: np.ndarray, oversampling: int = 4) -> MTFResult:
    """Compute the MTF of an imaging system from a slanted-edge image.

    Parameters
    ----------
    image:
        2-D grayscale image of a slanted edge (vertical or horizontal).
    oversampling:
        Number of ESF bins per pixel. 4 is the conventional choice.
    """
    img = np.asarray(image, dtype=float)
    img, _ = _orient_vertical(img)

    fit = find_edge(img)
    esf = _build_esf(img, fit.slope, fit.intercept, oversampling)

    lsf = np.gradient(esf)
    lsf *= np.hamming(lsf.size)

    spectrum = np.abs(np.fft.rfft(lsf))
    mtf = spectrum / spectrum[0]
    freqs = np.fft.rfftfreq(lsf.size, d=1.0 / oversampling)

    # Report up to the sensor Nyquist frequency (0.5 cycles/pixel) and a
    # little beyond, which is where the data is meaningful.
    keep = freqs <= 1.0
    return MTFResult(
        frequencies=freqs[keep],
        mtf=mtf[keep],
        esf=esf,
        lsf=lsf,
        edge_angle_deg=fit.angle_deg,
        oversampling=oversampling,
    )
