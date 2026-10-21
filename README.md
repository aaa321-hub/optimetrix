# OptiMetrix

Open-source Python tools for optical metrology analysis.

OptiMetrix computes the Modulation Transfer Function (MTF) of an imaging system from a slanted-edge image, using the general method described publicly in ISO 12233. It is written from scratch and aimed at lab engineers, students, and hobbyists who want a transparent, testable alternative to closed analysis software.

## Features

- Slanted-edge MTF with sub-pixel edge fitting and 4x oversampled ESF
- Handles vertical and horizontal edges automatically
- Noise-robust edge detection (windowed centroid per row)
- MTF50 and MTF10 summary metrics
- Synthetic Gaussian-blurred edge targets with analytic MTF for validation

## Install

```bash
git clone https://github.com/aaa321-hub/optimetrix.git
cd optimetrix
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np
from PIL import Image
from optimetrix import slanted_edge_mtf

img = np.asarray(Image.open("edge.png").convert("L"), dtype=float)
result = slanted_edge_mtf(img)

print(f"Edge angle: {result.edge_angle_deg:.2f} deg")
print(f"MTF50: {result.mtf50:.3f} cycles/pixel")
```

Try it without a real image:

```python
from optimetrix import make_slanted_edge, slanted_edge_mtf

result = slanted_edge_mtf(make_slanted_edge(blur_sigma=1.0))
print(result.mtf50)  # about 0.187 cycles/pixel, matching Gaussian theory
```

## How it works

1. **Edge fit.** Each row is differentiated and the centroid of the derivative near its peak gives a sub-pixel edge position. A line is fit through those positions.
2. **ESF.** Every pixel is projected onto the edge normal and binned at 4x oversampling to build the edge spread function.
3. **LSF.** The ESF is differentiated and Hamming-windowed.
4. **MTF.** The normalized magnitude of the LSF's Fourier transform.

## Tests

```bash
pytest
```

The suite checks recovered edge angle, MTF50 against analytic Gaussian values, curve agreement below Nyquist, horizontal edges, and noise robustness.

## Roadmap

- [ ] Refactor ESF/LSF pipeline into a vectorized, separately tested module
- [ ] CLI to batch-process image folders and export CSV/PDF reports
- [ ] ISO-style derivative correction for the finite-difference LSF
- [ ] ROI selection for multi-edge targets
- [ ] Distortion and lateral chromatic aberration metrics

## License

MIT. See [LICENSE](LICENSE).
