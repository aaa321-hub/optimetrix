import numpy as np
import pytest

from optimetrix import find_edge, make_slanted_edge, slanted_edge_mtf
from optimetrix.synthetic import gaussian_mtf


def test_edge_angle_recovered():
    img = make_slanted_edge(angle_deg=5.0, blur_sigma=1.0)
    fit = find_edge(img)
    assert fit.angle_deg == pytest.approx(5.0, abs=0.1)


def test_mtf_starts_at_one_and_decreases():
    result = slanted_edge_mtf(make_slanted_edge())
    assert result.mtf[0] == pytest.approx(1.0)
    assert result.mtf[-1] < result.mtf[0]


@pytest.mark.parametrize("sigma", [0.7, 1.0, 1.5])
def test_mtf50_matches_gaussian_theory(sigma):
    result = slanted_edge_mtf(make_slanted_edge(size=(128, 128), blur_sigma=sigma))
    expected = np.sqrt(np.log(2) / (2 * np.pi**2 * sigma**2))
    assert result.mtf50 == pytest.approx(expected, rel=0.05)


def test_mtf_curve_tracks_theory_below_nyquist():
    sigma = 1.0
    result = slanted_edge_mtf(make_slanted_edge(size=(128, 128), blur_sigma=sigma))
    band = result.frequencies <= 0.5
    theory = gaussian_mtf(result.frequencies[band], sigma)
    assert np.max(np.abs(result.mtf[band] - theory)) < 0.05


def test_horizontal_edge_handled():
    img = make_slanted_edge(angle_deg=5.0, blur_sigma=1.0).T
    vertical = slanted_edge_mtf(make_slanted_edge(angle_deg=5.0, blur_sigma=1.0))
    horizontal = slanted_edge_mtf(img)
    assert horizontal.mtf50 == pytest.approx(vertical.mtf50, rel=1e-6)


def test_robust_to_mild_noise():
    clean = slanted_edge_mtf(make_slanted_edge(size=(128, 128)))
    noisy = slanted_edge_mtf(make_slanted_edge(size=(128, 128), noise_std=0.005, seed=0))
    assert noisy.mtf50 == pytest.approx(clean.mtf50, rel=0.1)


def test_rejects_non_2d_input():
    with pytest.raises(ValueError):
        find_edge(np.zeros((4, 4, 3)))
