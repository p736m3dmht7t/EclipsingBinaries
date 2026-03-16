# -*- coding: utf-8 -*-
"""
Tests for multi_aperture_photometry.py

"""

import pytest
import numpy as np


# ===========================================================================
# Pure math helpers — extracted from multiple_AP
# These formulas are used directly in the photometry pipeline and are the
# most valuable things to unit-test since they affect all output magnitudes.
# ===========================================================================

# --- magnitude calculation --------------------------------------------------
# target_magnitude = (-log(sum(2.512**-mags)) / log(2.512))
#                    - 2.5 * log10(target_flx / sum(comparison_flx))

def _target_magnitude(magnitudes_comp, target_flx, comparison_flx):
    ensemble_flux = sum(2.512 ** -m for m in magnitudes_comp)
    return (-np.log(ensemble_flux) / np.log(2.512)) - (
        2.5 * np.log10(target_flx / sum(comparison_flx))
    )


def _target_magnitude_error(target_flux_err, target_flx, comp_flux_err, comparison_flx):
    return 2.5 * np.log10(
        1 + np.sqrt(
            (target_flux_err ** 2 / target_flx ** 2) +
            (sum(comp_flux_err ** 2) / sum(comparison_flx) ** 2)
        )
    )


# --- background subtraction -------------------------------------------------
def _background_subtracted_flux(aperture_sum, bkg_mean, aperture_area):
    return aperture_sum - bkg_mean * aperture_area


# --- flux error -------------------------------------------------------------
def _flux_error(aperture_sum, aperture_area, read_noise=10.83):
    return np.sqrt(aperture_sum + aperture_area * read_noise ** 2)


# --- relative flux ----------------------------------------------------------
def _rel_flux_comps(comparison_flx):
    result = []
    for i, flx in enumerate(comparison_flx):
        total_others = sum(f for j, f in enumerate(comparison_flx) if j != i)
        result.append(flx / total_others)
    return result


# ===========================================================================
# target magnitude
# ===========================================================================
def test_target_magnitude_equal_flux_gives_catalog():
    # If target flux == sum(comparison flux), differential term = 0
    # So result equals the ensemble magnitude of the comparison stars
    mags = [12.0, 12.0]
    target_flx = 2.0
    comp_flx = [1.0, 1.0]
    result = _target_magnitude(mags, target_flx, comp_flx)
    expected_ensemble = -np.log(sum(2.512 ** -m for m in mags)) / np.log(2.512)
    assert result == pytest.approx(expected_ensemble, abs=1e-8)


def test_target_magnitude_brighter_target_gives_lower_mag():
    mags = [14.0, 14.0]
    comp_flx = [1000.0, 1000.0]
    mag_faint = _target_magnitude(mags, 500.0, comp_flx)
    mag_bright = _target_magnitude(mags, 5000.0, comp_flx)
    assert mag_bright < mag_faint


def test_target_magnitude_returns_float():
    result = _target_magnitude([14.0, 14.5], 1000.0, [900.0, 950.0])
    assert isinstance(result, float)


def test_target_magnitude_single_comp_star():
    mags = [13.0]
    target_flx = 500.0
    comp_flx = [500.0]
    result = _target_magnitude(mags, target_flx, comp_flx)
    expected = -np.log(2.512 ** -13.0) / np.log(2.512)
    assert result == pytest.approx(expected, abs=1e-8)


def test_target_magnitude_scales_with_flux_ratio():
    mags = [14.0, 14.0]
    comp_flx = [1000.0, 1000.0]
    mag1 = _target_magnitude(mags, 1000.0, comp_flx)
    mag2 = _target_magnitude(mags, 4000.0, comp_flx)
    # quadrupling flux should decrease magnitude by ~1.5
    assert mag1 - mag2 == pytest.approx(2.5 * np.log10(4), abs=1e-6)


def test_target_magnitude_multiple_comps():
    mags = [13.0, 13.5, 14.0]
    comp_flx = [800.0, 600.0, 400.0]
    result = _target_magnitude(mags, 700.0, comp_flx)
    assert np.isfinite(result)


# ===========================================================================
# target magnitude error
# ===========================================================================
def test_magnitude_error_zero_flux_error_gives_minimum():
    # With zero flux errors, error = 2.5 * log10(1 + 0) = 0
    result = _target_magnitude_error(0.0, 1000.0, np.array([0.0, 0.0]), [500.0, 500.0])
    assert result == pytest.approx(0.0, abs=1e-10)


def test_magnitude_error_positive():
    result = _target_magnitude_error(50.0, 1000.0, np.array([30.0, 30.0]), [900.0, 850.0])
    assert result > 0


def test_magnitude_error_larger_noise_gives_larger_error():
    err_low = _target_magnitude_error(10.0, 1000.0, np.array([10.0, 10.0]), [900.0, 850.0])
    err_high = _target_magnitude_error(100.0, 1000.0, np.array([100.0, 100.0]), [900.0, 850.0])
    assert err_high > err_low


def test_magnitude_error_returns_float():
    result = _target_magnitude_error(50.0, 1000.0, np.array([30.0, 30.0]), [900.0, 850.0])
    assert isinstance(result, float)


def test_magnitude_error_finite():
    result = _target_magnitude_error(25.0, 2000.0, np.array([20.0, 15.0, 18.0]), [800.0, 700.0, 600.0])
    assert np.isfinite(result)


# ===========================================================================
# background subtracted flux
# ===========================================================================
def test_background_flux_no_background():
    assert _background_subtracted_flux(1000.0, 0.0, 100.0) == pytest.approx(1000.0)


def test_background_flux_subtracts_correctly():
    # aperture_sum=1000, bkg_mean=2.0, area=50 → 1000 - 100 = 900
    assert _background_subtracted_flux(1000.0, 2.0, 50.0) == pytest.approx(900.0)


def test_background_flux_large_background_can_go_negative():
    result = _background_subtracted_flux(100.0, 10.0, 50.0)
    assert result == pytest.approx(-400.0)


def test_background_flux_scales_with_area():
    area1 = _background_subtracted_flux(1000.0, 1.0, 100.0)
    area2 = _background_subtracted_flux(1000.0, 1.0, 200.0)
    assert area1 > area2


def test_background_flux_zero_aperture_sum():
    assert _background_subtracted_flux(0.0, 5.0, 20.0) == pytest.approx(-100.0)


# ===========================================================================
# flux error
# ===========================================================================
def test_flux_error_positive():
    assert _flux_error(1000.0, np.pi * 20 ** 2) > 0


def test_flux_error_increases_with_counts():
    err_low = _flux_error(100.0, 100.0)
    err_high = _flux_error(10000.0, 100.0)
    assert err_high > err_low


def test_flux_error_increases_with_area():
    err_small = _flux_error(1000.0, 100.0)
    err_large = _flux_error(1000.0, 1000.0)
    assert err_large > err_small


def test_flux_error_zero_counts_still_positive():
    # read noise alone contributes even with zero signal
    result = _flux_error(0.0, 100.0, read_noise=10.83)
    assert result == pytest.approx(np.sqrt(100.0 * 10.83 ** 2))


def test_flux_error_known_value():
    # aperture_sum=100, area=4, read_noise=10 → sqrt(100 + 4*100) = sqrt(500)
    assert _flux_error(100.0, 4.0, read_noise=10.0) == pytest.approx(np.sqrt(500.0), abs=1e-8)


def test_flux_error_default_read_noise():
    result = _flux_error(0.0, 1.0)
    assert result == pytest.approx(10.83, abs=1e-6)


# ===========================================================================
# relative flux of comparison stars
# ===========================================================================
def test_rel_flux_comps_two_equal_stars():
    # Each star gets 1/1 = 1.0 relative to the other
    result = _rel_flux_comps([500.0, 500.0])
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(1.0)


def test_rel_flux_comps_length_matches_input():
    comps = [100.0, 200.0, 300.0]
    assert len(_rel_flux_comps(comps)) == len(comps)


def test_rel_flux_comps_each_star_excluded_from_own_denominator():
    comps = [100.0, 200.0, 300.0]
    result = _rel_flux_comps(comps)
    # star 0: 100 / (200+300) = 0.2
    assert result[0] == pytest.approx(100.0 / 500.0, abs=1e-10)
    # star 1: 200 / (100+300) = 0.5
    assert result[1] == pytest.approx(200.0 / 400.0, abs=1e-10)
    # star 2: 300 / (100+200) = 1.0
    assert result[2] == pytest.approx(300.0 / 300.0, abs=1e-10)


def test_rel_flux_comps_single_star():
    # Only one comparison star — denominator is 0, will raise ZeroDivisionError
    with pytest.raises(ZeroDivisionError):
        _rel_flux_comps([500.0])


def test_rel_flux_comps_all_positive():
    comps = [300.0, 400.0, 500.0, 600.0]
    assert all(r > 0 for r in _rel_flux_comps(comps))


def test_rel_flux_comps_dominant_star_gives_high_relative_flux():
    comps = [10.0, 10.0, 1000.0]
    result = _rel_flux_comps(comps)
    # star 2 is dominant, should have highest relative flux
    assert result[2] == max(result)


# ===========================================================================
# ensemble magnitude formula internals
# ===========================================================================
def test_ensemble_flux_conversion_roundtrip():
    mags = [13.0, 14.0, 15.0]
    ensemble = sum(2.512 ** -m for m in mags)
    recovered = -np.log(ensemble) / np.log(2.512)
    assert np.isfinite(recovered)


def test_ensemble_dominated_by_brightest():
    mags_bright = [10.0, 14.0, 15.0]
    mags_faint = [14.0, 14.0, 15.0]
    ens_bright = sum(2.512 ** -m for m in mags_bright)
    ens_faint = sum(2.512 ** -m for m in mags_faint)
    # brighter ensemble has higher flux → lower (brighter) recovered magnitude
    assert (-np.log(ens_bright) / np.log(2.512)) < (-np.log(ens_faint) / np.log(2.512))


def test_differential_term_zero_when_equal_flux():
    target_flx = 1000.0
    comp_flx = [1000.0]
    diff = 2.5 * np.log10(target_flx / sum(comp_flx))
    assert diff == pytest.approx(0.0, abs=1e-10)


def test_differential_term_positive_when_target_brighter():
    # target brighter than ensemble → log10 > 0 → term > 0 → subtracted → lower magnitude
    diff = 2.5 * np.log10(2000.0 / 1000.0)
    assert diff > 0


# ===========================================================================
# aperture area consistency (used in background and error calculations)
# ===========================================================================
def test_circular_aperture_area():
    from photutils.aperture import CircularAperture
    ap = CircularAperture((100, 100), r=20)
    assert ap.area == pytest.approx(np.pi * 20 ** 2, abs=1e-6)


def test_circular_annulus_area():
    from photutils.aperture import CircularAnnulus
    ann = CircularAnnulus((100, 100), r_in=30, r_out=50)
    expected = np.pi * (50 ** 2 - 30 ** 2)
    assert ann.area == pytest.approx(expected, abs=1e-6)


def test_annulus_area_larger_than_aperture():
    from photutils.aperture import CircularAperture, CircularAnnulus
    ap = CircularAperture((100, 100), r=20)
    ann = CircularAnnulus((100, 100), r_in=30, r_out=50)
    assert ann.area > ap.area