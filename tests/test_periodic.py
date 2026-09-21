"""Tests for the periodic-regime extraction (src/periodic.py).

These use synthetic signals with analytically known extrema. They protect the extraction logic
(cycle detection, spline-refined peaks, frequency estimators, periodicity flag); none of them
asserts anything about the DFG benchmark intervals.
"""
import numpy as np
import pytest

from src.periodic import (analyse_periodic, find_cycles, frequency_from_fft,
                          frequency_from_harmonic_fit, spline_extrema)

F0 = 3.0178          # Hz, close to the physical shedding frequency
T0 = 1.0 / F0


def synthetic(dt, t_end=8.0, transient=True, cl_amp=1.0, cd_mean=3.2, cd_amp=0.04):
    """c_L = -A sin(2 pi f t) pattern with a 2f drag, an exponentially decaying amplitude
    transient, and a pressure signal in phase with c_L."""
    t = np.arange(0.0, t_end + 0.5 * dt, dt)
    env = 1.0 - np.exp(-t / 1.0) if transient else np.ones_like(t)
    cl = cl_amp * env * np.sin(2 * np.pi * F0 * t)
    cd = cd_mean + cd_amp * env * np.cos(2 * np.pi * 2 * F0 * t + 0.3)
    dp = 2.4 + 0.02 * env * np.sin(2 * np.pi * F0 * t)
    return t, cd, cl, dp


def test_spline_extrema_finds_peak_between_samples():
    """A sine sampled so that no sample lies on the crest: the largest raw sample underestimates
    the crest, the spline extremum does not."""
    dt = 0.0123
    t = np.arange(0.0, 3.0, dt)
    y = np.sin(2 * np.pi * F0 * t)
    tk, yk = spline_extrema(t, y, "max")
    assert len(tk) >= 8
    assert np.allclose(yk, 1.0, atol=2e-6)
    assert y.max() < 1.0 - 1e-5                      # sampled maximum is genuinely biased here


def test_cycles_are_delimited_by_lift_minima_with_correct_period():
    t, cd, cl, dp = synthetic(0.005, transient=False)
    cycles = find_cycles(t, cd, cl, dp, t_min=0.5)
    assert len(cycles) >= 20
    for c in cycles:
        assert c.period == pytest.approx(T0, rel=1e-5)
        assert c.cl_max == pytest.approx(1.0, abs=1e-6)
        assert c.cl_min == pytest.approx(-1.0, abs=1e-6)
        assert c.cd_max == pytest.approx(3.24, abs=1e-6)


def test_drag_maximum_is_taken_over_the_whole_cycle_not_one_peak():
    """Two drag peaks of different height per lift cycle (as in the real flow): the cycle
    maximum must be the taller one."""
    dt = 0.004
    t = np.arange(0.0, 8.0, dt)
    cl = np.sin(2 * np.pi * F0 * t)
    cd = 3.2 + 0.03 * np.cos(2 * np.pi * 2 * F0 * t) + 0.01 * np.cos(2 * np.pi * F0 * t)
    cycles = find_cycles(t, cd, cl, None, t_min=0.5)
    # exact maximum of the analytic drag over one period (fine-grid reference)
    tt = np.linspace(0, T0, 400001)
    exact = (3.2 + 0.03 * np.cos(2 * np.pi * 2 * F0 * tt) + 0.01 * np.cos(2 * np.pi * F0 * tt)).max()
    assert all(c.cd_max == pytest.approx(exact, abs=1e-6) for c in cycles)


def test_analysis_recovers_known_values_despite_transient_and_coarse_sampling():
    t, cd, cl, dp = synthetic(0.01)
    pa = analyse_periodic(t, cd, cl, dp, D=0.1, U_bar=1.0, t_min=0.5)
    assert pa.values["cl_max"] == pytest.approx(1.0, abs=2e-3)   # decayed transient, not sampling
    assert pa.values["period"] == pytest.approx(T0, rel=1e-4)
    assert pa.values["cl_amp"] == pytest.approx(2.0, abs=4e-3)
    for name, st in pa.strouhal.items():
        assert st == pytest.approx(0.1 * F0, rel=5e-4), name


def test_periodic_flag_rejects_a_still_growing_signal():
    """Amplitude growing ~1 % per cycle: the cycle-to-cycle drift must fail the criterion."""
    dt = 0.005
    t = np.arange(0.0, 8.0, dt)
    growth = 1.0 + 0.03 * t
    cl = growth * np.sin(2 * np.pi * F0 * t)
    cd = 3.2 + 0.04 * growth * np.cos(2 * np.pi * 2 * F0 * t)
    pa = analyse_periodic(t, cd, cl, None, t_min=0.5)
    assert not pa.periodic


def test_periodic_flag_accepts_a_settled_signal():
    t, cd, cl, dp = synthetic(0.005, t_end=14.0)
    pa = analyse_periodic(t, cd, cl, dp, t_min=0.5)
    assert pa.periodic


def test_too_short_a_series_raises():
    t, cd, cl, dp = synthetic(0.005, t_end=1.0)
    with pytest.raises(ValueError):
        analyse_periodic(t, cd, cl, dp, t_min=0.2)


def test_frequency_estimators_agree_on_off_bin_frequency():
    """A window that is not an integer number of periods (so the FFT peak is off-bin): the
    zero-padded interpolated FFT and the harmonic fit must both recover f to ~1e-4 relative."""
    dt = 0.005
    t = np.arange(0.0, 6.0, dt)
    y = np.sin(2 * np.pi * F0 * t) + 0.05 * np.sin(2 * np.pi * 2 * F0 * t + 1.0)
    f_fft, _, _ = frequency_from_fft(t, y, t_start=2.0)
    f_fit = frequency_from_harmonic_fit(t, y, t_start=2.0, f0=3.0)
    assert f_fft == pytest.approx(F0, rel=2e-3)
    assert f_fit == pytest.approx(F0, rel=1e-6)


def test_pressure_difference_conventions_are_reported_at_the_right_instants():
    t, cd, cl, dp = synthetic(0.005, transient=False)
    pa = analyse_periodic(t, cd, cl, dp, t_min=0.5)
    # dp is in phase with c_L: at the c_L crest dp is at its crest, half a period later at its trough
    assert pa.values["dp_at_cl_max"] == pytest.approx(2.42, abs=1e-5)
    assert pa.values["dp_half_period_after_cl_max"] == pytest.approx(2.38, abs=1e-5)
