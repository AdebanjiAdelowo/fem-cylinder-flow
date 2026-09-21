"""Periodic-regime analysis of the unsteady DFG 2D-2 force/pressure signals.

The benchmark (Schaefer & Turek 1996; FeatFlow "DFG benchmark 2D-2" page) asks for the maxima of
c_D and c_L over one cycle of the fully developed flow, where a cycle runs from one minimum of c_L to
the next. Taking the largest raw time sample of a run is not equivalent: the sampling interval is
only ~1/66 of the period, so a sampled peak can miss the true peak by ~1e-3 (the same size as the
effects being studied), and a transient that has not yet died away biases the value low.

This module therefore works cycle by cycle:

* extrema are located on a cubic-spline interpolant of the sampled signal, not on the samples;
* cycles are delimited by successive minima of c_L (the benchmark's own definition);
* each cycle reports c_D/c_L extrema, period, and pressure difference, so that convergence to the
  periodic regime can be judged from the cycle-to-cycle drift instead of assumed;
* the shedding frequency is estimated three independent ways (cycle period, windowed FFT with
  zero-padding, harmonic least-squares fit) so that they can be cross-checked.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


@dataclass
class Cycle:
    """One lift cycle [t0, t1], t0 and t1 being successive minima of c_L."""
    t0: float
    t1: float
    period: float
    cd_max: float
    cd_min: float
    cl_max: float
    cl_min: float
    t_cl_max: float
    dp_at_cl_max: float | None = None
    dp_half_period_after_cl_max: float | None = None
    dp_min: float | None = None
    dp_max: float | None = None

    @property
    def cd_mean(self) -> float:
        return 0.5 * (self.cd_max + self.cd_min)   # benchmark definition: (max + min) / 2

    @property
    def cl_amp(self) -> float:
        return self.cl_max - self.cl_min


def spline_extrema(t: np.ndarray, y: np.ndarray, kind: str = "max"):
    """All interior local maxima ('max') or minima ('min') of the cubic spline through (t, y).

    Returns (times, values), located from the roots of the spline derivative, so the result does
    not depend on the sample spacing (unlike the largest sample).
    """
    cs = CubicSpline(t, y)
    roots = cs.derivative().roots(extrapolate=False)
    if roots.size == 0:
        return np.array([]), np.array([])
    curv = cs.derivative(2)(roots)
    keep = curv < 0 if kind == "max" else curv > 0
    tk = roots[keep]
    return tk, cs(tk)


def _window_extremum(cs: CubicSpline, t0: float, t1: float, kind: str):
    """Global max/min of the spline on [t0, t1] (interior critical points plus the end points)."""
    d1 = cs.derivative()
    r = d1.roots(extrapolate=False)
    r = r[(r >= t0) & (r <= t1)]
    cand = np.concatenate([[t0, t1], r])
    vals = cs(cand)
    i = np.argmax(vals) if kind == "max" else np.argmin(vals)
    return float(cand[i]), float(vals[i])


def _local_quadratic_peak(t, y, t_peak, half_width=2, kind="max"):
    """Independent extremum estimate: least-squares parabola through the samples nearest t_peak.
    Used only to bound the interpolation uncertainty of the spline extremum."""
    i = int(np.argmin(np.abs(t - t_peak)))
    lo, hi = max(0, i - half_width), min(len(t), i + half_width + 1)
    c = np.polyfit(t[lo:hi] - t[i], y[lo:hi], 2)
    if c[0] == 0:
        return float(y[i])
    tv = -c[1] / (2 * c[0])
    return float(np.polyval(c, tv))


def find_cycles(t, cd, cl, dp=None, t_min: float = 1.0) -> list[Cycle]:
    """Split the series into lift cycles (successive minima of c_L) and compute per-cycle statistics.

    `t_min` skips the impulsive-start spike when locating minima; it is an analysis start, not a
    claim that the flow is periodic from there (that is judged from the per-cycle drift).
    """
    t, cd, cl = map(np.asarray, (t, cd, cl))
    m = t >= t_min
    cs_l = CubicSpline(t[m], cl[m])
    cs_d = CubicSpline(t[m], cd[m])
    cs_p = CubicSpline(t[m], np.asarray(dp)[m]) if dp is not None else None

    tmins, _ = spline_extrema(t[m], cl[m], "min")
    cycles: list[Cycle] = []
    for a, b in zip(tmins[:-1], tmins[1:]):
        T = float(b - a)
        t_clmax, clmax = _window_extremum(cs_l, a, b, "max")
        _, clmin = _window_extremum(cs_l, a, b, "min")
        _, cdmax = _window_extremum(cs_d, a, b, "max")
        _, cdmin = _window_extremum(cs_d, a, b, "min")
        c = Cycle(t0=float(a), t1=float(b), period=T, cd_max=cdmax, cd_min=cdmin,
                  cl_max=clmax, cl_min=clmin, t_cl_max=t_clmax)
        if cs_p is not None:
            c.dp_at_cl_max = float(cs_p(t_clmax))
            t_half = t_clmax + T / 2 if t_clmax + T / 2 <= b else t_clmax - T / 2
            c.dp_half_period_after_cl_max = float(cs_p(t_half))
            _, c.dp_max = _window_extremum(cs_p, a, b, "max")
            _, c.dp_min = _window_extremum(cs_p, a, b, "min")
        cycles.append(c)
    return cycles


# ---------------------------------------------------------------------------------------------
# Frequency estimators
# ---------------------------------------------------------------------------------------------

def frequency_from_cycles(cycles: list[Cycle], n_last: int) -> float:
    """Mean frequency over the last n_last cycles = n / (elapsed time between first and last minimum)."""
    use = cycles[-n_last:]
    return len(use) / (use[-1].t1 - use[0].t0)


def frequency_from_fft(t, y, t_start: float, pad_factor: int = 32):
    """Peak frequency of a Hann-windowed, zero-padded FFT of y(t) for t >= t_start, refined by
    three-point quadratic interpolation of the log-magnitude around the peak bin."""
    t, y = np.asarray(t), np.asarray(y)
    m = t >= t_start
    ts, ys = t[m], y[m] - np.mean(y[m])
    dt = ts[1] - ts[0]
    n = len(ys)
    nfft = pad_factor * n
    spec = np.abs(np.fft.rfft(ys * np.hanning(n), nfft))
    freqs = np.fft.rfftfreq(nfft, d=dt)
    k = int(np.argmax(spec[1:-1])) + 1
    y1, y2, y3 = np.log(spec[k - 1:k + 2] + 1e-300)
    den = y1 - 2 * y2 + y3
    delta = 0.5 * (y1 - y3) / den if den != 0 else 0.0
    return float(freqs[k] + delta * (freqs[1] - freqs[0])), freqs, spec


def frequency_from_harmonic_fit(t, y, t_start: float, f0: float, n_harm: int = 3):
    """Least-squares fit of y = a0 + sum_h [a_h cos(2 pi h f t) + b_h sin(2 pi h f t)]; the
    frequency f is found by 1-D minimisation of the residual (variable projection) around f0."""
    t, y = np.asarray(t), np.asarray(y)
    m = t >= t_start
    ts, ys = t[m], y[m]

    def resid(f):
        cols = [np.ones_like(ts)]
        for h in range(1, n_harm + 1):
            cols += [np.cos(2 * np.pi * h * f * ts), np.sin(2 * np.pi * h * f * ts)]
        A = np.stack(cols, axis=1)
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        return float(np.sum((A @ coef - ys) ** 2))

    res = minimize_scalar(resid, bounds=(0.9 * f0, 1.1 * f0), method="bounded",
                          options={"xatol": 1e-9})
    return float(res.x)


# ---------------------------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------------------------

@dataclass
class PeriodicAnalysis:
    cycles: list[Cycle]
    n_last: int
    D: float
    U_bar: float
    values: dict = field(default_factory=dict)      # value taken from the last complete cycle
    drift: dict = field(default_factory=dict)       # |last - previous| cycle increment
    spread: dict = field(default_factory=dict)      # max - min over the last n_last cycles
    interp_uncertainty: dict = field(default_factory=dict)
    strouhal: dict = field(default_factory=dict)
    fft_window: tuple = (0.0, 0.0)
    periodic: bool = False
    tolerance: dict = field(default_factory=dict)


# per-quantity absolute drift tolerances between the last two cycles (documented, not tuned to any
# benchmark interval): well below the smallest effect being resolved (~1e-3), above sampling noise.
DEFAULT_DRIFT_TOL = {"cd_max": 5e-5, "cl_max": 1e-4, "cl_min": 1e-4, "period": 2e-4}


def analyse_periodic(t, cd, cl, dp=None, *, D: float = 0.1, U_bar: float = 1.0, t_min: float = 1.0,
                     n_last: int = 4, fft_window_length: float | None = None,
                     drift_tol: dict | None = None) -> PeriodicAnalysis:
    """Extract benchmark observables from the last complete cycle and judge the periodic regime.

    values     : per-quantity value on the LAST complete cycle
    drift      : change of that quantity between the last two cycles (the periodic-regime criterion)
    spread     : max-min over the last `n_last` cycles
    interp_uncertainty : |spline extremum - local-parabola extremum| for c_L,max and c_D,max
    strouhal   : St from (i) mean cycle period, (ii) windowed zero-padded FFT, (iii) harmonic fit
    """
    t, cd, cl = map(np.asarray, (t, cd, cl))
    tol = dict(DEFAULT_DRIFT_TOL if drift_tol is None else drift_tol)
    cycles = find_cycles(t, cd, cl, dp, t_min)
    if len(cycles) < n_last + 1:
        raise ValueError(f"only {len(cycles)} complete cycles found; need at least {n_last + 1}")

    last, prev = cycles[-1], cycles[-2]
    tail = cycles[-n_last:]
    names = ["cd_max", "cd_min", "cl_max", "cl_min", "period", "cl_amp", "cd_mean"]
    if dp is not None:
        names += ["dp_at_cl_max", "dp_half_period_after_cl_max", "dp_min", "dp_max"]
    pa = PeriodicAnalysis(cycles=cycles, n_last=n_last, D=D, U_bar=U_bar, tolerance=tol)
    for n in names:
        pa.values[n] = getattr(last, n)
        pa.drift[n] = abs(getattr(last, n) - getattr(prev, n))
        vals = [getattr(c, n) for c in tail]
        pa.spread[n] = max(vals) - min(vals)
    pa.periodic = all(pa.drift[k] <= v for k, v in tol.items())

    # interpolation uncertainty of the spline extremum vs a local-parabola fit on the samples
    for name, sig, kind in (("cl_max", cl, "max"), ("cd_max", cd, "max")):
        t_pk = last.t_cl_max if name == "cl_max" else _window_extremum(
            CubicSpline(t[t >= t_min], cd[t >= t_min]), last.t0, last.t1, "max")[0]
        pa.interp_uncertainty[name] = abs(pa.values[name] - _local_quadratic_peak(t, sig, t_pk))
    # sampling error a naive 'largest raw sample' extraction would have made on the same cycle
    in_cycle = (t >= last.t0) & (t <= last.t1)
    pa.interp_uncertainty["cl_max_raw_sample_bias"] = pa.values["cl_max"] - float(np.max(cl[in_cycle]))
    pa.interp_uncertainty["cd_max_raw_sample_bias"] = pa.values["cd_max"] - float(np.max(cd[in_cycle]))

    # Strouhal: three independent estimates
    f_cyc = frequency_from_cycles(cycles, n_last)
    win = fft_window_length if fft_window_length is not None else max(4.0, 8 * last.period)
    win = min(win, t[-1] - t_min)
    t_start = t[-1] - win
    f_fft, _, _ = frequency_from_fft(t, cl, t_start)
    f_fit = frequency_from_harmonic_fit(t, cl, t_start, f_cyc)
    conv = D / U_bar
    pa.strouhal = {"cycle_period": f_cyc * conv, "fft": f_fft * conv, "harmonic_fit": f_fit * conv}
    pa.fft_window = (float(t_start), float(t[-1]))
    return pa
