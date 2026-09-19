"""DFG 2D-2 benchmark: unsteady periodic vortex shedding at Re = 100.

Reference values (Schäfer & Turek, 1996, Table 4, "lower/upper bound" row):
    c_D,max in [3.2200, 3.2400]
    c_L,max in [0.9900, 1.0100]
    St      in [0.2950, 0.3050]
    Delta P in [2.4600, 2.5000]

Time discretisation: Crank-Nicolson (theta = 0.5) on the Taylor-Hood
spatial discretisation, Newton solve each step (see src/navier_stokes.py).

Usage: python scripts/run_2d2_unsteady.py --config smoke|local|full
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ufl
import yaml
from dolfinx.fem import Function

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.geometry import build_mesh, CYLINDER
from src.spaces import taylor_hood_space
from src.cylinder_bcs import build_bcs
from src.navier_stokes import residual, newton_solve
from src.forces import compute_drag_lift, pressure_difference

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

NU = 1.0e-3
RHO = 1.0
D = 0.1
U_M = 1.5
U_BAR = 2.0 / 3.0 * U_M  # = 1.0, Re = U_BAR*D/nu = 100
THETA = 0.5  # Crank-Nicolson

REF_CD_MAX = (3.2200, 3.2400)
REF_CL_MAX = (0.9900, 1.0100)
REF_ST = (0.2950, 0.3050)
REF_DP = (2.4600, 2.5000)


def strouhal_from_series(t: np.ndarray, c_L: np.ndarray, discard_frac: float = 0.5):
    """Estimate the shedding frequency from an FFT of c_L(t), after
    discarding the initial transient (first `discard_frac` of the series).

    The raw FFT bin spacing is 1/(window duration): for an 8 s run with a
    50% transient discard, the analysis window is only 4 s, giving a bin
    spacing of 0.25 Hz (Delta-St = 0.1*0.25 = 0.025) -- coarser than the
    ENTIRE published reference interval width for St (0.01). Reporting the
    raw argmax bin frequency would therefore be a meaningless comparison.
    Instead, three-point parabolic (quadratic) interpolation on the
    log-magnitude spectrum around the peak bin is used to recover sub-bin
    frequency resolution (a standard DSP technique; see e.g. Smith, J.O.,
    "Spectral Audio Signal Processing", quadratic peak interpolation), and
    is independently cross-checked in the README against the mean period
    measured directly from successive lift-signal peak times.
    """
    n0 = int(len(t) * discard_frac)
    t_s, c_s = t[n0:], c_L[n0:]
    c_s = c_s - np.mean(c_s)
    dt = t_s[1] - t_s[0]
    n = len(c_s)
    spectrum = np.abs(np.fft.rfft(c_s * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=dt)
    k = np.argmax(spectrum[1:]) + 1  # skip the zero-frequency bin

    log_mag = np.log(spectrum + 1e-300)
    y1, y2, y3 = log_mag[k - 1], log_mag[k], log_mag[k + 1]
    denom = y1 - 2 * y2 + y3
    delta = 0.5 * (y1 - y3) / denom if denom != 0 else 0.0
    f_shed = freqs[k] + delta * (freqs[1] - freqs[0])

    return f_shed, freqs, spectrum, t_s, c_s


def period_from_peaks(t: np.ndarray, c_L: np.ndarray, discard_frac: float = 0.5):
    """Independent cross-check: mean period from successive local maxima of
    c_L(t) in the post-transient window (direct time-domain measurement,
    no FFT/windowing involved)."""
    n0 = int(len(t) * discard_frac)
    t_s, c_s = t[n0:], c_L[n0:]
    peak_times = [
        t_s[i] for i in range(1, len(t_s) - 1) if c_s[i] > c_s[i - 1] and c_s[i] > c_s[i + 1]
    ]
    if len(peak_times) < 2:
        return None, peak_times
    periods = np.diff(peak_times)
    return float(np.mean(periods)), peak_times


def postprocess_and_report(config_name: str, t_arr, cD_arr, cL_arr, dp_arr,
                            n_cells: int, n_dofs: int, dt: float, t_end: float,
                            elapsed: float | None = None) -> None:
    f_shed, freqs, spectrum, t_s, c_s = strouhal_from_series(t_arr, cL_arr)
    St_fft = f_shed * D / U_BAR

    mean_period, peak_times = period_from_peaks(t_arr, cL_arr)
    St_peaks = (1.0 / mean_period) * D / U_BAR if mean_period else float("nan")

    n0 = int(len(t_arr) * 0.5)
    cD_max = float(np.max(cD_arr[n0:]))
    cL_max = float(np.max(cL_arr[n0:]))
    dp_at_cLmax = float(dp_arr[n0:][np.argmax(cL_arr[n0:])])

    # saturation check: are the last few lift peaks flat (limit cycle reached)?
    peak_vals = []
    for tp in peak_times:
        idx = int(np.argmin(np.abs(t_arr - tp)))
        peak_vals.append(cL_arr[idx])
    tail_peaks = peak_vals[-6:] if len(peak_vals) >= 6 else peak_vals
    saturation_spread = (max(tail_peaks) - min(tail_peaks)) if tail_peaks else float("nan")

    elapsed_str = f"{elapsed:.2f}s" if elapsed is not None else "N/A (reprocessed from saved time series)"
    lines = [
        f"DFG 2D-2 benchmark (Re=100, unsteady), config={config_name}",
        f"mesh: cells={n_cells}, dofs={n_dofs}, dt={dt}, t_end={t_end}, theta={THETA} "
        f"(Crank-Nicolson), wall_time={elapsed_str}",
        f"Reference (Schafer-Turek 1996, Table 4): c_D,max in {REF_CD_MAX}, "
        f"c_L,max in {REF_CL_MAX}, St in {REF_ST}, dP in {REF_DP}",
        "",
        "--- Saturation check (last <=6 lift peaks in the post-transient window) ---",
        "  " + ", ".join(f"{v:.5f}" for v in tail_peaks),
        f"  spread = {saturation_spread:.5f} "
        f"({'flat: treated as converged limit cycle' if saturation_spread < 0.01 else 'NOT flat: transient may still be present'})",
        "",
        f"computed c_D,max = {cD_max:.4f}  (max over last half of run)",
        f"computed c_L,max = {cL_max:.4f}  (max over last half of run)",
        f"computed dP(t at c_L,max) = {dp_at_cLmax:.4f}",
        "",
        "--- Strouhal number: two independent estimates ---",
        f"  FFT (Hanning window, {t_s[-1]-t_s[0]:.2f}s post-transient window, "
        f"quadratic log-magnitude peak interpolation): f = {f_shed:.4f} Hz -> St = {St_fft:.4f}",
        f"  Direct peak-to-peak period ({len(peak_times)} peaks, mean period "
        f"{mean_period:.5f}s): f = {1.0/mean_period:.4f} Hz -> St = {St_peaks:.4f}"
        if mean_period else "  Direct peak-to-peak: insufficient peaks found",
        f"  agreement between the two methods: "
        f"{abs(St_fft-St_peaks)/St_fft*100:.2f}% relative difference" if mean_period else "",
        "",
    ]
    for name, val, ref in [("c_D,max", cD_max, REF_CD_MAX), ("c_L,max", cL_max, REF_CL_MAX),
                            ("St (FFT)", St_fft, REF_ST), ("dP", dp_at_cLmax, REF_DP)]:
        inside = ref[0] <= val <= ref[1]
        if inside:
            status = "INSIDE interval"
        else:
            nearest = ref[0] if val < ref[0] else ref[1]
            pct = (val - nearest) / nearest * 100
            status = f"OUTSIDE interval by {pct:+.3f}% (vs. nearest bound)"
        lines.append(f"  {name}: computed={val:.5f}, reference interval={ref}, {status}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"2d2_unsteady_{config_name}.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)

    # figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.plot(t_arr, cD_arr, label="$c_D(t)$")
    ax1.plot(t_arr, cL_arr, label="$c_L(t)$")
    ax1.set_ylim(-2, 5)
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("coefficient")
    ax1.set_title("Force coefficients (y-clipped; impulsive-start spike\n"
                   "at t~0 omitted from view, present in raw data)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    f_max_plot = max(3.0 * f_shed, 2.0)
    ax2.plot(freqs, spectrum)
    ax2.axvline(f_shed, color="r", linestyle="--",
                label=f"interpolated peak f={f_shed:.4f} Hz (St={St_fft:.4f})")
    ax2.set_xlim(0, f_max_plot)
    ax2.set_xlabel("frequency [Hz]")
    ax2.set_ylabel("|FFT($c_L$)|")
    ax2.set_title("Shedding-frequency spectrum (post-transient window)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"2d2_force_coefficients_{config_name}.png", dpi=150)

    np.savez(RESULTS_DIR / f"2d2_timeseries_{config_name}.npz",
             t=t_arr, c_D=cD_arr, c_L=cL_arr, dp=dp_arr)
    print(f"\nWrote results/2d2_unsteady_{config_name}.txt, "
          f"figures/2d2_force_coefficients_{config_name}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    parser.add_argument("--dt-override", type=float, default=None,
                         help="Use this dt instead of the config's, keeping its mesh -- for "
                              "isolating temporal from spatial sensitivity.")
    parser.add_argument("--label", type=str, default=None,
                         help="Output filename label; defaults to --config.")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())["2d2_unsteady"]
    h_far, h_cyl, dt, t_end = cfg["h_far"], cfg["h_cyl"], cfg["dt"], cfg["t_end"]
    if args.dt_override is not None:
        dt = args.dt_override
    label = args.label or args.config

    md = build_mesh(h_far, h_cyl)
    mesh = md.mesh
    W = taylor_hood_space(mesh)
    bcs = build_bcs(W, mesh, md.facet_tags, U_M)
    n_cells = mesh.topology.index_map(2).size_local
    n_dofs = W.dofmap.index_map.size_global * W.dofmap.index_map_bs

    w = Function(W)       # current time level
    w_prev = Function(W)  # previous time level, starts at rest
    w_test = ufl.TestFunction(W)

    n_steps = int(round(t_end / dt))
    t_hist, cD_hist, cL_hist, dp_hist = [0.0], [0.0], [0.0], [0.0]

    t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        F = residual(w, w_test, NU, w_prev=w_prev, dt=dt, theta=THETA)
        newton_solve(F, w, bcs, mesh, prefix=f"unsteady2d2_{label}_")

        u_h, p_h = w.sub(0), w.sub(1)
        c_D, c_L = compute_drag_lift(u_h, p_h, NU, mesh, md.facet_tags, CYLINDER, U_BAR, D, RHO)
        dp = pressure_difference(p_h, mesh)

        t_hist.append(step * dt)
        cD_hist.append(c_D)
        cL_hist.append(c_L)
        dp_hist.append(dp)

        w_prev.x.array[:] = w.x.array
    elapsed = time.perf_counter() - t0

    postprocess_and_report(
        label, np.array(t_hist), np.array(cD_hist), np.array(cL_hist), np.array(dp_hist),
        n_cells, n_dofs, dt, t_end, elapsed,
    )


if __name__ == "__main__":
    main()
