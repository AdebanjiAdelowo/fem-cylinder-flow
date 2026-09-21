"""DFG 2D-2 benchmark: unsteady periodic vortex shedding at Re = 100.

Time discretisation: Crank-Nicolson (theta = 0.5) on the Taylor-Hood spatial discretisation, Newton
solve each step (see src/navier_stokes.py). The residual, Newton problem, force forms and pressure
probe are built once and re-used every step.

Benchmark observables are extracted per lift cycle from cubic-spline extrema (src/periodic.py),
not from the largest raw time sample, and the periodic regime is judged from the cycle-to-cycle
drift. Four force evaluations are recorded side by side (src/forces.py); the reported benchmark
value uses ``--primary-force`` (default ``laplacian``, the benchmark's own definition).

Examples
--------
    python scripts/run_2d2_unsteady.py --config full                     # original baseline setup
    python scripts/run_2d2_unsteady.py --config full --geometry-order 2 --label full_p2
    python scripts/run_2d2_unsteady.py --h-far 0.03 --h-cyl 0.005 --dt 0.005 --t-end 12 \\
        --geometry-order 2 --init-state results/states/spinup.npz --label fine_p2
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ufl
import yaml
from dolfinx import fem
from dolfinx.fem import Function

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.geometry import build_mesh, cylinder_geometry_report, CYLINDER
from src.spaces import taylor_hood_space
from src.cylinder_bcs import build_bcs
from src.navier_stokes import (residual, make_newton_problem, solve_newton_problem, NewtonSettings)
from src.forces import ForceCoefficients, PointProbe, BENCHMARK_PRESSURE_POINTS, FORCE_METHODS
from src.periodic import analyse_periodic
from src.reference import INTERVALS_2D2, FEATFLOW_2D2, interval_status
from src.restart import load_state, save_state

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

NU = 1.0e-3
RHO = 1.0
D = 0.1
U_M = 1.5
U_BAR = 2.0 / 3.0 * U_M  # = 1.0, Re = U_BAR*D/nu = 100
THETA = 0.5  # Crank-Nicolson


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unavailable"


def environment_metadata() -> dict:
    import basix, dolfinx, gmsh, mpi4py, petsc4py, scipy  # noqa: E401
    from petsc4py import PETSc
    threads = {k: os.environ.get(k) for k in
               ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", "src", "scripts", "configs")),
        "dolfinx": dolfinx.__version__, "basix": basix.__version__, "ufl": ufl.__version__,
        "petsc": ".".join(map(str, PETSc.Sys.getVersion())), "gmsh": gmsh.__version__,
        "numpy": np.__version__, "scipy": scipy.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "cpu_count": os.cpu_count(), "thread_env": threads,
    }


def run_simulation(*, h_far, h_cyl, dt, t_end, geometry_order, label, force_methods,
                   init_state=None, save_state_path=None, theta=THETA, log_every=100,
                   checkpoint_every=200, resume=False) -> dict:
    """Integrate to t_end and return raw series plus run metadata.

    Every ``checkpoint_every`` steps the current state and the series so far are written to
    results/states/<label>_ckpt.npz and _partial.npz; ``resume=True`` continues from them, so a
    killed run (machine sleep, network-interface loss under MPICH) loses at most that many steps.
    """
    ckpt = RESULTS_DIR / "states" / f"{label}_ckpt.npz"
    partial = RESULTS_DIR / "states" / f"{label}_partial.npz"
    prior = None
    if resume and ckpt.exists() and partial.exists():
        init_state, prior = ckpt, dict(np.load(partial))
        print(f"[{label}] resuming from checkpoint at t={float(np.load(ckpt)['t'])}", flush=True)
    md = build_mesh(h_far, h_cyl, geometry_order=geometry_order)
    mesh = md.mesh
    geom = cylinder_geometry_report(mesh, md.facet_tags)
    W = taylor_hood_space(mesh)
    bcs = build_bcs(W, mesh, md.facet_tags, U_M)
    n_cells = mesh.topology.index_map(2).size_local
    n_dofs = W.dofmap.index_map.size_global * W.dofmap.index_map_bs

    if init_state is not None:
        w_prev, t_start, src_params = load_state(
            init_state, W, {"h_far": h_far, "h_cyl": h_cyl, "geometry_order": geometry_order})
        fem.set_bc(w_prev.x.array, bcs)      # interpolation leaves the boundary nodes approximate
        w_prev.x.scatter_forward()
        init = {"init_state": str(init_state), "t_start": t_start, "source_mesh": src_params}
    else:
        w_prev, t_start, init = Function(W), 0.0, {"init_state": None, "t_start": 0.0}
    w = Function(W)
    w.x.array[:] = w_prev.x.array

    w_test = ufl.TestFunction(W)
    settings = NewtonSettings()
    F = residual(w, w_test, NU, w_prev=w_prev, dt=dt, theta=theta)
    problem = make_newton_problem(F, w, bcs, settings, prefix=f"unsteady2d2_{label}_")
    forces = ForceCoefficients(w, md.facet_tags, CYLINDER, NU, U_BAR, D, RHO, methods=force_methods,
                               w_prev=w_prev, dt=dt, theta=theta)
    probe = PointProbe(mesh, BENCHMARK_PRESSURE_POINTS)

    original_init = init if prior is None else {
        "init_state": "checkpoint-resumed", "t_start": float(prior["t"][0]) - dt}
    n_steps = int(round((t_end - t_start) / dt))
    t_hist = np.empty(n_steps)
    fc = {m: np.empty((n_steps, 2)) for m in force_methods}
    dp_hist = np.empty(n_steps)
    newton_its = np.empty(n_steps, dtype=int)

    print(f"[{label}] cells={n_cells} dofs={n_dofs} geom_order={geometry_order} dt={dt} "
          f"t: {t_start} -> {t_end} ({n_steps} steps)", flush=True)
    wall0, cpu0 = time.perf_counter(), time.process_time()
    for step in range(n_steps):
        newton_its[step] = solve_newton_problem(problem)
        t_hist[step] = t_start + (step + 1) * dt
        for m, (cd, cl) in forces.all_coefficients().items():
            fc[m][step] = (cd, cl)
        dp_hist[step] = probe.evaluate(w.sub(1)) @ np.array([1.0, -1.0])
        w_prev.x.array[:] = w.x.array
        w_prev.x.scatter_forward()
        if checkpoint_every and (step + 1) % checkpoint_every == 0 and step + 1 < n_steps:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            save_state(ckpt, w, t_hist[step], h_far, h_cyl, geometry_order, dt)
            part = {"t": t_hist[:step + 1], "dp": dp_hist[:step + 1], "newton_it": newton_its[:step + 1]}
            for m in force_methods:
                part[f"c_D_{m}"], part[f"c_L_{m}"] = fc[m][:step + 1, 0], fc[m][:step + 1, 1]
            if prior is not None:
                part = {k: np.concatenate([prior[k], v]) for k, v in part.items()}
            np.savez(partial, **part)
        if log_every and (step + 1) % log_every == 0:
            print(f"[{label}] step {step+1}/{n_steps} t={t_hist[step]:.3f} "
                  f"wall={time.perf_counter()-wall0:.0f}s newton_it(last)={newton_its[step]}", flush=True)
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0

    if save_state_path is not None:
        pathlib.Path(save_state_path).parent.mkdir(parents=True, exist_ok=True)
        save_state(save_state_path, w, t_hist[-1], h_far, h_cyl, geometry_order, dt)

    meta = {
        "label": label, "h_far": h_far, "h_cyl": h_cyl, "geometry_order": geometry_order,
        "n_cells": int(n_cells), "n_dofs": int(n_dofs), "dt": dt, "t_end": t_end, "theta": theta,
        "n_steps": n_steps, "geometry": geom, "init": init, "force_methods": list(force_methods),
        "newton": {"rtol": settings.rtol, "atol": settings.atol, "max_it": settings.max_it,
                   "mean_iterations": float(newton_its.mean())},
        "linear_solver": "PETSc preonly + LU (MUMPS)", "mpi_ranks": mesh.comm.size,
        "wall_time_s": wall, "cpu_time_s": cpu, "wall_per_step_s": wall / max(n_steps, 1),
        "environment": environment_metadata(),
    }
    series = {"t": t_hist, "dp": dp_hist, "newton_it": newton_its}
    for m in force_methods:
        series[f"c_D_{m}"], series[f"c_L_{m}"] = fc[m][:, 0], fc[m][:, 1]
    if prior is not None:
        series = {k: np.concatenate([prior[k], v]) for k, v in series.items()}
        meta["init"] = original_init
        meta["resumed_from_checkpoint"] = True
        meta["n_steps"] = len(series["t"])
    for f in (ckpt, partial):
        f.unlink(missing_ok=True)
    return {"series": series, "meta": meta}


def _f(v, spec):
    """Format a metadata number, or 'n/a' when the series predates metadata capture."""
    return "n/a" if v is None else format(v, spec)


def analyse_and_report(label, series, meta, *, primary="laplacian", t_min=None, n_last=4,
                       write=True) -> dict:
    """Per-cycle analysis of every recorded force method; write JSON, text report, figure."""
    t = series["t"]
    dt = meta["dt"]
    methods = [k[len("c_D_"):] for k in series if k.startswith("c_D_")]
    if t_min is None:
        t_min = meta["init"]["t_start"] + 1.0
    out = {}
    for m in methods:
        # the variational force is centred at t_{n+1/2}; the traction forms at t_{n+1}
        tm = t - 0.5 * dt if m == "variational" else t
        out[m] = analyse_periodic(tm, series[f"c_D_{m}"], series[f"c_L_{m}"], series["dp"],
                                  D=D, U_bar=U_BAR, t_min=t_min, n_last=n_last)
    if not write:
        return out

    prim = out[primary]
    g = meta.get("geometry") or {}
    lines = [
        f"DFG 2D-2 benchmark (Re=100, unsteady), run label = {label}",
        f"mesh: cells={meta.get('n_cells')}, dofs={meta.get('n_dofs')}, geometry order={meta.get('geometry_order')} "
        f"(h_far={meta.get('h_far')}, h_cyl={meta.get('h_cyl')}), dt={dt}, theta={meta.get('theta')}, "
        f"t in [{meta['init']['t_start']}, {meta['t_end']}]",
        f"cylinder geometry: perimeter err {_f(g.get('perimeter_rel_err'), '+.3e')}, "
        f"area err {_f(g.get('area_rel_err'), '+.3e')}, "
        f"max radial deviation {_f(g.get('max_radial_dev_rel'), '.3e')} R, "
        f"{g.get('n_cyl_facets', 'n/a')} facets",
        f"init: {meta['init']}",
        f"cost: wall={_f(meta.get('wall_time_s'), '.1f')}s cpu={_f(meta.get('cpu_time_s'), '.1f')}s "
        f"({_f(meta.get('wall_per_step_s'), '.3f')} s/step), "
        f"Newton its/step={_f((meta.get('newton') or {}).get('mean_iterations'), '.2f')}",
        f"analysis from t >= {t_min}; {len(prim.cycles)} lift cycles (c_L min -> c_L min); "
        f"drift = |last - previous cycle|, spread = max-min over last {n_last} cycles",
        "",
    ]
    hdr = f"{'force method':<13}{'c_D,max':>10}{'c_L,max':>10}{'c_L,min':>10}{'period':>10}" \
          f"{'St(cyc)':>9}{'St(fft)':>9}{'St(fit)':>9}{'dP@clmax':>10}{'dP@+T/2':>9}  periodic"
    lines.append(hdr)
    for m, pa in out.items():
        v = pa.values
        lines.append(f"{m + ('*' if m == primary else ''):<13}{v['cd_max']:>10.5f}{v['cl_max']:>10.5f}"
                     f"{v['cl_min']:>10.5f}{v['period']:>10.5f}{pa.strouhal['cycle_period']:>9.5f}"
                     f"{pa.strouhal['fft']:>9.5f}{pa.strouhal['harmonic_fit']:>9.5f}"
                     f"{v['dp_at_cl_max']:>10.4f}{v['dp_half_period_after_cl_max']:>9.4f}  {pa.periodic}")
    lines += ["", f"(* primary = {primary}.  variational force is centred at t_(n+1/2).)", "",
              f"--- periodic-regime evidence, primary force ({primary}) ---",
              f"{'quantity':<14}{'last cycle':>12}{'drift':>11}{'spread(last %d)' % n_last:>16}{'tolerance':>11}"]
    for k in ("cd_max", "cl_max", "cl_min", "cl_amp", "period"):
        tol = prim.tolerance.get(k)
        lines.append(f"{k:<14}{prim.values[k]:>12.5f}{prim.drift[k]:>11.2e}{prim.spread[k]:>16.2e}"
                     f"{('%.0e' % tol) if tol else '-':>11}")
    iu = prim.interp_uncertainty
    lines += [
        f"extraction: spline vs local-parabola extremum differ by {iu['cl_max']:.1e} (c_L,max), "
        f"{iu['cd_max']:.1e} (c_D,max); a raw-sample maximum would have been low by "
        f"{iu['cl_max_raw_sample_bias']:.1e} (c_L,max), {iu['cd_max_raw_sample_bias']:.1e} (c_D,max)",
        f"Strouhal, primary: cycle period {prim.strouhal['cycle_period']:.5f}, "
        f"FFT {prim.strouhal['fft']:.5f} (window {prim.fft_window[0]:.2f}-{prim.fft_window[1]:.2f} s), "
        f"harmonic fit {prim.strouhal['harmonic_fit']:.5f}; definition St = f*D/Ubar, "
        f"f = 1/mean(cycle period over last {n_last} cycles) for the headline value",
        "",
        "--- position relative to the repository's comparison intervals (see src/reference.py) ---",
    ]
    for name, val, ref in [("c_D,max", prim.values["cd_max"], INTERVALS_2D2["cd_max"]),
                           ("c_L,max", prim.values["cl_max"], INTERVALS_2D2["cl_max"]),
                           ("St", prim.strouhal["cycle_period"], INTERVALS_2D2["st"]),
                           ("dP @ c_L,max", prim.values["dp_at_cl_max"], INTERVALS_2D2["dP"]),
                           ("dP @ +T/2", prim.values["dp_half_period_after_cl_max"], INTERVALS_2D2["dP"])]:
        lines.append(f"  {name:<14} {val:.5f}  interval {ref}: {interval_status(val, ref)}")
    f6 = FEATFLOW_2D2[6]["1/1600"]
    lines += ["", f"FeatFlow level-6 (667k dof) dt=1/1600 for orientation: c_D,max={f6[0]}, "
                  f"c_L,max={f6[1]}, period={f6[2]}, dP@c_L,max={f6[3]}"]

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"2d2_{label}.txt").write_text("\n".join(lines) + "\n")
    summary = {"meta": meta, "primary_force": primary, "analysis_t_min": t_min, "n_last": n_last,
               "methods": {m: {"values": pa.values, "drift": pa.drift, "spread": pa.spread,
                               "strouhal": pa.strouhal, "interp_uncertainty": pa.interp_uncertainty,
                               "periodic": pa.periodic, "n_cycles": len(pa.cycles),
                               "cycles": [vars(c) for c in pa.cycles]}
                           for m, pa in out.items()}}
    (RESULTS_DIR / f"2d2_{label}.json").write_text(json.dumps(summary, indent=1, default=float))
    np.savez(RESULTS_DIR / f"2d2_timeseries_{label}.npz", **series)
    print("\n".join(lines))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.plot(t, series[f"c_D_{primary}"], label="$c_D(t)$")
    ax1.plot(t, series[f"c_L_{primary}"], label="$c_L(t)$")
    ax1.set_ylim(-2, 5)
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("coefficient")
    ax1.set_title(f"Force coefficients ({primary} traction), y-clipped")
    ax1.legend()
    ax1.grid(alpha=0.3)
    idx = np.arange(len(prim.cycles))
    ax2.plot(idx, [c.cl_max for c in prim.cycles], "o-", label="$c_{L,max}$ per cycle")
    ax2.set_xlabel("lift cycle index")
    ax2.set_ylabel("$c_{L,max}$")
    ax2b = ax2.twinx()
    ax2b.plot(idx, [c.cd_max for c in prim.cycles], "s-", color="C1", label="$c_{D,max}$ per cycle")
    ax2b.set_ylabel("$c_{D,max}$")
    ax2.set_title("Approach to the periodic regime (cycle maxima)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"2d2_{label}.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote results/2d2_{label}.txt, .json, timeseries npz and figures/2d2_{label}.png")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, choices=["smoke", "local", "full"],
                        help="Take mesh/dt/t_end defaults from configs/<name>.yaml.")
    parser.add_argument("--h-far", type=float, default=None)
    parser.add_argument("--h-cyl", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--dt-override", type=float, default=None, help="Alias of --dt (kept for old scripts).")
    parser.add_argument("--t-end", type=float, default=None, help="Absolute final time.")
    parser.add_argument("--geometry-order", type=int, default=1, choices=[1, 2],
                        help="1 = straight-sided cells (polygonal cylinder, the original behaviour); "
                             "2 = curved isoparametric cylinder.")
    parser.add_argument("--force-methods", default=",".join(FORCE_METHODS))
    parser.add_argument("--primary-force", default="laplacian", choices=list(FORCE_METHODS))
    parser.add_argument("--init-state", default=None, help="Start from a saved state (see --save-state).")
    parser.add_argument("--save-state", default=None, help="Write the final state to this .npz path.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from results/states/<label>_ckpt.npz if a previous run of this label was killed.")
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--analysis-t-min", type=float, default=None)
    parser.add_argument("--n-last", type=int, default=4)
    parser.add_argument("--label", type=str, default=None)
    args = parser.parse_args()

    cfg = {}
    if args.config:
        cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())["2d2_unsteady"]
    h_far = args.h_far if args.h_far is not None else cfg.get("h_far")
    h_cyl = args.h_cyl if args.h_cyl is not None else cfg.get("h_cyl")
    dt = args.dt if args.dt is not None else (args.dt_override if args.dt_override is not None else cfg.get("dt"))
    t_end = args.t_end if args.t_end is not None else cfg.get("t_end")
    if None in (h_far, h_cyl, dt, t_end):
        parser.error("give --config or all of --h-far --h-cyl --dt --t-end")
    label = args.label or args.config or "run"
    methods = tuple(m for m in args.force_methods.split(",") if m)

    res = run_simulation(h_far=h_far, h_cyl=h_cyl, dt=dt, t_end=t_end, geometry_order=args.geometry_order,
                         label=label, force_methods=methods, init_state=args.init_state,
                         save_state_path=args.save_state, resume=args.resume,
                         checkpoint_every=args.checkpoint_every)
    RESULTS_DIR.mkdir(exist_ok=True)
    np.savez(RESULTS_DIR / f"2d2_timeseries_{label}.npz", **res["series"])   # raw data first, in case analysis fails
    analyse_and_report(label, res["series"], res["meta"], primary=args.primary_force,
                       t_min=args.analysis_t_min, n_last=args.n_last)


if __name__ == "__main__":
    main()
