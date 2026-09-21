"""DFG 2D-1 benchmark: steady flow around a cylinder at Re = 20.

Reference values (see src/reference.py for provenance):
    Schaefer & Turek (1996), Table 3, lower/upper bound row (carried over, not re-verified here):
        c_D in [5.5700, 5.5900], c_L in [0.0104, 0.0110], Delta P in [0.1172, 0.1176]
    High-order spectral values (Nabh 1998, FeatFlow page):
        c_D = 5.57953523384, c_L = 0.010618948146, Delta P = 0.11752016697

Four force evaluations are recorded (src/forces.py); the headline column uses ``--primary-force``
(default ``laplacian``, the benchmark's own stress definition).

Usage: python scripts/run_2d1_steady.py --config smoke|local|full [--geometry-order 1|2] [--label X]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import ufl
import yaml
from dolfinx.fem import Function

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.geometry import build_mesh, cylinder_geometry_report, CYLINDER
from src.spaces import taylor_hood_space
from src.cylinder_bcs import build_bcs
from src.navier_stokes import residual, make_newton_problem, solve_newton_problem, NewtonSettings
from src.forces import ForceCoefficients, PointProbe, BENCHMARK_PRESSURE_POINTS, FORCE_METHODS
from src.reference import INTERVALS_2D1, SPECTRAL_2D1, interval_status

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

NU = 1.0e-3
RHO = 1.0
D = 0.1
U_M = 0.3
U_BAR = 2.0 / 3.0 * U_M  # mean inflow velocity, Re = U_BAR*D/nu = 20

# kept for scripts that import the old names
REF_CD, REF_CL, REF_DP = INTERVALS_2D1["c_D"], INTERVALS_2D1["c_L"], INTERVALS_2D1["dP"]


def run_one(h_far: float, h_cyl: float, geometry_order: int = 1, primary: str = "laplacian"):
    md = build_mesh(h_far, h_cyl, geometry_order=geometry_order)
    mesh = md.mesh
    geom = cylinder_geometry_report(mesh, md.facet_tags)
    W = taylor_hood_space(mesh)
    bcs = build_bcs(W, mesh, md.facet_tags, U_M)

    w = Function(W)
    w_test = ufl.TestFunction(W)
    problem = make_newton_problem(residual(w, w_test, NU), w, bcs, NewtonSettings(),
                                  prefix=f"steady2d1_{h_far}_{h_cyl}_{geometry_order}_")
    t0 = time.perf_counter()
    n_it = solve_newton_problem(problem)
    elapsed = time.perf_counter() - t0

    forces = ForceCoefficients(w, md.facet_tags, CYLINDER, NU, U_BAR, D, RHO)
    all_f = forces.all_coefficients()
    probe = PointProbe(mesh, BENCHMARK_PRESSURE_POINTS)
    pa, pb = probe.evaluate(w.sub(1))
    c_D, c_L = all_f[primary]

    n_cells = mesh.topology.index_map(2).size_local
    n_dofs = W.dofmap.index_map.size_global * W.dofmap.index_map_bs
    return {
        "h_far": h_far, "h_cyl": h_cyl, "geometry_order": geometry_order, "n_cells": int(n_cells),
        "n_dofs": int(n_dofs), "newton_it": n_it, "wall_time_s": elapsed,
        "c_D": c_D, "c_L": c_L, "dp": float(pa - pb),
        "forces": {m: {"c_D": a, "c_L": b} for m, (a, b) in all_f.items()},
        "geometry": geom,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    parser.add_argument("--geometry-order", type=int, default=1, choices=[1, 2])
    parser.add_argument("--primary-force", default="laplacian", choices=list(FORCE_METHODS))
    parser.add_argument("--label", default=None, help="output label; defaults to --config")
    args = parser.parse_args()
    label = args.label or args.config

    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())
    rows = []
    for lvl in cfg["2d1_mesh_levels"]:
        r = run_one(lvl["h_far"], lvl["h_cyl"], args.geometry_order, args.primary_force)
        rows.append(r)
        print(f"h_far={r['h_far']:.4f} h_cyl={r['h_cyl']:.4f}  cells={r['n_cells']:6d}  "
              f"dofs={r['n_dofs']:7d}  Newton_it={r['newton_it']}  time={r['wall_time_s']:.2f}s  "
              f"c_D={r['c_D']:.5f}  c_L={r['c_L']:.6f}  dP={r['dp']:.5f}", flush=True)

    lines = [f"DFG 2D-1 benchmark (Re=20, steady), config={args.config}, geometry order={args.geometry_order}, "
             f"headline force={args.primary_force}",
             f"Comparison intervals (Schafer-Turek 1996 Table 3, carried over): c_D in {REF_CD}, "
             f"c_L in {REF_CL}, dP in {REF_DP}",
             f"Spectral reference (Nabh 1998 via FeatFlow): c_D={SPECTRAL_2D1['c_D']}, "
             f"c_L={SPECTRAL_2D1['c_L']}, dP={SPECTRAL_2D1['dP']}",
             "",
             f"{'cells':>7}  {'dofs':>7}  {'it':>3}  {'time(s)':>8}  {'c_D':>9}  {'c_L':>9}  {'dP':>9}"
             f"  {'area err':>10}  {'max r-dev/R':>11}"]
    for r in rows:
        g = r["geometry"]
        lines.append(f"{r['n_cells']:7d}  {r['n_dofs']:7d}  {r['newton_it']:3d}  "
                     f"{r['wall_time_s']:8.2f}  {r['c_D']:9.5f}  {r['c_L']:9.6f}  {r['dp']:9.5f}"
                     f"  {g['area_rel_err']:10.2e}  {g['max_radial_dev_rel']:11.2e}")
    lines += ["", "All four force definitions per level (c_D / c_L):",
              f"{'cells':>7}  " + "  ".join(f"{m:>22}" for m in FORCE_METHODS)]
    for r in rows:
        lines.append(f"{r['n_cells']:7d}  " + "  ".join(
            f"{r['forces'][m]['c_D']:11.5f} {r['forces'][m]['c_L']:10.6f}" for m in FORCE_METHODS))

    finest = rows[-1]
    lines += ["", "Finest level vs. comparison interval and vs. spectral reference:"]
    for name, val, ref, spec in [("c_D", finest["c_D"], REF_CD, SPECTRAL_2D1["c_D"]),
                                 ("c_L", finest["c_L"], REF_CL, SPECTRAL_2D1["c_L"]),
                                 ("dP", finest["dp"], REF_DP, SPECTRAL_2D1["dP"])]:
        lines.append(f"  {name}: computed={val:.5f}, interval {ref}: {interval_status(val, ref)}; "
                     f"relative difference to spectral value {100 * (val - spec) / spec:+.3f}%")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"2d1_steady_{label}.txt").write_text("\n".join(lines) + "\n")
    (RESULTS_DIR / f"2d1_steady_{label}.json").write_text(json.dumps(
        {"config": args.config, "geometry_order": args.geometry_order, "primary_force": args.primary_force,
         "levels": rows}, indent=1, default=float))
    print("\n".join(lines[-5:]))
    print(f"\nWrote results/2d1_steady_{label}.txt and .json")


if __name__ == "__main__":
    main()
