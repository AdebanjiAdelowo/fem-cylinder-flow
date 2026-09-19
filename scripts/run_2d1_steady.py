"""DFG 2D-1 benchmark: steady flow around a cylinder at Re = 20.

Reference values (Schäfer & Turek, 1996, Table 3, "lower/upper bound" row):
    c_D in [5.5700, 5.5900]
    c_L in [0.0104, 0.0110]
    Delta P in [0.1172, 0.1176]

Usage: python scripts/run_2d1_steady.py --config smoke|local|full
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

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
U_M = 0.3
U_BAR = 2.0 / 3.0 * U_M  # mean inflow velocity, Re = U_BAR*D/nu = 20

REF_CD = (5.5700, 5.5900)
REF_CL = (0.0104, 0.0110)
REF_DP = (0.1172, 0.1176)


def run_one(h_far: float, h_cyl: float):
    md = build_mesh(h_far, h_cyl)
    mesh = md.mesh
    W = taylor_hood_space(mesh)
    bcs = build_bcs(W, mesh, md.facet_tags, U_M)

    w = Function(W)
    w_test = ufl.TestFunction(W)
    F = residual(w, w_test, NU)

    t0 = time.perf_counter()
    n_it = newton_solve(F, w, bcs, mesh, prefix=f"steady2d1_{h_far}_{h_cyl}_")
    elapsed = time.perf_counter() - t0

    u_h, p_h = w.sub(0), w.sub(1)
    c_D, c_L = compute_drag_lift(u_h, p_h, NU, mesh, md.facet_tags, CYLINDER, U_BAR, D, RHO)
    dp = pressure_difference(p_h, mesh)

    n_cells = mesh.topology.index_map(2).size_local
    n_dofs = W.dofmap.index_map.size_global * W.dofmap.index_map_bs
    return {
        "h_far": h_far, "h_cyl": h_cyl, "n_cells": n_cells, "n_dofs": n_dofs,
        "newton_it": n_it, "wall_time_s": elapsed, "c_D": c_D, "c_L": c_L, "dp": dp,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())
    levels = cfg["2d1_mesh_levels"]

    rows = []
    for lvl in levels:
        r = run_one(lvl["h_far"], lvl["h_cyl"])
        rows.append(r)
        print(f"h_far={r['h_far']:.4f} h_cyl={r['h_cyl']:.4f}  cells={r['n_cells']:6d}  "
              f"dofs={r['n_dofs']:7d}  Newton_it={r['newton_it']}  time={r['wall_time_s']:.2f}s  "
              f"c_D={r['c_D']:.4f}  c_L={r['c_L']:.5f}  dP={r['dp']:.4f}")

    lines = ["DFG 2D-1 benchmark (Re=20, steady), config=" + args.config,
             f"Reference (Schafer-Turek 1996, Table 3): c_D in {REF_CD}, c_L in {REF_CL}, "
             f"dP in {REF_DP}",
             "",
             f"{'cells':>7}  {'dofs':>7}  {'it':>3}  {'time(s)':>8}  {'c_D':>9}  {'c_L':>9}  "
             f"{'dP':>9}"]
    for r in rows:
        lines.append(f"{r['n_cells']:7d}  {r['n_dofs']:7d}  {r['newton_it']:3d}  "
                      f"{r['wall_time_s']:8.2f}  {r['c_D']:9.5f}  {r['c_L']:9.6f}  {r['dp']:9.5f}")

    finest = rows[-1]
    lines.append("")
    lines.append(f"Finest-mesh discrepancy from reference INTERVAL (0 if inside):")
    for name, val, ref in [("c_D", finest["c_D"], REF_CD), ("c_L", finest["c_L"], REF_CL),
                            ("dP", finest["dp"], REF_DP)]:
        if ref[0] <= val <= ref[1]:
            disc = 0.0
        else:
            nearest = ref[0] if val < ref[0] else ref[1]
            disc = (val - nearest) / nearest * 100
        lines.append(f"  {name}: computed={val:.5f}, reference interval={ref}, "
                      f"{'INSIDE interval' if disc == 0.0 else f'{disc:+.3f}% outside nearest bound'}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"2d1_steady_{args.config}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))
    print(f"\nWrote results/2d1_steady_{args.config}.txt")


if __name__ == "__main__":
    main()
