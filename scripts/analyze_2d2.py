"""Re-run the periodic-regime analysis on a saved 2D-2 time series (no FEM solve).

Works on series written by run_2d2_unsteady.py (keys t, c_D_<method>, c_L_<method>, dp) and on the
original repository series (keys t, c_D, c_L, dp; symmetric-stress forces; a leading t = 0 row of
placeholder zeros, which is dropped).

    python scripts/analyze_2d2.py --npz results/2d2_timeseries_full.npz --label full_reanalysed \\
        --dt 0.005 --t-end 8 --cells 12020 --dofs 54860 --geometry-order 1 --primary-force symmetric
    python scripts/analyze_2d2.py --npz results/2d2_timeseries_sp_L5.npz --label sp_L5_b \\
        --meta-json results/2d2_sp_L5.json --n-last 6
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.run_2d2_unsteady import analyse_and_report

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--meta-json", default=None, help="results/2d2_<label>.json of the original run")
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--cells", type=int, default=None)
    ap.add_argument("--dofs", type=int, default=None)
    ap.add_argument("--geometry-order", type=int, default=None)
    ap.add_argument("--primary-force", default="laplacian")
    ap.add_argument("--t-min", type=float, default=None)
    ap.add_argument("--n-last", type=int, default=4)
    a = ap.parse_args()

    d = dict(np.load(a.npz))
    if "c_D" in d:                                   # original series: symmetric-stress forces only
        keep = d["t"] > 0.0                          # drop the t = 0 placeholder row
        d = {"t": d["t"][keep], "dp": d["dp"][keep],
             "c_D_symmetric": d["c_D"][keep], "c_L_symmetric": d["c_L"][keep]}
        if a.primary_force == "laplacian":
            a.primary_force = "symmetric"
    if a.meta_json:
        meta = json.loads(pathlib.Path(a.meta_json).read_text())["meta"]
    else:
        if a.dt is None:
            ap.error("--dt (or --meta-json) is required")
        t = d["t"]
        meta = {"n_cells": a.cells, "n_dofs": a.dofs, "geometry_order": a.geometry_order,
                "dt": a.dt, "theta": 0.5, "t_end": a.t_end or float(t[-1]),
                "init": {"init_state": None, "t_start": 0.0}}
    analyse_and_report(a.label, d, meta, primary=a.primary_force, t_min=a.t_min, n_last=a.n_last)


if __name__ == "__main__":
    main()
