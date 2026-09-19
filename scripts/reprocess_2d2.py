"""Regenerate 2D-2 results/figures from an already-saved time series (no
FEM re-solve). Used when post-processing logic (e.g. the Strouhal
frequency-estimation method) is corrected after a long run has already
completed -- re-running the FEM solve (up to ~35 minutes at full
resolution) would be wasteful when the raw force-coefficient time series is
already on disk.

Usage: python scripts/reprocess_2d2.py --config full --cells 12020 --dofs 54860 --dt 0.005 --t_end 8.0 --wall-time 2045.36
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.run_2d2_unsteady import postprocess_and_report

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--dofs", type=int, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--t_end", type=float, required=True)
    parser.add_argument("--wall-time", type=float, required=True)
    args = parser.parse_args()

    d = np.load(ROOT / "results" / f"2d2_timeseries_{args.config}.npz")
    postprocess_and_report(
        args.config, d["t"], d["c_D"], d["c_L"], d["dp"],
        args.cells, args.dofs, args.dt, args.t_end, args.wall_time,
    )


if __name__ == "__main__":
    main()
