"""Plot c_D, c_L, Delta P vs. mesh resolution for the 2D-1 mesh-convergence
ladder (run scripts/run_2d1_steady.py --config full first).

Usage: python scripts/plot_2d1_convergence.py --config full
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.run_2d1_steady import run_one, REF_CD, REF_CL, REF_DP
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="full", choices=["smoke", "local", "full"])
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())
    levels = cfg["2d1_mesh_levels"]

    rows = [run_one(lvl["h_far"], lvl["h_cyl"]) for lvl in levels]
    dofs = [r["n_dofs"] for r in rows]
    cDs = [r["c_D"] for r in rows]
    cLs = [r["c_L"] for r in rows]
    dps = [r["dp"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, vals, ref, name in zip(
        axes, [cDs, cLs, dps], [REF_CD, REF_CL, REF_DP], ["$c_D$", "$c_L$", r"$\Delta P$"]
    ):
        ax.semilogx(dofs, vals, "o-", color="#1f77b4")
        ax.axhspan(ref[0], ref[1], color="green", alpha=0.15, label="reference interval")
        ax.set_xlabel("degrees of freedom")
        ax.set_ylabel(name)
        ax.set_title(f"{name} vs. mesh resolution")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("DFG 2D-1 (Re=20): mesh convergence toward the reference interval")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"2d1_convergence_{args.config}.png", dpi=150)
    print(f"Wrote {FIG_DIR / f'2d1_convergence_{args.config}.png'}")


if __name__ == "__main__":
    main()
