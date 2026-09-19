"""Field visualisation for the steady 2D-1 solution: velocity magnitude,
pressure, vorticity, and streamlines.

Usage: python scripts/plot_fields_2d1.py --config full
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import ufl
import yaml
from dolfinx.fem import Function, functionspace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.geometry import build_mesh
from src.spaces import taylor_hood_space
from src.cylinder_bcs import build_bcs
from src.navier_stokes import residual, newton_solve

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"

NU = 1.0e-3
U_M = 0.3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="full", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())
    lvl = cfg["2d1_mesh_levels"][-1]  # finest level

    md = build_mesh(lvl["h_far"], lvl["h_cyl"])
    mesh = md.mesh
    W = taylor_hood_space(mesh)
    bcs = build_bcs(W, mesh, md.facet_tags, U_M)
    w = Function(W)
    w_test = ufl.TestFunction(W)
    F = residual(w, w_test, NU)
    newton_solve(F, w, bcs, mesh, prefix="plotfields_")
    u_h, p_h = w.sub(0).collapse(), w.sub(1).collapse()

    # interpolate onto P1 for plotting on the mesh's own (P1) vertices
    V1 = functionspace(mesh, ("Lagrange", 1))
    V1v = functionspace(mesh, ("Lagrange", 1, (2,)))

    u_p1 = Function(V1v)
    u_p1.interpolate(u_h)

    from dolfinx.fem import Expression

    speed = Function(V1)
    speed_expr = ufl.sqrt(ufl.inner(u_h, u_h))
    speed.interpolate(Expression(speed_expr, V1.element.interpolation_points))

    vort = Function(V1)
    vort_expr = u_h[1].dx(0) - u_h[0].dx(1)
    vort.interpolate(Expression(vort_expr, V1.element.interpolation_points))

    p_p1 = Function(V1)
    p_p1.interpolate(p_h)

    x = mesh.geometry.x
    cells = mesh.geometry.dofmap.reshape(-1, 3)
    triang = mtri.Triangulation(x[:, 0], x[:, 1], cells)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    im0 = axes[0].tricontourf(triang, speed.x.array, levels=40, cmap="viridis")
    axes[0].set_title(r"velocity magnitude $|u|$ [m/s]")
    fig.colorbar(im0, ax=axes[0])
    u_vals = u_p1.x.array.reshape(-1, 2)
    axes[0].streamplot(
        np.linspace(x[:, 0].min(), x[:, 0].max(), 200),
        np.linspace(x[:, 1].min(), x[:, 1].max(), 60),
        griddata_u(x, u_vals[:, 0], 200, 60),
        griddata_u(x, u_vals[:, 1], 200, 60),
        color="white", linewidth=0.5, density=1.2,
    )

    im1 = axes[1].tricontourf(triang, p_p1.x.array, levels=40, cmap="coolwarm")
    axes[1].set_title("pressure $p$ [Pa]")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].tricontourf(triang, vort.x.array, levels=40, cmap="RdBu_r")
    axes[2].set_title(r"vorticity $\omega = \partial v/\partial x - \partial u/\partial y$ [1/s]")
    fig.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_ylabel("y [m]")
    axes[-1].set_xlabel("x [m]")

    fig.suptitle(f"DFG 2D-1 steady solution (Re=20), {mesh.topology.index_map(2).size_local} cells")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"2d1_fields_{args.config}.png", dpi=150)
    print(f"Wrote {FIG_DIR / f'2d1_fields_{args.config}.png'}")


def griddata_u(x, vals, nx, ny):
    from scipy.interpolate import griddata
    xi = np.linspace(x[:, 0].min(), x[:, 0].max(), nx)
    yi = np.linspace(x[:, 1].min(), x[:, 1].max(), ny)
    Xi, Yi = np.meshgrid(xi, yi)
    return griddata(x[:, :2], vals, (Xi, Yi), method="linear")


if __name__ == "__main__":
    main()
