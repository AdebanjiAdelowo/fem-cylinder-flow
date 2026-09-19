"""Manufactured-solution verification of the Navier-Stokes FEM implementation.

Two checks on the unit square, Dirichlet velocity = exact solution on the
entire boundary, pressure pinned at one point (since an all-Dirichlet-
velocity Stokes/NS problem determines pressure only up to a constant):

(A) polynomial_solution: exactly representable by the P2/P1 Taylor-Hood
    space. Expect errors at the floating-point roundoff level at ANY mesh
    resolution (the FEM analogue of the Taylor-Green vortex check in the
    companion navier-stokes-2d project).
(B) trigonometric_solution: NOT exactly representable. Mesh refinement
    (N = 4, 8, 16, 32, 64) gives a genuine convergence study; expected rates
    for Taylor-Hood P2/P1: ||u-u_h||_{L2} ~ h^3, ||u-u_h||_{H1} ~ h^2,
    ||p-p_h||_{L2} ~ h^2.

Usage: python scripts/verify_mms.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh as dmesh
from dolfinx.fem import Function, dirichletbc, locate_dofs_topological, locate_dofs_geometrical

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.spaces import taylor_hood_space
from src.navier_stokes import residual, newton_solve
from src.manufactured import polynomial_solution, trigonometric_solution

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

NU = 1.0


def run_case(N: int, solution_fn, nu: float = NU):
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    W = taylor_hood_space(msh)
    x = ufl.SpatialCoordinate(msh)
    u_exact, p_exact, f_exact = solution_fn(x, nu)

    w = Function(W)
    w_test = ufl.TestFunction(W)
    F = residual(w, w_test, nu, f=f_exact)

    # Dirichlet velocity = exact solution on the whole boundary
    W0, _ = W.sub(0).collapse()
    u_bc = Function(W0)
    u_bc_expr = fem.Expression(u_exact, W0.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)

    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = dmesh.exterior_facet_indices(msh.topology)
    bdofs_u = locate_dofs_topological((W.sub(0), W0), tdim - 1, boundary_facets)
    bc_u = dirichletbc(u_bc, bdofs_u, W.sub(0))

    # pin pressure at one point (bottom-left corner) to remove the
    # additive-constant indeterminacy of an all-Dirichlet-velocity problem
    W1, _ = W.sub(1).collapse()
    p_pin = Function(W1)
    p_exact_expr = fem.Expression(p_exact, W1.element.interpolation_points)
    p_pin.interpolate(p_exact_expr)

    def corner(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)

    bdofs_p = locate_dofs_geometrical((W.sub(1), W1), corner)
    bc_p = dirichletbc(p_pin, bdofs_p, W.sub(1))

    newton_solve(F, w, [bc_u, bc_p], msh, prefix=f"mms_{id(solution_fn)}_{N}_")

    u_h, p_h = w.sub(0), w.sub(1)

    e_u_l2 = fem.assemble_scalar(fem.form(ufl.inner(u_h - u_exact, u_h - u_exact) * ufl.dx))
    e_u_h1 = fem.assemble_scalar(
        fem.form(ufl.inner(ufl.grad(u_h - u_exact), ufl.grad(u_h - u_exact)) * ufl.dx)
    )
    e_p_l2 = fem.assemble_scalar(fem.form((p_h - p_exact) ** 2 * ufl.dx))

    h = 1.0 / N
    return h, np.sqrt(e_u_l2), np.sqrt(e_u_h1 + e_u_l2), np.sqrt(e_p_l2)


def main() -> None:
    print("=== Check A: polynomial manufactured solution (exact FE recovery) ===")
    lines = ["Check A: polynomial solution (exact recovery, any N)",
             f"{'N':>4}  {'||u-u_h||_L2':>14}  {'||u-u_h||_H1':>14}  {'||p-p_h||_L2':>14}"]
    for N in [4, 8, 16]:
        h, e_l2, e_h1, e_p = run_case(N, polynomial_solution)
        line = f"{N:4d}  {e_l2:14.4e}  {e_h1:14.4e}  {e_p:14.4e}"
        print(line)
        lines.append(line)

    print("\n=== Check B: trigonometric manufactured solution (mesh convergence) ===")
    Ns = [4, 8, 16, 32, 64]
    hs, e_l2s, e_h1s, e_ps = [], [], [], []
    lines.append("")
    lines.append("Check B: trigonometric solution (mesh convergence)")
    lines.append(f"{'N':>4}  {'h':>8}  {'||u-u_h||_L2':>14}  {'order':>7}  "
                  f"{'||u-u_h||_H1':>14}  {'order':>7}  {'||p-p_h||_L2':>14}  {'order':>7}")
    for N in Ns:
        h, e_l2, e_h1, e_p = run_case(N, trigonometric_solution)
        hs.append(h); e_l2s.append(e_l2); e_h1s.append(e_h1); e_ps.append(e_p)

    for i, N in enumerate(Ns):
        if i == 0:
            o_l2 = o_h1 = o_p = float("nan")
        else:
            o_l2 = np.log(e_l2s[i - 1] / e_l2s[i]) / np.log(hs[i - 1] / hs[i])
            o_h1 = np.log(e_h1s[i - 1] / e_h1s[i]) / np.log(hs[i - 1] / hs[i])
            o_p = np.log(e_ps[i - 1] / e_ps[i]) / np.log(hs[i - 1] / hs[i])
        line = (f"{N:4d}  {hs[i]:8.4f}  {e_l2s[i]:14.4e}  {o_l2:7.3f}  "
                f"{e_h1s[i]:14.4e}  {o_h1:7.3f}  {e_ps[i]:14.4e}  {o_p:7.3f}")
        print(line)
        lines.append(line)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "mms_verification.txt").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.loglog(hs, e_l2s, "o-", label=r"$\|u-u_h\|_{L^2}$ (expect $O(h^3)$)")
    ax.loglog(hs, e_h1s, "s-", label=r"$\|u-u_h\|_{H^1}$ (expect $O(h^2)$)")
    ax.loglog(hs, e_ps, "^-", label=r"$\|p-p_h\|_{L^2}$ (expect $O(h^2)$)")
    ax.loglog(hs, e_l2s[0] * (np.array(hs) / hs[0]) ** 3, "k--", alpha=0.5, label=r"$O(h^3)$ ref.")
    ax.loglog(hs, e_h1s[0] * (np.array(hs) / hs[0]) ** 2, "k:", alpha=0.5, label=r"$O(h^2)$ ref.")
    ax.set_xlabel("mesh size $h$")
    ax.set_ylabel("error")
    ax.set_title("Taylor-Hood P2/P1 mesh convergence (manufactured solution)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "mms_convergence.png", dpi=150)
    print(f"\nWrote {RESULTS_DIR / 'mms_verification.txt'} and {FIG_DIR / 'mms_convergence.png'}")


if __name__ == "__main__":
    main()
