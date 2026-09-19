"""Fast (small-N) regression tests mirroring scripts/verify_mms.py's checks.

Full mesh-convergence verification (order estimation across several
resolutions) is run separately via `python scripts/verify_mms.py`, since it
is too slow for the default test-suite budget; these tests check the two
qualitative properties that must hold at any single resolution.
"""
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh
from dolfinx.fem import Function, dirichletbc, locate_dofs_topological, locate_dofs_geometrical

from src.spaces import taylor_hood_space
from src.navier_stokes import residual, newton_solve
from src.manufactured import polynomial_solution, trigonometric_solution


def _solve_mms(N, solution_fn, nu=1.0):
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    W = taylor_hood_space(msh)
    x = ufl.SpatialCoordinate(msh)
    u_exact, p_exact, f_exact = solution_fn(x, nu)

    w = Function(W)
    w_test = ufl.TestFunction(W)
    F = residual(w, w_test, nu, f=f_exact)

    W0, _ = W.sub(0).collapse()
    u_bc = Function(W0)
    u_bc.interpolate(fem.Expression(u_exact, W0.element.interpolation_points))
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = dmesh.exterior_facet_indices(msh.topology)
    bdofs_u = locate_dofs_topological((W.sub(0), W0), tdim - 1, boundary_facets)
    bc_u = dirichletbc(u_bc, bdofs_u, W.sub(0))

    W1, _ = W.sub(1).collapse()
    p_pin = Function(W1)
    p_pin.interpolate(fem.Expression(p_exact, W1.element.interpolation_points))

    def corner(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)

    bdofs_p = locate_dofs_geometrical((W.sub(1), W1), corner)
    bc_p = dirichletbc(p_pin, bdofs_p, W.sub(1))

    newton_solve(F, w, [bc_u, bc_p], msh, prefix=f"test_mms_{id(solution_fn)}_{N}_")
    u_h, p_h = w.sub(0), w.sub(1)
    e_u = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(u_h - u_exact, u_h - u_exact) * ufl.dx)))
    e_p = np.sqrt(fem.assemble_scalar(fem.form((p_h - p_exact) ** 2 * ufl.dx)))
    return e_u, e_p


def test_polynomial_solution_is_recovered_to_machine_precision():
    """phi degree 3 => u = curl(phi) degree <= 2, exactly representable by
    P2; p is degree 1, exactly representable by P1: the discrete solution
    should match to floating-point roundoff at any mesh resolution."""
    e_u, e_p = _solve_mms(N=6, solution_fn=polynomial_solution)
    assert e_u < 1e-10
    assert e_p < 1e-10


def test_trigonometric_solution_error_decreases_under_refinement():
    e_u_coarse, _ = _solve_mms(N=4, solution_fn=trigonometric_solution)
    e_u_fine, _ = _solve_mms(N=8, solution_fn=trigonometric_solution)
    assert e_u_fine < e_u_coarse / 4  # expect ~O(h^3): at least a 4x drop when h halves
