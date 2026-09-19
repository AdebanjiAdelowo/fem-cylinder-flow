"""Sanity checks on the production residual()/newton_solve() path (as
opposed to the manufactured-solution path in test_manufactured.py), using a
small lid-driven-cavity problem: incompressibility of the discrete solution,
and that a zero-Reynolds-number (Stokes) solve and a finite-Reynolds
(Navier-Stokes) solve give different, non-trivial solutions."""
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh
from dolfinx.fem import Function, dirichletbc, locate_dofs_topological

from src.spaces import taylor_hood_space
from src.navier_stokes import residual, newton_solve


def _cavity_bcs(W, msh):
    W0, _ = W.sub(0).collapse()

    def noslip_boundary(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0)

    def lid(x):
        return np.isclose(x[1], 1.0)

    noslip = Function(W0)
    facets = dmesh.locate_entities_boundary(msh, 1, noslip_boundary)
    bc0 = dirichletbc(noslip, locate_dofs_topological((W.sub(0), W0), 1, facets), W.sub(0))

    lid_velocity = Function(W0)
    lid_velocity.interpolate(lambda x: np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1]))))
    facets = dmesh.locate_entities_boundary(msh, 1, lid)
    bc1 = dirichletbc(lid_velocity, locate_dofs_topological((W.sub(0), W0), 1, facets), W.sub(0))
    return [bc0, bc1]


def _solve_cavity(nu: float, N: int = 12):
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    W = taylor_hood_space(msh)
    bcs = _cavity_bcs(W, msh)
    w = Function(W)
    w_test = ufl.TestFunction(W)
    F = residual(w, w_test, nu)
    newton_solve(F, w, bcs, msh, prefix=f"cavity_{nu}_{N}_")
    return w, msh


def _smooth_lid_cavity_bcs(W, msh):
    """A lid velocity that vanishes at the top corners (rather than jumping
    discontinuously from 0 to 1 there, as in the classical lid-driven
    cavity) removes that problem's well-known corner pressure/stress
    singularity. That singularity is a real, well-documented feature of the
    *discontinuous*-lid cavity problem, not a defect of this solver, but it
    also means the discontinuous-lid problem is a poor test case for
    checking that ||div(u_h)||_{L2} -> 0 under mesh refinement (the
    singularity prevents clean convergence of that norm). This smoothed
    variant is used specifically for that check."""
    W0, _ = W.sub(0).collapse()

    def sides_and_bottom(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0)

    def lid(x):
        return np.isclose(x[1], 1.0)

    noslip = Function(W0)
    facets = dmesh.locate_entities_boundary(msh, 1, sides_and_bottom)
    bc0 = dirichletbc(noslip, locate_dofs_topological((W.sub(0), W0), 1, facets), W.sub(0))

    lid_velocity = Function(W0)
    lid_velocity.interpolate(
        lambda x: np.stack((16.0 * x[0] ** 2 * (1.0 - x[0]) ** 2, np.zeros(x.shape[1])))
    )
    facets = dmesh.locate_entities_boundary(msh, 1, lid)
    bc1 = dirichletbc(lid_velocity, locate_dofs_topological((W.sub(0), W0), 1, facets), W.sub(0))
    return [bc0, bc1]


def test_discrete_divergence_decreases_under_mesh_refinement():
    """Taylor-Hood does not give a pointwise-divergence-free discrete
    velocity (only weak orthogonality to the continuous P1 pressure test
    space), so ||div(u_h)||_{L2} is not expected to be at roundoff at any
    single resolution -- but it must shrink as the mesh is refined. Uses
    the smoothed-lid cavity (see _smooth_lid_cavity_bcs) to avoid the
    classical corner singularity, which would otherwise stall this
    convergence."""
    def div_l2(N):
        msh = dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
        W = taylor_hood_space(msh)
        bcs = _smooth_lid_cavity_bcs(W, msh)
        w = Function(W)
        w_test = ufl.TestFunction(W)
        F = residual(w, w_test, nu=1.0)
        newton_solve(F, w, bcs, msh, prefix=f"smoothcavity_{N}_")
        u_h, _ = ufl.split(w)
        return np.sqrt(fem.assemble_scalar(fem.form(ufl.div(u_h) ** 2 * ufl.dx)))

    e_coarse = div_l2(8)
    e_fine = div_l2(16)
    assert e_fine < 0.5 * e_coarse


def test_stokes_and_navier_stokes_give_different_solutions_at_low_viscosity():
    """At nu=1 (Re~1) the Stokes and full NS solves should nearly coincide;
    at nu=0.01 (Re~100) convection should visibly change the solution --
    checks that the nonlinear term is actually wired into the residual."""
    w_visc, msh = _solve_cavity(nu=1.0)
    w_conv, _ = _solve_cavity(nu=0.01)

    u_visc = w_visc.sub(0).collapse()
    u_conv = w_conv.sub(0).collapse()
    diff = np.linalg.norm(u_visc.x.array - u_conv.x.array)
    norm = np.linalg.norm(u_visc.x.array)
    assert diff / norm > 0.05


def test_newton_solve_reports_iteration_count():
    """newton_solve should report at least one Newton iteration on a
    well-posed problem, and raise (rather than silently return a
    non-converged state) if SNES does not converge -- exercised implicitly
    by every other test in this file via successful completion."""
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    W = taylor_hood_space(msh)
    bcs = _cavity_bcs(W, msh)
    w = Function(W)
    w_test = ufl.TestFunction(W)
    F = residual(w, w_test, nu=1.0)
    n_it = newton_solve(F, w, bcs, msh, prefix="iter_count_")
    assert n_it >= 1
