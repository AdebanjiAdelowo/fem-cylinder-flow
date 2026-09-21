"""Weak formulation and Newton solvers for steady and unsteady incompressible
Navier-Stokes on a Taylor-Hood P2/P1 mixed space.

## Strong form

    du/dt - nu*Laplacian(u) + (u.grad)u + grad(p) = f   in Omega
    div(u) = 0                                           in Omega
    u = g                                                 on Gamma_D
    nu*(du/dn) - p*n = 0  ("do-nothing")                   on Gamma_N

(steady problems simply drop the du/dt term). nu is the kinematic viscosity
(the benchmark uses the momentum equation in this kinematic form, dividing
through by the constant density rho; see README).

## Weak form

Multiply by test functions (v, q), integrate over Omega, and integrate the
viscous and pressure-gradient terms by parts:

    -int nu*Laplacian(u).v dx = int nu*grad(u):grad(v) dx
                                  - int_{dOmega} nu*(du/dn).v ds
    int grad(p).v dx = -int p*div(v) dx + int_{dOmega} p*(v.n) ds

On Gamma_D, v = 0 (test functions vanish where the solution is prescribed),
so the boundary terms there drop out. On Gamma_N (the outlet), the two
boundary terms combine to int_{Gamma_N} (p*n - nu*du/dn).v ds; simply NOT
including this term in the discrete residual is exactly equivalent to
imposing the natural "do-nothing" condition nu*du/dn - p*n = 0 there. This
is a standard, well-posed (if imperfect at high Reynolds number where
backflow can occur) open-boundary condition for this benchmark class
(Heywood, Rannacher & Turek, 1996, "Artificial boundaries and flux and
pressure conditions for the incompressible Navier-Stokes equations",
Int. J. Numer. Meth. Fluids 22(5)), and is explicitly permitted by the
Schäfer-Turek benchmark specification ("the outflow condition can be
chosen by the user").

The resulting residual, find (u, p) in the Taylor-Hood space such that for
all (v, q):

    F((u,p);(v,q)) := int nu*grad(u):grad(v) dx - int p*div(v) dx
                       + int q*div(u) dx + int dot(grad(u)*u, v) dx
                       [+ int (u - u_prev)/dt . v dx  for the unsteady case]
                       - int f.v dx = 0

is nonlinear in u (through the convection term dot(grad(u)*u, v)) and is
solved by Newton's method (dolfinx.nls.petsc.NewtonSolver), with the exact
Jacobian obtained by UFL's automatic (symbolic) differentiation of F, and
the resulting linear systems solved directly (MUMPS via PETSc LU) at each
Newton iteration. A direct solver is used throughout this project in
preference to an iterative Krylov method with a fieldsplit preconditioner:
correctness and benchmark agreement are established first (per the project
brief), and the mesh sizes used here are small enough that a direct solve
is not a performance bottleneck; the "Limitations" section of the README
notes iterative solvers as a natural performance follow-up for much larger
3D-scale problems.
"""
from __future__ import annotations

from dataclasses import dataclass

import ufl
from dolfinx.fem import Function, form
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc


def residual(w, w_test, nu, f=None, w_prev=None, dt=None, theta: float = 1.0):
    """Build the nonlinear residual F(w; w_test) for the mixed Taylor-Hood space.

    w, w_test: current (trial) and test mixed Functions, split via ufl.split.
    f: optional UFL body-force expression (for manufactured-solution forcing).
    w_prev: previous-time-step mixed Function, for the unsteady (theta-method) case.
    theta: 1.0 = backward Euler, 0.5 = Crank-Nicolson. Only the momentum
        equation's viscous + convective (spatial) operator is theta-blended
        between time levels; the pressure-gradient/continuity coupling is
        kept fully implicit at the new time level, since pressure is an
        algebraic (Lagrange-multiplier) variable with no time derivative of
        its own -- standard practice for theta-methods applied to
        incompressible Navier-Stokes (e.g. Turek, 1999, *Efficient Solvers
        for Incompressible Flow Problems*, Springer, Ch. 3).
    """
    u, p = ufl.split(w)
    v, q = ufl.split(w_test)

    def viscous_convective(u_):
        r = nu * ufl.inner(ufl.grad(u_), ufl.grad(v)) * ufl.dx
        r += ufl.inner(ufl.dot(ufl.grad(u_), u_), v) * ufl.dx
        return r

    # pressure-gradient / continuity coupling: always fully implicit (new time level)
    F = -p * ufl.div(v) * ufl.dx + q * ufl.div(u) * ufl.dx

    if w_prev is None:
        F += viscous_convective(u)
    else:
        assert dt is not None
        u_prev, _ = ufl.split(w_prev)
        F += ufl.inner((u - u_prev) / dt, v) * ufl.dx
        F += theta * viscous_convective(u) + (1.0 - theta) * viscous_convective(u_prev)

    if f is not None:
        F += -ufl.inner(f, v) * ufl.dx

    return F


@dataclass
class NewtonSettings:
    rtol: float = 1e-10
    atol: float = 1e-10
    max_it: int = 25


def make_newton_problem(F, w, bcs, settings: NewtonSettings = NewtonSettings(), prefix: str = "ns_"):
    """Build (but do not solve) the SNES problem for F(w) = 0.

    Newton line search with the exact Jacobian from UFL's automatic differentiation of F and
    a direct LU (MUMPS) linear solve at every Newton iteration. The forms are compiled once, so a
    time loop can call ``problem.solve()`` repeatedly after updating any Coefficient that F
    depends on (e.g. ``w_prev``) in place -- the residual is then not rebuilt every step.
    """
    J = ufl.derivative(F, w)
    return NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix=prefix,
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": settings.rtol,
            "snes_atol": settings.atol,
            "snes_max_it": settings.max_it,
            "snes_error_if_not_converged": True,
            "ksp_type": "preonly",
            "ksp_error_if_not_converged": True,
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )


def solve_newton_problem(problem) -> int:
    """Solve a problem from ``make_newton_problem``; return the Newton iteration count."""
    problem.solve()
    n_it = problem.solver.getIterationNumber()
    reason = problem.solver.getConvergedReason()
    if reason <= 0:
        raise RuntimeError(f"SNES Newton solve did not converge (reason={reason})")
    return n_it


def newton_solve(F, w, bcs, mesh, settings: NewtonSettings = NewtonSettings(), prefix: str = "ns_"):
    """Solve F(w) = 0 with PETSc SNES (Newton line search + direct LU/MUMPS
    linear solves), using the exact Jacobian from UFL's automatic
    differentiation of F.

    dolfinx.fem.petsc.NonlinearProblem (>= 0.10) is a self-contained SNES
    wrapper: constructing it does not solve the problem, `.solve()` does.
    (The older two-class NonlinearProblem + dolfinx.nls.petsc.NewtonSolver
    pattern from earlier dolfinx versions was replaced by this single class.)
    """
    return solve_newton_problem(make_newton_problem(F, w, bcs, settings, prefix))
