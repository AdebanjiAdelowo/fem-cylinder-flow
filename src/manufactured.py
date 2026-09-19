"""Manufactured solutions for verifying the Stokes/Navier-Stokes FEM implementation.

Unlike src/manufactured-solution work in a spectral setting (sympy), here
the exact solution and its required forcing are built directly from UFL
expressions and differentiated symbolically by UFL itself (the same
machinery FFCX uses to form the exact Newton Jacobian), so there is no
external CAS dependency and no hand-transcribed algebra.

The exact velocity is built from a scalar streamfunction, u = curl(phi) =
(d(phi)/dy, -d(phi)/dx), which is divergence-free identically (to the level
of UFL's symbolic differentiation, i.e. exactly, not approximately) by
construction -- the same device used for the vorticity-streamfunction
Navier-Stokes solver in the companion `navier-stokes-2d` project, applied
here in a finite-element rather than spectral setting.

Two manufactured solutions are provided:

- `polynomial_solution`: phi and p chosen so that phi is degree <= 3
  (velocity is then degree <= 2 exactly, i.e. exactly representable by the
  P2 velocity space, and pressure is degree 1, exactly representable by the
  P1 pressure space). Used as an exact-recovery check (analogous to the
  Taylor-Green vortex in the companion spectral project): the discrete
  solution should match this to machine precision at ANY mesh resolution,
  because the finite-element space can represent it exactly.
- `trigonometric_solution`: phi and p are smooth but NOT polynomial (sine/
  cosine), so neither space can represent them exactly. Used for a genuine
  mesh-refinement convergence study; the expected rates for a Taylor-Hood
  P2/P1 pair are ||u-u_h||_{L2} = O(h^3), ||u-u_h||_{H1} = O(h^2),
  ||p-p_h||_{L2} = O(h^2) (Girault & Raviart, 1986, Thm. II.1.1; Elman,
  Silvester & Wathen, 2014, Ch. 3).
"""
from __future__ import annotations

import ufl


def _from_streamfunction(phi, p, x, nu):
    u = ufl.as_vector((phi.dx(1), -phi.dx(0)))
    convective = ufl.dot(ufl.grad(u), u)
    viscous = ufl.div(ufl.grad(u))
    f = -nu * viscous + convective + ufl.grad(p)
    return u, p, f


def polynomial_solution(x, nu):
    # A general bivariate polynomial of total degree <= 3 differentiates
    # (d/dx or d/dy) to total degree <= 2 in each component, so
    # u = curl(phi) is then exactly representable by P2. (x[0]**2*x[1]*(1
    # - x[1]) would NOT work here: expanded, it contains an x[0]**2*x[1]**2
    # term of total degree 4.)
    phi = x[0] ** 2 * x[1] + x[0] * x[1] ** 2
    p = x[0] + x[1] - 1.0
    return _from_streamfunction(phi, p, x, nu)


def trigonometric_solution(x, nu):
    phi = ufl.sin(ufl.pi * x[0]) ** 2 * ufl.sin(ufl.pi * x[1]) ** 2
    p = ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    return _from_streamfunction(phi, p, x, nu)
