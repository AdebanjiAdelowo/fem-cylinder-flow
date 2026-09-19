"""Taylor-Hood P2/P1 mixed finite-element space for velocity and pressure.

The Taylor-Hood pair (continuous piecewise-quadratic velocity, continuous
piecewise-linear pressure) is used because it is inf-sup (LBB) stable: the
velocity space is rich enough, relative to the pressure space, that the
discrete divergence operator has no spurious near-null modes, which rules
out the checkerboard pressure oscillations that an equal-order P1/P1 pair
would produce. This is the standard stable choice for incompressible Stokes
and Navier-Stokes FEM (see e.g. Girault & Raviart, 1986, *Finite Element
Methods for Navier-Stokes Equations*, or Elman, Silvester & Wathen, 2014,
*Finite Elements and Fast Iterative Solvers*, Ch. 3) and is exactly the
element pair the Schäfer-Turek benchmark paper itself uses in several of
its reference computations (Table 2, entries 8a/8b).
"""
from __future__ import annotations

from basix.ufl import element, mixed_element
from dolfinx.fem import functionspace
from dolfinx import default_real_type


def taylor_hood_space(mesh):
    P2 = element(
        "Lagrange", mesh.basix_cell(), degree=2, shape=(mesh.geometry.dim,),
        dtype=default_real_type,
    )
    P1 = element("Lagrange", mesh.basix_cell(), degree=1, dtype=default_real_type)
    TH = mixed_element([P2, P1])
    return functionspace(mesh, TH)
