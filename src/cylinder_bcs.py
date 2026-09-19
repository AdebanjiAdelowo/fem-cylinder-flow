"""Boundary conditions for the DFG cylinder benchmark, exactly as specified
in Schäfer & Turek (1996): no-slip on the walls and cylinder, a parabolic
inflow profile, and a "do-nothing" (natural) outflow condition (no explicit
BC object -- see src/navier_stokes.py docstring)."""
from __future__ import annotations

import numpy as np
from dolfinx import fem, mesh as dmesh
from dolfinx.fem import Function, dirichletbc, locate_dofs_topological

from src.geometry import H, INLET, WALLS, CYLINDER


def inflow_profile(U_m: float, t: float | None = None):
    """U(0, y[, t]) = 4*U_m*y*(H-y)/H**2 [* sin(pi*t/8) for the 2D-3 ramp,
    not used by the 2D-1/2D-2 cases implemented in this repository]."""
    def expr(x):
        u = np.zeros((2, x.shape[1]))
        u[0] = 4.0 * U_m * x[1] * (H - x[1]) / H**2
        return u
    return expr


def build_bcs(W, mesh, facet_tags, U_m: float):
    tdim = mesh.topology.dim
    fdim = tdim - 1
    W0, _ = W.sub(0).collapse()

    # no-slip: walls + cylinder
    noslip = Function(W0)
    noslip.x.array[:] = 0.0
    wall_facets = facet_tags.find(WALLS)
    cyl_facets = facet_tags.find(CYLINDER)
    noslip_facets = np.concatenate([wall_facets, cyl_facets])
    dofs_noslip = locate_dofs_topological((W.sub(0), W0), fdim, noslip_facets)
    bc_noslip = dirichletbc(noslip, dofs_noslip, W.sub(0))

    # parabolic inflow
    inflow = Function(W0)
    inflow.interpolate(inflow_profile(U_m))
    inlet_facets = facet_tags.find(INLET)
    dofs_inflow = locate_dofs_topological((W.sub(0), W0), fdim, inlet_facets)
    bc_inflow = dirichletbc(inflow, dofs_inflow, W.sub(0))

    return [bc_noslip, bc_inflow]
