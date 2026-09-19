"""Drag, lift and pressure-difference post-processing for the DFG cylinder benchmark.

The benchmark paper (Schäfer & Turek, 1996, Sec. 2.2) defines the drag and
lift forces via

    F_D = int_S (rho*nu*(dv_t/dn)*n_y - P*n_x) dS
    F_L = -int_S (rho*nu*(dv_t/dn)*n_x + P*n_y) dS

with tangential velocity v_t and tangent vector t = (n_y, -n_x). This is a
simplification of the general Cauchy traction that holds specifically on a
no-slip boundary (u = 0 identically on S, so only the derivative in the
normal direction survives in the viscous stress). Rather than differentiate
a boundary-restricted scalar (v_t is only defined on S, so "dv_t/dn" is not
directly expressible as an interior UFL gradient), this module evaluates
the mathematically equivalent general Cauchy stress traction directly, a
standard technique for this exact benchmark (used e.g. in the FEniCS
tutorial's Navier-Stokes cylinder-benchmark example, Langtangen & Logg,
2017, *Solving PDEs in Python*, Springer, Ch. 3):

    sigma(u, p) = -p*I + rho*nu*(grad(u) + grad(u)^T)
    F = -int_S sigma(u, p) . n dS

(the minus sign converts the fluid-domain traction, evaluated with the
domain's outward normal n which points from the fluid INTO the cylinder,
into the force the fluid exerts ON the cylinder). c_D and c_L follow from
F via the benchmark's normalisation:

    c_D = 2*F_x / (rho * Ubar^2 * D),   c_L = 2*F_y / (rho * Ubar^2 * D)

with Ubar the mean inflow velocity and D the cylinder diameter.
"""
from __future__ import annotations

import ufl
from dolfinx import fem


def stress_tensor(u, p, nu, rho: float = 1.0):
    return -p * ufl.Identity(2) + rho * nu * (ufl.grad(u) + ufl.grad(u).T)


def drag_lift_forms(u, p, nu, mesh, facet_tags, cylinder_marker: int, rho: float = 1.0):
    n = ufl.FacetNormal(mesh)
    sigma = stress_tensor(u, p, nu, rho)
    ds_cyl = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags, subdomain_id=cylinder_marker)
    traction = ufl.dot(sigma, n)
    F_x = fem.form(-traction[0] * ds_cyl)
    F_y = fem.form(-traction[1] * ds_cyl)
    return F_x, F_y


def compute_drag_lift(u, p, nu, mesh, facet_tags, cylinder_marker: int, U_bar: float, D: float,
                       rho: float = 1.0):
    F_x_form, F_y_form = drag_lift_forms(u, p, nu, mesh, facet_tags, cylinder_marker, rho)
    F_x = mesh.comm.allreduce(fem.assemble_scalar(F_x_form), op=__import__("mpi4py").MPI.SUM)
    F_y = mesh.comm.allreduce(fem.assemble_scalar(F_y_form), op=__import__("mpi4py").MPI.SUM)
    norm = 0.5 * rho * U_bar**2 * D
    c_D = F_x / norm
    c_L = F_y / norm
    return c_D, c_L


def pressure_difference(p, mesh, point_a=(0.15, 0.2), point_b=(0.25, 0.2)):
    """Delta P = P(point_a) - P(point_b), evaluated by point evaluation of
    the FE pressure field (front and back stagnation points of the
    cylinder, as defined in the benchmark paper)."""
    from dolfinx.geometry import (
        bb_tree, compute_collisions_points, compute_colliding_cells,
    )
    import numpy as np

    tree = bb_tree(mesh, mesh.topology.dim)

    def eval_at(point):
        pt = np.array([[point[0], point[1], 0.0]])
        cell_candidates = compute_collisions_points(tree, pt)
        colliding = compute_colliding_cells(mesh, cell_candidates, pt)
        cells = colliding.links(0)
        if len(cells) == 0:
            return None
        val = p.eval(pt, [cells[0]])
        return float(val[0])

    pa = eval_at(point_a)
    pb = eval_at(point_b)
    if pa is None or pb is None:
        return None
    return pa - pb
