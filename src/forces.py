"""Drag, lift and pressure-difference post-processing for the DFG cylinder benchmark.

Definition (Schaefer & Turek 1996; FeatFlow "DFG benchmark 2D-1/2D-2" pages)
---------------------------------------------------------------------------
    (F_D, F_L) = int_S sigma * eta ds,    sigma = nu*grad(u) - p*I,     eta = outer normal of the circle
    c_D = 2 F_D / (Ubar^2 D),  c_L = 2 F_L / (Ubar^2 D),  Ubar = (2/3) U_max,  D = 0.1,  rho = 1

The circle's outer normal eta points from the cylinder into the fluid, i.e. eta = -n with n the
fluid-domain outward normal (ufl.FacetNormal). Four evaluations of the force are provided so that
the post-processing choice can be separated from the discretisation error:

``laplacian``    F = -int (nu*grad(u) - p I) n ds. This is the benchmark's own definition and the
                 stress that belongs to the weak form solved here (nu*grad(u):grad(v)).
``symmetric``    F = -int (nu*(grad(u) + grad(u)^T) - p I) n ds, the general Cauchy traction. It is
                 identical to ``laplacian`` when div(u) = 0 on the cylinder, which the Taylor-Hood
                 velocity satisfies only in the limit h -> 0. This was the repository's original
                 choice; it is kept for comparison and for reproducing the earlier results.
``tangential``   the benchmark paper's tangential-derivative form,
                 F_D = int (nu d_eta(u_t) eta_y - p eta_x) ds, F_L = -int (nu d_eta(u_t) eta_x + p eta_y) ds,
                 t = (eta_y, -eta_x). Also equal to ``laplacian`` for div(u) = 0 on a no-slip wall.
``variational``  the discrete reaction force: the (steady or Crank-Nicolson) weak-form residual tested
                 with a velocity function equal to e_x (resp. e_y) on the cylinder nodes and zero
                 elsewhere, F = -R(phi). It involves no facet normals and no boundary quadrature,
                 so it is independent of how the normal is represented, and for a converged
                 solution it is the exact force of the discrete system. For the unsteady scheme it
                 is centred at t_{n+1/2} (the residual blends the two time levels), the traction
                 forms at t_{n+1}.
"""
from __future__ import annotations

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, geometry
from dolfinx.fem import Function, locate_dofs_topological

from src.navier_stokes import residual

FORCE_METHODS = ("laplacian", "symmetric", "tangential", "variational")


def stress_tensor(u, p, nu, rho: float = 1.0):
    """General Cauchy stress -p I + rho nu (grad u + grad u^T)  (the ``symmetric`` convention)."""
    return -p * ufl.Identity(2) + rho * nu * (ufl.grad(u) + ufl.grad(u).T)


def traction_force_expressions(u, p, nu, n, method: str, rho: float = 1.0):
    """UFL expressions (F_x, F_y) whose integral over the cylinder is the force ON the cylinder.

    n is the fluid-outward facet normal (pointing into the cylinder); eta = -n is the circle's
    outer normal used in the benchmark's definition.
    """
    I = ufl.Identity(2)
    if method == "symmetric":
        t = ufl.dot(stress_tensor(u, p, nu, rho), n)
        return -t[0], -t[1]
    if method == "laplacian":
        t = ufl.dot(-p * I + rho * nu * ufl.grad(u), n)
        return -t[0], -t[1]
    if method == "tangential":
        eta = -n
        tau = ufl.as_vector((eta[1], -eta[0]))
        dn_ut = ufl.dot(ufl.dot(ufl.grad(u), eta), tau)
        return rho * nu * dn_ut * eta[1] - p * eta[0], -(rho * nu * dn_ut * eta[0] + p * eta[1])
    raise ValueError(f"unknown traction method {method!r}")


class ForceCoefficients:
    """Drag/lift coefficients of a mixed Taylor-Hood Function ``w = (u, p)``, with all forms
    compiled once and evaluated repeatedly (``w`` is read in place, so updating ``w.x.array``
    between calls is enough).

    For the ``variational`` method pass the same ``w_prev``, ``dt`` and ``theta`` as the time
    stepper, so the residual being tested is the one the scheme actually solved.
    """

    def __init__(self, w: Function, facet_tags, cylinder_marker: int, nu: float, U_bar: float, D: float,
                 rho: float = 1.0, methods=FORCE_METHODS, w_prev: Function | None = None,
                 dt: float | None = None, theta: float = 1.0):
        self.w = w
        W = w.function_space
        mesh = W.mesh
        self.mesh = mesh
        self.norm = 0.5 * rho * U_bar ** 2 * D
        u, p = ufl.split(w)
        n = ufl.FacetNormal(mesh)
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags, subdomain_id=cylinder_marker)
        self._forms: dict[str, tuple] = {}
        for m in methods:
            if m == "variational":
                continue
            fx, fy = traction_force_expressions(u, p, nu, n, m, rho)
            self._forms[m] = (fem.form(fx * ds), fem.form(fy * ds))
        if "variational" in methods:
            self._forms["variational"] = self._variational_forms(
                W, mesh, facet_tags, cylinder_marker, nu, rho, w_prev, dt, theta)

    def _variational_forms(self, W, mesh, facet_tags, marker, nu, rho, w_prev, dt, theta):
        V0, _ = W.sub(0).collapse()
        fdim = mesh.topology.dim - 1
        dofs = locate_dofs_topological(V0, fdim, facet_tags.find(marker))
        forms = []
        self._phi = []                      # keep the test functions alive with the forms
        for comp in (0, 1):
            phi_v = Function(V0)
            phi_v.x.array[:] = 0.0
            phi_v.x.array[2 * dofs + comp] = 1.0
            phi_v.x.scatter_forward()
            phi = Function(W)
            phi.sub(0).interpolate(phi_v)
            phi.x.scatter_forward()
            self._phi.append(phi)
            # residual() is built for nu in kinematic form (rho = 1); rho multiplies the whole
            # momentum residual, so rescale afterwards.
            R = residual(self.w, phi, nu, w_prev=w_prev, dt=dt, theta=theta)
            forms.append(fem.form(-rho * R))
        return tuple(forms)

    def raw_forces(self, method: str):
        fx_form, fy_form = self._forms[method]
        comm = self.mesh.comm
        return (comm.allreduce(fem.assemble_scalar(fx_form), op=MPI.SUM),
                comm.allreduce(fem.assemble_scalar(fy_form), op=MPI.SUM))

    def coefficients(self, method: str = "laplacian"):
        Fx, Fy = self.raw_forces(method)
        return Fx / self.norm, Fy / self.norm

    def all_coefficients(self) -> dict[str, tuple[float, float]]:
        return {m: self.coefficients(m) for m in self._forms}


def compute_drag_lift(u, p, nu, mesh, facet_tags, cylinder_marker: int, U_bar: float, D: float,
                      rho: float = 1.0, method: str = "symmetric"):
    """One-shot traction-based (c_D, c_L). Compiles the forms on every call, so use
    ``ForceCoefficients`` inside a time loop. The default ``symmetric`` reproduces the original
    behaviour of this repository."""
    n = ufl.FacetNormal(mesh)
    fx, fy = traction_force_expressions(u, p, nu, n, method, rho)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags, subdomain_id=cylinder_marker)
    Fx = mesh.comm.allreduce(fem.assemble_scalar(fem.form(fx * ds)), op=MPI.SUM)
    Fy = mesh.comm.allreduce(fem.assemble_scalar(fem.form(fy * ds)), op=MPI.SUM)
    norm = 0.5 * rho * U_bar ** 2 * D
    return Fx / norm, Fy / norm


class PointProbe:
    """Evaluate a scalar FE function at fixed points, with the cells located once.

    The benchmark's pressure points (0.15, 0.2) and (0.25, 0.2) lie exactly ON the cylinder
    surface. On a curved mesh a point on the true circle can fall a few 1e-9 outside the meshed
    fluid domain, so a point that no cell contains is assigned to the nearest cell (GJK distance
    among bounding-box candidates, or the closest cell if there are none) and the finite-element
    function is evaluated there by extrapolation across that negligible distance. ``distance``
    records how far outside the mesh each point was, so a large value is visible, not silent. Works in parallel: each point is evaluated by the ranks that own a cell
    for it and the results are averaged.
    """

    def __init__(self, mesh, points):
        pts = np.zeros((len(points), 3))
        pts[:, :2] = np.asarray(points, dtype=float)
        self.mesh = mesh
        self.points = pts
        tree = geometry.bb_tree(mesh, mesh.topology.dim)
        cand = geometry.compute_collisions_points(tree, pts)
        coll = geometry.compute_colliding_cells(mesh, cand, pts)
        self.cells = np.full(len(pts), -1, dtype=np.int32)
        self.distance = np.full(len(pts), np.inf)      # 0 if inside a cell, else GJK distance
        x = mesh.geometry.x
        dofmap = mesh.geometry.dofmap
        for i in range(len(pts)):
            inside = coll.links(i)
            if len(inside):
                self.cells[i], self.distance[i] = inside[0], 0.0
                continue
            candidates = list(cand.links(i))
            if not candidates:      # no bounding box contains the point: search by distance instead
                ncell = mesh.topology.index_map(mesh.topology.dim).size_local
                mid_tree = geometry.create_midpoint_tree(
                    mesh, mesh.topology.dim, np.arange(ncell, dtype=np.int32))
                candidates = list(geometry.compute_closest_entity(tree, mid_tree, mesh, pts[i:i + 1]))
            best, best_d = -1, np.inf
            for c in candidates:
                d = float(np.linalg.norm(geometry.compute_distance_gjk(x[dofmap[c]], pts[i])))
                if d < best_d:
                    best, best_d = c, d
            self.cells[i], self.distance[i] = best, best_d

    def evaluate(self, f) -> np.ndarray:
        vals = np.zeros(len(self.points))
        found = np.zeros(len(self.points))
        for i, c in enumerate(self.cells):
            if c >= 0:
                vals[i] = float(f.eval(self.points[i:i + 1], np.array([c], dtype=np.int32))[0])
                found[i] = 1.0
        comm = self.mesh.comm
        vals, found = comm.allreduce(vals, op=MPI.SUM), comm.allreduce(found, op=MPI.SUM)
        if np.any(found == 0):
            raise RuntimeError("probe point not located in any cell")
        return vals / found


BENCHMARK_PRESSURE_POINTS = ((0.15, 0.2), (0.25, 0.2))


def pressure_difference(p, mesh, point_a=BENCHMARK_PRESSURE_POINTS[0], point_b=BENCHMARK_PRESSURE_POINTS[1],
                        probe: PointProbe | None = None):
    """Delta P = P(point_a) - P(point_b) at the benchmark's front/rear probe points. Pass a
    prebuilt ``PointProbe`` inside a time loop to avoid re-locating the cells each step."""
    probe = probe or PointProbe(mesh, [point_a, point_b])
    pa, pb = probe.evaluate(p)
    return float(pa - pb)
