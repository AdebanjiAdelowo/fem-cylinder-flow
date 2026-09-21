"""Sanity checks on the drag/lift/pressure-difference post-processing."""
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh
from dolfinx.fem import Function

from src.forces import stress_tensor, pressure_difference


def test_stress_tensor_reduces_to_minus_pressure_identity_for_zero_velocity():
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    Q = fem.functionspace(msh, ("Lagrange", 1))
    u = Function(V)
    u.x.array[:] = 0.0
    p = Function(Q)
    p.x.array[:] = 2.0

    sigma = stress_tensor(u, p, nu=1.0)
    sigma_00 = fem.assemble_scalar(fem.form(sigma[0, 0] * ufl.dx))
    area = fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * ufl.dx))
    assert np.isclose(sigma_00 / area, -2.0, atol=1e-10)


def test_pressure_difference_matches_known_linear_field():
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    Q = fem.functionspace(msh, ("Lagrange", 1))
    p = Function(Q)
    p.interpolate(lambda x: 3.0 * x[0] + x[1])  # p(x,y) = 3x + y

    dp = pressure_difference(p, msh, point_a=(0.2, 0.5), point_b=(0.8, 0.5))
    expected = (3.0 * 0.2 + 0.5) - (3.0 * 0.8 + 0.5)
    assert dp is not None
    assert np.isclose(dp, expected, atol=1e-6)


# ---------------------------------------------------------------------------------------------
# Force integration on the actual cylinder mesh, against exactly known answers
# ---------------------------------------------------------------------------------------------
import pytest
import ufl
from dolfinx.fem import Function

from src.geometry import build_mesh, cylinder_geometry_report, CYLINDER, R, CX, CY
from src.spaces import taylor_hood_space
from src.forces import ForceCoefficients, PointProbe, BENCHMARK_PRESSURE_POINTS, FORCE_METHODS

NU, D = 1.0e-3, 0.1
TRACTION_METHODS = ("laplacian", "symmetric")


def _state(order, velocity=None, pressure=None):
    md = build_mesh(h_far=0.1, h_cyl=0.02, geometry_order=order)
    W = taylor_hood_space(md.mesh)
    w = Function(W)
    if velocity is not None:
        w.sub(0).interpolate(velocity)
    if pressure is not None:
        w.sub(1).interpolate(pressure)
    w.x.scatter_forward()
    return md, w


def _polygon_area(md):
    return np.pi * R ** 2 * (1.0 + cylinder_geometry_report(md.mesh, md.facet_tags)["area_rel_err"])


@pytest.mark.parametrize("method", FORCE_METHODS[:3])   # traction-type definitions (u = 0 => pressure only)
@pytest.mark.parametrize("comp,pressure,sign", [
    (0, lambda x: -x[0], +1),    # p = -g x, g = 1: high pressure upstream pushes the body in +x
    (0, lambda x: +x[0], -1),    # reversed gradient reverses the force
    (1, lambda x: -x[1], +1),    # p = -g y: higher pressure below pushes the body up (positive lift)
])
def test_pressure_gradient_force_is_g_times_enclosed_area_with_correct_sign(method, comp, pressure, sign):
    """u = 0 and p linear: the force on a body is -int p n_out ds = -grad(p) * A (buoyancy).

    On the straight-sided mesh a physical-linear p is exactly representable, so the discrete
    integral equals g times the area of the meshed polygon to rounding error. That fixes the sign
    convention, the normal direction and the 2/(U^2 D) normalisation (checked for two U_bar)."""
    md, w = _state(1, pressure=pressure)
    A = _polygon_area(md)
    for U_bar in (1.0, 0.2):
        fc = ForceCoefficients(w, md.facet_tags, CYLINDER, NU, U_bar, D, methods=(method,))
        c = fc.coefficients(method)
        expected = sign * 2.0 * A / (U_bar ** 2 * D)                        # c = 2 F / (U^2 D)
        assert c[comp] == pytest.approx(expected, rel=1e-9)
        assert abs(c[1 - comp]) < 1e-9 * abs(expected)


@pytest.mark.parametrize("comp,pressure,sign", [(0, lambda x: -x[0], +1), (1, lambda x: -x[1], +1)])
def test_pressure_force_on_curved_mesh_has_the_right_sign_and_size(comp, pressure, sign):
    """On the curved mesh P1 pressure is linear in REFERENCE coordinates, so a physical-linear p is
    represented only to O(sagitta) (about 1% at this coarse h_cyl). Sign and magnitude must still
    be right to that level; a wrong normal, sign or normalisation would be off by O(1)."""
    md, w = _state(2, pressure=pressure)
    fc = ForceCoefficients(w, md.facet_tags, CYLINDER, NU, 1.0, D, methods=("laplacian",))
    c = fc.coefficients("laplacian")
    assert c[comp] == pytest.approx(sign * 2.0 * np.pi * R ** 2 / D, rel=3e-2)


@pytest.mark.parametrize("method", TRACTION_METHODS)
@pytest.mark.parametrize("velocity,comp", [
    (lambda x: np.vstack((x[1] ** 2, 0.0 * x[0])), 0),
    (lambda x: np.vstack((0.0 * x[0], x[0] ** 2)), 1),
])
def test_viscous_force_matches_divergence_theorem_on_the_meshed_boundary(method, velocity, comp):
    """u = (y^2, 0) (resp. (0, x^2)) is exactly P2 on affine cells, so int nu (grad u) eta ds over
    the meshed boundary is 2 nu A in the drag (lift) component by the divergence theorem, with A
    the enclosed (polygon) area and independent of the cylinder centre; the other component
    vanishes. Checks the viscous stress term, its orientation and the lift/drag assignment."""
    md, w = _state(1, velocity=velocity)
    fc = ForceCoefficients(w, md.facet_tags, CYLINDER, NU, 1.0, D, methods=(method,))
    F = fc.raw_forces(method)
    assert F[comp] == pytest.approx(2.0 * NU * _polygon_area(md), rel=1e-9)
    assert abs(F[1 - comp]) < 1e-9 * abs(F[comp])


def test_polygonal_boundary_lowers_the_integrated_force_by_its_area_error():
    """On the straight-sided mesh the same exact field integrates to the POLYGON's area, so the
    force is low by the measured area error (2D: a divergence-theorem integral over the enclosed
    region). Ties the force error directly to the measured geometry error."""
    md, w = _state(1, pressure=lambda x: -x[0])
    rep = cylinder_geometry_report(md.mesh, md.facet_tags)
    fc = ForceCoefficients(w, md.facet_tags, CYLINDER, NU, 1.0, D, methods=("laplacian",))
    Fx, _ = fc.raw_forces("laplacian")
    exact = np.pi * R ** 2
    assert Fx / exact - 1.0 == pytest.approx(rep["area_rel_err"], rel=1e-6)
    assert Fx < exact


def test_benchmark_normalisation_and_reynolds_numbers():
    """Ubar = (2/3) U_max, Re = Ubar D / nu: 20 for 2D-1 (U_max = 0.3), 100 for 2D-2 (U_max = 1.5)."""
    from scripts import run_2d1_steady as s1, run_2d2_unsteady as s2
    assert s1.D == s2.D == 2 * R
    assert s1.NU == s2.NU == 1.0e-3 and s1.RHO == s2.RHO == 1.0
    assert s1.U_BAR == pytest.approx(0.2) and s2.U_BAR == pytest.approx(1.0)
    assert s1.U_BAR * s1.D / s1.NU == pytest.approx(20.0)
    assert s2.U_BAR * s2.D / s2.NU == pytest.approx(100.0)


def test_pressure_probe_points_are_the_front_and_rear_stagnation_points():
    (ax, ay), (bx, by) = BENCHMARK_PRESSURE_POINTS
    assert (ax, ay, bx, by) == (0.15, 0.2, 0.25, 0.2)
    # both lie on the cylinder surface, on the horizontal line through its centre
    assert np.hypot(ax - CX, ay - CY) == pytest.approx(R)
    assert np.hypot(bx - CX, by - CY) == pytest.approx(R)


@pytest.mark.parametrize("order", [1, 2])
def test_probe_evaluates_a_linear_field_exactly_on_the_cylinder_surface(order):
    md, w = _state(order, pressure=lambda x: 3.0 * x[0] + x[1])
    probe = PointProbe(md.mesh, BENCHMARK_PRESSURE_POINTS)
    pa, pb = probe.evaluate(w.sub(1))
    assert pa == pytest.approx(3 * 0.15 + 0.2, abs=1e-9)
    assert pb == pytest.approx(3 * 0.25 + 0.2, abs=1e-9)


def test_probe_uses_nearest_cell_for_a_point_inside_the_cylinder():
    """A point inside the cylinder lies in no fluid cell; the probe must fall back to the nearest
    cell and extrapolate instead of failing (this is the situation of a probe on a curved
    boundary that the meshed boundary misses by a hair). On the straight-sided mesh the P1 field
    is exactly linear, so the extrapolated value is exact."""
    md, w = _state(1, pressure=lambda x: 3.0 * x[0] + x[1])
    pt = (0.15 + 5e-3, 0.2)
    probe = PointProbe(md.mesh, [pt])
    assert 0.0 < probe.distance[0] < 1e-2
    assert probe.evaluate(w.sub(1))[0] == pytest.approx(3 * pt[0] + pt[1], abs=1e-9)


def test_force_definitions_agree_on_a_solved_steady_flow():
    """On a converged 2D-1 solution the four force evaluations are different approximations of the
    same quantity: they must agree to within a few percent even on a coarse mesh, and the
    variational reaction must be closest to the benchmark's spectral value. (A loose consistency
    check; it does not test the benchmark interval.)"""
    from src.cylinder_bcs import build_bcs
    from src.navier_stokes import residual, newton_solve
    md = build_mesh(h_far=0.06, h_cyl=0.012, geometry_order=2)
    W = taylor_hood_space(md.mesh)
    w = Function(W)
    bcs = build_bcs(W, md.mesh, md.facet_tags, 0.3)
    newton_solve(residual(w, ufl.TestFunction(W), NU), w, bcs, md.mesh, prefix="tf_")
    fc = ForceCoefficients(w, md.facet_tags, CYLINDER, NU, 0.2, D)
    cd = {m: c[0] for m, c in fc.all_coefficients().items()}
    assert all(v > 5.0 for v in cd.values())                      # drag is positive (flow in +x)
    spread = (max(cd.values()) - min(cd.values())) / np.mean(list(cd.values()))
    assert spread < 0.03
    spectral = 5.57953523384
    assert abs(cd["variational"] - spectral) == min(abs(v - spectral) for v in cd.values())
