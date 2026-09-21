import numpy as np

from src.geometry import build_mesh, INLET, OUTLET, WALLS, CYLINDER, L, H, CX, CY, R


def test_geometry_constants_match_schafer_turek_1996():
    """Section 2.2 of Schafer & Turek (1996): channel [0,2.2]x[0,0.41],
    cylinder centre (0.2, 0.2), diameter 0.1."""
    assert L == 2.2
    assert H == 0.41
    assert CX == 0.2 and CY == 0.2
    assert R == 0.05


def test_mesh_has_all_four_boundary_markers():
    md = build_mesh(h_far=0.1, h_cyl=0.02)
    ft = md.facet_tags
    for marker in [INLET, OUTLET, WALLS, CYLINDER]:
        assert np.sum(ft.values == marker) > 0, f"marker {marker} has no facets"


def test_mesh_cell_count_increases_with_refinement():
    coarse = build_mesh(h_far=0.1, h_cyl=0.02)
    fine = build_mesh(h_far=0.04, h_cyl=0.008)
    n_coarse = coarse.mesh.topology.index_map(2).size_local
    n_fine = fine.mesh.topology.index_map(2).size_local
    assert n_fine > n_coarse


def test_inlet_facets_are_at_x_zero():
    md = build_mesh(h_far=0.1, h_cyl=0.02)
    mesh, ft = md.mesh, md.facet_tags
    mesh.topology.create_connectivity(1, 0)
    x = mesh.geometry.x
    inlet_facets = ft.find(INLET)
    conn = mesh.topology.connectivity(1, 0)
    for f in inlet_facets:
        verts = conn.links(f)
        assert np.allclose(x[verts, 0], 0.0, atol=1e-9)


def test_outlet_facets_are_at_x_L():
    md = build_mesh(h_far=0.1, h_cyl=0.02)
    mesh, ft = md.mesh, md.facet_tags
    mesh.topology.create_connectivity(1, 0)
    x = mesh.geometry.x
    outlet_facets = ft.find(OUTLET)
    conn = mesh.topology.connectivity(1, 0)
    for f in outlet_facets:
        verts = conn.links(f)
        assert np.allclose(x[verts, 0], L, atol=1e-9)


def test_domain_excludes_cylinder_interior():
    """No mesh vertex should lie strictly inside the cylinder."""
    md = build_mesh(h_far=0.1, h_cyl=0.02)
    x = md.mesh.geometry.x
    dist2 = (x[:, 0] - CX) ** 2 + (x[:, 1] - CY) ** 2
    assert np.all(dist2 >= (R - 1e-6) ** 2)


# ---------------------------------------------------------------------------------------------
# Cylinder representation: polygonal (order 1) vs curved isoparametric (order 2)
# ---------------------------------------------------------------------------------------------
import pytest
from dolfinx import fem
import ufl
from mpi4py import MPI

from src.geometry import cylinder_geometry_report


def test_default_geometry_is_straight_sided_degree_1():
    md = build_mesh(h_far=0.1, h_cyl=0.02)
    assert md.mesh.geometry.cmap.degree == 1
    assert cylinder_geometry_report(md.mesh, md.facet_tags)["geometry_degree"] == 1


def test_curved_geometry_order_survives_the_gmsh_to_dolfinx_import():
    """setOrder(2) in gmsh must arrive in dolfinx as a degree-2 coordinate element, with extra
    geometry nodes beyond the vertices (not a silently linearised mesh)."""
    md = build_mesh(h_far=0.1, h_cyl=0.02, geometry_order=2)
    mesh = md.mesh
    assert mesh.geometry.cmap.degree == 2
    assert mesh.geometry.x.shape[0] > mesh.topology.index_map(0).size_local
    assert cylinder_geometry_report(mesh, md.facet_tags)["geometry_degree"] == 2


def test_order2_mesh_shares_vertices_and_cells_with_order1_mesh():
    """The two geometries must differ only in the geometry map, so a polygon-vs-curved
    comparison isolates the boundary representation from the resolution."""
    a = build_mesh(h_far=0.1, h_cyl=0.02, geometry_order=1)
    b = build_mesh(h_far=0.1, h_cyl=0.02, geometry_order=2)
    assert a.mesh.topology.index_map(2).size_local == b.mesh.topology.index_map(2).size_local
    assert a.mesh.topology.index_map(0).size_local == b.mesh.topology.index_map(0).size_local
    assert len(a.facet_tags.find(CYLINDER)) == len(b.facet_tags.find(CYLINDER))


def test_polygon_area_error_is_the_inscribed_polygon_deficit():
    """Straight chords between points on the circle enclose less than pi R^2. For N vertices the
    area is at most (N/2) R^2 sin(2 pi / N) (concavity of sin), so the measured deficit must be at
    least that of the regular N-gon, and close to it for a near-uniform cylinder mesh."""
    md = build_mesh(h_far=0.1, h_cyl=0.02)
    rep = cylinder_geometry_report(md.mesh, md.facet_tags)
    N = rep["n_cyl_facets"]
    regular = N * np.sin(2 * np.pi / N) / (2 * np.pi) - 1.0
    assert rep["area_rel_err"] <= regular + 1e-12
    assert rep["area_rel_err"] == pytest.approx(regular, rel=0.15)
    assert rep["perimeter_rel_err"] < 0.0
    # vertices lie exactly on the circle, so the deviation is the chord sagitta ~ (pi/N)^2 / 2
    assert rep["max_radial_dev_rel"] >= 0.9 * (1 - np.cos(np.pi / N))


def test_curved_boundary_error_is_orders_of_magnitude_below_polygon():
    poly = cylinder_geometry_report(*_mesh_and_tags(1))
    curved = cylinder_geometry_report(*_mesh_and_tags(2))
    assert abs(curved["area_rel_err"]) < 1e-2 * abs(poly["area_rel_err"])
    assert abs(curved["perimeter_rel_err"]) < 1e-2 * abs(poly["perimeter_rel_err"])
    assert curved["max_radial_dev_rel"] < 1e-4
    # the curved boundary must stay on the circle of radius R centred at (CX, CY)
    assert curved["max_radial_dev_rel"] < poly["max_radial_dev_rel"] / 100


def _mesh_and_tags(order):
    md = build_mesh(h_far=0.1, h_cyl=0.02, geometry_order=order)
    return md.mesh, md.facet_tags


def test_curved_geometry_error_decreases_faster_than_polygon_under_refinement():
    """Order-1 error ~ h^2, order-2 error ~ h^4 (measured in area): refining h_cyl 0.02 -> 0.01
    must shrink the curved error by clearly more than the polygon's factor of ~4."""
    def area_err(order, hc):
        md = build_mesh(h_far=0.1, h_cyl=hc, geometry_order=order)
        return abs(cylinder_geometry_report(md.mesh, md.facet_tags)["area_rel_err"])
    poly_ratio = area_err(1, 0.02) / area_err(1, 0.01)
    curved_ratio = area_err(2, 0.02) / area_err(2, 0.01)
    assert 3.0 < poly_ratio < 5.5
    assert curved_ratio > 8.0


@pytest.mark.parametrize("order", [1, 2])
def test_domain_area_is_channel_minus_cylinder(order):
    md = build_mesh(h_far=0.1, h_cyl=0.02, geometry_order=order)
    area = md.mesh.comm.allreduce(
        fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=md.mesh))), op=MPI.SUM)
    rep = cylinder_geometry_report(md.mesh, md.facet_tags)
    exact = L * H - np.pi * R ** 2
    # the domain is short of the exact one by exactly the cylinder-area error (opposite sign)
    assert area == pytest.approx(exact - np.pi * R ** 2 * rep["area_rel_err"], rel=1e-9)


def test_wall_and_cylinder_facets_sit_at_the_specified_places():
    md = build_mesh(h_far=0.1, h_cyl=0.02)
    mesh, ft = md.mesh, md.facet_tags
    mesh.topology.create_connectivity(1, 0)
    x = mesh.geometry.x
    conn = mesh.topology.connectivity(1, 0)
    for f in ft.find(WALLS):
        y = x[conn.links(f), 1]
        assert np.all(np.isclose(y, 0.0, atol=1e-9)) or np.all(np.isclose(y, H, atol=1e-9))
    for f in ft.find(CYLINDER):
        v = x[conn.links(f)]
        assert np.allclose(np.hypot(v[:, 0] - CX, v[:, 1] - CY), R, atol=1e-9)
