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
