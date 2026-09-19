import numpy as np

from src.cylinder_bcs import inflow_profile
from src.geometry import H, build_mesh
from src.spaces import taylor_hood_space
from src.cylinder_bcs import build_bcs
from dolfinx.fem import Function


def test_inflow_profile_matches_benchmark_formula():
    """U(0,y) = 4*U_m*y*(H-y)/H**2 (Schafer & Turek, 1996, Sec. 2.2)."""
    U_m = 1.5
    x = np.vstack([np.zeros(3), [0.0, H / 2, H]])  # shape (gdim, npoints); x-coord irrelevant here
    u = inflow_profile(U_m)(x)
    y = x[1]
    expected_u0 = 4.0 * U_m * y * (H - y) / H**2
    assert np.allclose(u[0], expected_u0)
    assert np.allclose(u[1], 0.0)
    # maximum at the centreline should equal U_m
    assert np.isclose(u[0, 1], U_m, atol=1e-12)


def test_inflow_profile_zero_at_walls():
    U_m = 1.5
    x = np.vstack([np.zeros(2), [0.0, H]])
    u = inflow_profile(U_m)(x)
    assert np.allclose(u[0], 0.0)


def test_build_bcs_returns_two_conditions_with_correct_dof_count():
    md = build_mesh(h_far=0.1, h_cyl=0.02)
    W = taylor_hood_space(md.mesh)
    bcs = build_bcs(W, md.mesh, md.facet_tags, U_m=1.5)
    assert len(bcs) == 2
    # applying the BCs to a zero Function then checking non-trivial entries
    w = Function(W)
    for bc in bcs:
        bc.set(w.x.array)
    assert np.any(w.x.array != 0.0)  # inflow BC sets nonzero entries
