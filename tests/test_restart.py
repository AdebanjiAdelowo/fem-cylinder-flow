"""State save/load and coarse-to-fine interpolation used to start fine runs from a developed flow."""
import numpy as np
import pytest
from dolfinx.fem import Function

from src.forces import PointProbe
from src.geometry import build_mesh
from src.restart import load_state, save_state
from src.spaces import taylor_hood_space

FIELD_U = lambda x: np.vstack((x[1] ** 2, 0.0 * x[0]))     # exactly P2 on affine cells
FIELD_P = lambda x: x[0]                                    # exactly P1 on affine cells


def _coarse_state(tmp_path, order):
    md = build_mesh(0.1, 0.03, geometry_order=order)
    W = taylor_hood_space(md.mesh)
    w = Function(W)
    w.sub(0).interpolate(FIELD_U)
    w.sub(1).interpolate(FIELD_P)
    w.x.scatter_forward()
    path = tmp_path / "state.npz"
    save_state(path, w, t=1.25, h_far=0.1, h_cyl=0.03, geometry_order=order, dt=0.01)
    return md, W, w, path


def test_round_trip_on_identical_mesh_recipe_is_bitwise(tmp_path):
    md, W, w, path = _coarse_state(tmp_path, 2)
    w2, t, src = load_state(path, W, {"h_far": 0.1, "h_cyl": 0.03, "geometry_order": 2})
    assert t == 1.25 and src["geometry_order"] == 2
    assert np.array_equal(w2.x.array, w.x.array)


def test_interpolation_to_finer_straight_sided_mesh_reproduces_exact_fields(tmp_path):
    """On affine cells x and y^2 are exactly representable, so the fine-mesh state must equal the
    analytic fields at every fine dof, including nodes on the cylinder that lie outside the coarse
    polygon (extrapolated from the nearest coarse cell, not zeroed)."""
    _, _, _, path = _coarse_state(tmp_path, 1)
    fine = build_mesh(0.06, 0.015, geometry_order=1)
    Wf = taylor_hood_space(fine.mesh)
    wf, _, _ = load_state(path, Wf, {"h_far": 0.06, "h_cyl": 0.015, "geometry_order": 1})
    for i, fun in enumerate((FIELD_U, FIELD_P)):
        Vf, _ = Wf.sub(i).collapse()
        exact = Function(Vf)
        exact.interpolate(fun)
        assert np.max(np.abs(wf.sub(i).collapse().x.array - exact.x.array)) < 1e-10


def test_interpolation_on_curved_meshes_equals_coarse_function_evaluated_at_fine_nodes(tmp_path):
    """The definition of the restart interpolation: fine value = coarse FE function at the fine
    node. (Compared with this, not with an analytic field: on curved cells physical polynomials are
    not in the mapped FE space, so the coarse state itself differs from the analytic field.)"""
    md, W, w, path = _coarse_state(tmp_path, 2)
    fine = build_mesh(0.06, 0.015, geometry_order=2)
    Wf = taylor_hood_space(fine.mesh)
    wf, _, _ = load_state(path, Wf, {"h_far": 0.06, "h_cyl": 0.015, "geometry_order": 2})
    Vp, _ = Wf.sub(1).collapse()
    X = Vp.tabulate_dof_coordinates()
    rng = np.random.default_rng(0)
    sel = rng.choice(len(X), size=60, replace=False)
    coarse = PointProbe(md.mesh, X[sel, :2]).evaluate(w.sub(1))
    assert np.max(np.abs(wf.sub(1).collapse().x.array[sel] - coarse)) < 1e-9


def test_mismatched_dof_count_with_identical_recipe_is_rejected(tmp_path):
    md, W, w, path = _coarse_state(tmp_path, 2)
    other = taylor_hood_space(build_mesh(0.06, 0.015, geometry_order=2).mesh)
    with pytest.raises(RuntimeError):
        load_state(path, other, {"h_far": 0.1, "h_cyl": 0.03, "geometry_order": 2})
