from mpi4py import MPI
from dolfinx import mesh as dmesh

from src.spaces import taylor_hood_space


def test_taylor_hood_dof_counts():
    """P2 velocity (2 components) + P1 pressure on an NxN unit-square mesh:
    P2 dofs = vertices + edge midpoints, P1 dofs = vertices."""
    N = 4
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    W = taylor_hood_space(msh)

    msh.topology.create_connectivity(1, 0)
    n_vertices = msh.topology.index_map(0).size_local
    n_edges = msh.topology.index_map(1).size_local

    expected_p2_scalar_dofs = n_vertices + n_edges
    expected_p1_dofs = n_vertices
    expected_total = 2 * expected_p2_scalar_dofs + expected_p1_dofs

    n_dofs = W.dofmap.index_map.size_local * W.dofmap.index_map_bs
    assert n_dofs == expected_total


def test_subspaces_have_correct_value_shape():
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    W = taylor_hood_space(msh)
    V0, _ = W.sub(0).collapse()
    Q0, _ = W.sub(1).collapse()
    assert tuple(V0.element.value_shape) == (2,)
    assert tuple(Q0.element.value_shape) == ()
