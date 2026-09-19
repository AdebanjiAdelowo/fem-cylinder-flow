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
