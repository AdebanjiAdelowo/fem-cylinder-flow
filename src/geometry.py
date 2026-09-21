"""Schäfer-Turek DFG 2D benchmark geometry: channel with an off-centre cylinder.

Geometry, exactly as specified in Schäfer, M. & Turek, S. (1996), "Benchmark
Computations of Laminar Flow Around a Cylinder", in Flow Simulation with
High-Performance Computers II (Notes on Numerical Fluid Mechanics, Vol. 52),
Vieweg, Fig. 1 / Sec. 2.2:

    channel:  Omega = [0, 2.2] x [0, 0.41]  (length 2.2 m, height H = 0.41 m)
    cylinder: centre (0.2, 0.2), diameter D = 0.1 m (radius 0.05 m)

No geometry or boundary-condition values here are invented; every number is
taken directly from that paper. Facet physical-group tags:

    1 = inlet   (x = 0)
    2 = outlet  (x = 2.2)
    3 = walls   (y = 0 and y = 0.41)
    4 = cylinder (circle boundary)
"""
from __future__ import annotations

from dataclasses import dataclass

import gmsh
import numpy as np
import ufl
from mpi4py import MPI

from dolfinx import fem
from dolfinx.io.gmsh import model_to_mesh
from dolfinx.mesh import entities_to_geometry

L, H = 2.2, 0.41
CX, CY, R = 0.2, 0.2, 0.05

INLET, OUTLET, WALLS, CYLINDER = 1, 2, 3, 4


@dataclass
class MeshData:
    mesh: object
    cell_tags: object
    facet_tags: object
    h_far: float
    h_cyl: float
    geometry_order: int = 1


def build_mesh(h_far: float, h_cyl: float, comm=MPI.COMM_WORLD, rank: int = 0,
               geometry_order: int = 1) -> MeshData:
    """Build the channel-with-cylinder mesh via gmsh's OpenCASCADE kernel.

    h_far: target mesh size away from the cylinder.
    h_cyl: target mesh size on and near the cylinder surface (h_cyl < h_far
        to resolve the boundary layer and wake, as is standard practice for
        this benchmark).
    geometry_order: polynomial degree of the mesh geometry map. 1 gives straight-sided cells, so
        the cylinder is an inscribed polygon. 2 asks gmsh for second-order cells whose extra edge
        nodes are placed on the exact CAD circle, giving an isoparametric (curved) cylinder
        boundary; the vertices and cell connectivity are identical to the order-1 mesh, so the
        two differ only in the geometry map. The imported degree is checked below rather than
        assumed to survive the gmsh -> dolfinx conversion.
    """
    gmsh.initialize()
    if comm.rank == rank:
        gmsh.model.add("dfg_cylinder")
        rect = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
        disk = gmsh.model.occ.addDisk(CX, CY, 0, R, R)
        domain, _ = gmsh.model.occ.cut([(2, rect)], [(2, disk)])
        gmsh.model.occ.synchronize()

        surfaces = gmsh.model.getEntities(2)
        assert len(surfaces) == 1
        gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], 1)

        # classify boundary edges: the cylinder is the only non-"Line" (curved,
        # OCC reports it as "Ellipse" for a circle) entity; the four straight
        # edges are then sorted by their centre-of-mass location
        boundary = gmsh.model.getBoundary(surfaces, oriented=False)
        inlet_lines, outlet_lines, wall_lines, cyl_lines = [], [], [], []
        for dim, tag in boundary:
            if gmsh.model.getType(dim, tag) != "Line":
                cyl_lines.append(tag)
                continue
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            if np.isclose(com[0], 0.0, atol=1e-6):
                inlet_lines.append(tag)
            elif np.isclose(com[0], L, atol=1e-6):
                outlet_lines.append(tag)
            else:
                wall_lines.append(tag)

        gmsh.model.addPhysicalGroup(1, inlet_lines, INLET)
        gmsh.model.addPhysicalGroup(1, outlet_lines, OUTLET)
        gmsh.model.addPhysicalGroup(1, wall_lines, WALLS)
        gmsh.model.addPhysicalGroup(1, cyl_lines, CYLINDER)

        # mesh-size field: fine near the cylinder, coarser far away
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", cyl_lines)
        gmsh.model.mesh.field.setNumber(1, "Sampling", 200)

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", h_cyl)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", h_far)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 2 * R)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 8 * R)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        if geometry_order > 1:
            gmsh.model.mesh.setOrder(geometry_order)

    mesh_data = model_to_mesh(gmsh.model, comm, rank, gdim=2)
    gmsh.finalize()

    imported = mesh_data.mesh.geometry.cmap.degree
    if imported != geometry_order:
        raise RuntimeError(f"requested geometry order {geometry_order} but the imported dolfinx mesh "
                           f"has coordinate-element degree {imported}")

    return MeshData(
        mesh=mesh_data.mesh,
        cell_tags=mesh_data.cell_tags,
        facet_tags=mesh_data.facet_tags,
        h_far=h_far,
        h_cyl=h_cyl,
        geometry_order=geometry_order,
    )


def cylinder_geometry_report(mesh, facet_tags, n_sample: int = 33) -> dict:
    """Quantify how well the mesh represents the circle of radius R centred at (CX, CY).

    Everything is measured through the finite-element geometry map that the solver actually uses
    (UFL facet normals, ds measure), not inferred from the nominal mesh order:

    * geometry_degree           coordinate-element degree of the imported mesh
    * n_cyl_facets              number of boundary facets on the cylinder
    * perimeter_rel_err         (integral of ds over the cylinder) / (2 pi R) - 1
    * area_rel_err              enclosed area from the divergence theorem, -1/2 * closed integral of
                                (x - c).n ds with n the fluid-outward (into the cylinder) normal,
                                relative to pi R^2, minus 1
    * radius_eq_rel_err         equivalent-area radius error, sqrt(A/pi)/R - 1
    * max_radial_dev_rel        max over sampled boundary points of | |x - c| - R | / R, sampling
                                each facet's geometry map at `n_sample` parameter values
    """
    comm = mesh.comm
    fdim = mesh.topology.dim - 1
    facets = facet_tags.find(CYLINDER)
    n = ufl.FacetNormal(mesh)
    x = ufl.SpatialCoordinate(mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags, subdomain_id=CYLINDER)
    c = ufl.as_vector((CX, CY))

    perimeter = comm.allreduce(fem.assemble_scalar(fem.form(1.0 * ds)), op=MPI.SUM)
    area = -0.5 * comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.dot(x - c, n) * ds)), op=MPI.SUM)

    # radial deviation of the represented boundary from the exact circle
    degree = mesh.geometry.cmap.degree
    geo = entities_to_geometry(mesh, fdim, facets, False)     # (n_facets, nodes per facet)
    X = mesh.geometry.x[:, :2]
    s = np.linspace(0.0, 1.0, n_sample)
    if degree == 1:
        basis = np.stack([1 - s, s], axis=1)
    elif degree == 2:                                          # vertex, vertex, edge mid-node
        basis = np.stack([(1 - s) * (1 - 2 * s), s * (2 * s - 1), 4 * s * (1 - s)], axis=1)
    else:
        raise NotImplementedError("radial-deviation sampling implemented for degree 1 and 2")
    max_dev = 0.0
    if len(facets):
        pts = np.einsum("sk,fkd->fsd", basis, X[geo])
        max_dev = float(np.max(np.abs(np.hypot(pts[..., 0] - CX, pts[..., 1] - CY) - R))) / R
    max_dev = comm.allreduce(max_dev, op=MPI.MAX)

    return {
        "geometry_degree": int(degree),
        "n_cyl_facets": int(comm.allreduce(len(facets), op=MPI.SUM)),
        "perimeter_rel_err": perimeter / (2 * np.pi * R) - 1.0,
        "area_rel_err": area / (np.pi * R ** 2) - 1.0,
        "radius_eq_rel_err": float(np.sqrt(area / np.pi) / R - 1.0),
        "max_radial_dev_rel": max_dev,
    }
