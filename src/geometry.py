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
from mpi4py import MPI

from dolfinx.io.gmsh import model_to_mesh

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


def build_mesh(h_far: float, h_cyl: float, comm=MPI.COMM_WORLD, rank: int = 0) -> MeshData:
    """Build the channel-with-cylinder mesh via gmsh's OpenCASCADE kernel.

    h_far: target mesh size away from the cylinder.
    h_cyl: target mesh size on and near the cylinder surface (h_cyl < h_far
        to resolve the boundary layer and wake, as is standard practice for
        this benchmark).
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

    mesh_data = model_to_mesh(gmsh.model, comm, rank, gdim=2)
    gmsh.finalize()

    return MeshData(
        mesh=mesh_data.mesh,
        cell_tags=mesh_data.cell_tags,
        facet_tags=mesh_data.facet_tags,
        h_far=h_far,
        h_cyl=h_cyl,
    )
