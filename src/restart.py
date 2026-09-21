"""Save and reload a Taylor-Hood state (u, p), interpolating between different meshes.

A developed periodic flow is an attractor of the unsteady problem, so the benchmark observables do
not depend on how the fully developed state was reached. Spinning the flow up on a coarse mesh and
interpolating onto each target mesh therefore removes most of the (expensive) from-rest transient
on fine meshes. The consequence for the fine run is only that it must first shed the interpolation
error; whether it has done so is judged by the cycle-to-cycle drift in ``src.periodic``, never
assumed. Serial only (the saved array is the process-local dof vector).

The state file stores the mesh recipe (h_far, h_cyl, geometry_order), so the source mesh is rebuilt
deterministically instead of being serialised.
"""
from __future__ import annotations

import numpy as np
from dolfinx import fem
from dolfinx.fem import Function

from src.geometry import build_mesh
from src.spaces import taylor_hood_space


def save_state(path, w: Function, t: float, h_far: float, h_cyl: float, geometry_order: int,
               dt: float | None = None) -> None:
    if w.function_space.mesh.comm.size != 1:
        raise NotImplementedError("save_state is serial-only")
    np.savez(path, w=w.x.array.copy(), t=t, h_far=h_far, h_cyl=h_cyl,
             geometry_order=geometry_order, dt=np.nan if dt is None else dt)


def load_state(path, W_target, mesh_params_target: dict) -> tuple[Function, float, dict]:
    """Return (w_target, t_state, source_params) holding the saved state on the target space.

    If the source and target meshes have identical recipes the dof vector is copied verbatim,
    otherwise velocity and pressure are interpolated non-matching onto the target space.
    """
    if W_target.mesh.comm.size != 1:
        raise NotImplementedError("load_state is serial-only")
    d = np.load(path)
    src = {"h_far": float(d["h_far"]), "h_cyl": float(d["h_cyl"]),
           "geometry_order": int(d["geometry_order"])}
    w_t = Function(W_target)
    if src == {k: mesh_params_target[k] for k in src}:
        if len(d["w"]) != len(w_t.x.array):
            raise RuntimeError("state has matching mesh recipe but a different number of dofs")
        w_t.x.array[:] = d["w"]
    else:
        md = build_mesh(src["h_far"], src["h_cyl"], geometry_order=src["geometry_order"])
        W_s = taylor_hood_space(md.mesh)
        w_s = Function(W_s)
        w_s.x.array[:] = d["w"]
        for i in (0, 1):
            Vt, _ = W_target.sub(i).collapse()
            Vs, _ = W_s.sub(i).collapse()
            f_s = w_s.sub(i).collapse()
            f_t = Function(Vt)
            cells = np.arange(Vt.mesh.topology.index_map(Vt.mesh.topology.dim).size_local, dtype=np.int32)
            data = fem.create_interpolation_data(Vt, Vs, cells, padding=1e-6)
            f_t.interpolate_nonmatching(f_s, cells, interpolation_data=data)
            f_t.x.scatter_forward()
            w_t.sub(i).interpolate(f_t)
    w_t.x.scatter_forward()
    return w_t, float(d["t"]), src
