"""Unstructured triangular mesh of the liquid in the cylinder, rebuilt from the live free surface.

The meridional domain is ``0 <= r <= R_c`` below the free surface ``z = d + zeta(r)``, with named
boundaries ``axis`` (r = 0), ``bottom`` (z = 0), ``wall`` (r = R_c) and ``interface``. The first mesh
puts the interface through the initial elevation; every remesh rebuilds it through the current
interface nodes, ordered from the axis to the wall, with the end points snapped onto ``r = 0`` and
``r = R_c``.

Element size is graded with distance from two attractors: the whole interface (``h_interface``)
and the point where the interface meets the axis (``h_axis``), where the cavity collapses. The size
grows linearly with distance at rate ``grading`` and is capped at ``h_max``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from pyoomph.meshes.gmsh import GmshTemplate


@dataclass(frozen=True)
class MeshSettings:
    h_interface: float = 0.05
    h_axis: float = 0.05
    h_max: float = 0.4
    grading: float = 0.3
    interface_points: int = 401  # spline points of the initial interface

    def __post_init__(self):
        if min(self.h_interface, self.h_axis, self.h_max, self.grading) <= 0:
            raise ValueError("mesh sizes and grading must be positive")
        if self.interface_points < 3:
            raise ValueError("need at least three interface points")


class FaradayCylinderMesh(GmshTemplate):
    def __init__(self, radius: float, depth: float, elevation: Callable[[np.ndarray], np.ndarray],
                 settings: MeshSettings, domain_name: str = "liquid"):
        super().__init__()
        self.radius = radius
        self.depth = depth
        self.elevation = elevation  # initial zeta(r), numpy-vectorised
        self.settings = settings
        self.domain_name = domain_name

    def _interface_points(self) -> list[tuple[float, float]]:
        if self.is_first_time():
            r = np.linspace(0.0, self.radius, self.settings.interface_points)
            z = self.depth + self.elevation(r)
            return list(zip(r.tolist(), z.tolist()))
        segs = self.get_boundary_coordinates(self.domain_name + "/interface", sort_along_axis="x+",
                                             nondimensional=True)
        if len(segs) != 1:
            raise RuntimeError(f"free surface split into {len(segs)} segments; cannot remesh")
        pts = [(float(x), float(y)) for x, y in segs[0]]
        pts[0] = (0.0, pts[0][1])
        pts[-1] = (self.radius, pts[-1][1])
        return pts

    def define_geometry(self):
        s = self.settings
        self.mesh_mode = "tris"
        self.set_gmsh_parameter("General.NumThreads", 1)  # deterministic meshes
        self.set_gmsh_parameter("Mesh.MeshSizeMin", min(s.h_interface, s.h_axis))
        self.set_gmsh_parameter("Mesh.MeshSizeMax", s.h_max)
        self.set_gmsh_parameter("Mesh.MeshSizeFromCurvature", 0)
        self.set_gmsh_parameter("Mesh.MeshSizeFromPoints", 0)
        self.set_gmsh_parameter("Mesh.MeshSizeExtendFromBoundary", 0)
        self.set_gmsh_parameter("Mesh.Algorithm", 6)
        self.default_resolution = s.h_max

        pts = [self.point(x, y) for x, y in self._interface_points()]
        interface = self.spline(pts, name="interface")
        self.create_lines(pts[-1], "wall", (self.radius, 0.0), "bottom", (0.0, 0.0), "axis", pts[0])
        self.plane_surface("interface", "wall", "bottom", "axis", name=self.domain_name)

        d_if = self.add_mesh_size_field("Distance", CurvesList=[interface], Sampling=2000)
        d_ax = self.add_mesh_size_field("Distance", PointsList=[pts[0]])
        size_if = self.add_mesh_size_field(
            "MathEval", F=f"min({s.h_max!r}, {s.h_interface!r} + {s.grading!r}*F{d_if})")
        size_ax = self.add_mesh_size_field(
            "MathEval", F=f"min({s.h_max!r}, {s.h_axis!r} + {s.grading!r}*F{d_ax})")
        self.set_mesh_size_background_field(
            self.add_mesh_size_field("Min", FieldsList=[size_if, size_ax]))
