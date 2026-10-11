"""Unstructured triangular mesh of the liquid in the cylinder, rebuilt from the live free surface.

The meridional domain is ``0 <= r <= R_c`` below the free surface ``z = d + zeta(r)``, with named
boundaries ``axis`` (r = 0), ``bottom`` (z = 0), ``wall`` (r = R_c) and ``interface``. The first mesh
puts the interface through the initial elevation; every remesh rebuilds it through the current
interface nodes, ordered from the axis to the wall, with the end points snapped onto ``r = 0`` and
``r = R_c``.

Element size is graded with distance from two attractors: the whole interface (``h_interface``)
and a focus region near the axis, where the cavity collapses. The size grows linearly with distance
at rate ``grading`` and is capped at ``h_max``. Initially the focus is the point where the interface
meets the axis, with size ``h_axis``. During a collapse the problem sets ``focus`` (a
:class:`Focus`) before forcing a remesh: interface nodes within ``radius`` of the axis become
attractors with size ``size``, so the resolution follows the cavity bottom, the jet base and the jet.
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


@dataclass(frozen=True)
class Focus:
    size: float  # element size on the focus attractors
    radius: float  # interface nodes with r <= radius are attractors
    grading: float  # growth rate of the element size away from the focus


class FaradayCylinderMesh(GmshTemplate):
    def __init__(self, radius: float, depth: float, elevation: Callable[[np.ndarray], np.ndarray],
                 settings: MeshSettings, domain_name: str = "liquid"):
        super().__init__()
        self.radius = radius
        self.depth = depth
        self.elevation = elevation  # initial zeta(r), numpy-vectorised
        self.settings = settings
        self.domain_name = domain_name
        self.focus: Focus | None = None  # set by the problem before a resolution-driven remesh

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
        focus = self.focus if self.is_remeshing() else None
        h_focus = focus.size if focus is not None else s.h_axis
        self.mesh_mode = "tris"
        self.set_gmsh_parameter("General.NumThreads", 1)  # deterministic meshes
        self.set_gmsh_parameter("Mesh.MeshSizeMin", min(s.h_interface, h_focus))
        self.set_gmsh_parameter("Mesh.MeshSizeMax", s.h_max)
        self.set_gmsh_parameter("Mesh.MeshSizeFromCurvature", 0)
        self.set_gmsh_parameter("Mesh.MeshSizeFromPoints", 0)
        self.set_gmsh_parameter("Mesh.MeshSizeExtendFromBoundary", 0)
        self.set_gmsh_parameter("Mesh.Algorithm", 6)
        self.default_resolution = s.h_max

        # The rebuilt spline passes through the current interface nodes only; inserting points would
        # put kinks into it. Refining by at most about 2x per remesh keeps the size field between
        # attractor nodes within about 1.2 times the focus size.
        coords = self._interface_points()
        pts = [self.point(x, y) for x, y in coords]
        interface = self.spline(pts, name="interface")
        self.create_lines(pts[-1], "wall", (self.radius, 0.0), "bottom", (0.0, 0.0), "axis", pts[0])
        self.plane_surface("interface", "wall", "bottom", "axis", name=self.domain_name)

        d_if = self.add_mesh_size_field("Distance", CurvesList=[interface], Sampling=2000)
        if focus is None:
            attractors, g_focus = [pts[0]], s.grading
        else:
            attractors = [pt for pt, (x, _) in zip(pts, coords) if x <= focus.radius] or [pts[0]]
            g_focus = focus.grading
        d_ax = self.add_mesh_size_field("Distance", PointsList=attractors)
        size_if = self.add_mesh_size_field(
            "MathEval", F=f"min({s.h_max!r}, {s.h_interface!r} + {s.grading!r}*F{d_if})")
        size_ax = self.add_mesh_size_field(
            "MathEval", F=f"min({s.h_max!r}, {h_focus!r} + {g_focus!r}*F{d_ax})")
        self.set_mesh_size_background_field(
            self.add_mesh_size_field("Min", FieldsList=[size_if, size_ax]))

