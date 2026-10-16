"""Unstructured triangular mesh of the liquid in the cylinder, rebuilt from the live free surface.

The meridional domain is ``0 <= r <= R_c`` below the free surface ``z = d + zeta(r)``, with named
boundaries ``axis`` (r = 0), ``bottom`` (z = 0), ``wall`` (r = R_c) and ``interface``. The first mesh
puts the interface through the initial elevation; every remesh rebuilds it through the current
interface nodes, ordered from the axis to the wall, with the end points snapped onto ``r = 0`` and
``r = R_c``. The rebuild checks the control polygon and a sampled spline for self-contact before
meshing (:func:`~faraday_singularity.selfcontact.check_simple_interface`). Narrow spline loops can
escape this sampled check; detected contacts raise rather than entering mesh generation.

Element size is graded with distance from two attractors: the whole interface (``h_interface``)
and a focus region near the axis, where the cavity collapses. The size grows linearly with distance
at rate ``grading`` and is capped at ``h_max``. Initially the focus is the point where the interface
meets the axis, with size ``h_axis``. During a collapse the problem sets ``focus`` (a
:class:`Focus`) before forcing a remesh: interface nodes within ``radius`` of the axis become
attractors with size ``size``, so the resolution follows the cavity bottom, the jet base and the jet.

Reconnection. The template is a pyoomph ``TopologicalChangesGmshTemplate``, so a reconnection plan
parked on it (``surgery.py``) is built into the next mesh. With no plan pending and a single free
surface, nothing changes: the rebuild is the one described above, and points are not merged (the
template's tolerant point merge is off, ``point_snap_tolerance = 0``). With a plan, or with several
interface chains after a cut (only when the template was made with ``reconnect=True``), the liquid
is bounded by every chain (:func:`~faraday_singularity.reconnect.orient_chain`), the wall, the
bottom and the liquid spans of the axis (:func:`~faraday_singularity.reconnect.liquid_axis_spans`):
before a cut ``[0, z_tip]``, after one ``[0, z_bb]`` below the bubble and the bridge
``[z_bt, z_cb]``. The chains are checked one by one and against each other before meshing. A
:class:`CutSizing` set by the cut adds attractors at the fresh tips and along the bridge, and gmsh's
sizing from curvature with a floor below the tip size, which a cap needs so that no curved element
wraps around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from pyoomph.equations.topological_changes import TopologicalChangesGmshTemplate

from .reconnect import Chain, check_chains, liquid_axis_spans, orient_chain
from .selfcontact import check_simple_interface


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


@dataclass(frozen=True)
class CutSizing:
    """Element sizes after a cut: at the fresh tips on the axis and along the bridge of liquid."""
    h_tip: float
    h_bridge: float
    curvature: int  # gmsh MeshSizeFromCurvature, elements per 2 pi of turning (0: off)
    grading: float

    def __post_init__(self):  # plain numbers: they are written into gmsh field expressions
        for name in ("h_tip", "h_bridge", "grading"):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "curvature", int(self.curvature))
        if not (self.h_tip > 0 and self.h_bridge > 0 and self.grading > 0 and self.curvature >= 0):
            raise ValueError("cut sizes and grading must be positive")


class FaradayCylinderMesh(TopologicalChangesGmshTemplate):
    # The class name names the stored .msh files of state dumps; keep it.
    def __init__(self, radius: float, depth: float, elevation: Callable[[np.ndarray], np.ndarray],
                 settings: MeshSettings, domain_name: str = "liquid", reconnect: bool = False):
        super().__init__()
        self.radius = radius
        self.depth = depth
        self.elevation = elevation  # initial zeta(r), numpy-vectorised
        self.settings = settings
        self.domain_name = domain_name
        self.focus: Focus | None = None  # set by the problem before a resolution-driven remesh
        self.reconnect = reconnect  # several interface chains are allowed (after a cut)
        self.cut_sizing: CutSizing | None = None  # set by a cut, kept for the remeshes after it
        self.point_snap_tolerance = 0.0  # no merging of nearby points unless a cut needs it

    @property
    def interface_name(self) -> str:
        return self.domain_name + "/interface"

    def _interface_points(self, segs=None) -> list[tuple[float, float]]:
        if self.is_first_time():
            r = np.linspace(0.0, self.radius, self.settings.interface_points)
            z = self.depth + self.elevation(r)
            return list(zip(r.tolist(), z.tolist()))
        if segs is None:
            segs = self.get_boundary_coordinates(self.domain_name + "/interface",
                                                 sort_along_axis="x+", nondimensional=True)
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

        segs = None
        if self.is_remeshing() and self.has_pending_surgery_plan(self.interface_name):
            return self._define_chain_geometry(self._planned_chains(), focus, h_focus)
        if not self.is_first_time():
            segs = self.get_boundary_coordinates(self.interface_name, sort_along_axis="x+",
                                                 nondimensional=True)
            if len(segs) > 1 and self.reconnect:
                return self._define_chain_geometry(self._current_chains(segs), focus, h_focus)
        self.point_snap_tolerance = 0.0

        # The rebuilt spline passes through the current interface nodes only; inserting points would
        # put kinks into it. Refining by at most about 2x per remesh keeps the size field between
        # attractor nodes within about 1.2 times the focus size.
        coords = self._interface_points(segs)
        # Gmsh reports a self-crossing boundary only as warnings and an empty surface, which
        # pyoomph meets later as "There is no domain 'liquid'"; refuse it here, with its location.
        check_simple_interface(coords)
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


    # --- several interface chains: a planned reconnection, or the geometry after one ----------------

    def _end_tolerance(self) -> float:
        return 1e-7 * max(1.0, self.radius)

    def _planned_chains(self) -> list[Chain]:
        """The chains of the parked reconnection plan (its caps already replaced by the cut)."""
        rb = self.get_reconnected_boundaries(self.interface_name, self.domain_name + "/axis",
                                             nondimensional=True)
        tol = self._end_tolerance()
        return [orient_chain([(float(x), float(y)) for x, y in ch.points], self.radius, tol)
                for ch in rb.interface_chains]

    def _current_chains(self, segs) -> list[Chain]:
        tol = self._end_tolerance()
        chains = [orient_chain([(float(x), float(y)) for x, y in seg], self.radius, tol)
                  for seg in segs]
        return sorted(chains, key=lambda c: float(c.points[:, 1].min()))

    def _define_chain_geometry(self, chains: list[Chain], focus: Focus | None, h_focus: float):
        """The liquid bounded by several interface chains, the wall, the bottom and the axis."""
        s = self.settings
        cut = self.cut_sizing
        # The chain ends on the axis are shared with the axis lines; merge points that the surgery
        # and the remesh place within round-off of each other.
        self.point_snap_tolerance = 1e-9
        check_chains(chains)
        spans = liquid_axis_spans([z for c in chains for _, z in c.axis_ends()])

        curves, tips, attractor_pts = [], {}, []
        wall_pt = None
        for c in chains:
            pts = [self.point(x, y) for x, y in c.points.tolist()]
            curves.append(self.spline(pts, name="interface"))
            for k, z in c.axis_ends():
                tips[z] = pts[k]
            if c.is_open:
                wall_pt = pts[-1]
            if focus is not None:
                attractor_pts += [pt for pt, x in zip(pts, c.points[:, 0]) if x <= focus.radius]
        lowest = tips[spans[0][1]]
        self.create_lines(wall_pt, "wall", (self.radius, 0.0), "bottom", (0.0, 0.0), "axis", lowest)
        bridges = [self.line(tips[a], tips[b], name="axis") for a, b in spans[1:]]
        self.plane_surface("interface", "wall", "bottom", "axis", name=self.domain_name)

        h_min = min(s.h_interface, h_focus)
        d_if = self.add_mesh_size_field("Distance", CurvesList=curves, Sampling=2000)
        attractors = attractor_pts or [lowest]
        g_focus = focus.grading if focus is not None else s.grading
        d_ax = self.add_mesh_size_field("Distance", PointsList=attractors)
        fields = [self.add_mesh_size_field(
                      "MathEval", F=f"min({s.h_max!r}, {s.h_interface!r} + {s.grading!r}*F{d_if})"),
                  self.add_mesh_size_field(
                      "MathEval", F=f"min({s.h_max!r}, {h_focus!r} + {g_focus!r}*F{d_ax})")]
        if cut is not None:
            d_tip = self.add_mesh_size_field("Distance", PointsList=list(tips.values()))
            fields.append(self.add_mesh_size_field(
                "MathEval", F=f"min({s.h_max!r}, {cut.h_tip!r} + {cut.grading!r}*F{d_tip})"))
            if bridges:
                d_br = self.add_mesh_size_field("Distance", CurvesList=bridges, Sampling=200)
                fields.append(self.add_mesh_size_field(
                    "MathEval", F=f"min({s.h_max!r}, {cut.h_bridge!r} + {cut.grading!r}*F{d_br})"))
            h_min = min(h_min, cut.h_tip, cut.h_bridge)
            if cut.curvature > 0:
                self.set_gmsh_parameter("Mesh.MeshSizeFromCurvature", int(cut.curvature))
        self.set_gmsh_parameter("Mesh.MeshSizeMin", h_min)
        self.set_mesh_size_background_field(self.add_mesh_size_field("Min", FieldsList=fields))
