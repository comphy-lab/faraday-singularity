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
:class:`CutSizing` set by the cut adds attractors at the tips on the axis and along the bridge.
After a cut the focus is a set of regions (:class:`FocusRegion`), one per feature (each tip with
its jet, each neck, the bubble's bottom at the resolution it had at the landing), each meshed at its
own size; without regions it is the focus above.

gmsh's sizing from curvature (``CutSizing.curvature``, ``Mesh.MeshSizeFromCurvature``) is off
unless a cut asks for it. The caps do not need it: an arc of radius at least the cut radius ``delta``
meshed at ``tip_size <= delta/4`` has at least 25 elements per turn of 2 pi. Elsewhere it does harm:
gmsh reads the curvature of the uniform Catmull-Rom spline through the nodes, which bends sharply
where the node spacing jumps, refines there without grading, and so widens the jump at the next
rebuild, rebuild after rebuild, until the spline through the nodes crosses itself
(``testCases/test_reconnect_fem.py`` reproduces the feedback). Without it the ratio of adjacent
element lengths stays near the grading's, about 1.2.

After a cut gmsh's optimiser of curved elements is asked for the cut's ``high_order_threshold``
(``Mesh.HighOrderThresholdMin``, 0.1 by default), since the driver accepts a rebuilt mesh only if
the lower bound of the scaled Jacobian of every curved element reaches the cut's ``jacobian_min``;
and with ``rebuild_spline`` ``vertices`` (for a sensitivity test only) a rebuild splines each chain
through its vertices, every other node, rather than through every node.

In the ring stage (the ring of a pierced bubble, ``ringsurgery.ring_cut``: ``ring_stage`` set on
the template) every rebuild meshes the ring, a closed loop of the free surface, as a hole of the
liquid of two splines named ``interface``, at its frozen sizing (:class:`RingSizing`), from the
geometry a ring surgery plans (``pending_ring_geometry``) or from the current nodes
(``FaradayCylinderMesh._define_ring_geometry``). Before the stage the same build makes the
geometry the fill of a groove plans (``groovesurgery.groove_fill``, ``pending_ring_geometry`` with
the new chains and no loop), the only rebuild that sets one there; every other rebuild is as
before.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
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
class FocusRegion:
    """One feature of the free surface after a cut, meshed at its own element size.

    The interface nodes of every chain within ``radius`` of ``(r, z)`` are attractors of size
    ``size`` (``axial``: those with ``r <= radius``, at any height, as a focus without regions);
    with ``core`` (the twin of a region on the core of the upward jet, ``jetcore.core_twins``) only
    those of the open chain that the core's selection holds (``jetcore.core_select``)."""
    name: str
    size: float
    r: float
    z: float
    radius: float
    axial: bool = False
    core: tuple | None = None  # left out of the records without it (to_dict)

    def __post_init__(self):  # plain numbers: they are written into gmsh field expressions
        for name in ("size", "r", "z", "radius"):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "axial", bool(self.axial))
        if self.core is not None:
            object.__setattr__(self, "core", tuple(float(x) for x in self.core))
        if not (self.size > 0 and self.radius > 0):
            raise ValueError("a focus region needs a positive size and radius")

    def selects(self, P: np.ndarray, is_open: bool = True) -> np.ndarray:
        """Which of the points ``P`` (``(M, 2)``: r, z; the nodes of one chain, the open chain where
        ``is_open``) are attractors of this region."""
        if self.axial:
            inside = P[:, 0] <= self.radius
        else:
            inside = np.hypot(P[:, 0] - self.r, P[:, 1] - self.z) <= self.radius
        if self.core is None:
            return inside
        if not is_open:
            return np.zeros(len(P), dtype=bool)
        from .jetcore import core_select

        return inside & core_select(P, self.core)

    def to_dict(self) -> dict:
        """As a record holds it: every field, ``core`` only where it is set."""
        out = asdict(self)
        if self.core is None:
            del out["core"]
        return out


@dataclass(frozen=True)
class Focus:
    size: float  # element size on the focus attractors
    radius: float  # interface nodes with r <= radius are attractors
    grading: float  # growth rate of the element size away from the focus
    # after a cut: one region per feature, each with its own size (FocusRegion); when given, they
    # are the attractors, and size and radius only summarise them (smallest size, largest radius)
    regions: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "regions", tuple(
            g if isinstance(g, FocusRegion) else FocusRegion(**g) for g in self.regions))

    def to_dict(self) -> dict:
        """As a state dump or a record holds it: the three numbers, and the regions if any."""
        out = {"size": self.size, "radius": self.radius, "grading": self.grading}
        if self.regions:
            out["regions"] = [g.to_dict() for g in self.regions]
        return out

    def region(self, name: str) -> FocusRegion | None:
        return next((g for g in self.regions if g.name == name), None)


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


@dataclass(frozen=True)
class RingSizing:
    """The frozen sizing of the ring of a pierced bubble (``ringsurgery.ring_cut``): no element of
    the ring coarser than ``h_ring``, growing at ``grading`` away from it."""
    h_ring: float
    grading: float

    def __post_init__(self):  # plain numbers: they are written into gmsh field expressions
        for name in ("h_ring", "grading"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if not (self.h_ring > 0 and self.grading > 0):
            raise ValueError("the ring's size and grading must be positive")


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
        # set by a cut too (reconnect.ReconnectSettings): gmsh's Mesh.HighOrderThresholdMin for the
        # meshes after it (None: gmsh's default, 0.1), and the nodes their splines run through
        self.high_order_threshold: float | None = None
        self.rebuild_spline = "nodes"
        self.point_snap_tolerance = 0.0  # no merging of nearby points unless a cut needs it
        # the ring stage, once the ring of a pierced bubble is cut (ringsurgery.ring_cut): every
        # rebuild meshes the ring as a hole of the liquid (_define_ring_geometry) at its frozen
        # sizing; the geometry a ring surgery, or a liquid cut in the ring stage, plans for the
        # next build (chains and loops, consumed by it; no loop removes the ring, or keeps it
        # removed). Inert until set: before the ring nothing reads them
        self.ring_stage = False
        self.ring_sizing: RingSizing | None = None
        self.pending_ring_geometry: dict | None = None

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
        if self.is_remeshing() and (getattr(self, "ring_stage", False)
                                    or getattr(self, "pending_ring_geometry", None) is not None):
            # the ring stage (inert before it): the ring's own build; no pyoomph plan is parked
            # while a loop exists, whose open chart could not carry it; and, before the ring, the
            # build of a geometry a surgery the project owns plans (the fill of a groove,
            # groovesurgery.groove_fill), the only one that sets it then
            if self.has_pending_surgery_plan(self.interface_name):
                raise RuntimeError("a surgery plan is parked on the template in the ring stage")
            return self._define_ring_geometry(focus, h_focus)
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
        """The chains of the current interface nodes; with ``rebuild_spline`` ``vertices`` (after a
        cut, for a sensitivity test only) their vertices, every other node of a chain of quadratic
        elements, so that the spline of the rebuild ignores where the midside nodes have gone."""
        tol = self._end_tolerance()
        chains = [orient_chain([(float(x), float(y)) for x, y in seg], self.radius, tol)
                  for seg in segs]
        if self.rebuild_spline == "vertices":
            if any(len(c.points) % 2 != 1 for c in chains):
                raise RuntimeError("an interface chain with an even number of nodes is no chain of "
                                   "quadratic elements: its vertices are unknown")
            chains = [Chain(c.points[::2].copy(), c.end_types) for c in chains]
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

        regions = focus.regions if focus is not None else ()
        curves, tips, attractor_pts, nodes = [], {}, [], []
        region_pts = [[] for _ in regions]
        wall_pt = None
        for c in chains:
            pts = [self.point(x, y) for x, y in c.points.tolist()]
            curves.append(self.spline(pts, name="interface"))
            nodes += list(zip(pts, c.points.tolist()))
            for k, z in c.axis_ends():
                tips[z] = pts[k]
            if c.is_open:
                wall_pt = pts[-1]
            if focus is not None and not regions:
                attractor_pts += [pt for pt, x in zip(pts, c.points[:, 0]) if x <= focus.radius]
            for sel, g in zip(region_pts, regions):
                sel += [pt for pt, inside in zip(pts, g.selects(c.points, c.is_open)) if inside]
        lowest = tips[spans[0][1]]
        self.create_lines(wall_pt, "wall", (self.radius, 0.0), "bottom", (0.0, 0.0), "axis", lowest)
        bridges = [self.line(tips[a], tips[b], name="axis") for a, b in spans[1:]]
        self.plane_surface("interface", "wall", "bottom", "axis", name=self.domain_name)

        h_min = min(s.h_interface, h_focus)
        d_if = self.add_mesh_size_field("Distance", CurvesList=curves, Sampling=2000)
        g_focus = focus.grading if focus is not None else s.grading
        # the focus: its attractors near the axis at one size, or after a cut one set per feature
        # (FocusRegion), each at its own size; a region that holds no node keeps the node nearest
        # its centre, so that a feature between nodes is not dropped
        if regions:
            sized = [(g.size, pts or [min(nodes, key=lambda n: math.hypot(n[1][0] - g.r,
                                                                         n[1][1] - g.z))[0]])
                     for g, pts in zip(regions, region_pts)]
        else:
            sized = [(h_focus, attractor_pts or [lowest])]
        d_ax = [self.add_mesh_size_field("Distance", PointsList=pts) for _, pts in sized]
        fields = [self.add_mesh_size_field(
            "MathEval", F=f"min({s.h_max!r}, {s.h_interface!r} + {s.grading!r}*F{d_if})")]
        for (size, _), d in zip(sized, d_ax):
            fields.append(self.add_mesh_size_field(
                "MathEval", F=f"min({s.h_max!r}, {size!r} + {g_focus!r}*F{d})"))
            h_min = min(h_min, size)
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
        if self.high_order_threshold is not None:  # after a cut: curved elements of this quality
            self.set_gmsh_parameter("Mesh.HighOrderThresholdMin", float(self.high_order_threshold))
        self.set_mesh_size_background_field(self.add_mesh_size_field("Min", FieldsList=fields))

    # --- the ring stage: a closed loop of the free surface meshed as a hole -----------------------

    def _ring_stage_curves(self) -> tuple[list[Chain], list[np.ndarray]]:
        """The chains and the loops a ring-stage build is made of: the planned geometry of a ring
        surgery or of a liquid cut in the ring stage (``pending_ring_geometry``, consumed: the cut
        chain and the ring as it is), or the current interface nodes, split into
        the chains (``reconnect.orient_chain``) and the closed loops (a segment whose first point
        is its last, within 1e-12, or that ends on neither the axis nor the wall), each loop
        checked to be ordered (``ring.validate_loop_order``) and counter-clockwise."""
        from .ring import orient_loop, validate_loop_order

        tol = self._end_tolerance()
        pending, self.pending_ring_geometry = self.pending_ring_geometry, None
        if pending is not None:
            chains = [c if isinstance(c, Chain) else Chain(np.asarray(c["points"], dtype=float),
                                                           tuple(c["end_types"]))
                      for c in pending.get("chains") or []]
            loops = [np.asarray(x, dtype=float).reshape(-1, 2) for x in pending.get("loops") or []]
        else:
            chains, loops = [], []
            for seg in self.get_boundary_coordinates(self.interface_name, sort_along_axis="x+",
                                                     nondimensional=True):
                P = np.array([(float(x), float(y)) for x, y in seg])
                if len(P) > 2 and np.hypot(*(P[0] - P[-1])) <= 1e-12:
                    loops.append(P[:-1])
                elif min(abs(P[0, 0]), abs(P[-1, 0])) > tol and min(
                        abs(P[0, 0] - self.radius), abs(P[-1, 0] - self.radius)) > tol:
                    loops.append(P)
                else:
                    chains.append(orient_chain(P, self.radius, tol))
        out = []
        for L in loops:
            validate_loop_order(L)
            out.append(orient_loop(L)[0])
        return sorted(chains, key=lambda c: float(c.points[:, 1].min())), out

    def _define_ring_geometry(self, focus: Focus | None, h_focus: float):
        """The liquid of the ring stage: bounded by the chains of the free surface (exactly one of
        them open, as after any cut), the wall, the bottom and the liquid spans of the axis, with
        at most one ring, a closed loop of the free surface touching neither the axis nor the
        wall, as a hole, of two splines named ``interface`` (``ring.loop_halves``, split on
        vertices), so that its nodes are those of the free surface's own boundary. The chains and
        the loop are checked before meshing (``reconnect.check_chains``; the loop simple, its two
        splines without contact, ``ring.loop_spline_contact``, off the axis and away from every
        chain). The sizes are those of a build after the cut (the interface's distance field, the
        focus regions over the nodes of the chains and the loop, the cut's tip attractor at every
        end on the axis and its bridge attractor along every bridge), and the ring's frozen size
        besides (``ring_sizing``: ``h_ring + grading d`` from the loop, at most ``h_max``). After
        the ring's removal the stage builds with no loop, and keeps the cut's sizing and the
        regions, which the build of a single free surface would not."""
        from shapely.geometry import LineString

        from .ring import loop_crossings, loop_halves, loop_spline_contact
        from .selfcontact import InterfaceSelfIntersection

        s = self.settings
        cut = self.cut_sizing
        chains, loops = self._ring_stage_curves()
        if len(loops) > 1:
            raise RuntimeError(f"the ring stage holds {len(loops)} loops of the free surface, not "
                               "at most one")
        self.point_snap_tolerance = 1e-9
        check_chains(chains)
        for L in loops:
            if loop_crossings(L) or loop_spline_contact(L) is not None:
                raise InterfaceSelfIntersection("the ring's free surface, or the splines through "
                                                "it, cross themselves: no hole to mesh")
            ring_line = LineString(np.vstack([L, L[:1]]))
            if not float(L[:, 0].min()) > 0.0 or any(
                    ring_line.distance(LineString(c.points)) <= 0.0 for c in chains):
                raise RuntimeError("the ring touches the axis or a chain of the free surface")
        spans = liquid_axis_spans([z for c in chains for _, z in c.axis_ends()])

        regions = focus.regions if focus is not None else ()
        curves, tips, attractor_pts, nodes, ring_curves = [], {}, [], [], []
        region_pts = [[] for _ in regions]
        wall_pt = None
        for c in chains:
            pts = [self.point(x, y) for x, y in c.points.tolist()]
            curves.append(self.spline(pts, name="interface"))
            nodes += list(zip(pts, c.points.tolist()))
            for k, z in c.axis_ends():
                tips[z] = pts[k]
            if c.is_open:
                wall_pt = pts[-1]
            if focus is not None and not regions:
                attractor_pts += [pt for pt, x in zip(pts, c.points[:, 0]) if x <= focus.radius]
            for sel, g in zip(region_pts, regions):
                sel += [pt for pt, inside in zip(pts, g.selects(c.points, c.is_open)) if inside]
        holes = []
        for L in loops:
            pts = [self.point(x, y) for x, y in L.tolist()]
            a, b = loop_halves(L)
            m = len(a) - 1
            halves = [self.spline(pts[:m + 1], name="interface"),
                      self.spline(pts[m:] + pts[:1], name="interface")]
            holes.append(halves)
            ring_curves += halves
            nodes += list(zip(pts, L.tolist()))
            for sel, g in zip(region_pts, regions):
                sel += [pt for pt, inside in zip(pts, g.selects(L, False)) if inside]
        lowest = tips[spans[0][1]]
        outer = curves + self.create_lines(wall_pt, "wall", (self.radius, 0.0), "bottom",
                                           (0.0, 0.0), "axis", lowest)
        bridges = [self.line(tips[a], tips[b], name="axis") for a, b in spans[1:]]
        self.plane_surface(*outer, *bridges, name=self.domain_name, holes=holes or None)

        h_min = min(s.h_interface, h_focus)
        d_if = self.add_mesh_size_field("Distance", CurvesList=curves + ring_curves, Sampling=2000)
        g_focus = focus.grading if focus is not None else s.grading
        if regions:
            sized = [(g.size, pts or [min(nodes, key=lambda n: math.hypot(n[1][0] - g.r,
                                                                         n[1][1] - g.z))[0]])
                     for g, pts in zip(regions, region_pts)]
        else:
            sized = [(h_focus, attractor_pts or [lowest])]
        d_ax = [self.add_mesh_size_field("Distance", PointsList=pts) for _, pts in sized]
        fields = [self.add_mesh_size_field(
            "MathEval", F=f"min({s.h_max!r}, {s.h_interface!r} + {s.grading!r}*F{d_if})")]
        for (size, _), d in zip(sized, d_ax):
            fields.append(self.add_mesh_size_field(
                "MathEval", F=f"min({s.h_max!r}, {size!r} + {g_focus!r}*F{d})"))
            h_min = min(h_min, size)
        if cut is not None:
            d_tip = self.add_mesh_size_field("Distance", PointsList=list(tips.values()))
            fields.append(self.add_mesh_size_field(
                "MathEval", F=f"min({s.h_max!r}, {cut.h_tip!r} + {cut.grading!r}*F{d_tip})"))
            if bridges:
                d_br = self.add_mesh_size_field("Distance", CurvesList=bridges, Sampling=200)
                h_br = f"{cut.h_bridge!r} + {cut.grading!r}*F{d_br}"
                fields.append(self.add_mesh_size_field("MathEval", F=f"min({s.h_max!r}, {h_br})"))
            h_min = min(h_min, cut.h_tip, cut.h_bridge)
            if cut.curvature > 0:
                self.set_gmsh_parameter("Mesh.MeshSizeFromCurvature", int(cut.curvature))
        rs = self.ring_sizing
        if rs is not None and ring_curves:
            d_ring = self.add_mesh_size_field("Distance", CurvesList=ring_curves, Sampling=2000)
            fields.append(self.add_mesh_size_field(
                "MathEval", F=f"min({s.h_max!r}, {rs.h_ring!r} + {rs.grading!r}*F{d_ring})"))
            h_min = min(h_min, rs.h_ring)
        self.set_gmsh_parameter("Mesh.MeshSizeMin", h_min)
        if self.high_order_threshold is not None:
            self.set_gmsh_parameter("Mesh.HighOrderThresholdMin", float(self.high_order_threshold))
        self.set_mesh_size_background_field(self.add_mesh_size_field("Min", FieldsList=fields))
