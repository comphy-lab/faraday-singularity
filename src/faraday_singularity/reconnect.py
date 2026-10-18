"""Geometry of a reconnection through the neck pinch-off: settings, chains, caps and checks.

A sharp-interface mesh cannot change topology, so a closing gas neck ends a run at a cutoff radius.
A reconnection ("cut") continues it: at neck radius ``delta`` the gas thread is cut, the liquid gains
a short segment of the axis (the bridge), and the free surface splits into an open chain (from the
axis to the wall, bounding the cavity above) and a closed chain (from the axis to the axis, bounding
the bubble below). The liquid velocity is carried across and made divergence-free for the new
topology by a pressure-impulse projection (``surgery.py``). The bubble keeps the passive,
constant (zero) gas pressure of the rest of the free surface: the gas is the zero-density,
incompressible limit with no degrees of freedom of its own.

This module holds what needs no finite-element problem:

* :class:`ReconnectSettings`, the ``collapse.reconnect`` block of a case file;
* chains: interface polylines with their end types (``"axis"`` on ``r = 0``, ``"fixed"`` on the
  wall), their orientation, and the liquid spans of the axis between them (:func:`liquid_axis_spans`);
* the replacement of the planner's caps by a tangent circular arc (:func:`replace_caps`). pyoomph's
  planner (``pyoomph.meshes.axisymm_topology.detect_and_plan``) cuts the right topology, but on a
  slender gas thread its caps fold back on themselves (a needle inside the thread joined to it by a
  backward connector, with wedge angles of a few degrees) and cannot be meshed. Each cap is
  therefore always rebuilt: from the splice, a surviving node of the old interface, the new cap is
  the circular arc tangent to the old interface there whose centre lies on the axis, so it meets
  the axis at a right angle.

  The splice is the first survivor beyond the planner's exclusion window, ``cap_window_factor *
  delta`` about the event (6 by default), whose radius is at least ``delta``. The window sets the
  cut geometry: on slender threads (the synthetic one of the tests and the stored necks they can
  read) the first survivor already lies at ``r_s/delta`` of about 1.4-1.5, so the floor at
  ``delta`` drops no survivor (the cut record's ``in_cut_radii`` says ``splice_set_by``
  ``planner-window`` for each cap). The floor binds only on a thread that stays narrower than
  ``delta`` beyond the window (``delta-floor``): it keeps a fresh cap off a thread narrower than
  the post-cut neck limit ``delta`` (:func:`post_cut_stops`). There is no other splice setting. At
  ``delta/r_n = 1.05`` on those threads the bridge is 13-14.5 delta long and the cut removes 55-58
  delta^3 of gas (12.5-17 delta and 50-66 delta^3 over ``delta/r_n`` from 1.03 to 1.1); with a
  window of 4 about 10-11 delta and 36-39 delta^3, with 8 about 17-18 delta and 78-84 delta^3. The
  bridge is therefore much longer than the cut radius. The liquid that
  fills the bridge carries the inflow of the neck wall, so the impulse of the projection, and with
  it the first jet speeds, may depend on ``cap_window_factor``: a study in ``delta`` needs a
  sensitivity to it (e.g. 4 and 8), as to the fill;
* checks of the edited chains (simple, disjoint, monotone cap, no knife-edge angles), the volume
  the cut removes from the gas (:func:`chain_volumes`), and the gas volume within a box around the
  cut (:func:`gas_volume_in_box`), which separates the volume the cut adds to the liquid from the
  drift of rebuilding the whole free surface;
* the stops of a run after the cut (:func:`post_cut_stops`), on both chains, the self-contact a
  rebuild would meet among them (:func:`spline_contact_stop`), the fold across the groove at the
  base of a jet (``annular-contact-approach``), the reading of the interface on its quadratic
  elements (:func:`p2_edges`, :func:`edge_reading`, :func:`interface_guard`: the turn of an
  element's tangent and the place of its midside node; ``interface-unresolved``) and the
  ``kink_<chain>`` regions it asks for (:func:`kink_regions`), the base of each jet, the apex of the
  groove around it (:func:`jet_base_apex`, :func:`circle_radius`, :func:`groove_gap`), the lower
  bound of the scaled Jacobian of a quadratic triangle (:func:`p2_jacobian_bounds`), the features
  of both chains that the mesh after the cut follows, each in its own region at its own scale
  (:func:`feature_regions`, :func:`kept_region`, :func:`region_spacing`), and what the run
  records of both chains after it and steps by (:func:`post_cut_record`: the three tips on the axis
  with their radii of curvature, the bridge, the bubble and the necks of both chains;
  :func:`bubble_end` and :func:`tip_jet`: the jet bases of the bubble's two tips, measured as the
  open chain's, the bubble's top mirrored so that the jet the bridge drives down into the bubble is
  measured as one rising from a cavity; :func:`courant_limit`: the time in which the fastest node
  of either chain, relative to its element length, crosses an element), the size the cut's mesh
  sizing asks for along a chain
  (:func:`cut_size_field`, where the driver does not read the spacing of its focus), and which
  settings a continuation after the cut may change (``CONTINUATION_KEYS``: ``tau_end``).

Units are those of the solver: length ``1/k``. The neck radius ``r_n`` is the narrowest interior
local minimum of ``r`` along the free surface, as the collapse driver measures it
(:func:`neck_radius`). A cut needs ``delta`` a little above ``r_n``: just above it the planner
defers (the waist is not yet separable), so the landing requires ``delta/r_n >=
min_landing_ratio``. The cut geometry in units of ``delta`` (bridge, removed gas) depends on
``delta/r_n``, so a cut also requires ``delta/r_n <= max_landing_ratio``: a study in ``delta``
needs, for each ``delta``, a landing state whose neck has just passed below it. A run with
``collapse.landing`` keeps one for each of its cut radii, at the first measured state with
``delta/r_n >= LANDING_RATIO``, and none above ``MAX_LANDING_RATIO`` (``driver.CollapseDriver``);
the settings refuse a window ``[min_landing_ratio, max_landing_ratio]`` that does not contain
``[LANDING_RATIO, MAX_LANDING_RATIO]``, so that every landing state a run keeps can be cut.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from .collapse import Profile, axis_curvature, jet_base, necks
from .selfcontact import (
    FOLD_RESOLVED_ELEMENTS,
    InterfaceSelfIntersection,
    check_simple_interface,
    classify_contact,
    first_contact,
    fold_gap,
)

AXIS, FIXED = "axis", "fixed"
FILLS = ("blend", "zero")
RESTARTS = ("bdf1", "impulsive")
BUBBLE_MODELS = ("constant-pressure",)
CAPS = ("arc",)
REBUILD_SPLINES = ("nodes", "vertices")
LANDING_RATIO = 1.05  # a landing state for cut radius delta: the first with delta / r_n >= this
MAX_LANDING_RATIO = 1.1  # and a cut is refused above this delta / r_n (the default of the settings)
# Keys of the settings that decide only where a run after the cut ends: a continuation of a state
# written after the cut may change them; every other key must be the cut's (continues_cut).
CONTINUATION_KEYS = ("tau_end",)
# After a cut the collapse driver reads the spacing near the collapse only where the cut's sizing
# asks for at least this many focus sizes (cut_size_field; driver.CollapseDriver.measure).
CUT_SIZING_MARGIN = 1.5
# The three tips on the axis after a cut, as chains.txt names their jets (driver.CollapseDriver).
JET_TIPS = ("cavity_tip", "bubble_bottom", "bubble_top")
# For the bubble's two tips, the axial direction in which a jet from the tip runs into the gas: up
# from its bottom (liquid below, as at the cavity's tip), down from its top (the bridge above it).
BUBBLE_TIP_SIGNS = {"bubble_bottom": 1.0, "bubble_top": -1.0}


@dataclass(frozen=True)
class ReconnectSettings:
    """The ``collapse.reconnect`` block of a case: one cut at neck radius ``delta``.

    Only a constant-pressure bubble exists (the passive gas of the free surface); the planner's own
    volume correction is off; the planner's caps are always replaced by tangent arcs (``cap``).
    Every key but those of ``CONTINUATION_KEYS`` is a key of the cut, which a continuation keeps:
    the guard of the interface after the cut and the checks of its curved elements included.
    """
    delta: float  # cut radius
    reservoir_depth: float = 3.0  # closes the cavity chain through the gas above the free surface
    cap_window_factor: float = 6.0  # the planner's exclusion radius around the event, in delta
    min_landing_ratio: float = 1.03  # delta / r_n at least this
    max_landing_ratio: float = MAX_LANDING_RATIO  # and at most this, retries of the planner included
    plan_attempts: int = 5  # planner calls, the cut radius growing by retry_factor after a refusal
    retry_factor: float = 1.01
    cap: str = "arc"  # the splice: the first survivor beyond the window with r >= delta
    cap_min_points: int = 16  # control points of each arc, at least
    cap_tangent_fit: float = 2.0  # the tangent at the splice is fitted over this many r_s of arc
    min_cap_angle: float = 30.0  # smallest interior angle (deg) of a cap polyline
    # Mesh sizes of the cut, in cut radii, so that the meshes of a study in delta are similar: at
    # every tip on the axis (fresh caps and the bubble's far end) and along the bridge, kept for
    # every remesh after the cut; and the focus of the cut's remesh, which replaces the focus left
    # by the run that wrote the landing state (the driver's rule sets it again at its next rebuild)
    tip_size: float = 1.0 / 16.0  # at most 1/4: gmsh then follows a cap without inverting elements
    bridge_size: float = 1.0 / 16.0  # at most 1/4
    focus_size: float = 1.0 / 16.0
    focus_radius: float = 20.0  # interface nodes with r <= focus_radius * cut radius
    # gmsh MeshSizeFromCurvature after the cut, elements per 2 pi (0: off). Off by default: the caps
    # do not need it (tip_size <= 1/4), and it reads the curvature of the spline through the nodes,
    # which a jump in node spacing makes spurious, and refines there ungraded (mesh.py)
    curvature_sizing: int = 0
    fill: str = "blend"  # liquid velocity in the former gas: pyoomph's nearest-node blend, or zero
    projection: bool = True
    projection_tolerance: float = 1e-10  # absolute Newton tolerance of the projection
    energy_tolerance: float = 1e-10  # |KE(u~) - KE(u+) - |u+ - u~|^2/2| <= this * KE(u~)
    restore_pressure: bool = True  # Newton's initial guess for the pressure after the projection
    restart: str = "bdf1"  # first step after the cut: BDF1, or BDF2 on an impulsive history
    bubble_model: str = "constant-pressure"
    dt0: float = 1e-9  # first step after the cut
    dt_growth: float = 1.25  # growth of the step after the cut
    courant: float = 0.2  # and at most courant * (smallest tip element) / (largest surface speed)
    # the step after the cut also obeys the collapse's rule, courant h_focus / U (the finest focus
    # size over the largest surface speed anywhere); off, only every node's own Courant limit
    # (courant_limit) and the capillary time of the sharpest tip or jet base bound it
    focus_courant: bool = True
    # each step after the cut makes at least this many Newton iterations, whatever its tolerance:
    # at the steps after a cut (dt about 1e-9) the tolerance newton_floor/dt passes the first
    # iterate, whose error is systematic and accumulates into noise at the scale of the elements
    newton_min_iterations: int = 2
    tau_end: float = math.inf  # time after the cut at which the continuation ends
    max_events: int = 1
    bridge_stop: float = 0.1  # stop once the bridge is shorter than this * its length at the cut
    # a new neck on either chain stops the run (secondary-pinch-approach) only where it is narrower
    # than the cut radius and at least this fraction narrower than the lower of the widest points on
    # either side of it along the chain, or narrower than the mesh can follow (neck_elements h_min):
    # a ripple on a jet thinner than the cut radius is no neck
    neck_depth: float = 0.25
    bubble_stop: float = 0.01  # or the bubble has shrunk below this fraction of its volume at the cut
    # The interface after the cut, read on its quadratic elements (p2_edges, interface_guard) after
    # every accepted step and on the input of every rebuild: a rebuild refines where the tangent of
    # an element turns by more than edge_turn_refine degrees across it, or its midside node lies
    # more than midside_refine chords from mid-chord; the run stops (interface-unresolved) where the
    # turn exceeds edge_turn_stop_fresh on a freshly built mesh or edge_turn_stop on any, or the
    # midside node lies more than midside_stop chords from mid-chord (at a quarter the Jacobian of
    # the element vanishes at its vertex)
    edge_turn_refine: float = 30.0
    edge_turn_stop_fresh: float = 90.0
    edge_turn_stop: float = 120.0
    midside_refine: float = 0.1
    midside_stop: float = 0.2
    # a rebuilt mesh is accepted only if the lower bound of the scaled Jacobian of every curved
    # element (p2_jacobian_bounds) is at least jacobian_min, and gmsh's high-order optimiser is
    # asked for high_order_threshold after the cut (Mesh.HighOrderThresholdMin; 0.1 by default)
    jacobian_min: float = 0.2
    high_order_threshold: float = 0.3
    # an element whose quality (pyoomph's: smallest over mean Jacobian) falls below quality_floor
    # asks for a rebuild, besides pyoomph's relative criterion (a fifth of its quality at the build)
    quality_floor: float = 0.15
    # pyoomph's net for an element that turns inside out within a step (RemeshWhen's
    # on_inverted_element: restore the step's start, rebuild, retry), armed after the cut
    inverted_element_remesh: bool = True
    # the base of each jet (the apex of the groove around it) is meshed in a region of its own,
    # apex_extent apex radii wide, sized from the radius predicted apex_lookahead steps ahead
    apex_extent: float = 8.0
    apex_lookahead: int = 20
    # elements per local scale of the features after the cut (the tips, the bases of the jets, the
    # necks); None takes the collapse's elements_per_scale, so that one landing state can be cut at
    # several resolutions of the run after the cut
    elements_per_scale: float | None = None
    # the spline of a rebuild after the cut runs through every node of the interface ("nodes") or,
    # for a sensitivity test only, through its vertices ("vertices"), which smooths away the very
    # midside nodes the interface guard reads
    rebuild_spline: str = "nodes"

    def __post_init__(self):
        eps = self.elements_per_scale
        if eps is not None and (isinstance(eps, bool) or not isinstance(eps, (int, float))
                                or not math.isfinite(eps) or not eps > 0):
            raise ValueError("reconnect.elements_per_scale must be positive and finite, or null")

        def positive(name):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0 \
                    or not math.isfinite(value):
                raise ValueError(f"reconnect.{name} must be positive and finite")
        for name in ("delta", "reservoir_depth", "cap_window_factor", "retry_factor",
                     "cap_tangent_fit", "min_cap_angle", "tip_size", "bridge_size", "focus_size",
                     "focus_radius", "projection_tolerance", "energy_tolerance",
                     "dt0", "dt_growth", "courant", "bridge_stop", "bubble_stop",
                     "edge_turn_refine", "edge_turn_stop_fresh", "edge_turn_stop",
                     "midside_refine", "midside_stop", "jacobian_min", "high_order_threshold",
                     "quality_floor", "apex_extent"):
            positive(name)
        if not self.edge_turn_refine < self.edge_turn_stop_fresh <= self.edge_turn_stop < 180.0:
            raise ValueError("reconnect: edge_turn_refine < edge_turn_stop_fresh <= edge_turn_stop "
                             "< 180 degrees")
        if not self.midside_refine < self.midside_stop < 0.25:
            raise ValueError("reconnect: midside_refine < midside_stop < 1/4 (the quarter point)")
        for name in ("jacobian_min", "quality_floor"):
            if not getattr(self, name) < 1.0:
                raise ValueError(f"reconnect.{name} must be below one")
        if not self.high_order_threshold <= 1.0:
            raise ValueError("reconnect.high_order_threshold must be at most one")
        if isinstance(self.apex_lookahead, bool) or not isinstance(self.apex_lookahead, int) \
                or self.apex_lookahead < 0:
            raise ValueError("reconnect.apex_lookahead must be a non-negative integer")
        if self.rebuild_spline not in REBUILD_SPLINES:
            raise ValueError(f"reconnect.rebuild_spline must be one of {REBUILD_SPLINES}")
        if not self.min_landing_ratio > 1.0:
            raise ValueError("reconnect.min_landing_ratio must exceed one")
        if isinstance(self.max_landing_ratio, bool) or not isinstance(
                self.max_landing_ratio, (int, float)) or not math.isfinite(self.max_landing_ratio) \
                or not self.max_landing_ratio > self.min_landing_ratio:
            raise ValueError("reconnect.max_landing_ratio must be finite and exceed "
                             "min_landing_ratio")
        # every landing state a run keeps (driver.CollapseDriver: LANDING_RATIO <= delta/r_n <=
        # MAX_LANDING_RATIO) must be one the cut accepts
        if not self.min_landing_ratio <= LANDING_RATIO:
            raise ValueError(f"reconnect.min_landing_ratio must be at most {LANDING_RATIO}: runs "
                             f"keep landing states from delta/r_n = {LANDING_RATIO} on")
        if not self.max_landing_ratio >= MAX_LANDING_RATIO:
            raise ValueError(f"reconnect.max_landing_ratio must be at least {MAX_LANDING_RATIO}: "
                             f"runs keep landing states up to delta/r_n = {MAX_LANDING_RATIO}")
        if not self.retry_factor > 1.0:
            raise ValueError("reconnect.retry_factor must exceed one")
        for name in ("plan_attempts", "cap_min_points", "max_events"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"reconnect.{name} must be a positive integer")
        if self.max_events != 1:
            raise ValueError("reconnect.max_events: only one cut per run is supported")
        if isinstance(self.curvature_sizing, bool) or not isinstance(self.curvature_sizing, int) \
                or self.curvature_sizing < 0:
            raise ValueError("reconnect.curvature_sizing must be a non-negative integer")
        for name, allowed in (("fill", FILLS), ("restart", RESTARTS), ("cap", CAPS),
                              ("bubble_model", BUBBLE_MODELS)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"reconnect.{name} must be one of {allowed}")
        for name in ("projection", "restore_pressure", "inverted_element_remesh", "focus_courant"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"reconnect.{name} must be true or false")
        if not (self.tau_end > 0):
            raise ValueError("reconnect.tau_end must be positive")
        if isinstance(self.newton_min_iterations, bool) or not isinstance(
                self.newton_min_iterations, int) or self.newton_min_iterations < 1:
            raise ValueError("reconnect.newton_min_iterations must be a positive integer")
        if isinstance(self.neck_depth, bool) or not isinstance(self.neck_depth, (int, float)) \
                or not 0.0 <= self.neck_depth < 1.0:
            raise ValueError("reconnect.neck_depth must lie in [0, 1)")
        if not self.min_cap_angle < 90.0:
            raise ValueError("reconnect.min_cap_angle must be below 90 degrees")
        for name in ("tip_size", "bridge_size"):
            if not getattr(self, name) <= 0.25:
                raise ValueError(f"reconnect.{name} must be at most 1/4 (of the cut radius)")

    @classmethod
    def from_dict(cls, data: dict) -> "ReconnectSettings":
        if not isinstance(data, dict):
            raise ValueError("collapse.reconnect must be an object")
        removed = {"cap_splice_factor": "the planner's window (cap_window_factor) sets the splice, "
                                        "with a floor at delta",
                   "tip_elements": "tip_size gives the tip element size in cut radii",
                   "tip_floor": "tip_size gives the tip element size in cut radii",
                   "bridge_elements": "bridge_size gives the bridge element size in cut radii"}
        for key, why in removed.items():
            if key in data:
                raise ValueError(f"collapse.reconnect.{key} was removed: {why}")
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError("collapse.reconnect: unknown key(s) " + ", ".join(unknown))
        if "delta" not in data:
            raise ValueError("collapse.reconnect.delta is required")
        kw = dict(data)
        if kw.get("tau_end") is None:
            kw.pop("tau_end", None)
        return cls(**kw)

    def to_dict(self) -> dict:
        out = asdict(self)
        if not math.isfinite(out["tau_end"]):
            out["tau_end"] = None
        return out

    def continues_cut(self, recorded) -> bool:
        """Whether a run with these settings may continue a state written after a cut made with
        the ``recorded`` settings (the ``settings`` of its cut record, as JSON): equal in every key
        but those of ``CONTINUATION_KEYS``, which decide only where the run after the cut ends."""
        if not isinstance(recorded, dict):
            return False

        def cut_keys(d):
            return {k: v for k, v in d.items() if k not in CONTINUATION_KEYS}

        return cut_keys(json.loads(json.dumps(self.to_dict()))) == cut_keys(recorded)


def tau_end_reached(tau: float, tau_end: float) -> bool:
    """Whether the time after the cut ``tau`` has reached ``tau_end``, to round-off: where the
    collapse driver stops a run after a cut (``tau-end``), and where no continuation goes on."""
    return tau >= tau_end - max(1e-14, 8 * math.ulp(tau_end))


def split_collapse_block(block: dict) -> tuple[dict, dict | None, bool, tuple[float, ...]]:
    """Separate the reconnection keys from a ``collapse`` block.

    Returns the block for :class:`~faraday_singularity.driver.CollapseSettings` (unchanged apart
    from the three keys), the ``reconnect`` block or ``None``, ``keep_stop_state``, and the cut
    radii of ``landing`` (largest first; empty without the key), for each of which the run keeps a
    landing state.
    """
    rest = dict(block)
    recon = rest.pop("reconnect", None)
    keep = rest.pop("keep_stop_state", False)
    landing = rest.pop("landing", [])
    if not isinstance(keep, bool):
        raise ValueError("collapse.keep_stop_state must be true or false")
    if recon is not None and not isinstance(recon, dict):
        raise ValueError("collapse.reconnect must be an object")
    if not isinstance(landing, list) or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            or not x > 0 for x in landing):
        raise ValueError("collapse.landing must be a list of positive cut radii")
    if len(set(landing)) != len(landing):
        raise ValueError("collapse.landing: a cut radius is repeated")
    return rest, recon, keep, tuple(sorted((float(x) for x in landing), reverse=True))


# --- the neck -----------------------------------------------------------------------------------


def neck_radius(r, z) -> tuple[float, float, int]:
    """``(r_n, z_n, index)`` of the narrowest interior local minimum of ``r`` along the surface, as
    the collapse driver measures it; NaN without a neck. Necks with nearly equal radii can sit side
    by side; this is the global minimum among them."""
    nk = necks(Profile(np.asarray(r, dtype=float), np.asarray(z, dtype=float)))
    if not nk:
        return math.nan, math.nan, -1
    return nk[0]


def chain_neck(points, end_types, skip: int = 2) -> tuple[float, float, int]:
    """:func:`neck_radius` of one chain: ``(r, z, index)`` of its narrowest interior local minimum
    of ``r``, ignoring ``skip`` nodes at each end on the axis (a chain from the axis to the axis
    has two); NaN without one."""
    P = np.asarray(points, dtype=float)
    last = len(P) - 1 - (skip if end_types[1] == AXIS else 0)
    nk = [x for x in necks(Profile(P[:, 0], P[:, 1]), skip=skip) if x[2] <= last]
    return nk[0] if nk else (math.nan, math.nan, -1)


def chain_necks(points, end_types, skip: int = 2) -> list[dict]:
    """Every interior local minimum of ``r`` along one chain (as :func:`chain_neck`), narrowest
    first, with its depth: ``shoulder``, the lower of the largest ``r`` on either side of it along
    the chain, and ``depth = 1 - r/shoulder``;
    ``side`` is ``gas`` where the gas lies at smaller ``r`` (the chain runs upwards there, with the
    liquid on its right, as every chain is ordered) and ``liquid`` on a jet or thread of liquid."""
    P = np.asarray(points, dtype=float)
    r = P[:, 0]
    last = len(P) - 1 - (skip if end_types[1] == AXIS else 0)
    out = []
    for r_n, z_n, i in necks(Profile(r, P[:, 1]), skip=skip):
        if i > last:
            continue
        shoulder = min(float(r[:i].max()), float(r[i + 1:].max()))  # 1 <= i <= len - 2
        dz = float(P[min(i + 1, len(P) - 1), 1] - P[i - 1, 1])
        out.append({"r": r_n, "z": z_n, "index": i, "shoulder": shoulder,
                    "depth": 1.0 - r_n / shoulder if shoulder > 0.0 else 0.0,
                    "side": "gas" if dz > 0.0 else "liquid"})
    return out


# --- chains -------------------------------------------------------------------------------------


@dataclass
class Chain:
    """One interface polyline of the liquid boundary and the kind of its two ends."""
    points: np.ndarray  # (M, 2): r, z
    end_types: tuple[str, str]

    @property
    def is_open(self) -> bool:
        return FIXED in self.end_types

    def axis_ends(self) -> list[tuple[int, float]]:
        """``(end index 0 or -1, z)`` of the ends on the axis."""
        return [(k, float(self.points[k, 1])) for k in (0, -1) if self.end_types[k] == AXIS]


def end_type(point, radius: float, tol: float) -> str:
    """``"axis"`` for a point on ``r = 0``, ``"fixed"`` for one on the wall ``r = radius``."""
    x = float(point[0])
    if abs(x) <= tol:
        return AXIS
    if abs(x - radius) <= tol:
        return FIXED
    raise RuntimeError(f"an interface chain ends at r = {x:.12g}, on neither the axis nor the wall")


def orient_chain(points, radius: float, tol: float) -> Chain:
    """A chain with snapped ends, the open chain from the axis to the wall, a closed one upwards.

    The axis ends are put on ``r = 0`` exactly and the wall end on ``r = radius``, as the remesh
    of a single free surface does."""
    P = np.array(points, dtype=float).reshape(-1, 2)
    if len(P) < 2:
        raise RuntimeError("an interface chain needs at least two points")
    ends = (end_type(P[0], radius, tol), end_type(P[-1], radius, tol))
    if ends == (FIXED, FIXED):
        raise RuntimeError("an interface chain runs from the wall to the wall")
    if ends == (FIXED, AXIS) or (ends == (AXIS, AXIS) and P[0, 1] > P[-1, 1]):
        P, ends = P[::-1].copy(), (ends[1], ends[0])
    for k in (0, -1):
        P[k, 0] = 0.0 if ends[k] == AXIS else radius
    return Chain(P, ends)


def liquid_axis_spans(axis_tips: Sequence[float]) -> list[tuple[float, float]]:
    """The pieces of the axis inside the liquid, from the heights of the interface ends on it.

    The axis is liquid from the bottom ``z = 0`` up to the lowest interface end on it, and every
    further end toggles between gas and liquid: before the cut ``[0, z_tip]``; after it ``[0, z_bb]``
    below the bubble and the bridge ``[z_bt, z_cb]`` between the bubble's top and the cavity's
    bottom. The count of ends must be odd (one from the open chain, two from each closed one)."""
    tips = sorted(float(z) for z in axis_tips)
    if len(tips) % 2 != 1:
        raise RuntimeError(f"{len(tips)} interface ends on the axis; an odd number is needed")
    if tips[0] <= 0.0:
        raise RuntimeError("an interface end lies on the bottom of the container")
    spans = [(0.0, tips[0])] + [(tips[i], tips[i + 1]) for i in range(1, len(tips) - 1, 2)]
    for a, b in spans:
        if not b > a:
            raise RuntimeError(f"an empty liquid span of the axis between z = {a!r} and {b!r}")
    return spans


def chains_disjoint(chains: Sequence[Chain]) -> float:
    """Smallest distance between two different chains; raises if two chains touch or cross."""
    from shapely.geometry import LineString

    lines = [LineString(c.points) for c in chains]
    best = math.inf
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            d = float(lines[i].distance(lines[j]))
            if d <= 0.0:
                raise RuntimeError(f"interface chains {i} and {j} touch or cross")
            best = min(best, d)
    return best


def check_chains(chains: Sequence[Chain]) -> float:
    """Each chain simple (control polygon and sampled spline), all chains disjoint, exactly one
    open chain. Returns the smallest distance between chains (``inf`` for one chain)."""
    if sum(c.is_open for c in chains) != 1:
        raise RuntimeError("the free surface must have exactly one chain that reaches the wall")
    for c in chains:
        check_simple_interface(c.points)
    return chains_disjoint(chains) if len(chains) > 1 else math.inf


def closed_section(points, end_types, reservoir_depth: float | None) -> np.ndarray:
    """The half-section a chain bounds, closed as pyoomph's planner closes it: an axis end needs no
    closure; a wall end is closed up the wall by ``reservoir_depth`` and across to the axis."""
    P = np.asarray(points, dtype=float)
    extra = []
    if end_types[1] == FIXED:
        h = reservoir_depth or 0.0
        if h:
            extra.append((P[-1, 0], P[-1, 1] + h))
            extra.append((0.0, P[-1, 1] + h))
        else:
            extra.append((0.0, P[-1, 1]))
    if end_types[0] == FIXED:
        raise RuntimeError("closed_section expects the wall end last")
    return np.vstack([P, np.array(extra, dtype=float)]) if extra else P


def revolved_volume(section) -> float:
    """Volume of the body of revolution of a closed polygonal half-section."""
    p = np.asarray(section, dtype=float)
    r0, z0 = p[:, 0], p[:, 1]
    r1, z1 = np.roll(r0, -1), np.roll(z0, -1)
    return float(abs(np.sum((r0 * r0 + r0 * r1 + r1 * r1) / 3.0 * (z1 - z0)) * math.pi))


def chain_volumes(chains, reservoir_depth: float) -> list[float]:
    """Gas volume each chain bounds (the open chain closed through the gas above the surface)."""
    return [revolved_volume(closed_section(c.points, c.end_types, reservoir_depth)) for c in chains]


def gas_volume_in_box(chains, reservoir_depth: float, box: Sequence[float]) -> float:
    """Gas volume the chains bound within ``r <= box[0]``, ``box[1] <= z <= box[2]``.

    Before and after a cut, with the chains through the free-surface nodes of the two meshes, the
    difference is the gas the cut turned into liquid there, without the drift of the remesh over
    the rest of the free surface (the rebuilt spline sags differently from the old elements). The
    polygons through the nodes (with the midside nodes of the quadratic elements) bound the same
    region as the tessellated elements."""
    from shapely.geometry import Polygon
    from shapely.geometry import box as shapely_box

    clip = shapely_box(0.0, float(box[1]), float(box[0]), float(box[2]))
    total = 0.0
    for c in chains:
        section = Polygon(closed_section(c.points, c.end_types, reservoir_depth))
        part = section.intersection(clip)
        for poly in getattr(part, "geoms", [part]):
            if poly.geom_type != "Polygon" or poly.is_empty:
                continue
            total += revolved_volume(np.asarray(poly.exterior.coords)[:-1])
            total -= sum(revolved_volume(np.asarray(h.coords)[:-1]) for h in poly.interiors)
    return total


# --- caps ---------------------------------------------------------------------------------------


@dataclass
class CapEdit:
    """One replaced cap: where the arc starts, its shape, and what it replaced."""
    chain: int
    end: int  # 0 or -1: the end of the chain where the fresh tip is
    r_s: float  # splice radius
    z_s: float
    beta_deg: float  # angle of the old interface to the axis at the splice, narrowing to the tip
    radius: float  # arc radius
    z_tip: float
    points: int  # arc points, splice excluded, tip included
    replaced_fresh: int  # planner points dropped
    replaced_survivors: int  # surviving old points dropped between the planner cap and the splice
    splice_set_by: str  # "planner-window" (first survivor beyond it) or "delta-floor"
    backtrack: float  # fold depth of the planner cap, in delta (0: monotone)
    min_angle: float  # smallest interior angle of the new cap, degrees
    splice_angle: float  # interior angle at the splice, degrees (180: tangent)


def tip_size(delta: float, settings: ReconnectSettings) -> float:
    """Element size at the tips on the axis after a cut at radius ``delta``: ``tip_size * delta``, a
    fixed fraction of the cut radius (at most a quarter of it, which a cap needs for gmsh to follow
    its curvature without inverting curved elements), so that the meshes of a study in ``delta``
    are similar whatever the arc radius or the focus of the landing state."""
    return settings.tip_size * delta


def fresh_tip_run(origin: np.ndarray, end: int) -> int:
    """Number of planner points (``origin < 0``) at one end of a chain."""
    o = np.asarray(origin)
    seq = o if end == 0 else o[::-1]
    n = int(np.argmax(seq >= 0)) if np.any(seq >= 0) else len(seq)
    return n


def cap_backtrack(points_from_tip) -> float:
    """How far a polyline running from an axis tip comes back towards the tip axially: zero for a
    cap whose distance from the tip grows monotonically, the fold depth otherwise."""
    Q = np.asarray(points_from_tip, dtype=float)
    dz = np.abs(Q[:, 1] - Q[0, 1])
    return float((np.maximum.accumulate(dz) - dz).max()) if len(Q) else 0.0


def interior_angles(points) -> np.ndarray:
    """Interior angle (degrees) at each inner vertex of a polyline; 180 for a straight run."""
    P = np.asarray(points, dtype=float)
    if len(P) < 3:
        return np.zeros(0)
    a = P[:-2] - P[1:-1]
    b = P[2:] - P[1:-1]
    c = np.einsum("ij,ij->i", a, b) / np.maximum(np.hypot(*a.T) * np.hypot(*b.T), 1e-300)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _tangent_towards_tip(P: np.ndarray, k: int, step: int, length: float) -> np.ndarray:
    """Unit tangent of the old interface at ``P[k]``, pointing towards the tip.

    ``r`` and ``z`` are fitted by quadratics in the arc length over the nodes within ``length`` of
    ``P[k]`` away from the tip (``step`` is the index direction away from it), and differentiated at
    ``P[k]``."""
    idx = [k]
    s = [0.0]
    j = k
    while 0 <= j + step < len(P) and (s[-1] < length or len(idx) < 4):
        s.append(s[-1] + float(np.hypot(*(P[j + step] - P[j]))))
        j += step
        idx.append(j)
    Q = P[idx]
    S = np.asarray(s)
    deg = 2 if len(idx) >= 4 else 1
    dr = np.polyder(np.polyfit(S, Q[:, 0], deg))
    dz = np.polyder(np.polyfit(S, Q[:, 1], deg))
    d = -np.array([np.polyval(dr, 0.0), np.polyval(dz, 0.0)])  # s grows away from the tip
    return d / np.hypot(*d)


def tangent_arc(P_s: np.ndarray, t: np.ndarray, n_points: int) -> tuple[np.ndarray, float, float]:
    """The circular arc from ``P_s``, tangent to ``t`` there, that meets the axis at a right angle.

    Its centre lies on the axis, at radius ``r_s/|t_z|`` from ``P_s``. Returns the arc points from
    the splice (excluded) to the tip (included, on ``r = 0`` exactly), the radius and the tip
    height."""
    r_s, z_s = float(P_s[0]), float(P_s[1])
    tr, tz = float(t[0]), float(t[1])
    if abs(tz) < 0.2:
        raise RuntimeError(f"the interface at the splice is too oblique to the axis for a cap "
                           f"(tangent ({tr:.3g}, {tz:.3g}))")
    s = 1.0 if tz > 0 else -1.0
    n = np.array([-tz, tr]) if tz > 0 else np.array([tz, -tr])  # normal towards the axis
    R = r_s / -n[0]
    zc = z_s + R * n[1]
    phi_s = math.atan2(z_s - zc, r_s)
    phi_t = s * 0.5 * math.pi
    phi = np.linspace(phi_s, phi_t, n_points + 1)[1:]
    arc = np.column_stack([R * np.cos(phi), zc + R * np.sin(phi)])
    arc[-1] = (0.0, zc + s * R)
    return arc, R, float(arc[-1, 1])


def replace_caps(new_chains, delta: float, settings: ReconnectSettings) -> list[CapEdit]:
    """Replace every fresh axis tip of a planner's chains by a tangent arc, in place.

    ``new_chains`` are the planner's ``NewChain`` objects (``points``, ``sizes``, ``zeta``,
    ``origin``, ``end_types``). For each end on the axis made by the planner (``origin < 0`` there),
    the planner's points are dropped, and so are surviving old points up to the first one, walking
    away from the tip, whose radius is at least ``delta``: that node is the splice, normally the
    first survivor beyond the planner's window (``splice_set_by`` ``planner-window``), and further
    out only where survivors narrower than ``delta`` were dropped (``delta-floor``). The arc (:func:`tangent_arc`) is tangent there to the old interface (fitted over
    ``cap_tangent_fit`` splice radii) and ends on the axis. Its chart (``zeta``) is a strictly
    increasing ramp from the splice to the planner's tip value, so that the plan's chart still
    covers the cap. Point spacing is set so that no cap segment spans more chart than five times the
    median segment of all chains, which pyoomph's projection onto the plan would read as a jump.
    """
    s = settings
    all_dz = np.concatenate([np.abs(np.diff(np.asarray(c.zeta, dtype=float))) for c in new_chains])
    median_dz = max(float(np.median(all_dz)), 1e-300)
    edits: list[CapEdit] = []
    for ci, nc in enumerate(new_chains):
        for end in (0, -1):
            if nc.end_types[end] != AXIS or int(nc.origin[end]) >= 0:
                continue
            P = np.asarray(nc.points, dtype=float)
            Z = np.asarray(nc.zeta, dtype=float)
            S = np.asarray(nc.sizes, dtype=float)
            O = np.asarray(nc.origin, dtype=int)
            nf = fresh_tip_run(O, end)
            if nf >= len(P) - 3:
                raise RuntimeError("a planner chain has too few surviving old points for a cap")
            planner_cap = P[:nf + 1] if end == 0 else P[::-1][:nf + 1]
            backtrack = cap_backtrack(planner_cap) / delta if nf else 0.0
            # walk away from the tip over survivors to the splice
            step = 1 if end == 0 else -1
            k = nf if end == 0 else len(P) - 1 - nf
            dropped = 0
            while P[k, 0] < delta:
                if O[k] < 0:
                    raise RuntimeError("planner points between survivors next to a fresh cap")
                k += step
                dropped += 1
                if not (0 <= k < len(P)) or dropped > len(P) // 2:
                    raise RuntimeError(f"no surviving node reaches the splice radius {delta:.3g}")
            if O[k] < 0:
                raise RuntimeError("the splice of a cap is not an old interface node")
            t = _tangent_towards_tip(P, k, step, s.cap_tangent_fit * float(P[k, 0]))
            beta = math.degrees(math.atan2(-t[0], abs(t[1])))  # > 0: narrowing towards the tip
            z_tip_old = Z[0] if end == 0 else Z[-1]
            z_splice = Z[k]
            # enough points for an arc of at least cap_min_points with small chart steps
            n_pts = s.cap_min_points
            arc, R, z_tip = tangent_arc(P[k], t, n_pts)
            arc_len = R * abs(0.5 * math.pi - math.radians(beta))
            h_tip = tip_size(delta, s)
            n_pts = max(n_pts, int(math.ceil(arc_len / h_tip)),
                        int(math.ceil(abs(z_tip_old - z_splice) / (5.0 * median_dz))))
            arc, R, z_tip = tangent_arc(P[k], t, n_pts)
            # chart ramp along the arc, from the splice value to the planner's tip value
            seg = np.hypot(*np.diff(np.vstack([P[k], arc]), axis=0).T)
            frac = np.cumsum(seg) / seg.sum()
            zeta_arc = z_splice + frac * (z_tip_old - z_splice)
            zeta_arc[-1] = z_tip_old
            size_arc = np.full(len(arc), h_tip)
            orig_arc = np.full(len(arc), -1, dtype=int)
            if end == 0:  # chain order: tip first
                keep = slice(k, None)
                P2 = np.vstack([arc[::-1], P[keep]])
                Z2 = np.concatenate([zeta_arc[::-1], Z[keep]])
                S2 = np.concatenate([size_arc[::-1], S[keep]])
                O2 = np.concatenate([orig_arc, O[keep]])
                cap = P2[:len(arc) + 3]
            else:
                keep = slice(None, k + 1)
                P2 = np.vstack([P[keep], arc])
                Z2 = np.concatenate([Z[keep], zeta_arc])
                S2 = np.concatenate([S[keep], size_arc])
                O2 = np.concatenate([O[keep], orig_arc])
                cap = P2[::-1][:len(arc) + 3]
            if np.any(np.diff(Z2) <= 0.0):
                raise RuntimeError("the chart of a replaced cap is not strictly increasing")
            angles = interior_angles(cap)
            back_new = cap_backtrack(cap)
            if back_new > 1e-12 * delta:
                raise RuntimeError("a replaced cap is not monotone in z from its tip")
            if angles.min() < s.min_cap_angle:
                raise RuntimeError(f"a replaced cap has an interior angle of {angles.min():.3g} deg")
            splice_angle = float(angles[len(arc) - 1]) if len(angles) >= len(arc) else math.nan
            nc.points, nc.zeta, nc.sizes, nc.origin = P2, Z2, S2, O2
            edits.append(CapEdit(chain=ci, end=end, r_s=float(P[k, 0]), z_s=float(P[k, 1]),
                                 beta_deg=beta, radius=R, z_tip=z_tip, points=len(arc),
                                 replaced_fresh=nf, replaced_survivors=dropped,
                                 splice_set_by="delta-floor" if dropped else "planner-window",
                                 backtrack=backtrack, min_angle=float(angles.min()),
                                 splice_angle=splice_angle))
    return edits


def plan_chains(new_chains) -> list[Chain]:
    return [Chain(np.asarray(nc.points, dtype=float), tuple(nc.end_types)) for nc in new_chains]


@dataclass
class CutGeometry:
    """Geometry of the chains after a cut, as the mesh will be built from them."""
    z_bubble_bottom: float
    z_bubble_top: float
    z_cavity_tip: float
    bridge: float  # z_cavity_tip - z_bubble_top
    chain_gap: float  # smallest distance between the two chains
    spans: list = field(default_factory=list)  # liquid axis spans
    gas_before: float = math.nan  # gas volume the planner closed before the cut
    gas_after: list = field(default_factory=list)  # per chain, after the edit
    bubble_volume: float = math.nan
    removed_gas: float = math.nan  # gas_before - sum(gas_after): the liquid gains this

    def to_dict(self) -> dict:
        return asdict(self)


def in_cut_radii(edits: Sequence[CapEdit], geometry: CutGeometry, rmin: float,
                 settings: ReconnectSettings) -> dict:
    """The cut geometry in units of the cut radius ``rmin`` and what set its splices.

    ``splice_set_by`` gives, for each cap, ``planner-window`` when its splice is the first
    survivor beyond the planner's window (``cap_window_factor``), and ``delta-floor`` when survivors
    narrower than the cut radius were dropped first."""
    return {"cut_radius": rmin, "cap_window_factor": settings.cap_window_factor,
            "r_s": [e.r_s / rmin for e in edits], "arc_radius": [e.radius / rmin for e in edits],
            "replaced_survivors": [e.replaced_survivors for e in edits],
            "splice_set_by": [e.splice_set_by for e in edits],
            "bridge": geometry.bridge / rmin, "removed_gas": geometry.removed_gas / rmin**3}


def cut_geometry(chains: Sequence[Chain], reservoir_depth: float,
                 gas_before: float = math.nan) -> CutGeometry:
    """Check the chains after a pinch (one closed, one open) and describe them."""
    if len(chains) != 2:
        raise RuntimeError(f"a pinch should leave two interface chains, not {len(chains)}")
    gap = check_chains(chains)
    bubble = next(c for c in chains if not c.is_open)
    cavity = next(c for c in chains if c.is_open)
    tips = [z for c in chains for _, z in c.axis_ends()]
    spans = liquid_axis_spans(tips)
    vols = chain_volumes([bubble, cavity], reservoir_depth)
    zb0, zb1 = sorted([float(bubble.points[0, 1]), float(bubble.points[-1, 1])])
    zc = cavity.axis_ends()[0][1]
    if not zc > zb1:
        raise RuntimeError("the cavity tip does not lie above the bubble")
    return CutGeometry(z_bubble_bottom=zb0, z_bubble_top=zb1, z_cavity_tip=zc, bridge=zc - zb1,
                       chain_gap=gap, spans=spans, gas_before=gas_before, gas_after=vols,
                       bubble_volume=vols[0], removed_gas=gas_before - sum(vols))


# --- planner on a stored profile -----------------------------------------------------------------


def profile_chain(r, z, reservoir_depth: float):
    """A pyoomph ``InterfaceChain`` from a profile ordered from the axis to the wall, as the
    reconnection handler builds it from the interface nodes (sizes: twice the node spacing)."""
    from pyoomph.meshes.axisymm_topology import InterfaceChain

    pts = np.column_stack([np.asarray(r, dtype=float), np.asarray(z, dtype=float)])
    seg = np.hypot(*np.diff(pts, axis=0).T)
    sp = np.r_[seg[0], 0.5 * (seg[:-1] + seg[1:]), seg[-1]]
    return InterfaceChain(points=pts, sizes=2.0 * sp, end_types=(AXIS, FIXED),
                          reservoir=(None, reservoir_depth))


def plan_on_profile(r, z, rmin: float, settings: ReconnectSettings) -> tuple[str, Any]:
    """Run the planner on one stored profile at cut radius ``rmin``: ``(outcome, plan or message)``
    with outcome ``"none"``, ``"deferral"`` (waist not yet separable), ``"unplannable"``,
    ``"refused"`` (a plain ``RuntimeError``, e.g. a self-intersecting cap) or ``"plan"``."""
    from pyoomph.meshes import axisymm_topology as topo

    chain = profile_chain(r, z, settings.reservoir_depth)
    try:
        plan = topo.detect_and_plan([chain], rmin, None, cap_window_factor=settings.cap_window_factor,
                                    volume_conservation=False)
    except topo.WaistNotYetSeparable as e:
        return "deferral", str(e)
    except topo.InterfaceStateNotPlannable as e:
        return "unplannable", str(e)
    except RuntimeError as e:
        return "refused", str(e)
    return ("none", None) if plan is None else ("plan", plan)


# --- after the cut ---------------------------------------------------------------------------------


def fold_contact(arc: float, h_elem: float) -> str:
    """``fold`` for a fold pair whose arc spans at least ``FOLD_RESOLVED_ELEMENTS`` local element
    lengths, two parts of the resolved surface approaching across a gap; ``kink`` for a closer one,
    a corner at the scale of the mesh (:func:`~faraday_singularity.selfcontact.classify_contact`)."""
    return "fold" if arc >= FOLD_RESOLVED_ELEMENTS * h_elem else "kink"


def spline_contact_stop(named_chains) -> dict | None:
    """The ``self-contact-approach`` stop (trigger ``spline-contact``) of the first of the
    ``(name, chain)`` pairs whose spline through its nodes contacts itself, the test a remesh makes
    (:func:`check_chains`), with what met (:func:`~faraday_singularity.selfcontact.
    classify_contact`); None if every spline is simple. A chain whose nodes have crossed raises
    :class:`~faraday_singularity.selfcontact.InterfaceSelfIntersection`."""
    for name, c in named_chains:
        contact = first_contact(c.points)
        if contact is None:
            continue
        i, i1, j, j1 = contact["nodes"]
        if contact["what"] == "polygon":
            raise InterfaceSelfIntersection(
                f"the {name} chain's free surface crosses itself between nodes {i}-{i1} and {j}-{j1} "
                f"at (r, z) = ({contact['at'][0]:.9g}, {contact['at'][1]:.9g}) after the cut")
        return {"reason": "self-contact-approach", "trigger": "spline-contact", "chain": name,
                **classify_contact(c.points, contact), "nodes": [i, i1, j, j1],
                "contact_r": contact["at"][0], "contact_z": contact["at"][1],
                "crossings": contact["count"]}
    return None


def post_cut_stops(chains: Sequence[Chain], cut: CutGeometry, settings: ReconnectSettings,
                   h_min: float, *, neck_elements: float = 3.0, fold_ratio: float = 4.0,
                   fold_stop: float = 1e-3, fold_stop_elements: float = 3.0,
                   fold_search: float = 4.0, apexes: dict | None = None,
                   guard: dict | None = None, events=None, invalid=None) -> dict | None:
    """The stop a state after the cut calls for, if any.

    The mesh cannot follow a second change of topology: the bridge closing again (the bubble and
    cavity tips, on different chains, meeting across it), the chains touching elsewhere, the
    bubble's two tips meeting (it would become a ring), a new neck on either chain narrower than
    ``max(delta, neck_elements h_min)`` (``secondary-pinch-approach``: the thread behind a fresh cap
    can pinch again; a neck counts only where it is ``neck_depth`` below the lower of the widest
    points on either side of it along the chain, or narrower than ``neck_elements h_min`` whatever
    its depth: :func:`chain_necks`), a closing fold on either chain (with the driver's fold limits: ``fold_gap``
    with ``fold_ratio``, and a gap below ``max(fold_stop, fold_stop_elements h_elem)``): the
    ``annular-contact-approach`` (trigger ``fold-gap``) when the fold pair straddles the base of a
    jet, the apex of the groove around it (``apexes``: per chain, ``{"tip", "index", ...}`` with
    ``index`` the node of the apex along the chain), with that jet, its apex and ``R_apex``, and
    otherwise a ``self-contact-approach`` (trigger ``fold-gap``); a chain whose spline through its
    nodes, the curve every remesh is built on, contacts itself while its nodes do not
    (``self-contact-approach``, trigger ``spline-contact``, on either chain), or the bubble draining
    away under its constant pressure. The fold stops and the spline stop say what met (``contact``,
    :func:`~faraday_singularity.selfcontact.classify_contact`): ``fold`` when the two sides are at
    least ``FOLD_RESOLVED_ELEMENTS`` elements apart along the surface, a fold of the resolved
    surface, ``kink`` when they are closer, a contact at the scale of the mesh. A chain whose nodes
    have crossed (its control polygon) raises
    :class:`~faraday_singularity.selfcontact.InterfaceSelfIntersection`: that state is not valid.

    With ``guard`` (``fresh``: whether the mesh was built since its last step; ``h_floor`` and
    ``refine_ratio``, the driver's), the quadratic elements of both chains are read last
    (:func:`interface_guard`): ``interface-unresolved`` when the mesh does not represent the
    interface, an element bent or skewed beyond what a rebuild can be built on, where none of the
    stops above, of the geometry of the interface, applies; a spline that loops stays a
    ``spline-contact``.

    The spline test is the remesh's own (:func:`check_chains`, through
    :func:`~faraday_singularity.selfcontact.first_contact`, on the nodes the rebuild's spline runs
    through: every node, or the vertices with ``rebuild_spline`` ``vertices``), so a state that
    passes these stops passes the check of a rebuild from its nodes: the driver makes them before
    every rebuild after the cut, and a rebuild never meets a self-intersecting spline.

    The collapse driver's own neck stop covers only the open chain
    (``FaradayProblem.interface_arrays``); after a cut its fold stop is this one, on both chains,
    and the only one on the bubble, which before the cut was part of the open chain. Each chain is
    examined on its own, so the bridge between the two is never a fold.

    With ``events``, the events block (``events.EventSettings``), the stops are those of the events
    branch instead, in its own order and with its own limits (``events.post_cut_events``), and
    ``invalid`` is the check of a fresh mesh's curved elements the driver makes among the stops of
    an invalid state; without it the stops are those above, unchanged."""
    if events is not None:
        from .events import post_cut_events

        return post_cut_events(chains, cut, settings, h_min, events, neck_elements=neck_elements,
                               fold_ratio=fold_ratio, fold_stop=fold_stop,
                               fold_stop_elements=fold_stop_elements, fold_search=fold_search,
                               apexes=apexes, guard=guard, invalid=invalid)
    if invalid is not None:
        raise TypeError("post_cut_stops: invalid is a check of the events branch (events)")
    if len(chains) != 2:
        return {"reason": "topology-changed", "chains": len(chains)}
    bubble = next(c for c in chains if not c.is_open)
    cavity = next(c for c in chains if c.is_open)
    zb = sorted([float(bubble.points[0, 1]), float(bubble.points[-1, 1])])
    zc = cavity.axis_ends()[0][1]
    bridge = zc - zb[1]
    limit = max(settings.bridge_stop * cut.bridge, 3.0 * h_min)
    if bridge < limit:
        return {"reason": "bridge-closing", "bridge": bridge, "limit": limit}
    gap = chains_disjoint(chains)
    if gap < 3.0 * h_min:
        return {"reason": "chain-contact-approach", "chain_gap": gap, "limit": 3.0 * h_min}
    length = zb[1] - zb[0]
    if length < max(settings.delta, 3.0 * h_min):
        return {"reason": "bubble-jets-approach", "bubble_length": length}
    neck_limit = max(settings.delta, neck_elements * h_min)
    named = (("bubble", bubble), ("open", cavity))
    for name, c in named:
        for nk in chain_necks(c.points, c.end_types):
            if nk["r"] < neck_limit and (nk["depth"] >= settings.neck_depth
                                          or nk["r"] < neck_elements * h_min):
                return {"reason": "secondary-pinch-approach", "trigger": "neck-radius",
                        "chain": name, "r_neck": nk["r"], "z_neck": nk["z"],
                        "neck_limit": neck_limit, "neck_shoulder": nk["shoulder"],
                        "neck_depth": nk["depth"], "neck_side": nk["side"]}
    folds = []
    for name, c in named:
        fold = fold_gap(c.points[:, 0], c.points[:, 1], ratio=fold_ratio, stop=fold_stop,
                        stop_elements=fold_stop_elements, search_factor=fold_search)
        if fold.crossed:
            what = "bubble" if name == "bubble" else "open chain"
            raise InterfaceSelfIntersection(
                f"the {what}'s free surface crosses itself after the cut")
        fold_limit = max(fold_stop, fold_stop_elements * fold.h_elem)
        if math.isfinite(fold.gap) and fold.gap < fold_limit:
            folds.append((fold.gap / fold_limit, name, fold, fold_limit))
    if folds:  # the fold closest to its limit, on either chain
        _, name, fold, fold_limit = min(folds, key=lambda x: x[0])
        rec = {"trigger": "fold-gap", "chain": name, "contact": fold_contact(fold.arc, fold.h_elem),
               "fold_gap": fold.gap, "fold_r": fold.r, "fold_z": fold.z, "fold_arc": fold.arc,
               "fold_h_elem": fold.h_elem, "fold_limit": fold_limit, "fold_pair": list(fold.pair)}
        a, b = fold.pair
        jet = next((x for x in (apexes or {}).get(name, []) if a < int(x["index"]) < b), None)
        if jet is None:
            return {"reason": "self-contact-approach", **rec}
        return {"reason": "annular-contact-approach", **rec, "jet": jet["tip"],
                "apex_index": int(jet["index"]), "apex_r": jet.get("r", math.nan),
                "apex_z": jet.get("z", math.nan), "R_apex": jet.get("R_apex", math.nan),
                "groove_angle": jet.get("groove_angle", math.nan)}
    stride = 2 if settings.rebuild_spline == "vertices" else 1
    stop = spline_contact_stop([(name, Chain(c.points[::stride], c.end_types))
                                if stride > 1 and len(c.points) % 2 == 1 else (name, c)
                                for name, c in named])
    if stop is not None:
        return stop
    vb = revolved_volume(bubble.points)
    if vb < settings.bubble_stop * cut.bubble_volume:
        return {"reason": "bubble-vanishing", "bubble_volume": vb}
    if guard is not None:
        return interface_guard({name: edge_reading(c.points) for name, c in named}, settings,
                               fresh=bool(guard["fresh"]), h_floor=float(guard["h_floor"]),
                               refine_ratio=float(guard["refine_ratio"]))["stop"]
    return None


def cut_size_field(r, z, axis_tips: Sequence[float], sizing: dict) -> np.ndarray:
    """The element size the mesh sizing of a cut asks for at the points ``(r, z)``.

    ``sizing`` is the cut's ``mesh.CutSizing`` as its record holds it (``h_tip``, ``h_bridge``,
    ``grading``) and ``axis_tips`` the heights of the interface ends on the axis. As the mesh
    template builds every mesh after the cut (``mesh.FaradayCylinderMesh``): the finer of ``h_tip +
    grading d_tip``, with ``d_tip`` the distance to the nearest end on the axis, and ``h_bridge +
    grading d_bridge``, with ``d_bridge`` the distance to the bridge, the liquid spans of the axis
    between the ends (:func:`liquid_axis_spans`). The template's cap at ``h_max`` and its sizing
    from curvature are left out: this is where the cut's attractors make the mesh finer than a
    focus."""
    r, z = np.asarray(r, dtype=float), np.asarray(z, dtype=float)
    tips = sorted(float(x) for x in axis_tips)
    g = float(sizing["grading"])
    d_tip = np.min(np.hypot(r[:, None], z[:, None] - np.asarray(tips)[None, :]), axis=1)
    h = float(sizing["h_tip"]) + g * d_tip
    for a, b in liquid_axis_spans(tips)[1:]:
        dz = np.maximum(0.0, np.maximum(a - z, z - b))
        h = np.minimum(h, float(sizing["h_bridge"]) + g * np.hypot(r, dz))
    return h


FRESH_TIPS = ("cavity_tip", "bubble_top")  # the tips on the axis the cut makes


def kept_region(landing_focus: dict | None, cut_focus_size: float, z_bubble_bottom: float,
                z_bubble_top: float) -> dict | None:
    """The resolution the bubble's bottom keeps from the landing state: the landing's cavity bottom,
    with its jet and base, was meshed at the landing focus (``size`` within ``radius`` of the axis).
    It keeps the finer of that size and the cut's focus size within that radius of its tip, but
    within its half of the bubble, so that it never reaches the bubble's fresh top; None without a
    landing focus. Recorded by the cut (``focus.kept``) and kept by every later rebuild."""
    if not landing_focus:
        return None
    radius = min(float(landing_focus["radius"]), 0.5 * (z_bubble_top - z_bubble_bottom))
    return {"name": "bubble_bottom", "size": min(float(landing_focus["size"]), cut_focus_size),
            "radius": radius, "landing_size": float(landing_focus["size"]),
            "landing_radius": float(landing_focus["radius"])}


def feature_regions(segs, cavity: dict, tips: dict, necks: dict, cut: dict,
                    settings: ReconnectSettings, *, h_floor: float, h_axis: float,
                    elements_per_scale: float, focus_extent: float,
                    apexes: dict | None = None, bubble_bottom_sizing: str = "kept",
                    bubble_bottom_size: float | None = None, neck_floor: float | None = None,
                    neck_elements_per_scale: float | None = None) -> list[dict]:
    """The features of both chains after a cut, each to be meshed at its own size (the regions of
    the focus, ``mesh.FocusRegion``): ``name``, ``target`` (element size), centre ``r``, ``z``,
    ``radius``, its scale ``ell`` (``inf`` without one) and ``floor``, the smallest size it may ask
    for.

    * The two tips the cut made, the cavity's and the bubble's top (``FRESH_TIPS``), at their own
      scale ``ell_tip``, the smaller of the tip's radius of curvature (the cavity's apex curvature,
      ``R_bubble_top``) and the radius of the jet that stands on it (its base, where a resolved bump
      stands): ``clip(ell_tip/elements_per_scale, h_floor, h_cut)``, with ``h_cut`` the cut's focus
      size (``focus_size`` cut radii) the coarsest size allowed, not a fixed one. At ``h_cut`` the
      region spans ``focus_radius`` cut radii of the tip and the jet on it (twice the distance from
      the tip to the edge of its base), as the cut sized it; finer, it spans ``focus_extent
      ell_tip`` (at most that), and ``<tip>_cut`` keeps the rest of the cut's region at ``h_cut``,
      so that no rebuild coarsens what the cut resolved.
    * The base of each jet, the apex of the gas groove around it (``jet_base_<tip>``, for each tip
      in ``apexes``: ``r``, ``z`` the base, ``R_apex`` its radius of curvature,
      :func:`jet_base_apex`, ``R_pred`` the radius predicted some steps ahead, the driver's), at
      ``clip(R_pred/elements_per_scale, h_floor, cap)`` within ``max(apex_extent R_apex, 4
      target)`` of the base, ``cap`` the cut's focus size at the tips the cut made and the size
      the bubble's bottom asks for at its bottom, far from the cut; none where that is the cap,
      since the tip's region, which covers the base, then resolves it. Nothing else sizes it: the
      tips' regions end at the tip's scale, and at the cut's focus size the apex of a groove whose
      radius is a fraction of the jet's is met by one or two elements.
    * The bubble's bottom, which the landing state held as the cavity's bottom: at the size its jet
      asks for (``r_j/elements_per_scale``, floored at the landing run's ``h_floor``, as before the
      cut) but never coarser than it had at the landing (``kept``, :func:`kept_region`), over the
      larger of its jet's region and the one the landing resolved, within its half of the bubble.
      With ``bubble_bottom_sizing`` ``frozen`` (the events block's ``post``) the size it may not
      exceed is ``bubble_bottom_size`` instead of the landing's, whatever the landing state: the
      bubbles of a study in ``delta`` are then meshed alike there, far from the cut.
    * Each neck of either chain: ``r_n/elements_per_scale``, floored at the cut's focus size, within
      ``focus_extent r_n`` of it; with the events block's ``post`` at ``neck_elements_per_scale``
      elements per neck radius and floored at ``neck_floor``, so that a neck can be followed below
      the cut radius.

    ``cavity`` describes the open chain's tip (``z_tip``, ``kappa`` the apex curvature, ``r_j``,
    ``rise``, ``bump`` and ``flat_bump`` as the driver measures it), ``tips`` the bubble's two
    (:func:`tip_jet`), ``necks`` the record (:func:`post_cut_record`: ``r_neck_open``,
    ``z_neck_open``, ``r_neck_bubble``, ``z_neck_bubble``, ``R_bubble_top``); ``cut`` is the cut
    record (``rmin``, ``focus.kept``)."""
    rmin = float(cut.get("rmin") or cut["delta"])
    h_cut = settings.focus_size * rmin
    r_cut = settings.focus_radius * rmin
    ell_cap = h_axis * elements_per_scale
    bubble = next(c for c in segs if not c["open"])
    z_bb, z_bt = float(bubble["z"][0]), float(bubble["z"][-1])

    def jet(t: dict) -> tuple[list[float], float]:
        """The scale of the jet on a tip (its base radius, where a resolved bump stands) and the
        reach of the region that covers it."""
        if t.get("bump") and t.get("r_j", 0.0) > 0.0 and not t.get("flat_bump"):
            return [float(t["r_j"])], 2.0 * math.hypot(float(t["r_j"]), max(float(t["rise"]), 0.0))
        return [], 0.0

    def finite(xs):
        return [float(x) for x in xs if math.isfinite(x) and x > 0.0]

    def fresh_tip(name: str, z: float, scales: list[float], reach: float) -> list[dict]:
        ell = min(scales) if scales else math.inf
        target = float(np.clip(ell / elements_per_scale, h_floor, h_cut)) if scales else h_cut
        whole = max(r_cut, reach)
        if target >= h_cut:
            return [{"name": name, "target": h_cut, "r": 0.0, "z": z, "radius": whole, "ell": ell,
                     "floor": h_floor}]
        return [{"name": name, "target": target, "r": 0.0, "z": z,
                 "radius": max(min(whole, focus_extent * ell), 4.0 * target), "ell": ell,
                 "floor": h_floor},
                {"name": name + "_cut", "target": h_cut, "r": 0.0, "z": z, "radius": whole,
                 "ell": math.inf, "floor": h_cut}]

    out = []
    k = cavity.get("kappa", math.nan)
    scales, reach = jet(cavity)
    scales = finite(scales + ([1.0 / abs(k)] if math.isfinite(k) and k != 0.0 else []))
    out += fresh_tip("cavity_tip", float(cavity["z_tip"]), scales, reach)
    scales, reach = jet(tips.get("bubble_top", {}))
    out += fresh_tip("bubble_top", z_bt, finite(scales + [necks.get("R_bubble_top", math.nan)]),
                     reach)
    scales, reach = jet(tips.get("bubble_bottom", {}))
    kept = (cut.get("focus") or {}).get("kept")
    if bubble_bottom_sizing == "frozen":
        # never coarser than the frozen size, over the region the landing resolved, if any
        size = float(bubble_bottom_size)
        ell = min(scales + [ell_cap])
        target = min(float(np.clip(ell / elements_per_scale, h_floor, h_axis)), size)
        radius = max(focus_extent * ell if scales else 0.0, reach,
                     float(kept["radius"]) if kept else 0.0)
        if radius > 0.0:
            out.append({"name": "bubble_bottom", "target": target, "r": 0.0, "z": z_bb,
                        "radius": min(radius, 0.5 * (z_bt - z_bb)),
                        "ell": ell if scales else math.inf, "floor": h_floor, "kept": size})
    elif bubble_bottom_sizing != "kept":
        raise ValueError(f"unknown bubble_bottom_sizing {bubble_bottom_sizing!r}")
    elif scales or kept:
        ell = min(scales + [ell_cap])
        target = float(np.clip(ell / elements_per_scale, h_floor, h_axis))
        radius = max(focus_extent * ell if scales else 0.0, reach)
        if kept:
            target, radius = min(target, float(kept["size"])), max(radius, float(kept["radius"]))
        out.append({"name": "bubble_bottom", "target": target, "r": 0.0, "z": z_bb,
                    "radius": min(radius, 0.5 * (z_bt - z_bb)),
                    "ell": ell if scales else math.inf, "floor": h_floor,
                    "kept": None if not kept else float(kept["size"])})
    # the coarsest size of a jet's base: the cut's focus size near the cut, and at the bubble's
    # bottom, far from it, the size its own region asks for
    bottom = next((g for g in out if g["name"] == "bubble_bottom"), None)
    caps = {"cavity_tip": h_cut, "bubble_top": h_cut,
            "bubble_bottom": bottom["target"] if bottom is not None else h_cut}
    for tip in JET_TIPS:
        a = (apexes or {}).get(tip)
        R = float(a.get("R_apex", math.nan)) if a else math.nan
        if not (math.isfinite(R) and R > 0.0):
            continue
        R_pred = float(a.get("R_pred", R))
        cap = caps[tip]
        target = float(np.clip((R_pred if math.isfinite(R_pred) else R) / elements_per_scale,
                               h_floor, cap))
        if target >= cap:  # resolved by its tip's region, which covers it
            continue
        out.append({"name": f"jet_base_{tip}", "target": target, "r": float(a["r"]),
                    "z": float(a["z"]), "radius": max(settings.apex_extent * R, 4.0 * target),
                    "ell": R, "floor": h_floor})
    neck_eps = elements_per_scale if neck_elements_per_scale is None else neck_elements_per_scale
    neck_min = h_cut if neck_floor is None else neck_floor
    for name, chain in (("neck_open", "open"), ("neck_bubble", "bubble")):
        r_n, z_n = necks.get(f"r_neck_{chain}", math.nan), necks.get(f"z_neck_{chain}", math.nan)
        if math.isfinite(r_n) and r_n > 0.0:
            out.append({"name": name, "target": float(np.clip(r_n / neck_eps, neck_min, h_axis)),
                        "r": float(r_n), "z": float(z_n), "radius": focus_extent * float(r_n),
                        "ell": float(r_n), "floor": neck_min})
    return out


def region_spacing(segs, region: dict, axis_tips: Sequence[float], sizing: dict | None,
                   size: float, finer: Sequence[tuple[float, float, float]] = ()
                   ) -> tuple[float, float]:
    """``(h_max, h_min)``: the longest and the shortest element (twice the node spacing; the elements
    are quadratic) of the chains within ``region`` (a ball of ``radius`` about ``(r, z)``), the
    shortest only where the cut's sizing (:func:`cut_size_field`) asks for at least
    ``CUT_SIZING_MARGIN`` times ``size``, since no focus can coarsen what it refines, and outside
    the balls ``finer`` (``(r, z, reach)``: the regions of a finer size within this one, with the
    reach of their grading), whose elements are theirs to coarsen; NaN where the region holds no
    such element."""
    longest, shortest = [], []
    for c in segs:
        r, z = np.asarray(c["r"], dtype=float), np.asarray(c["z"], dtype=float)
        if len(r) < 2:
            continue
        inside = np.hypot(r - region["r"], z - region["z"]) <= region["radius"]
        edges = inside[:-1] & inside[1:]
        if not edges.any():
            continue
        h = 2.0 * np.hypot(np.diff(r), np.diff(z))
        longest.append(float(h[edges].max()))
        if sizing and all(k in sizing for k in ("h_tip", "h_bridge", "grading")):
            free = cut_size_field(r, z, axis_tips, sizing) >= CUT_SIZING_MARGIN * size
            edges = edges & free[:-1] & free[1:]
        for fr, fz, reach in finer:
            other = np.hypot(r - fr, z - fz) <= reach
            edges = edges & ~(other[:-1] | other[1:])
        if edges.any():
            shortest.append(float(h[edges].min()))
    return (max(longest) if longest else math.nan, min(shortest) if shortest else math.nan)


def tip_radius(r, z, nodes: int = 6) -> float:
    """Radius of curvature ``1/|z''(0)|`` of a free surface at its end on the axis, the arrays
    ordered from that end (:func:`~faraday_singularity.collapse.axis_curvature`); NaN when the fit
    fails or the tip is flat."""
    k = axis_curvature(Profile(np.asarray(r, dtype=float), np.asarray(z, dtype=float)), nodes)
    return 1.0 / abs(k) if math.isfinite(k) and k != 0.0 else math.nan


def bubble_end(seg: dict, tip: str) -> dict:
    """One end of the bubble chain in the frame of the jet its tip launches: the nodes from the tip
    on the axis to the widest point of the bubble, ordered from the tip, with ``z`` and ``u_z``
    multiplied by the tip's sign (``BUBBLE_TIP_SIGNS``, kept as ``sign``).

    ``seg`` is the bubble's entry of ``FaradayProblem.interface_segments``, which runs upwards from
    the bottom tip to the top tip. In this frame the liquid lies below the tip, as at the cavity's
    tip, so that the open chain's measurement (:func:`~faraday_singularity.collapse.jet_base`)
    applies unchanged: the bubble's top is mirrored (``z -> -z``), and the jet the bridge drives
    down into the bubble rises there from a base that is the highest point of the bubble's top near
    the axis. Ending each half at the widest point keeps the other tip out of it."""
    sign = BUBBLE_TIP_SIGNS[tip]
    r, z, ur, uz = (np.asarray(seg[k], dtype=float) for k in ("r", "z", "u_r", "u_z"))
    k = int(np.argmax(r))
    idx = np.arange(k + 1) if sign > 0 else np.arange(len(r) - 1, k - 1, -1)
    return {"r": r[idx], "z": sign * z[idx], "u_r": ur[idx], "u_z": sign * uz[idx], "sign": sign}


def tip_jet(end: dict, *, r_search: float, rise_tol: float, flat_aspect: float,
            jet_aspect: float, jet_max_radius: float) -> dict:
    """The jet base of one end of the bubble chain (:func:`bubble_end`), found as the collapse
    driver finds the open chain's, with the driver's settings of the same names.

    Returns, in the frame of the axis (not of the end): the jet-base radius ``r_j`` (0 while the tip
    is the lowest point of its end, in the end's frame) and height ``z_b``, the tip height
    ``z_tip``; and, in the end's frame, the rise of the tip above the base (``rise``), the index of
    the base (``index``, 0 for the tip), whether a bump stands there (``bump``, the tip above the
    base by more than ``rise_tol``), whether it has nearly vanished (``flat_bump``: a rise below
    ``flat_aspect r_j``) and whether it passes the driver's test of a born jet (``jet``: narrower
    than ``jet_max_radius`` and standing more than ``max(rise_tol, jet_aspect r_j)`` above its
    base). The flux through the base, which needs the bulk, is the driver's."""
    sign = float(end["sign"])
    r, z = np.asarray(end["r"], dtype=float), np.asarray(end["z"], dtype=float)
    jb = jet_base(Profile(r, z), r_search, rise_tol)
    rise = float(z[0]) - jb.z_b
    return {"r_j": jb.r_j, "z_b": sign * jb.z_b, "z_tip": sign * float(z[0]), "rise": rise,
            "index": jb.index, "bump": jb.jet_born,
            "flat_bump": bool(jb.index > 0 and rise <= flat_aspect * jb.r_j),
            "jet": bool(jb.jet_born and jb.r_j < jet_max_radius
                        and rise > max(rise_tol, jet_aspect * jb.r_j))}


def post_cut_record(segs) -> dict:
    """What a run records of both chains after a cut, from ``FaradayProblem.interface_segments``.

    The three tips on the axis (the bubble's bottom and top, the cavity's tip) with their heights,
    axial speeds and radii of curvature, the smallest of those radii (``R_tip_min``), the bridge
    of liquid between the bubble and the cavity, the bubble's volume and widest point, and the
    narrowest interior neck of each chain (NaN without one)."""
    bubble = next(c for c in segs if not c["open"])
    cavity = next(c for c in segs if c["open"])
    br, bz, bw = (np.asarray(bubble[k], dtype=float) for k in ("r", "z", "u_z"))
    cr, cz, cw = (np.asarray(cavity[k], dtype=float) for k in ("r", "z", "u_z"))
    k = int(np.argmax(br))
    nb = chain_neck(np.column_stack([br, bz]), tuple(bubble["end_types"]))
    no = chain_neck(np.column_stack([cr, cz]), tuple(cavity["end_types"]))
    out = {"z_bubble_bottom": float(bz[0]), "w_bubble_bottom": float(bw[0]),
           "R_bubble_bottom": tip_radius(br, bz),
           "z_bubble_top": float(bz[-1]), "w_bubble_top": float(bw[-1]),
           "R_bubble_top": tip_radius(br[::-1], bz[::-1]),
           "z_cavity_tip": float(cz[0]), "w_cavity_tip": float(cw[0]),
           "R_cavity_tip": tip_radius(cr, cz),
           "bridge": float(cz[0] - bz[-1]),
           "bubble_volume": revolved_volume(np.column_stack([br, bz])),
           "bubble_r_max": float(br[k]), "bubble_z_r_max": float(bz[k]),
           "r_neck_bubble": nb[0], "z_neck_bubble": nb[1], "r_neck_open": no[0], "z_neck_open": no[1]}
    radii = [out[f"R_{x}"] for x in ("bubble_bottom", "bubble_top", "cavity_tip")]
    finite = [x for x in radii if math.isfinite(x)]
    out["R_tip_min"] = min(finite) if finite else math.nan
    return out


def courant_limit(segs) -> dict:
    """The smallest ``h/|u|`` over the nodes of every chain, and where it is.

    ``h`` is a node's element length, twice the shorter of its two segments along the chain (the
    elements are quadratic, so consecutive nodes are half an element apart), and ``|u|`` its speed.
    A step at most ``courant`` times this moves no node by more than that fraction of its element,
    wherever the tips and the jets of either chain have gone: unlike a window about the cut, it
    follows them. ``chain`` is 0 for the open chain and 1 for the bubble; ``inf`` with every node at
    rest."""
    best = {"h_over_u": math.inf, "h": math.nan, "U": 0.0, "chain": -1, "r": math.nan,
            "z": math.nan}
    for c in segs:
        r, z = np.asarray(c["r"], dtype=float), np.asarray(c["z"], dtype=float)
        if len(r) < 2:
            continue
        ds = np.hypot(np.diff(r), np.diff(z))
        h = 2.0 * np.minimum(np.r_[ds[0], ds], np.r_[ds, ds[-1]])
        u = np.hypot(np.asarray(c["u_r"], dtype=float), np.asarray(c["u_z"], dtype=float))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(u > 0.0, h / u, math.inf)
        i = int(np.argmin(ratio))
        if ratio[i] < best["h_over_u"]:
            best = {"h_over_u": float(ratio[i]), "h": float(h[i]), "U": float(u[i]),
                    "chain": 0 if c["open"] else 1, "r": float(r[i]), "z": float(z[i])}
    return best


# --- the interface after the cut, on its quadratic elements ---------------------------------------


def p2_edges(points) -> dict:
    """The quadratic elements of a chain of interface nodes ``(M, 2)``, ``M = 2N + 1`` (vertex,
    midside node, vertex, ...), as the mesh holds them: for each element its chord ``L`` (vertex to
    vertex), the offset of its midside node from the chord in chords (``bulge``, ``s/L``), where
    that node lies along the chord (``f``, 0.5 at mid-chord: at ``f = 1/4`` or ``3/4`` the
    Jacobian of the element vanishes at a vertex), the turn of its tangent across it (``turn``,
    degrees: the angle between ``dx/dxi`` of the quadratic at its two vertices, ``2 atan(4 s/L)``
    at ``f = 1/2``; on a circle of radius ``R`` the angle the element subtends, ``s/L`` about
    ``L/(8R)``), its midside node (``mid``) and that node's index along the chain (``index``).
    Raises ``ValueError`` for a chain that is no chain of quadratic elements (an even count)."""
    P = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(P) < 3 or len(P) % 2 != 1:
        raise ValueError(f"a chain of quadratic elements has 2N + 1 >= 3 nodes, not {len(P)}")
    a, m, b = P[0:-2:2], P[1:-1:2], P[2::2]
    c = b - a
    L = np.hypot(c[:, 0], c[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = c / L[:, None]
        d = m - 0.5 * (a + b)
        s = np.abs(d[:, 0] * t[:, 1] - d[:, 1] * t[:, 0])
        f = 0.5 + np.einsum("ij,ij->i", d, t) / L
        t0 = -3.0 * a + 4.0 * m - b  # dx/dxi at the first vertex (xi = 0) and at the second
        t1 = a - 4.0 * m + 3.0 * b
        turn = np.degrees(np.arctan2(np.abs(t0[:, 0] * t1[:, 1] - t0[:, 1] * t1[:, 0]),
                                     np.einsum("ij,ij->i", t0, t1)))
    return {"L": L, "bulge": s / L, "f": f, "turn": turn, "mid": m,
            "index": 2 * np.arange(len(L)) + 1}


def edge_reading(points) -> dict:
    """The worst quadratic elements of a chain (:func:`p2_edges`): the largest turn of the tangent
    across an element (``turn``, degrees) and the largest offset of a midside node from mid-chord
    along the chord (``midside``, ``|f - 1/2|``), each with that element's midside node
    (``*_r``, ``*_z``, ``*_index``) and chord (``*_L``), and the elements themselves (``edges``);
    NaN (``edges`` None) for a chain that is no chain of quadratic elements."""
    out = {"turn": math.nan, "turn_r": math.nan, "turn_z": math.nan, "turn_L": math.nan,
           "turn_index": -1, "midside": math.nan, "midside_r": math.nan, "midside_z": math.nan,
           "midside_L": math.nan, "midside_index": -1, "edges": None}
    try:
        e = p2_edges(points)
    except ValueError:
        return out
    if not len(e["L"]):
        return out
    out["edges"] = e
    dev = np.abs(e["f"] - 0.5)
    for key, values in (("turn", e["turn"]), ("midside", dev)):
        v = np.where(np.isfinite(values), values, -np.inf)
        k = int(np.argmax(v))
        out.update({key: float(values[k]), f"{key}_r": float(e["mid"][k, 0]),
                    f"{key}_z": float(e["mid"][k, 1]), f"{key}_L": float(e["L"][k]),
                    f"{key}_index": int(e["index"][k])})
    return out


def interface_guard(readings: dict, settings: ReconnectSettings, *, fresh: bool, h_floor: float,
                    refine_ratio: float) -> dict:
    """What the quadratic elements of the chains after a cut call for (``readings``: per chain name,
    :func:`edge_reading`).

    ``stop``: the ``interface-unresolved`` stop, or None: trigger ``edge-turn`` where the tangent of
    an element turns by more than ``edge_turn_stop_fresh`` degrees on a freshly built mesh
    (``fresh``) or ``edge_turn_stop`` on any, ``edge-midside`` where a midside node lies more than
    ``midside_stop`` chords from mid-chord, and ``feature-below-floor`` where an element that asks
    for a refinement is already within ``refine_ratio`` of the run's smallest size ``h_floor``, so
    that no rebuild can resolve it. ``refine``: per chain, the elements (indices into its
    :func:`p2_edges`) whose turn exceeds ``edge_turn_refine`` or whose midside node lies more than
    ``midside_refine`` chords off mid-chord: a rebuild refines there (the driver's ``kink_<chain>``
    region). The worst element over the chains is reported (its chain, turn or offset, the limit,
    the place of its midside node and its chord)."""
    s = settings
    turn_limit = s.edge_turn_stop_fresh if fresh else s.edge_turn_stop
    worst = None
    for name, rd in readings.items():
        for trigger, key, limit in (("edge-turn", "turn", turn_limit),
                                    ("edge-midside", "midside", s.midside_stop)):
            value = rd.get(key, math.nan)
            if math.isfinite(value) and value > limit and (worst is None
                                                            or value / limit > worst[0]):
                worst = (value / limit, name, trigger, key, limit)
    if worst is not None:
        _, name, trigger, key, limit = worst
        rd = readings[name]
        return {"stop": {"reason": "interface-unresolved", "trigger": trigger, "chain": name,
                         key: rd[key], "limit": limit, "fresh": bool(fresh), "r": rd[f"{key}_r"],
                         "z": rd[f"{key}_z"], "L": rd[f"{key}_L"], "index": rd[f"{key}_index"]},
                "refine": {}}
    refine = {}
    for name, rd in readings.items():
        e = rd.get("edges")
        if e is None:
            continue
        with np.errstate(invalid="ignore"):
            bad = (e["turn"] > s.edge_turn_refine) | (np.abs(e["f"] - 0.5) > s.midside_refine)
        idx = np.flatnonzero(bad)
        if not len(idx):
            continue
        floored = idx[e["L"][idx] <= refine_ratio * h_floor]
        if len(floored):
            k = int(floored[np.argmax(e["turn"][floored])])
            return {"stop": {"reason": "interface-unresolved", "trigger": "feature-below-floor",
                             "chain": name, "turn": float(e["turn"][k]),
                             "midside": float(abs(e["f"][k] - 0.5)), "fresh": bool(fresh),
                             "r": float(e["mid"][k, 0]), "z": float(e["mid"][k, 1]),
                             "L": float(e["L"][k]), "index": int(e["index"][k]),
                             "h_floor": h_floor},
                    "refine": {}}
        refine[name] = idx
    return {"stop": None, "refine": refine}


def kink_regions(previous: dict, readings: dict, refine: dict, *, t: float, hold,
                 h_floor: float) -> dict:
    """The ``kink_<chain>`` regions of the chains after a reading (:func:`interface_guard`'s
    ``refine``), from those of the last accepted step (``previous``, by chain): where the guard asks
    for a refinement, a region about the worst element of the chain, at a quarter of its chord (at
    least ``h_floor``) within four chords of it, grown to hold it when the chain has one already;
    a region whose chain reads no such element for ``hold(h_before)`` of time (``hold``, a function
    of the chord it met, ``h_before``) is released: its size returns to that chord by the bounded
    coarsening of the rebuilds, and the driver drops it once its size is back there. Each entry:
    ``name``, ``chain``, ``r``, ``z``, ``radius``, ``target``, ``h_before``, ``t_set`` (the last
    time it was asked for) and
    ``released``."""
    out = {name: dict(v) for name, v in previous.items()}
    for chain, idx in refine.items():
        e = readings[chain]["edges"]
        k = int(idx[np.argmax(e["turn"][idx])])
        L, (r, z) = float(e["L"][k]), (float(e["mid"][k, 0]), float(e["mid"][k, 1]))
        target = max(0.25 * L, h_floor)
        g = out.get(chain)
        if g is None:
            out[chain] = {"name": f"kink_{chain}", "chain": chain, "r": r, "z": z,
                          "radius": 4.0 * L, "target": target, "h_before": L, "t_set": t,
                          "released": False}
            continue
        d = math.hypot(r - g["r"], z - g["z"])
        g.update(radius=max(g["radius"], d + 4.0 * L), target=min(
            target, g["target"] if not g["released"] else math.inf),
            h_before=max(g["h_before"], L), t_set=t, released=False)
    for chain, g in out.items():
        if chain in refine or g["released"]:
            continue
        if t >= g["t_set"] + hold(g["h_before"]):
            g.update(released=True, target=g["h_before"])
    return out


APEX_WINDOWS = (2, 4)  # vertices on either side of a jet's base in the circle fits of its radius
GROOVE_ANGLE_VERTICES = 6  # the chords of the groove's opening angle reach this many vertices away
GROOVE_WINDOW = 200  # nodes on either side of a jet's base searched for the groove's fold pairs


def circle_radius(points) -> float:
    """Radius of the circle fitted algebraically through ``points`` (least squares in
    ``x^2 + y^2 + D x + E y + F = 0``, about their mean and in units of their extent); ``inf`` for
    points on a line, NaN for fewer than three distinct points."""
    P = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(P) < 3:
        return math.nan
    Q = P - P.mean(axis=0)
    scale = float(np.max(np.abs(Q)))
    if not scale > 0.0:
        return math.nan
    Q = Q / scale
    A = np.column_stack([Q[:, 0], Q[:, 1], np.ones(len(Q))])
    sol, _, rank, sv = np.linalg.lstsq(A, -(Q**2).sum(axis=1), rcond=None)
    if rank < 3 or sv[-1] < 1e-12 * sv[0]:
        return math.inf
    D, E, F = sol
    R2 = 0.25 * (D * D + E * E) - F
    return scale * math.sqrt(R2) if R2 > 0.0 else math.nan


def jet_base_apex(r, z, index: int) -> dict:
    """The base of a jet on an end of a chain ordered from its tip on the axis (the open chain, or
    an end of the bubble in the frame of its jet, :func:`bubble_end`: the liquid below), at node
    ``index`` (:func:`~faraday_singularity.collapse.jet_base`): the vertex there (``apex``; the
    vertices are the even nodes; for a midside node, the lower of its two vertices), the radius of
    curvature of the surface there (``R_apex``: the smaller of the radii of the circles fitted
    through the vertices within ``APEX_WINDOWS`` of it, on vertices only, so that a midside node
    that has left its place cannot bend the fit; where they differ, a wedge is forming) and the
    opening angle of the groove around the jet (``groove_angle``, degrees: between the chords from
    the apex to the vertices ``GROOVE_ANGLE_VERTICES`` away on either side, as far as the end
    reaches; 180 for a flat surface). NaN where fewer than two vertices lie on either side."""
    r, z = np.asarray(r, dtype=float), np.asarray(z, dtype=float)
    out = {"apex": -1, "apex_r": math.nan, "apex_z": math.nan, "R_apex": math.nan,
           "groove_angle": math.nan}
    n = len(r)
    i = int(index)
    if not 0 < i < n - 1:
        return out
    if i % 2:  # a midside node: the lower of its two vertices
        i = i - 1 if z[i - 1] <= z[i + 1] else i + 1
    V = np.column_stack([r[0::2], z[0::2]])
    v = i // 2
    side = min(v, len(V) - 1 - v)
    if side < 2:
        return out
    radii = [circle_radius(V[v - w:v + w + 1]) for w in APEX_WINDOWS if w <= side]
    radii = [x for x in radii if math.isfinite(x) or x == math.inf]
    w = min(GROOVE_ANGLE_VERTICES, side)
    a, b = V[v - w] - V[v], V[v + w] - V[v]
    cosang = float(np.dot(a, b) / max(np.hypot(*a) * np.hypot(*b), 1e-300))
    out.update(apex=i, apex_r=float(V[v, 0]), apex_z=float(V[v, 1]),
               R_apex=float(min(radii)) if radii else math.nan,
               groove_angle=math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))
    return out


def groove_gap(r, z, apex: int, *, ratio: float, stop: float, stop_elements: float,
               window: int = GROOVE_WINDOW) -> dict:
    """The fold of the groove around a jet: of the pairs of nodes ``i < apex < j`` (the two sides of
    the groove, ``apex`` the node of its apex along the chain) within ``window`` nodes of the apex
    that are closer than ``1/ratio`` of the arc between them (the fold pairs of
    :func:`~faraday_singularity.selfcontact.fold_gap`), the one with the smallest distance
    relative to its limit ``max(stop, stop_elements (spacing_i + spacing_j))``: ``gap`` (``inf``
    without one), ``limit``, ``arc``, ``pair``."""
    P = np.column_stack([np.asarray(r, dtype=float), np.asarray(z, dtype=float)])
    out = {"gap": math.inf, "limit": math.nan, "arc": math.nan, "pair": (-1, -1)}
    n = len(P)
    if not 0 < apex < n - 1:
        return out
    ds = np.hypot(*np.diff(P, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    spacing = np.empty(n)
    spacing[0], spacing[-1] = ds[0], ds[-1]
    spacing[1:-1] = 0.5 * (ds[:-1] + ds[1:])
    I = np.arange(max(apex - window, 0), apex)
    J = np.arange(apex + 1, min(apex + window, n - 1) + 1)
    D = np.hypot(P[I, None, 0] - P[None, J, 0], P[I, None, 1] - P[None, J, 1])
    A = s[J][None, :] - s[I][:, None]
    limit = np.maximum(stop, stop_elements * (spacing[I][:, None] + spacing[J][None, :]))
    score = np.where(ratio * D < A, D / limit, np.inf)
    if not np.isfinite(score).any():
        return out
    a, b = np.unravel_index(int(np.argmin(score)), score.shape)
    return {"gap": float(D[a, b]), "limit": float(limit[a, b]), "arc": float(A[a, b]),
            "pair": (int(I[a]), int(J[b]))}


def p2_jacobian_bounds(xy, elements) -> np.ndarray:
    """A lower bound of the scaled Jacobian of each quadratic triangle: the smallest Bezier
    coefficient of ``det(dx/ds)``, a polynomial of degree two, over that of the straight-sided
    triangle through the vertices, ``det(x0 - x2, x1 - x2)``.

    ``xy`` holds the node positions and ``elements`` the six nodes of each triangle in oomph-lib's
    order (``TElement<2,3>``, gmsh's too): the vertices 0, 1, 2 (counterclockwise, as pyoomph
    orients every element), then the nodes on the edges 0-1, 1-2 and 2-0. The bound is the exact
    value at the vertices (the three corner coefficients), at most the smallest value in between,
    1 for a straight-sided element with its edge nodes at mid-edge, zero for a midside node at the
    quarter point of its edge and negative beyond (or for an element inside out)."""
    X = np.asarray(xy, dtype=float)[np.asarray(elements, dtype=int)]
    x0, x1, x2, x3, x4, x5 = (X[:, k] for k in range(6))
    b01 = 2.0 * x3 - 0.5 * (x0 + x1)  # the Bezier control points of the edges
    b12 = 2.0 * x4 - 0.5 * (x1 + x2)
    b20 = 2.0 * x5 - 0.5 * (x2 + x0)
    D1 = (x0 - b20, b01 - b12, b20 - x2)  # dx/ds1 = 2 sum_k D1_k lambda_k
    D2 = (b01 - b20, x1 - b12, b12 - x2)  # dx/ds2 = 2 sum_k D2_k lambda_k

    def det(u, v):
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    c = [4.0 * det(D1[k], D2[k]) for k in range(3)]
    c += [2.0 * (det(D1[k], D2[l]) + det(D1[l], D2[k])) for k, l in ((0, 1), (1, 2), (0, 2))]
    lin = np.abs(det(x0 - x2, x1 - x2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.min(np.vstack(c), axis=0) / lin
