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
  therefore always rebuilt: from a surviving node of the old interface at radius ``r_s >=
  cap_splice_factor * delta`` the new cap is the circular arc tangent to the old interface there
  whose centre lies on the axis, so it meets the axis at a right angle;
* checks of the edited chains (simple, disjoint, monotone cap, no knife-edge angles), the volume
  the cut removes from the gas (:func:`chain_volumes`), and the gas volume within a box around the
  cut (:func:`gas_volume_in_box`), which separates the volume the cut adds to the liquid from the
  drift of rebuilding the whole free surface;
* the stops of a run after the cut (:func:`post_cut_stops`), on both chains.

Units are those of the solver: length ``1/k``. The neck radius ``r_n`` is the narrowest interior
local minimum of ``r`` along the free surface, as the collapse driver measures it
(:func:`neck_radius`). A cut needs ``delta`` a little above ``r_n``: just above it the planner
defers (the waist is not yet separable), so the landing requires ``delta/r_n >=
min_landing_ratio``. The cut geometry in units of ``delta`` (bridge, removed gas) depends on
``delta/r_n``, so a cut also requires ``delta/r_n <= max_landing_ratio``: a study in ``delta``
needs, for each ``delta``, a landing state whose neck has just passed below it. A run with
``collapse.landing`` keeps one for each of its cut radii, at the first measured state with
``delta/r_n >= LANDING_RATIO`` (``driver.CollapseDriver``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from .collapse import Profile, necks
from .selfcontact import InterfaceSelfIntersection, check_simple_interface, fold_gap

AXIS, FIXED = "axis", "fixed"
FILLS = ("blend", "zero")
RESTARTS = ("bdf1", "impulsive")
BUBBLE_MODELS = ("constant-pressure",)
CAPS = ("arc",)
LANDING_RATIO = 1.05  # a landing state for cut radius delta: the first with delta / r_n >= this
MAX_LANDING_RATIO = 1.1  # and a cut is refused above this delta / r_n (the default of the settings)


@dataclass(frozen=True)
class ReconnectSettings:
    """The ``collapse.reconnect`` block of a case: one cut at neck radius ``delta``.

    Only a constant-pressure bubble exists (the passive gas of the free surface); the planner's own
    volume correction is off; the planner's caps are always replaced by tangent arcs (``cap``).
    """
    delta: float  # cut radius
    reservoir_depth: float = 3.0  # closes the cavity chain through the gas above the free surface
    cap_window_factor: float = 6.0  # the planner's exclusion radius around the event, in delta
    min_landing_ratio: float = 1.03  # delta / r_n at least this
    max_landing_ratio: float = MAX_LANDING_RATIO  # and at most this, retries of the planner included
    plan_attempts: int = 5  # planner calls, the cut radius growing by retry_factor after a refusal
    retry_factor: float = 1.01
    cap: str = "arc"
    cap_splice_factor: float = 1.0  # the arc starts at a surviving node with r >= this * delta
    cap_min_points: int = 16  # control points of each arc, at least
    cap_tangent_fit: float = 2.0  # the tangent at the splice is fitted over this many r_s of arc
    min_cap_angle: float = 30.0  # smallest interior angle (deg) of a cap polyline
    tip_elements: float = 8.0  # element size at the fresh tips: arc radius / tip_elements,
    tip_floor: float = 1.0 / 64.0  # at most delta/4 and not below tip_floor * delta
    bridge_elements: float = 8.0  # at least this many elements along the bridge of liquid
    curvature_sizing: int = 20  # gmsh MeshSizeFromCurvature after the cut (0: off)
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
    tau_end: float = math.inf  # time after the cut at which the continuation ends
    max_events: int = 1
    bridge_stop: float = 0.1  # stop once the bridge is shorter than this * its length at the cut
    bubble_stop: float = 0.01  # or the bubble has shrunk below this fraction of its volume at the cut

    def __post_init__(self):
        def positive(name):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0 \
                    or not math.isfinite(value):
                raise ValueError(f"reconnect.{name} must be positive and finite")
        for name in ("delta", "reservoir_depth", "cap_window_factor", "retry_factor",
                     "cap_splice_factor", "cap_tangent_fit", "min_cap_angle", "tip_elements",
                     "tip_floor", "bridge_elements", "projection_tolerance", "energy_tolerance",
                     "dt0", "dt_growth", "courant", "bridge_stop", "bubble_stop"):
            positive(name)
        if not self.min_landing_ratio > 1.0:
            raise ValueError("reconnect.min_landing_ratio must exceed one")
        if isinstance(self.max_landing_ratio, bool) or not isinstance(
                self.max_landing_ratio, (int, float)) or not math.isfinite(self.max_landing_ratio) \
                or not self.max_landing_ratio > self.min_landing_ratio:
            raise ValueError("reconnect.max_landing_ratio must be finite and exceed "
                             "min_landing_ratio")
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
        for name in ("projection", "restore_pressure"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"reconnect.{name} must be true or false")
        if not (self.tau_end > 0):
            raise ValueError("reconnect.tau_end must be positive")
        if not self.min_cap_angle < 90.0:
            raise ValueError("reconnect.min_cap_angle must be below 90 degrees")

    @classmethod
    def from_dict(cls, data: dict) -> "ReconnectSettings":
        if not isinstance(data, dict):
            raise ValueError("collapse.reconnect must be an object")
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
    backtrack: float  # fold depth of the planner cap, in delta (0: monotone)
    min_angle: float  # smallest interior angle of the new cap, degrees
    splice_angle: float  # interior angle at the splice, degrees (180: tangent)


def tip_size(radius: float, delta: float, settings: ReconnectSettings) -> float:
    """Element size at a fresh tip: ``radius/tip_elements`` but at most ``delta/4`` (a cap needs a
    size floor at or below a quarter of the cut radius for gmsh to follow its curvature without
    inverting curved elements), and at least ``tip_floor * delta``."""
    return max(min(radius / settings.tip_elements, 0.25 * delta), settings.tip_floor * delta)


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
    away from the tip, whose radius is at least ``cap_splice_factor * delta``: that node is the
    splice. The arc (:func:`tangent_arc`) is tangent there to the old interface (fitted over
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
            while P[k, 0] < s.cap_splice_factor * delta:
                if O[k] < 0:
                    raise RuntimeError("planner points between survivors next to a fresh cap")
                k += step
                dropped += 1
                if not (0 <= k < len(P)) or dropped > len(P) // 2:
                    raise RuntimeError("no surviving node reaches the splice radius "
                                       f"{s.cap_splice_factor * delta:.3g}")
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
            h_tip = tip_size(R, delta, s)
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


def post_cut_stops(chains: Sequence[Chain], cut: CutGeometry, settings: ReconnectSettings,
                   h_min: float, *, neck_elements: float = 3.0, fold_ratio: float = 4.0,
                   fold_stop: float = 1e-3, fold_stop_elements: float = 3.0,
                   fold_search: float = 4.0) -> dict | None:
    """The stop a state after the cut calls for, if any.

    The mesh cannot follow a second change of topology: the bridge closing again (the bubble and
    cavity tips, on different chains, meeting across it), the chains touching elsewhere, the
    bubble's two tips meeting (it would become a ring), a new neck on either chain narrower than
    ``max(delta, neck_elements h_min)`` (``secondary-pinch-approach``: the thread behind a fresh cap
    can pinch again), a closing fold on the bubble (``self-contact-approach``, with the driver's
    fold limits: ``fold_gap`` with ``fold_ratio``, and a gap below ``max(fold_stop,
    fold_stop_elements h_elem)``), or the bubble draining away under its constant pressure. A
    bubble that crosses itself raises
    :class:`~faraday_singularity.selfcontact.InterfaceSelfIntersection`.

    The collapse driver measures only the open chain (``FaradayProblem.interface_arrays``), and its
    fold stop covers that chain within its focus region; these stops are the only ones on the
    bubble, which before the cut was part of the open chain. Each chain is examined on its own, so
    the bridge between the two is never a fold."""
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
    for name, c in (("bubble", bubble), ("open", cavity)):
        r_n, z_n, _ = chain_neck(c.points, c.end_types)
        if math.isfinite(r_n) and r_n < neck_limit:
            return {"reason": "secondary-pinch-approach", "trigger": "neck-radius", "chain": name,
                    "r_neck": r_n, "z_neck": z_n, "neck_limit": neck_limit}
    fold = fold_gap(bubble.points[:, 0], bubble.points[:, 1], ratio=fold_ratio, stop=fold_stop,
                    stop_elements=fold_stop_elements, search_factor=fold_search)
    if fold.crossed:
        raise InterfaceSelfIntersection("the bubble's free surface crosses itself after the cut")
    fold_limit = max(fold_stop, fold_stop_elements * fold.h_elem)
    if math.isfinite(fold.gap) and fold.gap < fold_limit:
        return {"reason": "self-contact-approach", "trigger": "fold-gap", "chain": "bubble",
                "fold_gap": fold.gap, "fold_r": fold.r, "fold_z": fold.z, "fold_arc": fold.arc,
                "fold_h_elem": fold.h_elem, "fold_limit": fold_limit}
    vb = revolved_volume(bubble.points)
    if vb < settings.bubble_stop * cut.bubble_volume:
        return {"reason": "bubble-vanishing", "bubble_volume": vb}
    return None
