"""The fill of a closing gas groove: its geometry (numpy and shapely only, no pyoomph).

After the gas cut a fold of a chain (the open chain or the bubble before the ring, the open chain or
the ring's loop in the ring stage) closing below ``max(gap.kappa_g delta, fold_stop_elements
h_elem)`` is an event (``events.post_cut_events``, class ``A-groove``; ``ring.ring_stage_events``,
class ``RF``). With the block's ``groove.enabled`` the driver meets it by the fill of the groove
(``groovesurgery.groove_fill``): the gas beyond the groove's mouth becomes liquid and the free
surface is closed across the mouth, so that the run goes on with the groove shortened instead of
following its gas to a closure the mesh cannot make. This module is the fill's geometry
(:func:`groove_fill_geometry`, :class:`GrooveFill`):

* **the pocket** the fold pair closes off: on a chain the arc between the pair and the chord across
  it; on the ring's loop, of its two arcs between the pair, the one whose polygon has the smaller
  volume of revolution. Its **side** is read: the share of its area inside the chain's gas (the
  ring's loop, the bubble's section, the open chain closed through the gas above it). Only a pocket
  of gas is filled; a pocket of liquid, a thin liquid sheet between two faces of gas, which the
  fold test meets alike, is refused;
* **the mouth**: from each node of the pair along its wall, away from the pocket, while the wall
  lies within the fill's width ``w_f`` (``groove.mouth`` times the fold's limit) of the other wall
  and has not turned out of the groove (``MOUTH_TURN`` from its tangent at the pair); the first
  node at least ``w_f`` from it, or so turned, is the splice. Every part of the groove narrower than
  ``w_f`` is filled at once, so that the fold test does not meet the rest of the groove again on
  the fresh mesh, but not the surface the groove opens onto;
* **the closure** from splice to splice, a cubic curve whose control points lie on the walls'
  tangents at the splices (C1 there), its handles ``eta`` times those of a circular arc turning by
  the angle between the tangents (``2c/3`` for a U-turn across the mouth chord ``c``), spaced from
  each splice's node spacing growing by at most ``events.CAP_GROWTH`` per point to ``c /
  elements_per_scale``, at least ``events.CAP_MIN_POINTS`` points;
* **the filled region** ``Q``: the old arc between the splices and the closure back, the old gas
  that becomes liquid, of volume ``V_fill``; and the new chain, or the new loop, oriented and
  seamed as the ring's is.

The side of a fold (:func:`side_folds`, with the block's ``groove.sheets`` ``log``): the fold test
meets a thin liquid sheet between two faces of gas as it meets a groove, so the stops then select
the fold of each chain and of the ring on each side apart (the pocket's polygon counter-clockwise
in the chain's direction, gas; clockwise, liquid), a sheet is no event but a record
(:func:`sheet_reading`: its thickness at the pair, its length beyond it, its tip, area, mean
thickness and volume), and only its faces closing to the scale of the mesh is one.

Hard checks refuse a fill before anything changes (:class:`GrooveFillRefused`): a pair at a chain's
end, a pocket that is not simple or not gas, a mouth not found within the chain, a fill on the core
of the upward jet (``protected``), a closure that leaves the old gas or makes a knife-edge angle at a
splice, a filled region that is not simple or not inside the old gas, a fill larger than
``max_fill V_b``, and new chains or a new loop a mesh cannot be built on. A fill never splits a
chain or the ring: what it deletes is the pocket with its mouth, the rest stays one chain or one
loop.

Lengths are in the solver's units (``1/k``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .events import (
    CAP_ANGLE_NODES,
    CAP_CLEARANCE,
    CAP_MIN_POINTS,
    CAP_SKIP,
    liquid_clearance,
    revolved_area,
)

SIDE_TOL = 1e-3  # a pocket is gas where at least 1 - SIDE_TOL of its area lies in the gas
# the walk to the mouth ends where a wall's tangent has turned by this many degrees from its
# tangent at the pair: the walls then open onto the surface around the groove, and the closure
# across them still turns by at least 180 - 2 MOUTH_TURN degrees into the groove (30)
MOUTH_TURN = 75.0
BEZIER_SAMPLES = 4097  # samples of the closure's curve for its arclength
CHAINS = ("open", "bubble", "ring")


class GrooveFillRefused(RuntimeError):
    """The fill of a groove fails a hard check of its geometry before anything has changed:
    ``checks`` holds what was read, with the checks that failed under ``failed``."""

    def __init__(self, message: str, checks: dict | None = None):
        super().__init__(message)
        self.checks = dict(checks or {})


@dataclass
class GrooveFill:
    """The geometry of a fill (:func:`groove_fill_geometry`). ``points`` is the new chain (in the
    order of the old one) or the new loop (counter-clockwise, seamed on a vertex of the old walk,
    start not repeated), ``origin`` the old node each of its points is (``-1`` on the closure);
    ``closure`` the closure from next to the first splice to next to the second; ``fill`` the
    filled region ``Q`` (old gas that becomes liquid), of volume ``V_fill``; ``pocket`` the pocket
    of the fold pair; ``A_rem`` the area of the old surface between the splices, ``A_cl`` that of
    the closure; ``i_s``, ``j_s`` the splices as old indices; ``h_arc`` the shortest chord of the
    old surface between the splices; ``checks`` the hard checks as read."""
    chain: str
    pair: tuple
    gap: float
    width: float
    i_s: int
    j_s: int
    mouth: float
    eta: float
    handle: float
    turn_deg: float
    h_c: float
    closure: np.ndarray
    points: np.ndarray
    origin: np.ndarray
    fill: np.ndarray
    pocket: np.ndarray
    side: str
    gas_share: float
    V_fill: float
    V_pocket: float
    A_rem: float
    A_cl: float
    L_fill: float
    r_c: float
    z_c: float
    V_gas_before: float
    V_gas_after: float
    U_max: float = math.nan
    h_arc: float = math.nan
    checks: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """What a record keeps of the geometry, without its point arrays."""
        return {"chain": self.chain, "pair": list(self.pair), "gap": self.gap,
                "width": self.width, "splices": [self.i_s, self.j_s], "mouth": self.mouth,
                "eta": self.eta, "handle": self.handle, "turn_deg": self.turn_deg, "h_c": self.h_c,
                "closure_points": len(self.closure), "points": len(self.points),
                "side": self.side, "gas_share": self.gas_share, "V_fill": self.V_fill,
                "V_pocket": self.V_pocket, "A_rem": self.A_rem, "A_cl": self.A_cl,
                "L_fill": self.L_fill, "r_c": self.r_c, "z_c": self.z_c,
                "V_gas_before": self.V_gas_before, "V_gas_after": self.V_gas_after,
                "U_max": self.U_max, "h_arc": self.h_arc, "checks": dict(self.checks)}


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / max(float(np.hypot(*v)), 1e-300)


def gas_section(chain: str, points, end_types, reservoir_depth: float) -> np.ndarray:
    """The half-section of the gas a chain bounds: the ring's loop itself, the bubble from the axis
    to the axis, the open chain closed through the gas above it (``reconnect.closed_section``)."""
    from .reconnect import closed_section

    P = np.asarray(points, dtype=float)
    if chain == "ring":
        return P
    return closed_section(P, tuple(end_types), reservoir_depth if chain == "open" else None)


def gas_share(polygon, gas) -> float:
    """The share of the area of ``polygon`` inside the half-section ``gas``: one for a pocket of
    gas, zero for one of liquid; NaN for a polygon without area."""
    from shapely.geometry import Polygon

    A = Polygon(np.asarray(polygon, dtype=float))
    if not A.is_valid or not A.area > 0.0:
        return math.nan
    G = Polygon(np.asarray(gas, dtype=float))
    if not G.is_valid:
        G = G.buffer(0)
    return float(A.intersection(G).area / A.area)


def side_of(share: float) -> str:
    """``gas``, ``liquid`` or ``mixed`` for a share of :func:`gas_share`."""
    if share >= 1.0 - SIDE_TOL:
        return "gas"
    if share <= SIDE_TOL:
        return "liquid"
    return "mixed"


def _bezier(B, t) -> np.ndarray:
    t = np.asarray(t, dtype=float)[:, None]
    return ((1 - t) ** 3 * B[0] + 3 * (1 - t) ** 2 * t * B[1] + 3 * (1 - t) * t**2 * B[2]
            + t**3 * B[3])


def closure_curve(P0, d0, P1, d1, eta: float) -> tuple[np.ndarray, float, float]:
    """The control points of the closure from ``P0`` (leaving along ``d0``) to ``P1`` (arriving
    along ``d1``): handles ``eta c / (3 cos^2(theta/4))``, ``c = |P1 - P0|`` and ``theta`` the angle
    between ``d0`` and ``d1``, the handle of a circular arc turning by ``theta`` over the chord
    ``c``. Returns the four control points, the handle and ``theta`` in degrees."""
    P0, P1 = np.asarray(P0, dtype=float), np.asarray(P1, dtype=float)
    d0, d1 = _unit(d0), _unit(d1)
    c = float(np.hypot(*(P1 - P0)))
    theta = math.acos(float(np.clip(np.dot(d0, d1), -1.0, 1.0)))
    lam = eta * c / (3.0 * math.cos(0.25 * theta) ** 2)
    return np.array([P0, P0 + lam * d0, P1 - lam * d1, P1]), lam, math.degrees(theta)


def closure_points(B, h_start: float, h_end: float, h_c: float) -> tuple[np.ndarray, np.ndarray,
                                                                        float]:
    """The points of the closure of the control points ``B`` (:func:`closure_curve`), both
    splices excluded: from each end at the node spacing of its splice, growing by at most
    ``events.CAP_GROWTH`` per point to ``h_c`` (``ring._closure_arclengths``), at least
    ``CAP_MIN_POINTS`` (``h_c`` made finer until there are). Returns the points, their arclengths
    and the closure's length."""
    from .ring import _closure_arclengths

    t = np.linspace(0.0, 1.0, BEZIER_SAMPLES)
    X = _bezier(B, t)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(X, axis=0).T))])
    length = float(s[-1])
    while True:
        u = _closure_arclengths(length, 0.5 * length, h_start, h_end, h_c)
        if len(u) >= CAP_MIN_POINTS or h_c < 1e-6 * length:
            break
        h_c /= 1.5
    tu = np.interp(u, s, t)
    return _bezier(B, tu), u, length


def _tangent(P, k: int) -> np.ndarray:
    return _unit(P[min(k + 1, len(P) - 1)] - P[max(k - 1, 0)])


def _march(P, start: int, step: int, other, width: float,
           max_turn: float | None = None) -> tuple[int, str]:
    """From node ``start`` of ``P`` in the direction ``step`` (-1 or +1), the first node at least
    ``width`` from the polyline ``other`` (``width``), or whose tangent has turned by more than
    ``max_turn`` degrees (``MOUTH_TURN`` when None) from that at ``start`` (``turn``); ``(-1,
    "end")`` where the walk reaches the end of ``P`` first."""
    import shapely

    line = shapely.LineString(np.asarray(other, dtype=float))
    d0 = _tangent(P, int(start))
    cos_max = math.cos(math.radians(MOUTH_TURN if max_turn is None else max_turn))
    k = int(start)
    while 0 <= k < len(P):
        if not float(shapely.distance(line, shapely.Point(P[k]))) < width:
            return k, "width"
        if 0 < k < len(P) - 1 and float(np.dot(_tangent(P, k), d0)) < cos_max:
            return k, "turn"
        k += step
    return -1, "end"


def groove_fill_geometry(chain: str, points, end_types, pair, *, width: float, eta: float,
                         elements_per_scale: float, min_angle: float, reservoir_depth: float,
                         V_b: float, max_fill: float, others=(), loops=(), protected=None,
                         velocity=None) -> GrooveFill:
    """The geometry of the fill of the groove a fold pair ``pair`` of ``chain`` (``open``,
    ``bubble`` or ``ring``) closes (module docstring): ``points`` the chain as the stops read it
    (the ends on the axis on ``r = 0``) or the ring's loop (counter-clockwise, seamed on a vertex),
    ``end_types`` its ends (ignored for the ring), ``width`` the fill's width ``w_f``, ``eta`` the
    closure's handle factor, ``elements_per_scale`` the closure's points per mouth width,
    ``min_angle`` the smallest interior angle about a splice (``reconnect.min_cap_angle``),
    ``reservoir_depth`` the closure of the open chain through the gas above it, ``V_b`` the
    bubble's volume at the gas cut and ``max_fill`` the largest share of it a fill takes;
    ``others`` the other chains of the free surface (``reconnect.Chain``), ``loops`` the ring's
    loop beside a chain filled in the ring stage, ``protected`` a mask of the chain's nodes no fill
    may take (the core of the upward jet), ``velocity`` the chain's nodal velocities (``(n, 2)``,
    for the largest speed on the old surface filled, ``U_max``). Raises
    :class:`GrooveFillRefused` unless the hard checks hold (``checks``, with ``failed``)."""
    from shapely.geometry import LinearRing, LineString

    from .reconnect import Chain, check_chains, interior_angles, revolved_volume
    from .ring import loop_crossings, loop_spline_contact, orient_loop, rotate, seam_index

    if chain not in CHAINS:
        raise ValueError(f"no fill of a groove of the chain {chain!r}; the chains are {CHAINS}")
    P = np.asarray(points, dtype=float).reshape(-1, 2)
    n = len(P)
    loop = chain == "ring"
    failed: list[str] = []
    checks: dict = {"failed": failed, "width": float(width), "eta": float(eta)}

    def refuse(name, message):
        failed.append(name)
        raise GrooveFillRefused(f"the fill of the groove fails its checks: {message}", checks)

    a, b = sorted(int(k) for k in pair)
    checks["pair"] = [a, b]
    if not (0 <= a < b < n) or b - a < 2 or (loop and n - (b - a) < 2) or (
            not loop and not (0 < a and b < n - 1)):
        refuse("pair", f"the pair {[a, b]} is no pair of a groove of the chain's {n} nodes")
    gap = float(np.hypot(*(P[a] - P[b])))
    checks["gap"] = gap
    gas = gas_section(chain, P, end_types, reservoir_depth)
    # the pocket, and the order in which the walls are walked: on a chain as it is; on the loop
    # rotated so that the middle of its other arc is its first node
    off = 0
    if loop:
        arcs = [(a, b), (b, a)]
        polys = [np.vstack([np.roll(P, -p, axis=0)[:(q - p) % n + 1]]) for p, q in arcs]
        vols = [revolved_volume(x) for x in polys]
        p, q = arcs[int(np.argmin(vols))]
        m = (q - p) % n
        off = (q + (n - m) // 2) % n
        Pw = np.roll(P, -off, axis=0)
        a_w, b_w = (p - off) % n, (q - off) % n
        checks["pocket_arc"] = "short" if m <= n - m else "long"
    else:
        Pw, a_w, b_w = P, a, b
    pocket = Pw[a_w:b_w + 1].copy()
    checks["pocket_simple"] = bool(len(pocket) >= 3 and LinearRing(pocket).is_simple)
    if not checks["pocket_simple"]:
        refuse("pocket", "the pocket of the fold pair is not a simple polygon")
    share = gas_share(pocket, gas)
    side = side_of(share) if math.isfinite(share) else "none"
    V_pocket = revolved_volume(pocket)
    checks.update(gas_share=share, side=side, V_pocket=V_pocket)
    if side != "gas":
        refuse("side", f"the pocket of the fold pair is {side} (gas share {share:.6g}), not a "
                       "groove of gas")
    # the mouth: each wall walked away from the pocket while within the fill's width of the other
    i_s, why_s = _march(Pw, a_w, -1, Pw[b_w:], width)
    j_s, why_e = _march(Pw, b_w, +1, Pw[:a_w + 1], width)
    checks.update(splices_walked=[int(i_s), int(j_s)], mouth_ends=[why_s, why_e])
    if not (1 <= i_s < a_w and b_w < j_s <= len(Pw) - 2):
        refuse("mouth", "the mouth of the groove is not found within the chain (the walls stay "
                        f"within {width:.6g} of each other to its ends)")
    old = (np.arange(len(Pw)) + off) % n  # the old index of each node walked
    if protected is not None:
        mask = np.asarray(protected, dtype=bool)
        hit = [int(k) for k in old[i_s:j_s + 1] if 0 <= k < len(mask) and mask[k]]
        checks["protected_nodes"] = len(hit)
        if hit:
            refuse("core", f"the fill would take {len(hit)} nodes of the upward jet's core")
    # the closure
    d_s = _unit(Pw[i_s + 1] - Pw[i_s - 1])
    d_e = _unit(Pw[j_s + 1] - Pw[j_s - 1])
    B, lam, turn = closure_curve(Pw[i_s], d_s, Pw[j_s], d_e, eta)
    mouth = float(np.hypot(*(Pw[j_s] - Pw[i_s])))
    h_c = mouth / float(elements_per_scale)
    h_start = float(np.hypot(*(Pw[i_s] - Pw[i_s - 1])))
    h_end = float(np.hypot(*(Pw[j_s + 1] - Pw[j_s])))
    C, u, length = closure_points(B, h_start, h_end, h_c)
    checks.update(mouth=mouth, handle=lam, turn_deg=turn, h_c=h_c, closure_points=len(C),
                  closure_length=length)
    clearance = liquid_clearance(C, [gas])
    skip = CAP_SKIP * h_c * (1.0 + 1e-12)
    inner = (u > skip) & (u < length - skip)
    inner[:CAP_SKIP] = inner[len(C) - CAP_SKIP:] = False
    checks.update(clearance=float(clearance[inner].max()) if inner.any() else -math.inf,
                  clearance_limit=-CAP_CLEARANCE * mouth, closure_r_min=float(C[:, 0].min()))
    if not checks["clearance"] <= checks["clearance_limit"] or not checks["closure_r_min"] > 0.0:
        failed.append("closure")
    k = CAP_ANGLE_NODES
    around_s = np.vstack([Pw[max(i_s - k - 1, 0):i_s + 1], C[:k + 1]])
    around_e = np.vstack([C[len(C) - k - 1:], Pw[j_s:j_s + k + 2]])
    angles = np.concatenate([interior_angles(around_s), interior_angles(around_e)])
    checks.update(min_angle=float(angles.min()) if len(angles) else math.nan,
                  min_angle_limit=float(min_angle))
    if not checks["min_angle"] >= min_angle:
        failed.append("angle")
    # the filled region: the old surface between the splices, the closure back
    Q = np.vstack([Pw[i_s:j_s + 1], C[::-1]])
    V_fill = revolved_volume(Q)
    fill_share = gas_share(Q, gas)
    checks.update(fill_simple=bool(LinearRing(Q).is_simple), V_fill=V_fill, fill_share=fill_share,
                  V_fill_limit=float(max_fill) * float(V_b))
    if not (checks["fill_simple"] and V_fill > 0.0 and fill_share >= 1.0 - SIDE_TOL):
        failed.append("fill")
    if not V_fill <= float(max_fill) * float(V_b):
        failed.append("volume")
    # the new chain, or the new loop
    new = np.vstack([Pw[:i_s + 1], C, Pw[j_s:]])
    origin = np.concatenate([old[:i_s + 1], np.full(len(C), -1), old[j_s:]])
    V_gas_before = revolved_volume(gas) if chain != "open" else math.nan
    if loop:
        vertex = (origin >= 0) & (origin % 2 == 0)
        new, origin, vertex = orient_loop(new, origin, vertex)
        new, origin, vertex = rotate(new, seam_index(new, vertex), origin, vertex)
        crossings = loop_crossings(new)
        spline = loop_spline_contact(new) if not crossings else None
        gap_open = min((float(LineString(np.vstack([new, new[:1]])).distance(LineString(c.points)))
                        for c in others), default=math.inf)
        checks.update(loop_crossings=len(crossings), loop_spline_contact=spline,
                      loop_r_min=float(new[:, 0].min()), open_gap=gap_open)
        if crossings or spline is not None or not checks["loop_r_min"] > 0.0 or \
                not gap_open > 0.0:
            failed.append("chains")
        V_gas_after = revolved_volume(new)
    else:
        try:
            check_chains([Chain(new, tuple(end_types))] + list(others))
            checks["chains"] = "ok"
        except RuntimeError as e:  # InterfaceSelfIntersection is one
            checks["chains"] = f"{type(e).__name__}: {e}"
            failed.append("chains")
        ring_gap = min((float(LineString(np.vstack([L, L[:1]])).distance(LineString(new)))
                        for L in loops), default=math.inf)
        checks["ring_gap"] = ring_gap
        if not ring_gap > 0.0 and "chains" not in failed:
            failed.append("chains")
        V_gas_after = revolved_volume(gas_section(chain, new, end_types, reservoir_depth)) \
            if chain != "open" else math.nan
    checks.update(V_gas_before=V_gas_before, V_gas_after=V_gas_after)
    if failed:
        raise GrooveFillRefused(f"the fill of the groove fails its checks: {', '.join(failed)}",
                                checks)
    U_max = math.nan
    if velocity is not None:
        V = np.asarray(velocity, dtype=float).reshape(-1, 2)
        U_max = float(np.max(np.hypot(*V[old[i_s:j_s + 1]].T)))
    A = 0.5 * float(np.sum(Q[:, 0] * np.roll(Q[:, 1], -1) - np.roll(Q[:, 0], -1) * Q[:, 1]))
    if A != 0.0:
        cross = Q[:, 0] * np.roll(Q[:, 1], -1) - np.roll(Q[:, 0], -1) * Q[:, 1]
        r_c = float(np.sum((Q[:, 0] + np.roll(Q[:, 0], -1)) * cross) / (6.0 * A))
        z_c = float(np.sum((Q[:, 1] + np.roll(Q[:, 1], -1)) * cross) / (6.0 * A))
    else:
        r_c, z_c = float(Q[:, 0].mean()), float(Q[:, 1].mean())
    return GrooveFill(chain=chain, pair=(a, b), gap=gap, width=float(width), i_s=int(old[i_s]),
                      j_s=int(old[j_s]), mouth=mouth, eta=float(eta), handle=lam, turn_deg=turn,
                      h_c=h_c, closure=C, points=new, origin=origin, fill=Q, pocket=pocket,
                      side=side, gas_share=share, V_fill=V_fill, V_pocket=V_pocket,
                      A_rem=revolved_area(Pw[i_s:j_s + 1]),
                      A_cl=revolved_area(np.vstack([Pw[i_s], C, Pw[j_s]])),
                      L_fill=float(np.max(np.hypot(Q[:, 0] - r_c, Q[:, 1] - z_c))), r_c=r_c,
                      z_c=z_c, V_gas_before=V_gas_before, V_gas_after=V_gas_after, U_max=U_max,
                      h_arc=float(np.min(np.hypot(*np.diff(Pw[i_s:j_s + 1], axis=0).T))),
                      checks=checks)


# --- the side of a fold: a groove of gas, or a sheet of liquid ------------------------------------


def side_folds(points, *, loop: bool, ratio: float, stop: float, stop_elements: float,
               search_factor: float) -> dict:
    """The fold of a chain or of the ring's loop on each side of the free surface, selected as
    ``selfcontact.fold_gap`` (``ring.loop_fold_gap`` for a loop) selects the fold of the whole
    chain, among the pairs whose pocket (the arc between them, the shorter way round a loop, and
    the chord) is gas and among those whose pocket is liquid, apart: ``{"gas": ..., "liquid":
    ...}``, each ``{"fold": FoldGap, "h_fine": ...}`` with ``h_fine`` the element length of the
    finer side of its pair (twice the smaller node spacing; NaN without a fold). The gas lies on the
    left of every chain as the project orders them (the open chain from the axis to the wall, the
    bubble upwards, the ring counter-clockwise), so a pocket is gas where its polygon, the arc in
    the chain's direction closed by the chord, is counter-clockwise (its signed area, from prefix
    sums of the chords' cross products, not negative). The fold of the whole chain is the one of
    the two closer to its limit, so that a sheet of liquid, met apart, hides no groove of gas.
    ``crossed`` is left False: the caller reads it from the fold of the whole chain."""
    from scipy.spatial import cKDTree

    from .ring import loop_spacing, loop_steps
    from .selfcontact import NO_FOLD, FoldGap

    P = np.asarray(points, dtype=float).reshape(-1, 2)
    n = len(P)
    empty = {side: {"fold": NO_FOLD, "h_fine": math.nan} for side in ("gas", "liquid")}
    if n < (4 if loop else 3):
        return empty
    if loop:
        ds = loop_steps(P)
        s = np.concatenate([[0.0], np.cumsum(ds)[:-1]])
        S = float(ds.sum())
        spacing = loop_spacing(P)
    else:
        ds = np.hypot(*np.diff(P, axis=0).T)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        S = float(s[-1])
        spacing = np.empty(n)
        spacing[0], spacing[-1] = ds[0], ds[-1]
        spacing[1:-1] = 0.5 * (ds[:-1] + ds[1:])
    r, z = P[:, 0], P[:, 1]
    cross = r * np.roll(z, -1) - np.roll(r, -1) * z  # chord k to k + 1, the closing one last
    C = np.concatenate([[0.0], np.cumsum(cross[:-1])])  # C[k]: the chords before node k
    A_loop = 0.5 * float(cross.sum())
    radii = search_factor * np.maximum(stop, 2.0 * stop_elements * spacing)
    tree = cKDTree(P)
    best = {"gas": (math.inf, None), "liquid": (math.inf, None)}
    for start in range(0, n, 128):
        nodes = np.arange(start, min(start + 128, n))
        near = tree.query_ball_point(P[nodes], radii[nodes])
        i = np.repeat(nodes, [len(x) for x in near])
        j = np.concatenate(near).astype(int) if len(i) else np.zeros(0, dtype=int)
        unique = (spacing[i] > spacing[j]) | ((spacing[i] == spacing[j]) & (i < j))
        i, j = i[unique], j[unique]
        D = np.hypot(*(P[i] - P[j]).T)
        A = np.abs(s[i] - s[j])
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        area = 0.5 * (C[hi] - C[lo] + (r[hi] * z[lo] - r[lo] * z[hi]))
        if loop:
            back = A > S - A  # the shorter way round runs through the seam
            A = np.minimum(A, S - A)
            area = np.where(back, A_loop - area, area)
        fold = ratio * D < A
        if not fold.any():
            continue
        score = D / np.maximum(stop, stop_elements * (spacing[i] + spacing[j]))
        gas = area >= 0.0
        for side, mask in (("gas", fold & gas), ("liquid", fold & ~gas)):
            if mask.any():
                k = np.flatnonzero(mask)[np.argmin(score[mask])]
                if score[k] < best[side][0]:
                    best[side] = (float(score[k]), (int(i[k]), int(j[k])))
    out = {}
    for side, (_, pair) in best.items():
        if pair is None:
            out[side] = empty[side]
            continue
        a, b = sorted(pair)
        m = 0.5 * (P[a] + P[b])
        arc = abs(float(s[b] - s[a]))
        if loop:
            arc = min(arc, S - arc)
        out[side] = {"fold": FoldGap(float(np.hypot(*(P[a] - P[b]))), float(m[0]), float(m[1]),
                                     (a, b), arc, float(spacing[a] + spacing[b]), False),
                     "h_fine": 2.0 * float(min(spacing[a], spacing[b]))}
    return out


def sheet_reading(points, pair, *, loop: bool) -> dict:
    """What the record of a sheet holds of the pocket of liquid a fold pair closes off (the arc
    between the pair, the shorter way round a loop, and the chord): its thickness at the pair (the
    pair's distance), the arc, the length beyond the pair (half the arc, and the distance of its
    tip, the node of the pocket farthest from the pair's midpoint), its area, mean thickness (the
    area over half the arc), volume of revolution and centroid, the tip's place."""
    from .reconnect import revolved_volume

    P = np.asarray(points, dtype=float).reshape(-1, 2)
    n = len(P)
    a, b = sorted(int(k) for k in pair)
    Q = P[a:b + 1]
    if loop:
        back = np.vstack([P[b:], P[:a + 1]])
        if np.sum(np.hypot(*np.diff(back, axis=0).T)) < np.sum(np.hypot(*np.diff(Q, axis=0).T)):
            Q = back
    arc = float(np.sum(np.hypot(*np.diff(Q, axis=0).T)))
    m = 0.5 * (P[a] + P[b])
    k = int(np.argmax(np.hypot(Q[:, 0] - m[0], Q[:, 1] - m[1])))
    x, y = Q[:, 0], Q[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    A = 0.5 * float(cross.sum())
    rc = float(np.sum((x + np.roll(x, -1)) * cross) / (6.0 * A)) if A else math.nan
    zc = float(np.sum((y + np.roll(y, -1)) * cross) / (6.0 * A)) if A else math.nan
    return {"pair": [a, b], "nodes": int(len(Q)), "thickness": float(np.hypot(*(P[a] - P[b]))),
            "r": float(m[0]), "z": float(m[1]), "arc": arc, "length": 0.5 * arc,
            "tip_distance": float(np.hypot(Q[k, 0] - m[0], Q[k, 1] - m[1])),
            "tip_r": float(Q[k, 0]), "tip_z": float(Q[k, 1]), "area": abs(A),
            "mean_thickness": abs(A) / (0.5 * arc) if arc > 0.0 else math.nan,
            "volume": revolved_volume(Q), "r_c": rc, "z_c": zc, "n_chain": n}
