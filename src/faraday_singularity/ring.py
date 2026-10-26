"""The ring of a pierced bubble: the geometry of its cut and the chain helpers of a closed loop.

After the gas cut the bubble's top carries the jet the bridge drives down into it. When that jet's
tip comes within ``max(ring.gap delta, 3 h_min)`` of the plug below it (the piercing, class ``W1``
of ``events.post_cut_events``), the ring cut (``ringsurgery.ring_cut``) cuts the gas on the axis:
the bubble's two tips and the gas between them become a column of liquid joining the plug to the
jet, and the bubble's rest is a ring of gas, a closed loop of the free surface that touches neither
the axis nor the wall. The ring keeps the passive, constant (zero) gas pressure of the whole free
surface, as the bubble had it (``reconnect.py``): it lives on the free surface's own boundary
``interface``, with the same equations, so nothing new is compiled. pyoomph refuses a problem with
equations on a boundary its mesh lacks, at ``initialise`` and at ``load_state`` alike, so a
boundary of the ring's own could never load a state written before the ring.

This module needs no finite-element problem (numpy and shapely only):

* the geometry of the cut (:func:`ring_cut_geometry`, :class:`RingCut`): on the bubble chain as
  ``reconnect.orient_chain`` gives it (upwards, from the plug's tip on the axis to the downward
  jet's tip on the axis), the jet's shoulder ``s_t`` (walking from the top tip while ``r`` does not
  decrease), the splice radius ``rho = ring.rho s_t`` and the waist ``r_w = ring.waist rho``, the
  splices (the plug's first node with ``r >= rho``, the cap's last), and the closure between them:
  an arc tangent to the cap at its splice turning down to ``r = r_w``, a vertical segment, an arc
  turning out to the plug's direction at its splice (C1 at both splices). Its points start at the
  node spacing of each splice and grow by at most ``events.CAP_GROWTH`` per point to ``r_w /
  elements_per_scale``. The ring is the loop of the bubble's wall between the splices closed by
  the closure (``L``, counter-clockwise, its first node, the seam, the vertex of the old walk of
  largest ``r``); the filled region ``Q``, the old gas the cut turns into liquid, is the plug
  inside its splice, the axis, the cap inside its splice and the closure. The cut has two length
  scales, the trigger gap ``ring.gap delta`` and the splice radius ``rho``; the waist is a shape
  constant. Hard checks refuse a cut the old mesh does not hold (``rho`` below
  ``RHO_MIN_NODES`` element lengths at the splices), a closure that leaves the old gas or makes a
  knife-edge angle, a filled region or a ring that is not simple, and a ring born below
  ``bubble_stop`` of the bubble's volume at the gas cut (:class:`RingCutRefused`);
* the helpers of a closed loop (``L``, an ``(n, 2)`` array without its start repeated): every chain
  helper of the project reads it as its closed polyline (:func:`closed`, the seam repeated at the
  end, on a vertex node, so that the ``2N + 1`` nodes of ``reconnect.edge_reading`` and the
  closing element of ``events.predict_unlocated`` are there); the crossings of its chords with the
  closing chord adjacent to the first (:func:`loop_crossings`), its fold with periodic arclength
  (:func:`loop_fold_gap`), the two splines a mesh is built on (:func:`loop_halves`) and their
  contacts (:func:`loop_spline_contact`), its quadratic elements (:func:`loop_edges`), what a run
  records of it (:func:`loop_record`) and the check that a loop read off a mesh is ordered
  (:func:`validate_loop_order`, after pyoomph's own test of a closed loop);
* a liquid cut of the open chain in the ring stage (``surgery.liquid_cut``), whose rebuild is the
  ring stage's own and keeps the ring: the cap and the removed head clear of the ring
  (:func:`cap_ring_clearance`, check (v) of ``events.truncate_chain`` against the loop) and the
  ring kept across the rebuild (:func:`loop_kept`: the count of its nodes, and its volume within
  the ring's noise floor besides what resampling its quadratic surface can change its polyline's
  volume by, :func:`loop_p2_excess`);
* the ring stage: what a run records of a state of it (:func:`ring_stage_record`, the keys of
  ``reconnect.post_cut_record`` with the bubble's those of the ring or NaN; :func:`loop_segment`)
  and its stops and events (:func:`ring_stage_events`: the open chain read as before the ring, the
  ring's emptying below ``bubble_stop`` of ``V_b`` the request of its removal, class ``W2``, its
  approach to the axis, ``RA``, its folds, ``RF``, and its approach to the open chain, ``X``, each
  without a handler but the removal's and, with ``groove.enabled``, the fill of a groove for
  ``RF`` (``groovesurgery.groove_fill``), and so a resumable stop); the chains whose elements the
  guard carries at the floor (:func:`ring_floored_ok`) and the wedge of gas of the ring at an
  element the guard reads at the run's floor (:func:`floor_wedge`, the fold test read at the scale
  of the floor), which the driver fills (``CollapseDriver._floor_fill_request``).

Lengths are in the solver's units (``1/k``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .events import (
    CAP_ANGLE_NODES,
    CAP_CLEARANCE,
    CAP_GROWTH,
    CAP_MIN_POINTS,
    CAP_SKIP,
    RING_RHO_MAX,
    cap_spacing,
    gas_sections,
    liquid_clearance,
    revolved_area,
)
from .selfcontact import FoldGap, catmull_rom, chord_crossings

RHO_MAX = RING_RHO_MAX  # the largest splice radius, in radii of the jet's shoulder
# the old mesh must hold the cut's scale: the splice radius at least this many element lengths
# (twice the shorter node spacing) at the splices, the larger of the two
RHO_MIN_NODES = 1.5
CLOSURE_SIN_MARGIN = 0.05  # 1 - sin(phi) at least this: the cap's tangent not along the axis
WAIST_MIN = 0.1  # the waist at least this times the splice radius
LOOP_STEP_RATIO = 5.0  # no chord of a loop longer than this times the shorter chord beside it


class RingCutRefused(RuntimeError):
    """The ring cut fails a hard check of its geometry before anything has changed: ``checks``
    holds what was read, with the checks that failed under ``failed``."""

    def __init__(self, message: str, checks: dict | None = None):
        super().__init__(message)
        self.checks = dict(checks or {})


# --- the helpers of a closed loop -----------------------------------------------------------------


def closed(L) -> np.ndarray:
    """The loop ``L`` (``(n, 2)``, its start not repeated) as its closed polyline: the seam repeated
    at the end, the form every chain helper of the project reads it in."""
    L = np.asarray(L, dtype=float).reshape(-1, 2)
    return np.vstack([L, L[:1]])


def signed_area(L) -> float:
    """The signed area of the polygon ``L`` in ``(r, z)``: positive counter-clockwise."""
    L = np.asarray(L, dtype=float).reshape(-1, 2)
    r, z = L[:, 0], L[:, 1]
    return 0.5 * float(np.sum(r * np.roll(z, -1) - np.roll(r, -1) * z))


def loop_steps(L) -> np.ndarray:
    """The length of each chord of the loop, the closing one (last node to first) last."""
    L = np.asarray(L, dtype=float).reshape(-1, 2)
    return np.hypot(*(np.roll(L, -1, axis=0) - L).T)


def orient_loop(L, origin=None, vertex=None) -> tuple[np.ndarray, np.ndarray | None,
                                                       np.ndarray | None]:
    """The loop counter-clockwise in ``(r, z)`` (its first node kept), with the arrays ``origin``
    and ``vertex`` beside it (per node, reordered alike), each None where not given."""
    L = np.asarray(L, dtype=float).reshape(-1, 2)
    if signed_area(L) >= 0.0:
        return L.copy(), origin, vertex
    order = np.concatenate([[0], np.arange(len(L) - 1, 0, -1)])
    return (L[order].copy(), None if origin is None else np.asarray(origin)[order],
            None if vertex is None else np.asarray(vertex)[order])


def seam_index(L, vertex) -> int:
    """The seam of a loop: of its nodes that are vertices (``vertex``, a boolean per node), the one
    of largest ``r``, ties to the larger ``z``."""
    L = np.asarray(L, dtype=float).reshape(-1, 2)
    idx = np.flatnonzero(np.asarray(vertex, dtype=bool))
    if not len(idx):
        raise ValueError("a loop without a vertex node has no seam")
    best = max(idx, key=lambda k: (float(L[k, 0]), float(L[k, 1])))
    return int(best)


def rotate(L, k: int, *others):
    """The loop and the arrays beside it rotated so that node ``k`` comes first."""
    out = [np.roll(np.asarray(L), -int(k), axis=0)]
    out += [None if x is None else np.roll(np.asarray(x), -int(k), axis=0) for x in others]
    return tuple(out) if others else out[0]


def loop_crossings(L) -> list[tuple[int, int]]:
    """The contacts ``(i, j)`` of non-adjacent chords of the loop, chord ``k`` from node ``k`` to
    ``(k + 1) mod n``, the closing chord ``n - 1`` adjacent to chord 0 (otherwise as
    ``selfcontact.chord_crossings``)."""
    L = np.asarray(L, dtype=float).reshape(-1, 2)
    n = len(L)
    if n < 4:
        return []
    C = closed(L)
    return [(i, j) for i, j in chord_crossings(C[:, 0], C[:, 1]) if (i, j) != (0, n - 1)]


def loop_spacing(L) -> np.ndarray:
    """The node spacing of the loop, periodic: the mean of the chords on either side of a node."""
    d = loop_steps(L)
    return 0.5 * (np.roll(d, 1) + d)


def loop_fold_gap(L, *, ratio: float, stop: float, stop_elements: float,
                  search_factor: float) -> FoldGap:
    """The fold of a loop, selected as ``selfcontact.fold_gap`` selects that of a chain with local
    stop limits, with the arclength between two nodes the shorter way round the loop
    (``min(|s_i - s_j|, S - |s_i - s_j|)``) and the node spacing periodic: each node searches
    ``search_factor max(stop, 2 stop_elements spacing)``, a pair is a fold where ``ratio D`` lies
    below the arclength between them, and the smallest ``D / max(stop, stop_elements (spacing_i +
    spacing_j))`` selects it. ``crossed`` reports contacts of its chords (:func:`loop_crossings`).
    The result does not depend on where the loop's seam lies but through the pair's indices."""
    from scipy.spatial import cKDTree

    from .selfcontact import NO_FOLD

    P = np.asarray(L, dtype=float).reshape(-1, 2)
    n = len(P)
    if n < 4:
        return NO_FOLD
    ds = loop_steps(P)
    s = np.concatenate([[0.0], np.cumsum(ds)[:-1]])
    S = float(ds.sum())
    spacing = loop_spacing(P)
    crossed = bool(loop_crossings(P))
    radii = search_factor * np.maximum(stop, 2.0 * stop_elements * spacing)
    tree = cKDTree(P)
    best, pair = math.inf, None
    for start in range(0, n, 128):
        nodes = np.arange(start, min(start + 128, n))
        neighbours = tree.query_ball_point(P[nodes], radii[nodes])
        i = np.repeat(nodes, [len(x) for x in neighbours])
        j = np.concatenate(neighbours).astype(int) if len(i) else np.zeros(0, dtype=int)
        unique = (spacing[i] > spacing[j]) | ((spacing[i] == spacing[j]) & (i < j))
        i, j = i[unique], j[unique]
        D = np.hypot(*(P[i] - P[j]).T)
        A = np.abs(s[i] - s[j])
        A = np.minimum(A, S - A)
        fold = ratio * D < A
        if not fold.any():
            continue
        score = D / np.maximum(stop, stop_elements * (spacing[i] + spacing[j]))
        k = np.flatnonzero(fold)[np.argmin(score[fold])]
        if score[k] < best:
            best, pair = float(score[k]), (int(i[k]), int(j[k]))
    if pair is None:
        return FoldGap(math.inf, math.nan, math.nan, (-1, -1), math.nan, math.nan, crossed)
    a, b = sorted(pair)
    m = 0.5 * (P[a] + P[b])
    arc = abs(float(s[b] - s[a]))
    return FoldGap(float(np.hypot(*(P[a] - P[b]))), float(m[0]), float(m[1]), (a, b),
                   min(arc, S - arc), float(spacing[a] + spacing[b]), crossed)


def loop_halves(L) -> tuple[np.ndarray, np.ndarray]:
    """The two open polylines a mesh builds the loop's two splines on: nodes ``0..m`` and ``m..n-1``
    then node 0 again, ``m = n // 2`` rounded down to an even position, so that with the seam on a
    vertex the split lands on a vertex too."""
    L = np.asarray(L, dtype=float).reshape(-1, 2)
    n = len(L)
    if n < 4:
        raise ValueError(f"a loop needs at least four nodes, not {n}")
    m = n // 2
    m -= m % 2
    return L[:m + 1].copy(), np.vstack([L[m:], L[:1]])


def loop_spline_contact(L, samples: int | None = None) -> dict | None:
    """The first self-contact of the loop's control polygon (:func:`loop_crossings`) or, if it is
    simple, of the curve a mesh is built on: the two open Catmull-Rom splines of
    :func:`loop_halves` (``selfcontact.catmull_rom``, each with its phantom end points), sampled,
    against themselves and each other, as one closed polyline whose joints at the two splits are
    adjacent chords. ``what`` (``polygon`` or ``spline``), ``count``, ``nodes`` (the loop's nodes
    at the start and end of the two intervals that meet), ``at`` and ``other`` as
    ``selfcontact.first_contact`` gives them; None if neither meets itself."""
    from .selfcontact import REMESH_SAMPLES

    samples = REMESH_SAMPLES if samples is None else int(samples)
    L = np.asarray(L, dtype=float).reshape(-1, 2)
    n = len(L)
    hits = loop_crossings(L)
    if hits:
        i, j = hits[0]
        return {"what": "polygon", "count": len(hits), "nodes": (i, (i + 1) % n, j, (j + 1) % n),
                "at": (float(L[i, 0]), float(L[i, 1])),
                "other": (float(L[j, 0]), float(L[j, 1]))}
    a, b = loop_halves(L)
    m = len(a) - 1
    A, B = catmull_rom(a[:, 0], a[:, 1], samples), catmull_rom(b[:, 0], b[:, 1], samples)
    Q = np.vstack([A, B[1:-1]])  # the closed curve, its start not repeated
    hits = loop_crossings(Q)
    if not hits:
        return None

    def node(k):  # the loop's node at the start of the sampled chord k
        return k // samples if k < m * samples else (m + (k - m * samples) // samples) % n

    i, j = hits[0]
    return {"what": "spline", "count": len(hits),
            "nodes": (node(i), (node(i) + 1) % n, node(j), (node(j) + 1) % n),
            "at": (float(Q[i, 0]), float(Q[i, 1])), "other": (float(Q[j, 0]), float(Q[j, 1]))}


def loop_edges(L) -> dict:
    """The quadratic elements of a loop read off a mesh (``reconnect.edge_reading`` of its closed
    polyline, :func:`closed`): ``2N`` nodes with the seam on a vertex make ``N`` elements, the
    closing one included; NaN for a loop of an odd count."""
    from .reconnect import edge_reading

    return edge_reading(closed(L))


def loop_record(L, u=None) -> dict:
    """What a run records of a loop: the volume of the ring of revolution it bounds (``V``), the
    area of its section and its centroid (``r_c``, ``z_c``), its innermost radius and the height
    there (``r_min``, ``z_r_min``), its outermost radius and its extent in height, the count of its
    nodes, their shortest and longest spacing, and with the nodes' velocities ``u`` (``(n, 2)``)
    the largest speed and the loop's Courant time ``min h/|u|`` (``h`` an element length, twice
    the shorter chord at a node)."""
    from .reconnect import revolved_volume

    P = np.asarray(L, dtype=float).reshape(-1, 2)
    r, z = P[:, 0], P[:, 1]
    r1, z1 = np.roll(r, -1), np.roll(z, -1)
    cross = r * z1 - r1 * z
    A = 0.5 * float(np.sum(cross))
    if A != 0.0:
        rc = float(np.sum((r + r1) * cross) / (6.0 * A))
        zc = float(np.sum((z + z1) * cross) / (6.0 * A))
    else:
        rc = zc = math.nan
    ds = loop_steps(P)
    k = int(np.argmin(r))
    out = {"V": revolved_volume(P), "area_section": abs(A), "r_c": rc, "z_c": zc,
           "r_min": float(r[k]), "z_r_min": float(z[k]), "r_max": float(r.max()),
           "z_min": float(z.min()), "z_max": float(z.max()), "points": len(P),
           "h_min": float(ds.min()), "h_max": float(ds.max()), "u_max": math.nan,
           "courant": math.nan}
    if u is not None:
        U = np.hypot(*np.asarray(u, dtype=float).reshape(-1, 2).T)
        h = 2.0 * np.minimum(np.roll(ds, 1), ds)
        out["u_max"] = float(U.max()) if len(U) else math.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(U > 0.0, h / U, math.inf)
        out["courant"] = float(ratio.min()) if len(ratio) else math.inf
    return out


def _simple_loop(P) -> bool:
    """Whether the closed polyline through ``P`` is simple (it crosses itself nowhere)."""
    from shapely.geometry import LinearRing

    return bool(LinearRing(P).is_simple)


def validate_loop_order(L) -> None:
    """Raise ``RuntimeError`` unless the loop is geometrically ordered: no chord, the closing one
    included, longer than ``LOOP_STEP_RATIO`` times the shorter of the two chords beside it (a
    misordered walk jumps across the loop and back, each jump beside a chord of the walk's own
    spacing; a graded mesh changes its spacing by a small factor per node, so that the test, unlike
    pyoomph's against the median chord of its periodic chart, holds on a ring meshed finely near
    the axis and coarsely far from it) unless the loop is simple (such a jump across the loop and
    back crosses it; a rebuild's spacing may jump by more than that factor beside a region refined
    to the floor, as at the corner of r1h's fill at step 7228, 3.6e-6 beside 6.2e-7, a loop in
    order), and its start not repeated at its end."""
    P = np.asarray(L, dtype=float).reshape(-1, 2)
    if len(P) < 3:
        raise RuntimeError(f"a loop needs at least three nodes, not {len(P)}")
    d = loop_steps(P)
    if not d[-1] > 0.0:
        raise RuntimeError("the loop repeats its start at its end: pass it without the repeat")
    if np.any(d <= 0.0):
        raise RuntimeError("the loop has a zero-length interior chord")
    beside = np.minimum(np.roll(d, 1), np.roll(d, -1))
    ratio = np.where(beside > 0.0, d / np.maximum(beside, 1e-300), np.inf)
    worst = int(np.argmax(ratio))
    if ratio[worst] > LOOP_STEP_RATIO and not _simple_loop(P):
        a, b = P[worst], P[(worst + 1) % len(P)]
        raise RuntimeError(f"the loop is not geometrically ordered: the chord from ({a[0]:.9g}, "
                           f"{a[1]:.9g}) to ({b[0]:.9g}, {b[1]:.9g}) is {d[worst]:.6g}, more "
                           f"than {LOOP_STEP_RATIO:g} times the shorter chord beside it "
                           f"({beside[worst]:.6g})")


def loop_p2_excess(L) -> float:
    """The volume of revolution between a loop's quadratic surface and its polyline: over each of
    its quadratic elements (vertex, midside node, vertex, the seam a vertex, so that ``2N`` nodes
    make ``N`` elements, the closing one included), the sliver between the element's parabola and
    its two chords, a third of the triangle of its three nodes (Archimedes: the parabolic segment
    over the chord of the two vertices is four thirds of that triangle), revolved about the axis at
    the triangle's centroid. Positive where the elements bulge out of the polygon: the gas the
    quadratic loop bounds is ``reconnect.revolved_volume(L)`` plus this, whatever the loop's
    orientation. NaN for an odd count of nodes, whose elements are unknown."""
    P = np.asarray(L, dtype=float).reshape(-1, 2)
    n = len(P)
    if n < 4 or n % 2:
        return math.nan
    v0, m, v1 = P[0::2], P[1::2], np.roll(P, -2, axis=0)[0::2]
    tri = 0.5 * ((m[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
                 - (m[:, 1] - v0[:, 1]) * (v1[:, 0] - v0[:, 0]))
    rbar = (v0[:, 0] + m[:, 0] + v1[:, 0]) / 3.0
    sign = 1.0 if signed_area(P) >= 0.0 else -1.0
    return sign * float(np.sum(tri * 2.0 * math.pi * rbar)) / 3.0


# a liquid cut in the ring stage keeps the ring (loop_kept): its nodes within this share of their
# count before the cut, its volume within this many of the ring's noise floor ring.sigma_V besides
# what resampling one quadratic surface can change its polyline's volume by
LOOP_NODE_SLACK = 0.1
LOOP_NOISE_FLOORS = 5.0


def loop_kept(before, after, *, sigma_V: float) -> dict:
    """Whether a rebuild that is not the ring's own (a liquid cut of the open chain in the ring
    stage, ``surgery.liquid_cut``) kept the ring: as many loops after it as before (one, or none
    once the ring is removed); the loop's nodes within ``LOOP_NODE_SLACK`` of their count before;
    the volume its polyline bounds changed by at most ``LOOP_NOISE_FLOORS sigma_V`` (the ring's
    noise floor, ``ring.sigma_V``) plus the drift of the rebuild, ``|S_before| + |S_after|``
    (:func:`loop_p2_excess`): the transfer places the new nodes on the old quadratic surface, so
    that each polyline lies within its slivers of that surface and resampling it moves the
    polyline's volume by at most their sum. ``before`` and ``after`` are lists of loops (``(n, 2)``
    arrays); returns what was read, the change of the quadratic surface's own volume (``dV_p2``,
    recorded only) and the ``problems`` found, none where the ring is kept."""
    from .reconnect import revolved_volume

    out = {"loops_before": len(before), "loops_after": len(after), "sigma_V": float(sigma_V),
           "node_slack": LOOP_NODE_SLACK, "noise_floors": LOOP_NOISE_FLOORS, "problems": []}
    if len(before) != len(after):
        out["problems"].append(f"{len(after)} loops after the rebuild, {len(before)} before it")
        return out
    if not before:
        return out
    if len(before) > 1:
        out["problems"].append(f"{len(before)} loops of the free surface, not at most one")
        return out
    A, B = (np.asarray(x[0], dtype=float).reshape(-1, 2) for x in (before, after))
    V0, V1 = revolved_volume(A), revolved_volume(B)
    S0, S1 = loop_p2_excess(A), loop_p2_excess(B)
    drift = abs(S0) + abs(S1)
    limit = LOOP_NOISE_FLOORS * float(sigma_V) + drift
    out.update(nodes_before=len(A), nodes_after=len(B), V_before=V0, V_after=V1, dV=V1 - V0,
               S_before=S0, S_after=S1, dV_p2=(V1 + S1) - (V0 + S0), drift=drift, limit=limit)
    if not abs(len(B) - len(A)) <= LOOP_NODE_SLACK * len(A):
        out["problems"].append(f"the ring has {len(B)} nodes after the rebuild, {len(A)} before "
                               f"it (beyond {LOOP_NODE_SLACK:.0%})")
    if not abs(V1 - V0) <= limit:  # NaN (an odd count) is a miss too
        out["problems"].append(f"the ring's volume changed by {V1 - V0:.6g}, beyond {limit:.6g} "
                               f"({LOOP_NOISE_FLOORS:g} sigma_V and the drift of the rebuild)")
    return out


def cap_ring_clearance(points, cap, removed, r_n: float, loops) -> dict:
    """Check (v) of ``events.truncate_chain`` against the ring, for a liquid cut of the open chain
    in the ring stage: the new chain (``points``) apart from every loop (``gap``, the smallest
    distance, positive), every point of the cap at least ``events.CAP_CLEARANCE r_n`` outside the
    ring's gas (``clearance``, ``events.liquid_clearance`` with the loop as the gas's section) and
    the removed region (``removed``, the head's half-section) sharing no area with it
    (``overlap``). ``failed`` lists the checks that do not hold (``gap``, ``inside``,
    ``removed``); none without a loop."""
    from shapely.geometry import LineString, Polygon

    out = {"loops": len(loops), "gap": math.inf, "clearance": math.inf,
           "clearance_limit": CAP_CLEARANCE * float(r_n), "overlap": 0.0, "failed": []}
    chain = LineString(np.asarray(points, dtype=float).reshape(-1, 2))
    head = Polygon(np.asarray(removed, dtype=float).reshape(-1, 2))
    for L in loops:
        C = closed(L)
        out["gap"] = min(out["gap"], float(LineString(C).distance(chain)))
        out["clearance"] = min(out["clearance"],
                               float(np.min(liquid_clearance(cap, [C]))) if len(cap) else math.inf)
        out["overlap"] += float(head.intersection(Polygon(C)).area)
    if not out["gap"] > 0.0:
        out["failed"].append("gap")
    if not out["clearance"] >= out["clearance_limit"]:
        out["failed"].append("inside")
    if not out["overlap"] <= 0.0:
        out["failed"].append("removed")
    return out


# --- the geometry of the ring cut -----------------------------------------------------------------


@dataclass
class RingCut:
    """The geometry of the ring cut (:func:`ring_cut_geometry`). ``loop`` is the ring ``L``
    (counter-clockwise, its seam first, start not repeated) and ``origin`` the old bubble node each
    of its points is (``-1`` on the closure); ``closure`` the closure from next to the jet's splice
    to next to the plug's; ``fill`` the filled region ``Q`` (old gas that becomes liquid), of volume
    ``V_fill``; ``A_rem`` the area of the bubble's surface the cut removes (its two tips inside the
    splices), ``A_cl`` that of the closure; ``checks`` the hard checks as read."""
    s_t: float
    i_sh: int
    z_shoulder: float
    rho: float
    rho_over_s_t: float
    waist: float
    r_w: float
    h_splice: float
    i_b: int
    i_t: int
    r_b: float
    z_b: float
    r_t: float
    z_t: float
    chi_deg: float
    phi_deg: float
    a_t: float
    a_b: float
    z_1: float
    z_2: float
    h_c: float
    closure: np.ndarray
    loop: np.ndarray
    origin: np.ndarray
    fill: np.ndarray
    V_fill: float
    V_ring: float
    A_rem: float
    A_cl: float
    L_fill: float
    z_mid: float
    gap: float
    checks: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """What a record keeps of the geometry, without its point arrays."""
        return {"s_t": self.s_t, "z_shoulder": self.z_shoulder, "rho": self.rho,
                "rho_over_s_t": self.rho_over_s_t, "waist": self.waist, "r_w": self.r_w,
                "h_splice": self.h_splice,
                "splice_plug": {"i": self.i_b, "r": self.r_b, "z": self.z_b,
                                "chi_deg": self.chi_deg},
                "splice_jet": {"i": self.i_t, "r": self.r_t, "z": self.z_t,
                               "phi_deg": self.phi_deg},
                "a_t": self.a_t, "a_b": self.a_b, "vertical_length": self.z_1 - self.z_2,
                "closure_points": len(self.closure), "h_c": self.h_c,
                "V_fill": self.V_fill, "V_ring": self.V_ring, "A_rem": self.A_rem,
                "A_cl": self.A_cl, "L_fill": self.L_fill, "z_mid": self.z_mid, "gap": self.gap,
                "checks": dict(self.checks)}


def jet_shoulder(P) -> int:
    """The shoulder of the downward jet on the bubble chain ``P`` (upwards, its last node the jet's
    tip on the axis): the node reached walking from the tip towards lower indices while ``r`` does
    not decrease."""
    P = np.asarray(P, dtype=float)
    k = len(P) - 1
    while k > 1 and P[k - 1, 0] >= P[k, 0]:
        k -= 1
    return k


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / max(float(np.hypot(*v)), 1e-300)


def _element_length(P, i: int) -> float:
    """Twice the shorter of the two chords at node ``i`` of a chain: the length of an element."""
    a = float(np.hypot(*(P[i] - P[i - 1]))) if i > 0 else math.inf
    b = float(np.hypot(*(P[i + 1] - P[i]))) if i < len(P) - 1 else math.inf
    return 2.0 * min(a, b)


class _Closure:
    """The closure of the ring cut as a curve: the arc ``A_t`` from the jet's splice, the vertical
    segment at ``r = r_w`` and the arc ``A_b`` to the plug's splice, parametrised by arclength from
    the jet's splice."""

    def __init__(self, Pt, phi, a_t, Pb, chi, a_b, r_w):
        self.Pt, self.Pb, self.phi, self.chi = np.asarray(Pt), np.asarray(Pb), phi, chi
        self.a_t, self.a_b, self.r_w = a_t, a_b, r_w
        d_t = np.array([math.cos(math.pi + phi), math.sin(math.pi + phi)])
        d_b = np.array([math.cos(chi), math.sin(chi)])
        self.c_t = self.Pt + a_t * np.array([-d_t[1], d_t[0]])  # left of the direction of travel
        self.c_b = self.Pb + a_b * np.array([-d_b[1], d_b[0]])
        self.psi0 = math.pi + phi
        self.z_1 = float(self.Pt[1] - a_t * math.cos(phi))
        self.z_2 = float(self.Pb[1] + a_b * math.cos(chi))
        self.L1 = a_t * (1.5 * math.pi - self.psi0)
        self.Lv = self.z_1 - self.z_2
        self.L3 = a_b * (0.5 * math.pi + chi)
        self.length = self.L1 + self.Lv + self.L3

    def at(self, s) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        out = np.empty((len(s), 2))
        first = s <= self.L1
        psi = self.psi0 + s[first] / self.a_t
        out[first] = self.c_t + self.a_t * np.column_stack([np.sin(psi), -np.cos(psi)])
        mid = (s > self.L1) & (s <= self.L1 + self.Lv)
        out[mid, 0] = self.r_w
        out[mid, 1] = self.z_1 - (s[mid] - self.L1)
        last = s > self.L1 + self.Lv
        psi = 1.5 * math.pi + (s[last] - self.L1 - self.Lv) / self.a_b
        out[last] = self.c_b + self.a_b * np.column_stack([np.sin(psi), -np.cos(psi)])
        return out


def _closure_arclengths(length: float, s_mid: float, h_top: float, h_bottom: float,
                        h_c: float) -> np.ndarray:
    """The arclengths of the closure's points, both splices excluded: from each end at the node
    spacing of its splice, growing by at most ``CAP_GROWTH`` per point to ``h_c``
    (``events.cap_spacing``), the two halves meeting at ``s_mid``."""
    top = cap_spacing(s_mid, h_top, h_c, CAP_GROWTH)
    bottom = length - cap_spacing(length - s_mid, h_bottom, h_c, CAP_GROWTH)[:-1]
    return np.concatenate([top, bottom[::-1]])


def ring_cut_geometry(bubble, open_chain, ring, *, elements_per_scale: float, min_angle: float,
                      reservoir_depth: float, V_b: float, bubble_stop: float) -> RingCut:
    """The geometry of the ring cut of the pierced ``bubble`` (a ``reconnect.Chain`` from the axis
    to the axis, upwards: its first node the plug's tip, its last the downward jet's), the open
    chain beside it ``open_chain``, under the block's ``ring`` (``events.RingEventSettings``), with
    ``elements_per_scale`` elements per waist radius on the closure (``ring.cut_elements`` or the
    reconnect block's), ``min_angle`` the smallest interior angle about a splice
    (``reconnect.min_cap_angle``), ``reservoir_depth`` the closure of the open chain through the gas
    above it, ``V_b`` the bubble's volume at the gas cut and ``bubble_stop`` the fraction of it
    below which the ring is removed (module docstring).

    The closure, in angles ``psi`` from ``+r`` counter-clockwise, with ``d_t`` the unit tangent of
    the cap at its splice towards its tip (central difference), ``phi = atan2(d_t) - 180 deg``, and
    ``d_b`` that of the plug at its splice away from its tip, ``chi = atan2(d_b)``: the arc ``A_t``
    of radius ``a_t = (r_t - r_w) / (1 - sin phi)`` from the jet's splice, turning from ``180 deg +
    phi`` to ``270 deg`` and ending at ``(r_w, z_t - a_t cos phi)``; the vertical segment at
    ``r_w`` down to ``z_b + a_b cos chi``; the arc ``A_b`` of radius ``a_b = (r_b - r_w) / (1 + sin
    chi)`` turning from ``270 deg`` to ``360 deg + chi`` and ending at the plug's splice with the
    plug's tangent. Its points, ``CAP_MIN_POINTS`` at least, start at the node spacing next to each
    splice and grow by at most ``CAP_GROWTH`` per point to ``h_c = r_w / elements_per_scale``.

    Raises :class:`RingCutRefused` unless the hard checks hold (``checks``, with ``failed``):
    ``splice`` (``r`` not decreasing from the plug's tip to its splice nor increasing from the
    cap's splice to the jet's tip; the cap's splice beyond the shoulder; ``rho`` at most ``RHO_MAX
    s_t`` and at least ``RHO_MIN_NODES h_splice``, ``h_splice`` the larger element length at the two
    splices; the jet's tip and its splice above the plug's tip and splice); ``closure`` (``1 - sin
    phi >= CLOSURE_SIN_MARGIN``, both arcs of positive radius, a vertical segment of length at least
    zero, ``r_w >= WAIST_MIN rho``); ``inside`` (every closure point but those within ``CAP_SKIP``
    of its own intervals, ``CAP_SKIP h_c`` of arc, of either splice, and at least the ``CAP_SKIP``
    nearest each, which leave the splices tangentially, at least ``CAP_CLEARANCE rho`` inside the
    old gas, ``events.liquid_clearance``, as a liquid cut's cap);
    ``angle`` (the interior angles over each splice and ``CAP_ANGLE_NODES`` nodes on either side at
    least ``min_angle``); ``fill`` (``Q`` simple, of positive volume); ``ring`` (``L`` simple, off
    the axis, disjoint from the open chain, its two splines without contact, :func:`
    loop_spline_contact`); ``volume`` (the ring at least ``bubble_stop V_b``, so that no ring is
    born due for its removal)."""
    from shapely.geometry import LinearRing, LineString

    from .reconnect import AXIS, interior_angles, revolved_volume

    P = np.asarray(bubble.points, dtype=float)
    n = len(P)
    if tuple(bubble.end_types) != (AXIS, AXIS) or n < 8:
        raise RingCutRefused("the ring cut needs the bubble chain, from the axis to the axis")
    failed: list[str] = []
    checks: dict = {"failed": failed}
    i_sh = jet_shoulder(P)
    s_t = float(P[i_sh, 0])
    rho = float(ring.rho) * s_t
    r_w = float(ring.waist) * rho
    gap = float(P[-1, 1] - P[0, 1])
    checks.update(s_t=s_t, i_sh=i_sh, rho=rho, r_w=r_w, gap=gap)
    i_b = int(np.argmax(P[:, 0] >= rho)) if np.any(P[:, 0] >= rho) else 0
    above = np.flatnonzero(P[:, 0] >= rho)
    i_t = int(above[-1]) if len(above) else n - 1
    h_splice = max(_element_length(P, i_b), _element_length(P, i_t)) if 0 < i_b < i_t < n - 1 \
        else math.inf
    checks.update(i_b=i_b, i_t=i_t, h_splice=h_splice,
                  rho_min=RHO_MIN_NODES * h_splice, rho_max=RHO_MAX * s_t)
    splice_ok = (0 < i_b < i_sh < i_t < n - 1
                 and bool(np.all(np.diff(P[:i_b + 1, 0]) >= 0.0))
                 and bool(np.all(np.diff(P[i_t:, 0]) <= 0.0))
                 and rho <= RHO_MAX * s_t * (1.0 + 1e-12) and rho >= RHO_MIN_NODES * h_splice
                 and gap > 0.0 and P[i_t, 1] - P[i_b, 1] > 0.0)
    if not splice_ok:
        failed.append("splice")
        raise RingCutRefused("the splices of the ring cut fail their checks: splice", checks)
    Pt, Pb = P[i_t], P[i_b]
    d_t = _unit(P[i_t + 1] - P[i_t - 1])
    d_b = _unit(P[i_b + 1] - P[i_b - 1])
    phi = math.atan2(d_t[1], d_t[0]) % (2.0 * math.pi) - math.pi
    chi = math.atan2(d_b[1], d_b[0])
    r_t, z_t, r_b, z_b = float(Pt[0]), float(Pt[1]), float(Pb[0]), float(Pb[1])
    margin = 1.0 - math.sin(phi)
    a_t = (r_t - r_w) / margin if margin > 0.0 else math.inf
    a_b = (r_b - r_w) / (1.0 + math.sin(chi)) if 1.0 + math.sin(chi) > 0.0 else math.inf
    checks.update(phi_deg=math.degrees(phi), chi_deg=math.degrees(chi), a_t=a_t, a_b=a_b,
                  sin_margin=margin)
    closure_ok = (margin >= CLOSURE_SIN_MARGIN and 0.0 < a_t < math.inf and 0.0 < a_b < math.inf
                  and -0.5 * math.pi < chi < 0.5 * math.pi and r_w >= WAIST_MIN * rho)
    curve = None
    if closure_ok:
        curve = _Closure(Pt, phi, a_t, Pb, chi, a_b, r_w)
        checks.update(z_1=curve.z_1, z_2=curve.z_2, vertical_length=curve.Lv)
        closure_ok = curve.Lv >= 0.0
    if not closure_ok:
        failed.append("closure")
        raise RingCutRefused("the closure of the ring cut fails its checks: closure", checks)
    h_c = r_w / float(elements_per_scale)
    h_top = float(np.hypot(*(P[i_t] - P[i_t - 1])))
    h_bottom = float(np.hypot(*(P[i_b + 1] - P[i_b])))
    s_mid = curve.L1 + 0.5 * curve.Lv
    while True:
        u = _closure_arclengths(curve.length, s_mid, h_top, h_bottom, h_c)
        if len(u) >= CAP_MIN_POINTS or h_c < 1e-6 * curve.length:
            break
        h_c /= 1.5
    C = curve.at(u)
    checks.update(h_c=h_c, closure_points=len(C))
    # the closure inside the old gas, but where it leaves the splices tangentially: as a liquid
    # cut's cap, the points within CAP_SKIP of its own intervals (h_c of arc) of either splice,
    # and at least the CAP_SKIP nearest each
    clearance = liquid_clearance(C, gas_sections([bubble], reservoir_depth))
    skip = CAP_SKIP * h_c * (1.0 + 1e-12)
    inner = (u > skip) & (u < curve.length - skip)
    inner[:CAP_SKIP] = inner[len(C) - CAP_SKIP:] = False
    checks.update(clearance=float(clearance[inner].max()) if inner.any() else -math.inf,
                  clearance_limit=-CAP_CLEARANCE * rho, skip_arc=skip,
                  skipped=int(len(C) - inner.sum()))
    if not checks["clearance"] <= checks["clearance_limit"]:
        failed.append("inside")
    # no knife-edge angle about either splice
    k = CAP_ANGLE_NODES
    around_t = np.vstack([P[max(i_t - k - 1, 0):i_t + 1], C[:k + 1]])
    around_b = np.vstack([C[len(C) - k - 1:], P[i_b:i_b + k + 2]])
    angles = np.concatenate([interior_angles(around_t), interior_angles(around_b)])
    checks.update(min_angle=float(angles.min()) if len(angles) else math.nan,
                  min_angle_limit=float(min_angle))
    if not checks["min_angle"] >= min_angle:
        failed.append("angle")
    # the filled region: the plug inside its splice, the axis, the cap inside its splice, the
    # closure back to the plug's splice
    Q = np.vstack([P[i_b::-1], P[n - 1:i_t - 1:-1], C])
    V_fill = revolved_volume(Q)
    checks["fill_simple"] = bool(LinearRing(Q).is_simple)
    if not (checks["fill_simple"] and V_fill > 0.0):
        failed.append("fill")
    # the ring: the bubble's wall between the splices closed by the closure
    L = np.vstack([P[i_b:i_t + 1], C])
    origin = np.concatenate([np.arange(i_b, i_t + 1), np.full(len(C), -1)])
    vertex = (origin >= 0) & (origin % 2 == 0)
    L, origin, vertex = orient_loop(L, origin, vertex)
    L, origin, vertex = rotate(L, seam_index(L, vertex), origin, vertex)
    crossings = loop_crossings(L)
    spline = loop_spline_contact(L) if not crossings else None
    open_gap = float(LineString(closed(L)).distance(LineString(np.asarray(open_chain.points))))
    V_ring = revolved_volume(L)
    checks.update(ring_crossings=len(crossings), ring_spline_contact=spline,
                  ring_r_min=float(L[:, 0].min()), open_gap=open_gap, V_ring=V_ring,
                  V_ring_limit=float(bubble_stop) * float(V_b))
    if crossings or spline is not None or not checks["ring_r_min"] > 0.0 or not open_gap > 0.0:
        failed.append("ring")
    if not V_ring >= float(bubble_stop) * float(V_b):
        failed.append("volume")
    if failed:
        raise RingCutRefused(f"the ring cut fails its checks: {', '.join(failed)}", checks)
    z_mid = 0.5 * (z_t + z_b)
    return RingCut(s_t=s_t, i_sh=i_sh, z_shoulder=float(P[i_sh, 1]), rho=rho,
                   rho_over_s_t=float(ring.rho), waist=float(ring.waist), r_w=r_w,
                   h_splice=h_splice, i_b=i_b, i_t=i_t, r_b=r_b, z_b=z_b, r_t=r_t, z_t=z_t,
                   chi_deg=math.degrees(chi), phi_deg=math.degrees(phi), a_t=a_t, a_b=a_b,
                   z_1=curve.z_1, z_2=curve.z_2, h_c=h_c, closure=C, loop=L, origin=origin,
                   fill=Q, V_fill=V_fill, V_ring=V_ring,
                   A_rem=revolved_area(P[:i_b + 1]) + revolved_area(P[i_t:]),
                   A_cl=revolved_area(np.vstack([Pt, C, Pb])),
                   L_fill=float(np.max(np.hypot(Q[:, 0], Q[:, 1] - z_mid))), z_mid=z_mid,
                   gap=gap, checks=checks)


# --- the ring stage: what a run records, and its stops and events ---------------------------------

# the record of a state of the ring stage has every key of reconnect.post_cut_record: the open
# chain's as before, the bubble's volume and widest point those of the ring, every other key of
# the bubble (its tips, the bridge, its neck) NaN, so that chains.txt keeps its columns
BUBBLE_KEYS = ("z_bubble_bottom", "w_bubble_bottom", "R_bubble_bottom", "z_bubble_top",
               "w_bubble_top", "R_bubble_top", "bridge", "r_neck_bubble", "z_neck_bubble")


def ring_stage_record(open_seg: dict, loop: dict | None) -> dict:
    """What a run records of a state of the ring stage, with the keys of
    ``reconnect.post_cut_record``: the open chain's tip, its radius of curvature and its neck as
    before; ``bubble_volume``, ``bubble_r_max`` and ``bubble_z_r_max`` those of the ring (NaN once
    it is removed); the bubble's tips, the bridge and the bubble's neck NaN; ``R_tip_min`` the open
    chain's tip's radius. ``open_seg`` and ``loop`` as ``FaradayProblem.interface_segments`` and
    ``interface_loops`` give them."""
    from .reconnect import chain_neck, revolved_volume, tip_radius

    cr, cz, cw = (np.asarray(open_seg[k], dtype=float) for k in ("r", "z", "u_z"))
    no = chain_neck(np.column_stack([cr, cz]), tuple(open_seg["end_types"]))
    out = {k: math.nan for k in BUBBLE_KEYS}
    out.update(z_cavity_tip=float(cz[0]), w_cavity_tip=float(cw[0]),
               R_cavity_tip=tip_radius(cr, cz), r_neck_open=no[0], z_neck_open=no[1],
               bubble_volume=math.nan, bubble_r_max=math.nan, bubble_z_r_max=math.nan)
    if loop is not None:
        L = np.column_stack([loop["r"], loop["z"]])
        k = int(np.argmax(L[:, 0]))
        out.update(bubble_volume=revolved_volume(L), bubble_r_max=float(L[k, 0]),
                   bubble_z_r_max=float(L[k, 1]))
    out["R_tip_min"] = out["R_cavity_tip"]
    return out


def loop_segment(loop: dict) -> dict:
    """A loop of ``FaradayProblem.interface_loops`` as a segment the readers of chains take (its
    closed polyline, :func:`closed`, its velocities alike; ``open`` False): what
    ``reconnect.courant_limit`` and ``reconnect.region_spacing`` read of it."""
    def wrap(x):
        x = np.asarray(x, dtype=float)
        return np.concatenate([x, x[:1]])

    return {"open": False, "end_types": ("loop", "loop"), "r": wrap(loop["r"]),
            "z": wrap(loop["z"]), "u_r": wrap(loop["u_r"]), "u_z": wrap(loop["u_z"])}


def ring_stage_events(chains, loops, cut, settings, h_min: float, events, ring: dict, *,
                      neck_elements: float, fold_ratio: float, fold_stop: float,
                      fold_stop_elements: float, fold_search: float, apexes: dict | None,
                      guard: dict | None, invalid=None, pinch: dict | None = None,
                      sheet=None) -> dict | None:
    """The stops and events of a state of the ring stage (after the ring cut,
    ``ringsurgery.ring_cut``; ``ring`` the problem's record of the stage), on the open chain
    (``chains``, ``reconnect.Chain``) and the ring (``loops``, ``(n, 2)`` arrays as
    ``FaradayProblem.interface_loops`` orders them), with the arguments of
    ``events.post_cut_events``, in its three ranks.

    1. The state itself: not one open chain, more than one loop, a loop after the ring's removal
       or none before it (``topology-changed``, with ``chains`` and ``loops``); the open chain's
       nodes crossing, or the ring's (raised, ``InterfaceSelfIntersection``), the ring touching the
       open chain or the axis (raised); a spline through the open chain's nodes that contacts
       itself, or one of the ring's two (``self-contact-approach``, trigger ``spline-contact``,
       chain ``ring`` for the ring's); on a freshly built mesh a tracked liquid neck of the open
       chain narrower than ``liquid.resolution`` element lengths, the guard of the interface over
       the open chain and the ring, and ``invalid()``.
    2. The end of the ring: before its removal, its volume below ``bubble_stop`` of the bubble's at
       the gas cut (``ring-collapse``, class ``W2``, handler ``ring-removal``).
    3. The events, in this order: the ring coming up to the axis, ``r_min < max(ring.axis_ratio
       r_w, 3 h_col)`` with ``r_w`` the column's waist and ``h_col`` its size (``ring-axis``,
       class ``RA``); the open chain's liquid necks as ``events.post_cut_events`` reads them
       (``L``, ``L-unhandled``; with liquid events off the cut's second-pinch test); the closing
       folds of the open chain and of its tracked heads (``A-head`` first, then a contact at the
       scale of the mesh, which stops the run) together with the ring's fold
       (:func:`loop_fold_gap`, against ``max(gap.kappa_g delta, fold_stop_elements h_elem)``: at
       the scale of the mesh a ``self-contact-approach`` of chain ``ring``, otherwise
       ``ring-fold``, class ``RF``), the closest to its limit first among the open chain's grooves
       (``A-groove``) and the ring's folds; the open chain's gas neck (``G``); the ring within
       ``3 h_min`` of the open chain (``chain-contact``, ``X``); on a mesh a step has run on, the
       guard last. Not tested: the piercing (no bubble), the bridge (the axis is liquid up to the
       cavity's tip) and the bubble emptying. The ring's own events have no handler but the
       removal's, and with the block's ``groove.enabled`` the fill of a groove for ``RF`` (its
       request then names it, ``handler`` ``groove-fill``): each other is a resumable stop. With
       ``groove.sheets`` ``log`` the folds of the open chain and of the ring are those of each side
       apart (``groove.side_folds``): a sheet of liquid is no event unless its faces close to the
       scale of the mesh (``sheet-contact``, without a handler). With ``pinch.enabled`` the open chain's necks land
       only when they close, as in ``events.post_cut_events`` (``pinch``, the driver's history).
       With ``sheet`` (the driver's reading of the ring's lip, ``CollapseDriver._sheet_context``:
       called with the ring's liquid fold and the element length of its finer side, it keeps the
       lip's thickness history and returns ``h_s``, ``thinning`` and ``refine_limit``) the ring's
       sheet is a ``sheet-contact`` also where its faces lie within ``refine_limit`` (the driver
       resolves it, ``driver.SHEET_REFINE_ELEMENTS`` element lengths, while it can refine) and
       where it is thinning at ``h_s`` or below (the driver ruptures it), with ``sheet_rupture``
       (``h_s``, ``due``, the thinning reading) in the request; the ring's liquid fold is read up
       to ``sheet.h_s`` also beyond the fold limit, as a sheet alone."""
    from shapely.geometry import LineString

    from .events import (GROOVE_HANDLER, SHEET_CONTACT, _neck_fields, _pinch_field, _request,
                         closing_readings, fold_stop_after_cut, head_fold, head_lands,
                         head_of_fold, lands, liquid_necks)
    from .reconnect import (Chain, adjacent_shoulders, chain_necks, edge_reading, fold_contact,
                            interface_guard, revolved_volume, spline_contact_stop)
    from .selfcontact import InterfaceSelfIntersection, fold_gap

    removed = ring.get("removed") is not None
    opens = [c for c in chains if c.is_open]
    if len(chains) != 1 or len(opens) != 1 or len(loops) > 1 or (loops and removed) or (
            not loops and not removed):
        return {"reason": "topology-changed", "chains": len(chains), "loops": len(loops)}
    cavity = opens[0]
    L = np.asarray(loops[0], dtype=float) if loops else None
    named = (("open", cavity),)
    ls = events.liquid
    # 1. the state itself
    stop_abs = fold_stop_after_cut(events, settings.delta, fold_stop)
    fold = fold_gap(cavity.points[:, 0], cavity.points[:, 1], ratio=fold_ratio, stop=stop_abs,
                    stop_elements=fold_stop_elements, search_factor=fold_search)
    if fold.crossed:
        raise InterfaceSelfIntersection("the open chain's free surface crosses itself after the "
                                        "cut")
    ring_gap = math.inf
    if L is not None:
        if loop_crossings(L):
            raise InterfaceSelfIntersection("the ring's free surface crosses itself")
        ring_gap = float(LineString(closed(L)).distance(LineString(cavity.points)))
        if not ring_gap > 0.0 or not float(L[:, 0].min()) > 0.0:
            raise RuntimeError("the ring touches or crosses the open chain or the axis")
    stride = 2 if settings.rebuild_spline == "vertices" else 1
    stop = spline_contact_stop([("open", Chain(cavity.points[::stride], cavity.end_types))
                                if stride > 1 and len(cavity.points) % 2 == 1
                                else ("open", cavity)])
    if stop is not None:
        return stop
    if L is not None:
        contact = loop_spline_contact(L)
        if contact is not None:
            return {"reason": "self-contact-approach", "trigger": "spline-contact",
                    "chain": "ring", "contact": contact["what"], "nodes": list(contact["nodes"]),
                    "contact_r": contact["at"][0], "contact_z": contact["at"][1],
                    "crossings": contact["count"]}
    necks = liquid_necks(named, ls) if ls.enabled else []
    closing_readings(necks, events, settings.delta, pinch)
    fresh = bool(guard["fresh"]) if guard is not None else False
    if fresh:
        unresolved = [x for x in necks if x["r_n"] < ls.resolution * x["h_loc"]]
        if unresolved:
            x = min(unresolved, key=lambda x: x["r_n"] / x["h_loc"])
            return {"reason": "liquid-neck-unresolved", **_neck_fields(x),
                    "limit": ls.resolution * x["h_loc"], "resolution": ls.resolution,
                    "fresh": True}

    def guarded():  # the guard of the interface on the quadratic elements of both
        readings = {"open": edge_reading(cavity.points)}
        if L is not None:
            readings["ring"] = loop_edges(L)
        return interface_guard(readings, settings, fresh=fresh, h_floor=float(guard["h_floor"]),
                               floored_ok=ring_floored_ok(events),
                               refine_ratio=float(guard["refine_ratio"]))["stop"]

    if guard is not None and fresh:
        stop = guarded()
        if stop is not None:
            return stop
    if invalid is not None:
        stop = invalid()
        if stop is not None:
            return stop
    # 2. the end of the ring
    V_b = float(cut.bubble_volume)
    if L is not None:
        V = revolved_volume(L)
        limit = settings.bubble_stop * V_b
        if V < limit:
            return _request("ring-collapse", "W2", chain="ring", ring_volume=V,
                            ratio=V / V_b, limit=limit, V_b=V_b)
    # 3. the events
    if L is not None:
        r_w = float(ring.get("r_w", math.nan))
        h_col = float((ring.get("sizing") or {}).get("h_col", math.nan))
        axis_limit = max(ring_axis_limits(ring, events))
        k = int(np.argmin(L[:, 0]))
        if float(L[k, 0]) < axis_limit:
            return _request("ring-axis", "RA", chain="ring", r_min=float(L[k, 0]),
                            z_r_min=float(L[k, 1]), limit=axis_limit, r_w=r_w, h_col=h_col)
    neck_limit = max(settings.delta, neck_elements * h_min)

    def secondary(side):  # the open chain's part of the test of events.post_cut_events
        for nk in chain_necks(cavity.points, cavity.end_types):
            if nk["side"] != side or not nk["r"] < neck_limit:
                continue
            found = {"chain": "open", "r_neck": nk["r"], "z_neck": nk["z"],
                     "index": int(nk["index"]), "neck_limit": neck_limit,
                     "neck_shoulder": nk["shoulder"], "neck_depth": nk["depth"],
                     "neck_side": side}
            deep = nk["depth"] >= settings.neck_depth
            if side == "gas":
                left, right = adjacent_shoulders(cavity.points[:, 0], int(nk["index"]))
                adj = float(np.minimum(left, right))
                adj_depth = 1.0 - nk["r"] / adj if adj > 0.0 else 0.0
                deep = deep and adj_depth >= settings.neck_depth
                found.update(neck_shoulder_left=left, neck_shoulder_right=right,
                             neck_shoulder_adjacent=adj, neck_depth_adjacent=adj_depth)
            if deep or nk["r"] < neck_elements * h_min:
                return found
        return None

    if ls.enabled:
        landed = sorted((x for x in necks if lands(x, events)), key=lambda x: x["ratio"])
        if landed:
            x = landed[0]
            return _request("liquid-neck", "L" if x["status"] == "head" else "L-unhandled",
                            eps_d=ls.eps_d, **_neck_fields(x), **_pinch_field(x))
    else:
        found = secondary("liquid")
        if found is not None:
            return _request("liquid-neck", "L-unhandled", detected_by="secondary-pinch", **found)
    closing = {}

    # the driver's reading of the ring's lip (sheet) reads a liquid fold of the ring up to its
    # rupture thickness h_s even where that lies beyond the fold limit, so that h_s, not the limit,
    # sets which lip is ruptured; such a fold is read as a sheet alone (``beyond``)
    lip_reach = float(getattr(sheet, "h_s", math.nan)) if sheet is not None else math.nan

    def offer(name, f, side=None, h_fine=math.nan):
        fold_limit = max(stop_abs, fold_stop_elements * f.h_elem)
        if not math.isfinite(f.gap):
            return
        beyond = (f.gap >= fold_limit and name == "ring" and side == "liquid"
                  and f.gap <= lip_reach)
        if f.gap < fold_limit or beyond:
            closing.setdefault((name, tuple(f.pair)), (f.gap / fold_limit, name, f, fold_limit,
                                                       side, h_fine, beyond))

    # with groove.sheets log the fold of the open chain and of the ring on each side apart
    # (groove.side_folds), so that a sheet of liquid hides no groove of gas
    sheets = events.groove.sheets == "log"
    fold_kw = {"ratio": fold_ratio, "stop": stop_abs, "stop_elements": fold_stop_elements,
               "search_factor": fold_search}
    if sheets:
        from .groove import side_folds

        for side, x in side_folds(cavity.points, loop=False, **fold_kw).items():
            offer("open", x["fold"], side, x["h_fine"])
    else:
        offer("open", fold)
    for x in necks:
        if x.get("head") is not None:
            offer("open", head_fold(cavity.points, x["head"], x["index"], ratio=fold_ratio,
                                    stop=stop_abs, stop_elements=fold_stop_elements,
                                    search_factor=fold_search))
    if L is not None:
        if sheets:
            for side, x in side_folds(L, loop=True, **fold_kw).items():
                offer("ring", x["fold"], side, x["h_fine"])
        else:
            offer("ring", loop_fold_gap(L, ratio=fold_ratio, stop=stop_abs,
                                        stop_elements=fold_stop_elements,
                                        search_factor=fold_search))
    landing, kinks, grooves = [], [], []
    for score, name, f, fold_limit, side, h_fine, beyond in closing.values():
        rec = {"chain": name, "contact": fold_contact(f.arc, f.h_elem), "fold_gap": f.gap,
               "fold_r": f.r, "fold_z": f.z, "fold_arc": f.arc, "fold_h_elem": f.h_elem,
               "fold_limit": fold_limit, "fold_pair": list(f.pair),
               "kappa_g": events.gap.kappa_g}
        if side is not None:
            rec["side"] = side
        jet = None
        if name == "open":
            a, b = f.pair
            jet = next((x for x in (apexes or {}).get("open", []) if a < int(x["index"]) < b),
                       None)
            if jet is not None:
                rec.update(jet=jet["tip"], apex_index=int(jet["index"]),
                           apex_r=jet.get("r", math.nan), apex_z=jet.get("z", math.nan),
                           R_apex=jet.get("R_apex", math.nan),
                           groove_angle=jet.get("groove_angle", math.nan))
        if rec["contact"] == "kink" and not beyond:
            reason = "self-contact-approach" if jet is None else "annular-contact-approach"
            kinks.append((score, {"reason": reason, "trigger": "fold-gap", **rec}))
            continue
        if side == "liquid":  # a sheet: no event, unless its faces close to the scale of the mesh
            contact_limit = events.groove.sheet_contact * h_fine
            lip = sheet(f, h_fine) if sheet is not None and name == "ring" else None
            due = lip is not None and lip["thinning"]["thinning"] and f.gap <= lip["h_s"]
            resolve = lip is not None and f.gap < lip["refine_limit"]
            if f.gap < contact_limit or due or resolve:
                extra = ({"sheet_rupture": {"h_s": lip["h_s"], "due": bool(due),
                                            "refine_limit": lip["refine_limit"],
                                            "thinning": lip["thinning"]}}
                         if lip is not None else {})
                grooves.append((score, "RF" if name == "ring" else "A-groove",
                                {**rec, "h_fine": h_fine, "sheet_contact_limit": contact_limit,
                                 "sheet_contact": True, **extra}))
            continue
        if name == "ring":
            grooves.append((score, "RF", rec))
            continue
        head = head_of_fold(f.pair, necks)
        if head is not None:
            rec["a_head"] = {**_neck_fields(head), "r_n_over_s": head["r_n"] / head["s"],
                             "eps_max": ls.eps_max, **_pinch_field(head)}
            if head_lands(head, events):
                landing.append(((head["r_n"] / head["s"], score), head, rec))
                continue
        grooves.append((score, "A-groove", rec))
    if landing:
        _, head, rec = min(landing, key=lambda x: x[0])
        return _request("a-head", "A-head" if head["status"] == "head" else "L-unhandled", **rec)
    if kinks:
        return min(kinks, key=lambda x: x[0])[1]
    if grooves:
        _, cls, rec = min(grooves, key=lambda x: x[0])
        if rec.pop("sheet_contact", False):  # a sheet whose faces meet: no handler
            req = _request(SHEET_CONTACT, cls, **rec)
            req["handler"] = None
            return req
        req = _request("groove" if cls == "A-groove" else "ring-fold", cls, **rec)
        if cls == "RF" and events.groove.enabled:  # its handler, the fill of a groove, made
            req["handler"] = GROOVE_HANDLER
        return req
    found = secondary("gas")
    if found is not None:
        return _request("gas-neck", "G", **found)
    if L is not None and ring_gap < 3.0 * h_min:
        return _request("chain-contact", "X", chain="ring", chain_gap=ring_gap,
                        limit=3.0 * h_min)
    if guard is not None and not fresh:
        return guarded()
    return None


# the fold test of a wedge of gas of the ring at the run's floor (floor_wedge): a pair across the
# wedge is a fold at the scale of the floor where FLOOR_FOLD_RATIO times its gap lies below the arc
# between the two nodes (a wedge narrower than about 84 degrees) and its gap below
# fold_stop_elements element lengths there
FLOOR_FOLD_RATIO = 1.5
# a corner of gas at the floor, open up to about a right angle (1/sin(50 degrees) = 1.31: up to about
# 100 degrees), read so where no wedge is: the root of the radial sheet that the downward jet's head
# drives out along the plug, where the sheet's top meets the column's wall (r1 at b36c9c1, step 7506:
# a gas corner of 90 degrees at the floor, turning by 99 degrees within an element of 1.08e-6)
FLOOR_CORNER_RATIO = 1.3


def floor_wedge(L, index: int, *, stop_elements: float, ratio: float = FLOOR_FOLD_RATIO,
                chords: int = 2) -> dict | None:
    """The wedge of gas of the ring at an element the guard reads at the run's floor, the fold test
    read at the scale of the floor (the driver's floor fill, ``CollapseDriver.
    _floor_fill_request``): ``L`` the loop (counter-clockwise, its gas inside), ``index`` the
    guard's element (its midside node, an index of :func:`loop_edges`' closed polyline). The apex
    is a vertex of the gas where the loop turns left across it (over ``chords`` chords on either
    side; a right turn is the tip of a liquid finger, and None); the pair a fill would close is the
    widest symmetric one about it, ``(index - m, index + m)``, that is still a fold at the scale of
    the floor: its gap below ``stop_elements`` times the element length at the pair (``h_elem``,
    the sum of the two nodes' spacings, :func:`loop_spacing`) and ``ratio`` times its gap below the
    arc between them (the shorter way round). None where the apex turns right or no pair is such a
    fold. Returns ``pair`` (sorted), ``m``, ``gap``, ``h_elem``, ``limit`` (``stop_elements
    h_elem``), ``arc``, ``ratio``, the apex (``apex``, ``r``, ``z``), its ``turn`` (degrees, left
    positive) and the gas side's interior ``angle``."""
    P = np.asarray(L, dtype=float).reshape(-1, 2)
    n = len(P)
    if n < 2 * chords + 4:
        return None
    i = int(index) % n
    d1, d2 = P[i] - P[(i - chords) % n], P[(i + chords) % n] - P[i]
    turn = math.degrees(math.atan2(float(d1[0] * d2[1] - d1[1] * d2[0]), float(d1 @ d2)))
    if not turn > 0.0:  # a right turn: the tip of a finger of liquid, no wedge of gas
        return None
    sp = loop_spacing(P)
    ds = loop_steps(P)
    s = np.concatenate([[0.0], np.cumsum(ds)[:-1]])
    S = float(ds.sum())
    best = None
    for m in range(1, n // 4):
        a, b = (i - m) % n, (i + m) % n
        gap = float(np.hypot(*(P[a] - P[b])))
        h_elem = float(sp[a] + sp[b])
        arc = abs(float(s[b] - s[a]))
        arc = min(arc, S - arc)
        if gap < stop_elements * h_elem and ratio * gap < arc:
            best = {"pair": sorted([int(a), int(b)]), "m": m, "gap": gap, "h_elem": h_elem,
                    "limit": float(stop_elements) * h_elem, "arc": arc}
        elif best is not None:
            break
    if best is None:
        return None
    best.update(ratio=float(ratio), apex=i, r=float(P[i, 0]), z=float(P[i, 1]), turn=turn,
                angle=180.0 - turn)
    return best


def ring_floored_ok(events) -> tuple:
    """The chains whose elements the guard carries at the floor in the ring stage
    (``reconnect.interface_guard``'s ``floored_ok``): the ring, with the block's jetcore and its
    recentred far kinks (the ring a nuisance, decision 22: a feature thinner than the run's floor,
    the rim of a thin sheet, is carried at the floor within the limits of the turn and the midside
    node rather than ending the run); none otherwise."""
    jet = getattr(events, "jetcore", None)
    return ("ring",) if jet is not None and jet.enabled and jet.far_kinks == "recentre" else ()


def ring_axis_limits(ring: dict, events) -> tuple[float, float]:
    """The two terms of the ring's approach to the axis (:func:`ring_stage_events`, class ``RA``):
    ``ring.axis_ratio`` times the column's waist ``r_w`` and three times the column's element size
    ``h_col``, from the problem's record of the stage (``ring``; NaN terms count as zero)."""
    r_w = float(ring.get("r_w", math.nan))
    h_col = float((ring.get("sizing") or {}).get("h_col", math.nan))
    a = events.ring.axis_ratio * r_w if math.isfinite(r_w) else 0.0
    b = 3.0 * h_col if math.isfinite(h_col) else 0.0
    return a, b
