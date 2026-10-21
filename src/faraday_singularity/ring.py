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
  (:func:`validate_loop_order`, pyoomph's own test of a closed loop).

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
LOOP_STEP_RATIO = 5.0  # no step of a loop longer than this times its median (pyoomph's own test)


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


def validate_loop_order(L) -> None:
    """Raise ``RuntimeError`` unless the loop is geometrically ordered: no chord, the closing one
    included, longer than ``LOOP_STEP_RATIO`` times the median chord (pyoomph's own test of a
    closed loop, which a misordered walk of a remeshed loop once failed), and its start not
    repeated at its end."""
    P = np.asarray(L, dtype=float).reshape(-1, 2)
    if len(P) < 3:
        raise RuntimeError(f"a loop needs at least three nodes, not {len(P)}")
    d = loop_steps(P)
    if not d[-1] > 0.0:
        raise RuntimeError("the loop repeats its start at its end: pass it without the repeat")
    median = float(np.median(d))
    worst = int(np.argmax(d))
    if median > 0.0 and d[worst] > LOOP_STEP_RATIO * median:
        a, b = P[worst], P[(worst + 1) % len(P)]
        raise RuntimeError(f"the loop is not geometrically ordered: the chord from ({a[0]:.9g}, "
                           f"{a[1]:.9g}) to ({b[0]:.9g}, {b[1]:.9g}) is {d[worst]:.6g}, more "
                           f"than {LOOP_STEP_RATIO:g} times the median chord {median:.6g}")


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
