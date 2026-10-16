"""Self-contact of the free surface: folds that close, and a surface that crosses itself.

A sharp-interface mesh cannot change topology, so the run must stop before two parts of the free
surface touch, just as it stops before a neck pinches off. Near the jet base a fold can form: an
air slot between the jet and the cavity wall whose walls approach each other. Its closure is an
event that would require a topology change the mesh cannot follow; once the discrete surface has crossed
itself it bounds no planar domain, and Gmsh then writes a mesh without a single liquid element.

The free surface is passed as arrays ordered along the curve from the axis to the wall.

* **Fold pair**: two nodes whose distance ``D`` is less than ``1/ratio`` of the arc length ``A``
  between them along the surface. A smooth surface has ``D ~ A`` for nearby nodes; a V-shaped
  groove of half-angle ``theta`` has ``D/A = sin(theta)``, so ``ratio = 4`` admits grooves sharper
  than about 14.5 degrees, and a single node turning by less than about 151 degrees is never one.
* **Fold gap**: the smallest ``D`` within a fixed search, or the pair with smallest ``D/limit``
  when local stop limits are supplied; ``inf`` without a fold pair in the search.
* **Crossing**: an intersection, touch or overlap of non-adjacent chords of the polyline.

The remesh rebuilds the surface as the uniform Catmull-Rom spline through the nodes, the curve of
both Gmsh's built-in ``Spline`` and pyoomph's ``CurvedEntityCatmullRomSpline`` (pyoomph
``src/meshtemplate.cpp``, ``CurvedEntityCatmullRomSpline::interpolate``: phantom end points
``2 p0 - p1`` and ``2 pN - pN-1``, a uniform parameter per interval). :func:`check_simple_interface`
refuses contacts of the control polygon or a sampled spline. The spline is sampled at ``samples``
points per interval; loops narrower than that sample spacing can pass this check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


class InterfaceSelfIntersection(RuntimeError):
    """The free surface crosses itself: the liquid is no longer a planar domain."""


def _points(r, z) -> np.ndarray:
    return np.column_stack([np.asarray(r, dtype=float), np.asarray(z, dtype=float)])


def chord_crossings(r, z, chords=None) -> list[tuple[int, int]]:
    """Contacts ``(i, j)``, ``i < j - 1``, of chords ``k = (P_k, P_k+1)`` of a polyline.

    ``chords`` restricts the test to those chord indices. Two chords can cross only if their
    midpoints are at most the longer chord's length apart, so each chord searches a ball of its
    own length and keeps partners no longer than itself.
    """
    P = _points(r, z)
    k = np.arange(len(P) - 1) if chords is None else np.asarray(sorted(chords), dtype=int)
    if len(k) < 2:
        return []
    a, b = P[k], P[k + 1]
    L = np.hypot(*(b - a).T)
    mid = 0.5 * (a + b)
    pairs = set()
    for u, nbrs in enumerate(cKDTree(mid).query_ball_point(mid, L * (1.0 + 1e-12))):
        for v in nbrs:
            if L[v] <= L[u] and abs(int(k[u]) - int(k[v])) > 1:
                pairs.add((min(u, v), max(u, v)))
    if not pairs:
        return []
    I, J = np.array(sorted(pairs)).T

    def orient(p, q, x):
        return (q[:, 0] - p[:, 0]) * (x[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (x[:, 0] - p[:, 0])

    hit = ((orient(a[I], b[I], a[J]) * orient(a[I], b[I], b[J]) <= 0)
           & (orient(a[J], b[J], a[I]) * orient(a[J], b[J], b[I]) <= 0)
           & np.all(np.maximum(np.minimum(a[I], b[I]), np.minimum(a[J], b[J]))
                    <= np.minimum(np.maximum(a[I], b[I]), np.maximum(a[J], b[J])), axis=1))
    return [(int(k[i]), int(k[j])) for i, j in zip(I[hit], J[hit])]


def catmull_rom(r, z, samples: int = 8) -> np.ndarray:
    """The uniform Catmull-Rom spline through the nodes, ``samples`` points per interval."""
    P = _points(r, z)
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    t = np.linspace(0.0, 1.0, samples, endpoint=False)[:, None]
    w = [-.5 * t**3 + t**2 - .5 * t, 1.5 * t**3 - 2.5 * t**2 + 1,
         -1.5 * t**3 + 2 * t**2 + .5 * t, .5 * t**3 - .5 * t**2]
    seg = sum(w[q][None, :, :] * ext[q:q + len(P) - 1][:, None, :] for q in range(4))
    return np.vstack([seg.reshape(-1, 2), P[-1:]])


REMESH_SAMPLES = 8  # spline points per interval of the remesh's check (mesh.py)
# a contact spanning at least this many elements of the surface between its two sides is a fold of
# the resolved surface; one within fewer is a kink at the scale of the mesh (classify_contact)
FOLD_RESOLVED_ELEMENTS = 4


def first_contact(points, samples: int = REMESH_SAMPLES) -> dict | None:
    """The first self-contact of the control polygon ``points`` (``(r, z)`` pairs ordered from the
    axis) or, if the polygon is simple, of its sampled Catmull-Rom spline; None if neither has one.

    ``what`` is ``"polygon"`` or ``"spline"``; ``nodes`` the first and last node of each of the two
    intervals that meet, ``(i, i + 1, j, j + 1)``; ``at`` and ``other`` the start of each sampled
    chord; ``count`` the number of contacts. :func:`check_simple_interface` raises on the same
    contact, so a state whose nodes pass here passes the remesh's check."""
    P = np.asarray(points, dtype=float)
    for what, Q, scale in (("polygon", P, 1),
                           ("spline", catmull_rom(P[:, 0], P[:, 1], samples), samples)):
        hits = chord_crossings(Q[:, 0], Q[:, 1])
        if hits:
            i, j = hits[0]
            return {"what": what, "count": len(hits),
                    "nodes": (i // scale, i // scale + 1, j // scale, j // scale + 1),
                    "at": (float(Q[i, 0]), float(Q[i, 1])),
                    "other": (float(Q[j, 0]), float(Q[j, 1]))}
    return None


def classify_contact(points, contact: dict) -> dict:
    """Whether a contact of :func:`first_contact` is a fold of the resolved surface or a kink at
    the scale of the mesh, with the evidence: the node intervals between the two sides (``span``,
    two per quadratic element), the shortest and longest node spacing over the nodes involved and
    their ratio, and the largest turn of the polygon there (degrees). A ``kink`` spans fewer than
    ``FOLD_RESOLVED_ELEMENTS`` elements: its two sides are a few nodes apart along the surface, not
    two parts of the surface approaching across a gap that the mesh resolves."""
    P = np.asarray(points, dtype=float)
    i, _, _, j1 = contact["nodes"]
    span = j1 - i
    lo, hi = max(0, i - 2), min(len(P) - 1, j1 + 2)
    ds = np.hypot(*np.diff(P[lo:hi + 1], axis=0).T)
    a = np.diff(P[lo:hi + 1], axis=0)
    ang = np.degrees(np.arctan2(a[:, 1], a[:, 0]))
    turn = np.abs((np.diff(ang) + 180.0) % 360.0 - 180.0)
    return {"contact": "kink" if span < 2 * FOLD_RESOLVED_ELEMENTS else "fold", "span": int(span),
            "spacing_min": float(ds.min()), "spacing_max": float(ds.max()),
            "spacing_ratio": float(ds.max() / ds.min()) if ds.min() > 0 else math.inf,
            "max_turn_deg": float(turn.max()) if len(turn) else 0.0}


def check_simple_interface(points, samples: int = REMESH_SAMPLES):
    """Raise :class:`InterfaceSelfIntersection` if the control polygon ``points`` (``(r, z)`` pairs
    ordered from the axis) or its sampled Catmull-Rom spline contacts itself (:func:`first_contact`).

    The spline uses ``samples`` points per interval; loops narrower than the sample spacing can
    pass. This is a geometric guard, not a universal certificate that the spline is simple.
    """
    c = first_contact(points, samples)
    if c is not None:
        what = "free surface" if c["what"] == "polygon" else "spline through the free surface"
        i, i1, j, j1 = c["nodes"]
        raise InterfaceSelfIntersection(
            f"the {what} crosses itself ({c['count']} crossing(s)); the first between nodes "
            f"{i}-{i1} at (r, z) = ({c['at'][0]:.9g}, {c['at'][1]:.9g}) and nodes "
            f"{j}-{j1} at ({c['other'][0]:.9g}, {c['other'][1]:.9g}): no planar liquid "
            "domain to mesh (interface contact the sharp-interface mesh cannot follow)")


@dataclass(frozen=True)
class FoldGap:
    gap: float  # distance of the selected fold pair, inf without one within the search
    r: float  # midpoint of that pair
    z: float
    pair: tuple[int, int]  # node indices along the surface
    arc: float  # arc length between them
    h_elem: float  # local element length there: twice the mean node spacing of the pair
    crossed: bool  # some chords within r <= r_max cross


NO_FOLD = FoldGap(math.inf, math.nan, math.nan, (-1, -1), math.nan, math.nan, False)


def fold_gap(r, z, *, r_max: float = math.inf, ratio: float = 4.0,
             search: float | None = None, stop: float | None = None,
             stop_elements: float = 0.0, search_factor: float = 4.0) -> FoldGap:
    """Select a fold among nodes with ``r <= r_max``.

    With ``stop`` supplied, each node searches ``search_factor * max(stop, 2 * stop_elements *
    spacing)`` and the smallest ``D / max(stop, stop_elements * (spacing_i + spacing_j))`` selects
    the pair. A fold that violates its local limit cannot be hidden by a narrower, resolved fold.
    Otherwise only pairs closer than ``search`` are examined and the smallest distance selects.
    ``crossed`` reports contacts of chords with both ends in the selection. These are conservative
    geometric limits; a static sharp corner can also meet them.
    """
    P = _points(r, z)
    if len(P) < 3:
        return NO_FOLD
    ds = np.hypot(*np.diff(P, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    spacing = np.empty(len(P))
    spacing[0], spacing[-1] = ds[0], ds[-1]
    spacing[1:-1] = 0.5 * (ds[:-1] + ds[1:])
    sel = np.flatnonzero(P[:, 0] <= r_max)
    in_sel = np.zeros(len(P), dtype=bool)
    in_sel[sel] = True
    chords = np.flatnonzero(in_sel[:-1] & in_sel[1:])
    crossed = bool(chord_crossings(P[:, 0], P[:, 1], chords))
    if len(sel) < 3:
        return FoldGap(math.inf, math.nan, math.nan, (-1, -1), math.nan, math.nan, crossed)
    if stop is None and search is None:
        raise ValueError("fold_gap requires a fixed search or local stop limits")
    radii = (np.full(len(sel), search) if stop is None else
             search_factor * np.maximum(stop, 2.0 * stop_elements * spacing[sel]))
    tree = cKDTree(P[sel])
    best, pair = math.inf, None

    def candidate_pairs():
        if stop is None:
            # Retain the fixed-search ordering, including its choice between equal distances.
            pairs = tree.query_pairs(search, output_type="ndarray")
            yield sel[pairs[:, 0]], sel[pairs[:, 1]]
            return
        # Bound temporary storage; a coarse region must not widen every fine-node search.
        for start in range(0, len(sel), 128):
            nodes = sel[start:start + 128]
            neighbours = tree.query_ball_point(P[nodes], radii[start:start + len(nodes)])
            i = np.repeat(nodes, [len(partners) for partners in neighbours])
            j = sel[np.concatenate(neighbours).astype(int)]
            unique = (spacing[i] > spacing[j]) | ((spacing[i] == spacing[j]) & (i < j))
            yield i[unique], j[unique]

    for i, j in candidate_pairs():
        D = np.hypot(*(P[i] - P[j]).T)
        A = np.abs(s[i] - s[j])
        fold = ratio * D < A
        if not fold.any():
            continue
        score = D if stop is None else D / np.maximum(stop, stop_elements * (spacing[i] + spacing[j]))
        k = np.flatnonzero(fold)[np.argmin(score[fold])]
        if score[k] < best:
            best, pair = float(score[k]), (int(i[k]), int(j[k]))
    if pair is None:
        return FoldGap(math.inf, math.nan, math.nan, (-1, -1), math.nan, math.nan, crossed)
    a, b = sorted(pair)
    m = 0.5 * (P[a] + P[b])
    return FoldGap(float(np.hypot(*(P[a] - P[b]))), float(m[0]), float(m[1]), (a, b), float(s[b] - s[a]),
                   float(spacing[a] + spacing[b]), crossed)
