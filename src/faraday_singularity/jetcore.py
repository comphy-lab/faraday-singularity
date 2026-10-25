"""The core of the upward jet after the cut and the split of the step's Courant limit: the resolution
policy of the events block's ``jetcore`` part (``events.JetCoreSettings``), in plain numpy.

The science of a run after the cut is the upward liquid jet the gas cut launched: its base (the
apex ``A = (r_j, z_b)`` of the gas groove around it, the point the jet's observables use), the
walls that feed it and its stem. The core ``K`` (:func:`core_mask`) is that part of the open chain:
its nodes within ``R_K = max(core_radius L, apex_extent R_apex)`` of ``A``, ``L = max(r_j, delta)``
and ``R_apex`` the groove's radius of curvature at the base, but not above the jet's tracked liquid
neck (the root of the head on its tip, :func:`upward_neck`), and the walls beyond the apex within
``wall_radius L`` of ``A``. Without a base of the jet (no resolved bump) the whole open chain is
the core. Every other node of the free surface (the head on the jet's tip, the bubble, the downward
jet, the ring) is the far region, which stays in the domain.

The step's Courant limit (``reconnect.courant_limit``: a step moves no node by more than
``courant`` of its element) is split (:func:`courant_split`): ``min(courant min_K h/|u|,
far_courant min_F h/|u|)``. With ``far_courant`` equal to ``courant`` it is the limit of every node,
to the bit (a positive factor commutes with the minimum of doubles).

The far region is meshed under simplifying assumptions, every feature kept in the domain
(:func:`region_class`, :func:`split_refine`, :func:`far_kink_regions`; the driver's
``CollapseDriver._feature_regions``): every far region's target at least ``far_floor``; the guard's
far kinks (an element of the far region whose tangent turns by more than ``edge_turn_refine``) in a
region of their own per chain, recentred on the worst element at each request instead of grown,
at the floor, below it only while the turn exceeds ``far_yield_turn`` (a yield, halving per
rebuild); the far regions refined beyond ``far_refine_ratio`` of their targets; the ring's kinks no
finer than the column. The tracked necks' regions are exempt, so that a neck is followed to its
landing. Where the mesh of the ring cannot represent a feature (an element of the ring turning
beyond the guard's limit, on a fresh or a stepped mesh), the driver rebuilds at once with that
element refined to the size its turn asks for, in a far kink of its own
(``CollapseDriver._ring_refine``, ``refined``, ``kink_ring_r<n>_far``) that follows its feature
keeping its size until the feature asks no more; every other region as it was; while such a region
remains, every far kink of the ring follows its own feature and a new feature has a region of its
own (``kink_ring_m<n>_far``), so that a feature never loses its refinement at once because another
one asks.

Before the piercing (:func:`prep_regions`), once the bubble is shorter than ``prep_gap delta``, its
bottom and the downward jet's tip are refined so that the ring cut (``ring.ring_cut_geometry``,
which needs the splice radius ``rho`` at least ``ring.RHO_MIN_NODES`` element lengths at both
splices) admits ``rho = prep_rho s_t``, ``s_t`` the radius of the jet's shoulder, with a margin of
``PREP_MARGIN``; these regions are exempt from the far floor and their nodes are stepped at the
core's Courant number (:func:`prep_mask`).
"""

from __future__ import annotations

import math
import re

import numpy as np

# jetcore.txt, with the events block's jetcore.enabled: a row per step after the cut, the step and
# what set it (``limiter``: the candidate of the driver's step equal to it, ``courant-core`` or
# ``courant-far`` for the split Courant limit), the split limit (``dt_courant`` the one the step
# took, ``dt_core``, ``dt_far``, which binds, the far Courant number, the node of each class that
# sets its part: element length, speed, place and chain), the core (``L``, ``R_K``, the base ``A``,
# the height of the neck that caps it, its nodes and the far nodes)
COLUMNS = ("step", "time", "tau", "tau_p", "dt", "limiter", "dt_courant", "dt_core", "dt_far",
           "binds", "far_courant", "h_core", "U_core", "r_core", "z_core", "chain_core", "h_far",
           "U_far", "r_far", "z_far", "chain_far", "L", "R_K", "A_r", "A_z", "z_cap",
           "core_nodes", "far_nodes", "far_regions", "far_floored", "far_yields", "prep",
           "s_t")


def upward_neck(points, end_types, apex_index: int, track_depth: float) -> dict | None:
    """The tracked liquid neck of the upward jet on the open chain (its nodes from the tip on the
    axis, index 0, to the wall): of ``reconnect.chain_necks``, one on the liquid side at least
    ``track_depth`` deep, between the tip and the jet's base (``0 < index < apex_index``), the one
    nearest the base; None without one."""
    from .reconnect import chain_necks

    best = None
    for nk in chain_necks(np.asarray(points, dtype=float), tuple(end_types)):
        i = int(nk["index"])
        if nk["side"] != "liquid" or nk["depth"] < track_depth or not 0 < i < apex_index:
            continue
        if best is None or i > best["index"]:
            best = {"index": i, "r": float(nk["r"]), "z": float(nk["z"]),
                    "depth": float(nk["depth"])}
    return best


def core_mask(points, end_types, apex: dict | None, *, delta: float, core_radius: float,
              wall_radius: float, apex_extent: float, track_depth: float) -> dict:
    """The core ``K`` on the open chain ``points`` (module docstring): ``mask`` by node, with the
    base ``A`` (``A_r``, ``A_z``, ``apex_index``), ``L``, ``R_K``, the capping neck (``neck``,
    ``z_cap``, NaN without one) and ``whole`` (no base of a jet: every node is core). ``apex`` is the
    open chain's jet base as the driver reads it (``CollapseDriver._apexes``' ``cavity_tip``: ``r``,
    ``z`` the base, ``index`` the apex's node, ``R_apex``)."""
    P = np.asarray(points, dtype=float).reshape(-1, 2)
    n = len(P)
    out = {"mask": np.ones(n, dtype=bool), "whole": True, "A_r": math.nan, "A_z": math.nan,
           "apex_index": -1, "L": math.nan, "R_K": math.inf, "neck": None, "z_cap": math.nan}
    if apex is None or n < 2:
        return out
    i_a = int(apex["index"])
    if not 0 < i_a < n:
        return out
    A = (float(apex["r"]), float(apex["z"]))
    L = max(A[0], float(delta)) if math.isfinite(A[0]) else float(delta)
    R_apex = float(apex.get("R_apex", math.nan))
    R_K = max(core_radius * L, apex_extent * R_apex if math.isfinite(R_apex) else 0.0)
    d = np.hypot(P[:, 0] - A[0], P[:, 1] - A[1])
    idx = np.arange(n)
    ball = d <= R_K
    neck = upward_neck(P, end_types, i_a, track_depth)
    if neck is not None:
        ball &= idx >= neck["index"]  # at or below the neck: the jet's stem, not its head
    walls = (idx > i_a) & (d <= wall_radius * L)
    out.update(mask=ball | walls, whole=False, A_r=A[0], A_z=A[1], apex_index=i_a, L=L, R_K=R_K,
               neck=neck, z_cap=neck["z"] if neck is not None else math.nan)
    return out


def node_ratios(seg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(h, |u|, h/|u|)`` by node of a chain, exactly as ``reconnect.courant_limit`` reads them:
    ``h`` twice the shorter of a node's two segments, ``inf`` where the node is at rest."""
    r, z = np.asarray(seg["r"], dtype=float), np.asarray(seg["z"], dtype=float)
    ds = np.hypot(np.diff(r), np.diff(z))
    h = 2.0 * np.minimum(np.r_[ds[0], ds], np.r_[ds, ds[-1]])
    u = np.hypot(np.asarray(seg["u_r"], dtype=float), np.asarray(seg["u_z"], dtype=float))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = np.where(u > 0.0, h / u, math.inf)
    return h, u, ratio


def _none() -> dict:
    return {"h_over_u": math.inf, "h": math.nan, "U": 0.0, "chain": -1, "r": math.nan,
            "z": math.nan, "nodes": 0}


def courant_split(segs, masks, *, core: float, far: float) -> dict:
    """The Courant limit of the step split between the core and the far nodes (module docstring):
    ``segs`` the chains as ``reconnect.courant_limit`` takes them (``chain`` 0 the open chain, 1 any
    other), ``masks`` the core nodes of each (a boolean array, or None for a chain wholly far);
    ``dt = min(core min_K h/|u|, far min_F h/|u|)``, with ``dt_core``, ``dt_far``, ``binds``
    (``core`` or ``far``, the core on a tie) and the node of each class that sets its part
    (``core``, ``far``: ``h/|u|``, ``h``, ``U``, place, chain and the class's node count)."""
    best = {"core": _none(), "far": _none()}
    for c, m in zip(segs, masks):
        if len(c["r"]) < 2:
            continue
        h, u, ratio = node_ratios(c)
        k = np.zeros(len(ratio), dtype=bool) if m is None else np.asarray(m, dtype=bool)
        chain = 0 if c.get("open") else 1
        for cls, sel in (("core", k), ("far", ~k)):
            if not sel.any():
                continue
            best[cls]["nodes"] += int(sel.sum())
            j = int(np.flatnonzero(sel)[np.argmin(ratio[sel])])
            if ratio[j] < best[cls]["h_over_u"]:
                best[cls].update(h_over_u=float(ratio[j]), h=float(h[j]), U=float(u[j]),
                                 chain=chain, r=float(c["r"][j]), z=float(c["z"][j]))
    dt_core, dt_far = core * best["core"]["h_over_u"], far * best["far"]["h_over_u"]
    return {"dt": min(dt_core, dt_far), "dt_core": dt_core, "dt_far": dt_far,
            "binds": "core" if dt_core <= dt_far else "far", "core": best["core"],
            "far": best["far"], "core_courant": core, "far_courant": far}


# --- the far region's sizing ----------------------------------------------------------------------

PREP_MARGIN = 1.5  # the margin of the refinement before the piercing on the ring cut's check

# the classes of the regions of a rebuild after the cut with jetcore (region_class): the core's,
# resolved as without the part; the far region's, floored and refined at the far ratio; a tracked
# neck's, exempt from the floor (its region follows the neck to its landing); the pre-pierce
# refinement's, exempt too; the ring's frozen regions, sized at the ring cut
CORE, FAR, NECK, PREP, FROZEN = "core", "far", "neck", "prep", "frozen"
# the regions of the open chain by name: the jet's base is core whatever its place
OPEN_REGIONS = ("cavity_tip", "cavity_tip_cut", "kink_open")


def nearest_node(points, r: float, z: float) -> int:
    """The node of ``points`` nearest ``(r, z)``."""
    P = np.asarray(points, dtype=float).reshape(-1, 2)
    return int(np.argmin(np.hypot(P[:, 0] - r, P[:, 1] - z)))


def region_class(region: dict, core: dict, open_points) -> str:
    """The class of a region of a rebuild after the cut (``CORE``, ``FAR``, ``NECK``, ``PREP`` or
    ``FROZEN``): the ring's frozen regions frozen; the pre-pierce regions (``pierce_*``) and the
    tracked necks' (``neck_*``) exempt; the base of the upward jet (``jet_base_cavity_tip``) core;
    a region of the open chain's tip or a kink of the open chain (``OPEN_REGIONS``, the core's own
    kinks ``kink_open``) core where the node of the open chain nearest its centre is a core node
    (``core["mask"]``); every other region (the bubble's, the ring's kinks, a far kink,
    ``kink_<chain>_far``) far."""
    name = region["name"]
    if region.get("frozen"):
        return FROZEN
    if name.startswith("pierce_"):
        return PREP
    if name.startswith("neck_"):
        return NECK
    if name == "jet_base_cavity_tip":
        return CORE
    if name in OPEN_REGIONS and open_points is not None and len(open_points):
        k = nearest_node(open_points, float(region["r"]), float(region["z"]))
        return CORE if bool(np.asarray(core["mask"])[k]) else FAR
    return FAR


def split_refine(refine: dict, readings: dict, core: dict) -> tuple[dict, dict]:
    """The guard's elements to refine (``reconnect.interface_guard``'s ``refine``, per chain) split
    into the core's and the far region's: an element of the open chain is the core's where its
    midside node is a core node; every element of another chain (the bubble, the ring) is far."""
    near, far = {}, {}
    mask = np.asarray(core["mask"], dtype=bool)
    for chain, idx in refine.items():
        idx = np.asarray(idx, dtype=int)
        if chain == "open":
            mid = np.asarray(readings[chain]["edges"]["index"], dtype=int)[idx]
            inside = mask[np.clip(mid, 0, len(mask) - 1)]
            if inside.any():
                near[chain] = idx[inside]
            if (~inside).any():
                far[chain] = idx[~inside]
        else:
            far[chain] = idx
    return near, far


# with a region the driver refined on a chain (CollapseDriver._ring_refine), at most this many far
# kinks on that chain, each following its own feature (far_kink_regions)
MULTI_FAR_KINKS = 8


def far_kink_regions(previous: dict, readings: dict, refine: dict, *, t: float, hold,
                     h_far: float, h_floor: float, yield_turn: float, sizes: dict,
                     floors: dict | None = None) -> dict:
    """The far kink regions (``kink_<chain>_far``, keyed ``<chain>_far``) after a reading, from
    those of the last accepted step (``previous``): where the guard asks for a far element of a
    chain to be refined (``refine``, its far part, :func:`split_refine`), one region about the worst
    of them, recentred on it at every request with a radius of four of its chords (no growth), at
    ``max(chord/4, h_far, floors[chain])`` (``floors``: the ring's column size, say); while that
    element turns by more than ``yield_turn`` degrees the region yields below the floor, halving its
    last target down to ``h_floor`` each time the current mesh has met it (``sizes``: the size of
    each region in the current mesh), each halving a yield (``yields``, ``yield_t``). A region whose
    chain asks for none for ``hold(h_before)`` of time is released, its target back to the chord it
    met, as a core kink is (``reconnect.kink_regions``).

    On a chain with a region the driver refined itself on a freshly built mesh (``refined``, the
    ring's ``CollapseDriver._ring_refine``, keyed ``<chain>_r<n>_far``: the rim of a thin sheet
    that a rebuild at the yielded size could not represent), while one such region remains, every
    far kink of the chain follows its own feature: the elements that ask within the reach of a
    live region (its radius and four of their chords) are its own, and it is recentred on the
    worst of them,
    four of its chords wide, at the rule's target (a refined one keeping its own, never coarser),
    yielding past ``yield_turn`` by halving the finer of its target and its size in the mesh; an
    element beyond every reach gives a region of its own (``<chain>_m<n>_far``, ``multi``; at most
    ``MULTI_FAR_KINKS`` regions on the chain), and each region none of whose elements asks for
    ``hold(h_before)`` is released, coarsening by the rule's bounded factor. A feature so never
    loses its refinement at once because another one asks. Without a refined region every region
    is as the rule makes it above."""
    out = {name: dict(v) for name, v in previous.items()}
    floors = floors or {}
    touched = set()
    multi_chains = {g.get("chain") for g in out.values() if g.get("refined") or g.get("multi")}
    for chain, idx in refine.items():
        idx = np.asarray(idx, dtype=int)
        if not len(idx):
            continue
        e = readings[chain]["edges"]
        floor = max(h_far if h_far is not None else 0.0, floors.get(chain, 0.0), h_floor)
        multi = chain in multi_chains

        def renewed(g, name, k, *, kept):
            """The region ``name`` (its last entry ``g``, or None) about element ``k``."""
            L, turn = float(e["L"][k]), float(e["turn"][k])
            target = max(0.25 * L, floor)
            yields, yield_t = (g.get("yields", 0), g.get("yield_t")) if g is not None else (0, None)
            live = g is not None and not g["released"]
            if live:
                last = float(g["target"])
                size = sizes.get(name)
                if multi and size is not None:
                    last = min(last, size)  # the finer of its target and its size in the mesh
                if turn > yield_turn:
                    if size is not None and size <= last * (1.0 + 1e-9) and last > h_floor:
                        target = max(0.5 * last, h_floor)  # the mesh has met the last: halve it
                        yields, yield_t = yields + 1, t
                    else:
                        target = min(target, last)  # not met yet: kept
                elif kept:
                    target = min(target, last)  # the driver's refinement: kept
            entry = {"name": name, "chain": chain, "r": float(e["mid"][k, 0]),
                     "z": float(e["mid"][k, 1]), "radius": 4.0 * L, "target": target,
                     "h_before": max(L, g["h_before"]) if live else L, "t_set": t,
                     "released": False, "turn": turn, "yields": yields, "yield_t": yield_t,
                     "far": True}
            for mark in ("refined", "multi"):
                if live and g.get(mark):
                    entry[mark] = True
            return entry

        def within(key, pool):
            g = out[key]
            reach = float(g["radius"]) + 4.0 * e["L"][pool]
            return pool[np.hypot(e["mid"][pool, 0] - float(g["r"]),
                                 e["mid"][pool, 1] - float(g["z"])) <= reach]

        pool = idx
        for key in sorted(k for k, g in out.items() if g.get("chain") == chain
                          and not g["released"] and (multi or g.get("refined"))):
            near = within(key, pool)
            if not len(near):
                continue
            g = out[key]
            out[key] = renewed(g, g["name"], int(near[np.argmax(e["turn"][near])]),
                               kept=bool(g.get("refined")))
            touched.add(key)
            pool = np.setdiff1d(pool, near)
        if not len(pool):
            continue
        if multi:  # each feature beyond every reach a region of its own
            live = sum(1 for g in out.values() if g.get("chain") == chain and not g["released"])
            while len(pool) and live < MULTI_FAR_KINKS:
                n = [int(x.group(1)) for x in
                     (re.fullmatch(rf"{re.escape(chain)}_m(\d+)_far", k) for k in out) if x]
                key = f"{chain}_m{max(n, default=0) + 1}_far"
                out[key] = renewed(None, f"kink_{key}", int(pool[np.argmax(e["turn"][pool])]),
                                   kept=False)
                out[key]["multi"] = True
                touched.add(key)
                pool = np.setdiff1d(pool, within(key, pool))
                live += 1
            continue
        key, name = f"{chain}_far", f"kink_{chain}_far"
        out[key] = renewed(out.get(key), name, int(pool[np.argmax(e["turn"][pool])]), kept=False)
    for key, g in out.items():
        chain = g["chain"]
        if g["released"] or key in touched:
            continue
        if chain not in multi_chains and chain in refine and len(refine[chain]):
            continue
        if t >= g["t_set"] + hold(g["h_before"]):
            g.update(released=True, target=g["h_before"])
    return out


def prep_regions(bubble_seg, *, delta: float, prep_gap: float | None, prep_rho: float,
                 h_floor: float) -> list[dict]:
    """The regions of the refinement before the piercing (module docstring): none unless the
    bubble (``bubble_seg``, its nodes upwards from its bottom on the axis to the downward jet's tip
    on the axis) is shorter than ``prep_gap delta``; else ``pierce_tip`` about the jet's tip and
    ``pierce_bottom`` about the bubble's bottom, each of radius ``s_t`` (the jet's shoulder,
    ``ring.jet_shoulder``) at ``max(prep_rho s_t / (RHO_MIN_NODES PREP_MARGIN), h_floor)``."""
    from .ring import RHO_MIN_NODES, jet_shoulder

    if prep_gap is None or bubble_seg is None:
        return []
    P = np.column_stack([np.asarray(bubble_seg["r"], dtype=float),
                         np.asarray(bubble_seg["z"], dtype=float)])
    if len(P) < 8:
        return []
    length = float(P[-1, 1] - P[0, 1])
    if not length < prep_gap * delta:
        return []
    s_t = float(P[jet_shoulder(P), 0])
    if not s_t > 0.0:
        return []
    size = max(prep_rho * s_t / (RHO_MIN_NODES * PREP_MARGIN), h_floor)
    return [{"name": name, "target": size, "r": 0.0, "z": float(P[k, 1]), "radius": s_t,
             "ell": s_t, "floor": h_floor, "s_t": s_t, "bubble_length": length}
            for name, k in (("pierce_tip", -1), ("pierce_bottom", 0))]


def prep_mask(seg, regions: list[dict]) -> np.ndarray | None:
    """The nodes of a chain within any of the regions of :func:`prep_regions` (stepped at the
    core's Courant number); None without a region."""
    if not regions:
        return None
    r, z = np.asarray(seg["r"], dtype=float), np.asarray(seg["z"], dtype=float)
    out = np.zeros(len(r), dtype=bool)
    for g in regions:
        out |= np.hypot(r - g["r"], z - g["z"]) <= g["radius"]
    return out
