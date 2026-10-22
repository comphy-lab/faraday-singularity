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
landing.
"""

from __future__ import annotations

import math

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
           "core_nodes", "far_nodes", "far_regions", "far_floored", "far_yields")


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
    met, as a core kink is (``reconnect.kink_regions``)."""
    out = {name: dict(v) for name, v in previous.items()}
    floors = floors or {}
    for chain, idx in refine.items():
        idx = np.asarray(idx, dtype=int)
        if not len(idx):
            continue
        e = readings[chain]["edges"]
        k = int(idx[np.argmax(e["turn"][idx])])
        L, turn = float(e["L"][k]), float(e["turn"][k])
        r, z = float(e["mid"][k, 0]), float(e["mid"][k, 1])
        key, name = f"{chain}_far", f"kink_{chain}_far"
        floor = max(h_far if h_far is not None else 0.0, floors.get(chain, 0.0), h_floor)
        target = max(0.25 * L, floor)
        g = out.get(key)
        yields, yield_t = (g.get("yields", 0), g.get("yield_t")) if g is not None else (0, None)
        if g is not None and not g["released"] and turn > yield_turn:
            last = float(g["target"])
            size = sizes.get(name)
            if size is not None and size <= last * (1.0 + 1e-9) and last > h_floor:
                target = max(0.5 * last, h_floor)  # the mesh has met the last target: halve it
                yields, yield_t = yields + 1, t
            else:
                target = min(target, last)  # not met yet: kept
        out[key] = {"name": name, "chain": chain, "r": r, "z": z, "radius": 4.0 * L,
                    "target": target, "h_before": max(L, g["h_before"]) if g is not None
                    and not g["released"] else L, "t_set": t, "released": False,
                    "turn": turn, "yields": yields, "yield_t": yield_t, "far": True}
    for key, g in out.items():
        chain = g["chain"]
        if chain in refine and len(refine[chain]) or g["released"]:
            continue
        if t >= g["t_set"] + hold(g["h_before"]):
            g.update(released=True, target=g["h_before"])
    return out
