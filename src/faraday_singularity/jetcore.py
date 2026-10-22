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
           "core_nodes", "far_nodes")


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
