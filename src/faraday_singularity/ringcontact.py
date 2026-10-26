"""Read local contacts of the ring without changing its topology.

A crossing of the rebuild spline within two neighbouring elements can terminate a
gas wedge whose control polygon remains simple. Fill that wedge with the existing
groove surgery. A liquid kink instead asks for refinement: never fill or rupture
its film. Its rebuild gains no liquid and carries no momentum, within the ring's
calibrated noise floors; the upward jet's base is checked separately.
"""
from __future__ import annotations

import math

import numpy as np


class RingStepContact(RuntimeError):
    """A local polygon contact of a candidate step, before the driver accepts it."""
    def __init__(self, reading: dict):
        super().__init__("a step crossed a local part of the ring")
        self.reading = reading


def loop_node_map(source, target) -> np.ndarray | None:
    """Locate source node identities in a cyclic rotation of the same ordered walk.

    Cache indices are meaningful only without a rebuild. Duplicate, missing,
    reversed or reordered identities keep the contact unreadable.
    """
    source, target = np.asarray(source), np.asarray(target)
    if len(source) != len(target) or not len(source) \
            or len(set(source.tolist())) != len(source) \
            or len(set(target.tolist())) != len(target):
        return None
    positions = {int(k): i for i, k in enumerate(target)}
    if any(int(k) not in positions for k in source):
        return None
    indices = np.array([positions[int(k)] for k in source])
    return indices if np.all(np.diff(indices) % len(indices) == 1) else None


def step_contact(points) -> dict | None:
    """One crossing within two neighbouring quadratic elements, otherwise None."""
    from .ring import loop_crossings
    from shapely.geometry import LineString

    P = np.asarray(points, dtype=float)
    n = len(P)
    if n < 6 or n % 2:
        return None
    hits = loop_crossings(P)
    if len(hits) != 1:
        return None
    a, b = hits[0]
    if (b-a) % n > (a-b) % n:
        a, b = b, a
    span = (b-a) % n+1
    if span > 4 or (a % 2+span-1)//2 > 1:
        return None
    hit = LineString(P[[a, (a+1) % n]]).intersection(LineString(P[[b, (b+1) % n]]))
    if hit.geom_type != "Point":
        return None
    indices = ((a+np.arange(span+1)) % n).tolist()
    return {"trigger": "stepped-polygon", "chain": "ring", "contact": "polygon",
            "node_count": n, "nodes": [a, (a+1) % n, b, (b+1) % n],
            "local_indices": indices, "after_local": P[indices].tolist(),
            "contact_r": float(hit.x), "contact_z": float(hit.y), "crossings": 1}


def step_wedge(points, reading: dict, *, stop_elements: float) -> tuple[dict | None, str]:
    """Certify the pre-step pocket as gas; never operate on the crossed polygon."""
    from .groove import gas_share, side_of
    from .ring import floor_wedge, loop_spline_contact

    P = np.asarray(points, dtype=float)
    n = len(P)
    if n != reading["node_count"] or loop_spline_contact(P) is not None:
        return None, "unreadable"
    indices = reading["local_indices"]
    phase = side_of(gas_share(P[indices], P))
    if phase != "gas":
        return None, phase
    apex = indices[(len(indices)-1)//2]
    return floor_wedge(P, apex, stop_elements=stop_elements), phase


def spline_wedge(points, stop: dict, *, stop_elements: float) -> dict | None:
    """A gas wedge at a single spline crossing within two adjacent quadratic elements.

    The actual crossing, the simple polygon and the phase of its shorter pocket
    are checked again. Nonlocal crossings and liquid pockets remain stops.
    """
    from .groove import gas_share, side_of
    from .ring import floor_wedge, loop_halves, loop_spline_contact

    if stop.get("reason") != "self-contact-approach" or stop.get("trigger") != "spline-contact" \
            or stop.get("chain") != "ring" or stop.get("contact") != "spline" \
            or stop.get("crossings") != 1:
        return None
    P = np.asarray(points, dtype=float).reshape(-1, 2)
    n = len(P)
    nodes = stop.get("nodes") or []
    if n < 6 or n % 2 or len(nodes) != 4 or any(not 0 <= int(k) < n for k in nodes):
        return None
    hit = loop_spline_contact(P)
    if hit is None or hit["what"] != "spline" or hit["count"] != 1 \
            or tuple(hit["nodes"]) != tuple(nodes):
        return None
    # The interval starts can straddle the seam. Each interval is one node chord.
    a, a1, b, b1 = (int(k) for k in nodes)
    if a1 != (a + 1) % n or b1 != (b + 1) % n:
        return None
    if (b - a) % n > (a - b) % n:
        a, b = b, a
    span = (b - a) % n + 1
    if span > 4 or (a % 2 + span - 1) // 2 > 1:  # at most two quadratic elements
        return None
    split = len(loop_halves(P)[0]) - 1  # the actual split is on a vertex, not necessarily n//2
    if span == 1 and a in (0, split - 1, split, n - 1):
        return None  # the two open splines use phantom controls at their ends
    # A cubic interval can loop within itself. Its two end nodes alone
    # enclose no pocket; its four Catmull-Rom control nodes read its phase.
    Q = P[(a - 1 + np.arange(4)) % n] if span == 1 else P[(a + np.arange(span + 1)) % n]
    if side_of(gas_share(Q, P)) != "gas":
        return None
    apexes = (a, (a + 1) % n) if span == 1 else ((a + span // 2) % n,)
    wedges = [floor_wedge(P, i, stop_elements=stop_elements) for i in apexes]
    return max((w for w in wedges if w is not None), key=lambda w: w["turn"], default=None)


def liquid_reading(points, stop: dict | None) -> dict | None:
    """Read a liquid kink as a request to halve the pair's element size.

    Only a simple ring and a liquid pocket qualify. A resolved fold, a spline
    crossing, a gas kink and a contact without a phase reading remain stops.
    """
    from .groove import gas_share, side_of
    from .ring import loop_spacing
    from shapely.geometry import LinearRing

    if stop is None or stop.get("reason") != "self-contact-approach" \
            or stop.get("trigger") != "fold-gap" or stop.get("chain") != "ring" \
            or stop.get("contact") != "kink" or stop.get("side") != "liquid":
        return stop
    P = np.asarray(points, dtype=float).reshape(-1, 2)
    n = len(P)
    pair = stop.get("fold_pair") or []
    if n < 6 or n % 2 or len(pair) != 2 or any(not 0 <= int(k) < n for k in pair) \
            or not LinearRing(P).is_simple:
        return stop
    a, b = sorted(int(k) for k in pair)
    if a == b:
        return stop
    Q = P[a:b + 1]
    other = np.vstack([P[b:], P[:a + 1]])
    if np.linalg.norm(np.diff(other, axis=0), axis=1).sum() < \
            np.linalg.norm(np.diff(Q, axis=0), axis=1).sum():
        Q = other
    if side_of(gas_share(Q, P)) != "liquid":
        return stop
    h = float(loop_spacing(P)[a] + loop_spacing(P)[b])
    mid = 0.5 * (P[a] + P[b])
    return {"reason": "interface-unresolved", "trigger": "liquid-kink", "chain": "ring",
            "L": h, "r": float(mid[0]), "z": float(mid[1]),
            "liquid_contact": dict(stop)}


def sample(problem, grid: dict, reservoir_depth: float) -> dict:
    """The existing box integrals and polyline volumes, with the ring counted as gas."""
    from .reconnect import Chain
    from .ring import closed
    from .surgery import _named_chains, box_integrals, box_polyline_budgets

    named, _ = _named_chains(problem)
    loops = [np.column_stack([x["r"], x["z"]]) for x in problem.interface_loops()]
    chains = [c for _, c in named] + [Chain(closed(L), ("loop", "loop")) for L in loops]
    return {"box": box_integrals(problem, grid, cutoff=True),
            "poly": box_polyline_budgets(chains, grid["z"], grid["R"], reservoir_depth,
                                         r_centre=grid["r_c"]), "loops": len(loops)}


def closures(before: dict, after: dict, sigma) -> dict:
    """Zero added/removed volume and momentum, with the ring surgery's existing floors.

    No projection or surgery is made. The two hard balances use five times the
    calibrated ring noise, as R1 and R2 do; a nonfinite value is a failure.
    """
    from .ringsurgery import RING_NOISE_FLOORS

    values = {"C_V": abs(after["poly"]["V_liquid"] - before["poly"]["V_liquid"]),
              "C_P": abs(after["box"]["Pz"] - before["box"]["Pz"])}
    limits = {"C_V": RING_NOISE_FLOORS["R1"][1] * sigma.sigma_V,
              "C_P": RING_NOISE_FLOORS["R2"][1] * sigma.sigma_Pz}
    hard = [k for k in values if not math.isfinite(values[k]) or not values[k] <= limits[k]]
    if before["loops"] != 1 or after["loops"] != 1:
        hard.append("C_loop")
    return {"values": values, "limits": limits, "hard": hard, "V_added": 0.0, "V_removed": 0.0}
