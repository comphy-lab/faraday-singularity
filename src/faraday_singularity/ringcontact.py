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
    """The existing box integrals and polyline volumes, with the ring counted as gas; and, for
    :func:`closures`, the readings of the box's cells (:func:`cell_readings`: each cell's
    midpoint sample, and for each cell within a diagonal of a free surface its liquid fraction
    and its velocity at its liquid part's centroid), which the record does not keep
    (:func:`plain_sample`)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    from .reconnect import Chain, closed_section
    from .ring import closed
    from .surgery import _named_chains, box_integrals, box_polyline_budgets, sample_fields

    named, _ = _named_chains(problem)
    loops = [np.column_stack([x["r"], x["z"]]) for x in problem.interface_loops()]
    chains = [c for _, c in named] + [Chain(closed(L), ("loop", "loop")) for L in loops]
    gas = unary_union([Polygon(closed_section(c.points, c.end_types, reservoir_depth)).buffer(0)
                       for c in chains])

    def sampler(r, z):
        f, found = sample_fields(problem, r, z, ("velocity_y",))
        return np.where(found, f["velocity_y"], 0.0), found

    return {"box": box_integrals(problem, grid, cutoff=True),
            "poly": box_polyline_budgets(chains, grid["z"], grid["R"], reservoir_depth,
                                         r_centre=grid["r_c"]), "loops": len(loops),
            "_cells": cell_readings(grid, gas, sampler)}


def plain_sample(sampled: dict) -> dict:
    """A sample of :func:`sample` as the record keeps it: without its cells' readings."""
    return {k: v for k, v in sampled.items() if not k.startswith("_")}


def _cells(grid: dict, index, points: int = 3):
    """The polar cells ``index`` of ``grid`` (``surgery.site_grid``) as polygons: each within half
    a radial step of its sample's distance from the centre and half an angular step of its
    angle (``points`` along each arc), at ``r >= 0`` (a cell whose midpoint lies off the axis can
    reach across it; the domain ends there)."""
    import shapely

    h_rho, h_th = grid["R"] / grid["n"], 2.0 * math.pi / grid["n"]
    half = shapely.box(0.0, grid["z"] - 2.0 * grid["R"], grid["r_c"] + 2.0 * grid["R"],
                       grid["z"] + 2.0 * grid["R"])
    out = []
    for k in np.asarray(index, dtype=int):
        rho = float(grid["rho"][k])
        th = math.atan2(float(grid["zz"][k]) - grid["z"], float(grid["r"][k]) - grid["r_c"])
        t = np.linspace(th - 0.5 * h_th, th + 0.5 * h_th, points)
        ring = np.column_stack([np.cos(t), np.sin(t)])
        P = np.vstack([ring * (rho + 0.5 * h_rho), (ring * max(rho - 0.5 * h_rho, 0.0))[::-1]])
        out.append(shapely.Polygon(P + np.array([grid["r_c"], grid["z"]])).intersection(half))
    return np.asarray(out, dtype=object)


def cell_readings(grid: dict, gas, sampler) -> dict:
    """The readings of the cells of ``grid`` for :func:`momentum_change`, ``gas`` the gas the
    polylines bound (a shapely geometry) and ``sampler(r, z)`` the axial velocity and whether each
    point lies in the liquid: each cell's midpoint sample (``found``, ``uz``), and for each cell
    whose midpoint lies within a cell's diagonal of the gas's boundary (``near``, the cells a
    rebuild can move a free surface across) the volume of revolution of the cell (``V_cell``) and
    of its liquid part (``V_liquid``: the cell less the gas) and the liquid part's axial momentum
    (``P_liquid``, ``int u_z 2 pi r dA``), each by the seven-point, degree-five triangle rule of
    ``axisrepair.polygon_quadrature_points`` with the velocity sampled at its points (the points
    the finite element does not find in the liquid take the mean of those it does;
    ``read_liquid`` false where it finds none of a part that has liquid)."""
    import shapely

    from .axisrepair import polygon_quadrature_points

    uz, found = sampler(grid["r"], grid["zz"])
    h_rho, h_th = grid["R"] / grid["n"], 2.0 * math.pi / grid["n"]
    disc = shapely.Point(grid["r_c"], grid["z"]).buffer(grid["R"] * (1.0 + 2.0 / grid["n"]),
                                                         quad_segs=256)
    local = gas.intersection(disc)
    diag = np.hypot(h_rho, np.asarray(grid["rho"]) * h_th)
    if local.is_empty or not len(diag):
        near = np.zeros(0, dtype=int)
    else:  # the boundary sampled at most half the finest diagonal apart: a distance within it
        from scipy.spatial import cKDTree

        step = 0.5 * float(diag.min())
        B = shapely.get_coordinates(shapely.segmentize(local.boundary, step))
        d, _ = cKDTree(B).query(np.column_stack([grid["r"], grid["zz"]]))
        near = np.flatnonzero(d <= diag + step)
    n = len(near)
    V_cell, V_liq, P_liq = np.zeros(n), np.zeros(n), np.zeros(n)
    read = np.ones(n, dtype=bool)
    if n:
        cells = _cells(grid, near)
        liquid = shapely.difference(cells, local)
        points, weights, owner = [], [], []
        for i, (cell, part) in enumerate(zip(cells, liquid)):
            V_cell[i] = float(np.sum(polygon_quadrature_points(cell)[1]))
            if part.is_empty or part.area <= 0.0:
                continue
            X, dV, _ = polygon_quadrature_points(part)
            points.append(X)
            weights.append(dV)
            owner.append(np.full(len(X), i))
        if points:
            X, dV, own = np.vstack(points), np.concatenate(weights), np.concatenate(owner)
            u, ok = sampler(X[:, 0], X[:, 1])
            V_liq = np.bincount(own, weights=dV, minlength=n)
            V_ok = np.bincount(own, weights=np.where(ok, dV, 0.0), minlength=n)
            P_ok = np.bincount(own, weights=np.where(ok, dV * u, 0.0), minlength=n)
            with np.errstate(divide="ignore", invalid="ignore"):
                P_liq = np.where(V_ok > 0.0, P_ok * V_liq / V_ok, 0.0)
            read = (V_liq <= 0.0) | (V_ok > 0.0)
    return {"found": found, "uz": uz, "near": near, "V_cell": V_cell, "V_liquid": V_liq,
            "P_liquid": P_liq, "read_liquid": read}


def momentum_change(grid: dict, before: dict, after: dict) -> dict:
    """The axial momentum the rebuild moved within the box (``surgery.box_integrals``' midpoint
    rule, weighted by the box's cutoff ``phi``), its cells read consistently across the rebuild
    (:func:`cell_readings`): a cell whose midpoint lies in the liquid both times counts the change
    of its velocity there (``matched``); a cell whose midpoint the rebuild moved across a free
    surface (in the liquid on one side only, ``crossed``) counts the change of its liquid part's
    momentum, ``phi dV/V_cell (P_a - P_b)``, ``P`` the volume integral of the axial velocity over
    the cell less the polylines' gas on each side (revolved, ``2 pi r dA``), instead of the whole
    cell that the midpoint rule moves across the surface with its sample; a crossed cell either
    side could not read so (not near a surface by its polylines, or none of its liquid part's
    points in the liquid) counts the midpoint rule's change, as before (``unresolved``). A value
    that is not finite, a sample the finite element found in the liquid on either side included,
    makes the change not finite (C_P then fails, :func:`closures`). The
    midpoint rule's change itself is kept as ``grid``. r1 at step 7817 and r1s4 at 8208 (bf80e74):
    the lip's refine moved one heavy cell's midpoint (dV 1.4e-11 at 2.7e-3 from the lip, 330-380
    in u_z) across the free surface of the plug, the surface through the cell's middle both
    times: 3.9e-9 and 3.6e-9 of the grid's change, the matched cells 1.8e-10 and -4.1e-11, the
    limit 1.62e-9."""
    cb, ca = before["_cells"], after["_cells"]
    w = np.asarray(grid["dV"], dtype=float) * np.asarray(grid["phi"], dtype=float)
    both = cb["found"] & ca["found"]
    crossed = np.flatnonzero(cb["found"] != ca["found"])
    matched = float(np.sum(w[both] * (ca["uz"][both] - cb["uz"][both])))
    where = [{int(k): i for i, k in enumerate(c["near"])} for c in (cb, ca)]
    cells = []
    for k in crossed:
        k = int(k)
        grid_k = float(w[k] * (ca["uz"][k] - cb["uz"][k]))
        sides = []
        for c, at in zip((cb, ca), where):
            i = at.get(k)
            ok = i is not None and bool(c["read_liquid"][i]) and float(c["V_cell"][i]) > 0.0
            sides.append((float(c["V_liquid"][i]), float(c["P_liquid"][i]), float(c["V_cell"][i]))
                         if ok else None)
        resolved = None not in sides
        if resolved:
            scale = float(w[k]) / sides[0][2]  # the midpoint rule's weight over the cell's volume
            value = scale * (sides[1][1] - sides[0][1])
        else:
            value = grid_k
        cells.append({"r": float(grid["r"][k]), "z": float(grid["zz"][k]),
                      "rho": float(grid["rho"][k]), "w": float(w[k]),
                      "uz": [float(cb["uz"][k]), float(ca["uz"][k])],
                      "fraction": [s[0] / s[2] if s else None for s in sides],
                      "uz_liquid": [s[1] / s[0] if s and s[0] > 0.0 else None for s in sides],
                      "grid": grid_k, "resolved": resolved, "value": value})
    values = [c["value"] for c in cells]
    # every sample the finite element found in the liquid must be finite, on either side, before
    # any of them is read (a crossed cell's midpoint included, which its liquid part replaces)
    samples_finite = all(bool(np.all(np.isfinite(c["uz"][c["found"]]))) for c in (cb, ca))
    finite = samples_finite and math.isfinite(matched) and all(math.isfinite(v) for v in values)
    crossed_sum = math.fsum(values) if finite else math.nan
    change = matched + crossed_sum if finite else math.nan
    grid_change = float(np.sum(np.where(ca["found"], w * ca["uz"], 0.0))
                        - np.sum(np.where(cb["found"], w * cb["uz"], 0.0)))
    return {"change": change, "matched": matched, "crossed": crossed_sum,
            "grid": grid_change, "cells_crossed": len(cells),
            "cells_unresolved": sum(1 for c in cells if not c["resolved"]),
            "largest_crossed": max(cells, key=lambda c: abs(c["grid"])
                                   if math.isfinite(c["grid"]) else math.inf) if cells else None}


def closures(before: dict, after: dict, sigma, grid: dict | None = None) -> dict:
    """Zero added/removed volume and momentum, with the ring surgery's existing floors.

    No projection or surgery is made. The two hard balances use five times the
    calibrated ring noise, as R1 and R2 do; a nonfinite value is a failure. With the
    box's ``grid`` and samples that hold their cells (:func:`sample`), C_P is the
    momentum the rebuild moved read consistently across it (:func:`momentum_change`:
    a cell whose midpoint crossed a free surface counts the change of its liquid part's
    momentum, not its whole volume); the midpoint rule's change is kept as ``C_P_grid``.
    """
    from .ringsurgery import RING_NOISE_FLOORS

    values = {"C_V": abs(after["poly"]["V_liquid"] - before["poly"]["V_liquid"]),
              "C_P": abs(after["box"]["Pz"] - before["box"]["Pz"])}
    split = None
    if grid is not None and "_cells" in before and "_cells" in after:
        split = momentum_change(grid, before, after)
        # the reading replaces the midpoint rule's change, never a change that is not finite
        values["C_P"] = abs(split["change"]) if math.isfinite(values["C_P"]) else math.nan
    limits = {"C_V": RING_NOISE_FLOORS["R1"][1] * sigma.sigma_V,
              "C_P": RING_NOISE_FLOORS["R2"][1] * sigma.sigma_Pz}
    hard = [k for k in values if not math.isfinite(values[k]) or not values[k] <= limits[k]]
    if before["loops"] != 1 or after["loops"] != 1:
        hard.append("C_loop")
    out = {"values": values, "limits": limits, "hard": hard, "V_added": 0.0, "V_removed": 0.0}
    if split is not None:
        out["C_P_grid"] = abs(after["box"]["Pz"] - before["box"]["Pz"])
        out["C_P_split"] = split
    return out
