"""The rupture of a thinning liquid lip of the ring by the removal of its rim (``sheet_rupture``).

In the ring stage the downward jet's head, landed on the plug, drives a tongue of liquid up into the
ring's gas whose top curls outward into a lip, a sheet of liquid between two faces of the ring's gas
that stretches and thins as it expands (r1 and r1h of the delta triple from tau/delta^2 25.19: its
thickness at its thinnest section from 4.7e-5 to 2.0e-5 within 0.15 tau/delta^2, the faces closing at
about the extensional rate ``h (d_s u_s + u_r/r)``). The lip is a part of the ring's nuisance
(decision 22): it is resolved by ``driver.SHEET_ELEMENTS`` elements across its thickness down to the
run's floor (``driver.CollapseDriver._ring_sheet_reading``, the ring refine with its zero-removal
budgets), and once its thinnest section reaches ``h_s = jetcore.sheet_rupture_floors h_floor`` while
its thickness falls over its recent samples (:func:`lip_thinning`) it is ruptured there: the rim
beyond the section, liquid, is removed from the domain and logged, as the head of a drop is
(decisions 19 and 20), and the ring's gas joins across the cut.

The surgery is a rebuild the project owns, as the ring surgeries and the fill of a groove are
(``ringsurgery.py``, ``groovesurgery.py``), the fill's mirror: liquid removed instead of gas filled.

1. **Admission**: a gas cut made, the ring stage before the ring's removal with its one loop and
   the open chain alone, a request of kind ``sheet-contact`` on the ring whose rupture is due
   (``sheet_rupture``), a positive last step. ``precut_e<k>`` kept.
2. **Geometry** (:func:`sheet_rupture_geometry`): the lip's pair read again as the ring's liquid
   fold (``groove.side_folds``) at the request's pair, its pocket liquid; the rim the pocket beyond
   the section, closed on the remaining lip by a half-circle cap of diameter the section's width
   (``CAP_POINTS`` points, bulging into the rim); refused (``surgery.CutRefused``) unless the new
   loop is simple, the removed region liquid and simple, clear of the open chain by
   ``OPEN_CLEARANCE`` section widths, and at most ``MAX_REMOVED_V_B`` of the bubble's volume.
3. **Budgets before**, within ``max(ring.box_radius delta, 2 L_rim)`` of the rim's centroid (a disc
   at the site, ``surgery.site_grid``; the polylines clipped to it): the box's on the grid, the
   global ones, and the rim's volume, axial momentum and kinetic energy on the rim itself
   (:func:`rim_quadrature`: every point located and its volume the rim's within the R1 tolerance,
   or the rupture is refused; the grid's reading of the rim reported beside it).
4. **Remesh at once** from the planned geometry (the template's ``pending_ring_geometry``: the open
   chain as it is and the new loop), under the driver's focus with the cap's region
   (``sheet_e<k>``, at ``max(h_floor, w / CAP_ELEMENTS)``, ``w`` the section's width); the transfer
   places the interface nodes by projection onto the old interface (``FaradayProblem.
   _ring_transfer``). From here a failure is ``surgery.CutFailed``.
5. **Mesh check**: the chains and their ends as before, one loop, the axis spans unchanged; the
   nodes the transfer leaves unlocated those the geometry predicts (``ringsurgery._transfer_check``):
   of the interface on the cap, none of the bulk (the new liquid lies in the old) or the axis.
6. **Projection** (``surgery.project`` with ``groove.projection`` and ``groove.restart`` or the
   cut's, the run's last step as the history's, the time unchanged, the disc its box): ``Pi = 0`` on
   every free surface, the cap's included.
7. **Budgets after and closures** (``ringsurgery._ring_closures``, prefix ``S``, the removal a
   negative fill): S1 the liquid volume the disc loses against the rim's volume, S2 the axial
   momentum the transfer takes away against the rim's (its quadrature's), S3 the projection's momentum balance in the
   disc (the cutoff identity) and ``S3_box`` the disc's change of momentum by the projection, hard
   (``surgery.BudgetUnclosed``); S4-S6 (energies, area) flags; the base of the upward jet besides
   (``base_check``, the hard closure ``S_base``). Then the record, the driver's bookkeeping
   (``on_made``), ``postcut_e<k>`` and ``events/event_<k>.json`` (kind ``sheet-rupture``) with the
   removed V, Pz and KE, the rim's centroid, the section and the lip's thickness history.

A refusal leaves the problem unchanged, and a rupture whose hard closures miss leaves it holding a
rebuilt state that is not accepted: either is a resumable stop.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .events import EVENT_SCHEMA, PRECUT_EVENT, SHEET_CONTACT, SHEET_RUPTURE, SHEET_RUPTURE_CLASS
from .mesh import Focus, FocusRegion
from .reconnect import Chain, liquid_axis_spans

CAP_POINTS = 9  # the points of the half-circle cap that closes the remaining lip, its ends included
CAP_ELEMENTS = 4  # the cap's region: the section's width over this, at least the floor
REGION_RADIUS = 4.0  # the cap's region reaches this many section widths from the cap's middle
CAP_TOL = 0.25  # a new interface node within this times the section's width of the cap lies on it
OPEN_CLEARANCE = 4.0  # the removed region clear of the open chain by this many section widths
MAX_REMOVED_V_B = 1e-3  # the largest rim removed, in the bubble's volume at the gas cut
LIP_SAMPLES = 6  # the recent samples (t, thickness) of the lip the driver keeps
LIP_MIN_SAMPLES = 3  # a lip is read as thinning only over at least this many of them


class SheetRuptureRefused(RuntimeError):
    """The geometry of a rupture refused, with what its checks found (``checks``)."""

    def __init__(self, message: str, checks: dict):
        super().__init__(message)
        self.checks = checks


def lip_thinning(samples) -> dict:
    """Whether the lip's thickness falls over its recent samples (``samples``: ``[t, thickness,
    ...]``,
    in time order, the driver's last ``LIP_SAMPLES``): at least ``LIP_MIN_SAMPLES`` of them at
    distinct times, the least-squares slope of the thickness against time negative and the last
    sample below the first, as the drop rule cuts only a neck that closes. Returns ``thinning`` with
    the slope and the count read."""
    S = np.asarray([x[:2] for x in (samples or []) if len(x) >= 2], dtype=float).reshape(-1, 2)
    out = {"thinning": False, "samples": int(len(S)), "slope": math.nan}
    if len(S) < LIP_MIN_SAMPLES or not np.all(np.isfinite(S)):
        return out
    t, h = S[:, 0], S[:, 1]
    if not np.all(np.diff(t) > 0.0):
        return out
    tc = t - t.mean()
    slope = float(np.dot(tc, h - h.mean()) / np.dot(tc, tc))
    out.update(slope=slope, thinning=bool(slope < 0.0 and h[-1] < h[0]))
    return out


def _arc(P, a: int, b: int) -> list[int]:
    """The loop ``P``'s nodes from ``a`` to ``b`` (both included) the shorter way round in length,
    as the fold detector reads a pocket (``groove.side_folds``, ``groove.sheet_reading``): the walk
    from the lower index of the pair to the higher unless the walk through the seam is strictly
    shorter, whatever the nodes each holds (a refined lip holds many short elements)."""
    P = np.asarray(P, dtype=float).reshape(-1, 2)
    n = len(P)
    lo, hi = sorted((int(a) % n, int(b) % n))
    inner = list(range(lo, hi + 1))
    seam = list(range(hi, n)) + list(range(0, lo + 1))

    def length(walk):
        return float(np.sum(np.hypot(*np.diff(P[walk], axis=0).T)))

    walk = seam if length(seam) < length(inner) else inner
    return walk if walk[0] == int(a) % n else walk[::-1]


def _cap(A, B, toward) -> np.ndarray:
    """The half circle from ``A`` to ``B`` of diameter ``|AB|``, bulging to the side of the line
    ``AB`` that ``toward`` lies on (``CAP_POINTS`` points, ``A`` and ``B`` included)."""
    A, B, toward = (np.asarray(x, dtype=float) for x in (A, B, toward))
    c, half = 0.5 * (A + B), 0.5 * (B - A)
    rho = float(np.hypot(*half))
    e = half / rho
    nrm = np.array([-e[1], e[0]])
    if float(np.dot(toward - c, nrm)) < 0.0:
        nrm = -nrm
    phi = np.linspace(math.pi, 0.0, CAP_POINTS)  # from A (phi = pi) to B (phi = 0)
    return c + rho * (np.cos(phi)[:, None] * e + np.sin(phi)[:, None] * nrm)


def sheet_rupture_geometry(P, pair, *, open_points, V_b: float, velocity=None) -> dict:
    """The rupture of the ring's loop ``P`` (counter-clockwise, its gas inside) at the lip's
    section ``pair`` (two nodes across the lip, its pocket of liquid the arc between them the
    shorter way round in length, :func:`_arc`, and the chord): the rim, the pocket beyond the section, removed and the
    remaining lip closed by the half circle of diameter the section's width bulging into the rim
    (:func:`_cap`). Returns the new loop's points (the loop with the rim's arc replaced by the cap,
    counter-clockwise), the removed region (``removed``: the rim's arc and the cap back), its area,
    volume of revolution ``V_rim``, centroid, extent ``L_rim``, the cap, the section's width ``w``,
    the rim's arc length, the largest speed on the rim (``U_max``, with ``velocity`` the loop's) and
    the checks; raises :class:`SheetRuptureRefused` where the new loop is not simple, the removed
    region not liquid or not simple, nearer the open chain (``open_points``) than
    ``OPEN_CLEARANCE w``, or larger than ``MAX_REMOVED_V_B V_b``."""
    import shapely
    from shapely.geometry import LinearRing, Polygon

    from .groove import gas_share, side_of
    from .reconnect import revolved_volume

    P = np.asarray(P, dtype=float).reshape(-1, 2)
    n = len(P)
    a, b = (int(k) % n for k in pair)
    checks: dict = {"failed": []}
    if a == b or n < 8:
        raise SheetRuptureRefused("the section's pair is not two nodes of the loop", checks)
    arc = _arc(P, a, b)
    rim = P[arc]
    share = gas_share(rim, P)
    checks.update(side=side_of(share), gas_share=share, rim_nodes=len(arc))
    if checks["side"] != "liquid":
        checks["failed"].append("liquid")
        raise SheetRuptureRefused(f"the pocket beyond the section is not liquid ({checks['side']})",
                                  checks)
    w = float(np.hypot(*(P[a] - P[b])))
    tip = rim[int(np.argmax(np.hypot(*(rim - 0.5 * (P[a] + P[b])).T)))]
    cap = _cap(P[arc[0]], P[arc[-1]], tip)  # from the arc's first node to its last
    removed = np.vstack([rim, cap[::-1][1:-1]])
    poly = Polygon(removed)
    checks["removed_simple"] = bool(poly.is_valid and poly.area > 0.0)
    if not checks["removed_simple"]:
        checks["failed"].append("removed")
        raise SheetRuptureRefused("the removed region is not a simple polygon", checks)
    inside = set(arc[1:-1])
    keep = [i for i in range(n) if i not in inside]
    # the new loop: the loop's nodes off the rim, the cap from the arc's first node to its last
    # in the loop's own direction
    forward = (arc[1] - arc[0]) % n == 1
    start = keep.index(arc[-1] if forward else arc[0])
    ring = keep[start:] + keep[:start]  # from the rim's last node (going forward) round to the first
    if forward:  # ring runs b, ..., a: close a -> cap -> b
        new = np.vstack([P[ring], cap[1:-1]])
    else:  # the arc runs backwards from a: ring runs a, ..., b; close b -> cap reversed -> a
        new = np.vstack([P[ring], cap[::-1][1:-1]])
    checks["loop_simple"] = bool(LinearRing(new).is_simple)
    area_new = 0.5 * float(np.sum(new[:, 0] * np.roll(new[:, 1], -1) - np.roll(new[:, 0], -1)
                                  * new[:, 1]))
    checks["loop_ccw"] = area_new > 0.0
    if not (checks["loop_simple"] and checks["loop_ccw"]):
        checks["failed"].append("loop")
        raise SheetRuptureRefused("the ring's new loop is not simple and counter-clockwise", checks)
    clearance = float(shapely.distance(poly, shapely.LineString(np.asarray(open_points,
                                                                           dtype=float))))
    checks.update(open_clearance=clearance, open_clearance_limit=OPEN_CLEARANCE * w)
    if not clearance > OPEN_CLEARANCE * w:
        checks["failed"].append("open")
        raise SheetRuptureRefused("the rim lies within reach of the open chain", checks)
    V_rim = revolved_volume(removed)
    checks.update(V_rim=V_rim, V_limit=MAX_REMOVED_V_B * V_b)
    if not V_rim <= MAX_REMOVED_V_B * V_b:
        checks["failed"].append("volume")
        raise SheetRuptureRefused("the rim is larger than a lip's", checks)
    c = poly.centroid
    U = (float(np.max(np.hypot(*np.asarray(velocity, dtype=float)[arc].T)))
         if velocity is not None else math.nan)
    arc_len = float(np.sum(np.hypot(*np.diff(rim, axis=0).T)))
    cap_len = math.pi * 0.5 * w
    return {"points": new, "removed": removed, "cap": cap, "pair": [a, b], "arc": arc,
            "w": w, "area": float(poly.area), "V_rim": V_rim, "r_c": float(c.x),
            "z_c": float(c.y), "L_rim": float(max(np.ptp(removed[:, 0]), np.ptp(removed[:, 1]))),
            "tip": tip.tolist(), "arc_length": arc_len, "cap_length": cap_len, "U_max": U,
            "A_rim": 2.0 * math.pi * float(np.sum(0.5 * (rim[1:, 0] + rim[:-1, 0])
                                                  * np.hypot(*np.diff(rim, axis=0).T))),
            "A_cap": 2.0 * math.pi * float(np.sum(0.5 * (cap[1:, 0] + cap[:-1, 0])
                                                  * np.hypot(*np.diff(cap, axis=0).T))),
            "checks": checks}


def rim_quadrature(pb, polygon, V_rim: float) -> dict:
    """The rim's volume, axial momentum and kinetic energy integrated on the rim itself: the
    positive triangle rule of ``axisrepair.polygon_quadrature_points`` on ``polygon`` (the removed
    region), its points located in the liquid mesh, whatever the box grid's spacing (the grid of
    ``surgery.site_grid``, a few hundred cells across a box of ``ring.box_radius delta``, can hold
    no point of a rim at the floor's scale, and would log its momentum and energy as zero).
    ``checked`` where every point is located and the rule's volume reproduces ``V_rim``, the
    polygon's volume of revolution, within the R1 tolerance (``ringsurgery.RING_BUDGET_LIMITS``),
    as the axis repair's sliver quadrature is checked; a rupture requires it."""
    from .axisrepair import polygon_quadrature_points
    from .ringsurgery import RING_BUDGET_LIMITS
    from .surgery import sample_fields

    X, dV, triangles = polygon_quadrature_points(polygon)
    if len(X):
        values, found = sample_fields(pb, X[:, 0], X[:, 1], ("velocity_x", "velocity_y"))
    else:
        values, found = {"velocity_x": np.zeros(0), "velocity_y": np.zeros(0)}, np.zeros(0, bool)
    V = math.fsum(dV)
    located = bool(len(X)) and bool(found.all())
    ur, uz = values["velocity_x"], values["velocity_y"]
    error, limit = abs(V - V_rim), RING_BUDGET_LIMITS["R1"] * abs(V_rim)
    return {"V_quad": V, "V_rim": float(V_rim), "error": error, "limit": limit,
            "Pz": math.fsum(dV * uz) if located else math.nan,
            "KE": 0.5 * math.fsum(dV * (ur**2 + uz**2)) if located else math.nan,
            "points": int(len(X)), "found": int(found.sum()), "triangles": int(triangles),
            "rule": "triangle-degree-five", "checked": bool(located and error <= limit)}


def sheet_rupture(problem, events, reconnect, outdir: Path, request: dict, *, k: int,
                  dt_prev: float, h_floor: float, lip: dict, driver=None, on_made=None, log=print,
                  base_check=None) -> dict:
    """Rupture the ring's thinning lip the request names (module docstring), the ``k``-th event
    after the gas cut; ``h_floor`` the run's floor; ``lip`` what the driver read of the lip (its
    thickness history, ``h_s``, the thinning reading), recorded; ``base_check`` the driver's reading
    of the upward jet's base on the fresh mesh against its reading before (``base_guard`` in the
    record), a base it finds changed the hard closure ``S_base``. Returns the record (schema
    ``faraday-event/2``, kind ``sheet-rupture``), also written to ``events/event_<k>.json``
    whatever the outcome; raises ``surgery.CutRefused`` with the problem unchanged,
    ``surgery.BudgetUnclosed`` where a hard closure misses (a stop to resume from ``precut_e<k>``),
    ``surgery.CutFailed`` otherwise after the remesh."""
    from .ringsurgery import _run_surgery

    out = Path(outdir)
    record = {"schema": EVENT_SCHEMA, "k": int(k), "kind": SHEET_RUPTURE,
              "class": SHEET_RUPTURE_CLASS, "handler": SHEET_RUPTURE}
    path = out / "events" / f"event_{int(k):03d}.json"
    return _run_surgery(lambda progress: _sheet_rupture(
        problem, events, reconnect, out, record, progress, request, path, k=int(k),
        dt_prev=float(dt_prev), h_floor=float(h_floor), lip=dict(lip), driver=driver,
        on_made=on_made, log=log, base_check=base_check), problem, out, record, path, log)


def _sheet_rupture(pb, events, s, out: Path, record: dict, progress: dict, request: dict,
                   path: Path, *, k: int, dt_prev: float, h_floor: float, lip: dict, driver,
                   on_made, log, base_check=None) -> dict:
    """The sequence of :func:`sheet_rupture`, filling ``record`` as it goes."""
    import shapely

    from .events import predict_unlocated
    from .ring import closed
    from .ringcontact import momentum_change, surface_cells
    from .ringsurgery import (CORNER_TOL, _axis_ends, _base_record, _corners, _focus_regions,
                              _node_sets, _rebuild, _ring_closures, _transfer_check)
    from .surgery import (INTERFACE, BudgetUnclosed, CutFailed, CutRefused, _axis_spans_of_mesh,
                          _jsonable, _keep, _named_chains, box_integrals, box_polyline_budgets,
                          orient_interface_chart, project, site_grid, transfer_warnings)

    def stage(name):
        progress["stage"] = name

    wall0 = time.time()
    if getattr(pb, "recon", None) is None:
        raise CutRefused("the problem was not made with reconnect settings")
    if pb.cut is None:
        raise CutRefused("a rupture of the ring's lip follows the gas cut: the problem has none")
    if getattr(pb, "ring", None) is None or pb.ring.get("removed") is not None:
        raise CutRefused("a rupture of the ring's lip is made in the ring stage, before the ring's "
                         "removal")
    if request.get("kind") != SHEET_CONTACT or request.get("chain") != "ring" \
            or not (request.get("sheet_rupture") or {}).get("due"):
        raise CutRefused(f"no rupture for a request of kind {request.get('kind')!r} on the chain "
                         f"{request.get('chain')!r} not due for one")
    if not (dt_prev > 0.0 and math.isfinite(dt_prev)):
        raise CutRefused(f"the step before the rupture must be positive, not {dt_prev!r}")
    delta = float(s.delta)
    record.update(_base_record(pb, events, s, request, k, SHEET_RUPTURE, SHEET_RUPTURE_CLASS,
                               SHEET_RUPTURE, "ring", dt_prev))
    record["stage"] = "ring"
    record["lip"] = _jsonable(lip)
    named, segs = _named_chains(pb)
    chains = dict(named)
    loop_segs = pb.interface_loops()
    loops = [np.column_stack([x["r"], x["z"]]) for x in loop_segs]
    if len(loops) != 1 or set(chains) != {"open"}:
        raise CutRefused(f"a lip of the ring is ruptured on its one loop beside the open chain: "
                         f"chains {sorted(chains)}, {len(loops)} loops")
    P = loops[0]
    velocity = np.column_stack([loop_segs[0]["u_r"], loop_segs[0]["u_z"]])
    V_b = float(pb.cut["geometry"]["bubble_volume"])
    record["dumps"] = {}

    stage("precut")
    event = {**record["request"], "k": k}
    record["dumps"]["precut"] = _keep(pb, driver, out, f"precut_e{k:03d}",
                                      {"cut_stage": PRECUT_EVENT, "event": event})["path"]

    stage("geometry")
    try:
        g = sheet_rupture_geometry(P, request.get("fold_pair") or (), V_b=V_b,
                                   open_points=chains["open"].points, velocity=velocity)
    except SheetRuptureRefused as e:
        record["geometry"] = {"checks": e.checks}
        raise CutRefused(f"the rupture of the lip is refused: {e}") from e
    record["geometry"] = {key: g[key] for key in ("pair", "w", "area", "V_rim", "r_c", "z_c",
                                                   "L_rim", "tip", "arc_length", "cap_length",
                                                   "U_max", "A_rim", "A_cap", "checks")}
    record["geometry"].update(V_rim_over_V_b=g["V_rim"] / V_b, V_rim_over_delta3=g["V_rim"]
                              / delta**3, h_floor=h_floor)

    stage("budgets")
    R_b = max(events.ring.box_radius * delta, 2.0 * g["L_rim"])
    grid = site_grid(g["r_c"], g["z_c"], R_b)
    global_before = pb.cut_observables()
    box_before = box_integrals(pb, grid, {"rim": g["removed"]}, cutoff=True)
    rim = box_before.pop("rim")  # the grid's reading of the rim, reported beside the quadrature's
    quad = rim_quadrature(pb, g["removed"], g["V_rim"])
    record["rim_quadrature"] = quad
    if not quad["checked"]:
        raise CutRefused(f"the rim's quadrature does not resolve it: {quad['found']} of "
                         f"{quad['points']} points located, its volume {quad['V_quad']:.6g} "
                         f"against the rim's {g['V_rim']:.6g} (limit {quad['limit']:.3g})")
    rings_before = [Chain(closed(L), ("loop", "loop")) for L in loops]
    poly_before = box_polyline_budgets([c for _, c in named] + rings_before, g["z_c"], R_b,
                                       s.reservoir_depth, r_centre=g["r_c"])
    # S2 the transfer's change alone: the rim it removes read as gas before the rebuild
    cells_before = surface_cells(pb, grid, [c for _, c in named] + rings_before,
                                 s.reservoir_depth, region=g["removed"])
    U = float(g["U_max"]) if math.isfinite(g["U_max"]) else 0.0
    scales = {"V_fill": g["V_rim"], "U": U, "P_ref": g["V_rim"] * U,
              "KE_ref": 0.5 * g["V_rim"] * U**2}
    size = max(h_floor, g["w"] / CAP_ELEMENTS)
    mid = g["cap"][len(g["cap"]) // 2]
    region = FocusRegion(f"sheet_e{k:03d}", size=size, r=float(mid[0]), z=float(mid[1]),
                         radius=REGION_RADIUS * g["w"])
    template = pb.get_mesh("liquid")._templatemesh
    old_focus = template.focus
    regions, grading = _focus_regions(old_focus, pb.mesh_settings.grading)
    new_regions = tuple(regions) + (region,)
    focus = Focus(size=min(x.size for x in new_regions), radius=max(x.radius for x in new_regions),
                  grading=grading, regions=new_regions)
    record["mesh"] = {"focus_before": None if old_focus is None else old_focus.to_dict(),
                      "region": {"name": region.name, "size": size, "r": region.r, "z": region.z,
                                 "radius": region.radius, "h_floor": h_floor}}
    geometry = {"chains": [chains["open"]], "loops": [g["points"]]}
    old_chains = [c.points for _, c in named] + [closed(L) for L in loops]

    def restore():
        template.focus = old_focus

    stage("park")
    printed = _rebuild(pb, template, focus, geometry, {"axis_bulk": False, "reach": 0.0},
                       stage=stage, precut=record["dumps"]["precut"], restore=restore)

    stage("mesh-check")
    transfer = transfer_warnings("".join(printed))
    record["transfer"] = transfer
    new_named, _ = _named_chains(pb)
    new_loops = [np.column_stack([x["r"], x["z"]]) for x in pb.interface_loops()]
    record["mesh"].update(chains=[{"chain": x, "nodes": len(c.points),
                                   "end_types": list(c.end_types)} for x, c in new_named],
                          loops=[len(L) for L in new_loops], nodes=pb.get_mesh("liquid").nnode(),
                          planned_loop_points=int(len(g["points"])))
    if {x: c.end_types for x, c in new_named} != {x: c.end_types for x, c in named}:
        raise CutFailed(f"the rupture's mesh has the chains {record['mesh']['chains']}, not those "
                        "planned", record["dumps"]["precut"])
    if len(new_loops) != 1:
        raise CutFailed(f"the rupture's mesh has {len(new_loops)} loops, not one",
                        record["dumps"]["precut"])
    spans = _axis_spans_of_mesh(pb)
    expected = liquid_axis_spans([z for _, c in new_named for _, z in c.axis_ends()])
    old_axis = liquid_axis_spans([z for _, c in named for _, z in c.axis_ends()])
    record["mesh"].update(axis_spans=spans, expected_axis_spans=[list(x) for x in expected])
    tol = 1e-9 * max(1.0, pb.nd.radius)

    def same(xs, ys) -> bool:
        return len(xs) == len(ys) and all(abs(a - b) <= tol for x, y in zip(xs, ys)
                                          for a, b in zip(x, y))

    if not same(spans, expected):
        raise CutFailed(f"the rupture's mesh has the axis spans {spans}, expected {expected}",
                        record["dumps"]["precut"])
    if not same(old_axis, expected):
        raise CutFailed(f"the rupture moved the axis spans from {old_axis} to {expected}",
                        record["dumps"]["precut"])
    sets = _node_sets(pb)
    X = sets["interface"][1]
    predicted = predict_unlocated(X, old_chains)
    cap_line = shapely.LineString(g["cap"])
    on_cap = (np.asarray(shapely.distance(cap_line, shapely.points(X)) <= CAP_TOL * g["w"],
                         dtype=bool) if len(X) else np.zeros(0, dtype=bool))
    bulk_inside = np.zeros(len(sets["bulk"][1]), dtype=bool)  # the new liquid lies in the old
    axis_gap = np.zeros(len(sets["axis"][1]), dtype=bool)
    corners = _corners(pb, _axis_ends([c for _, c in named]),
                       CORNER_TOL * float(pb.cut["sizing"]["h_tip"]))
    problems = _transfer_check(transfer, sets, predicted, on_cap, bulk_inside, axis_gap, corners)
    transfer["corner_fields"] = sorted(pb.get_mesh(INTERFACE + "/axis").get_nodal_field_indices())
    if problems:
        raise CutFailed("the transfer of the rupture is not as planned: " + "; ".join(problems),
                        record["dumps"]["precut"])
    global_transfer = pb.cut_observables()
    rings_after = [Chain(closed(L), ("loop", "loop")) for L in new_loops]
    poly_after = box_polyline_budgets([c for _, c in new_named] + rings_after, g["z_c"], R_b,
                                      s.reservoir_depth, r_centre=g["r_c"])
    moved = momentum_change(grid, {"_cells": cells_before}, {"_cells": surface_cells(
        pb, grid, [c for _, c in new_named] + rings_after, s.reservoir_depth)})

    stage("projection")
    record["projection"] = {}
    gs = events.groove
    settings = replace(s, dt0=dt_prev, restart=gs.restart or s.restart, projection=gs.projection)
    project(pb, settings, float(record["t"]), record["projection"], box=grid, cutoff=True)
    stage("chart")
    orient_interface_chart(pb)

    stage("postcut")
    global_after = pb.cut_observables()
    box = record["projection"].get("box")
    if box is not None:  # the cutoff's sums, those without it kept beside them
        box_transfer = dict(box["before_phi"], weight="cutoff", unweighted=box["before"])
        box_after = dict(box["after_phi"], weight="cutoff", unweighted=box["after"])
    else:
        box_transfer = box_after = box_integrals(pb, grid, cutoff=True)
    record["budgets"] = {
        "box": {"r": g["r_c"], "z": g["z_c"], "R": R_b, "n": grid["n"],
                "cells": int(len(grid["r"])), "weight": "cutoff"},
        "polylines": {"before": poly_before, "after": poly_after},
        "grid": {"before": box_before, "after_transfer": box_transfer,
                 "after_projection": box_after},
        "global": {"before": global_before, "after_transfer": global_transfer,
                   "after": global_after, "includes_ring": True}}
    # the rim's momentum and energy those of its own quadrature, which S2 and S6 hold the box's
    # change against; the grid's reading of the rim beside them
    removed = {"V": g["V_rim"], "Pz": float(quad["Pz"]), "KE": float(quad["KE"]),
               "V_grid": float(rim["V_grid"]), "Pz_grid": float(rim["Pz"]),
               "KE_grid": float(rim["KE"])}
    ring = events.ring
    closures = _ring_closures({"V": -removed["V"], "Pz": -removed["Pz"], "KE": -removed["KE"]},
                              record["budgets"]["polylines"],
                              {"before": box_before, "after_transfer": box_transfer,
                               "after_projection": box_after}, box, scales,
                              {"A_cl": g["A_cap"], "A_rem": g["A_rim"]},
                              sigma={"V": ring.sigma_V, "Pz": ring.sigma_Pz, "KE": ring.sigma_KE},
                              prefix="S", moved_change=moved["change"])
    closures["S2_split"] = moved  # the reading's split; S2_grid the midpoint rule's S2
    closures["global_volume"] = {"delta_V": global_after["volume"] - global_before["volume"],
                                 "V_rim": removed["V"]}
    closures["global_identity"] = {x: record["projection"].get(x, math.nan)
                                   for x in ("delta_Pz", "Pi_on_bottom")}
    record["closures"] = closures
    if closures["hard"]:
        missed = {x: closures[x] for x in closures["hard"]}
        raise BudgetUnclosed(f"the budget of the rupture of the lip does not close: "
                             f"{closures['hard']} beyond their limits ({missed})")
    if base_check is not None:  # the base of the upward jet, which the rupture must leave as it was
        stage("base")
        guard = base_check()
        record["base_guard"] = guard
        if not guard["ok"]:
            closures["flags"] = list(closures["flags"]) + ["S_base"]
            closures["hard"] = list(closures["hard"]) + ["S_base"]
            raise BudgetUnclosed(f"the rupture changed the upward jet's base: {guard['change']} "
                                 f"beyond {guard['tol']}")
    record["removed"] = {**removed, "V_over_V_b": removed["V"] / V_b,
                         "V_over_delta3": removed["V"] / delta**3,
                         "centroid": [g["r_c"], g["z_c"]]}
    record["filled"] = {"V": -removed["V"], "Pz": -removed["Pz"], "KE": -removed["KE"]}
    record["outcome"] = "made"
    record["wall_seconds"] = time.time() - wall0
    made = _jsonable(record)
    pb.events = list(getattr(pb, "events", None) or []) + [made]
    if on_made is not None:
        on_made(made)
    made["dumps"]["postcut"] = _keep(pb, driver, out, f"postcut_e{k:03d}",
                                     {"cut_stage": "postcut-event", "event_k": k})["path"]
    pb.events[-1] = made
    full = dict(made, removed_polygon=g["removed"].tolist(), cap=g["cap"].tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(full), indent=1) + "\n")
    rel = ", ".join(f"{x} {closures['relative'][x]:.3g}" for x in closures["relative"])
    log(f"sheet rupture {k}: section {g['w']:.6g}, rim {removed['V']:.6g} "
        f"({removed['V'] / V_b:.3g} V_b); closures {rel}")
    return made
