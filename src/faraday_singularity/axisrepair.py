"""The repair of a chain's end folded along the axis (``axis_repair``), in the nuisance region.

A fast jet's tip on the axis (the downward jet's tip, the bubble's top end, before the piercing)
can fold within one element: the surface meets the axis, and its last element runs on along the
axis to the axis node, a node of it at ``r < 0`` (r3 of the delta triple at e923ead, step 7338:
the tip moving down at 1.08e4 met the axis at z 4.677043, its last element ran 5.5e-6 up the axis
to the axis node, a vertex at r -1.6e-7, and the element before it turned by 102.8 degrees). A
rebuild at the size the turn asks for exposes it as a gas neck at the axis, not a feature it
resolves. The tip is a nuisance (decision 22): the fold is removed, the nodes on the axis beyond
the surface dropped and the axis node put where the surface meets the axis, and the sliver of gas
between the folded tail and the new tip becomes liquid, its volume and momentum a closure that is
hard above a small bound.

1. **The fold** (:func:`axis_fold`): walked from the chain's axis end inward, the nodes within
   ``AXIS_FOLD_TOL`` of the chain's spacing there of the axis (``r`` at most that, or below zero),
   beyond the end itself; none (no fold) where the node next to the end lies off the axis, as at a
   tip that meets the axis square. The first node off the axis is kept, and the new end is where
   the line through it and the node before it meets the axis, at most its own radius from its
   height (the surface meets the axis square).
2. **The geometry**: the chain without the folded nodes, ending on the axis at the new end; the
   sliver, the polygon of the folded tail from the first kept node to where the tail first reaches
   the axis (beyond, it runs on or across the axis and bounds no area) and the new end, its volume
   of revolution ``V_sliver``; the old chain cut there for the budgets (a polygon through a node at
   ``r < 0`` is no valid section).
3. **Budgets before**, within ``groove.box_radius delta`` of the new end on the axis (the polar
   grid of ``surgery.site_grid`` about it, weighted by the box's cutoff), and the box's polylines.
4. **Rebuild** from the planned chains (the template's ``pending_ring_geometry``), the interface
   placed by projection onto the old one and the axis located in the old bulk
   (``FaradayProblem._ring_transfer``), the focus unchanged. The fill's transfer check compares
   the unlocated-node counts with the old chains and the sliver plus its band, requires the
   chain names and end types to stay unchanged and checks the planned axis spans. A miss is a
   failed repair, rolled back by the driver.
5. **Closures** (``ringsurgery._ring_closures``, prefix ``A``): A1 the liquid gained against the
   sliver's volume, A2 the momentum the transfer carried against the sliver's, hard, floored at
   the ``groove`` part's noise; A3 none (no projection); the flags as the fill's; and the bound,
   hard: ``V_sliver <= AXIS_REPAIR_BOUND delta^3`` (``A_V``) and ``|Pz_sliver| <= AXIS_REPAIR_BOUND
   delta^3 U`` (``A_P``, the sliver's sampled momentum after transfer, independently of its
   geometric volume; ``U`` the largest speed of the folded nodes). Its positive triangle
   quadrature must locate every point and reproduce the sliver's geometric volume within the
   existing R1 volume tolerance; otherwise A_P is unchecked and the repair is refused.
   The box-grid sample is retained for A2, and reported beside the sliver quadrature. The bound: a fold of one
   element of the tip's size ``h`` leaves a sliver of at most about ``pi h^3``; the elements at a
   tip of the far region are at most a few hundredths of ``delta`` (r3's 5.5e-6, 0.022 delta, so
   ``pi h^3`` = 3.3e-5 ``delta^3``), and ``1e-4 delta^3`` (a sphere of radius 0.029 delta) admits
   such a fold and nothing that a resolved feature could be; r3's sliver is about 1e-6 ``delta^3``.
   A larger one is no fold of the mesh but a feature, and is refused (the stop
   ``axis-repair-unclosed``, the problem rolled back), for the cap of option B.

The base of the upward jet is read before and after (``base_check``, the hard closure
``A_base``).
"""

from __future__ import annotations

import math
import time

import numpy as np

AXIS_FOLD_TOL = 0.25  # a node within this share of the chain's spacing of the axis lies on it
AXIS_FOLD_SEARCH = 16  # the most nodes from the axis end a fold may hold
AXIS_REPAIR_BOUND = 1e-4  # the sliver's volume at most this times delta^3 (hard above)
AXIS_REPAIR_RECUR_STEPS = 5  # a repair at the same end within this many steps: the fold recurs


def axis_fold(P, end: str) -> dict | None:
    """The fold of a chain's end on the axis (``end`` ``first`` or ``last``; ``P`` the chain's
    nodes, the end on ``r = 0``), or None: walked from the end inward, the nodes beyond the end
    within ``AXIS_FOLD_TOL`` of the chain's spacing there (the median of its last eight chords) of
    the axis, or across it; the first node off the axis kept and the new end where the line
    through it and the node before it meets the axis, at most its own radius from its height.
    Returns the fold: ``removed`` (indices of ``P``, the end among them), ``kept`` (the first kept
    node's index), the new end, the new chain's points, the sliver (polygon) and its volume of
    revolution, the spacing, the tolerance, the fold nodes' smallest ``r`` and the old end."""
    from .reconnect import revolved_volume

    P = np.asarray(P, dtype=float).reshape(-1, 2)
    n = len(P)
    if n < 6 or end not in ("first", "last"):
        return None
    walk = np.arange(n)[::-1] if end == "last" else np.arange(n)
    Q = P[walk]
    h = float(np.median(np.hypot(*np.diff(Q[:9], axis=0).T)))
    tol = AXIS_FOLD_TOL * h
    k = 1
    while k < min(n - 3, AXIS_FOLD_SEARCH) and Q[k, 0] <= tol:
        k += 1
    if k == 1 or Q[k, 0] <= tol:
        return None  # the node beside the end lies off the axis: a tip that meets it square
    a, b = Q[k], Q[k + 1]  # the first node kept and the one before it
    dr = a[0] - b[0]
    z_new = a[1] + (a[1] - b[1]) * (0.0 - a[0]) / dr if dr < 0.0 else a[1]
    z_new = float(np.clip(z_new, a[1] - a[0], a[1] + a[0]))
    new_end = np.array([0.0, z_new])
    kept = Q[k:]
    pts = np.vstack([new_end[None, :], kept])
    points = pts[::-1] if end == "last" else pts
    # the folded tail from the kept node to where it first reaches the axis (a node at r <= 0, the
    # end at the latest; between the two the crossing of r = 0): beyond, the tail runs on the axis
    # or across it and bounds no area, so that the old surface ends there for the budgets
    j = next(i for i in range(k - 1, -1, -1) if Q[i, 0] <= 0.0)
    a_in, a_out = Q[j + 1], Q[j]
    t = a_in[0] / (a_in[0] - a_out[0]) if a_in[0] - a_out[0] > 0.0 else 1.0
    cross = np.array([0.0, a_in[1] + t * (a_out[1] - a_in[1])])
    tail = np.vstack([Q[k:j:-1], cross[None, :]])  # the kept node, ..., the crossing
    sliver = np.vstack([new_end[None, :], tail])
    old_pts = np.vstack([Q[:j:-1], cross[None, :]])  # the old chain from its far end, as read
    old_points = old_pts if end == "last" else old_pts[::-1]
    return {"end": end, "removed": [int(i) for i in walk[:k]], "kept": int(walk[k]),
            "new_end": new_end, "old_end": Q[0].copy(), "crossing": cross, "points": points,
            "old_points": old_points, "sliver": sliver,
            "V_sliver": abs(float(revolved_volume(sliver))), "h": h, "tol": tol,
            "r_min": float(Q[1:k, 0].min()), "k": int(k)}


def polygon_quadrature_points(polygon) -> tuple[np.ndarray, np.ndarray, int]:
    """The points ``X`` (``(m, 2)``) and volume weights ``dV`` (``2 pi r`` times the area weight)
    of the seven-point, degree-five triangle rule on the constrained Delaunay triangles of
    ``polygon`` (the points of a ring, or a shapely polygonal geometry, holes and parts included),
    and the triangles' count: positive weights whose points follow the polygon at any scale
    (:func:`sliver_quadrature`; ``sheetsurgery.rim_quadrature``; ``ringcontact.cell_readings``)."""
    import shapely

    geom = polygon if isinstance(polygon, shapely.Geometry) else shapely.Polygon(polygon)
    triangles = shapely.constrained_delaunay_triangles(geom).geoms
    bary = [(1 / 3, 1 / 3, 1 / 3)]
    weights = [0.225]
    for a, b, w in ((0.059715871789770, 0.470142064105115, 0.132394152788506),
                     (0.797426985353087, 0.101286507323456, 0.125939180544827)):
        bary.extend(((a, b, b), (b, a, b), (b, b, a)))
        weights.extend((w, w, w))
    points, dV = [], []
    for triangle in triangles:
        P = np.asarray(triangle.exterior.coords, dtype=float)[:3]
        A, B = P[1] - P[0], P[2] - P[0]
        area = 0.5 * abs(A[0] * B[1] - A[1] * B[0])
        X = np.asarray(bary) @ P
        points.extend(X)
        dV.extend(2 * math.pi * X[:, 0] * area * np.asarray(weights))
    X = np.asarray(points, dtype=float).reshape(-1, 2)
    dV = np.asarray(dV, dtype=float)
    return X, dV, len(X) // len(bary)


def sliver_quadrature(pb, polygon) -> dict:
    """Integrate momentum on the sliver itself, with positive triangle Gauss weights and the
    axisymmetric factor 2 pi r. The degree-five rule resolves a quadratic velocity times r;
    unlike the box grid, its points and weights follow the sliver at any scale."""
    from .reconnect import revolved_volume
    from .surgery import sample_fields

    X, dV, triangles = polygon_quadrature_points(polygon)
    values, found = sample_fields(pb, X[:, 0], X[:, 1], ("velocity_y",))
    V = math.fsum(dV)
    geometric = abs(float(revolved_volume(polygon)))
    return {"Pz": math.fsum(dV * values["velocity_y"]) if found.all() else math.nan,
            "V_quad": V, "V_sliver": geometric,
            "V_quad_over_V_sliver": V / geometric if geometric > 0 else math.nan,
            "V_found": math.fsum(dV[found]), "points": len(X), "found": int(found.sum()),
            "rule": "triangle-degree-five", "triangles": triangles}


def axis_repair(pb, events, reconnect, outdir, chain: str, fold: dict, *, velocity,
                base_check=None, log=print) -> dict:
    """Repair the fold ``fold`` (:func:`axis_fold`) of the chain ``chain`` (module docstring):
    rebuild without the folded nodes, read the closures and the base. Returns the record; raises
    ``surgery.BudgetUnclosed`` (the record as its ``record``) where a hard closure misses, the
    problem then holding the repaired mesh, which the caller rolls back; any other failure of the
    rebuild as ``surgery.CutFailed``. ``velocity`` the chain's nodes' velocities (``(n, 2)``)."""
    from .events import predict_unlocated, revolved_area
    from .reconnect import Chain, liquid_axis_spans
    from .ringcontact import momentum_change, surface_cells
    from .ringsurgery import (CORNER_TOL, RING_BUDGET_LIMITS, _axis_ends, _corners, _inside, _node_sets, _rebuild,
                              _ring_closures, _transfer_check)
    from .surgery import (BudgetUnclosed, CutFailed, _axis_spans_of_mesh, _named_chains,
                          box_integrals, box_polyline_budgets, site_grid, transfer_warnings)
    import shapely

    wall0 = time.time()
    gs, delta = events.groove, float(reconnect.delta)
    record = {"kind": "axis-repair", "chain": chain, "end": fold["end"],
              "t": pb.get_current_time(dimensional=False, as_float=True),
              "fold": {"removed": fold["removed"], "kept": fold["kept"],
                       "new_end": fold["new_end"].tolist(), "old_end": fold["old_end"].tolist(),
                       "h": fold["h"], "tol": fold["tol"], "r_min": fold["r_min"],
                       "V_sliver": fold["V_sliver"], "sliver": fold["sliver"].tolist(),
                       "V_sliver_over_delta3": fold["V_sliver"] / delta**3}}
    named, _ = _named_chains(pb)
    chains = dict(named)
    V = np.asarray(velocity, dtype=float).reshape(-1, 2)
    folded = V[fold["removed"] + [fold["kept"]]]
    U = float(np.max(np.hypot(*folded.T))) if len(folded) else 0.0
    uz = float(np.mean(folded[:, 1])) if len(folded) else 0.0
    V_sl = float(fold["V_sliver"])
    bound_V = AXIS_REPAIR_BOUND * delta**3
    record["bound"] = {"V": bound_V, "P": bound_V * U, "U": U, "u_z": uz,
                       "Pz_sliver_estimate": V_sl * uz, "AXIS_REPAIR_BOUND": AXIS_REPAIR_BOUND}

    if V_sl > bound_V:  # the geometric bound needs no whole-domain rebuild
        record["closures"] = {"A_V": V_sl, "limits": {"A_V": bound_V},
                              "hard": ["A_V"], "flags": ["A_V"]}
        record.update(outcome="unclosed", wall_seconds=time.time() - wall0)
        error = BudgetUnclosed("the axis fold exceeds the volume bound")
        error.record = record
        raise error

    # budgets before
    z_c = float(fold["new_end"][1])
    R_b = float(gs.box_radius) * delta
    grid = site_grid(0.0, z_c, R_b)
    sliver = fold["sliver"]
    before = box_integrals(pb, grid, {"sliver": sliver}, cutoff=True)
    before.pop("sliver")
    depth = float(reconnect.reservoir_depth)
    old_clean = Chain(np.asarray(fold["old_points"], dtype=float), tuple(chains[chain].end_types))
    poly_before = box_polyline_budgets([old_clean if x == chain else c for x, c in named], z_c,
                                       R_b, depth, r_centre=0.0)
    # A2 read cell by cell across the rebuild, the gas bounded as the polyline budgets bound it
    cells_before = surface_cells(pb, grid, [old_clean if x == chain else c for x, c in named],
                                 depth)
    # the rebuild
    new = Chain(np.asarray(fold["points"], dtype=float).copy(), tuple(chains[chain].end_types))
    geometry = {"chains": [new if x == chain else c for x, c in named], "loops": []}
    template = pb.get_mesh("liquid")._templatemesh
    focus = template.focus
    progress = {}

    def stage(name):
        progress["stage"] = name

    printed = _rebuild(pb, template, focus, geometry,
                       {"axis_bulk": True, "reach": 4.0 * float(fold["h"])}, stage=stage,
                       precut=None, restore=lambda: None)
    record["transfer"] = transfer_warnings("".join(printed))
    new_named, _ = _named_chains(pb)
    after_chains = dict(new_named)
    record["mesh"] = {"chains": {x: {"nodes": len(c.points), "end_types": list(c.end_types)}
                                   for x, c in new_named},
                      "nodes": pb.get_mesh("liquid").nnode()}
    problems = []
    if {x: c.end_types for x, c in new_named} != {x: c.end_types for x, c in named}:
        problems.append("the repair changed the chain names or end types")
    if pb.interface_loops():
        problems.append("the repair made an unplanned loop")
    spans = _axis_spans_of_mesh(pb)
    expected = liquid_axis_spans([z for _, c in new_named for _, z in c.axis_ends()])
    record["mesh"].update(axis_spans=spans, expected_axis_spans=[list(x) for x in expected])
    if len(spans) != len(expected) or any(abs(a - b) > 1e-9 * max(1.0, pb.nd.radius)
            for x, y in zip(spans, expected) for a, b in zip(x, y)):
        problems.append("the repair's axis spans differ from the planned spans")
    sets = _node_sets(pb)
    predicted = predict_unlocated(sets["interface"][1], [c.points for _, c in named])
    band = 0.05 * float(fold["h"])
    closure = shapely.LineString([fold["new_end"], chains[chain].points[fold["kept"]]])
    on_closure = np.asarray(shapely.distance(closure, shapely.points(sets["interface"][1]))
                            <= band, dtype=bool)
    bulk_inside = _inside(sets["bulk"][1], sliver, -band)
    A = sets["axis"][1]
    low, high = sorted([float(fold["old_end"][1]), float(fold["new_end"][1])])
    axis_gap = (A[:, 1] >= low - band) & (A[:, 1] <= high + band)
    corners = _corners(pb, _axis_ends([c for _, c in named]),
                       CORNER_TOL * float(pb.cut["sizing"]["h_tip"]))
    problems += _transfer_check(record["transfer"], sets, predicted, on_closure,
                                 bulk_inside, axis_gap, corners)
    record["transfer"]["band"] = band
    record["transfer"]["problems"] = problems
    if problems:
        error = CutFailed("the transfer of the axis repair is not as planned: " + "; ".join(problems))
        error.record = record
        raise error
    after = box_integrals(pb, grid, {"sliver": sliver}, cutoff=True)
    filled = after.pop("sliver")
    # A2 the transfer's change alone: the sliver read as gas after the rebuild
    moved = momentum_change(grid, {"_cells": cells_before}, {"_cells": surface_cells(
        pb, grid, [c for _, c in new_named], depth, region=sliver)})
    poly_after = box_polyline_budgets([c for _, c in new_named], z_c, R_b, depth, r_centre=0.0)
    A_rem = float(revolved_area(fold["sliver"][1:]))  # the folded tail, the kept node to the axis
    A_cl = float(revolved_area(fold["sliver"][:2]))  # the new tail, the kept node to the new end
    scales = {"V_fill": V_sl, "U": U, "P_ref": V_sl * U, "KE_ref": 0.5 * V_sl * U**2}
    closures = _ring_closures({"V": V_sl, "Pz": filled["Pz"], "KE": filled["KE"]},
                              {"before": poly_before, "after": poly_after},
                              {"before": before, "after_transfer": after,
                               "after_projection": after}, None, scales,
                              {"A_cl": A_cl, "A_rem": A_rem},
                              sigma={"V": gs.sigma_V, "Pz": gs.sigma_Pz, "KE": gs.sigma_KE},
                              prefix="A", moved_change=moved["change"])
    closures["A2_split"] = moved  # the reading's split; A2_grid the midpoint rule's A2
    closures["A_V"] = V_sl
    resolved = sliver_quadrature(pb, sliver)
    closures["A_P"] = abs(float(resolved["Pz"]))
    record["bound"]["Pz_sliver"] = float(resolved["Pz"])
    coverage_error = abs(resolved["V_quad"] - V_sl)
    coverage_limit = RING_BUDGET_LIMITS["R1"] * V_sl
    resolved["checked"] = bool(resolved["points"] and resolved["found"] == resolved["points"]
                                and coverage_error <= coverage_limit)
    resolved["coverage_limit"] = coverage_limit
    if not resolved["checked"]:
        closures["hard"].append("A_P_coverage")
        closures["flags"].append("A_P_coverage")
    closures["limits"].update(A_V=bound_V, A_P=bound_V * U)
    for x in ("A_V", "A_P"):
        if not closures[x] <= closures["limits"][x]:
            closures["flags"] = list(closures["flags"]) + [x]
            closures["hard"] = list(closures["hard"]) + [x]
    end_new = after_chains[chain].points[-1 if fold["end"] == "last" else 0]
    record["new_end_mesh"] = [float(end_new[0]), float(end_new[1])]
    record["budgets"] = {"box": {"r": 0.0, "z": z_c, "R": R_b, "weight": "cutoff"},
                         "polylines": {"before": poly_before, "after": poly_after},
                         "grid": {"before": before, "after": after}, "sliver": filled,
                         "sliver_quadrature": resolved}
    record["closures"] = closures
    if not closures["hard"] and base_check is not None:
        guard = base_check()
        record["base_guard"] = guard
        if not guard["ok"]:
            closures["flags"] = list(closures["flags"]) + ["A_base"]
            closures["hard"] = list(closures["hard"]) + ["A_base"]
    record["wall_seconds"] = time.time() - wall0
    if closures["hard"]:
        record["outcome"] = "unclosed"
        err = BudgetUnclosed(f"the repair of the axis end does not close: {closures['hard']}")
        err.record = record
        raise err
    record["outcome"] = "made"
    log(f"axis repair ({chain}, {fold['end']}): {len(fold['removed'])} nodes removed, sliver "
        f"{V_sl:.3g} ({V_sl / delta**3:.3g} delta^3), new end z {fold['new_end'][1]:.9g}")
    return record
