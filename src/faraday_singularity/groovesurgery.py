"""The fill of a closing gas groove on the finite-element problem (``groove_fill``).

A fold of a chain after the gas cut closing below ``max(gap.kappa_g delta, fold_stop_elements
h_elem)`` is an event of class ``A-groove`` (the open chain or the bubble, ``events.
post_cut_events``; the open chain in the ring stage, ``ring.ring_stage_events``) or ``RF`` (the
ring's loop). With the block's ``groove.enabled`` the driver meets it by this fill
(``driver.CollapseDriver._meet_groove``): the gas of the groove beyond its mouth becomes liquid, the
free surface is closed across the mouth (``groove.groove_fill_geometry``), and the run goes on from
the fresh mesh. It is one of the simplifying assumptions the run makes away from the upward jet
(decision 22): the gas of a thin groove is deleted, and logged, instead of being followed to a
closure the mesh cannot make; it is never made on the base of the upward jet or its core.

The fill is a rebuild the project owns, as the ring surgeries are (``ringsurgery.py``):

1. **Admission**: a gas cut made, ``groove.enabled``, a request of class ``A-groove`` (kind
   ``groove``) or ``RF`` (kind ``ring-fold``) on a chain the state has (``open`` or ``bubble``
   before the ring, ``open`` or ``ring`` in its stage), its trigger read again at its pair under
   the block (the pair closer than ``max(kappa_g delta, fold_stop_elements h_elem)`` and a fold,
   ``fold_ratio D`` below the arc between them), a positive last step; a pair across the base of
   the upward jet (``jet`` ``cavity_tip``) refused. ``precut_e<k>`` kept.
2. **Geometry** (``groove.groove_fill_geometry``, width ``groove.mouth`` times the trigger's limit,
   the closure's handles ``groove.cap_height``, once more at ``events.CAP_RETRY_HEIGHT``), refused
   (``surgery.CutRefused``, ``groove-fill-refused``) unless its hard checks hold, among them a
   pocket of gas (a sheet of liquid is refused), the core of the upward jet untouched (jetcore's
   mask) and the fill at most ``groove.max_fill V_b``.
3. **Budgets before**, within ``max(groove.box_radius delta, 2 L_fill)`` of the filled region's
   centroid, a disc at the site (``surgery.site_grid``; the polylines clipped to it,
   ``surgery.box_polyline_budgets`` with ``r_centre``), and the global ones; the scales ``P_ref =
   V_fill U`` and ``KE_ref = V_fill U^2 / 2``, ``U`` the largest speed on the old surface filled.
4. **Remesh at once** from the planned geometry (the template's ``pending_ring_geometry``: the new
   chains, and the ring as it is or the new loop; the template builds it as the ring stage's
   rebuild does, ``mesh.FaradayCylinderMesh._define_ring_geometry``, also before the ring),
   under the driver's focus with the closure's region (``groove_e<k>``, at the closure's own
   spacing within ``REGION_RADIUS`` mouth widths of its middle); the transfer places the interface
   nodes by projection onto the old interface (``FaradayProblem._ring_transfer``, the zeta
   transfer off). From here a failure is ``surgery.CutFailed``.
5. **Mesh check**: the chains and their ends, the loops (the ring kept across an open-chain fill
   in the ring stage, ``ring.loop_kept``; one loop after a fill of the ring), the axis spans as
   planned; the nodes the transfer leaves unlocated those the geometry predicts
   (``ringsurgery._transfer_check``): of the interface on the closure, of the bulk inside the filled
   region, none of the axis, no node without a value but on the corner set, none new.
6. **Fill** (``groove.fill``: the blend, or zero) and the **projection** (``surgery.project`` with
   ``groove.projection`` and ``groove.restart`` or the cut's, the run's last step as the history's,
   the time unchanged, the disc its box): ``Pi = 0`` on every free surface, the closure included.
7. **Budgets after and closures** (``ringsurgery._ring_closures``, prefix ``G``): G1 the volume the
   liquid gains against the filled volume, G2 the momentum the transfer carries against the filled
   region's, G3 the projection's momentum balance in the disc (the cutoff identity) and ``G3_box``
   the disc's change of momentum by the projection, hard (``surgery.BudgetUnclosed``,
   ``event-unclosed``), but ``G3_box`` beyond its limit only a flag where the identity G3 resolves
   it, the impulse of the merge (:func:`merge_impulse`); G4-G6 (energies, area) flags; then the
   record, the driver's bookkeeping (``on_made``), ``postcut_e<k>`` and ``events/event_<k>.json``
   (kind ``groove-fill``) with the filled region, the closure and the pocket.

A refusal leaves the problem unchanged; a fill whose hard closures miss leaves it rebuilt, wholly,
under a state that is not accepted. On a far head the driver then falls back to the far-head cut of
that head from the state before the event, which it loads back from ``precut_e<k>`` after a miss
(``driver.CollapseDriver._groove_fallback``); elsewhere either is a resumable stop.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .events import EVENT_SCHEMA, GROOVE_CLASSES, GROOVE_HANDLER, PRECUT_EVENT
from .mesh import Focus, FocusRegion
from .reconnect import Chain, liquid_axis_spans

GROOVE_KIND = "groove-fill"  # the kind of a fill made (events.EVENT_CLASSES["gap"])
REQUEST_KINDS = {"A-groove": "groove", "RF": "ring-fold"}  # the kind of the request of each class
REGION_RADIUS = 2.0  # the closure's region reaches this many mouth widths from its middle
CLOSURE_TOL = 0.05  # a new interface node within this times the mouth of the closure lies on it
LOOP_NODE_RANGE = (0.5, 1.5)  # the new loop's nodes against the planned points (recorded only)
# a refusal of the geometry that a flatter closure may not meet (the closure's own checks): retried
RETRIED = {"closure", "angle", "fill", "chains"}
assert set(REQUEST_KINDS) == set(GROOVE_CLASSES)
# G3_box beyond its limit is the impulse of the merge where the cutoff identity resolves it: its
# larger side at least this share of the disc's change of momentum (merge_impulse)
MERGE_RESOLVED = 0.5


def merge_impulse(closures: dict, floor_Pz: float = 0.0) -> dict:
    """The closures of a fill (``ringsurgery._ring_closures``, prefix ``G``) with ``G3_box`` read as
    the impulse of the merge the fill makes. A fill joins liquid that moved at different velocities
    across the gas it removes (the walls of the groove closing and sliding past each other); the
    projection is that merge's inelastic impulse, which acts near free surfaces and so moves
    momentum out of any disc about the site (its far field a dipole, whose flux through a sphere
    does not fall with its radius), physically, balanced in the end at the container's bottom
    (``global_identity``). The disc's change of momentum against ``P_ref = V_fill U`` (``U`` the
    surface's largest speed, a translation that makes no impulse) is then no test of the momentum
    balance, which is the cutoff identity G3: the disc's change equals the impulse's flux through
    its boundary. So ``G3_box`` beyond its limit stays a flag, out of ``hard``, where G3 holds and
    resolves it: the identity's larger side at least ``MERGE_RESOLVED`` of ``G3_box`` and not below
    ``floor_Pz`` (the momentum floor of the closures, five ``groove.sigma_Pz``), so that the
    momentum the disc lost is a measured flux, not noise the identity cannot see; otherwise it stays
    hard.
    Returns the closures, unchanged where ``G3_box`` is within its limit, else with ``G3_merge``
    (``side``, ``share``, ``floor``, ``resolved``)."""
    if "G3_box" not in closures.get("hard", ()):
        return closures
    phi = closures.get("G3_phi") or {}
    sides = [abs(float(phi.get(x, math.nan))) for x in ("phi_lhs", "phi_rhs")]
    side = max(sides) if all(math.isfinite(x) for x in sides) else math.nan
    box = float(closures["G3_box"])
    share = side / box if box > 0.0 and math.isfinite(side) else math.nan
    resolved = bool("G3" not in closures.get("flags", ()) and math.isfinite(share)
                    and share >= MERGE_RESOLVED and side >= float(floor_Pz))
    out = dict(closures, G3_merge={"side": side, "share": share, "floor": float(floor_Pz),
                                   "resolved": resolved})
    if resolved:
        out["hard"] = [x for x in closures["hard"] if x != "G3_box"]
    return out


def _spacing(P, loop: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """The node spacing of a chain (``selfcontact.fold_gap``'s: the mean of the chords beside a
    node, the end chord at an end) or of a loop (periodic), the arclength at each node and the
    whole length."""
    from .ring import loop_spacing, loop_steps

    if loop:
        ds = loop_steps(P)
        return loop_spacing(P), np.concatenate([[0.0], np.cumsum(ds)[:-1]]), float(ds.sum())
    ds = np.hypot(*np.diff(P, axis=0).T)
    sp = np.empty(len(P))
    sp[0], sp[-1] = ds[0], ds[-1]
    sp[1:-1] = 0.5 * (ds[:-1] + ds[1:])
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return sp, s, float(s[-1])


def fill_trigger(P, pair, *, loop: bool, fold: dict, mouth: float) -> dict:
    """The trigger of a fill read again at its pair (``fold``: the stops' ``ratio``, ``stop`` the
    absolute part of the limit, ``kappa_g delta``, and ``stop_elements``): the pair's distance
    ``gap``, the limit ``max(stop, stop_elements h_elem)``, the arclength between the two nodes
    (the shorter way round a loop), whether they still make a fold (``fold_ratio gap < arc``) that
    closes (``gap < limit``), ``holds``, and the fill's width ``mouth * limit``."""
    P = np.asarray(P, dtype=float).reshape(-1, 2)
    a, b = sorted(int(x) for x in pair)
    out = {"pair": [a, b], "holds": False}
    if not (0 <= a < b < len(P)):
        return out
    sp, s, S = _spacing(P, loop)
    gap = float(np.hypot(*(P[a] - P[b])))
    h_elem = float(sp[a] + sp[b])
    arc = float(s[b] - s[a])
    if loop:
        arc = min(arc, S - arc)
    limit = max(float(fold["stop"]), float(fold["stop_elements"]) * h_elem)
    out.update(gap=gap, h_elem=h_elem, arc=arc, limit=limit, width=float(mouth) * limit,
               ratio=float(fold["ratio"]),
               holds=bool(gap < limit and float(fold["ratio"]) * gap < arc))
    return out


def groove_fill(problem, events, reconnect, outdir: Path, request: dict, *, k: int, dt_prev: float,
                elements_per_scale: float, h_floor: float, fold: dict, core=None, driver=None,
                on_made=None, log=print) -> dict:
    """Fill the closing gas groove an event request names (``A-groove`` or ``RF``; module
    docstring), the ``k``-th event after the gas cut, under the block ``events`` with
    ``groove.enabled``. ``elements_per_scale`` is the run's after the cut, which
    ``groove.cut_elements`` replaces on the closure; ``h_floor`` the run's floor of the sizes after
    it; ``fold`` the stops' fold limits (``ratio``, ``stop``, ``stop_elements``) the trigger is read
    again with; ``core`` jetcore's mask of the open chain's core (None without the part), whose
    nodes no fill takes. Returns the record (schema ``faraday-event/2``, kind ``groove-fill``),
    also written to ``events/event_<k>.json`` whatever the outcome; raises ``surgery.CutRefused``
    with the problem unchanged, ``surgery.BudgetUnclosed`` where a hard closure misses (a stop to
    resume from ``precut_e<k>``), ``surgery.CutFailed`` otherwise after the remesh."""
    from .ringsurgery import _run_surgery

    out = Path(outdir)
    record = {"schema": EVENT_SCHEMA, "k": int(k), "kind": GROOVE_KIND,
              "class": request.get("class"), "handler": GROOVE_HANDLER}
    path = out / "events" / f"event_{int(k):03d}.json"
    return _run_surgery(lambda progress: _groove_fill(
        problem, events, reconnect, out, record, progress, request, path, k=int(k),
        dt_prev=float(dt_prev), elements_per_scale=float(elements_per_scale),
        h_floor=float(h_floor), fold=dict(fold), core=core, driver=driver, on_made=on_made,
        log=log), problem, out, record, path, log)


def _groove_fill(pb, events, s, out: Path, record: dict, progress: dict, request: dict,
                 path: Path, *, k: int, dt_prev: float, elements_per_scale: float, h_floor: float,
                 fold: dict, core, driver, on_made, log) -> dict:
    """The sequence of :func:`groove_fill`, filling ``record`` as it goes."""
    import shapely

    from .events import CAP_RETRY_HEIGHT, predict_unlocated
    from .groove import GrooveFillRefused, groove_fill_geometry
    from .ring import closed, loop_kept
    from .ringsurgery import (CORNER_TOL, RING_NOISE_FLOORS, _axis_ends, _base_record, _corners,
                              _focus_regions, _inside, _node_sets, _rebuild, _ring_closures,
                              _transfer_check, _zero_inside)
    from .surgery import (INTERFACE, BudgetUnclosed, CutFailed, CutRefused,
                          _axis_spans_of_mesh, _jsonable, _keep, _named_chains, box_integrals,
                          box_polyline_budgets, orient_interface_chart, project, site_grid,
                          transfer_warnings)

    def stage(name):
        progress["stage"] = name

    wall0 = time.time()
    gs = events.groove
    if getattr(pb, "recon", None) is None:
        raise CutRefused("the problem was not made with reconnect settings")
    if pb.cut is None:
        raise CutRefused("a fill of a groove follows the gas cut: the problem has none")
    if not gs.enabled:
        raise CutRefused("the fill of a groove is made only with the block's groove.enabled")
    kind, cls = request.get("kind"), request.get("class")
    if cls not in GROOVE_CLASSES or REQUEST_KINDS[cls] != kind:
        raise CutRefused(f"no fill of a groove for an event of kind {kind!r} and class {cls!r}")
    if not (dt_prev > 0.0 and math.isfinite(dt_prev)):
        raise CutRefused(f"the step before the fill must be positive, not {dt_prev!r}")
    delta = float(s.delta)
    name = request.get("chain")
    record.update(_base_record(pb, events, s, request, k, GROOVE_KIND, cls, GROOVE_HANDLER, name,
                               dt_prev))
    named, segs = _named_chains(pb)
    chains = dict(named)
    ring_stage = getattr(pb, "ring", None) is not None
    loop_segs = pb.interface_loops() if ring_stage else []
    loops = [np.column_stack([x["r"], x["z"]]) for x in loop_segs]
    if ring_stage:
        record["stage"] = "ring"
        removed_ring = pb.ring.get("removed") is not None
        if len(loops) != (0 if removed_ring else 1):
            raise CutRefused(f"the ring stage holds {len(loops)} loops of the free surface, not "
                             f"{0 if removed_ring else 1}")
    if cls == "RF":
        if not ring_stage or name != "ring" or len(loops) != 1 or set(chains) != {"open"}:
            raise CutRefused(f"a fold of the ring is filled in the ring stage, on its one loop: "
                             f"chain {name!r}, chains {sorted(chains)}, {len(loops)} loops")
        P, ends = loops[0], ("loop", "loop")
        velocity = np.column_stack([loop_segs[0]["u_r"], loop_segs[0]["u_z"]])
        others, beside = [chains["open"]], []
    else:
        wanted = {"open"} if ring_stage else {"open", "bubble"}
        if set(chains) != wanted or name not in chains:
            raise CutRefused(f"a groove of the chain {name!r} is filled on the chains "
                             f"{sorted(wanted)}, not {sorted(chains)}")
        P, ends = chains[name].points, chains[name].end_types
        velocity = np.column_stack([segs[name]["u_r"], segs[name]["u_z"]])
        others, beside = [c for x, c in named if x != name], loops
    trigger = fill_trigger(P, request.get("fold_pair") or (), loop=cls == "RF", fold=fold,
                           mouth=gs.mouth)
    trigger.update(kappa_g=events.gap.kappa_g, request_gap=request.get("fold_gap"),
                   request_limit=request.get("fold_limit"), jet=request.get("jet"))
    record["trigger"] = trigger
    if not trigger["holds"]:
        raise CutRefused(f"the fold of the request does not close under this block: {trigger}")
    if request.get("jet") == "cavity_tip":
        raise CutRefused("the groove at the base of the upward jet is not filled: its jet is the "
                         "one resolved, not a part of the far region (decision 22)")
    V_b = float(pb.cut["geometry"]["bubble_volume"])
    record["dumps"] = {}

    stage("precut")
    event = {**record["request"], "k": k}
    record["dumps"]["precut"] = _keep(pb, driver, out, f"precut_e{k:03d}",
                                      {"cut_stage": PRECUT_EVENT, "event": event})["path"]

    stage("geometry")
    eps = float(gs.cut_elements) if gs.cut_elements is not None else elements_per_scale
    protected = core if name == "open" else None
    attempts, g = [], None
    heights = (gs.cap_height,) + ((CAP_RETRY_HEIGHT,) if gs.cap_height != CAP_RETRY_HEIGHT else ())
    for eta in heights:
        try:
            g = groove_fill_geometry(name, P, ends, trigger["pair"], width=trigger["width"],
                                     eta=eta, elements_per_scale=eps, min_angle=s.min_cap_angle,
                                     reservoir_depth=s.reservoir_depth, V_b=V_b,
                                     max_fill=gs.max_fill, others=others, loops=beside,
                                     protected=protected, velocity=velocity)
            attempts.append({"eta": eta, "outcome": "ok"})
            break
        except GrooveFillRefused as e:
            attempts.append({"eta": eta, "outcome": "refused", "error": str(e),
                             "checks": e.checks})
            if set(e.checks.get("failed") or ()) - RETRIED:
                break  # a refusal no flatter closure meets
    if g is None:
        last = attempts[-1]["checks"]
        record["geometry"] = {"attempts": attempts, "side": last.get("side"),
                              "gas_share": last.get("gas_share"), "failed": last.get("failed")}
        raise CutRefused(f"the fill of the groove is refused: {attempts[-1]['error']}")
    record["geometry"] = {**g.summary(), "elements_per_scale": eps, "attempts": attempts,
                          "V_fill_over_V_b": g.V_fill / V_b,
                          "V_fill_over_delta3": g.V_fill / delta**3}

    stage("budgets")
    R_b = max(gs.box_radius * delta, 2.0 * g.L_fill)
    grid = site_grid(g.r_c, g.z_c, R_b)
    global_before = pb.cut_observables()
    box_before = box_integrals(pb, grid, {"fill": g.fill})
    box_before.pop("fill")
    rings_before = [Chain(closed(L), ("loop", "loop")) for L in loops]
    poly_before = box_polyline_budgets([c for _, c in named] + rings_before, g.z_c, R_b,
                                       s.reservoir_depth, r_centre=g.r_c)
    U = float(g.U_max) if math.isfinite(g.U_max) else 0.0
    scales = {"V_fill": g.V_fill, "U": U, "P_ref": g.V_fill * U, "KE_ref": 0.5 * g.V_fill * U**2}
    rmin = float(pb.cut.get("rmin") or pb.cut["delta"])
    h_cut = s.focus_size * rmin
    size = float(np.clip(g.h_c, h_floor, h_cut))
    mid = g.closure[len(g.closure) // 2]
    region = FocusRegion(f"groove_e{k:03d}", size=size, r=float(mid[0]), z=float(mid[1]),
                         radius=REGION_RADIUS * g.mouth)
    template = pb.get_mesh("liquid")._templatemesh
    old_focus = template.focus
    regions, grading = _focus_regions(old_focus, pb.mesh_settings.grading)
    new_regions = tuple(regions) + (region,)
    focus = Focus(size=min(x.size for x in new_regions), radius=max(x.radius for x in new_regions),
                  grading=grading, regions=new_regions)
    record["mesh"] = {"focus_before": None if old_focus is None else old_focus.to_dict(),
                      "region": {"name": region.name, "size": size, "r": region.r, "z": region.z,
                                 "radius": region.radius, "h_c": g.h_c, "h_floor": h_floor,
                                 "h_cut": h_cut}}
    if cls == "RF":
        geometry = {"chains": [chains["open"]], "loops": [g.points]}
    else:
        new_chain = Chain(np.asarray(g.points, dtype=float).copy(), tuple(ends))
        geometry = {"chains": [new_chain if x == name else c for x, c in named],
                    "loops": [L.copy() for L in loops]}
    old_chains = [c.points for _, c in named] + [closed(L) for L in loops]

    def restore():
        template.focus = old_focus

    stage("park")
    printed = _rebuild(pb, template, focus, geometry, {"axis_bulk": False, "reach": 0.0},
                       stage=stage, precut=record["dumps"]["precut"], restore=restore)

    stage("mesh-check")
    transfer = transfer_warnings("".join(printed))
    record["transfer"] = transfer
    new_named, new_segs = _named_chains(pb)
    new_loops = [np.column_stack([x["r"], x["z"]]) for x in pb.interface_loops()]
    record["mesh"].update(chains=[{"chain": x, "nodes": len(c.points),
                                   "end_types": list(c.end_types)} for x, c in new_named],
                          loops=[len(L) for L in new_loops], nodes=pb.get_mesh("liquid").nnode())
    if {x: c.end_types for x, c in new_named} != {x: c.end_types for x, c in named}:
        raise CutFailed(f"the fill's mesh has the chains {record['mesh']['chains']}, not those "
                        "planned", record["dumps"]["precut"])
    if len(new_loops) != len(loops):
        raise CutFailed(f"the fill's mesh has {len(new_loops)} loops, not {len(loops)}",
                        record["dumps"]["precut"])
    if cls == "RF":
        n_loop = len(new_loops[0])
        record["mesh"]["planned_loop_points"] = int(len(g.points))
        if not LOOP_NODE_RANGE[0] * len(g.points) <= n_loop <= LOOP_NODE_RANGE[1] * len(g.points):
            record["mesh"]["loop_nodes_flag"] = True  # recorded: the ring's sizing resamples it
    elif loops:  # a fill of the open chain in the ring stage keeps the ring
        kept = loop_kept(loops, new_loops, sigma_V=events.ring.sigma_V)
        record["loop"] = kept
        if kept["problems"]:
            raise CutFailed("the fill did not keep the ring: " + "; ".join(kept["problems"]),
                            record["dumps"]["precut"])
    spans = _axis_spans_of_mesh(pb)
    expected = liquid_axis_spans([z for _, c in new_named for _, z in c.axis_ends()])
    record["mesh"].update(axis_spans=spans, expected_axis_spans=[list(x) for x in expected])
    if len(spans) != len(expected) or any(abs(a - b) > 1e-9 * max(1.0, pb.nd.radius)
                                          for x, y in zip(spans, expected) for a, b in zip(x, y)):
        raise CutFailed(f"the fill's mesh has the axis spans {spans}, expected {expected}",
                        record["dumps"]["precut"])
    old_axis = liquid_axis_spans([z for _, c in named for _, z in c.axis_ends()])
    if len(old_axis) != len(expected) or any(abs(a - b) > 1e-9 * max(1.0, pb.nd.radius)
                                             for x, y in zip(old_axis, expected)
                                             for a, b in zip(x, y)):
        raise CutFailed(f"the fill moved the axis spans from {old_axis} to {expected}",
                        record["dumps"]["precut"])
    sets = _node_sets(pb)
    X = sets["interface"][1]
    predicted = predict_unlocated(X, old_chains)
    Pw = np.asarray(P, dtype=float)
    closure_line = shapely.LineString(np.vstack([Pw[g.i_s], g.closure, Pw[g.j_s]]))
    on_closure = (np.asarray(shapely.distance(closure_line, shapely.points(X))
                             <= CLOSURE_TOL * g.mouth, dtype=bool)
                  if len(X) else np.zeros(0, dtype=bool))
    # the bulk nodes in the old gas: inside the filled region but within a twentieth of the old
    # surface's shortest chord there, the sag of its quadratic elements
    bulk_inside = _inside(sets["bulk"][1], g.fill, 0.05 * g.h_arc)
    axis_gap = np.zeros(len(sets["axis"][1]), dtype=bool)
    corners = _corners(pb, _axis_ends([c for _, c in named]),
                       CORNER_TOL * float(pb.cut["sizing"]["h_tip"]))
    problems = _transfer_check(transfer, sets, predicted, on_closure, bulk_inside, axis_gap,
                               corners)
    transfer["corner_fields"] = sorted(pb.get_mesh(INTERFACE + "/axis").get_nodal_field_indices())
    if problems:
        raise CutFailed("the transfer of the fill is not as planned: " + "; ".join(problems),
                        record["dumps"]["precut"])
    global_transfer = pb.cut_observables()
    rings_after = [Chain(closed(L), ("loop", "loop")) for L in new_loops]
    poly_after = box_polyline_budgets([c for _, c in new_named] + rings_after, g.z_c, R_b,
                                      s.reservoir_depth, r_centre=g.r_c)

    stage("fill")
    fill = {"mode": gs.fill, "zeroed_bulk": 0, "zeroed_axis": 0}
    if gs.fill == "zero":
        fill.update(_zero_inside(pb, sets, g.fill, (math.inf, -math.inf), 0.0))
        pb.invalidate_cached_mesh_data()
    filled = box_integrals(pb, grid, {"fill": g.fill})["fill"]
    fill.update(Pz_fill=filled["Pz"], KE_fill=filled["KE"], V_grid=filled["V_grid"])
    record["fill"] = fill

    stage("projection")
    record["projection"] = {}
    settings = replace(s, dt0=dt_prev, restart=gs.restart or s.restart, projection=gs.projection)
    t_now = float(record["t"])
    project(pb, settings, t_now, record["projection"], box=grid)
    stage("chart")
    orient_interface_chart(pb)

    stage("postcut")
    global_after = pb.cut_observables()
    box = record["projection"].get("box")
    if box is not None:
        box_transfer, box_after = box["before"], box["after"]
    else:
        box_transfer = box_after = box_integrals(pb, grid)
    record["budgets"] = {
        "box": {"r": g.r_c, "z": g.z_c, "R": R_b, "n": grid["n"], "cells": int(len(grid["r"]))},
        "polylines": {"before": poly_before, "after": poly_after},
        "grid": {"before": box_before, "after_transfer": box_transfer,
                 "after_projection": box_after},
        "global": {"before": global_before, "after_transfer": global_transfer,
                   "after": global_after, "includes_ring": bool(loops)}}
    closures = _ring_closures({"V": g.V_fill, "Pz": fill["Pz_fill"], "KE": fill["KE_fill"]},
                              record["budgets"]["polylines"],
                              {"before": box_before, "after_transfer": box_transfer,
                               "after_projection": box_after}, box, scales,
                              {"A_cl": g.A_cl, "A_rem": g.A_rem},
                              sigma={"V": gs.sigma_V, "Pz": gs.sigma_Pz, "KE": gs.sigma_KE},
                              prefix="G")
    closures = merge_impulse(closures, RING_NOISE_FLOORS["R3_box"][1] * gs.sigma_Pz)
    closures["global_identity"] = {x: record["projection"].get(x, math.nan)
                                   for x in ("delta_Pz", "Pi_on_bottom")}
    record["closures"] = closures
    if closures["hard"]:
        missed = {x: closures[x] for x in closures["hard"]}
        raise BudgetUnclosed(f"the budget of the fill of the groove does not close: "
                             f"{closures['hard']} beyond their limits ({missed})")
    record["filled"] = {"V": g.V_fill, "Pz": fill["Pz_fill"], "KE": fill["KE_fill"]}
    record["gas_removed"] = {"V": g.V_fill, "V_over_V_b": g.V_fill / V_b,
                             "V_over_delta3": g.V_fill / delta**3,
                             "V_gas_before": g.V_gas_before, "V_gas_after": g.V_gas_after}
    record["outcome"] = "made"
    record["wall_seconds"] = time.time() - wall0
    made = _jsonable(record)
    pb.events = list(getattr(pb, "events", None) or []) + [made]
    if on_made is not None:
        on_made(made)
    made["dumps"]["postcut"] = _keep(pb, driver, out, f"postcut_e{k:03d}",
                                     {"cut_stage": "postcut-event", "event_k": k})["path"]
    pb.events[-1] = made
    full = dict(made, fill_polygon=g.fill.tolist(), closure=g.closure.tolist(),
                pocket=g.pocket.tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(full), indent=1) + "\n")
    rel = ", ".join(f"{x} {closures['relative'][x]:.3g}" for x in closures["relative"])
    log(f"groove fill {k} ({cls}, chain {name}): mouth {g.mouth:.6g}, filled {g.V_fill:.6g} "
        f"({g.V_fill / V_b:.3g} V_b); closures {rel}")
    return made
