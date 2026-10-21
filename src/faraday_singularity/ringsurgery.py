"""The ring of a pierced bubble on the finite-element problem: its cut, its removal and the stage
between them.

When the downward jet pierces the bubble (the request ``piercing``, class ``W1``, of
``events.post_cut_events``, made with ``ring.enabled``), the ring cut (:func:`ring_cut`) cuts the
gas on the axis: the column between the plug below and the jet's cap above becomes liquid, closed
by the closure of ``ring.ring_cut_geometry`` (an arc, a vertical segment of radius ``r_w`` and an
arc, tangent to both), and the bubble's rest is the ring, a closed loop of the free surface's own
boundary ``interface`` with the passive constant (zero) gas pressure of the whole free surface, the
bubble's. pyoomph's planner refuses a toroidal topology and parks no plan the project could use
for a loop (its chart is open on the whole boundary), so the cut is a rebuild the project owns:

1. **Admission**: a gas cut made and no ring yet, the block's ``ring.enabled``, a request of kind
   ``piercing`` and class ``W1`` whose trigger holds again under the block, a positive last step,
   the chains the open chain and the bubble; ``precut_e<k>`` kept.
2. **Geometry** (``ring.ring_cut_geometry``), refused (:class:`~faraday_singularity.surgery.
   CutRefused`, ``ring-cut-refused``) unless its hard checks hold.
3. **Budgets before**, within ``max(ring.box_radius delta, 2 L_fill)`` of the filled region's
   middle on the axis (``surgery.box_polyline_budgets``, ``surgery.box_integrals``), the global ones
   and those within ``surgery.LOCAL_RADIUS delta``.
4. **Remesh at once** from the planned geometry (the template's ``pending_ring_geometry``: the open
   chain and the ring; ``ring_stage`` and the ring's frozen sizing set, ``mesh.RingSizing``), under
   the driver's focus with the bubble's regions replaced by the frozen copies of those finer than
   the ring's size (``ring_<name>``) and the column's region (``ring_cut``). The transfer places the
   interface nodes by projection onto the old interface (the problem's chart turns the zeta
   transfer off with a loop on either mesh, ``problem.RingTolerantArclength``) and locates the axis
   nodes in the old bulk (``FaradayProblem._ring_transfer``). From here a failure is
   ``surgery.CutFailed``.
5. **Mesh check**: one open chain, one ring, the axis one span up to the cavity's tip, and the
   nodes the transfer leaves unlocated those the geometry predicts: of the interface
   (``events.predict_unlocated``, on the closure, which must hold nodes), of the bulk (inside the
   filled region) and of the axis (between the bubble's old tips); no node without a value but on
   the corner set, at most those of it the geometry makes new (none: :func:`_corners`), as
   ``surgery.transfer_problems`` checks a liquid cut's.
6. **Fill** (``ring.fill``: the blend, or zero) and the **projection** (``surgery.project``, with
   ``ring.projection`` and ``ring.restart`` or the cut's, the run's last step as the history's, the
   time unchanged), ``Pi = 0`` on every free surface, the ring's closure included.
7. **Budgets after and closures** (:func:`_ring_closures`, R1-R6 and ``R3_box``), of which the
   volume the liquid gains, the momentum the transfer carries and both momentum statements of the
   projection in its box, the cutoff identity of ``surgery._box_projection`` (R3) and the box's
   change of momentum (``R3_box``, the liquid cut's F3), are hard checks
   (``surgery.BudgetUnclosed``, ``event-unclosed``); then the record of the stage on the problem
   (``FaradayProblem.ring``), the driver's bookkeeping (``on_made``), ``postcut_e<k>``, the record
   ``events/event_<k>.json`` with the filled region and the ring.

Both momentum statements of the projection are hard, since neither alone settles what happens at a
piercing. Where the pressure impulse reaches the box's boundary (a source of it beside a free
surface is a dipole as seen from there, whose flux does not fall with the box's radius), the box's
momentum changes through that boundary, and the identity is the box's balance with the flux; where
``Pi = 0`` on a closure close to the axis keeps the impulse within the column, the box's momentum
barely changes, both sides of the identity lie far below ``P_ref`` and cancel, and its relative
error is set by the quadrature of the box's grid. The identity is therefore measured against the
larger of its sides (not against ``P_ref``, which would hide a miss of its whole size) and the
change against ``P_ref``, each floored at the box's noise; the projection's energy (R4) is a flag,
the energy identity of ``surgery.project`` the numerical check.

The ring stage lives on the problem (``FaradayProblem.ring``, its record, dumped with every state of
it) and on the mesh template (``ring_stage``, ``ring_sizing``: every rebuild meshes the ring as a
hole of the liquid at the sizing frozen at the cut, ``mesh.FaradayCylinderMesh.
_define_ring_geometry``). pyoomph rebuilds a loaded mesh from a sibling of the problem's template,
which in a new process holds neither, so a restart in the ring stage puts them back
(:func:`restore_ring`, called by ``driver.CollapseDriver.restore`` after the cut's own
``surgery.restore_cut``); the chart of the loaded interface needs nothing, the problem's arclength
chart charting a loop whatever the template.

Once the ring's volume falls below ``reconnect.bubble_stop`` of the bubble's at the gas cut (the
request ``ring-collapse``, class ``W2``, ``ring.ring_stage_events``), :func:`ring_removal` removes
it, the instantaneous inelastic collapse of its last part: the same sequence with no loop planned,
the bulk nodes inside the old ring the transfer's unlocated ones, the closures Q1-Q6 of the same
form, the removal logged in the record of the stage (``removed``). The window then ends
(``ring.after_removal`` ``stop``, the driver's ``bubble-vanishing``) or the run goes on with the
open chain alone (``continue``).
"""

from __future__ import annotations

import contextlib
import json
import math
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from .events import EVENT_SCHEMA, PRECUT_EVENT
from .mesh import Focus, FocusRegion, RingSizing
from .reconnect import Chain, liquid_axis_spans

# surgery (and with it pyoomph) is imported where a ring surgery runs, not with this module, which
# the driver's restore of the ring stage imports on any problem

# the stages of a ring surgery before it changes the problem: any failure up to them is a refusal
RING_UNCHANGED_STAGES = ("admission", "precut", "geometry", "budgets", "park")
# the closures of a ring surgery within its box (_ring_closures): the coefficient of the limit of
# each, times its own scale, and the hard ones (the volume the liquid gains, the momentum the
# transfer carries, the projection's momentum balance in the box, the cutoff identity, against the
# larger of its sides, and the box's momentum the projection changes, against P_ref, the liquid
# cut's F3); the kinetic energy and the area are flagged
RING_BUDGET_LIMITS = {"R1": 1e-2, "R2": 0.05, "R3": 1e-2, "R3_box": 0.05, "R4": 1.0, "R5": 1e-2,
                      "R6": 0.05}
HARD_RING_CLOSURES = ("R1", "R2", "R3", "R3_box")
# the floors of those limits: a multiple of the noise of the box (the block's ring.sigma_V,
# sigma_Pz, sigma_KE), the noise's name and multiple
RING_NOISE_FLOORS = {"R1": ("V", 5.0), "R2": ("Pz", 5.0), "R3": ("Pz", 5.0), "R3_box": ("Pz", 5.0),
                     "R4": ("KE", 3.0), "R6": ("KE", 5.0)}
CORNER_TOL = 0.05  # a new node of the interface's axis within this times h_tip of an old axis end
# the bubble's regions of the driver's focus, which the ring cut replaces by its frozen copies
BUBBLE_REGIONS = ("bubble_top", "bubble_bottom", "jet_base_bubble_top", "jet_base_bubble_bottom",
                  "neck_bubble", "kink_bubble")
LOOP_NODE_RANGE = (0.5, 1.5)  # the ring's nodes in the cut's mesh, against the planned points
BULK_SLACK = (0.1, 5)  # the bulk's unlocated nodes within this share and count of those predicted
AXIS_SLACK = 2  # and the axis's within this count


def restore_ring(problem, record: dict) -> None:
    """After loading a state of the ring stage: the stage and the ring's frozen sizing
    (``record["sizing"]``: ``h_ring``, ``grading``) on the template the loaded mesh is rebuilt from
    (``FaradayProblem.active_mesh_template``), with no ring geometry pending; the record on the
    problem (``FaradayProblem.ring``)."""
    sizing = (record or {}).get("sizing") or {}
    if "h_ring" not in sizing or "grading" not in sizing:
        raise RuntimeError("the record of the ring stage has no ring sizing")
    template = problem.active_mesh_template()
    template.ring_stage = True
    template.ring_sizing = RingSizing(h_ring=float(sizing["h_ring"]),
                                      grading=float(sizing["grading"]))
    template.pending_ring_geometry = None
    problem.ring = record


# --- shared pieces --------------------------------------------------------------------------------


def _bubble_region(name: str) -> bool:
    """Whether a region of the driver's focus belongs to the bubble (its tips, the bases of their
    jets, its neck, its kink region), under its own name or an event's (``<name>_e<k>``,
    ``<name>_cut``)."""
    return any(name == b or name.startswith(b + "_") for b in BUBBLE_REGIONS)


def frozen_ring_regions(old_focus: Focus | None, h_ring: float, size: float | None,
                        column: dict) -> list[FocusRegion]:
    """The ring's frozen regions: of the bubble's regions of the focus (:func:`_bubble_region`) each
    finer than the ring's size ``h_ring``, but its cut regions (``*_cut``), copied as
    ``ring_<name>`` at its centre and radius, at ``max(size, ring.size)``; and the column's
    (``ring_cut``, ``column``)."""
    out = []
    for g in (old_focus.regions if old_focus is not None else ()):
        if not _bubble_region(g.name) or g.name.endswith("_cut") or not g.size < h_ring:
            continue
        out.append(FocusRegion(f"ring_{g.name}", size=max(g.size, size or 0.0), r=g.r, z=g.z,
                               radius=g.radius))
    out.append(FocusRegion("ring_cut", size=column["size"], r=column["r"], z=column["z"],
                           radius=column["radius"]))
    return out


def _region_dict(g: FocusRegion) -> dict:
    return {"name": g.name, "size": g.size, "r": g.r, "z": g.z, "radius": g.radius}


def _inside(points, polygon, shrink: float) -> np.ndarray:
    """Which points lie inside ``polygon`` shrunk by ``shrink``."""
    import shapely
    from shapely.geometry import Polygon

    P = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(P):
        return np.zeros(0, dtype=bool)
    core = Polygon(polygon).buffer(-shrink)
    return np.asarray(shapely.contains_xy(core, P[:, 0], P[:, 1]), dtype=bool)


def _node_sets(problem) -> dict:
    """The nodes of the new mesh by kind, with their positions: the interface's, the axis's, and the
    bulk's (neither)."""
    from .surgery import INTERFACE, mesh_nodes

    iface = mesh_nodes(problem.get_mesh(INTERFACE))
    axis = mesh_nodes(problem.get_mesh("liquid/axis"))
    on = {(n.x(0), n.x(1)) for n in iface + axis}
    bulk = [n for n in problem.get_mesh("liquid").nodes() if (n.x(0), n.x(1)) not in on]

    def xy(nodes):
        return np.array([[n.x(0), n.x(1)] for n in nodes]).reshape(-1, 2)

    return {"interface": (iface, xy(iface)), "axis": (axis, xy(axis)), "bulk": (bulk, xy(bulk))}


def _reported(transfer: dict, *names) -> int:
    """pyoomph's count of unlocated nodes over the meshes ``names``."""
    return int(sum(transfer["unlocated"].get(n, [0, 0])[0] for n in names))


def _zero_inside(problem, sets: dict, polygon, z_range, shrink: float) -> dict:
    """Zero the velocity (current level) of the bulk nodes inside ``polygon`` and of the axis nodes
    with ``z`` in ``z_range``; interface nodes are left alone, as the gas cut's zero fill leaves
    them."""
    fi = problem.get_mesh("liquid").get_nodal_field_indices()
    iu, iw = fi["velocity_x"], fi["velocity_y"]
    nodes, X = sets["bulk"]
    inside = _inside(X, polygon, -shrink) if len(X) else np.zeros(0, dtype=bool)
    axis, A = sets["axis"]
    on_axis = (A[:, 1] > z_range[0]) & (A[:, 1] < z_range[1]) if len(A) else np.zeros(0, bool)
    for group, flags in ((nodes, inside), (axis, on_axis)):
        for n, flag in zip(group, flags):
            if flag:
                n.set_value(iu, 0.0)
                n.set_value(iw, 0.0)
    return {"zeroed_bulk": int(inside.sum()), "zeroed_axis": int(on_axis.sum())}


def _ring_closures(fill: dict, poly: dict, box: dict, phi: dict | None, scales: dict,
                   areas: dict, sigma: dict | None = None, prefix: str = "R") -> dict:
    """The closures of a ring surgery within its box, each against its limit (``RING_BUDGET_LIMITS``
    times its scale, never below the floor of ``RING_NOISE_FLOORS``, a multiple of the ring's noise
    ``sigma``: ``V``, ``Pz``, ``KE``, zero where not given): R1 the liquid volume the box gained
    (its polylines before and after, ``poly``) against the filled volume ``V_fill``; R2 the axial
    momentum the transfer carried (``box``: before, after the transfer and the fill, after the
    projection) against the filled region's momentum ``Pz_fill``, at the scale ``P_ref``; R3 the
    projection's momentum balance in the box, the cutoff identity ``|phi_lhs - phi_rhs|`` (``phi``,
    the projection's ``box``, None where no projection was made, R3 then zero), against the larger
    of its two sides, as ``surgery._box_projection`` states its relative error (a side missing or
    not finite makes R3 NaN, a miss); ``R3_box`` the box's axial momentum the projection changed,
    against ``P_ref``, the liquid cut's F3; R4 the kinetic energy the projection changed, against
    ``KE_ref`` (a flag: an inelastic merge in the box takes kinetic energy physically); R5 the
    area of the interface against the closure's less the removed tips' (``areas``); R6 the kinetic
    energy the transfer carried against ``KE_fill``. The scales: ``V_fill``, ``P_ref = V_fill U``,
    ``KE_ref = V_fill U^2 / 2`` (``U`` the closing speed of the cut, the ring's largest speed at
    its removal). ``flags`` the closures beyond their limits, ``hard`` those of
    ``HARD_RING_CLOSURES`` among them (named ``prefix`` R, or Q for the removal: ``Q3_box``).
    Both momentum statements of the projection are hard: the identity, that the box's momentum
    balance holds with the impulse's flux through the box's boundary, and the change itself, that
    the projection moved little momentum across that boundary; neither alone discriminates where
    the impulse stays within a thin column whose sides cancel (the identity's sides then lie far
    below ``P_ref``, its relative error set by the quadrature of the box's grid; below the floor it
    states only that the momentum it misses is below the box's noise)."""
    V, Pz_fill, KE_fill = fill["V"], fill["Pz"], fill["KE"]
    P_ref, KE_ref = scales["P_ref"], scales["KE_ref"]
    before, moved, after = box["before"], box["after_transfer"], box["after_projection"]
    dA = areas["A_cl"] - areas["A_rem"]
    projected = phi is not None
    phi = phi or {}
    lhs, rhs = float(phi.get("phi_lhs", math.nan)), float(phi.get("phi_rhs", math.nan))
    sides = math.isfinite(lhs) and math.isfinite(rhs)
    values = {"R1": abs(poly["after"]["V_liquid"] - poly["before"]["V_liquid"] - V),
              "R2": abs(moved["Pz"] - before["Pz"] - Pz_fill),
              "R3": (abs(lhs - rhs) if sides else math.nan) if projected else 0.0,
              "R3_box": abs(after["Pz"] - moved["Pz"]),
              "R4": abs(after["KE"] - moved["KE"]),
              "R5": abs(poly["after"]["A"] - poly["before"]["A"] - dA),
              "R6": abs(moved["KE"] - before["KE"] - KE_fill)}
    scale = {"R1": V, "R2": P_ref,
             "R3": (max(abs(lhs), abs(rhs)) if sides else math.nan) if projected else 0.0,
             "R3_box": P_ref, "R4": KE_ref, "R5": abs(dA), "R6": KE_ref}
    noise = {"V": 0.0, "Pz": 0.0, "KE": 0.0, **{k: float(v) for k, v in (sigma or {}).items()}}

    def limit(k):
        floor = (RING_NOISE_FLOORS[k][1] * noise[RING_NOISE_FLOORS[k][0]]
                 if k in RING_NOISE_FLOORS else 0.0)
        own = RING_BUDGET_LIMITS[k] * scale[k]
        return own if math.isnan(own) else max(own, floor)  # NaN stays a miss

    limits = {k: limit(k) for k in values}
    flags = [k for k in values if not values[k] <= limits[k]]

    def name(k):
        return prefix + k[1:]

    def named(d):
        return {name(k): v for k, v in d.items()}

    return {**named(values), "limits": named(limits),
            "relative": named({k: values[k] / scale[k] if scale[k] else math.nan
                               for k in values}),
            "flags": [name(k) for k in flags],
            "hard": [name(k) for k in flags if k in HARD_RING_CLOSURES], "sigma": noise,
            "scales": dict(scales), f"{prefix}3_projected": projected,
            f"{prefix}3_phi": {x: phi.get(x, math.nan) for x in ("phi_lhs", "phi_rhs",
                                                                 "phi_error",
                                                                 "phi_relative_error")}}


def _run_surgery(body, problem, outdir: Path, record: dict, path: Path, log) -> dict:
    """Run a ring surgery's sequence ``body(progress)`` as ``surgery.liquid_cut`` runs a liquid
    cut's: its record written to ``events/event_<k>.json`` whatever the outcome (``made``,
    ``refused``, ``unclosed``, ``failed`` or ``interrupted``, with ``failed_at``), any other
    ``Exception`` raised as a refusal up to ``RING_UNCHANGED_STAGES`` and as a failure from the
    remesh on, an interruption never turned into an outcome."""
    from .surgery import BudgetUnclosed, CutFailed, CutRefused, _jsonable

    progress = {"stage": "admission"}
    try:
        return body(progress)
    except BaseException as e:
        stage, err, extra = progress["stage"], e, {}
        if isinstance(e, Exception) and not isinstance(e, (CutRefused, CutFailed)):
            message = f"at stage {stage}: {type(e).__name__}: {e}"
            err = (CutRefused(f"no ring surgery was made ({message})")
                   if stage in RING_UNCHANGED_STAGES else
                   CutFailed(f"the ring surgery failed after changing the problem ({message})"))
            extra["traceback"] = traceback.format_exception(e)
        outcome = ("refused" if isinstance(err, CutRefused) else
                   "unclosed" if isinstance(err, BudgetUnclosed) else
                   "failed" if isinstance(err, Exception) else "interrupted")
        failed = _jsonable({**record, "outcome": outcome, "failed_at": stage,
                            "error": f"{type(err).__name__}: {err}", **extra})
        if isinstance(err, (CutRefused, CutFailed)):
            err.record = failed
            if err.precut is None:
                err.precut = (record.get("dumps") or {}).get("precut")
        with contextlib.suppress(OSError):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(failed, indent=1) + "\n")
        if err is e:
            raise
        raise err from e


def _rebuild(pb, template, focus: Focus, geometry: dict, transfer: dict, *, stage, precut,
             restore) -> list[str]:
    """Remesh at once from the planned ring geometry (the template's ``pending_ring_geometry``)
    under ``focus``, with the transfer settings ``transfer`` on the problem while it lasts; returns
    what the process printed (pyoomph's counts of the transfer). A failure before the remesh puts
    back what ``restore`` saved and is a refusal, from it on a failure."""
    from .surgery import CutFailed, CutRefused, captured_stdout

    remeshing = False
    try:
        template.focus = focus
        template.pending_ring_geometry = geometry
        pb._ring_transfer = transfer
        stage("remesh")
        remeshing = True
        with captured_stdout() as printed:
            pb.force_remesh({template})
        pb._domains_to_remesh.clear()
        return printed
    except BaseException as e:
        if not remeshing:
            restore()
        if not isinstance(e, Exception):
            raise
        if not remeshing:
            raise CutRefused(f"the ring surgery's rebuild could not be set up "
                             f"({type(e).__name__}: {e})") from e
        raise CutFailed(f"the remesh of the ring surgery failed ({type(e).__name__}: {e})",
                        precut) from e
    finally:
        pb._ring_transfer = None
        template.pending_ring_geometry = None


def _base_record(pb, events, s, request, k, kind, cls, handler, chain, dt_prev) -> dict:
    t_now = pb.get_current_time(dimensional=False, as_float=True)
    t_cut = float(pb.cut["t_cut"])
    t_p = (pb.cut.get("pinch") or {}).get("t_p")
    return {"schema": EVENT_SCHEMA, "k": int(k), "kind": kind, "class": cls, "handler": handler,
            "chain": chain, "tip": None, "t": t_now, "tau": t_now - t_cut,
            "tau_p": t_now - t_p if t_p is not None else math.nan, "delta": float(s.delta),
            "request": {key: v for key, v in request.items() if key != "reason"},
            "settings": events.to_dict(), "settings_sha256": events.sha256(),
            "dt": {"history": dt_prev}}


def _focus_regions(old_focus: Focus | None, grading_default: float):
    if old_focus is None:
        return (), grading_default
    regions = tuple(old_focus.regions) or (FocusRegion("focus", size=old_focus.size, r=0.0, z=0.0,
                                                       radius=old_focus.radius, axial=True),)
    return regions, old_focus.grading


def _axis_ends(chains) -> list:
    """The ends on the axis of the chains (``reconnect.Chain``) of the old interface."""
    return [c.points[k] for c in chains for k, end in ((0, 0), (-1, 1))
            if c.end_types[end] == "axis"]


def _corners(problem, old_ends, tol: float) -> dict:
    """The nodes of the new interface's axis (its corner set, the mesh ``interface/axis``) and how
    many of them lie farther than ``tol`` from every axis end of the old interface (``old_ends``,
    points): those a transfer may leave without a value of their own fields (``_kin_bc``), as the
    stump's new tip of a liquid cut; the ring surgeries make none, the corner set staying the
    open chain's tip on the axis."""
    from .surgery import INTERFACE, mesh_nodes

    X = np.array([[n.x(0), n.x(1)] for n in mesh_nodes(problem.get_mesh(INTERFACE + "/axis"))])
    X = X.reshape(-1, 2)
    E = np.asarray(old_ends, dtype=float).reshape(-1, 2)
    if not len(X):
        return {"nodes": 0, "new": 0}
    if not len(E):
        return {"nodes": len(X), "new": len(X)}
    d = np.min(np.hypot(X[:, None, 0] - E[None, :, 0], X[:, None, 1] - E[None, :, 1]), axis=1)
    return {"nodes": len(X), "new": int(np.sum(d > tol))}


def _transfer_check(transfer: dict, sets: dict, predicted_iface: np.ndarray, on_closure,
                    bulk_inside: np.ndarray, axis_in_gap: np.ndarray,
                    corners: dict) -> list[str]:
    """What the checks of a ring surgery's transfer find wrong in its record, as
    ``surgery.transfer_problems`` checks a liquid cut's: pyoomph's count of unlocated interface
    nodes within ``UNLOCATED_SLACK`` of the prediction and every predicted one on the closure
    (``on_closure`` None, the removal: none predicted at all), and the closure holding nodes of the
    new interface (``closure_nodes``, the liquid cut's ``cap_nodes``); its count of the bulk's
    within ``BULK_SLACK`` of the bulk nodes predicted (inside the filled region) and of the axis's
    within ``AXIS_SLACK`` of those predicted (in the gap of the old axis); and no node without a
    value but on the corner set (``interface/axis``), at most those of it the geometry makes new
    (``corners``, :func:`_corners`: none for either ring surgery)."""
    from .surgery import INTERFACE, UNLOCATED_SLACK

    problems = []
    rep_if = _reported(transfer, INTERFACE)
    if abs(rep_if - int(predicted_iface.sum())) > UNLOCATED_SLACK:
        problems.append(f"{rep_if} interface nodes unlocated, {int(predicted_iface.sum())} "
                        "predicted")
    if on_closure is None:
        if predicted_iface.any():
            problems.append(f"{int(predicted_iface.sum())} interface nodes predicted unlocated")
    else:
        if np.any(predicted_iface & ~on_closure):
            problems.append(f"{int(np.sum(predicted_iface & ~on_closure))} predicted unlocated "
                            "interface nodes off the closure")
        if not np.any(on_closure):
            problems.append("no interface node on the planned closure")
    rep_bulk = _reported(transfer, "liquid")
    want = int(bulk_inside.sum())
    if abs(rep_bulk - want) > max(BULK_SLACK[0] * want, BULK_SLACK[1]):
        problems.append(f"{rep_bulk} bulk nodes unlocated, {want} predicted")
    rep_axis = _reported(transfer, "liquid/axis")
    want = int(axis_in_gap.sum())
    if abs(rep_axis - want) > AXIS_SLACK:
        problems.append(f"{rep_axis} axis nodes unlocated, {want} predicted")
    no_value, corner = transfer.get("no_value") or {}, INTERFACE + "/axis"
    if set(no_value) - {corner} or no_value.get(corner, [0])[0] > corners["new"]:
        problems.append(f"nodes without a value beyond the {corners['new']} new node(s) of the "
                        f"corner set: {no_value}")
    transfer.update(predicted_unlocated=int(predicted_iface.sum()),
                    reported_unlocated=rep_if,
                    predicted_off_closure=int(np.sum(predicted_iface & ~on_closure))
                    if on_closure is not None else int(predicted_iface.sum()),
                    closure_nodes=int(np.sum(on_closure)) if on_closure is not None else None,
                    corners=dict(corners),
                    former_gas={"bulk": rep_bulk, "axis": rep_axis,
                                "predicted_bulk": int(bulk_inside.sum()),
                                "predicted_axis": int(axis_in_gap.sum())})
    return problems


# --- the ring cut ---------------------------------------------------------------------------------


def ring_cut(problem, events, reconnect, outdir: Path, request: dict, *, k: int, dt_prev: float,
             elements_per_scale: float, h_floor: float, driver=None, on_made=None,
             log=print) -> dict:
    """Cut the gas of a bubble its downward jet pierces on the axis, and keep its rest as a ring
    (module docstring): the ``k``-th event after the gas cut, the request ``piercing`` (class
    ``W1``) of ``events.post_cut_events`` under the block ``events`` with ``ring.enabled``.
    ``elements_per_scale`` is the run's after the cut (``reconnect.elements_per_scale`` or the
    collapse's), which ``ring.cut_elements`` replaces on the closure, and ``h_floor`` the run's
    floor of the sizes after the cut (the column's size is ``clip(r_w / eps, h_floor, h_cut)``).
    Returns the record (schema ``faraday-event/2``), also written to ``events/event_<k>.json``
    whatever the outcome; raises ``surgery.CutRefused`` with the problem unchanged,
    ``surgery.BudgetUnclosed`` where a hard closure misses (a stop to resume from ``precut_e<k>``),
    ``surgery.CutFailed`` otherwise after the remesh."""
    out = Path(outdir)
    record = {"schema": EVENT_SCHEMA, "k": int(k), "kind": request.get("kind"),
              "class": request.get("class"), "handler": "ring"}
    path = out / "events" / f"event_{int(k):03d}.json"
    return _run_surgery(lambda progress: _ring_cut(
        problem, events, reconnect, out, record, progress, request, path, k=int(k),
        dt_prev=float(dt_prev), elements_per_scale=float(elements_per_scale),
        h_floor=float(h_floor), driver=driver, on_made=on_made, log=log),
        problem, out, record, path, log)


def _ring_cut(pb, events, s, out: Path, record: dict, progress: dict, request: dict, path: Path, *,
              k: int, dt_prev: float, elements_per_scale: float, h_floor: float, driver, on_made,
              log) -> dict:
    """The sequence of :func:`ring_cut`, filling ``record`` as it goes."""
    import shapely

    from .events import predict_unlocated
    from .reconnect import revolved_volume
    from .ring import RingCutRefused, closed, loop_record, ring_cut_geometry
    from .surgery import (INTERFACE, LOCAL_RADIUS, BudgetUnclosed, CutFailed, CutRefused,
                          _axis_spans_of_mesh, _jsonable, _keep, _named_chains, box_grid,
                          box_integrals,
                          box_polyline_budgets, orient_interface_chart, project,
                          transfer_warnings)

    def stage(name):
        progress["stage"] = name

    wall0 = time.time()
    ring = events.ring
    if getattr(pb, "recon", None) is None:
        raise CutRefused("the problem was not made with reconnect settings")
    if pb.cut is None:
        raise CutRefused("a ring cut follows the gas cut: the problem has none")
    if getattr(pb, "ring", None) is not None:
        raise CutRefused("the problem holds a ring already: one ring per run")
    if not ring.enabled:
        raise CutRefused("the ring cut is made only with the block's ring.enabled")
    kind, cls = request.get("kind"), request.get("class")
    if (kind, cls) != ("piercing", "W1"):
        raise CutRefused(f"no ring cut for an event of kind {kind!r} and class {cls!r}")
    if not (dt_prev > 0.0 and math.isfinite(dt_prev)):
        raise CutRefused(f"the step before the cut must be positive, not {dt_prev!r}")
    delta = float(s.delta)
    record.update(_base_record(pb, events, s, request, k, kind, cls, "ring", "bubble", dt_prev))
    named, segs = _named_chains(pb)
    chains = dict(named)
    if set(chains) != {"open", "bubble"}:
        raise CutRefused(f"the ring cut needs the open chain and the bubble, not {sorted(chains)}")
    bubble, cavity = chains["bubble"], chains["open"]
    P = bubble.points
    seg = segs["bubble"]
    h_min = float(pb.cut["sizing"]["h_tip"])
    length = float(P[-1, 1] - P[0, 1])
    limit = max(ring.gap * delta, 3.0 * h_min)
    V_b = float(pb.cut["geometry"]["bubble_volume"])
    V_bubble = revolved_volume(P)
    ghosts = [{"k": e.get("k"), "status": (e.get("ghost") or {}).get("status"),
               "far_tip": (e.get("ghost") or {}).get("far_tip")}
              for e in (pb.events or []) if isinstance(e.get("ghost"), dict)]
    record["trigger"] = {"bubble_length": length, "limit": limit, "gap": ring.gap,
                         "gap_over_delta": length / delta, "z_bubble_bottom": float(P[0, 1]),
                         "w_bubble_bottom": float(seg["u_z"][0]), "z_bubble_top": float(P[-1, 1]),
                         "w_bubble_top": float(seg["u_z"][-1]),
                         "U_close": abs(float(seg["u_z"][0]) - float(seg["u_z"][-1])),
                         "V_bubble": V_bubble, "V_bubble_over_V_b": V_bubble / V_b,
                         "ghosts": ghosts}
    if not length < limit:
        raise CutRefused(f"the bubble's tips lie {length:.6g} apart, not within the trigger "
                         f"max(ring.gap delta, 3 h_min) = {limit:.6g}")
    record["dumps"] = {}

    stage("precut")
    event = {**record["request"], "k": k}
    record["dumps"]["precut"] = _keep(pb, driver, out, f"precut_e{k:03d}",
                                      {"cut_stage": PRECUT_EVENT, "event": event})["path"]

    stage("geometry")
    eps = float(ring.cut_elements) if ring.cut_elements is not None else elements_per_scale
    try:
        g = ring_cut_geometry(bubble, cavity, ring, elements_per_scale=eps,
                              min_angle=s.min_cap_angle, reservoir_depth=s.reservoir_depth,
                              V_b=V_b, bubble_stop=s.bubble_stop)
    except RingCutRefused as e:
        record["geometry"] = {"checks": e.checks}
        raise CutRefused(f"the ring cut is refused: {e}") from e
    record["geometry"] = {**g.summary(), "elements_per_scale": eps}
    record["trigger"].update(s_t=g.s_t, z_shoulder=g.z_shoulder,
                             w_splice_plug=float(seg["u_z"][g.i_b]),
                             w_splice_jet=float(seg["u_z"][g.i_t]))
    record["ring"] = {**loop_record(g.loop), "V0_over_V_b": g.V_ring / V_b,
                      "open_gap": g.checks.get("open_gap")}

    stage("budgets")
    R_b = max(ring.box_radius * delta, 2.0 * g.L_fill)
    grid, grid20 = box_grid(g.z_mid, R_b), box_grid(g.z_mid, LOCAL_RADIUS * delta)
    global_before = pb.cut_observables()
    box_before = box_integrals(pb, grid, {"fill": g.fill})
    box_before.pop("fill")
    box20_before = box_integrals(pb, grid20)
    poly_before = box_polyline_budgets([cavity, bubble], g.z_mid, R_b, s.reservoir_depth)
    U_close = abs(float(seg["u_z"][g.i_b]) - float(seg["u_z"][g.i_t]))
    scales = {"V_fill": g.V_fill, "U": U_close, "P_ref": g.V_fill * U_close,
              "KE_ref": 0.5 * g.V_fill * U_close**2}
    rmin = float(pb.cut.get("rmin") or pb.cut["delta"])
    h_cut = s.focus_size * rmin
    h_ring = max(h_cut, ring.size or 0.0)
    h_col = float(np.clip(g.r_w / eps, h_floor, h_cut))
    column = {"name": "ring_cut", "size": h_col, "r": 0.0, "z": g.z_mid,
              "radius": 0.5 * (g.z_t - g.z_b) + 4.0 * g.rho}

    template = pb.get_mesh("liquid")._templatemesh
    old_focus = template.focus
    old_stage = (getattr(template, "ring_stage", False), getattr(template, "ring_sizing", None))
    regions, grading = _focus_regions(old_focus, pb.mesh_settings.grading)
    frozen = frozen_ring_regions(old_focus, h_ring, ring.size, column)
    kept = [x for x in regions if not _bubble_region(x.name)]
    new_regions = tuple(kept) + tuple(frozen)
    focus = Focus(size=min(x.size for x in new_regions), radius=max(x.radius for x in new_regions),
                  grading=grading, regions=new_regions)
    record["mesh"] = {"focus_before": None if old_focus is None else old_focus.to_dict(),
                      "frozen": [_region_dict(x) for x in frozen if x.name != "ring_cut"],
                      "ring_cut_region": column, "h_ring": h_ring, "h_col": h_col,
                      "h_cut": h_cut,
                      "dropped": [x.name for x in regions if _bubble_region(x.name)]}
    old_chains = [cavity.points, P]  # the old interface: the open chain and the bubble

    def restore():
        template.focus = old_focus
        template.ring_stage, template.ring_sizing = old_stage

    stage("park")
    template.ring_stage = True
    template.ring_sizing = RingSizing(h_ring=h_ring, grading=grading)
    printed = _rebuild(pb, template, focus, {"chains": [cavity], "loops": [g.loop]},
                       {"axis_bulk": True, "reach": 2.0 * g.rho}, stage=stage,
                       precut=record["dumps"]["precut"], restore=restore)

    stage("mesh-check")
    transfer = transfer_warnings("".join(printed))
    record["transfer"] = transfer
    new_segs, loops = pb.interface_segments(), pb.interface_loops()
    record["mesh"].update(chains=[{"open": c["open"], "nodes": len(c["r"]),
                                   "end_types": list(c["end_types"])} for c in new_segs],
                          loops=[len(x["r"]) for x in loops],
                          planned_loop_points=int(len(g.loop)),
                          nodes=pb.get_mesh("liquid").nnode())
    if len(new_segs) != 1 or not new_segs[0]["open"] or len(loops) != 1:
        raise CutFailed(f"the ring cut's mesh has {len(new_segs)} chains and {len(loops)} loops, "
                        "not the open chain and the ring", record["dumps"]["precut"])
    n_loop = len(loops[0]["r"])
    if not LOOP_NODE_RANGE[0] * len(g.loop) <= n_loop <= LOOP_NODE_RANGE[1] * len(g.loop):
        record["mesh"]["loop_nodes_flag"] = True  # recorded: the frozen sizing resamples the ring
    spans = _axis_spans_of_mesh(pb)
    expected = liquid_axis_spans([float(new_segs[0]["z"][0])])
    record["mesh"].update(axis_spans=spans, expected_axis_spans=[list(x) for x in expected])
    if len(spans) != 1 or any(abs(a - b) > 1e-9 * max(1.0, pb.nd.radius)
                              for a, b in zip(spans[0], expected[0])):
        raise CutFailed(f"the ring cut's mesh has the axis spans {spans}, expected {expected}",
                        record["dumps"]["precut"])
    sets = _node_sets(pb)
    X = sets["interface"][1]
    predicted = predict_unlocated(X, old_chains)
    closure_line = shapely.LineString(np.vstack([[g.r_t, g.z_t], g.closure, [g.r_b, g.z_b]]))
    on_closure = np.asarray(shapely.distance(closure_line, shapely.points(X)) <= 0.05 * g.rho,
                            dtype=bool) if len(X) else np.zeros(0, dtype=bool)
    bulk_inside = _inside(sets["bulk"][1], g.fill, 0.05 * h_col)
    A = sets["axis"][1]
    tol = 0.05 * h_col
    axis_gap = ((A[:, 1] > float(P[0, 1]) + tol) & (A[:, 1] < float(P[-1, 1]) - tol)
                if len(A) else np.zeros(0, dtype=bool))
    corners = _corners(pb, _axis_ends([cavity, bubble]), CORNER_TOL * h_min)
    problems = _transfer_check(transfer, sets, predicted, on_closure, bulk_inside, axis_gap,
                               corners)
    transfer["corner_fields"] = sorted(pb.get_mesh(INTERFACE + "/axis").get_nodal_field_indices())
    if problems:
        raise CutFailed("the transfer of the ring cut is not as planned: " + "; ".join(problems),
                        record["dumps"]["precut"])
    global_transfer = pb.cut_observables()
    L_new = np.column_stack([loops[0]["r"], loops[0]["z"]])
    open_new = Chain(np.column_stack([new_segs[0]["r"], new_segs[0]["z"]]),
                     tuple(new_segs[0]["end_types"]))
    open_new.points[0, 0] = 0.0
    poly_after = box_polyline_budgets([open_new, Chain(closed(L_new), ("loop", "loop"))], g.z_mid,
                                      R_b, s.reservoir_depth)

    stage("fill")
    fill = {"mode": ring.fill, "zeroed_bulk": 0, "zeroed_axis": 0}
    if ring.fill == "zero":
        fill.update(_zero_inside(pb, sets, g.fill, (float(P[0, 1]), float(P[-1, 1])), 0.0))
        pb.invalidate_cached_mesh_data()
    filled = box_integrals(pb, grid, {"fill": g.fill})["fill"]
    fill.update(Pz_fill=filled["Pz"], KE_fill=filled["KE"], V_grid=filled["V_grid"])
    record["fill"] = fill

    stage("projection")
    record["projection"] = {}
    settings = replace(s, dt0=dt_prev, restart=ring.restart or s.restart,
                       projection=ring.projection)
    t_now = float(record["t"])
    project(pb, settings, t_now, record["projection"], box=(g.z_mid, R_b))
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
        "box": {"z": g.z_mid, "R": R_b, "n": grid["n"]},
        "polylines": {"before": poly_before, "after": poly_after},
        "grid": {"before": box_before, "after_transfer": box_transfer,
                 "after_projection": box_after},
        "box20": {"before": box20_before, "after": box_integrals(pb, grid20)},
        "global": {"before": global_before, "after_transfer": global_transfer,
                   "after": global_after, "includes_ring": True}}
    closures = _ring_closures({"V": g.V_fill, "Pz": fill["Pz_fill"], "KE": fill["KE_fill"]},
                              record["budgets"]["polylines"],
                              {"before": box_before, "after_transfer": box_transfer,
                               "after_projection": box_after}, box, scales,
                              {"A_cl": g.A_cl, "A_rem": g.A_rem},
                              sigma={"V": ring.sigma_V, "Pz": ring.sigma_Pz, "KE": ring.sigma_KE})
    closures["global_identity"] = {x: record["projection"].get(x, math.nan)
                                   for x in ("delta_Pz", "Pi_on_bottom")}
    record["closures"] = closures
    if closures["hard"]:
        missed = {x: closures[x] for x in closures["hard"]}
        raise BudgetUnclosed(f"the budget of the ring cut does not close: {closures['hard']} "
                             f"beyond their limits ({missed})")
    record["filled"] = {"V": g.V_fill, "Pz": fill["Pz_fill"], "KE": fill["KE_fill"]}
    pb.ring = _jsonable({
        "k_cut": k, "t_cut_ring": t_now, "V_b": V_b, "V0": g.V_ring, "rho": g.rho,
        "r_w": g.r_w, "z_mid": g.z_mid,
        "sizing": {"h_ring": h_ring, "h_col": h_col, "grading": grading,
                   "frozen": [_region_dict(x) for x in frozen if x.name != "ring_cut"],
                   "ring_cut": column},
        "fill": dict(record["filled"]), "removed": None,
        "settings_sha256": events.sha256()})
    record["outcome"] = "made"
    record["wall_seconds"] = time.time() - wall0
    made = _jsonable(record)
    pb.events = list(getattr(pb, "events", None) or []) + [made]
    if on_made is not None:
        on_made(made)
    made["dumps"]["postcut"] = _keep(pb, driver, out, f"postcut_e{k:03d}",
                                     {"cut_stage": "postcut-event", "event_k": k})["path"]
    pb.events[-1] = made
    full = dict(made, fill_polygon=g.fill.tolist(), ring_loop=g.loop.tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(full), indent=1) + "\n")
    rel = ", ".join(f"{x} {closures['relative'][x]:.3g}" for x in closures["relative"])
    log(f"ring cut {k} (rho {g.rho:.6g}, r_w {g.r_w:.6g}): filled {g.V_fill:.6g}, ring "
        f"{g.V_ring / V_b:.4g} V_b; closures {rel}")
    return made


# --- the ring's removal ---------------------------------------------------------------------------


def ring_removal(problem, events, reconnect, outdir: Path, request: dict, *, k: int,
                 dt_prev: float, elements_per_scale: float, h_floor: float, driver=None,
                 on_made=None, log=print) -> dict:
    """Remove the ring once its volume has fallen below ``reconnect.bubble_stop`` of the bubble's at
    the gas cut (the request ``ring-collapse``, class ``W2``, of ``ring.ring_stage_events``), the
    ``k``-th event after the gas cut, and project (module docstring): the rebuild with no loop, the
    ring's region filled (``ring.fill``), the closures Q1-Q6 (Q1-Q3 hard), the removal logged in the
    record of the stage (``FaradayProblem.ring["removed"]``) and in the event's record, with what
    the run does next (``after_removal``). Arguments, record and errors as :func:`ring_cut`."""
    out = Path(outdir)
    record = {"schema": EVENT_SCHEMA, "k": int(k), "kind": request.get("kind"),
              "class": request.get("class"), "handler": "ring-removal"}
    path = out / "events" / f"event_{int(k):03d}.json"
    return _run_surgery(lambda progress: _ring_removal(
        problem, events, reconnect, out, record, progress, request, path, k=int(k),
        dt_prev=float(dt_prev), driver=driver, on_made=on_made, log=log),
        problem, out, record, path, log)


def _ring_removal(pb, events, s, out: Path, record: dict, progress: dict, request: dict,
                  path: Path, *, k: int, dt_prev: float, driver, on_made, log) -> dict:
    """The sequence of :func:`ring_removal`, filling ``record`` as it goes."""
    from .events import predict_unlocated
    from .reconnect import revolved_volume
    from .ring import closed, loop_record, signed_area
    from .surgery import (LOCAL_RADIUS, BudgetUnclosed, CutFailed, CutRefused,
                          _axis_spans_of_mesh, _jsonable, _keep, _named_chains, box_grid,
                          box_integrals,
                          box_polyline_budgets, orient_interface_chart, project,
                          transfer_warnings)

    def stage(name):
        progress["stage"] = name

    wall0 = time.time()
    ring = events.ring
    st = getattr(pb, "ring", None)
    if getattr(pb, "recon", None) is None or pb.cut is None:
        raise CutRefused("the ring's removal follows the gas cut and the ring cut")
    if st is None or st.get("removed") is not None:
        raise CutRefused("the problem holds no ring to remove")
    if not ring.enabled:
        raise CutRefused("the ring's removal is made only with the block's ring.enabled")
    kind, cls = request.get("kind"), request.get("class")
    if (kind, cls) != ("ring-collapse", "W2"):
        raise CutRefused(f"no removal of the ring for an event of kind {kind!r} and class {cls!r}")
    if not (dt_prev > 0.0 and math.isfinite(dt_prev)):
        raise CutRefused(f"the step before the removal must be positive, not {dt_prev!r}")
    delta = float(s.delta)
    record.update(_base_record(pb, events, s, request, k, kind, cls, "ring-removal", "ring",
                               dt_prev))
    loops = pb.interface_loops()
    named, _ = _named_chains(pb)
    chains = dict(named)
    if len(loops) != 1 or set(chains) != {"open"}:
        raise CutRefused(f"the removal needs the open chain and one ring, not {sorted(chains)} "
                         f"and {len(loops)} loops")
    L = np.column_stack([loops[0]["r"], loops[0]["z"]])
    U = np.hypot(loops[0]["u_r"], loops[0]["u_z"])
    V_b = float(st["V_b"])
    V_ring = revolved_volume(L)
    limit = s.bubble_stop * V_b
    record["trigger"] = {"ring_volume": V_ring, "ratio": V_ring / V_b, "limit": limit, "V_b": V_b}
    record["ring"] = loop_record(L, np.column_stack([loops[0]["u_r"], loops[0]["u_z"]]))
    if not V_ring < limit:
        raise CutRefused(f"the ring holds {V_ring / V_b:.4g} of V_b, not below bubble_stop "
                         f"{s.bubble_stop:g}")
    record["dumps"] = {}

    stage("precut")
    event = {**record["request"], "k": k}
    record["dumps"]["precut"] = _keep(pb, driver, out, f"precut_e{k:03d}",
                                      {"cut_stage": PRECUT_EVENT, "event": event})["path"]

    stage("budgets")
    Q = closed(L)
    z_c = float(record["ring"]["z_c"])
    L_fill = float(np.max(np.hypot(Q[:, 0], Q[:, 1] - z_c)))
    R_b = max(ring.box_radius * delta, 2.0 * L_fill)
    grid, grid20 = box_grid(z_c, R_b), box_grid(z_c, LOCAL_RADIUS * delta)
    cavity = chains["open"]
    global_before = pb.cut_observables()
    box_before = box_integrals(pb, grid)
    box20_before = box_integrals(pb, grid20)
    ring_chain = Chain(Q, ("loop", "loop"))
    poly_before = box_polyline_budgets([cavity, ring_chain], z_c, R_b, s.reservoir_depth)
    U_ring = float(U.max()) if len(U) else 0.0
    scales = {"V_fill": V_ring, "U": U_ring, "P_ref": V_ring * U_ring,
              "KE_ref": 0.5 * V_ring * U_ring**2}
    h_ring = float(st["sizing"]["h_ring"])
    record["geometry"] = {"V_fill": V_ring, "z_c": z_c, "L_fill": L_fill,
                          "area": signed_area(L)}

    template = pb.get_mesh("liquid")._templatemesh
    old_focus = template.focus
    regions, grading = _focus_regions(old_focus, pb.mesh_settings.grading)
    kept = tuple(x for x in regions if not x.name.startswith("ring_")) or regions
    focus = Focus(size=min(x.size for x in kept), radius=max(x.radius for x in kept),
                  grading=grading, regions=kept)
    record["mesh"] = {"focus_before": None if old_focus is None else old_focus.to_dict(),
                      "dropped": [x.name for x in regions if x.name.startswith("ring_")]}

    def restore():
        template.focus = old_focus

    stage("park")
    printed = _rebuild(pb, template, focus, {"chains": [cavity], "loops": []},
                       {"axis_bulk": False, "reach": 0.0}, stage=stage,
                       precut=record["dumps"]["precut"], restore=restore)

    stage("mesh-check")
    transfer = transfer_warnings("".join(printed))
    record["transfer"] = transfer
    new_segs, new_loops = pb.interface_segments(), pb.interface_loops()
    record["mesh"].update(chains=len(new_segs), loops=len(new_loops),
                          nodes=pb.get_mesh("liquid").nnode())
    if len(new_segs) != 1 or not new_segs[0]["open"] or new_loops:
        raise CutFailed("the removal's mesh is not the open chain alone",
                        record["dumps"]["precut"])
    spans = _axis_spans_of_mesh(pb)
    expected = liquid_axis_spans([float(new_segs[0]["z"][0])])
    record["mesh"].update(axis_spans=spans, expected_axis_spans=[list(x) for x in expected])
    if len(spans) != 1 or any(abs(a - b) > 1e-9 * max(1.0, pb.nd.radius)
                              for a, b in zip(spans[0], expected[0])):
        raise CutFailed(f"the removal's mesh has the axis spans {spans}, expected {expected}",
                        record["dumps"]["precut"])
    sets = _node_sets(pb)
    predicted = predict_unlocated(sets["interface"][1], [cavity.points, Q])
    bulk_inside = _inside(sets["bulk"][1], L, 0.05 * h_ring)
    corners = _corners(pb, _axis_ends([cavity]), CORNER_TOL * float(pb.cut["sizing"]["h_tip"]))
    problems = _transfer_check(transfer, sets, predicted, None, bulk_inside,
                               np.zeros(len(sets["axis"][1]), dtype=bool), corners)
    if problems:
        raise CutFailed("the transfer of the ring's removal is not as planned: "
                        + "; ".join(problems), record["dumps"]["precut"])
    global_transfer = pb.cut_observables()
    open_new = Chain(np.column_stack([new_segs[0]["r"], new_segs[0]["z"]]),
                     tuple(new_segs[0]["end_types"]))
    open_new.points[0, 0] = 0.0
    poly_after = box_polyline_budgets([open_new], z_c, R_b, s.reservoir_depth)

    stage("fill")
    fill = {"mode": ring.fill, "zeroed_bulk": 0, "zeroed_axis": 0}
    if ring.fill == "zero":
        fill.update(_zero_inside(pb, sets, L, (math.inf, -math.inf), 0.0))
        pb.invalidate_cached_mesh_data()
    filled = box_integrals(pb, grid, {"fill": L})["fill"]
    fill.update(Pz_fill=filled["Pz"], KE_fill=filled["KE"], V_grid=filled["V_grid"])
    record["fill"] = fill

    stage("projection")
    record["projection"] = {}
    settings = replace(s, dt0=dt_prev, restart=ring.restart or s.restart,
                       projection=ring.projection)
    t_now = float(record["t"])
    project(pb, settings, t_now, record["projection"], box=(z_c, R_b))
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
        "box": {"z": z_c, "R": R_b, "n": grid["n"]},
        "polylines": {"before": poly_before, "after": poly_after},
        "grid": {"before": box_before, "after_transfer": box_transfer,
                 "after_projection": box_after},
        "box20": {"before": box20_before, "after": box_integrals(pb, grid20)},
        "global": {"before": global_before, "after_transfer": global_transfer,
                   "after": global_after, "includes_ring": True}}
    from .events import revolved_area

    closures = _ring_closures({"V": V_ring, "Pz": fill["Pz_fill"], "KE": fill["KE_fill"]},
                              record["budgets"]["polylines"],
                              {"before": box_before, "after_transfer": box_transfer,
                               "after_projection": box_after}, box, scales,
                              {"A_cl": 0.0, "A_rem": revolved_area(Q)},
                              sigma={"V": ring.sigma_V, "Pz": ring.sigma_Pz, "KE": ring.sigma_KE},
                              prefix="Q")
    closures["global_identity"] = {x: record["projection"].get(x, math.nan)
                                   for x in ("delta_Pz", "Pi_on_bottom")}
    record["closures"] = closures
    if closures["hard"]:
        missed = {x: closures[x] for x in closures["hard"]}
        raise BudgetUnclosed(f"the budget of the ring's removal does not close: "
                             f"{closures['hard']} beyond their limits ({missed})")
    record["filled"] = {"V": V_ring, "Pz": fill["Pz_fill"], "KE": fill["KE_fill"]}
    record["after_removal"] = ring.after_removal
    removed = {"k": k, "t": t_now, "tau": record["tau"], "V": V_ring, "V_over_V_b": V_ring / V_b,
               "Pz_fill": fill["Pz_fill"], "KE_fill": fill["KE_fill"]}
    pb.ring = _jsonable({**st, "removed": removed})
    record["outcome"] = "made"
    record["wall_seconds"] = time.time() - wall0
    made = _jsonable(record)
    pb.events = list(getattr(pb, "events", None) or []) + [made]
    if on_made is not None:
        on_made(made)
    made["dumps"]["postcut"] = _keep(pb, driver, out, f"postcut_e{k:03d}",
                                     {"cut_stage": "postcut-event", "event_k": k})["path"]
    pb.events[-1] = made
    full = dict(made, ring_polygon=L.tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(full), indent=1) + "\n")
    log(f"ring removed (event {k}) at V/V_b {V_ring / V_b:.4g}: filled {V_ring:.6g}, "
        f"closures " + ", ".join(f"{x} {closures['relative'][x]:.3g}"
                                 for x in closures["relative"]))
    return made
