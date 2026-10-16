"""The cut: reconnection of the free surface through a closing neck, on the finite-element problem.

The sequence (:func:`cut`), for a :class:`~faraday_singularity.problem.FaradayProblem` made with
``reconnect`` settings, at a state whose neck radius ``r_n`` is a little below the cut radius
``delta``:

1. **Record** the budgets (volume, kinetic energy, axial momentum ``P_z``, ``int (div u)^2``) and keep
   a ``precut`` state dump.
2. **Plan.** pyoomph's ``AxisymmetricReconnection`` runs its planner once, at ``rmin = delta``, with
   the cavity closed through the gas above the free surface (``reservoir_depth``) so that the neck is
   a waist of the gas and the event a pinch; its volume correction is off. It only parks a plan on
   the mesh template. A deferral (waist not yet separable), a declined state or a refusal (a plain
   ``RuntimeError``) is retried at a cut radius larger by ``retry_factor``, ``plan_attempts`` times.
3. **Caps.** Both planner caps are replaced by tangent arcs (``reconnect.replace_caps``), and the
   template is given the sizes of the tips and of the bridge (``mesh.CutSizing``).
4. **Remesh at once**, with nothing in between that could solve, move a node or remesh: the plan
   refers to the current interface nodes exactly. pyoomph transfers the fields; nodes in the former
   gas thread and on the bridge are not located in the old mesh and get its two-nearest-node blend.
   Any failure clears the plan; the caller restores the ``precut`` state.
5. **Fill**: keep the blend, or zero the velocity at bulk nodes inside the former gas.
6. **History := transferred field** (``assign_initial_values_impulsive``), so that level 1 holds it.
7. **Project**: find ``u+`` (P2) and the pressure impulse ``Pi`` (P1) with
   ``int (u+ - u~).v - Pi div v = 0`` and ``int q div u+ = 0`` for all admissible ``v`` and all ``q``;
   ``Pi = 0`` holds weakly on every free surface (the bubble keeps the passive gas pressure). This is
   an auxiliary residual, solved as one *transient* step that does not shift the history: in a
   stationary solve pyoomph replaces the history value ``u~`` by the current one and the system is
   singular. ``u+`` is the L2 projection of ``u~`` onto discretely divergence-free fields with the
   (homogeneous) pinned values, so ``KE(u~) - KE(u+) = |u+ - u~|^2/2`` holds to round-off; the cut
   is refused unless it does, to ``energy_tolerance``.
8. **History := u+**, the time and the step sizes put back (``dt0`` for every level), and the step
   counter set for a BDF1 first step (``restart = "bdf1"``) or a BDF2 step on the flat history.
9. The interface chart is oriented from a far point so that both chains keep their orientation
   at later remeshes; the driver's bookkeeping restarts for the new tips; a ``postcut`` dump and
   ``cut_diagnostics.json`` are written.

Only private names of the pinned pyoomph release make step 2 possible; ``testCases`` pins them.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from pyoomph.equations.topological_changes import AxisymmetricReconnection
from pyoomph.expressions import div, evaluate_in_past, testfunction, var, weak
from pyoomph.generic.codegen import Equations

from .mesh import CutSizing
from .reconnect import (ReconnectSettings, Chain, chain_volumes, cut_geometry, liquid_axis_spans,
                        neck_radius, plan_chains, replace_caps)

INTERFACE = "liquid/interface"
FAR_POINT = (1.0e4, -100.0)  # the chart starts at the chain end nearest this: the wall, else the bottom


class CutRefused(RuntimeError):
    """No cut was made; the problem is unchanged."""


class CutFailed(RuntimeError):
    """The cut failed after the problem was changed; restore the ``precut`` state."""


class DriverTriggeredReconnection(AxisymmetricReconnection):
    """pyoomph's reconnection handler, inert until the driver asks for one pinch.

    With ``rmin`` and ``distmin`` both ``None`` the handler's own hook after every Newton solve only
    drops a stale plan. :meth:`request_pinch` runs that hook once at a given cut radius, inside the
    interface's code-generator scope, which the hook needs. Volume correction and the check of the
    mesh motion are off; the cavity chain is closed through the gas above the free surface.
    """

    def __init__(self, *, reservoir_depth: float, cap_window_factor: float):
        super().__init__(rmin=None, distmin=None, volume_conservation=False,
                         check_mesh_motion_direction=False, reservoir_depth=reservoir_depth,
                         cap_window_factor=cap_window_factor)

    def request_pinch(self, problem, rmin: float) -> str:
        """Plan a pinch at cut radius ``rmin``: ``"plan"`` (parked on the template), ``"none"``
        (no neck below ``rmin``), ``"deferral"`` or ``"unplannable"`` (the planner's two retryable
        refusals, which the handler swallows and counts). Other refusals raise ``RuntimeError``."""
        tree = problem.get_mesh(INTERFACE).get_eqtree()
        if not any(e is self for e in tree._equations):
            raise RuntimeError("the reconnection handler is not an equation of the free surface")
        template = problem.get_mesh("liquid")._templatemesh
        deferred, unplannable = self._deferred_events, self._unplannable_states
        self.rmin = rmin
        self._nondim_done = False
        try:
            with tree._on_this_domain():
                AxisymmetricReconnection.after_newton_solve(self)
        finally:
            self.rmin = None
            self._nondim_done = False
        if template.has_pending_surgery_plan(INTERFACE):
            return "plan"
        if self._deferred_events > deferred:
            return "deferral"
        if self._unplannable_states > unplannable:
            return "unplannable"
        return "none"


class PressureImpulseProjection(Equations):
    """``int (u - u~).v - Pi div v - q div u`` with ``u~`` the velocity at history level 1, as the
    auxiliary residual ``projection``; ``Pi`` is held in the pressure field."""

    DESTINATION = "projection"

    def define_residuals(self):
        u, v = var("velocity"), testfunction("velocity")
        Pi, q = var("pressure"), testfunction("pressure")
        u_old = evaluate_in_past(var("velocity"), 1)
        self.add_residual(weak(u - u_old, v) - weak(Pi, div(v)) - weak(div(u), q),
                          destination=self.DESTINATION)


# --- pieces of the sequence ---------------------------------------------------------------------


def clear_plan(problem) -> None:
    """Drop a parked plan and the remesh request that came with it."""
    template = problem.get_mesh("liquid")._templatemesh
    template._clear_pending_surgery_plan(INTERFACE)
    if problem.recon is not None:
        problem.recon._pending_on = None
    problem._domains_to_remesh.clear()


@contextlib.contextmanager
def captured_stdout():
    """Capture what C, C++ and Python write to the process's standard output, and write it out
    again afterwards. pyoomph reports the nodes its transfer could not locate there, not in its
    log file. Yields a list that holds the captured text once the block has ended."""
    out: list[str] = []
    libc = ctypes.CDLL(None)
    sys.stdout.flush()
    libc.fflush(None)
    saved = os.dup(1)
    with tempfile.TemporaryFile() as tmp:
        os.dup2(tmp.fileno(), 1)
        try:
            yield out
        finally:
            sys.stdout.flush()
            libc.fflush(None)
            os.dup2(saved, 1)
            os.close(saved)
            tmp.seek(0)
            data = tmp.read()
            out.append(data.decode(errors="replace"))
            view = memoryview(data)
            while view:  # straight to the descriptor: pyoomph wraps sys.stdout
                view = view[os.write(1, view):]


UNLOCATED = re.compile(r"interpolating (\S+): (\d+) of (\d+) node\(s\) could not be located")
NO_VALUE = re.compile(r"interpolating '([^']+)': (\d+) of (\d+) nodes received NO value")


def transfer_warnings(text: str) -> dict:
    """Counts of nodes pyoomph could not locate in the old mesh (blended from the two nearest old
    nodes instead), and of codimension-2 nodes that received no value, per mesh."""
    out = {"unlocated": {}, "no_value": {}}
    for m in UNLOCATED.finditer(text):
        out["unlocated"][m.group(1)] = [int(m.group(2)), int(m.group(3))]
    for m in NO_VALUE.finditer(text):
        out["no_value"][m.group(1)] = [int(m.group(2)), int(m.group(3))]
    return out


def _axis_spans_of_mesh(problem) -> list[list[float]]:
    """Liquid spans of the axis as built: the z ranges of connected runs of axis nodes."""
    edges = []
    for e in problem.get_mesh("liquid/axis").elements():
        z = sorted(e.node_pt(k).x(1) for k in range(e.nnode()))
        edges.append((z[0], z[-1]))
    edges.sort()
    spans: list[list[float]] = []
    for a, b in edges:
        if spans and a <= spans[-1][1] + 1e-14:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    return spans


def mesh_nodes(mesh) -> list:
    """The nodes of a mesh; for an interface mesh, those of its elements, each once."""
    if mesh.nnode() > 0:
        return list(mesh.nodes())
    seen, out = set(), []
    for e in mesh.elements():
        for k in range(e.nnode()):
            n = e.node_pt(k)
            key = (n.x(0), n.x(1))
            if key not in seen:
                seen.add(key)
                out.append(n)
    return out


def _gas_polygon(old_open: Chain, reservoir_depth: float):
    from shapely.geometry import Polygon

    P = old_open.points
    return Polygon(np.vstack([P, [[P[-1, 0], P[-1, 1] + reservoir_depth],
                                  [0.0, P[-1, 1] + reservoir_depth]]]))


def former_gas_nodes(problem, old_open: Chain, reservoir_depth: float, tol: float) -> dict:
    """``[count, total]`` of nodes of the new mesh that lie in the old gas, farther than ``tol`` from
    the old free surface: interior bulk nodes, free-surface nodes and axis nodes (``z`` above the
    old tip). These are the nodes the transfer cannot locate in the old liquid: the former gas
    thread, the fresh caps and the bridge. (Without ``tol`` the sag of the rebuilt spline between
    old nodes would count half the free surface.)"""
    import shapely
    from shapely.geometry import LineString

    gas = _gas_polygon(old_open, reservoir_depth)
    core = gas.buffer(-tol)
    line = LineString(old_open.points)
    iface = mesh_nodes(problem.get_mesh(INTERFACE))
    axis = mesh_nodes(problem.get_mesh("liquid/axis"))
    on_boundary = {(n.x(0), n.x(1)) for n in iface + axis}
    bulk = [n for n in problem.get_mesh("liquid").nodes() if (n.x(0), n.x(1)) not in on_boundary]

    def count(nodes, test):
        return [int(sum(1 for n in nodes if test(n.x(0), n.x(1)))), len(nodes)]

    z_tip = float(old_open.points[0, 1])
    return {"liquid": count(bulk, lambda x, y: bool(shapely.contains_xy(core, x, y))),
            INTERFACE: count(iface, lambda x, y: bool(shapely.contains_xy(gas, x, y))
                             and line.distance(shapely.Point(x, y)) > tol),
            "liquid/axis": count(axis, lambda x, y: y > z_tip + tol),
            "tolerance": tol}


def zero_fill(problem, old_open: Chain, reservoir_depth: float, centre: tuple[float, float],
              reach: float) -> int:
    """Zero the velocity (current level) at bulk nodes inside the former gas region near the cut.

    The former gas is the half-section the pre-cut open chain bounds, closed up the wall and across
    to the axis; only nodes within ``reach`` of ``centre`` are considered, and interface nodes are
    left alone."""
    import shapely

    gas = _gas_polygon(old_open, reservoir_depth)
    mesh = problem.get_mesh("liquid")
    fi = mesh.get_nodal_field_indices()
    iu, iw = fi["velocity_x"], fi["velocity_y"]
    surf = {(n.x(0), n.x(1)) for n in mesh_nodes(problem.get_mesh(INTERFACE))}
    cand = [n for n in mesh.nodes()
            if math.hypot(n.x(0) - centre[0], n.x(1) - centre[1]) <= reach
            and (n.x(0), n.x(1)) not in surf]
    if not cand:
        return 0
    xy = np.array([[n.x(0), n.x(1)] for n in cand])
    inside = shapely.contains_xy(gas, xy[:, 0], xy[:, 1])
    count = 0
    for n, flag in zip(cand, inside):
        if flag:
            n.set_value(iu, 0.0)
            n.set_value(iw, 0.0)
            count += 1
    return count


def _pressures(problem):
    mesh = problem.get_mesh("liquid")
    ip = mesh.get_nodal_field_indices()["pressure"]
    return ip, [(n, n.value(ip)) for n in mesh.nodes() if n.nvalue() > ip]


def _pressure_impulse_stats(problem) -> dict:
    _, vals = _pressures(problem)
    pi = np.array([v for _, v in vals])
    return {"max_abs_Pi": float(np.max(np.abs(pi))) if len(pi) else math.nan}


def project(problem, settings: ReconnectSettings, t_cut: float) -> dict:
    """Steps 6-8: impulsive history, the projection, the restart of the history at ``t_cut``.

    Returns the projection diagnostics. Raises :class:`CutFailed` if the projection solve did not
    converge, remeshed, or broke the energy identity."""
    s = settings
    pb = problem
    before = pb.cut_observables()
    pb.assign_initial_values_impulsive()  # levels 1, 2 := the transferred (and filled) field
    out: dict = {"before": before}
    if s.projection:
        ip, saved = _pressures(pb)
        pb._projecting = True
        try:
            pb.solve_auxiliary_residual(PressureImpulseProjection.DESTINATION, timestep=s.dt0,
                                        shift_values=False, do_not_set_IC=True,
                                        newton_solver_tolerance=s.projection_tolerance)
        finally:
            pb._projecting = False
        residuals = [float(x) for x in pb.get_last_residual_convergence()]
        if len(pb._domains_to_remesh) or getattr(pb, "_remesh_requested_inside_solve", False):
            raise CutFailed("a remesh was requested during the projection")
        after = pb.cut_observables()  # level 1 still holds u~: projection_change = |u+ - u~|^2/2
        dke = before["kinetic_energy"] - after["kinetic_energy"]
        identity = dke - after["projection_change"]
        out.update(after=after, newton_residuals=residuals, delta_KE=dke,
                   energy_identity_error=identity,
                   delta_Pz=after["momentum_z"] - before["momentum_z"],
                   Pi_rms_on_surface=math.sqrt(max(after["pressure_sq"], 0.0) / after["area"]),
                   divergence_L2_before=math.sqrt(max(before["divergence_sq"], 0.0)),
                   divergence_L2_after=math.sqrt(max(after["divergence_sq"], 0.0)),
                   **_pressure_impulse_stats(pb))
        if not residuals or residuals[-1] > s.projection_tolerance:
            raise CutFailed(f"the projection did not converge: residuals {residuals}")
        if dke < -s.energy_tolerance * before["kinetic_energy"] or \
                abs(identity) > s.energy_tolerance * before["kinetic_energy"]:
            raise CutFailed(f"the projection broke the energy identity: delta KE {dke!r}, "
                            f"|u+ - u~|^2/2 {after['projection_change']!r}")
        if s.restore_pressure:  # the pressure field now holds Pi; give Newton the old guess back
            for n, v in saved:
                n.set_value(ip, v)
    pb.set_current_time(t_cut, dimensional=False)
    pb.assign_initial_values_impulsive(s.dt0)  # every level := u+, every step size := dt0
    pb.timestepper.set_num_unsteady_steps_done(0 if s.restart == "bdf1" else 1)
    out["restart"] = s.restart
    return out


def orient_interface_chart(problem) -> None:
    """Chart each chain from its end nearest a far point below the wall: the open chain from the
    wall, a chain from axis to axis from its lower end. ``x+`` cannot orient the latter."""
    z = problem.interface_zeta
    tree = problem.get_mesh(INTERFACE).get_eqtree()
    if not any(e is z for e in tree._equations):
        raise RuntimeError("the arclength chart is not an equation of the free surface")
    z.sort_along_axis = None
    z.start_near_point = FAR_POINT
    with tree._on_this_domain():
        z.assign_zetas(problem.get_mesh(INTERFACE))


# --- the whole cut ------------------------------------------------------------------------------


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def cut(problem, settings: ReconnectSettings, outdir: Path, *, driver=None, tag: str | None = None,
        log=print) -> dict:
    """Cut the neck of the current state at ``settings.delta`` and project (module docstring).

    ``driver``, if given, keeps the ``precut``/``postcut`` dumps with its bookkeeping and restarts
    the bookkeeping for the new tips; without it the dumps hold the pyoomph state only. Returns the
    cut record, also written to ``cut_diagnostics.json``. Raises :class:`CutRefused` when no cut
    was made, :class:`CutFailed` when the problem must be restored from the ``precut`` dump."""
    s = settings
    pb = problem
    out = Path(outdir)
    if pb.recon is None:
        raise CutRefused("the problem was not made with reconnect settings")
    if pb.cut is not None:
        raise CutRefused("only one cut per run")
    wall0 = time.time()
    t_cut = pb.get_current_time(dimensional=False, as_float=True)
    tag = tag or f"{s.delta:.6g}"
    chains_before = pb.interface_segments()
    if len(chains_before) != 1:
        raise CutRefused(f"the free surface has {len(chains_before)} chains, not one")
    old = chains_before[0]
    old_open = Chain(np.column_stack([old["r"], old["z"]]), tuple(old["end_types"]))
    r_n, z_n, _ = neck_radius(old["r"], old["z"])
    if not math.isfinite(r_n):
        raise CutRefused("the free surface has no neck")
    if s.delta < s.min_landing_ratio * r_n:
        raise CutRefused(f"delta {s.delta:.6g} is below {s.min_landing_ratio} times the neck "
                         f"radius {r_n:.6g}")
    record: dict = {"delta": s.delta, "t_cut": t_cut, "r_neck": r_n, "z_neck": z_n,
                    "delta_over_r_neck": s.delta / r_n, "settings": s.to_dict()}
    record["budgets_before"] = pb.cut_observables()
    record["precut_state"] = _keep(pb, driver, out, f"precut_{tag}", {"cut_stage": "precut"})

    # 2. plan
    attempts = []
    rmin = s.delta
    template = pb.get_mesh("liquid")._templatemesh
    for _ in range(s.plan_attempts):
        try:
            outcome, message = pb.recon.request_pinch(pb, rmin), ""
        except RuntimeError as e:
            clear_plan(pb)
            outcome, message = "refused", str(e)
        attempts.append({"rmin": rmin, "outcome": outcome, "message": message})
        if outcome == "plan":
            break
        if outcome == "none":
            raise CutRefused(f"no pinch planned at cut radius {rmin:.6g} (neck {r_n:.6g})")
        rmin *= s.retry_factor
    record["plan_attempts"] = attempts
    if attempts[-1]["outcome"] != "plan":
        raise CutRefused(f"no plan after {len(attempts)} attempts: {attempts[-1]}")
    record["rmin"] = rmin

    try:
        plan, _extra = template._pending_surgery_plans[INTERFACE]
        gas_before = float(sum(plan.fragment_volumes_before))
        record["plan"] = {"events": [{"kind": e.kind, "z_center": e.z_center} for e in plan.events],
                          "gas_before": gas_before,
                          "planner_gas_after": [float(v) for v in plan.fragment_volumes_after],
                          "fresh_points": [int(np.sum(np.asarray(c.origin) < 0))
                                           for c in plan.new_chains]}
        # 3. caps, and the sizes of the new mesh
        edits = replace_caps(plan.new_chains, rmin, s)
        chains = plan_chains(plan.new_chains)
        geom = cut_geometry(chains, s.reservoir_depth, gas_before)
        # the plan's gas spans and volumes, for the edited chains (informational in pyoomph)
        plan.axis_spans_inside = np.array(
            [[c.points[0, 1], c.points[-1, 1] + (s.reservoir_depth if c.is_open else 0.0)]
             for c in chains])
        plan.fragment_volumes_after = chain_volumes(chains, s.reservoir_depth)
        h_tip = max(min(e.radius for e in edits) / s.tip_elements, s.tip_floor * rmin)
        sizing = CutSizing(h_tip=h_tip, h_bridge=min(h_tip, geom.bridge / s.bridge_elements),
                           curvature=s.curvature_sizing,
                           grading=pb.active_mesh_template().settings.grading)
        record["caps"] = [asdict(e) for e in edits]
        record["geometry"] = geom.to_dict()
        record["sizing"] = asdict(sizing)
        template.cut_sizing = sizing
        # 4. the remesh, at once
        with captured_stdout() as printed:
            pb.force_remesh({template})
        pb._domains_to_remesh.clear()
    except BaseException as e:
        clear_plan(pb)
        template.cut_sizing = None
        raise CutFailed(f"the cut remesh failed ({type(e).__name__}: {e}); restore "
                        f"{record['precut_state']['path']}") from e
    record["transfer"] = transfer_warnings("".join(printed))
    record["transfer"]["former_gas_nodes"] = former_gas_nodes(pb, old_open, s.reservoir_depth,
                                                              0.05 * rmin)
    # the fresh tips on the axis get no transferred value of the interface's only own field, the
    # kinematic Lagrange multiplier _kin_bc; it has no time history, so this is Newton's start only
    record["transfer"]["corner_fields"] = sorted(
        pb.get_mesh(INTERFACE + "/axis").get_nodal_field_indices())

    try:
        segs = pb.interface_segments()
        if len(segs) != 2 or sum(c["open"] for c in segs) != 1:
            raise CutFailed(f"the cut mesh has {len(segs)} interface chains")
        spans = _axis_spans_of_mesh(pb)
        tips = [float(c["z"][k]) for c in segs for k in (0, -1)
                if c["end_types"][0 if k == 0 else 1] == "axis"]
        expected = liquid_axis_spans(tips)
        record["mesh"] = {"chains": [{"open": c["open"], "nodes": int(len(c["r"])),
                                      "z_ends": [float(c["z"][0]), float(c["z"][-1])]} for c in segs],
                          "axis_spans": spans, "expected_axis_spans": [list(x) for x in expected]}
        if len(spans) != len(expected):
            raise CutFailed(f"the cut mesh has axis spans {spans}, expected {expected}")
        record["budgets_after_transfer"] = pb.cut_observables()
        # 5. fill
        record["fill"] = {"kind": s.fill, "zeroed_nodes": 0}
        if s.fill == "zero":
            reach = 20.0 * max(geom.bridge, rmin)
            record["fill"]["zeroed_nodes"] = zero_fill(pb, old_open, s.reservoir_depth, (0.0, z_n),
                                                       reach)
        record["budgets_after_fill"] = pb.cut_observables()
        # 6-8. projection and restart
        record["projection"] = project(pb, s, t_cut)
        orient_interface_chart(pb)
    except CutFailed:
        raise
    except BaseException as e:
        raise CutFailed(f"after the cut remesh: {type(e).__name__}: {e}; restore "
                        f"{record['precut_state']['path']}") from e
    record["budgets_after"] = pb.cut_observables()
    volumes = {"liquid_before": record["budgets_before"]["volume"],
               "liquid_after": record["budgets_after"]["volume"],
               "liquid_gain_plan": geom.removed_gas}
    volumes["liquid_gain_mesh"] = volumes["liquid_after"] - volumes["liquid_before"]
    record["volumes"] = volumes
    record["wall_seconds"] = time.time() - wall0
    pb.cut = _jsonable(record)
    if driver is not None:
        driver.reset_after_cut(pb.cut)
    record["postcut_state"] = _keep(pb, driver, out, f"postcut_{tag}", {"cut_stage": "postcut"})
    pb.cut = _jsonable(record)
    (out / "cut_diagnostics.json").write_text(json.dumps(pb.cut, indent=1) + "\n")
    log(f"cut at delta {s.delta:.6g} (rmin {rmin:.6g}, r_n {r_n:.6g}): bridge {geom.bridge:.4g}, "
        f"removed gas {geom.removed_gas:.4g}, delta KE {record['projection'].get('delta_KE')}")
    return pb.cut


def _keep(problem, driver, outdir: Path, name: str, extra: dict) -> dict:
    """A kept state dump under ``states/``, outside the driver's rotation, with its JSON."""
    path = Path(outdir) / "states" / f"{name}.dump"
    if driver is not None:
        driver.keep_state(path, **extra)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        problem.save_state(str(path), quiet=True)
        meta = {"time": problem.get_current_time(dimensional=False, as_float=True), **extra}
        if problem.cut is not None:
            meta["cut"] = problem.cut
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    return {"path": str(path), "time": problem.get_current_time(dimensional=False, as_float=True)}


def restore_cut(problem, record: dict) -> None:
    """After loading a state written after a cut: the record, and the chart orientation."""
    problem.cut = record
    orient_interface_chart(problem)
