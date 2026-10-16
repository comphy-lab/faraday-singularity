"""The cut: reconnection of the free surface through a closing neck, on the finite-element problem.

The sequence (:func:`cut`), for a :class:`~faraday_singularity.problem.FaradayProblem` made with
``reconnect`` settings, at a state whose neck radius ``r_n`` is a little below the cut radius
``delta`` (``min_landing_ratio <= delta/r_n <= max_landing_ratio``, otherwise the cut is refused):

1. **Record** the budgets (volume, kinetic energy, axial momentum ``P_z``, ``int (div u)^2``), the
   kinetic energy within ``LOCAL_RADIUS delta`` of the neck (:func:`local_kinetic_energy`, again
   after the transfer, the fill and the projection) and the pinch time ``t_p`` of the neck, fitted
   by the driver to its latest neck samples (``pinch``: the fit, its window, and ``t_cut - t_p``),
   and keep a ``precut`` state dump, which records the neck (``neck``) as a landing state does, so
   that a new run can cut it again. Times after the cut are compared at fixed ``t - t_p``, since
   ``t_cut`` depends on ``delta``.
2. **Plan.** pyoomph's ``AxisymmetricReconnection`` runs its planner once, at ``rmin = delta``, with
   the cavity closed through the gas above the free surface (``reservoir_depth``) so that the neck is
   a waist of the gas and the event a pinch; its volume correction is off. It only parks a plan on
   the mesh template. A deferral (waist not yet separable), a declined state or a refusal (a plain
   ``RuntimeError``) is retried at a cut radius larger by ``retry_factor``, ``plan_attempts`` times,
   but never above ``max_landing_ratio r_n``.
3. **Caps.** Both planner caps are replaced by tangent arcs (``reconnect.replace_caps``). The mesh
   of the cut is sized in cut radii, so that the cuts of a study in ``delta`` are meshed alike: the
   template is given the element sizes at the tips on the axis and along the bridge
   (``mesh.CutSizing``: ``tip_size`` and ``bridge_size`` times the cut radius, kept for every later
   remesh), and the focus of this remesh is set to ``focus_size`` cut radii on the interface nodes
   within ``focus_radius`` cut radii of the axis. That focus replaces the one the run that wrote the
   landing state left, whose size depends on when that run last rebuilt its mesh and on its
   ``h_floor``; both are recorded (``focus``). The collapse driver sets the focus again by its own
   rule at its next rebuild.
4. **Remesh at once**, with nothing in between that could solve, move a node or remesh: the plan
   refers to the current interface nodes exactly. pyoomph transfers the fields; nodes in the former
   gas thread and on the bridge are not located in the old mesh and get its two-nearest-node blend.
   Any failure clears the plan. A failure from the remesh on can leave the problem half rebuilt, so
   the ``precut`` state is never loaded back into it: the run ends, and a new process starts from
   the ``precut`` dump if wanted (a restart that cuts it again).
5. **Fill**: keep the blend, or zero the velocity at the nodes inside the former gas, the nodes on
   the axis of the bridge included. Nodes on the free surface are never zeroed: the fresh caps take
   the velocity of the nearest point of the old neck surface (pyoomph's transfer projects them onto
   it), and zeroing them would put a jump in the velocity at the splice.
6. **History := transferred field** (``assign_initial_values_impulsive``), so that level 1 holds it.
7. **Project**: find ``u+`` (P2) and the pressure impulse ``Pi`` (P1) with
   ``int (u+ - u~).v - Pi div v = 0`` and ``int q div u+ = 0`` for all admissible ``v`` and all ``q``;
   ``Pi = 0`` holds weakly on every free surface (the bubble keeps the passive gas pressure). This is
   an auxiliary residual, solved as one *transient* step that does not shift the history: in a
   stationary solve pyoomph replaces the history value ``u~`` by the current one and the system is
   singular. ``u+`` is the L2 projection of ``u~`` onto discretely divergence-free fields with the
   (homogeneous) pinned values, so ``KE(u~) - KE(u+) = |u+ - u~|^2/2`` holds to round-off; the cut
   is refused unless it does, to ``energy_tolerance``, and unless the solve converged to
   ``projection_tolerance``. That residual includes ``int q div u+`` for every pressure test
   function: ``u+`` is divergence-free weakly, against the P1 pressures, and these two tests are
   the acceptance of the projection. The pointwise divergence of the P2 velocity is not zero: its
   L2 norm (``strong_divergence_L2_after``, and the per-step ``divergence_sq``, its square) is a
   discretisation diagnostic of order ``h``. On the manufactured field of the tests it is 4.2, 2.3
   and 1.3 on meshes of 8, 16 and 32 elements per side (an observed order of 0.8-0.9), while the
   residual stays near 5e-15.
8. **History := u+**, the time and the step sizes put back (``dt0`` for every level), and the step
   counter set for a BDF1 first step (``restart = "bdf1"``) or a BDF2 step on the flat history.
9. The interface chart is oriented from a far point so that both chains keep their orientation
   at later remeshes; the driver's bookkeeping restarts for the new tips; a ``postcut`` dump and
   ``cut_diagnostics.json`` are written.

A refused or failed cut also writes ``cut_diagnostics.json``: the record as far as it got (plan
attempts, caps, geometry, transfer counts, axis spans, budgets, projection residuals and the energy
identity), with ``outcome`` ``refused`` or ``failed``, the stage reached (``failed_at``) and the
error. Every failure of the cut is one of the two: any other exception is raised as a refusal
before the remesh and as a failure from it on, the bookkeeping after the cut and the ``postcut``
dump included (:func:`cut`). An interruption (``KeyboardInterrupt``, ``SystemExit`` or any other
exception that is not an ``Exception``) is no outcome of the cut: it propagates unchanged, after
the plan is dropped and the partial record is written with ``outcome`` ``interrupted``. The record
also gives the cut geometry in cut radii (``in_cut_radii``: the splice radii, arc radii, bridge and
removed gas, the number of survivors each cap dropped, and for each cap whether the planner's
window or the floor at the cut radius set its splice; see ``reconnect.py``).

A state written after the cut carries its record; :func:`restore_cut` puts the cut's mesh sizing
back when such a state is loaded. The production runner (``simulationCases/run_faraday.py``)
makes the cut at the start of a restart from a landing state under a case that adds the
``collapse.reconnect`` block, continues under the collapse driver in its post-cut mode (the stops
on both chains, the step rule of both chains), and continues a run from any state written after
the cut; the prototype (``simulationCases/reconnect_prototype.py``) refuses such a state.

The volume the liquid gains (``volumes``) is recorded twice: over the whole mesh, which includes the
drift of rebuilding the whole free surface (a remesh changes the volume by more than the cut at
production radii), and as the change of the gas volume within a box around the cut
(``reconnect.gas_volume_in_box``), which is the cut's own and is compared with the plan.

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
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
from pyoomph.equations.topological_changes import AxisymmetricReconnection
from pyoomph.expressions import div, evaluate_in_past, testfunction, var, weak
from pyoomph.generic.codegen import Equations

from .mesh import CutSizing, Focus
from .reconnect import (Chain, ReconnectSettings, chain_volumes, cut_geometry, gas_volume_in_box,
                        in_cut_radii, liquid_axis_spans, neck_radius, orient_chain, plan_chains,
                        replace_caps, tip_size)

INTERFACE = "liquid/interface"
FAR_POINT = (1.0e4, -100.0)  # the chart starts at the chain end nearest this: the wall, else the bottom
AXIS_TOL = 1e-9  # a node with |r| below this lies on the axis (the template's point merge tolerance)


class CutRefused(RuntimeError):
    """No cut was made; the problem is unchanged. ``record``: the partial cut record, with the
    stage reached (``failed_at``), as written to ``cut_diagnostics.json``; ``precut``: the path of
    the ``precut`` dump once it is written (refusals from the plan on), else None."""

    def __init__(self, message: str, record: dict | None = None, precut: str | None = None):
        super().__init__(message)
        self.record = record
        self.precut = precut


class CutFailed(RuntimeError):
    """The cut failed after the problem was changed, which may be half rebuilt: end the run and do
    not load anything back into it. The ``precut`` dump (``precut``) holds the state before the
    cut, for a new process: a landing state, which a restart under a case with a reconnect block
    cuts again (``run_case.restart_admission``). ``record`` as for :class:`CutRefused`."""

    def __init__(self, message: str, precut: str | None = None, record: dict | None = None):
        super().__init__(message)
        self.precut = precut
        self.record = record


# The stages of a cut before it changes the problem: any failure up to them is a refusal.
UNCHANGED_STAGES = ("admission", "precut", "plan", "caps")


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

    def xy(nodes):
        return np.array([[n.x(0), n.x(1)] for n in nodes]).reshape(-1, 2)

    B, S, A = xy(bulk), xy(iface), xy(axis)
    z_tip = float(old_open.points[0, 1])
    in_bulk = shapely.contains_xy(core, B[:, 0], B[:, 1]) if len(B) else np.zeros(0, bool)
    in_surf = np.zeros(len(S), bool)
    if len(S):
        in_surf = shapely.contains_xy(gas, S[:, 0], S[:, 1])
        idx = np.flatnonzero(in_surf)
        if len(idx):
            far = shapely.distance(line, shapely.points(S[idx])) > tol
            in_surf[idx] = far
    return {"liquid": [int(in_bulk.sum()), len(B)], INTERFACE: [int(in_surf.sum()), len(S)],
            "liquid/axis": [int(np.sum(A[:, 1] > z_tip + tol)), len(A)], "tolerance": tol}


def zero_fill(problem, old_open: Chain, reservoir_depth: float, centre: tuple[float, float],
              reach: float) -> dict:
    """Zero the velocity (current level) at the nodes inside the former gas region near the cut.

    The former gas is the half-section the pre-cut open chain bounds, closed up the wall, across
    and down the axis to the old tip. The nodes of the bridge lie on that closing segment of the
    axis, where a point-in-polygon test is false: they are the nodes on the axis above the old tip.
    Only nodes within ``reach`` of ``centre`` are considered, and interface nodes are left alone.
    Returns the counts of zeroed nodes off the axis (``bulk``) and on it (``axis``)."""
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
        return {"bulk": 0, "axis": 0}
    xy = np.array([[n.x(0), n.x(1)] for n in cand])
    z_tip = float(old_open.points[0, 1])
    on_axis = (np.abs(xy[:, 0]) <= AXIS_TOL) & (xy[:, 1] > z_tip + AXIS_TOL)
    inside = shapely.contains_xy(gas, xy[:, 0], xy[:, 1]) | on_axis
    for n, flag in zip(cand, inside):
        if flag:
            n.set_value(iu, 0.0)
            n.set_value(iw, 0.0)
    return {"bulk": int(np.sum(inside & ~on_axis)), "axis": int(np.sum(on_axis))}


def _pressures(problem):
    mesh = problem.get_mesh("liquid")
    ip = mesh.get_nodal_field_indices()["pressure"]
    return ip, [(n, n.value(ip)) for n in mesh.nodes() if n.nvalue() > ip]


def _pressure_impulse_stats(problem) -> dict:
    _, vals = _pressures(problem)
    pi = np.array([v for _, v in vals])
    return {"max_abs_Pi": float(np.max(np.abs(pi))) if len(pi) else math.nan}


LOCAL_RADIUS = 20.0  # the local budgets of a cut are taken within this many cut radii of the neck
LOCAL_SAMPLES = 128  # points per direction of the polar grid of the local kinetic energy


def local_kinetic_energy(problem, z_centre: float, radius: float, n: int = LOCAL_SAMPLES) -> float:
    """Kinetic energy of the liquid within ``radius`` of the point ``(0, z_centre)`` of the axis.

    The midpoint rule on an ``n x n`` polar grid over the half-disc, with the fields at the points
    pyoomph locates in the liquid mesh and the ``2 pi r`` measure; points outside the liquid (in the
    gas, below the bottom) count zero. A diagnostic, not an exact integral: its error comes from the
    grid near the free surface (on the neck of the FEM tests, 128 and 256 points per direction agree
    to about 1e-4)."""
    bulk = problem.get_mesh("liquid")
    fi = bulk.get_nodal_field_indices()
    iu, iw = 1 + fi["velocity_x"], 1 + fi["velocity_y"]
    h_rho, h_th = radius / n, math.pi / n
    rho = (np.arange(n) + 0.5) * h_rho
    theta = -0.5 * math.pi + (np.arange(n) + 0.5) * h_th
    P, T = np.meshgrid(rho, theta, indexing="ij")
    r, z = (P * np.cos(T)).ravel(), (z_centre + P * np.sin(T)).ravel()
    rows = bulk.evaluate_at_points(np.column_stack([r, z]).tolist(), lagrangian=False)
    u2 = np.array([row[iu] ** 2 + row[iw] ** 2 if row[0] == 1.0 else 0.0 for row in rows])
    return float(np.sum(u2 * math.pi * r * P.ravel()) * h_rho * h_th)


def project(problem, settings: ReconnectSettings, t_cut: float, out: dict | None = None) -> dict:
    """Steps 6-8: impulsive history, the projection, the restart of the history at ``t_cut``.

    Returns the projection diagnostics, filled into ``out`` as they are made, so that they survive
    a failure. Raises :class:`CutFailed` if the projection solve did not converge, remeshed, or
    broke the energy identity."""
    s = settings
    pb = problem
    out = {} if out is None else out
    before = pb.cut_observables()
    pb.assign_initial_values_impulsive()  # levels 1, 2 := the transferred (and filled) field
    out["before"] = before
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
        out["newton_residuals"] = residuals
        if len(pb._domains_to_remesh) or getattr(pb, "_remesh_requested_inside_solve", False):
            raise CutFailed("a remesh was requested during the projection")
        after = pb.cut_observables()  # level 1 still holds u~: projection_change = |u+ - u~|^2/2
        dke = before["kinetic_energy"] - after["kinetic_energy"]
        identity = dke - after["projection_change"]
        out.update(after=after, delta_KE=dke,
                   energy_identity_error=identity,
                   delta_Pz=after["momentum_z"] - before["momentum_z"],
                   # u+ - u~ = -grad Pi with Pi = 0 on the free surfaces: delta P_z ~ int_bottom Pi
                   Pi_on_bottom=after.get("pressure_bottom", math.nan),
                   Pi_rms_on_surface=math.sqrt(max(after["pressure_sq"], 0.0) / after["area"]),
                   # the L2 norm of the pointwise divergence of the P2 velocity: a discretisation
                   # diagnostic of order h (module docstring), not the acceptance of the cut
                   strong_divergence_L2_before=math.sqrt(max(before["divergence_sq"], 0.0)),
                   strong_divergence_L2_after=math.sqrt(max(after["divergence_sq"], 0.0)),
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

    ``driver``, if given, keeps the ``precut``/``postcut`` dumps with its bookkeeping, gives the
    pinch time of the neck (:meth:`~faraday_singularity.driver.CollapseDriver.pinch_time`) and
    restarts the bookkeeping for the new tips; without it the dumps hold the pyoomph state only and
    the pinch time is unknown. Returns the cut record, also written to ``cut_diagnostics.json``.
    Raises :class:`CutRefused` when no cut was made, :class:`CutFailed` when the problem was changed
    (the run must end; a new process may start from the ``precut`` dump). Either way
    ``cut_diagnostics.json`` then holds the partial record, with ``outcome`` (``refused`` or
    ``failed``), the stage reached (``failed_at``) and the error, and the exception carries it as
    ``record`` and the path of the ``precut`` dump, once written, as ``precut``. Every other
    ``Exception`` is raised as one of the two, by the stage it came from (:data:`UNCHANGED_STAGES`:
    a refusal; from the remesh on, the bookkeeping and the ``postcut`` dump included: a failure),
    with it as the cause and its traceback in the record (``traceback``), so that a caller sees
    every outcome of the cut as one of them. An interruption (``KeyboardInterrupt``, ``SystemExit``,
    ...) is never turned into an outcome of the cut: the file records ``outcome`` ``interrupted``
    and the interruption propagates unchanged."""
    out = Path(outdir)
    record: dict = {"delta": settings.delta}
    progress = {"stage": "admission"}
    try:
        return _cut(problem, settings, out, record, progress, driver=driver, tag=tag, log=log)
    except BaseException as e:
        stage, err, extra = progress["stage"], e, {}
        if isinstance(e, Exception) and not isinstance(e, (CutRefused, CutFailed)):
            message = f"at stage {stage}: {type(e).__name__}: {e}"
            err = (CutRefused(f"no cut was made ({message})") if stage in UNCHANGED_STAGES else
                   CutFailed(f"the cut failed after changing the problem ({message})"))
            extra["traceback"] = traceback.format_exception(e)
        outcome = ("refused" if isinstance(err, CutRefused) else
                   "failed" if isinstance(err, Exception) else "interrupted")
        failed = _jsonable({**record, "outcome": outcome, "failed_at": stage,
                            "error": f"{type(err).__name__}: {err}", **extra})
        if isinstance(err, (CutRefused, CutFailed)):
            err.record = failed
            if err.precut is None:
                err.precut = (record.get("precut_state") or {}).get("path")
        with contextlib.suppress(OSError):  # the failure itself must reach the caller
            out.mkdir(parents=True, exist_ok=True)
            (out / "cut_diagnostics.json").write_text(json.dumps(failed, indent=1) + "\n")
        if err is e:
            raise
        raise err from e


def _cut(pb, s: ReconnectSettings, out: Path, record: dict, progress: dict, *, driver, tag,
         log) -> dict:
    """The sequence of :func:`cut`, filling ``record`` as it goes and ``progress["stage"]``."""
    def stage(name):
        progress["stage"] = name

    if pb.recon is None:
        raise CutRefused("the problem was not made with reconnect settings")
    if pb.cut is not None:
        raise CutRefused("only one cut per run")
    if driver is not None and getattr(driver, "reconnect", None) != s:
        # the driver continues the run after the cut, with the stops and steps of these settings
        raise CutRefused("the driver was not made with the reconnect settings of this cut")
    wall0 = time.time()
    t_cut = pb.get_current_time(dimensional=False, as_float=True)
    tag = tag or f"{s.delta:.6g}"
    chains_before = pb.interface_segments()
    if len(chains_before) != 1:
        raise CutRefused(f"the free surface has {len(chains_before)} chains, not one")
    old = chains_before[0]
    old_open = Chain(np.column_stack([old["r"], old["z"]]), tuple(old["end_types"]))
    r_n, z_n, _ = neck_radius(old["r"], old["z"])
    record.update(t_cut=t_cut, r_neck=r_n, z_neck=z_n,
                  delta_over_r_neck=s.delta / r_n if r_n > 0.0 else math.nan,
                  settings=s.to_dict())
    if not math.isfinite(r_n):
        raise CutRefused("the free surface has no neck")
    if s.delta < s.min_landing_ratio * r_n:
        raise CutRefused(f"delta {s.delta:.6g} is below {s.min_landing_ratio} times the neck "
                         f"radius {r_n:.6g}")
    if s.delta > s.max_landing_ratio * r_n:
        # the cut geometry in units of delta depends on delta/r_n: a cut far above the neck is not
        # comparable with cuts at other delta that land just below their radius
        raise CutRefused(f"delta {s.delta:.6g} is above {s.max_landing_ratio} times the neck "
                         f"radius {r_n:.6g}; cut from a state whose neck has just passed below delta")
    # the time origin of a study in delta: the pinch time of the neck before the cut (decision 16)
    pinch = (driver.pinch_time() if driver is not None else
             {"t_p": math.nan, "source": None, "reason": "no driver: no neck samples"})
    pinch["t_cut_minus_t_p"] = t_cut - pinch["t_p"]
    record["pinch"] = pinch
    stage("precut")
    record["budgets_before"] = pb.cut_observables()
    local_radius = LOCAL_RADIUS * s.delta
    local = {"radius": local_radius, "centre": [0.0, z_n], "samples": LOCAL_SAMPLES,
             "kinetic_energy": {"before": local_kinetic_energy(pb, z_n, local_radius)}}
    record["local_budgets"] = local
    # the state the cut starts from, with its neck: a landing state, which a new run can cut again
    neck = {"delta": s.delta, "r_neck": r_n, "z_neck": z_n, "delta_over_r_neck": s.delta / r_n}
    record["precut_state"] = _keep(pb, driver, out, f"precut_{tag}",
                                   {"cut_stage": "precut", "pinch": pinch, "neck": neck})

    # 2. plan
    stage("plan")
    attempts = []
    record["plan_attempts"] = attempts
    rmin = s.delta
    template = pb.get_mesh("liquid")._templatemesh
    for _ in range(s.plan_attempts):
        try:
            outcome, message = pb.recon.request_pinch(pb, rmin), ""
        except RuntimeError as e:
            clear_plan(pb)
            outcome, message = "refused", str(e)
        except BaseException:  # an interruption: drop a half-made plan and propagate
            clear_plan(pb)
            raise
        attempts.append({"rmin": rmin, "outcome": outcome, "message": message})
        if outcome == "plan":
            break
        if outcome == "none":
            raise CutRefused(f"no pinch planned at cut radius {rmin:.6g} (neck {r_n:.6g})")
        if rmin * s.retry_factor > s.max_landing_ratio * r_n:
            break  # a retry would cut too far above the neck
        rmin *= s.retry_factor
    if attempts[-1]["outcome"] != "plan":
        raise CutRefused(f"no plan after {len(attempts)} attempts: {attempts[-1]}")
    record["rmin"] = rmin
    record["rmin_over_r_neck"] = rmin / r_n

    remeshing = False
    try:
        stage("caps")
        plan, _extra = template._pending_surgery_plans[INTERFACE]
        gas_before = float(sum(plan.fragment_volumes_before))
        record["plan"] = {"events": [{"kind": e.kind, "z_center": e.z_center} for e in plan.events],
                          "gas_before": gas_before,
                          "planner_gas_after": [float(v) for v in plan.fragment_volumes_after],
                          "fresh_points": [int(np.sum(np.asarray(c.origin) < 0))
                                           for c in plan.new_chains]}
        # 3. caps, and the sizes of the new mesh
        edits = replace_caps(plan.new_chains, rmin, s)
        record["caps"] = [asdict(e) for e in edits]
        chains = plan_chains(plan.new_chains)
        geom = cut_geometry(chains, s.reservoir_depth, gas_before)
        record["geometry"] = geom.to_dict()
        # the splices, bridge and removed gas in cut radii: the planner's window usually sets them
        record["in_cut_radii"] = in_cut_radii(edits, geom, rmin, s)
        # the plan's gas spans and volumes, for the edited chains (informational in pyoomph)
        plan.axis_spans_inside = np.array(
            [[c.points[0, 1], c.points[-1, 1] + (s.reservoir_depth if c.is_open else 0.0)]
             for c in chains])
        plan.fragment_volumes_after = chain_volumes(chains, s.reservoir_depth)
        # the mesh of the cut in units of the cut radius, so that the cuts of a study in delta are
        # meshed alike: the tips and the bridge (kept for every later remesh) and the focus of
        # this remesh, which replaces the one the landing run left (sized by its own history)
        sizing = CutSizing(h_tip=tip_size(rmin, s), h_bridge=s.bridge_size * rmin,
                           curvature=s.curvature_sizing, grading=template.settings.grading)
        landing_focus = template.focus
        cut_focus = Focus(size=s.focus_size * rmin, radius=min(pb.nd.radius, s.focus_radius * rmin),
                          grading=(landing_focus.grading if landing_focus is not None
                                   else template.settings.grading))
        record["focus"] = {"landing": None if landing_focus is None else asdict(landing_focus),
                           "cut": asdict(cut_focus)}
        # every change of the free surface lies between the two splices
        z_lo, z_hi = min(e.z_s for e in edits), max(e.z_s for e in edits)
        margin = 2.0 * rmin
        box = (z_hi - z_lo + 2.0 * margin, z_lo - margin, z_hi + margin)
        gas_box_before = gas_volume_in_box([old_open], s.reservoir_depth, box)
        record["sizing"] = asdict(sizing)
        record["sizing_in_cut_radii"] = {"h_tip": s.tip_size, "h_bridge": s.bridge_size,
                                         "focus_size": s.focus_size, "focus_radius": s.focus_radius}
        template.cut_sizing = sizing
        template.focus = cut_focus
        # 4. the remesh, at once
        stage("remesh")
        remeshing = True
        with captured_stdout() as printed:
            pb.force_remesh({template})
        pb._domains_to_remesh.clear()
    except BaseException as e:
        clear_plan(pb)
        template.cut_sizing = None
        if not remeshing and "focus" in record:  # put back the landing state's focus
            template.focus = landing_focus
        if not isinstance(e, Exception):  # an interruption is no outcome of the cut
            raise
        if not remeshing:  # nothing has changed but the plan, now dropped
            raise CutRefused(f"the plan could not be built ({type(e).__name__}: {e})") from e
        raise CutFailed(f"the cut remesh failed ({type(e).__name__}: {e})",
                        record["precut_state"]["path"]) from e

    try:
        stage("mesh-check")
        record["transfer"] = transfer_warnings("".join(printed))
        record["transfer"]["former_gas_nodes"] = former_gas_nodes(pb, old_open, s.reservoir_depth,
                                                                  0.05 * rmin)
        # the fresh tips on the axis get no transferred value of the interface's only own field,
        # the kinematic Lagrange multiplier _kin_bc; it has no time history, so this is Newton's
        # start only
        record["transfer"]["corner_fields"] = sorted(
            pb.get_mesh(INTERFACE + "/axis").get_nodal_field_indices())
        segs = pb.interface_segments()
        record["mesh"] = {"chains": [{"open": c["open"], "nodes": len(c["r"]),
                                      "z_ends": [float(c["z"][0]), float(c["z"][-1])]}
                                     for c in segs]}
        if len(segs) != 2 or sum(c["open"] for c in segs) != 1:
            raise CutFailed(f"the cut mesh has {len(segs)} interface chains")
        spans = _axis_spans_of_mesh(pb)
        tips = [float(c["z"][k]) for c in segs for k in (0, -1)
                if c["end_types"][0 if k == 0 else 1] == "axis"]
        expected = liquid_axis_spans(tips)
        record["mesh"].update(axis_spans=spans, expected_axis_spans=[list(x) for x in expected])
        if len(spans) != len(expected):
            raise CutFailed(f"the cut mesh has axis spans {spans}, expected {expected}")
        tol = 1e-7 * max(1.0, pb.nd.radius)
        gas_box_after = gas_volume_in_box(
            [orient_chain(np.column_stack([c["r"], c["z"]]), pb.nd.radius, tol) for c in segs],
            s.reservoir_depth, box)
        record["budgets_after_transfer"] = pb.cut_observables()
        local["kinetic_energy"]["after_transfer"] = local_kinetic_energy(pb, z_n, local_radius)
        # 5. fill
        stage("fill")
        record["fill"] = {"kind": s.fill, "zeroed_nodes": 0, "zeroed_axis_nodes": 0}
        if s.fill == "zero":
            reach = 20.0 * max(geom.bridge, rmin)
            zeroed = zero_fill(pb, old_open, s.reservoir_depth, (0.0, z_n), reach)
            record["fill"].update(zeroed_nodes=zeroed["bulk"] + zeroed["axis"],
                                  zeroed_axis_nodes=zeroed["axis"])
        record["budgets_after_fill"] = pb.cut_observables()
        local["kinetic_energy"]["after_fill"] = local_kinetic_energy(pb, z_n, local_radius)
        # 6-8. projection and restart
        stage("projection")
        record["projection"] = {}
        project(pb, s, t_cut, record["projection"])
        stage("chart")
        orient_interface_chart(pb)
    except CutFailed as e:
        e.precut = record["precut_state"]["path"]
        raise
    except Exception as e:  # an interruption propagates unchanged
        raise CutFailed(f"after the cut remesh: {type(e).__name__}: {e}",
                        record["precut_state"]["path"]) from e
    stage("postcut")
    record["budgets_after"] = pb.cut_observables()
    local["kinetic_energy"]["after"] = local_kinetic_energy(pb, z_n, local_radius)
    volumes = {"liquid_before": record["budgets_before"]["volume"],
               "liquid_after": record["budgets_after"]["volume"],
               "liquid_gain_plan": geom.removed_gas,
               "box": list(box), "gas_in_box_before": gas_box_before,
               "gas_in_box_after": gas_box_after, "liquid_gain_box": gas_box_before - gas_box_after}
    volumes["liquid_gain_mesh"] = volumes["liquid_after"] - volumes["liquid_before"]
    # what the rebuild of the free surface outside the box changed
    volumes["remesh_drift"] = volumes["liquid_gain_mesh"] - volumes["liquid_gain_box"]
    record["volumes"] = volumes
    record["outcome"] = "cut"
    record["wall_seconds"] = time.time() - wall0
    pb.cut = _jsonable(record)
    if driver is not None:
        driver.reset_after_cut(pb.cut)
    record["postcut_state"] = _keep(pb, driver, out, f"postcut_{tag}", {"cut_stage": "postcut"})
    pb.cut = _jsonable(record)
    (out / "cut_diagnostics.json").write_text(json.dumps(pb.cut, indent=1) + "\n")
    g = record["in_cut_radii"]
    log(f"cut at delta {s.delta:.6g} (rmin {rmin:.6g}, r_n {r_n:.6g}): splices at r_s/rmin "
        f"{', '.join(f'{x:.3g}' for x in g['r_s'])} ({', '.join(g['splice_set_by'])}), bridge "
        f"{g['bridge']:.3g} rmin, removed gas {g['removed_gas']:.3g} rmin^3, "
        f"delta KE {record['projection'].get('delta_KE')}, t_cut - t_p {pinch['t_cut_minus_t_p']}")
    return pb.cut


def _keep(problem, driver, outdir: Path, name: str, extra: dict) -> dict:
    """A kept state dump under ``states/``, outside the driver's rotation, with its JSON."""
    path = Path(outdir) / "states" / f"{name}.dump"
    if driver is not None:
        driver.keep_state(path, **extra)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        problem.save_state(str(path), quiet=True)
        focus = problem.active_mesh_template().focus
        meta = {"time": problem.get_current_time(dimensional=False, as_float=True),
                "focus": None if focus is None else asdict(focus), **extra}
        if problem.cut is not None:
            meta["cut"] = problem.cut
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    return {"path": str(path), "time": problem.get_current_time(dimensional=False, as_float=True)}


def restore_cut(problem, record: dict) -> None:
    """After loading a state written after a cut: the record, the mesh sizing of the cut, and the
    chart orientation.

    Every remesh after a cut sizes the fresh tips and the bridge by the cut's ``CutSizing``, which
    lives on the mesh template. pyoomph builds a reloaded mesh from a sibling of the problem's
    original template that copies that template's attributes; in a new process the original
    template has no cut sizing, so it is put back here from the record, or the remeshes after the
    restart would differ from those of the run that wrote the state. The focus is not restored
    here: it is the driver's, dumped with its bookkeeping (``CollapseDriver.restore``)."""
    if problem.reconnect is None:
        raise RuntimeError("a state written after a cut needs a problem made with reconnect settings")
    if "sizing" not in record:
        raise RuntimeError("the cut record of the state has no mesh sizing")
    problem.active_mesh_template().cut_sizing = CutSizing(**record["sizing"])
    problem.cut = record
    orient_interface_chart(problem)
