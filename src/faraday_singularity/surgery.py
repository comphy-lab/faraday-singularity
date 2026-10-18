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
   ``h_floor``; both are recorded (``focus``). Only the bubble's bottom, the landing's cavity bottom
   with its jet, far from the cut, keeps the landing focus where it is finer
   (``reconnect.kept_region``: a second region of the focus, within the landing focus radius of its
   tip and its half of the bubble; ``focus.kept``), so that the cut coarsens no feature it does
   not make. The collapse driver sets the focus again by its own rule, feature by feature, at its
   next rebuild, and keeps that resolution of the bubble's bottom.
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

**The cut of a liquid neck** after the gas cut (:func:`liquid_cut`; the driver does not make it
yet) removes the head of a neck the events branch lands (``events.post_cut_events``) and closes the
stump at the neck by a half-spheroid (``events.truncate_chain``). pyoomph's planner sees only the
necks of the gas, so the plan of the new chains is the project's (``events.build_liquid_plan``),
parked as pyoomph parks its own (:func:`park_plan`): the remesh builds it and the transfer carries
the fields across through its chart. The transfer is checked against the geometry
(:func:`transfer_problems`): pyoomph's count of the interface nodes it leaves unlocated lies within
``UNLOCATED_SLACK`` of the count of those farther than half an element from every old element its
locator tries (``events.predict_unlocated``), every predicted one on the cap, and at most the
stump's tip on the axis has no value. pyoomph gives the unlocated nodes the velocity of the old
surface near the neck, so the cap's nodes are given the old bulk velocity at their places, sampled
along the planned cap before the remesh (``cap_fill`` ``sample``). The projection follows, with the
step before the cut as the history's and the time unchanged. The budgets are taken within a box
about the neck: the liquid volume and the area of the interface from the polylines clipped to it
(:func:`box_polyline_budgets`), the axial momentum and the kinetic energy on a polar grid
(:func:`box_grid`, :func:`box_integrals`) of the box and of the removed head, before the cut, after
the transfer and the fill, and after the projection (:func:`project` with ``box``, which also reads
the identity of the box's cutoff in its weak form). The removed volume and momentum must close
against the box's change, and the projection may change the momentum there by little
(``HARD_CLOSURES``); the energy, the area and the identity are recorded against their limits.
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

from .mesh import CutSizing, Focus, FocusRegion
from .reconnect import (Chain, ReconnectSettings, chain_volumes, cut_geometry, gas_volume_in_box,
                        in_cut_radii, kept_region, liquid_axis_spans, neck_radius, orient_chain,
                        plan_chains, replace_caps, tip_size)

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


BOX_SAMPLES = 256  # points per direction of the polar grid of the budgets of a liquid cut


def box_grid(z_centre: float, radius: float, n: int = BOX_SAMPLES) -> dict:
    """The polar grid of the budgets within ``radius`` of the point ``(0, z_centre)`` of the
    axis, as :func:`local_kinetic_energy` lays it over the half-disc: the midpoints of ``n x n``
    cells in the distance ``rho`` from the centre and the angle from the axis, their places ``r``,
    ``zz``, the volume of each cell revolved about the axis (``dV``, ``2 pi r rho drho dtheta``),
    and the C1 cutoff ``phi`` of the box with its axial derivative ``dphi_dz``
    (:func:`box_cutoff`)."""
    h_rho, h_th = radius / n, math.pi / n
    rho = (np.arange(n) + 0.5) * h_rho
    theta = -0.5 * math.pi + (np.arange(n) + 0.5) * h_th
    P, T = np.meshgrid(rho, theta, indexing="ij")
    r, z = (P * np.cos(T)).ravel(), (z_centre + P * np.sin(T)).ravel()
    phi, dphi = box_cutoff(P.ravel(), radius)
    return {"z": float(z_centre), "R": float(radius), "n": int(n), "r": r, "zz": z,
            "rho": P.ravel(), "dV": 2.0 * math.pi * r * P.ravel() * h_rho * h_th, "phi": phi,
            "dphi_dz": dphi * np.sin(T).ravel()}


def box_cutoff(rho, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """The cutoff of the box of radius ``radius`` at the distances ``rho`` from its centre and its
    derivative in ``rho``: one within half the radius, zero beyond it, a cubic step in between
    (continuously differentiable), so that ``v = phi e_z`` is a velocity the projection's weak form
    holds for, up to its interpolation, wherever the box lies inside the liquid or reaches only its
    free surfaces."""
    t = np.clip((np.asarray(rho, dtype=float) - 0.5 * radius) / (0.5 * radius), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t), -6.0 * t * (1.0 - t) / (0.5 * radius)


def sample_fields(problem, r, z, names) -> tuple[dict, np.ndarray]:
    """The nodal fields ``names`` of the liquid at the points ``(r, z)``, each located in the mesh
    in one call (``nan`` where a point lies outside the liquid), and which points were found."""
    bulk = problem.get_mesh("liquid")
    fi = bulk.get_nodal_field_indices()
    rows = bulk.evaluate_at_points(np.column_stack([r, z]).tolist(), lagrangian=False)
    found = np.array([row[0] == 1.0 for row in rows], dtype=bool)
    out = {}
    for name in names:
        col = 1 + fi[name]
        out[name] = np.array([row[col] if ok else math.nan for row, ok in zip(rows, found)],
                             dtype=float)
    return out, found


def box_integrals(problem, grid: dict, polygons: dict | None = None) -> dict:
    """The axial momentum ``Pz`` and the kinetic energy ``KE`` of the liquid within the box of
    ``grid`` (:func:`box_grid`), by the midpoint rule over its cells, points outside the liquid
    counting zero, and the volume the found points cover (``V_grid``); for each half-section of
    ``polygons`` (by name) the same over the cells inside it, from the same samples."""
    import shapely
    from shapely.geometry import Polygon

    f, found = sample_fields(problem, grid["r"], grid["zz"], ("velocity_x", "velocity_y"))
    ur, uz = np.where(found, f["velocity_x"], 0.0), np.where(found, f["velocity_y"], 0.0)
    w = np.where(found, grid["dV"], 0.0)

    def sums(mask):
        return {"Pz": float(np.sum(w[mask] * uz[mask])),
                "KE": float(np.sum(0.5 * w[mask] * (ur[mask] ** 2 + uz[mask] ** 2))),
                "V_grid": float(np.sum(w[mask])), "points": int(np.count_nonzero(mask & found))}

    out = {"z": grid["z"], "R": grid["R"], "n": grid["n"], **sums(np.ones(len(w), dtype=bool))}
    for name, section in (polygons or {}).items():
        inside = np.asarray(shapely.contains_xy(Polygon(section), grid["r"], grid["zz"]),
                            dtype=bool)
        out[name] = sums(inside)
    return out


def box_polyline_budgets(chains, z_centre: float, radius: float, reservoir_depth: float) -> dict:
    """The volumes and the area within the half-disc of ``radius`` about ``(0, z_centre)`` from the
    polylines of the ``chains`` (``reconnect.Chain``), clipped to it (as ``reconnect.
    gas_volume_in_box`` clips to a box): the gas they bound (``V_gas``; the open chain closed
    through the gas above it), the liquid (``V_liquid``, the half-disc's revolved volume, a sphere,
    less the gas) and the area of the interface (``A``), each polyline segment revolved."""
    from shapely.geometry import LineString, Point, Polygon
    from shapely.geometry import box as shapely_box

    from .reconnect import closed_section, revolved_volume

    disc = Point(0.0, z_centre).buffer(radius, quad_segs=256).intersection(
        shapely_box(0.0, z_centre - radius, radius, z_centre + radius))
    V_disc = revolved_volume(np.asarray(disc.exterior.coords)[:-1])
    gas = area = 0.0
    for c in chains:
        part = Polygon(closed_section(c.points, c.end_types, reservoir_depth)).intersection(disc)
        for poly in getattr(part, "geoms", [part]):
            if poly.geom_type != "Polygon" or poly.is_empty:
                continue
            gas += revolved_volume(np.asarray(poly.exterior.coords)[:-1])
            gas -= sum(revolved_volume(np.asarray(h.coords)[:-1]) for h in poly.interiors)
        line = LineString(c.points).intersection(disc)
        for seg in getattr(line, "geoms", [line]):
            if seg.geom_type != "LineString" or seg.is_empty:
                continue
            Q = np.asarray(seg.coords)
            d = np.diff(Q, axis=0)
            area += float(np.sum(math.pi * (Q[:-1, 0] + Q[1:, 0]) * np.hypot(d[:, 0], d[:, 1])))
    return {"z": float(z_centre), "R": float(radius), "V_disc": V_disc, "V_gas": gas,
            "V_liquid": V_disc - gas, "A": area}


def _box_projection(grid: dict, before: dict, after: dict, found: np.ndarray) -> dict:
    """What the projection changed within the box of ``grid``, from the velocity ``u~`` sampled
    before it (``before``) and ``u+`` and the pressure impulse ``Pi`` after it, before the old
    pressure is put back (``after``), on the same points (``found`` in the liquid both times): the
    axial momentum and the kinetic energy there before and after (as :func:`box_integrals`), their
    changes (``delta_Pz``, ``delta_KE``) and the identity of the cutoff ``phi``
    (:func:`box_cutoff`), ``int (u+ - u~)_z phi dV = int Pi dphi/dz dV`` (``phi_lhs``,
    ``phi_rhs``, ``phi_error``), which the weak form of the projection makes hold up to the
    interpolation of ``phi e_z`` and to the quadrature of the grid."""
    w = np.where(found, grid["dV"], 0.0)

    def z(x):
        return np.where(found, x, 0.0)

    def sums(f):
        return {"Pz": float(np.sum(w * z(f["velocity_y"]))),
                "KE": float(np.sum(0.5 * w * (z(f["velocity_x"]) ** 2 + z(f["velocity_y"]) ** 2))),
                "V_grid": float(np.sum(w)), "points": int(found.sum())}

    du = z(after["velocity_y"]) - z(before["velocity_y"])
    lhs = float(np.sum(w * du * grid["phi"]))
    rhs = float(np.sum(w * z(after["pressure"]) * grid["dphi_dz"]))
    scale = max(abs(lhs), abs(rhs), 1e-300)
    out = {"z": grid["z"], "R": grid["R"], "n": grid["n"], "points": int(found.sum()),
           "before": sums(before), "after": sums(after)}
    out.update(delta_Pz=out["after"]["Pz"] - out["before"]["Pz"],
               delta_KE=out["after"]["KE"] - out["before"]["KE"], phi_lhs=lhs, phi_rhs=rhs,
               phi_error=lhs - rhs, phi_relative_error=(lhs - rhs) / scale,
               max_abs_Pi=float(np.max(np.abs(z(after["pressure"])))) if found.any() else math.nan)
    return out


def project(problem, settings: ReconnectSettings, t_cut: float, out: dict | None = None, *,
            box: tuple[float, float] | None = None) -> dict:
    """Steps 6-8: impulsive history, the projection, the restart of the history at ``t_cut``.

    Returns the projection diagnostics, filled into ``out`` as they are made, so that they survive
    a failure. Raises :class:`CutFailed` if the projection solve did not converge, remeshed, or
    broke the energy identity. With ``box`` (``(z_centre, radius)``, a liquid cut's) the velocity is
    sampled on the polar grid of that box (:func:`box_grid`) before the solve and again, with the
    pressure impulse, after it and before the old pressure is put back, for what the projection
    changed there (``out["box"]``, :func:`_box_projection`); without it nothing else is read."""
    s = settings
    pb = problem
    out = {} if out is None else out
    before = pb.cut_observables()
    pb.assign_initial_values_impulsive()  # levels 1, 2 := the transferred (and filled) field
    out["before"] = before
    if s.projection:
        grid = sampled = None
        if box is not None:
            grid = box_grid(*box)
            sampled = sample_fields(pb, grid["r"], grid["zz"], ("velocity_x", "velocity_y"))
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
        if grid is not None:  # u+ and Pi on the box, before the old pressure is put back
            fields, found = sample_fields(pb, grid["r"], grid["zz"],
                                          ("velocity_x", "velocity_y", "pressure"))
            out["box"] = _box_projection(grid, sampled[0], fields, found & sampled[1])
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


def park_plan(problem, plan, old_zeta_table, *, rmin: float) -> None:
    """Park a surgery plan built by the project (``events.build_liquid_plan``) on the mesh template,
    as pyoomph's reconnection handler parks its own (``AxisymmetricReconnection.
    after_newton_solve``): with the names of the interface and of its axis read in the
    interface's scope, the old chart as the table the transfer looks the old nodes up in, and
    ``rmin`` as the cut radius, of which the corner nodes of the new interface may lie at most
    twice from an old one to take its interface fields. The next remesh (``force_remesh``) builds
    the plan's chains and transfers the fields through its chart; nothing may solve in between,
    since the handler drops a parked plan at every Newton solve."""
    recon = problem.recon
    if recon is None:
        raise RuntimeError("the problem was not made with reconnect settings")
    tree = problem.get_mesh(INTERFACE).get_eqtree()
    if not any(e is recon for e in tree._equations):
        raise RuntimeError("the reconnection handler is not an equation of the free surface")
    template = problem.get_mesh("liquid")._templatemesh
    with tree._on_this_domain():
        cg = recon.get_current_code_generator()
        iname = cg.get_full_name()
        try:
            _, axis_name = recon.get_interface_and_axisymm_name()
        except RuntimeError:
            axis_name = None
        try:
            _, opposite_axis = recon.get_opposite_and_opposite_axisymm_name()
        except RuntimeError:
            opposite_axis = None
        try:
            opposite = cg._get_opposite_interface()
            opposite_interface = None if opposite is None else opposite.get_full_name()
        except RuntimeError:
            opposite_interface = None
        scale = problem.get_scaling("spatial")
    recon._pending_on = (template, iname)
    template._set_pending_surgery_plan(iname, plan, {
        "spatial_scale": scale, "interface_name": iname, "axis_name": axis_name,
        "opposite_axis_name": opposite_axis, "opposite_interface_name": opposite_interface,
        "old_zeta_table": np.asarray(old_zeta_table, dtype=float), "rmin_nd": float(rmin),
        "distmin_nd": None})
    problem._domains_to_remesh.add(template)


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
        # the bubble's bottom, the landing's cavity bottom with its jet, keeps the resolution it
        # had at the landing (reconnect.kept_region): a focus region beside the cut's own
        h_cut, r_cut = s.focus_size * rmin, min(pb.nd.radius, s.focus_radius * rmin)
        kept = kept_region(None if landing_focus is None else landing_focus.to_dict(), h_cut,
                           geom.z_bubble_bottom, geom.z_bubble_top)
        regions = () if kept is None else (
            FocusRegion("cut", size=h_cut, r=0.0, z=0.0, radius=r_cut, axial=True),
            FocusRegion("bubble_bottom", size=kept["size"], r=0.0, z=geom.z_bubble_bottom,
                        radius=kept["radius"]))
        cut_focus = Focus(size=min([h_cut] + [g.size for g in regions]), radius=r_cut,
                          grading=(landing_focus.grading if landing_focus is not None
                                   else template.settings.grading), regions=regions)
        record["focus"] = {"landing": None if landing_focus is None else landing_focus.to_dict(),
                           "cut": cut_focus.to_dict(), "kept": kept}
        # every change of the free surface lies between the two splices
        z_lo, z_hi = min(e.z_s for e in edits), max(e.z_s for e in edits)
        margin = 2.0 * rmin
        box = (z_hi - z_lo + 2.0 * margin, z_lo - margin, z_hi + margin)
        gas_box_before = gas_volume_in_box([old_open], s.reservoir_depth, box)
        record["sizing"] = asdict(sizing)
        record["sizing_in_cut_radii"] = {"h_tip": s.tip_size, "h_bridge": s.bridge_size,
                                         "focus_size": s.focus_size, "focus_radius": s.focus_radius}
        template.cut_sizing = sizing
        # the curved elements of every mesh after the cut, from this one on (mesh.py)
        template.high_order_threshold = s.high_order_threshold
        template.rebuild_spline = s.rebuild_spline
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
        template.high_order_threshold, template.rebuild_spline = None, "nodes"
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
                "focus": None if focus is None else focus.to_dict(), **extra}
        if problem.cut is not None:
            meta["cut"] = problem.cut
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    return {"path": str(path), "time": problem.get_current_time(dimensional=False, as_float=True)}


def restore_cut(problem, record: dict) -> None:
    """After loading a state written after a cut: the record, the mesh sizing of the cut (with the
    threshold of gmsh's optimiser of curved elements and the nodes the spline of a rebuild runs
    through, ``high_order_threshold`` and ``rebuild_spline`` of its settings), and the chart
    orientation.

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
    template = problem.active_mesh_template()
    template.cut_sizing = CutSizing(**record["sizing"])
    # cuts recorded before these settings existed keep gmsh's defaults and every node
    settings = record.get("settings") or {}
    template.high_order_threshold = settings.get("high_order_threshold")
    template.rebuild_spline = settings.get("rebuild_spline", "nodes")
    problem.cut = record
    orient_interface_chart(problem)


# --- the cut of a liquid neck after the gas cut ---------------------------------------------------

EVENT_SCHEMA = "faraday-event/2"
# the kinds of event a liquid cut is made for, and the class of each (events.post_cut_events)
LIQUID_CUT_CLASSES = {"liquid-neck": "L", "a-head": "A-head"}
# the stages of a liquid cut before it changes the problem: any failure up to them is a refusal
LIQUID_UNCHANGED_STAGES = ("admission", "precut", "caps", "plan", "budgets", "park")
CAP_FILL_DENSITY = 4  # samples of the old bulk velocity per node of the new mesh along the cap
CAP_NODE_TOL = 0.05  # a new interface node within this times r_n of the planned cap lies on it
UNLOCATED_SLACK = 2  # the transfer may leave this many interface nodes more or fewer than predicted
# the budgets of a liquid cut within its box (closures F1-F5; sigma, the box noise of an ordinary
# rebuild, not yet calibrated, counts zero): the coefficient of each limit, and the closures of the
# volume and the axial momentum of the removal, which are hard checks
BUDGET_LIMITS = {"F1": 1e-2, "F2": 0.05, "F3": 0.05, "F4": 0.1, "F5": 1e-2}
HARD_CLOSURES = ("F1", "F2", "F3")


def _named_chains(problem) -> tuple[tuple, dict]:
    """The chains of the interface by name (``open``, ``bubble``) as the stops after a cut read
    them, the ends on the axis put on ``r = 0``, and the segments themselves."""
    named, segs = [], {}
    for c in problem.interface_segments():
        name = "open" if c["open"] else "bubble"
        if name in segs:
            raise RuntimeError(f"the free surface has more than one {name} chain")
        P = np.column_stack([c["r"], c["z"]]).astype(float)
        for k, end in ((0, 0), (-1, 1)):
            if c["end_types"][end] == "axis":
                P[k, 0] = 0.0
        named.append((name, Chain(P, tuple(c["end_types"]))))
        segs[name] = c
    return tuple(sorted(named, key=lambda x: x[0])), segs


def _cap_polyline(lc) -> tuple[np.ndarray, np.ndarray]:
    """The planned cap from the neck to its tip on the axis, and the arclength along it."""
    Q = np.vstack([[lc.r_n, lc.z_n], lc.cap])
    return Q, np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Q, axis=0).T))])


def _onto_polyline(X, Q, s) -> tuple[np.ndarray, np.ndarray]:
    """The distance of each point ``X`` from the polyline ``Q`` and the arclength ``s`` at the
    nearest point of it."""
    X = np.asarray(X, dtype=float).reshape(-1, 2)
    A, d = Q[:-1], np.diff(Q, axis=0)
    L2 = np.maximum(np.einsum("ij,ij->i", d, d), 1e-300)
    best, at = np.full(len(X), np.inf), np.zeros(len(X))
    for j in range(len(A)):
        t = np.clip(((X - A[j]) @ d[j]) / L2[j], 0.0, 1.0)
        dist = np.hypot(*(X - A[j] - t[:, None] * d[j]).T)
        better = dist < best
        best[better], at[better] = dist[better], s[j] + t[better] * (s[j + 1] - s[j])
    return best, at


def _on_cap(lc, X, chain) -> np.ndarray:
    """Which of the interface nodes at ``X`` lie on the planned cap of the cut ``lc``: nodes of the
    chain it cut (``chain``, its segment of ``FaradayProblem.interface_segments``) beyond the neck
    and within ``CAP_NODE_TOL r_n`` of the cap."""
    Q, s = _cap_polyline(lc)
    dist, u = _onto_polyline(X, Q, s)
    own = {(float(r), float(z)) for r, z in zip(chain["r"], chain["z"])}
    mine = np.array([(float(x), float(y)) in own for x, y in X], dtype=bool)
    return mine & (u > 1e-9 * s[-1]) & (dist <= CAP_NODE_TOL * lc.r_n)


def _fill_cap(problem, lc, samples: dict, chain: dict, *, apply: bool) -> dict:
    """The cap fill of a liquid cut: the new interface nodes on the planned cap (:func:`_on_cap`,
    of the chain it cut, ``chain``) take the old bulk velocity at their places, interpolated by
    arclength along the cap between the samples taken before the remesh (``samples``), at the
    current level (``project`` makes the history impulsive); ``u_r`` stays pinned on the axis. With
    ``apply`` false only what the fill would change is read. Returns the nodes, the samples used and
    the largest change of the velocity."""
    Q, s = _cap_polyline(lc)
    nodes = mesh_nodes(problem.get_mesh(INTERFACE))
    X = np.array([[n.x(0), n.x(1)] for n in nodes]).reshape(-1, 2)
    _, u = _onto_polyline(X, Q, s)
    on = _on_cap(lc, X, chain)
    ok = samples["found"]
    fi = problem.get_mesh("liquid").get_nodal_field_indices()
    iu, iw = fi["velocity_x"], fi["velocity_y"]
    out = {"mode": "sample" if apply else "transfer", "nodes": int(on.sum()),
           "samples": int(len(ok)), "samples_found": int(ok.sum()), "max_change": 0.0,
           "max_speed": 0.0}
    if not on.any() or not ok.any():
        return out
    ur = np.interp(u[on], samples["s"][ok], samples["u_r"][ok])
    uz = np.interp(u[on], samples["s"][ok], samples["u_z"][ok])
    change = 0.0
    for n, a, b in zip((nodes[k] for k in np.flatnonzero(on)), ur, uz):
        a = 0.0 if n.is_pinned(iu) else float(a)
        change = max(change, math.hypot(a - n.value(iu), b - n.value(iw)))
        if apply:
            if not n.is_pinned(iu):
                n.set_value(iu, a)
            n.set_value(iw, float(b))
    out.update(max_change=float(change), max_speed=float(np.max(np.hypot(ur, uz))))
    if apply:
        problem.reapply_boundary_conditions()
    return out


def transfer_problems(transfer: dict) -> list[str]:
    """What the checks of a liquid cut's transfer find wrong in its record (``transfer``: pyoomph's
    counts, ``unlocated`` and ``no_value`` by mesh, and the prediction), nothing where it is as
    planned: the count of the interface nodes reported unlocated within ``UNLOCATED_SLACK`` of the
    count predicted (``events.predict_unlocated``; pyoomph reports counts, not the nodes), no
    predicted one off the planned cap, a cap with nodes, no node of the bulk or the axis unlocated,
    and no node without a value but at most one on the axis of the interface, the stump's new tip
    (its own field ``_kin_bc``, Newton's start only)."""
    problems = []
    reported, predicted = transfer["reported_unlocated"], transfer["predicted_unlocated"]
    if abs(reported - predicted) > UNLOCATED_SLACK:
        problems.append(f"{reported} interface nodes unlocated, {predicted} predicted")
    if transfer["predicted_off_cap"]:
        problems.append(f"{transfer['predicted_off_cap']} unlocated interface nodes off the cap")
    if not transfer["cap_nodes"] > 0:
        problems.append("no interface node on the planned cap")
    others = {name: v for name, v in transfer["unlocated"].items() if name != INTERFACE}
    if any(v[0] > 0 for v in others.values()):
        problems.append(f"unlocated nodes of the bulk or the axis: {others}")
    no_value, corner = transfer["no_value"], INTERFACE + "/axis"
    if set(no_value) - {corner} or no_value.get(corner, [0])[0] > 1:
        problems.append(f"nodes without a value beyond the stump's tip: {no_value}")
    return problems


def _closures(removed: dict, poly: dict, box: dict) -> dict:
    """The budgets of a liquid cut within its box (``poly``: the polyline volumes and areas before
    and after the transfer; ``box``: the grid's momentum and kinetic energy before it, after the
    transfer and the cap fill, and after the projection), each against its limit
    (``BUDGET_LIMITS``, the box noise counting zero until calibrated): F1 the liquid volume lost
    against the removed head's, F2 the axial momentum the transfer lost against the head's, F3 the
    momentum the projection changed, F4 the kinetic energy it took, F5 the area of the interface
    against the head's less the cap's. ``flags`` lists the closures beyond their limits; those of
    ``HARD_CLOSURES`` among them (``hard``) fail the cut, after its remesh."""
    V, P, KE = removed["V"], removed["Pz"], removed["KE"]
    dA = removed["area"] - removed["cap_area"]
    values = {
        "F1": abs(poly["after"]["V_liquid"] - poly["before"]["V_liquid"] + V),
        "F2": abs(box["after_transfer"]["Pz"] - box["before"]["Pz"] + P),
        "F3": abs(box["after_projection"]["Pz"] - box["after_transfer"]["Pz"]),
        "F4": box["after_transfer"]["KE"] - box["after_projection"]["KE"],
        "F5": abs(poly["after"]["A"] - poly["before"]["A"] + dA)}
    limits = {"F1": BUDGET_LIMITS["F1"] * V, "F2": BUDGET_LIMITS["F2"] * abs(P),
              "F3": BUDGET_LIMITS["F3"] * abs(P), "F4": BUDGET_LIMITS["F4"] * KE,
              "F5": BUDGET_LIMITS["F5"] * abs(dA)}
    flags = [k for k in values if not values[k] <= limits[k]]
    return {**values, "limits": limits, "relative": {
        "F1": values["F1"] / V if V else math.nan, "F2": values["F2"] / abs(P) if P else math.nan,
        "F3": values["F3"] / abs(P) if P else math.nan, "F4": values["F4"] / KE if KE else math.nan,
        "F5": values["F5"] / abs(dA) if dA else math.nan},
        "flags": flags, "hard": [k for k in flags if k in HARD_CLOSURES], "sigma": 0.0}


def liquid_cut(problem, events, reconnect: ReconnectSettings, outdir: Path, request: dict, *,
               k: int, dt_prev: float, elements_per_scale: float, h_floor: float, driver=None,
               on_made=None, log=print) -> dict:
    """Cut the liquid neck an event request lands (``events.post_cut_events``: ``liquid-neck``,
    class ``L``, or ``a-head``, class ``A-head``), the ``k``-th event after the gas cut, and
    project.

    1. **Admission**: a gas cut was made (``problem.cut``), the request is of a kind and class
       this cut is for (``LIQUID_CUT_CLASSES``) and names a tracked neck of the state with a single
       head (``events.liquid_necks`` with ``events.liquid``) that lands under ``events.liquid`` as
       its class says (``r_n <= eps_d s`` for ``L``, ``r_n <= eps_max s`` for ``A-head``), whatever
       settings the request was made under; ``precut_e<k>`` is kept (``cut_stage``
       ``precut-event``, the request as its ``event``).
    2. **Cap** (``events.truncate_chain``): the head removed, the stump closed at the neck by the
       half-spheroid of ``liquid.cap_height`` (tried once more at ``events.CAP_RETRY_HEIGHT`` if
       refused), ``elements_per_scale`` elements per neck radius, refused unless its hard checks
       hold; and the plan for the transfer (``events.build_liquid_plan``).
    3. **Budgets before**: the global ones (``cut_observables``), within the box of radius
       ``max(liquid.box_radius delta, 2 L_rem)`` about the neck the volume and the area of the
       interface from the polylines (:func:`box_polyline_budgets`) and the momentum and kinetic
       energy on a polar grid (:func:`box_integrals`), of the box and of the removed region ``Q``,
       the same within ``LOCAL_RADIUS delta``, and the old bulk velocity along the cap
       (``CAP_FILL_DENSITY`` samples per node of the new mesh there).
    4. **Remesh at once** from the plan (:func:`park_plan`, ``rmin`` the neck radius), the focus the
       driver's with the region ``event_cap`` at the cap's tip, meshed at ``clip(r_n/
       elements_per_scale, h_floor, h_cut)``, ``h_cut`` the gas cut's focus size. From here a
       failure is :class:`CutFailed`. The transfer is checked (:func:`transfer_problems`): the
       count of the interface nodes it leaves unlocated within ``UNLOCATED_SLACK`` of the count
       predicted from the geometry (``events.predict_unlocated``), every predicted one on the cap,
       none of the bulk or the axis, and at most the stump's tip on the axis without a value; the
       chains, their ends and the axis spans as planned.
    5. **Cap fill** (``liquid.cap_fill`` ``sample``): the cap's nodes take the old bulk velocity at
       their places, not the nearest old surface's that pyoomph's transfer gives them.
    6. **Projection** (:func:`project`, with ``liquid.projection`` and ``liquid.restart`` or the gas
       cut's): ``Pi = 0`` on every free surface, the cap's included, the step before the cut
       (``dt_prev``) as the history's, the time unchanged; within the box, what it changed and the
       identity of the box's cutoff (``projection.box``).
    7. **Budgets after** and the closures (:func:`_closures`), of which the volume and the momentum
       of the removal are hard checks; the removed head as the ghost the run is to follow
       (``ghost``: its volume, momentum and kinetic energy, its mean axial speed ``U``, its
       equivalent radius, the retraction speed of its rear ``r_n^(-1/2)``, its two ends and the
       speed of the stump's tip relative to it) and the totals of the removals since the gas cut
       (``removed_total``); the record is appended to ``problem.events``, ``on_made`` is called
       with it (the driver's bookkeeping) and ``postcut_e<k>`` kept.

    The record (schema ``faraday-event/2``) is written to ``events/event_<k>.json`` whatever the
    outcome (``made``, ``refused``, ``failed`` or ``interrupted``, with ``failed_at``), as
    :func:`cut` writes its own; a refusal leaves the problem unchanged, a failure ends the run."""
    out = Path(outdir)
    record: dict = {"schema": EVENT_SCHEMA, "k": int(k), "kind": request.get("kind"),
                    "class": request.get("class")}
    progress = {"stage": "admission"}
    path = out / "events" / f"event_{int(k):03d}.json"
    try:
        return _liquid_cut(problem, events, reconnect, out, record, progress, request, path,
                           k=int(k), dt_prev=float(dt_prev),
                           elements_per_scale=float(elements_per_scale), h_floor=float(h_floor),
                           driver=driver, on_made=on_made, log=log)
    except BaseException as e:
        stage, err, extra = progress["stage"], e, {}
        if isinstance(e, Exception) and not isinstance(e, (CutRefused, CutFailed)):
            message = f"at stage {stage}: {type(e).__name__}: {e}"
            err = (CutRefused(f"no liquid cut was made ({message})")
                   if stage in LIQUID_UNCHANGED_STAGES else
                   CutFailed(f"the liquid cut failed after changing the problem ({message})"))
            extra["traceback"] = traceback.format_exception(e)
        outcome = ("refused" if isinstance(err, CutRefused) else
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


def _liquid_cut(pb, events, s: ReconnectSettings, out: Path, record: dict, progress: dict,
                request: dict, path: Path, *, k: int, dt_prev: float, elements_per_scale: float,
                h_floor: float, driver, on_made, log) -> dict:
    """The sequence of :func:`liquid_cut`, filling ``record`` as it goes."""
    from dataclasses import replace

    from .events import (CAP_RETRY_HEIGHT, EVENT_CLASSES, PRECUT_EVENT, CapRefused,
                         build_liquid_plan, event_neck, liquid_necks, predict_unlocated,
                         truncate_chain)

    def stage(name):
        progress["stage"] = name

    wall0 = time.time()
    ls = events.liquid
    if pb.recon is None:
        raise CutRefused("the problem was not made with reconnect settings")
    if pb.cut is None:
        raise CutRefused("a liquid cut follows the gas cut: the problem has none")
    kind, cls = request.get("kind"), request.get("class")
    if kind not in EVENT_CLASSES["liquid"] or LIQUID_CUT_CLASSES.get(kind) != cls:
        raise CutRefused(f"no liquid cut for an event of kind {kind!r} and class {cls!r}")
    if not (dt_prev > 0.0 and math.isfinite(dt_prev)):
        raise CutRefused(f"the step before the cut must be positive, not {dt_prev!r}")
    t_now = pb.get_current_time(dimensional=False, as_float=True)
    t_cut = float(pb.cut["t_cut"])
    t_p = (pb.cut.get("pinch") or {}).get("t_p")
    delta = float(s.delta)
    named, segs = _named_chains(pb)
    wanted = event_neck(request)
    necks = liquid_necks(named, ls)
    neck = next((x for x in necks if x["chain"] == wanted.get("chain")
                 and x["index"] == wanted.get("index")), None)
    record.update(t=t_now, tau=t_now - t_cut, tau_p=t_now - t_p if t_p is not None else math.nan,
                  delta=delta, request={key: v for key, v in request.items() if key != "reason"},
                  settings=events.to_dict(), settings_sha256=events.sha256(),
                  dt={"history": dt_prev})
    if neck is None or neck["status"] != "head":
        raise CutRefused(f"the request names no tracked neck with a single head: {wanted}")
    seg = segs[neck["chain"]]
    u_r = float(seg["u_r"][neck["index"]])
    Oh = float(getattr(pb.nd, "Oh", math.nan))
    record.update(chain=neck["chain"], trigger={
        **neck, "eps_d": ls.eps_d, "eps_max": ls.eps_max, "landed": bool(neck["landed"]),
        "r_n_over_s": neck["r_n"] / neck["s"], "u_r": u_r,
        "Re_n": neck["r_n"] * abs(u_r) / Oh if Oh > 0.0 else math.nan,
        "We_n": neck["r_n"] * u_r**2})
    # the class of the request holds of the neck under these settings, not only under those it was
    # made with (a restart from the state before an event may change the keys of the class)
    if cls == "L" and not neck["landed"]:
        raise CutRefused(f"the neck has not landed at liquid.eps_d {ls.eps_d:g}: r_n/s "
                         f"{neck['r_n'] / neck['s']:.6g}")
    if cls == "A-head" and not neck["r_n"] <= ls.eps_max * neck["s"]:
        raise CutRefused(f"the neck of the head the fold reaches lies above liquid.eps_max "
                         f"{ls.eps_max:g}: r_n/s {neck['r_n'] / neck['s']:.6g}")
    record["dumps"] = {}

    stage("precut")
    event = {**record["request"], "k": k}
    record["dumps"]["precut"] = _keep(pb, driver, out, f"precut_e{k:03d}",
                                      {"cut_stage": PRECUT_EVENT, "event": event})["path"]

    stage("caps")
    attempts, lc = [], None
    heights = (ls.cap_height,) + ((CAP_RETRY_HEIGHT,) if ls.cap_height != CAP_RETRY_HEIGHT else ())
    for eta in heights:
        try:
            lc = truncate_chain(named, neck, eta=eta, elements_per_scale=elements_per_scale,
                                min_angle=s.min_cap_angle, reservoir_depth=s.reservoir_depth)
            attempts.append({"eta": eta, "outcome": "ok", "checks": lc.checks})
            break
        except CapRefused as e:
            attempts.append({"eta": eta, "outcome": "refused", "error": str(e),
                             "checks": e.checks})
    record["cap"] = {"shape": ls.cap, "attempts": attempts, "retried": len(attempts) > 1}
    if lc is None:
        raise CutRefused(f"the cap of the liquid cut is refused at every height: {attempts}")
    summary = lc.summary()
    summary.pop("checks")
    record["cap"].update(summary, tip=lc.tip)
    record["tip"] = lc.tip

    stage("plan")
    lp = build_liquid_plan(named, lc, s.reservoir_depth)
    record["plan"] = {"cap_zeta": list(lp.cap_zeta), "table_rows": int(len(lp.table)),
                      "new_chains": [{"points": len(nc.points), "zeta": [float(nc.zeta[0]),
                                                                         float(nc.zeta[-1])],
                                      "cap_points": int(np.sum(np.asarray(nc.origin) < 0))}
                                     for nc in lp.plan.new_chains]}

    stage("budgets")
    R_b = max(ls.box_radius * delta, 2.0 * lc.L_rem)
    grid, grid20 = box_grid(lc.z_n, R_b), box_grid(lc.z_n, LOCAL_RADIUS * delta)
    global_before = pb.cut_observables()
    box_before = box_integrals(pb, grid, {"removed": lc.removed})
    box20_before = box_integrals(pb, grid20)
    poly_before = box_polyline_budgets([c for _, c in named], lc.z_n, R_b, s.reservoir_depth)
    q = box_before.pop("removed")
    removed = {"V": lc.V_rem, "V_grid": q["V_grid"], "Pz": q["Pz"], "KE": q["KE"],
               "area": lc.A_rem, "cap_area": lc.A_cap, "length": lc.L_rem,
               "U": q["Pz"] / lc.V_rem, "points": q["points"]}
    record["removed"] = removed
    rmin = float(pb.cut.get("rmin") or pb.cut["delta"])
    h_cut = s.focus_size * rmin
    size = float(np.clip(lc.r_n / elements_per_scale, h_floor, h_cut))
    Qc, sc = _cap_polyline(lc)
    m = CAP_FILL_DENSITY * 2 * int(math.ceil(sc[-1] / size)) + 1
    su = np.linspace(0.0, sc[-1], m)
    pts = np.column_stack([np.interp(su, sc, Qc[:, 0]), np.interp(su, sc, Qc[:, 1])])
    f, found = sample_fields(pb, pts[:, 0], pts[:, 1], ("velocity_x", "velocity_y"))
    samples = {"s": su, "u_r": f["velocity_x"], "u_z": f["velocity_y"], "found": found}

    template = pb.get_mesh("liquid")._templatemesh
    old_focus = template.focus
    region = FocusRegion("event_cap", size=size, r=0.0, z=lc.z_tip_new, radius=8.0 * lc.r_n)
    if old_focus is None:
        regions, grading = (region,), pb.mesh_settings.grading
    else:  # a focus without regions is its own axial region beside the cap's
        regions = tuple(old_focus.regions) or (FocusRegion("focus", size=old_focus.size, r=0.0,
                                                           z=0.0, radius=old_focus.radius,
                                                           axial=True),)
        regions, grading = regions + (region,), old_focus.grading
    record["mesh"] = {"focus_before": None if old_focus is None else old_focus.to_dict(),
                      "event_cap": {"size": size, "radius": 8.0 * lc.r_n, "z": lc.z_tip_new,
                                    "h_floor": h_floor, "h_cut": h_cut}}
    old_chains = [np.column_stack([segs[name]["r"], segs[name]["z"]]) for name, _ in named]
    stage("park")
    remeshing = False
    try:
        template.focus = Focus(size=min(g.size for g in regions),
                               radius=max(g.radius for g in regions), grading=grading,
                               regions=regions)
        park_plan(pb, lp.plan, lp.table, rmin=lc.r_n)
        stage("remesh")
        remeshing = True
        with captured_stdout() as printed:
            pb.force_remesh({template})
        pb._domains_to_remesh.clear()
    except BaseException as e:
        clear_plan(pb)
        if not remeshing:  # nothing has changed but the focus and the plan: put them back
            template.focus = old_focus
        if not isinstance(e, Exception):  # an interruption is no outcome of the cut
            raise
        if not remeshing:
            raise CutRefused(f"the plan could not be parked ({type(e).__name__}: {e})") from e
        raise CutFailed(f"the remesh of the liquid cut failed ({type(e).__name__}: {e})",
                        record["dumps"]["precut"]) from e

    stage("mesh-check")
    transfer = transfer_warnings("".join(printed))
    record["transfer"] = transfer
    new_named, new_segs = _named_chains(pb)
    ends = {name: c.end_types for name, c in named}
    record["mesh"]["chains"] = [{"chain": name, "nodes": len(c.points),
                                 "end_types": list(c.end_types)} for name, c in new_named]
    if {name: c.end_types for name, c in new_named} != ends:
        raise CutFailed(f"the liquid cut's mesh has the chains {record['mesh']['chains']}, not "
                        f"those of the plan")
    spans = _axis_spans_of_mesh(pb)
    expected = liquid_axis_spans([z for _, c in new_named for _, z in c.axis_ends()])
    record["mesh"].update(axis_spans=spans, expected_axis_spans=[list(x) for x in expected],
                          nodes=pb.get_mesh("liquid").nnode())
    if len(spans) != len(expected) or any(abs(a - b) > 1e-9 * max(1.0, pb.nd.radius)
                                          for x, y in zip(spans, expected) for a, b in zip(x, y)):
        raise CutFailed(f"the liquid cut's mesh has the axis spans {spans}, expected {expected}")
    nodes = mesh_nodes(pb.get_mesh(INTERFACE))
    X = np.array([[n.x(0), n.x(1)] for n in nodes]).reshape(-1, 2)
    predicted = predict_unlocated(X, old_chains)
    on_cap = _on_cap(lc, X, new_segs[lc.chain])
    reported = transfer["unlocated"].get(INTERFACE, [0, 0])[0]
    transfer.update(predicted_unlocated=int(predicted.sum()), reported_unlocated=int(reported),
                    predicted_off_cap=int(np.sum(predicted & ~on_cap)),
                    cap_nodes=int(on_cap.sum()),
                    corner_fields=sorted(pb.get_mesh(INTERFACE + "/axis")
                                         .get_nodal_field_indices()))
    problems = transfer_problems(transfer)
    if problems:
        raise CutFailed("the transfer of the liquid cut is not as planned: " + "; ".join(problems))
    global_transfer = pb.cut_observables()
    poly_after = box_polyline_budgets([c for _, c in new_named], lc.z_n, R_b, s.reservoir_depth)

    stage("fill")
    record["cap_fill"] = _fill_cap(pb, lc, samples, new_segs[lc.chain],
                                   apply=ls.cap_fill == "sample")

    stage("projection")
    record["projection"] = {}
    settings = replace(s, dt0=dt_prev, restart=ls.restart or s.restart, projection=ls.projection)
    project(pb, settings, t_now, record["projection"], box=(lc.z_n, R_b))
    stage("chart")
    orient_interface_chart(pb)

    stage("postcut")
    global_after = pb.cut_observables()
    box = record["projection"].get("box")
    if box is not None:  # the box before and after the projection, from its own samples
        box_transfer, box_after = box["before"], box["after"]
    else:
        box_transfer = box_after = box_integrals(pb, grid)
    box20_after = box_integrals(pb, grid20)
    record["budgets"] = {
        "box": {"z": lc.z_n, "R": R_b, "n": grid["n"]},
        "polylines": {"before": poly_before, "after": poly_after},
        "grid": {"before": box_before, "after_transfer": box_transfer,
                 "after_projection": box_after},
        "box20": {"before": box20_before, "after": box20_after},
        "global": {"before": global_before, "after_transfer": global_transfer,
                   "after": global_after}}
    closures = _closures(removed, record["budgets"]["polylines"],
                         {"before": box_before, "after_transfer": box_transfer,
                          "after_projection": box_after})
    closures["F3_phi"] = {x: (box or {}).get(x, math.nan)
                          for x in ("phi_lhs", "phi_rhs", "phi_error", "phi_relative_error")}
    record["closures"] = closures
    if closures["hard"]:
        raise CutFailed(f"the budget of the liquid cut does not close: {closures['hard']} beyond "
                        f"their limits ({ {x: closures[x] for x in closures['hard']} })")
    # the removed head as the run is to follow it: moving rigidly at its mean axial speed, its rear
    # (the cap's side) retracting towards it at the capillary speed of the end of a thread
    _, segs_after = _named_chains(pb)
    u_tip = float(segs_after[lc.chain]["u_z"][0 if lc.end == 0 else -1])
    record["ghost"] = {"V": lc.V_rem, "Pz": removed["Pz"], "KE": removed["KE"],
                       "U": removed["U"], "a": (0.75 * lc.V_rem / math.pi) ** (1.0 / 3.0),
                       "v_ret": lc.r_n ** -0.5, "direction": lc.direction, "t0": t_now,
                       "z_front0": lc.z_tip_old, "z_rear0": lc.z_tip_new,
                       "du_stump": u_tip - removed["U"], "status": "free"}
    # what the removals have taken from the liquid since the gas cut, this one included
    before = [e for e in (getattr(pb, "events", None) or [])
              if e.get("kind") in EVENT_CLASSES["liquid"]]
    record["removed_total"] = {"events": len(before) + 1, **{
        key: sum(float((e.get("removed") or {}).get(key, 0.0)) for e in before) + removed[key]
        for key in ("V", "Pz", "KE")}}
    record["outcome"] = "made"
    record["wall_seconds"] = time.time() - wall0
    made = _jsonable(record)
    pb.events = list(getattr(pb, "events", None) or []) + [made]
    if on_made is not None:
        on_made(made)
    made["dumps"]["postcut"] = _keep(pb, driver, out, f"postcut_e{k:03d}",
                                     {"cut_stage": "postcut-event", "event_k": k})["path"]
    pb.events[-1] = made
    full = dict(made, removed=dict(made["removed"], polygon=lc.removed.tolist()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(full), indent=1) + "\n")
    rel = ", ".join(f"{x} {closures['relative'][x]:.3g}" for x in BUDGET_LIMITS)
    log(f"liquid cut {k} at the {lc.chain} neck (r_n {lc.r_n:.6g}, r_n/s {lc.r_n / lc.s:.3g}): "
        f"removed {lc.V_rem:.6g} ({lc.V_rem / lc.s**3:.3g} s^3), P_z {removed['Pz']:.6g}; "
        f"closures relative to the removal {rel}")
    return made
