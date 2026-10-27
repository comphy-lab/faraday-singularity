"""Small finite-element tests of the reconnection (``surgery.py``, ``mesh.py``, ``problem.py``).

* pins of the private pyoomph names the cut relies on;
* the pressure-impulse projection on a manufactured field: ``u~ = u_s + grad(phi)`` with ``u_s``
  solenoidal and admissible and ``phi = 0`` on the free top, so that ``u+ -> u_s``, ``Pi -> phi``
  and ``KE(u~) - KE(u+) = |u+ - u~|^2/2``;
* a cut of a synthetic necked gas cavity in a shallow container (a "kinematic" cut: the velocity
  field is imposed, not computed): topology, volumes, transfer, projection, history, chart, a few
  steps, a restart from the ``postcut`` state, a quality remesh of the two chains after the cut and
  a step after it, the remesh after a restart in a new process, the zero fill on every node of the
  former gas, and the zero fill with an impulsive restart;
* the collapse driver after a cut of that cavity, whose rebuilds of the mesh take effect and stop
  although the tips keep the cut's sizing;
* the refusal and the failure of a cut, which leave their partial record in
  ``cut_diagnostics.json``;
* the prototype entry point on a landing state of that cavity, with the lineage of its parent run,
  the pinch time fitted to the neck samples of the landing state, and a failing cut;
* the production runner: the cut at the start of a restart from a landing state of a case with the
  default dump cadence, continuations from its first rotating dump, its ``postcut`` dump and the
  state it ended on at ``tau_end`` (under a later ``tau_end``), and failures of the cut at several
  stages, each recorded under ``cut`` in ``run_status.json``;
* the ordinary path: meshes as with the mesh class of commit 9e0d8a1 (to the byte), steps
  (to round-off), dumps that load across the two kinds of problem, a short run of the collapse
  driver against that of 9e0d8a1, every output compared (to round-off, dump keys exactly), and the
  same run with and without its dumps (to the byte).

Each problem compiles in seconds; the module takes a little over three minutes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pyoomph = pytest.importorskip("pyoomph")

from faraday_singularity import surgery  # noqa: E402
from faraday_singularity.driver import CollapseDriver, CollapseSettings  # noqa: E402
from faraday_singularity.mesh import FaradayCylinderMesh, Focus, MeshSettings  # noqa: E402
from faraday_singularity.parameters import CaseParameters, nondimensionalise  # noqa: E402
from faraday_singularity.problem import FaradayProblem  # noqa: E402
from faraday_singularity.reconnect import Chain, ReconnectSettings, neck_radius  # noqa: E402

R_N = 0.02  # neck radius of the synthetic cavity
# the cut of the synthetic cavity in most tests: its mesh sizes, a quarter of the cut radius, are
# about those of the cavity's mesh, so that the tests stay small
FAST_CUT = {"delta": 1.05 * R_N, "dt0": 1e-5, "tip_size": 0.25, "bridge_size": 0.25,
            "focus_size": 0.25}


# --- private names ------------------------------------------------------------------------------


def test_private_pyoomph_names_the_cut_relies_on():
    import inspect

    from pyoomph.equations.topological_changes import (
        AxisymmetricReconnection,
        TopologicalChangesGmshTemplate,
    )
    from pyoomph.generic.codegen import EquationTree
    from pyoomph.generic.problem import Problem
    from pyoomph.meshes import axisymm_topology as topo

    assert hasattr(EquationTree, "_on_this_domain")
    h = surgery.DriverTriggeredReconnection(reservoir_depth=3.0, cap_window_factor=6.0)
    for name in ("_nondim_done", "_pending_on", "_deferred_events", "_unplannable_states"):
        assert hasattr(h, name), name
    assert h.rmin is None and h.distmin is None and h.volume_conservation is False
    assert h.check_mesh_motion_direction is False and h.reservoir_depth == 3.0
    for name in ("_clear_pending_surgery_plan", "_set_pending_surgery_plan",
                 "has_pending_surgery_plan", "get_reconnected_boundaries"):
        assert hasattr(TopologicalChangesGmshTemplate, name), name
    t = FaradayCylinderMesh(3.8, 2.0, lambda r: 0 * r, MeshSettings())
    assert t._pending_surgery_plans == {} and t.point_snap_tolerance == 0.0
    assert "_domains_to_remesh" in inspect.getsource(Problem.__init__)
    hook = inspect.getsource(AxisymmetricReconnection.after_newton_solve)
    for needle in ("_clear_pending_surgery_plan", "_set_pending_surgery_plan",
                   "_domains_to_remesh.add", "ensure_nondimensional_distance_parameters"):
        assert needle in hook, needle
    detect = inspect.getsource(AxisymmetricReconnection._detect_from_data)
    assert "except WaistNotYetSeparable" in detect and "except InterfaceStateNotPlannable" in detect
    assert issubclass(topo.WaistNotYetSeparable, RuntimeError)
    assert issubclass(topo.InterfaceStateNotPlannable, RuntimeError)
    # after a cut: pyoomph's net for an element turned inside out within a step, armed at run time
    # (FaradayProblem.arm_inverted_element_remesh), and the export the curved-element check reads
    from pyoomph import _pyoomph_core

    assert hasattr(_pyoomph_core, "set_detect_inverted_elements")
    assert "_domains_remesh_on_inversion" in inspect.getsource(Problem.__init__)
    assert "self._domains_remesh_on_inversion" in inspect.getsource(
        Problem._solve_with_inversion_remesh)
    assert "force_remesh(self._domains_remesh_on_inversion)" in inspect.getsource(
        Problem._solve_with_inversion_remesh)


# --- the projection on a manufactured field ---------------------------------------------------------


def _manufactured():
    from pyoomph.expressions import cos, pi, sin, var, vector

    r, z = var("coordinate_x"), var("coordinate_y")
    us = vector(-2 * z * r * (1 - r) ** 2, 2 * z**2 * (1 - r) * (1 - 2 * r))  # solenoidal
    phi = cos(pi * r) * cos(pi * z / 2)  # zero on the top z = 1
    gphi = vector(-pi * sin(pi * r) * cos(pi * z / 2), -pi / 2 * cos(pi * r) * sin(pi * z / 2))
    return us, gphi, phi


class ManufacturedProblem(pyoomph.Problem):
    """The unit square ``0 <= r, z <= 1``: axis on the left, free slip on the right and bottom, free
    top; the projection's history field is ``u_s + grad(phi)``."""

    def __init__(self, N: int):
        super().__init__()
        self.N = N
        self._projecting = False
        self.cut = None

    def define_problem(self):
        from pyoomph.equations.generic import (
            AxisymmetryBC,
            DirichletBC,
            InitialCondition,
            IntegralObservables,
        )
        from pyoomph.equations.navier_stokes import NavierStokesEquations
        from pyoomph.expressions import div, dot, evaluate_in_past, var
        from pyoomph.meshes.simplemeshes import RectangularQuadMesh

        self.set_coordinate_system("axisymmetric")
        self.add_mesh(RectangularQuadMesh(N=self.N, size=[1, 1], name="liquid"))
        us, gphi, phi = _manufactured()
        u, p = var("velocity"), var("pressure")
        u1 = evaluate_in_past(u, 1)
        eqs = NavierStokesEquations(mass_density=1, dynamic_viscosity=1)
        eqs += InitialCondition(velocity=us + gphi)
        eqs += AxisymmetryBC() @ "left"
        eqs += DirichletBC(velocity_x=0) @ "right"
        eqs += DirichletBC(velocity_y=0) @ "bottom"
        eqs += surgery.PressureImpulseProjection()
        eqs += IntegralObservables(volume=1, kinetic_energy=dot(u, u) / 2, momentum_z=u[1],
                                   divergence_sq=div(u) ** 2,
                                   projection_change=dot(u - u1, u - u1) / 2,
                                   err_us=dot(u - us, u - us), err_pi=(p - phi) ** 2,
                                   gphi_energy=dot(gphi, gphi) / 2)
        eqs += IntegralObservables(area=1, pressure_sq=p**2) @ "top"
        self.add_equations(eqs @ "liquid")

    def cut_observables(self):
        bulk, top = self.get_mesh("liquid"), self.get_mesh("liquid/top")
        out = {n: float(bulk.evaluate_observable(n)) for n in
               ("volume", "kinetic_energy", "momentum_z", "divergence_sq", "projection_change")}
        out.update({n: float(top.evaluate_observable(n)) for n in ("area", "pressure_sq")})
        return out


@pytest.fixture(scope="module")
def manufactured(tmp_path_factory):
    res = {}
    for N in (8, 16):
        pb = ManufacturedProblem(N)
        pb.set_output_directory(str(tmp_path_factory.mktemp(f"proj{N}")))
        with pb:
            pb.initialise()
            pb.set_initial_condition()
            t0 = pb.get_current_time(dimensional=False, as_float=True)
            s = ReconnectSettings(delta=1.0, dt0=0.25, restore_pressure=False)
            out = surgery.project(pb, s, t0)
            bulk = pb.get_mesh("liquid")
            ip = bulk.get_nodal_field_indices()["velocity_y"]
            node = list(bulk.nodes())[len(list(bulk.nodes())) // 2]
            out.update(err_us=float(bulk.evaluate_observable("err_us")),
                       err_pi=float(bulk.evaluate_observable("err_pi")),
                       gphi_energy=float(bulk.evaluate_observable("gphi_energy")),
                       time=pb.get_current_time(dimensional=False, as_float=True), t0=t0,
                       counter=pb.timestepper.get_num_unsteady_steps_done(),
                       levels=[node.value_at_t(k, ip) for k in range(3)],
                       dt=pb.timestepper.time_pt().dt(0))
            res[N] = out
    return res


def test_projection_energy_identity_and_convergence(manufactured):
    for out in manufactured.values():
        ke = out["before"]["kinetic_energy"]
        assert out["newton_residuals"][-1] < 1e-10
        assert abs(out["energy_identity_error"]) <= 1e-12 * ke
        assert out["delta_KE"] > 0
    a, b = manufactured[8], manufactured[16]
    assert b["err_us"] < a["err_us"] / 3  # u+ -> u_s
    assert b["err_pi"] < a["err_pi"] / 3  # Pi -> phi
    # the energy removed tends to |grad phi|^2/2, only as h -> 0
    exact = b["gphi_energy"]
    assert abs(b["delta_KE"] - exact) < abs(a["delta_KE"] - exact)
    # Pi vanishes weakly on the free top
    assert b["Pi_rms_on_surface"] < a["Pi_rms_on_surface"] / 2
    assert b["Pi_rms_on_surface"] < 1e-2 * b["max_abs_Pi"]


def test_the_strong_divergence_is_a_discretisation_diagnostic(manufactured):
    """``u+`` is divergence-free weakly (its residual, which holds ``int q div u+`` for every P1
    pressure, converges to round-off); the L2 norm of its pointwise divergence is not zero and falls
    at about first order in the mesh size (observed order 0.83 here)."""
    a, b = manufactured[8], manufactured[16]
    for out in (a, b):
        assert "divergence_L2_after" not in out
        assert out["strong_divergence_L2_after"] > 1e8 * out["newton_residuals"][-1]
        assert out["strong_divergence_L2_after"] < out["strong_divergence_L2_before"]
    ratio = a["strong_divergence_L2_after"] / b["strong_divergence_L2_after"]
    assert 1.5 < ratio < 2.5  # first order, not second


def test_projection_restores_time_and_history(manufactured):
    for out in manufactured.values():
        assert out["time"] == out["t0"] and out["counter"] == 0
        assert out["dt"] == 0.25
        assert out["levels"][0] == out["levels"][1] == out["levels"][2]


# --- a synthetic necked cavity --------------------------------------------------------------------


def neck_interface(R, z_n=0.6, r_n=R_N, beta=12.0, L=0.25):
    """A gas boundary from the axis to the wall: spherical bubble bottom, conical thread through a
    hyperbolic neck of radius ``r_n`` at ``z_n``, a funnel, and a nearly flat surface."""
    tb = math.tan(math.radians(beta))

    def cone(z):
        return np.sqrt(r_n**2 + ((z - z_n) * tb) ** 2)

    z_j = z_n - L
    a = float(cone(z_j))
    th = np.linspace(0, 0.5 * math.pi, 2000)
    bottom = np.column_stack([a * np.sin(th), z_j - a * np.cos(th)])
    zz = np.linspace(z_j, z_n + 0.4, 200001)[1:]
    rr = cone(zz) + 8.0 * np.maximum(0, zz - z_n - 0.03) ** 2
    k = int(np.argmax(rr >= 0.3))
    zz, rr = zz[:k + 1], rr[:k + 1]
    flat_r = np.linspace(rr[-1], R, 20000)[1:]
    flat = np.column_stack([flat_r, zz[-1] + 0.02 * (flat_r - rr[-1]) / (R - rr[-1])])
    P = np.vstack([bottom, np.column_stack([rr, zz]), flat])
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    out, si = [P[0]], 0.0
    while True:
        d = math.hypot(out[-1][0] - r_n, out[-1][1] - z_n)
        si += min(0.03, max(r_n / 4, 0.15 * d))
        if si >= s[-1]:
            break
        out.append(np.array([np.interp(si, s, P[:, 0]), np.interp(si, s, P[:, 1])]))
    out.append(P[-1].copy())
    Q = np.array(out)
    Q[0, 0], Q[-1, 0] = 0.0, R
    return [(float(x), float(y)) for x, y in Q]


class NeckMesh(FaradayCylinderMesh):
    def _interface_points(self, segs=None):
        if self.is_first_time():
            return neck_interface(self.radius)
        return super()._interface_points(segs)


class NeckProblem(FaradayProblem):
    mesh_class = NeckMesh


PHYSICS = {"fluid": {"Oh": 0.05, "Bo": 0.0}, "container": {"mode": 1, "depth": 1.0},
           "forcing": {"Gamma": 0.0, "detuning": 1.0, "phase": 0.0}, "initial": {"amplitude": 0.0}}
MESH = {"h_interface": R_N / 4, "h_axis": R_N / 4, "h_max": 0.25, "grading": 0.3,
        "interface_points": 401}


def neck_problem(outdir, settings):
    nd = nondimensionalise(CaseParameters.from_dict(PHYSICS))
    pb = NeckProblem(nd, mesh=MeshSettings(**MESH), reconnect=settings)
    pb.set_output_directory(str(outdir))
    return pb


def impose_closing_flow(pb):
    """A smooth flow that closes the neck radially and stretches it axially (not solenoidal)."""
    m = pb.get_mesh("liquid")
    fi = m.get_nodal_field_indices()
    seg = pb.interface_segments()[0]
    _, z_n, _ = neck_radius(seg["r"], seg["z"])
    for n in m.nodes():
        x, y = n.x(0), n.x(1)
        g = math.exp(-((y - z_n) / 0.05) ** 2) * math.exp(-((x / 0.3) ** 2))
        n.set_value(fi["velocity_x"], -5.0 * x / (x + 0.02) * g)
        n.set_value(fi["velocity_y"], 2.0 * (y - z_n) * g)
    pb.reapply_boundary_conditions()
    pb.assign_initial_values_impulsive()


def _state_vector(pb):
    return np.concatenate([np.concatenate([c["r"], c["z"], c["u_r"], c["u_z"]])
                           for c in pb.interface_segments()])


def _step(pb, dt):
    t = pb.get_current_time(dimensional=False, as_float=True)
    pb.run(t + dt, timestep=dt, maxstep=dt, outstep=False, temporal_error=1.0,
           newton_solver_tolerance=1e-8)


def mesh_signature(pb) -> dict:
    """Node and element counts of the liquid mesh and a hash of its sorted node coordinates."""
    bulk = pb.get_mesh("liquid")
    xy = np.array(sorted((n.x(0), n.x(1)) for n in bulk.nodes()))
    return {"nodes": len(xy), "elements": sum(1 for _ in bulk.elements()),
            "sha256": hashlib.sha256(xy.tobytes()).hexdigest()}


def chart_ends(pb) -> dict:
    """The interface chart at the two ends of each chain."""
    bulk = pb.get_mesh("liquid")
    bind = bulk.get_boundary_index("interface")
    pos_zeta = {(n.x(0), n.x(1)): n.get_coordinates_on_boundary(bind)[0]
                for n in surgery.mesh_nodes(pb.get_mesh("liquid/interface"))}
    return {("open" if c["open"] else "bubble"):
            (pos_zeta[(c["r"][0], c["z"][0])], pos_zeta[(c["r"][-1], c["z"][-1])])
            for c in pb.interface_segments()}


@pytest.fixture(scope="module")
def neck_cut(tmp_path_factory):
    out = tmp_path_factory.mktemp("neck")
    s = ReconnectSettings(**FAST_CUT)
    res = {}
    with neck_problem(out, s) as pb:
        pb.initialise()
        impose_closing_flow(pb)
        res["neck"] = neck_radius(pb.interface_segments()[0]["r"], pb.interface_segments()[0]["z"])
        # the local kinetic energy over a half-disc that holds all the imposed flow
        res["ke_total"] = pb.cut_observables()["kinetic_energy"]
        res["ke_wide"] = surgery.local_kinetic_energy(pb, res["neck"][1], 1.0)
        old = pb.interface_segments()[0]
        old_open = Chain(np.column_stack([old["r"], old["z"]]), tuple(old["end_types"]))
        rec = surgery.cut(pb, s, out)
        res["record"] = rec
        res["file"] = json.loads((out / "cut_diagnostics.json").read_text())
        res["counter"] = pb.timestepper.get_num_unsteady_steps_done()
        res["time"] = pb.get_current_time(dimensional=False, as_float=True)
        res["dt"] = pb.timestepper.time_pt().dt(0)
        bulk = pb.get_mesh("liquid")
        iu = bulk.get_nodal_field_indices()["velocity_x"]
        nodes = list(bulk.nodes())
        res["levels"] = [[n.value_at_t(k, iu) for k in range(3)] for n in nodes[::97]]
        zeta = pb.interface_zeta
        res["chart_rule"] = (zeta.sort_along_axis, zeta.start_near_point)
        res["chart_ends"] = chart_ends(pb)
        res["bridge0"] = rec["geometry"]["bridge"]
        # a few steps, and the same first step again from the postcut state
        _step(pb, 1e-5)
        res["after_first"] = _state_vector(pb)
        post = Path(rec["postcut_state"]["path"])
        res["postcut_meta"] = json.loads(post.with_suffix(".json").read_text())
        pb.load_state(str(post))
        _step(pb, 1e-5)
        res["after_first_again"] = _state_vector(pb)
        for k in range(2):
            _step(pb, 1e-5 * 1.25 ** (k + 1))
        segs = pb.interface_segments()
        res["chains_after"] = len(segs)
        cav = next(c for c in segs if c["open"])
        bub = next(c for c in segs if not c["open"])
        res["bridge_after"] = float(cav["z"][0] - bub["z"][-1])
        res["axis_state"] = pb.axis_state()
        res["open_tip"] = (float(cav["z"][0]), float(cav["u_z"][0]))
        # a remesh of the two chains, as a quality remesh after the cut would make, and a step
        before = pb.cut_observables()
        spans_before = surgery._axis_spans_of_mesh(pb)
        pb.force_remesh()
        after = pb.cut_observables()
        res["quality_remesh"] = {
            "chains": len(pb.interface_segments()), "spans": (len(spans_before),
                                                              len(surgery._axis_spans_of_mesh(pb))),
            "volume_change": after["volume"] - before["volume"], "volume": before["volume"],
            "ke_change": after["kinetic_energy"] / before["kinetic_energy"] - 1.0,
            "chart_ends": chart_ends(pb)}
        _step(pb, 1e-5 * 1.25 ** 3)
        res["quality_remesh"]["chains_after_step"] = len(pb.interface_segments())
        # the remesh right after an in-process restart from the postcut state, for the comparison
        # with one in a new process
        pb.load_state(str(post))
        pb.force_remesh()
        res["restart_remesh"] = mesh_signature(pb)
        res["postcut_path"] = str(post)
        # again from the precut state, with the zero fill and an impulsive restart
        pb.load_state(rec["precut_state"]["path"])
        pb.cut = None
        s2 = ReconnectSettings(**dict(FAST_CUT, fill="zero", restart="impulsive"))
        out2 = out / "zero"
        out2.mkdir()
        res["zero"] = surgery.cut(pb, s2, out2)
        res["zero_counter"] = pb.timestepper.get_num_unsteady_steps_done()
        _step(pb, 1e-5)
        res["zero_chains_after"] = len(pb.interface_segments())
        # the zero fill on the blended postcut field: every node in the former gas but on the free
        # surface, the axis of the bridge included, found here without the fill's own tests
        pb.load_state(str(post))
        geo = rec["geometry"]
        filled = surgery.zero_fill(pb, old_open, s.reservoir_depth, (0.0, rec["z_neck"]), 1e3)
        import shapely
        from shapely.geometry import Polygon

        P = old_open.points
        gas = Polygon(np.vstack([P, [[P[-1, 0], P[-1, 1] + s.reservoir_depth],
                                     [0.0, P[-1, 1] + s.reservoir_depth]]]))
        bulk = pb.get_mesh("liquid")
        fi = bulk.get_nodal_field_indices()
        surface = {(n.x(0), n.x(1)) for n in surgery.mesh_nodes(pb.get_mesh("liquid/interface"))}
        bridge = [n for n in surgery.mesh_nodes(pb.get_mesh("liquid/axis"))
                  if geo["z_bubble_top"] < n.x(1) < geo["z_cavity_tip"]
                  and (n.x(0), n.x(1)) not in surface]
        inner = [n for n in bulk.nodes() if (n.x(0), n.x(1)) not in surface and n.x(0) > 0.0
                 and shapely.contains_xy(gas, n.x(0), n.x(1))]
        res["zero_check"] = {
            "counts": filled, "bridge_nodes": len(bridge), "inner_nodes": len(inner),
            "max_speed": max(math.hypot(n.value(fi["velocity_x"]), n.value(fi["velocity_y"]))
                             for n in bridge + inner)}
    return res


def test_cut_plans_once_and_builds_two_chains(neck_cut):
    rec = neck_cut["record"]
    assert rec == neck_cut["file"] and rec["outcome"] == "cut"
    assert [a["outcome"] for a in rec["plan_attempts"]] == ["plan"]
    assert rec["plan"]["events"][0]["kind"] == "pinch"
    chains = rec["mesh"]["chains"]
    assert sorted(c["open"] for c in chains) == [False, True]
    assert len(rec["mesh"]["axis_spans"]) == 2
    for got, want in zip(rec["mesh"]["axis_spans"], rec["mesh"]["expected_axis_spans"]):
        assert got == pytest.approx(want, abs=1e-12)
    g = rec["geometry"]
    # on this conical neck (12 degrees) the planner's window gives a bridge of about 10.6 cut radii
    # and 86 cut radii cubed of removed gas
    assert g["z_bubble_top"] < g["z_cavity_tip"] and 8 * rec["rmin"] < g["bridge"] < 14 * rec["rmin"]
    assert 60 * rec["rmin"] ** 3 < g["removed_gas"] < 110 * rec["rmin"] ** 3
    for cap in rec["caps"]:
        assert cap["min_angle"] >= 30.0 and cap["r_s"] >= rec["delta"]


def test_cut_records_the_geometry_in_cut_radii(neck_cut):
    rec = neck_cut["record"]
    g, rmin = rec["in_cut_radii"], rec["rmin"]
    assert g["cut_radius"] == rmin and g["cap_window_factor"] == 6.0
    assert g["r_s"] == pytest.approx([c["r_s"] / rmin for c in rec["caps"]])
    assert g["replaced_survivors"] == [c["replaced_survivors"] for c in rec["caps"]]
    assert g["splice_set_by"] == [c["splice_set_by"] for c in rec["caps"]]
    assert set(g["splice_set_by"]) <= {"planner-window", "delta-floor"}
    assert "cap_splice_factor" not in g and "cap_splice_factor" not in rec["settings"]
    assert g["bridge"] == pytest.approx(rec["geometry"]["bridge"] / rmin)
    assert g["removed_gas"] == pytest.approx(rec["geometry"]["removed_gas"] / rmin**3)


def _bottom_spacing(pb, radius: float) -> float:
    """The longest element (twice the node spacing) of the bubble's chain within ``radius`` of its
    bottom tip; before a cut, of the free surface within ``radius`` of its tip on the axis."""
    segs = pb.interface_segments()
    c = next((c for c in segs if not c["open"]), segs[0])
    r, z = np.asarray(c["r"]), np.asarray(c["z"])
    inside = np.hypot(r, z - z[0]) <= radius
    return float(2 * np.hypot(np.diff(r), np.diff(z))[inside[:-1] & inside[1:]].max())


def test_the_cut_mesh_is_sized_in_cut_radii_whatever_the_landing_focus(tmp_path):
    """The tips, the bridge and the focus of the cut's remesh are fixed fractions of the cut radius:
    two landing states that differ only in a focus coarser than the cut's are cut onto the same
    mesh. A landing focus finer than the cut's is kept on the bubble's bottom, which the landing
    state held as the cavity's bottom (``focus.kept``, within its half of the bubble), so that the
    cut does not coarsen it there, while the open chain is meshed as after a coarse landing."""
    meshes, records, opens, bottoms = [], [], [], []
    landings = (Focus(size=R_N / 3, radius=0.1, grading=0.3),
                Focus(size=R_N / 2.5, radius=0.6, grading=0.3),
                Focus(size=R_N / 12, radius=0.6, grading=0.3))
    for k, landing in enumerate(landings):
        out = tmp_path / f"cut{k}"
        s = ReconnectSettings(**FAST_CUT)
        with neck_problem(out, s) as pb:
            pb.initialise()
            impose_closing_flow(pb)
            pb.active_mesh_template().focus = landing
            rec = surgery.cut(pb, s, out)
            meshes.append(mesh_signature(pb))
            records.append(rec)
            opens.append(next(np.column_stack([c["r"], c["z"]]) for c in pb.interface_segments()
                              if c["open"]))
            kept = rec["focus"]["kept"]
            bottoms.append(_bottom_spacing(pb, kept["radius"]))
            assert pb.active_mesh_template().focus == Focus(**rec["focus"]["cut"])
    assert meshes[0] == meshes[1] and meshes[2] != meshes[0]
    assert np.array_equal(opens[0], opens[2])  # the open chain, near the cut, as after either
    for rec, landing in zip(records, landings):
        rmin = rec["rmin"]
        h_cut = 0.25 * rmin
        assert rec["focus"]["landing"] == landing.to_dict()
        cut = rec["focus"]["cut"]
        assert (cut["size"], cut["radius"], cut["grading"]) == (
            pytest.approx(min(h_cut, landing.size)), pytest.approx(20 * rmin), 0.3)
        regions = {g["name"]: g for g in cut["regions"]}
        assert regions["cut"]["axial"] and regions["cut"]["size"] == pytest.approx(h_cut)
        assert regions["cut"]["radius"] == pytest.approx(20 * rmin)
        kept = rec["focus"]["kept"]
        assert kept["size"] == pytest.approx(min(h_cut, landing.size))
        g = rec["geometry"]
        assert kept["radius"] == pytest.approx(min(landing.radius,
                                                   0.5 * (g["z_bubble_top"] - g["z_bubble_bottom"])))
        assert regions["bubble_bottom"]["size"] == kept["size"]
        assert regions["bubble_bottom"]["z"] == g["z_bubble_bottom"]
        assert rec["sizing"]["h_tip"] == pytest.approx(0.25 * rmin)
        assert rec["sizing"]["h_bridge"] == pytest.approx(0.25 * rmin)
        assert rec["sizing"]["curvature"] == 0  # gmsh's sizing from curvature is off
        assert rec["sizing_in_cut_radii"] == {"h_tip": 0.25, "h_bridge": 0.25, "focus_size": 0.25,
                                              "focus_radius": 20.0}
    # the bottom is meshed at the finer landing focus there, at the cut's otherwise
    assert bottoms[2] <= 1.25 * landings[2].size < 0.5 * bottoms[0]
    assert bottoms[0] == pytest.approx(0.25 * records[0]["rmin"], rel=0.25)


def test_cut_records_the_local_kinetic_energy(neck_cut):
    """The kinetic energy within 20 delta of the neck, at the four stages of the cut; sampled over
    a half-disc that holds all the flow it is the global integral, to the sampling error."""
    assert neck_cut["ke_wide"] == pytest.approx(neck_cut["ke_total"], rel=1e-2)
    rec = neck_cut["record"]
    loc = rec["local_budgets"]
    assert loc["radius"] == pytest.approx(surgery.LOCAL_RADIUS * rec["delta"])
    assert loc["centre"] == [0.0, rec["z_neck"]]
    ke = loc["kinetic_energy"]
    assert set(ke) == {"before", "after_transfer", "after_fill", "after"}
    assert 0.5 * rec["budgets_before"]["kinetic_energy"] < ke["before"] \
        <= rec["budgets_before"]["kinetic_energy"]
    assert ke["after"] <= rec["budgets_after"]["kinetic_energy"] and ke["after"] < ke["after_fill"]
    # without a driver the cut has no neck samples, so no pinch time
    assert rec["pinch"]["t_p"] is None and rec["pinch"]["source"] is None


def test_cut_volume_is_the_planned_gain(neck_cut):
    """The gas the cut turns into liquid within a box around it is the planned gain; the change of
    the whole mesh adds the drift of rebuilding the free surface, which at production radii exceeds
    the gain (the box separates the two)."""
    v = neck_cut["record"]["volumes"]
    assert v["liquid_gain_plan"] > 0
    assert v["liquid_gain_box"] == pytest.approx(v["liquid_gain_plan"], rel=2e-3)
    assert v["liquid_gain_mesh"] == pytest.approx(v["liquid_gain_box"] + v["remesh_drift"],
                                                  rel=1e-12)
    assert v["liquid_gain_mesh"] == pytest.approx(v["liquid_gain_plan"], rel=1e-2)
    r_box, z_lo, z_hi = v["box"]
    caps = neck_cut["record"]["caps"]
    assert z_lo < min(c["z_s"] for c in caps) and z_hi > max(c["z_s"] for c in caps)
    assert r_box > max(c["r_s"] for c in caps)


def test_cut_transfer_falls_back_only_in_the_former_gas(neck_cut):
    tr = neck_cut["record"]["transfer"]
    gas = tr["former_gas_nodes"]
    unloc = tr["unlocated"]
    # pyoomph's fallback counts match the nodes of the new mesh in the old gas (tolerance band)
    assert unloc["liquid"][0] == pytest.approx(gas["liquid"][0], rel=0.1)
    assert unloc["liquid/interface"][0] == pytest.approx(gas["liquid/interface"][0], rel=0.2)
    assert unloc["liquid/axis"][0] == gas["liquid/axis"][0]
    # the two fresh tips get no value of the interface-only field; it is _kin_bc
    assert tr["no_value"]["liquid/interface/axis"][0] == 2
    assert "_kin_bc" in tr["corner_fields"]


def test_cut_projection_and_history(neck_cut):
    pr = neck_cut["record"]["projection"]
    ke = pr["before"]["kinetic_energy"]
    assert pr["newton_residuals"][-1] <= 1e-10
    assert abs(pr["energy_identity_error"]) <= 1e-10 * ke and pr["delta_KE"] > 0
    assert pr["Pi_rms_on_surface"] < 1e-3 * pr["max_abs_Pi"]
    # the axial momentum the projection removes is the impulse on the bottom, to discretisation
    assert pr["delta_Pz"] == pytest.approx(pr["Pi_on_bottom"], rel=0.05)
    assert neck_cut["counter"] == 0 and neck_cut["dt"] == 1e-5
    assert neck_cut["time"] == neck_cut["record"]["t_cut"]
    for lv in neck_cut["levels"]:
        assert lv[0] == lv[1] == lv[2]


def test_cut_chart_runs_from_the_wall_and_up_the_bubble(neck_cut):
    assert neck_cut["chart_rule"] == (None, surgery.FAR_POINT)
    ends = neck_cut["chart_ends"]
    assert ends["open"][0] > ends["open"][1]  # charted from the wall end
    assert ends["bubble"][0] < ends["bubble"][1]  # charted upwards


def test_quality_remesh_after_the_cut(neck_cut):
    """A remesh of the two chains after the cut (the path of pyoomph's quality remesh) keeps both
    chains, the two liquid spans of the axis and the chart orientation, changes volume and energy
    little, and a step converges after it."""
    q = neck_cut["quality_remesh"]
    assert q["chains"] == 2 and q["spans"] == (2, 2) and q["chains_after_step"] == 2
    assert abs(q["volume_change"]) < 1e-8 * q["volume"]
    assert abs(q["ke_change"]) < 5e-3
    ends = q["chart_ends"]
    assert ends["open"][0] > ends["open"][1] and ends["bubble"][0] < ends["bubble"][1]


RESTART_REMESH = """
import json, sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, sys.argv[1] + "/src")
sys.path.insert(0, sys.argv[1] + "/testCases")
import test_reconnect_fem as t
from faraday_singularity import surgery
from faraday_singularity.mesh import Focus
from faraday_singularity.reconnect import ReconnectSettings

post, out = Path(sys.argv[2]), Path(sys.argv[3])
meta = json.loads(post.with_suffix(".json").read_text())
report = {}
with t.neck_problem(out, ReconnectSettings(**t.FAST_CUT)) as pb:
    pb.initialise()
    for mode in ("plain", "restored"):
        pb.load_state(str(post))
        if mode == "restored":  # as CollapseDriver.restore does: the dumped focus, then the cut
            pb.active_mesh_template().focus = Focus(**meta["focus"])
            surgery.restore_cut(pb, meta["cut"])
        sizing = pb.active_mesh_template().cut_sizing
        pb.force_remesh()
        report[mode] = {"sizing": None if sizing is None else asdict(sizing),
                        "mesh": t.mesh_signature(pb)}
print("REPORT " + json.dumps(report))
"""


def test_restart_in_a_new_process_keeps_the_cut_sizing(neck_cut, tmp_path):
    """In a new process the template that pyoomph rebuilds a stored mesh from has no cut sizing;
    ``restore_cut`` puts it back, so the first remesh after the restart is the one an in-process
    restart makes."""
    proc = subprocess.run([sys.executable, "-c", RESTART_REMESH, str(ROOT),
                           neck_cut["postcut_path"], str(tmp_path / "out")],
                          capture_output=True, text=True, timeout=900, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    line = next(x for x in proc.stdout.splitlines() if x.startswith("REPORT "))
    report = json.loads(line[len("REPORT "):])
    assert report["plain"]["sizing"] is None  # what restore_cut repairs
    assert report["restored"]["sizing"] == neck_cut["record"]["sizing"]
    assert report["restored"]["mesh"] == neck_cut["restart_remesh"]
    assert report["plain"]["mesh"] != neck_cut["restart_remesh"]


def test_steps_after_the_cut_and_restart(neck_cut):
    a, b = neck_cut["after_first"], neck_cut["after_first_again"]
    meta = neck_cut["postcut_meta"]
    assert meta["cut_stage"] == "postcut" and meta["cut"]["t_cut"] == neck_cut["record"]["t_cut"]
    assert a.shape == b.shape
    # a reload is not bit for bit here (the mesh is read back from its stored file, and MKL runs
    # threaded); 1e-10 bounds what it changes
    assert np.max(np.abs(a - b)) <= 1e-10 * np.max(np.abs(a))
    assert neck_cut["chains_after"] == 2
    assert neck_cut["bridge_after"] > neck_cut["bridge0"]  # the caps move apart
    assert neck_cut["axis_state"] == neck_cut["open_tip"]


def test_zero_fill_and_impulsive_restart(neck_cut):
    z = neck_cut["zero"]
    fill, tr = z["fill"], z["transfer"]
    # the bridge's axis nodes, which pyoomph counts on the axis mesh, are zeroed too, all but its
    # two ends: the fresh tips, on the free surface
    assert fill["zeroed_axis_nodes"] == tr["unlocated"]["liquid/axis"][0] - 2 > 0
    assert fill["zeroed_axis_nodes"] == tr["former_gas_nodes"]["liquid/axis"][0] - 2
    assert fill["zeroed_nodes"] - fill["zeroed_axis_nodes"] == pytest.approx(
        tr["unlocated"]["liquid"][0], rel=0.1)
    pr = z["projection"]
    assert abs(pr["energy_identity_error"]) <= 1e-10 * pr["before"]["kinetic_energy"]
    assert neck_cut["zero_counter"] == 1 and neck_cut["zero_chains_after"] == 2


def test_zero_fill_zeroes_every_node_of_the_former_gas(neck_cut):
    c = neck_cut["zero_check"]
    assert c["bridge_nodes"] > 0 and c["counts"]["axis"] == c["bridge_nodes"]
    assert c["counts"]["bulk"] == c["inner_nodes"]
    assert c["max_speed"] == 0.0


def test_after_a_cut_the_driver_does_not_coarsen_the_tips_the_cut_made(tmp_path):
    """Every mesh after the cut keeps the cut's sizing at the tips, here h_tip = delta/32, and the
    tips the cut made keep the cut's focus size, delta/16, within 20 cut radii: the fresh cap's
    scale asks the rule for a coarser focus (before this was a rebuild at each of 6 steps without
    effect, read at the tip, and then coarsenings read away from it), but the driver makes no
    rebuild at all. The spacing near the collapse in collapse.txt is still read away from the tips,
    where the cut's sizing leaves it to the focus."""
    from faraday_singularity.reconnect import CUT_SIZING_MARGIN

    s = ReconnectSettings(delta=1.05 * R_N, dt0=1e-5, tip_size=1 / 32)
    settings = CollapseSettings(t_end=10.0, dt_max=0.01, max_steps=6)
    with neck_problem(tmp_path, s) as pb:
        pb.initialise()
        impose_closing_flow(pb)
        driver = CollapseDriver(pb, settings, tmp_path, reconnect=s)
        rec = surgery.cut(pb, s, tmp_path, driver=driver, log=lambda *_: None)
        result = driver.run(1e-8)
        driver.close()
        tips = _mesh_report(pb, 0.1, 5 * rec["rmin"])["tips"]
    events = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert result["stop"]["reason"] == "max-steps" and result["steps"] == 6
    assert result["remesh_count"] == 1  # the cut's own
    assert not [e for e in events if e["event"].startswith("remesh")]
    table = _columns(tmp_path / "collapse.txt")
    h_tip = rec["sizing"]["h_tip"]
    assert np.all(table["h_near"] > CUT_SIZING_MARGIN * h_tip)  # read away from the tips
    assert np.all(table["h_focus"] > s.focus_size * rec["rmin"])  # the rule's coarser target
    for name, h in tips.items():
        assert 0.9 * h_tip <= h.min() and h.max() <= 1.25 * s.focus_size * rec["rmin"], (name, h)


def _bend_element(pb, turn_deg: float) -> dict:
    """Move the midside node of the bubble's element at its widest point off the chord, towards the
    gas, so that the element's tangent turns by ``turn_deg`` across it (``s/L = tan(turn/2)/4``),
    at every time level and in its Lagrangian position (the reference of the mesh's smoothing):
    a kink of the interface in the state itself, as a rebuild would read it. Returns the element's
    chord ``L`` and the new place of its midside node ``at``."""
    seg = next(c for c in pb.interface_segments() if not c["open"])
    P = np.column_stack([seg["r"], seg["z"]])
    k = 2 * int(np.argmax(P[1::2, 0])) + 1
    a, b = P[k - 1], P[k + 1]
    L = float(np.hypot(*(b - a)))
    t = (b - a) / L
    new = 0.5 * (a + b) + L * math.tan(math.radians(turn_deg) / 2) / 4 * np.array([-t[1], t[0]])
    node = next(n for n in surgery.mesh_nodes(pb.get_mesh("liquid/interface"))
                if (n.x(0), n.x(1)) == (P[k, 0], P[k, 1]))
    for level in range(node.ntstorage()):
        for i in (0, 1):
            node.set_x_at_t(level, i, float(new[i]))
    for i in (0, 1):
        node.set_x_lagr(i, float(new[i]))
    pb.invalidate_cached_mesh_data()
    return {"L": L, "at": new}


def test_after_a_cut_an_element_bent_by_40_degrees_is_refined_away(tmp_path):
    """An element of the bubble bent by 40 degrees after the cut (below the stop of 90 on the
    freshly built mesh, above the 30 at which the guard asks for a refinement) is refined by the
    rebuild after the first step, in a region of its own (kink_bubble, a quarter of its chord):
    on the new mesh no element there turns by more than 30 degrees, and the run goes on."""
    from faraday_singularity.reconnect import p2_edges

    s = ReconnectSettings(**FAST_CUT)
    settings = CollapseSettings(t_end=10.0, dt_max=0.01, max_steps=2)
    with neck_problem(tmp_path, s) as pb:
        pb.initialise()
        impose_closing_flow(pb)
        driver = CollapseDriver(pb, settings, tmp_path, reconnect=s)
        surgery.cut(pb, s, tmp_path, driver=driver, log=lambda *_: None)
        bent = _bend_element(pb, 40.0)
        driver._post_cut_cache = None  # the state has changed
        from pyoomph import _pyoomph_core

        assert not _pyoomph_core.get_detect_inverted_elements()
        result = driver.run(1e-8)
        # the net for an element turned inside out was armed for the run after the cut, and is
        # disarmed when the driver closes (the detection is process-wide)
        assert result["cut"]["inverted_element_remesh"] is True
        assert result["cut"]["inverted_element_remeshes"] == 0
        assert _pyoomph_core.get_detect_inverted_elements()
        driver.close()
        assert not _pyoomph_core.get_detect_inverted_elements()
        assert not pb._domains_remesh_on_inversion
        bubble = next(c for c in pb.interface_segments() if not c["open"])
        e = p2_edges(np.column_stack([bubble["r"], bubble["z"]]))
        # the absolute floor of the quality of an element (pyoomph's: smallest over mean
        # Jacobian), for an element that has fallen below it since its build
        assert pb.quality_floor_reading(1e-3) is None
        q = [(e.get_quality_factor(), e.get_initial_quality_factor())
             for e in pb.get_mesh("liquid").elements()]
        q_k, q0_k = max(q, key=lambda x: x[1] - x[0])
        assert q0_k > q_k
        floor = 0.5 * (q_k + q0_k)
        reading = pb.quality_floor_reading(floor)
        assert reading["trigger"] == "absolute-quality" and reading["ratio"] <= q_k < floor
        assert reading["quality_floor"] == floor and "size_ratio_min" in reading
        # only elements that have fallen below the floor since their build count: one built below
        # it would be made again by every rebuild
        for f in (0.5, 0.9, 0.99, 0.999):
            expect = [x for x, q0 in q if x < f <= q0]
            got = pb.quality_floor_reading(f)
            assert (got is None) == (not expect), f
            if expect:
                assert got["ratio"] == pytest.approx(min(expect))
    near = np.hypot(e["mid"][:, 0] - bent["at"][0], e["mid"][:, 1] - bent["at"][1]) <= 2 * bent["L"]
    assert near.sum() >= 4 and e["turn"][near].max() < 30.0, e["turn"][near]
    assert result["stop"]["reason"] == "max-steps"
    rows = _columns(tmp_path / "chains.txt")
    assert rows["edge_turn_bubble"][0] == pytest.approx(40.0, abs=0.5)
    assert rows["edge_turn_bubble"][-1] < 30.0
    events = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    remesh = [x for x in events if x["event"] == "remesh"]
    assert remesh and remesh[0]["step"] == 1 and "kink_bubble" in remesh[0]["asked_by"]
    kink = {g["name"]: g for g in remesh[0]["regions"]}["kink_bubble"]
    assert kink["target"] == pytest.approx(bent["L"] / 4, rel=0.1)
    assert kink["new_size"] <= 0.3 * bent["L"]


def test_after_a_cut_a_failed_step_leaves_the_profiles_of_the_last_good_state(tmp_path):
    """After a cut no failure of a step ends a run cleanly, and the problem then holds a rejected
    iterate: the driver writes the profiles of both chains of the state the failing step started
    from (profile_<step>_lastgood.txt and its _bubble beside it) before it raises."""
    s = ReconnectSettings(**FAST_CUT)
    with neck_problem(tmp_path, s) as pb:
        pb.initialise()
        impose_closing_flow(pb)
        driver = CollapseDriver(pb, CollapseSettings(t_end=10.0, dt_max=0.01), tmp_path,
                                reconnect=s)
        surgery.cut(pb, s, tmp_path, driver=driver, log=lambda *_: None)
        expected = {("open" if c["open"] else "bubble"):
                    np.column_stack([c[k] for k in ("r", "z", "u_r", "u_z")])
                    for c in pb.interface_segments()}

        def failing(*_args, **_kw):
            raise RuntimeError("OomphException")

        pb.run = failing
        with pytest.raises(RuntimeError, match="OomphException"):
            driver.run(1e-8)
        driver.close()
    good = tmp_path / "profiles" / "profile_000000_lastgood.txt"
    for path, chain in ((good, "open"), (good.with_name("profile_000000_lastgood_bubble.txt"),
                                         "bubble")):
        header = json.loads(path.read_text().splitlines()[0][2:])
        assert header["reason"] == "lastgood" and header["chain"] == chain and header["step"] == 0
        assert "OomphException" in header["error"] and header["tau"] == 0.0
        assert np.array_equal(np.loadtxt(path), expected[chain])


class FlaggingNeckProblem(NeckProblem):
    """The neck problem with thresholds of element distortion that flag every step pyoomph checks,
    so that the driver rebuilds the mesh after a cut at about every other step."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        from pyoomph.equations.generic import RemeshingOptions

        self.remeshing_options = RemeshingOptions(max_expansion=1 + 1e-9, min_expansion=1 - 1e-9)


def _mesh_report(pb, kept_radius: float, near: float) -> dict:
    """The mesh of a state after a cut: the longest element of the bubble's bottom within
    ``kept_radius`` of its tip; the elements of each tip the cut made within ``near`` of it, beyond
    its first two (which the cut's tip sizing sets); and the largest ratio of adjacent element
    lengths on each chain."""
    segs = pb.interface_segments()
    bubble = next(c for c in segs if not c["open"])
    cavity = next(c for c in segs if c["open"])
    out = {"bottom": _bottom_spacing(pb, kept_radius), "tips": {}, "adjacent": {}}
    for name, c, k in (("cavity_tip", cavity, 0), ("bubble_top", bubble, -1)):
        r, z = np.asarray(c["r"]), np.asarray(c["z"])
        if k == -1:
            r, z = r[::-1], z[::-1]
        h = 2 * np.hypot(np.diff(r), np.diff(z))
        d = np.hypot(r[:-1], z[:-1] - z[0])
        out["tips"][name] = h[(d <= near) & (np.arange(len(h)) >= 4)]
    for name, c in (("bubble", bubble), ("open", cavity)):
        ds = np.hypot(np.diff(c["r"]), np.diff(c["z"]))
        h = ds[0::2][:len(ds[1::2])] + ds[1::2]  # quadratic elements: vertex, midside, vertex
        out["adjacent"][name] = float(np.max(np.maximum(h[1:] / h[:-1], h[:-1] / h[1:])))
    return out


# the cut of the post-cut feature runs: tips, bridge and focus at an eighth of the cut radius, finer
# than the cavity's h_axis, and a landing focus finer still (the bubble's bottom keeps it)
FEATURE_CUT = dict(FAST_CUT, tip_size=0.125, focus_size=0.125)
FEATURE_LANDING = Focus(size=R_N / 12, radius=0.6, grading=0.3)
# the landing run's h_floor of those runs: half the cut's focus size, above the size the rule asks
# for at the tips (ell/400), so that the tips are refined to it, twofold, and the run stays small
FEATURE_FLOOR = 0.5 * 0.125 * 1.05 * R_N


@pytest.fixture(scope="module")
def post_cut_features(tmp_path_factory):
    """A run after a cut of the synthetic cavity whose landing state was meshed at a focus of R_N/12
    near the axis, finer than the cut's focus (delta/8), with a rule that asks for ell/400 at the
    tips the cut made (about 1e-4, below an h_floor of delta/16) and distortion flags that make the
    driver rebuild the mesh: the mesh after the cut and after every rebuild of the driver, and the
    lower bound of the scaled Jacobian of its curved elements (``curved_jacobian_bound``), with the
    nodes and elements of the last mesh."""
    out = tmp_path_factory.mktemp("features")
    s = ReconnectSettings(**FEATURE_CUT)
    settings = CollapseSettings(t_end=10.0, dt_max=0.01, max_steps=4, elements_per_scale=400.0,
                                h_floor=FEATURE_FLOOR)
    nd = nondimensionalise(CaseParameters.from_dict(PHYSICS))
    pb = FlaggingNeckProblem(nd, mesh=MeshSettings(**MESH), reconnect=s)
    pb.set_output_directory(str(out))
    res = {"meshes": [], "out": out, "jacobians": []}
    with pb:
        pb.initialise()
        pb.active_mesh_template().focus = FEATURE_LANDING
        pb.force_remesh()  # the landing state, its cavity bottom meshed at the landing focus
        impose_closing_flow(pb)
        res["landing_bottom"] = _bottom_spacing(pb, 0.1)
        driver = CollapseDriver(pb, settings, out, reconnect=s)
        rec = surgery.cut(pb, s, out, driver=driver, log=lambda *_: None)
        res["record"] = rec
        report = (lambda: _mesh_report(pb, rec["focus"]["kept"]["radius"], 5 * rec["rmin"]))
        res["meshes"].append(report())
        res["jacobians"].append(pb.curved_jacobian_bound(s.jacobian_min))
        original = pb.force_remesh

        def recording(*args, **kw):
            original(*args, **kw)
            res["meshes"].append(report())
            res["jacobians"].append(pb.curved_jacobian_bound(s.jacobian_min))

        pb.force_remesh = recording
        res["result"] = driver.run(1e-8)
        driver.close()
        res["focus"] = pb.active_mesh_template().focus
        values, elements, _, fields = pb.get_mesh("liquid").to_numpy(False, True, 0, False)[:4]
        res["last_mesh"] = (np.asarray(values)[:, [fields["coordinate_x"], fields["coordinate_y"]]],
                            np.asarray(elements))
    res["events"] = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
    return res


def test_after_a_cut_the_bubble_bottom_keeps_its_landing_resolution(post_cut_features):
    """The bubble's bottom, far below the cut (its tip 0.3 from the neck, 14 cut radii), was meshed
    at the landing focus R_N/12; the cut's focus (delta/8, 1.6 times coarser) and every rebuild of
    the driver after it keep it there, never coarser than at the landing."""
    res = post_cut_features
    rec = res["record"]
    landing = FEATURE_LANDING.size
    assert res["landing_bottom"] <= 1.25 * landing
    assert rec["focus"]["kept"]["size"] == pytest.approx(landing)
    assert landing < 0.125 * rec["rmin"]
    meshes = res["meshes"]
    assert len(meshes) >= 3  # the cut's and at least two rebuilds of the driver
    for m in meshes:
        assert m["bottom"] <= 1.25 * landing, m["bottom"]
    regions = {g.name: g for g in res["focus"].regions}
    assert regions["bubble_bottom"].size == pytest.approx(landing)
    assert res["result"]["stop"]["reason"] == "max-steps"


def test_after_a_cut_the_tips_are_meshed_at_their_own_scale_down_to_h_floor(post_cut_features):
    """Near the tips the cut made, the rule's own size, ell/400 (about 1e-4), lies below the landing
    run's h_floor (half the cut's focus size here): the tips are meshed at their own scale clipped
    to between h_floor and the cut's focus size, so the first rebuild refines their regions to
    h_floor, twofold, and every later one keeps them there; the driver's focus is floored at h_floor
    only, not at the cut's focus size, which met a sharp tip with one or two elements. A rebuild
    for distortion keeps the size of every region."""
    res = post_cut_features
    rmin = res["record"]["rmin"]
    h_cut = 0.125 * rmin
    table = _columns(res["out"] / "collapse.txt")
    assert np.all(table["ell"] / 400 < FEATURE_FLOOR)  # the size the rule asks for
    assert np.all(table["h_focus"] == pytest.approx(FEATURE_FLOOR, rel=1e-12))
    first, later = res["meshes"][0], res["meshes"][2:]
    for name, h in first["tips"].items():  # the cut's mesh: its focus size
        assert len(h) > 10 and 0.6 * h_cut <= h.min() and h.max() <= 1.4 * h_cut, (name, h)
    assert later
    for m in later:
        for name, h in m["tips"].items():
            assert len(h) > 20, (name, h)
            assert 0.6 * FEATURE_FLOOR <= h.min() and h.max() <= 1.4 * FEATURE_FLOOR, (name, h)
    regions = {g.name: g for g in res["focus"].regions}
    for name in ("cavity_tip", "bubble_top"):
        assert regions[name].size == pytest.approx(FEATURE_FLOOR)
        assert regions[name].radius >= 20 * rmin * (1 - 1e-12)
    rows = _columns(res["out"] / "chains.txt")
    sized = np.isfinite(rows["size_cavity_tip"])
    assert sized.any()
    assert np.all(rows["size_cavity_tip"][sized][1:] == pytest.approx(FEATURE_FLOOR))
    assert np.all(rows["h_cavity_tip"][sized][1:] <= 1.4 * FEATURE_FLOOR)
    distortion = [e for e in res["events"] if e.get("reason") == "distortion"]
    assert distortion
    for e in distortion:  # a rebuild for distortion keeps the size of every region it has
        sizes = {g["name"]: g for g in e["regions"]}
        for name in ("cavity_tip", "bubble_top"):
            g = sizes[name]
            if math.isfinite(g["size"]):
                assert g["new_size"] == pytest.approx(g["size"])
        assert sizes["bubble_bottom"]["new_size"] == pytest.approx(FEATURE_LANDING.size)


def test_after_a_cut_every_mesh_has_valid_curved_elements(post_cut_features):
    """The cut's mesh and every rebuild after it: the lower bound of the scaled Jacobian of every
    curved element (the Bezier bound) is at least the cut's jacobian_min, 0.2, as the run accepts
    them; on the last mesh it is checked against the Jacobian sampled over its worst elements,
    which also checks the order of the nodes it assumes."""
    from faraday_singularity.reconnect import p2_jacobian_bounds

    res = post_cut_features
    assert len(res["jacobians"]) == len(res["meshes"]) >= 3
    for j in res["jacobians"]:
        assert j["min"] >= 0.2 and j["below"] == 0 and j["elements"] > 1000, j
    xy, elements = res["last_mesh"]  # the state the run ended on, stepped since its last rebuild
    bounds = p2_jacobian_bounds(xy, elements)
    assert bounds.min() >= 0.2
    assert np.median(bounds) == pytest.approx(1.0, abs=1e-2)  # nearly straight-sided elements
    for k in np.argsort(bounds)[:20]:
        X = xy[elements[k]]
        lin = (X[0, 0] - X[2, 0]) * (X[1, 1] - X[2, 1]) - (X[0, 1] - X[2, 1]) * (X[1, 0] - X[2, 0])
        assert lin > 0.0  # counterclockwise, as pyoomph orients every element
        assert bounds[k] <= _jacobian_sampled(X).min() / lin + 1e-9


def test_after_a_cut_a_rebuild_puts_a_misplaced_midside_node_of_the_axis_back(tmp_path):
    """The transfer to a rebuilt mesh interpolates the node positions; after a cut it has put a
    midside node of the axis 40 edge lengths off its edge (at the bubble's top, in a cut at delta
    5e-4), which turned its curved element inside out. Every rebuild after a cut runs
    ``repair_axis_midsides``, which puts such a node back at the mean of its edge's vertices at every
    time level, in position and velocity; it finds nothing to do on a sound mesh and leaves it as it
    is, and it does not run before a cut."""
    s = ReconnectSettings(**FAST_CUT)
    with neck_problem(tmp_path, s) as pb:
        pb.initialise()
        impose_closing_flow(pb)
        pb.force_remesh()  # before the cut: no repair
        assert pb.axis_repair is None and pb.axis_repairs_total == 0
        surgery.cut(pb, s, tmp_path)
        _step(pb, 1e-5)
        pb.force_remesh()
        sound = pb.axis_repair
        assert sound["repaired"] == 0 and sound["edges"] > 10 and sound["max_offset"] < 1e-3
        before = mesh_signature(pb)
        assert pb.repair_axis_midsides()["repaired"] == 0
        assert mesh_signature(pb) == before
        # misplace a midside node of the bridge at every time level, as the transfer did
        segs = pb.interface_segments()
        z_cavity = float(next(c for c in segs if c["open"])["z"][0])
        z_bubble = float(next(c for c in segs if not c["open"])["z"][-1])
        iv = pb.get_mesh("liquid").get_nodal_field_indices()["velocity_y"]
        edge = next(e for e in pb.get_mesh("liquid/axis").elements() if e.nnode() == 3
                    and all(z_bubble < e.node_pt(k).x(1) < z_cavity for k in range(3)))
        a, mid, b = (edge.node_pt(k) for k in range(3))
        length = abs(b.x(1) - a.x(1))
        assert abs(mid.x(1) - 0.5 * (a.x(1) + b.x(1))) < 1e-3 * length  # the middle node
        levels = range(pb.timestepper.nprev_values() + 1)  # the current level and the history
        assert len(levels) == 3  # BDF2
        for t in levels:
            mid.set_x_at_t(t, 1, mid.x_at_t(t, 1) + 40 * length)
            mid.set_value_at_t(t, iv, 1e3)
        pb._jacobian_cache = None
        assert pb.curved_jacobian_bound(0.2)["min"] < 0.0
        rec = pb.repair_axis_midsides()
        assert rec["repaired"] == 1 and rec["max_offset"] == pytest.approx(40.0, rel=1e-2)
        assert rec["z"] == pytest.approx(0.5 * (a.x(1) + b.x(1)) + 40 * length)
        for t in levels:
            for i in range(2):
                assert mid.x_at_t(t, i) == pytest.approx(0.5 * (a.x_at_t(t, i) + b.x_at_t(t, i)),
                                                         rel=0, abs=1e-14)
            assert mid.value_at_t(t, iv) == pytest.approx(
                0.5 * (a.value_at_t(t, iv) + b.value_at_t(t, iv)), rel=0, abs=1e-12)
        pb._jacobian_cache = None
        assert pb.curved_jacobian_bound(0.2)["min"] >= 0.2
        _step(pb, 1.25e-5)  # and the problem steps on
        assert len(pb.interface_segments()) == 2


def _jacobian_sampled(X, n=40):
    """``det(dx/ds)`` of a quadratic triangle (oomph-lib's order of nodes) sampled over its
    reference triangle, from the derivatives of its Lagrange shape functions."""
    out = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            s1, s2 = i / n, j / n
            s3 = 1.0 - s1 - s2
            d1 = np.array([4 * s1 - 1, 0.0, -(4 * s3 - 1), 4 * s2, -4 * s2, 4 * (s3 - s1)])
            d2 = np.array([0.0, 4 * s2 - 1, -(4 * s3 - 1), 4 * s1, 4 * (s3 - s2), -4 * s1])
            a, b = d1 @ X, d2 @ X
            out.append(a[0] * b[1] - a[1] * b[0])
    return np.array(out)


def test_after_a_cut_no_mesh_has_a_jump_in_element_length(post_cut_features):
    """With gmsh's sizing from curvature off, adjacent elements of either chain differ by no more
    than the grading allows, on the cut's mesh and every rebuild after it: no cluster of short
    elements beside long ones, through which the spline of a later rebuild could loop. A guard: this
    short run does not grow such a cluster even with the sizing on; the feedback that does is
    pinned by test_gmsh_sizing_from_curvature_amplifies_a_jump_in_node_spacing."""
    for m in post_cut_features["meshes"]:
        for chain, ratio in m["adjacent"].items():
            assert ratio < 2.0, (chain, ratio)


def test_the_vertex_spline_of_a_rebuild_is_an_option_for_sensitivity_tests_only():
    """With ``rebuild_spline`` ``vertices`` (off by default) a rebuild after a cut splines each
    chain through its vertices, every other node, so that a midside node off its place along the
    curve does not shape the new surface; the stops test the spline of those same nodes. A chain
    with an even number of nodes has no vertices to take."""
    from faraday_singularity.reconnect import ReconnectSettings as RS

    assert RS(delta=1e-3).rebuild_spline == "nodes"
    t = FaradayCylinderMesh(3.8, 2.0, lambda r: 0 * r, MeshSettings(), reconnect=True)
    th = [math.pi * k / 8 for k in range(1, 8)]
    bubble = ([(0.0, 0.48)] + [(0.02 * math.sin(x), 0.5 - 0.02 * math.cos(x)) for x in th]
              + [(0.0, 0.52)])
    cavity = [(0.0, 0.6)] + [(0.1 * k, 0.6 + 0.01 * k) for k in range(1, 38)] + [(3.8, 1.0)]
    assert len(bubble) % 2 == 1 and len(cavity) % 2 == 1
    every = t._current_chains([bubble, cavity])
    t.rebuild_spline = "vertices"
    vertices = t._current_chains([bubble, cavity])
    for a, b in zip(every, vertices):
        assert np.array_equal(b.points, a.points[::2]) and b.end_types == a.end_types
    with pytest.raises(RuntimeError, match="even number of nodes"):
        t._current_chains([bubble, cavity[:5] + cavity[6:]])


def test_gmsh_sizing_from_curvature_amplifies_a_jump_in_node_spacing(tmp_path):
    """Why gmsh's sizing from curvature is off after a cut (``reconnect.ReconnectSettings``,
    ``curvature_sizing``): on a gently curved wall meshed at a uniform size, a jump in the spacing
    of the nodes the spline runs through (two intervals six times shorter than their neighbours)
    comes back from a rebuild with 20 elements per turn as a jump of more than ten in adjacent
    element lengths, and without it as none: gmsh reads the curvature of the uniform Catmull-Rom
    spline, which the jump makes spurious, and does not grade what it asks for."""
    gmsh = pytest.importorskip("gmsh")

    def rebuilt(P, h0, curvature):
        geo = tmp_path / f"c{curvature}.geo"
        lines = [f"Point({k}) = {{{float(x)!r}, {float(y)!r}, 0, 1}};"
                 for k, (x, y) in enumerate(P, 1)]
        lines.append("Spline(1) = {" + ", ".join(str(k) for k in range(1, len(P) + 1)) + "};")
        lines.append(f'Field[1] = MathEval; Field[1].F = "{float(h0)!r}"; Background Field = 1;')
        geo.write_text("\n".join(lines) + "\n")
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.NumThreads", 1)
            gmsh.open(str(geo))
            for k, v in (("Mesh.MeshSizeMin", h0 / 100), ("Mesh.MeshSizeMax", 1.0),
                         ("Mesh.MeshSizeFromCurvature", curvature), ("Mesh.MeshSizeFromPoints", 0),
                         ("Mesh.MeshSizeExtendFromBoundary", 0)):
                gmsh.option.setNumber(k, v)
            gmsh.model.mesh.generate(1)
            _, xyz, par = gmsh.model.mesh.getNodes(1, 1, includeBoundary=False,
                                                   returnParametricCoord=True)
            X = np.asarray(xyz).reshape(-1, 3)[np.argsort(par), :2]
        finally:
            gmsh.finalize()
        h = np.hypot(*np.diff(np.vstack([P[:1], X, P[-1:]]), axis=0).T)
        return float(np.max(np.maximum(h[1:] / h[:-1], h[:-1] / h[1:])))

    long, short = 8.3e-5, 1.47e-5
    n = int(3e-3 / long)
    r = np.concatenate([[0.0], np.cumsum([long] * n + [short] * 2 + [long] * n)])
    P = np.column_stack([r, 0.12 * r**2 / 1e-3 + 2e4 * r**4 / 1e-3])
    assert rebuilt(P, 4e-4, 20) > 10.0
    assert rebuilt(P, 4e-4, 0) < 1.05
    assert ReconnectSettings(delta=1e-3).curvature_sizing == 0


@pytest.mark.parametrize("ratio", [1.2, 2.0])
def test_cut_is_refused_far_above_the_neck(tmp_path, ratio):
    s = ReconnectSettings(delta=ratio * R_N)
    with neck_problem(tmp_path, s) as pb:
        pb.initialise()
        with pytest.raises(surgery.CutRefused, match="above") as refused:
            surgery.cut(pb, s, tmp_path)
        assert pb.cut is None and not pb.get_mesh("liquid")._templatemesh.has_pending_surgery_plan()
    # the refusal leaves its record
    rec = json.loads((tmp_path / "cut_diagnostics.json").read_text())
    assert rec == refused.value.record
    assert rec["outcome"] == "refused" and rec["failed_at"] == "admission"
    assert rec["delta_over_r_neck"] == pytest.approx(ratio, rel=1e-2) and "above" in rec["error"]


def test_a_failed_cut_writes_its_diagnostics(tmp_path, monkeypatch):
    """A failure after the remesh (here in the last stage, the chart) leaves the record as far as
    it got in ``cut_diagnostics.json``, and on the exception."""
    def broken_chart(_problem):
        raise RuntimeError("no chart")

    monkeypatch.setattr(surgery, "orient_interface_chart", broken_chart)
    s = ReconnectSettings(**FAST_CUT)
    with neck_problem(tmp_path, s) as pb:
        pb.initialise()
        impose_closing_flow(pb)
        with pytest.raises(surgery.CutFailed, match="no chart") as failed:
            surgery.cut(pb, s, tmp_path)
    rec = json.loads((tmp_path / "cut_diagnostics.json").read_text())
    assert rec == failed.value.record
    assert rec["outcome"] == "failed" and rec["failed_at"] == "chart" and "no chart" in rec["error"]
    assert failed.value.precut == rec["precut_state"]["path"]
    assert Path(failed.value.precut).is_file()
    for key in ("plan_attempts", "caps", "geometry", "in_cut_radii", "transfer", "mesh",
                "budgets_after_transfer", "budgets_after_fill", "local_budgets"):
        assert key in rec, key
    pr = rec["projection"]
    assert pr["newton_residuals"] and abs(pr["energy_identity_error"]) <= 1e-10 * pr["before"][
        "kinetic_energy"]
    assert "postcut_state" not in rec and "volumes" not in rec


@pytest.mark.parametrize("stage, target", [("caps", "replace_caps"),
                                           ("chart", "orient_interface_chart")])
@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_an_interruption_is_no_outcome_of_the_cut(tmp_path, monkeypatch, stage, target,
                                                   interruption):
    """An interruption before the remesh (the caps) or after it (the chart) propagates unchanged:
    it is neither refused nor failed. The plan is dropped, and the partial record says
    ``interrupted`` with the stage reached."""
    def interrupt(*_args, **_kw):
        raise interruption("stop now")

    monkeypatch.setattr(surgery, target, interrupt)
    s = ReconnectSettings(**FAST_CUT)
    with neck_problem(tmp_path, s) as pb:
        pb.initialise()
        impose_closing_flow(pb)
        with pytest.raises(interruption) as raised:
            surgery.cut(pb, s, tmp_path)
        assert type(raised.value) is interruption
        assert pb.cut is None and not pb.get_mesh("liquid")._templatemesh.has_pending_surgery_plan()
    rec = json.loads((tmp_path / "cut_diagnostics.json").read_text())
    assert rec["outcome"] == "interrupted" and rec["failed_at"] == stage
    assert rec["error"].startswith(interruption.__name__)


def test_cut_is_refused_too_close_to_the_neck(tmp_path):
    s = ReconnectSettings(delta=1.01 * R_N)
    with neck_problem(tmp_path, s) as pb:
        pb.initialise()
        with pytest.raises(surgery.CutRefused, match="below"):
            surgery.cut(pb, s, tmp_path)
        assert pb.cut is None and not pb.get_mesh("liquid")._templatemesh.has_pending_surgery_plan()


# --- the prototype entry point ----------------------------------------------------------------------


PINCH = (1e-4, 2e-2, 0.56)  # t_p, A, alpha of the neck samples given to the landing state


@pytest.fixture(scope="module")
def landing_run(tmp_path_factory):
    """A landing state of the synthetic cavity, with the records of the run that wrote it (a
    stand-in commit) and the neck samples of a power-law neck ``A (t_p - t)^alpha`` before it; the
    case of that run, one derived from it for the prototype, and one for the production runner's
    cut (the landing run's case with a ``collapse.reconnect`` block at ``delta = 1.05 r_n``)."""
    root = tmp_path_factory.mktemp("prototype")
    landing = root / "landing"
    landing.mkdir()
    s = ReconnectSettings(**FAST_CUT)
    with neck_problem(landing, s) as pb:
        pb.initialise()
        impose_closing_flow(pb)
        driver = CollapseDriver(pb, CollapseSettings(t_end=10.0, dt_max=0.01), landing)
        t_p, A, alpha = PINCH
        ts = -1e-6 * np.arange(60)[::-1]  # up to the state, at t = 0
        driver.state.neck_samples = [[float(t), float(A * (t_p - t) ** alpha)] for t in ts]
        seg = pb.interface_segments()[0]
        r_n, z_n, _ = neck_radius(seg["r"], seg["z"])
        delta = 1.05 * r_n
        info = {"delta": delta, "r_neck": r_n, "z_neck": z_n, "delta_over_r_neck": 1.05}
        state = driver.keep_state(landing / "states" / f"landing_{delta:.6g}.dump", landing=info,
                                  pinch=driver.pinch_time())
        driver.close()
    # the default state_dt (0.1), as in a production landing case: far beyond the run after the cut,
    # whose dumps follow from the doubling of tau after the cut and the end of the run
    parent_case = {"schema": "pyoomph-case-v1", "case_id": "neck-landing",
                   "geometry": {"kind": "axisymmetric"}, "physics": PHYSICS, "mesh": MESH,
                   "tolerances": {"newton": 1e-8},
                   "time_stepping": {"scheme": "BDF2", "steps_per_forcing_period": 100},
                   "collapse": {"keep_stop_state": True, "state_keep": 10},
                   "output": {"cadence": 0.25, "cadence_unit": "forcing periods",
                              "write_states": False},
                   "completion": {"kind": "event", "t_max": 10, "unit": "forcing periods"}}
    # the parent run's records, as run_case writes them (a stand-in commit)
    parent_path = root / "landing-case.json"
    parent_path.write_text(json.dumps(parent_case))
    digest = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    (landing / "run-manifest.json").write_text(json.dumps(
        {"schema": "pyoomph-runner-manifest-v1", "component_commit": "b" * 40,
         "case_sha256": digest}))
    (landing / "parameters.json").write_text(json.dumps(
        {"case": parent_case, "provenance": {"case_sha256": digest}}))
    case = json.loads(json.dumps(parent_case))
    case["case_id"] = "neck-prototype"
    case["collapse"]["reconnect"] = dict(FAST_CUT, delta=1.0)
    case_path = root / "case.json"
    case_path.write_text(json.dumps(case))
    cut = json.loads(json.dumps(parent_case))
    cut["case_id"] = "neck-cut"
    cut["collapse"]["reconnect"] = dict(FAST_CUT, delta=delta, tau_end=3e-5)
    cut_path = root / "cut-case.json"
    cut_path.write_text(json.dumps(cut))
    return {"state": state, "case": case_path, "digest": digest, "cut_case": cut_path,
            "delta": delta, "r_neck": r_n}


def _columns(path: Path) -> dict[str, np.ndarray]:
    cols = path.read_text().splitlines()[0][2:].split("\t")
    data = np.loadtxt(path, ndmin=2)
    return {c: data[:, k] for k, c in enumerate(cols)}


def test_prototype_cuts_a_landing_state_and_steps(landing_run, tmp_path):
    """The prototype cuts at a ratio to the state's neck and steps under the collapse driver in its
    post-cut mode, as the production runner does."""
    state = landing_run["state"]
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, str(ROOT / "simulationCases" / "reconnect_prototype.py"),
                           str(landing_run["case"]), "--state", str(state), "--outdir", str(out),
                           "--delta-ratio", "1.05", "--steps", "2"],
                          capture_output=True, text=True, timeout=900, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    status = json.loads((out / "run_status.json").read_text())
    assert status["state"] == "completed" and status["steps"] == 2
    assert status["stop"]["reason"] == "max-steps"
    cut = json.loads((out / "cut_diagnostics.json").read_text())
    assert cut["delta"] == pytest.approx(1.05 * cut["r_neck"])
    # the pinch time, fitted to the neck samples the landing state carries
    t_p, _, alpha = PINCH
    pinch = cut["pinch"]
    assert pinch["source"] == "neck_samples" and pinch["samples"] >= 10
    assert pinch["t_p"] == pytest.approx(t_p, rel=1e-3)
    assert pinch["alpha"] == pytest.approx(alpha, rel=1e-2)
    assert pinch["t_cut_minus_t_p"] == pytest.approx(cut["t_cut"] - pinch["t_p"], abs=1e-15)
    rows = _columns(out / "chains.txt")
    assert len(rows["step"]) == 3
    tau, dt = rows["tau"], rows["dt"]
    assert tau[0] == 0.0 and dt[1] == pytest.approx(1e-5) and dt[2] == pytest.approx(1.25e-5)
    assert np.allclose(rows["tau_p"], rows["time"] - pinch["t_p"], rtol=0, atol=1e-15)
    # each step within the limits both chains set at its start
    assert np.all(dt[1:] <= np.minimum(rows["dt_courant"], rows["dt_capillary"])[:-1] * (1 + 1e-12))
    assert np.all(np.isfinite(rows["momentum_z"])) and np.all(rows["bridge"] > 0)
    table = _columns(out / "collapse.txt")
    assert np.allclose(table["tau_p"], table["time"] - pinch["t_p"], rtol=0, atol=1e-15)
    for path in sorted((out / "profiles").glob("profile_*_bubble.txt")):
        header = json.loads(path.read_text().splitlines()[0][2:])
        assert header["chain"] == "bubble" and header["tau_p"] == pytest.approx(
            header["time"] - pinch["t_p"], abs=1e-15)
        assert path.with_name(path.name.replace("_bubble", "")).is_file()
    for name in (f"precut_{cut['delta']:.6g}", f"postcut_{cut['delta']:.6g}"):
        assert (out / "states" / f"{name}.dump").is_file()
        assert (out / "states" / f"{name}.json").is_file()
    precut = json.loads((out / "states" / f"precut_{cut['delta']:.6g}.json").read_text())
    assert precut["pinch"] == pinch and precut["cut_stage"] == "precut"
    events = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events[:2]] == ["restart", "cut"]
    assert events[1]["t_pinch"] == pinch["t_p"] and events[1]["in_cut_radii"] == cut["in_cut_radii"]
    assert events[1]["tau"] == 0.0 and events[-1]["event"] == "stop"
    manifest = json.loads((out / "run-manifest.json").read_text())
    assert manifest["schema"] == "faraday-reconnect-prototype-v1"
    assert manifest["state"]["path"] == str(state.resolve())
    parent = manifest["state"]["parent_record"]
    assert parent["case_sha256"] == landing_run["digest"] and parent["component_commit"] == "b" * 40
    assert parent["derived_keys"] == ["case_id", "collapse.reconnect"]
    # the prototype makes one cut per run: it does not continue its postcut state
    post = out / "states" / f"postcut_{cut['delta']:.6g}.dump"
    proc = subprocess.run([sys.executable, str(ROOT / "simulationCases" / "reconnect_prototype.py"),
                           str(landing_run["case"]), "--state", str(post), "--outdir",
                           str(tmp_path / "again"), "--delta-ratio", "1.05", "--check"],
                          capture_output=True, text=True, timeout=300, check=False)
    assert proc.returncode != 0 and "one cut per run" in proc.stderr


FAILING_PROTOTYPE = """
import sys
sys.path.insert(0, sys.argv[1] + "/simulationCases")
sys.path.insert(0, sys.argv[1] + "/src")
from faraday_singularity import problem, surgery
import reconnect_prototype

def broken_chart(_problem):
    raise RuntimeError("no chart")

load, loads = problem.FaradayProblem.load_state, []

def first_load_only(self, *args, **kw):  # the landing state; a reload after the cut fails
    loads.append(args)
    if len(loads) > 1:
        raise RuntimeError("a reload after the failed cut")
    return load(self, *args, **kw)

surgery.orient_interface_chart = broken_chart
problem.FaradayProblem.load_state = first_load_only
reconnect_prototype.main(sys.argv[2:])
"""


def test_prototype_records_a_failed_cut(landing_run, tmp_path):
    """A cut that fails after its remesh ends the run as ``failed``, with the failure's own message
    and stage, its partial record in ``cut_diagnostics.json`` and the precut dump named. The problem
    is not reloaded, so a reload that would fail too cannot replace the first failure."""
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, "-c", FAILING_PROTOTYPE, str(ROOT),
                           str(landing_run["case"]), "--state", str(landing_run["state"]),
                           "--outdir", str(out), "--delta-ratio", "1.05", "--steps", "2"],
                          capture_output=True, text=True, timeout=900, check=False)
    status = json.loads((out / "run_status.json").read_text())
    assert status["state"] == "failed", proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "no chart" in status["detail"] and status["failed_at"] == "chart"
    assert status["cut_diagnostics"] == "cut_diagnostics.json" and "restored_precut" not in status
    assert Path(status["precut"]).is_file()
    rec = json.loads((out / "cut_diagnostics.json").read_text())
    assert rec["outcome"] == "failed" and rec["failed_at"] == "chart"
    assert rec["projection"]["newton_residuals"] and rec["pinch"]["source"] == "neck_samples"


# --- the production runner: a cut at the start of a restart, and a restart after it ------------------


def _run_faraday(*args, script=None):
    """``simulationCases/run_faraday.py`` (or ``script``, which runs it) in a fresh process, MKL on
    one thread so that a continuation can be compared with the run it continues."""
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    head = ([sys.executable, "-c", script, str(ROOT)] if script is not None else
            [sys.executable, str(ROOT / "simulationCases" / "run_faraday.py")])
    return subprocess.run(head + [str(a) for a in args], capture_output=True, text=True,
                          timeout=900, env=env, check=False)


SHA256, COMMIT = re.compile(r"[0-9a-f]{64}"), re.compile(r"[0-9a-f]{40}")


def _runner_manifest_problems(m: dict) -> list[str]:
    """The fields validate-run-output requires of a pyoomph-runner-manifest-v1, but for a dirty
    commit, which a tree under edit has (the test tree can be dirty)."""
    out = []
    if m.get("schema") != "pyoomph-runner-manifest-v1":
        out.append("schema")
    if not SHA256.fullmatch(str(m.get("case_sha256"))):
        out.append("case_sha256")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", str(m.get("case_id"))):
        out.append("case_id")
    if not re.fullmatch(r"[0-9a-f]{40}(-dirty)?", str(m.get("component_commit"))):
        out.append("component_commit")
    if not COMMIT.fullmatch(str((m.get("pyoomph") or {}).get("commit"))):
        out.append("pyoomph.commit")
    if not (isinstance(m.get("argv"), list) and m["argv"] and all(
            isinstance(x, str) and x for x in m["argv"])):
        out.append("argv")
    fields = m.get("case_fields_used")
    if not (isinstance(fields, list) and fields and len(set(fields)) == len(fields)):
        out.append("case_fields_used")
    restart = m.get("restart_from")
    if restart is not None and not (isinstance(restart.get("path"), str)
                                    and SHA256.fullmatch(str(restart.get("sha256")))):
        out.append("restart_from")
    return out


@pytest.fixture(scope="module")
def production_cut(landing_run, tmp_path_factory):
    """The production runner cuts the landing state at its start (the case of the landing run with
    a reconnect block), then continues from dumps written after the cut: the first rotating one,
    the ``postcut`` dump, and the dump of the state it ended on at ``tau_end``, under a case that
    raises ``tau_end`` and dumps every step (and, refused before any output, under its own)."""
    root = tmp_path_factory.mktemp("production")
    cut_run = root / "cut" / "runtime"
    res = {"cut": cut_run, "proc_cut": _run_faraday(landing_run["cut_case"], "--restart-from",
                                                    landing_run["state"], "--out", cut_run)}
    if res["proc_cut"].returncode == 0:
        # a restart needs a parent with a clean source commit; the test tree may be under edit
        path = cut_run / "run-manifest.json"
        m = json.loads(path.read_text())
        res["cut_commit"] = m["component_commit"]
        m["component_commit"] = m["component_commit"].removesuffix("-dirty")
        path.write_text(json.dumps(m))
        status = json.loads((cut_run / "run_status.json").read_text())
        dumps = sorted((cut_run / "states").glob("state_*.dump"))
        res["from"] = dumps[0]
        res["post"] = root / "post" / "runtime"
        res["proc_post"] = _run_faraday(landing_run["cut_case"], "--restart-from", dumps[0],
                                        "--out", res["post"])
        res["postcut"] = Path(status["cut"]["postcut"])
        res["from_postcut"] = root / "from-postcut" / "runtime"
        res["proc_from_postcut"] = _run_faraday(landing_run["cut_case"], "--restart-from",
                                                res["postcut"], "--out", res["from_postcut"])
        res["end"] = Path(status["collapse"]["stop"]["end_state"])
        later = json.loads(landing_run["cut_case"].read_text())
        later["case_id"] = "neck-cut-later"
        later["collapse"]["reconnect"]["tau_end"] = 6e-5
        later["collapse"]["state_dt"] = 1e-6
        res["later_case"] = root / "later-case.json"
        res["later_case"].write_text(json.dumps(later))
        res["later"] = root / "later" / "runtime"
        res["proc_later"] = _run_faraday(res["later_case"], "--restart-from", res["end"], "--out",
                                         res["later"])
        res["same"] = root / "same" / "runtime"
        res["proc_same"] = _run_faraday(landing_run["cut_case"], "--restart-from", res["end"],
                                        "--out", res["same"], "--check")
    return res


def test_production_runner_cuts_a_landing_state_at_its_start(production_cut, landing_run):
    proc = production_cut["proc_cut"]
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    run = production_cut["cut"]
    status = json.loads((run / "run_status.json").read_text())
    assert status["state"] == "completed" and status["collapse"]["stop"]["reason"] == "tau-end"
    t_p = PINCH[0]
    assert status["cut"]["outcome"] == "cut" and status["cut"]["t_p"] == pytest.approx(t_p, rel=1e-3)
    assert status["collapse"]["cut"]["tau"] == pytest.approx(3e-5)
    # the standard runner manifest, with the lineage of the landing state
    manifest = json.loads((run / "run-manifest.json").read_text())
    manifest["component_commit"] = production_cut["cut_commit"]
    assert _runner_manifest_problems(manifest) == []
    assert manifest["case_sha256"] == hashlib.sha256(
        landing_run["cut_case"].read_bytes()).hexdigest() and manifest["case_id"] == "neck-cut"
    assert manifest["restart_mode"] == "cut" and manifest["reconnect"]["delta"] == landing_run["delta"]
    restart = manifest["restart_from"]
    assert restart["path"] == str(landing_run["state"].resolve())
    assert restart["sha256"] == hashlib.sha256(landing_run["state"].read_bytes()).hexdigest()
    parent = restart["parent_record"]
    assert parent["case_sha256"] == landing_run["digest"] and parent["component_commit"] == "b" * 40
    assert parent["derived_keys"] == ["case_id", "collapse.reconnect"]
    params = json.loads((run / "parameters.json").read_text())
    assert params["restart_mode"] == "cut"
    assert params["inactive_after_cut"] == {"keep_stop_state": True}
    # the cut, its record and its kept states
    rec = json.loads((run / "cut_diagnostics.json").read_text())
    assert rec["outcome"] == "cut" and rec["pinch"]["t_p"] == status["cut"]["t_p"]
    assert rec["focus"]["cut"]["size"] == pytest.approx(0.25 * rec["rmin"])
    for name in ("precut", "postcut"):
        assert Path(status["cut"][name]).is_file() and Path(status["cut"][name]).parent == run / "states"
    # after the cut: the post-cut mode of the driver
    chains, table = _columns(run / "chains.txt"), _columns(run / "collapse.txt")
    assert list(chains["dt"]) == pytest.approx([0.0, 1e-5, 1.25e-5, 7.5e-6])  # tau_end 3e-5
    assert chains["tau"][0] == 0.0 and np.allclose(chains["tau_p"], chains["time"] - rec["pinch"]["t_p"])
    assert np.all(chains["dt"][1:] <= np.minimum(chains["dt_courant"],
                                                 chains["dt_capillary"])[:-1] * (1 + 1e-12))
    assert np.array_equal(table["tau"], chains["tau"]) and np.all(np.isfinite(table["z_axis"]))
    assert {"r_j", "V_j", "Q_j", "tau_p"} <= set(table)
    # the jets of the three tips, the open chain's as collapse.txt has it; the bubble of the
    # synthetic cavity has no jet, and its scale (its length here) lies above the cap of the scale
    for tip in ("cavity_tip", "bubble_bottom", "bubble_top"):
        names = {f"{k}_{tip}" for k in ("r_j", "z_b", "Q_j", "V_j", "We_j", "jet_born")}
        assert names <= set(chains)
    assert np.array_equal(chains["r_j_cavity_tip"], table["r_j"], equal_nan=True)
    assert np.array_equal(chains["V_j_cavity_tip"], table["V_j"], equal_nan=True)
    assert np.all(chains["jet_born_bubble_top"] == 0) and np.all(chains["r_j_bubble_top"] == 0.0)
    assert np.all(np.isfinite(chains["ell_bubble"]))
    assert np.all(table["ell"] <= chains["ell_bubble"])
    events = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds[:2] == ["restart", "cut"] and kinds[-1] == "stop"
    assert not any(k.startswith("landing") for k in kinds)
    assert all("tau_p" in e for e in events[1:])
    bubbles = sorted((run / "profiles").glob("profile_*_bubble.txt"))
    assert bubbles and all(json.loads(b.read_text().splitlines()[0][2:])["chain"] == "bubble"
                           for b in bubbles)
    # the landing case's state_dt (0.1) exceeds the run after the cut: its dumps are the first
    # step, the next at twice its tau, and the state it ended on at tau_end
    dumps = sorted((run / "states").glob("state_*.json"))
    step0 = json.loads(landing_run["state"].with_suffix(".json").read_text())["step"]
    assert [json.loads(d.read_text())["step"] - step0 for d in dumps] == [1, 2, 3]
    assert status["collapse"]["stop"]["end_state"] == str(dumps[-1].with_suffix(".dump"))
    for d in dumps:
        meta = json.loads(d.read_text())
        assert meta["cut"]["t_cut"] == rec["t_cut"] and meta["tau"] == pytest.approx(
            meta["time"] - rec["t_cut"], abs=1e-15)
        assert "landed" not in meta and "resume_at" not in meta
        assert meta["coarsen_blocked_until"] == -10**9


def test_production_runner_continues_a_post_cut_dump(production_cut, landing_run):
    """A restart from a dump written after the cut restores the cut and continues in the post-cut
    mode: the same steps as the run it continues, to round-off."""
    proc = production_cut["proc_post"]
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    run, cut_run = production_cut["post"], production_cut["cut"]
    status = json.loads((run / "run_status.json").read_text())
    assert status["state"] == "completed" and status["collapse"]["stop"]["reason"] == "tau-end"
    assert "cut" not in status  # no second cut
    manifest = json.loads((run / "run-manifest.json").read_text())
    assert _runner_manifest_problems(manifest) == []
    assert manifest["restart_mode"] == "post-cut"
    assert manifest["restart_from"]["path"] == str(production_cut["from"].resolve())
    assert "derived_keys" not in manifest["restart_from"]["parent_record"]  # the same case
    events = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "restart" and "tau" in events[0]
    assert "cut" not in [e["event"] for e in events] and events[-1]["event"] == "stop"
    a, b = _columns(cut_run / "chains.txt"), _columns(run / "chains.txt")
    k = int(np.searchsorted(a["step"], b["step"][0]))
    assert b["step"][0] == json.loads(production_cut["from"].with_suffix(".json").read_text())["step"]
    for name in b:
        if name == "dt":  # the restart's first row records no step
            assert np.allclose(b[name][1:], a[name][k + 1:], rtol=1e-8, atol=1e-15)
        else:
            assert np.allclose(b[name], a[name][k:], rtol=1e-8, atol=1e-13, equal_nan=True), name
    meta = json.loads(max((run / "states").glob("state_*.json")).read_text())
    assert meta["cut"]["sizing"] == json.loads(
        (cut_run / "cut_diagnostics.json").read_text())["sizing"]


def test_production_runner_continues_from_the_postcut_dump(production_cut):
    """The postcut dump, the one dump after the cut that a run of a production case had: a
    continuation from it makes the steps of the cut run, to round-off."""
    proc = production_cut["proc_from_postcut"]
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    run, cut_run = production_cut["from_postcut"], production_cut["cut"]
    status = json.loads((run / "run_status.json").read_text())
    assert status["state"] == "completed" and status["collapse"]["stop"]["reason"] == "tau-end"
    manifest = json.loads((run / "run-manifest.json").read_text())
    assert manifest["restart_mode"] == "post-cut"
    assert manifest["restart_from"]["path"] == str(production_cut["postcut"].resolve())
    a, b = _columns(cut_run / "chains.txt"), _columns(run / "chains.txt")
    assert b["tau"][0] == 0.0 and np.array_equal(a["step"], b["step"])
    for name in b:
        first = 1 if name == "dt" else 0  # the restart's first row records no step
        assert np.allclose(b[name][first:], a[name][first:], rtol=1e-8, atol=1e-13,
                           equal_nan=True), name


def test_production_runner_continues_a_run_that_ended_at_tau_end_under_a_later_tau_end(
        production_cut):
    """The state a cut run ended on at tau_end is dumped; under the cut's case a continuation from
    it is refused before any output (it would stop at once), and under a case derived from it that
    raises tau_end, and dumps every step, it goes on to the new tau_end. The cut's record keeps its
    own settings."""
    same = production_cut["proc_same"]
    assert same.returncode != 0 and "raise tau_end" in same.stderr
    assert not production_cut["same"].exists()
    proc = production_cut["proc_later"]
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    run = production_cut["later"]
    status = json.loads((run / "run_status.json").read_text())
    assert status["state"] == "completed" and status["collapse"]["stop"]["reason"] == "tau-end"
    assert status["collapse"]["cut"]["tau"] == pytest.approx(6e-5)
    assert status["collapse"]["cut"]["tau_end"] == 6e-5
    manifest = json.loads((run / "run-manifest.json").read_text())
    assert manifest["restart_mode"] == "post-cut" and manifest["reconnect"]["tau_end"] == 6e-5
    assert manifest["restart_from"]["path"] == str(production_cut["end"].resolve())
    assert manifest["restart_from"]["parent_record"]["derived_keys"] == [
        "case_id", "collapse.reconnect.tau_end", "collapse.state_dt"]
    chains = _columns(run / "chains.txt")
    assert chains["tau"][0] == pytest.approx(3e-5) and len(chains["tau"]) >= 3
    assert np.all(np.diff(chains["tau"]) > 0) and np.all(chains["bridge"] > 0)
    dumps = [json.loads(d.read_text()) for d in sorted((run / "states").glob("state_*.json"))]
    assert [d["step"] for d in dumps] == list(chains["step"][1:].astype(int))  # every step
    assert all(d["cut"]["settings"]["tau_end"] == 3e-5 for d in dumps)


FAILING_AT = """
import runpy, sys
sys.path.insert(0, sys.argv[1] + "/src")
from faraday_singularity import driver, surgery
from faraday_singularity.selfcontact import InterfaceSelfIntersection

def crossed(self, record):  # the fresh open chain crosses itself in the driver's measurement
    raise InterfaceSelfIntersection("the free surface crosses itself")

def planner_bug(self, problem, rmin):
    raise ValueError("a planner bug")

where = sys.argv[2]
if where == "postcut":
    driver.CollapseDriver.reset_after_cut = crossed
elif where == "plan":
    surgery.DriverTriggeredReconnection.request_pinch = planner_bug
else:  # every attempt deferred: the cut itself refuses
    surgery.DriverTriggeredReconnection.request_pinch = lambda self, problem, rmin: "deferral"
sys.argv = [sys.argv[1] + "/simulationCases/run_faraday.py"] + sys.argv[3:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


@pytest.mark.parametrize("where, outcome, failed_at, cause", [
    ("postcut", "failed", "postcut", "InterfaceSelfIntersection"),
    ("plan", "refused", "plan", "ValueError"),
    ("deferral", "refused", "plan", None)])
def test_production_runner_records_every_failure_of_a_cut_under_cut(landing_run, tmp_path, where,
                                                                    outcome, failed_at, cause):
    """A failure in the bookkeeping after the cut, an unexpected error in the plan and a refusal of
    every plan attempt all end the run as failed with the cut's record under ``cut`` in
    ``run_status.json``, the stage reached and the precut dump named; an unexpected error is the
    cause, with its traceback in ``cut_diagnostics.json``."""
    out = tmp_path / "runtime"
    proc = _run_faraday(where, landing_run["cut_case"], "--restart-from", landing_run["state"],
                        "--out", out, script=FAILING_AT)
    assert proc.returncode == 1, proc.stdout[-3000:] + proc.stderr[-3000:]
    status = json.loads((out / "run_status.json").read_text())
    assert status["state"] == "failed" and "collapse" not in status
    assert status["cut"]["outcome"] == outcome and status["cut"]["failed_at"] == failed_at
    precut = Path(status["cut"]["precut"])
    assert precut.is_file() and precut.parent == out / "states"
    rec = json.loads((out / "cut_diagnostics.json").read_text())
    assert rec["outcome"] == outcome and rec["failed_at"] == failed_at
    assert rec["precut_state"]["path"] == str(precut)
    if cause is None:
        assert "traceback" not in rec
        assert {a["outcome"] for a in rec["plan_attempts"]} == {"deferral"}
    else:
        assert cause in status["detail"] and cause in rec["error"]
        assert any(cause in line for line in rec["traceback"])
    # the precut dump records the neck it was cut at: a landing state for a new cut
    meta = json.loads(precut.with_suffix(".json").read_text())
    assert meta["cut_stage"] == "precut" and meta["neck"]["r_neck"] == pytest.approx(
        landing_run["r_neck"], rel=1e-12)


FAILING_CUT_RUN = """
import runpy, sys
sys.path.insert(0, sys.argv[1] + "/src")
from faraday_singularity import surgery

def broken_chart(_problem):
    raise RuntimeError("no chart")

surgery.orient_interface_chart = broken_chart
sys.argv = [sys.argv[1] + "/simulationCases/run_faraday.py"] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def test_production_runner_records_a_failed_cut(landing_run, tmp_path):
    out = tmp_path / "runtime"
    proc = _run_faraday(landing_run["cut_case"], "--restart-from", landing_run["state"], "--out",
                        out, script=FAILING_CUT_RUN)
    assert proc.returncode == 1, proc.stdout[-3000:] + proc.stderr[-3000:]
    status = json.loads((out / "run_status.json").read_text())
    assert status["state"] == "failed" and "no chart" in status["detail"]
    assert status["cut"]["outcome"] == "failed" and status["cut"]["failed_at"] == "chart"
    assert Path(status["cut"]["precut"]).is_file() and "collapse" not in status
    rec = json.loads((out / "cut_diagnostics.json").read_text())
    assert rec["outcome"] == "failed" and rec["failed_at"] == "chart"
    assert json.loads((out / "run-manifest.json").read_text())["restart_mode"] == "cut"


# --- the ordinary path ----------------------------------------------------------------------------


HAIRPIN_RUN = """
import runpy, sys
sys.path.insert(0, sys.argv[1] + "/src")
import numpy as np
from faraday_singularity import driver, surgery

original = driver.CollapseDriver.reset_after_cut


def hairpin(self, record):  # right after the cut: a midside node 0.55 chords off its chord
    m = original(self, record)
    pb = self.pb
    seg = next(c for c in pb.interface_segments() if not c["open"])
    P = np.column_stack([seg["r"], seg["z"]])
    k = 2 * int(np.argmax(P[1::2, 0])) + 1
    a, b = P[k - 1], P[k + 1]
    L = float(np.hypot(*(b - a)))
    t = (b - a) / L
    new = 0.5 * (a + b) + 0.55 * L * np.array([-t[1], t[0]])  # towards the gas
    node = next(n for n in surgery.mesh_nodes(pb.get_mesh("liquid/interface"))
                if (n.x(0), n.x(1)) == (P[k, 0], P[k, 1]))
    for level in range(node.ntstorage()):
        for i in (0, 1):
            node.set_x_at_t(level, i, float(new[i]))
    for i in (0, 1):
        node.set_x_lagr(i, float(new[i]))
    pb.invalidate_cached_mesh_data()
    self._post_cut_cache = None
    return m


driver.CollapseDriver.reset_after_cut = hairpin
sys.argv = [sys.argv[1] + "/simulationCases/run_faraday.py"] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def test_production_runner_stops_cleanly_on_an_unresolved_interface(landing_run, tmp_path):
    """An element of the bubble whose midside node stands 0.55 chords off its chord after the cut,
    the hairpin of 131 degrees on which a production run failed with a Newton residual of inf at
    every step size, ends the run before its first step as a clean stop: interface-unresolved,
    trigger edge-turn (above the 90 degrees of a freshly built mesh), the end state dumped, and the
    run completed, not failed."""
    out = tmp_path / "runtime"
    proc = _run_faraday(landing_run["cut_case"], "--restart-from", landing_run["state"], "--out",
                        out, script=HAIRPIN_RUN)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    status = json.loads((out / "run_status.json").read_text())
    assert status["state"] == "completed" and status["cut"]["outcome"] == "cut"
    stop = status["collapse"]["stop"]
    assert stop["reason"] == "interface-unresolved" and stop["trigger"] == "edge-turn"
    assert stop["chain"] == "bubble" and stop["turn"] == pytest.approx(131.1, abs=0.5)
    assert stop["fresh"] is True and stop["limit"] == 90.0 and status["collapse"]["steps"] == 0
    end = Path(stop["end_state"])
    assert end.is_file() and end.with_suffix(".json").is_file() and end.parent == out / "states"
    events = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "stop" and events[-1]["reason"] == "interface-unresolved"
    assert _columns(out / "chains.txt")["edge_turn_bubble"][0] == pytest.approx(131.1, abs=0.5)


def _reference_mesh_class():
    """``FaradayCylinderMesh`` as at commit 9e0d8a1 (on pyoomph's plain ``GmshTemplate``)."""
    try:
        src = subprocess.run(["git", "-C", str(ROOT), "show",
                              "9e0d8a1:src/faraday_singularity/mesh.py"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("commit 9e0d8a1 is not available")
    mod = types.ModuleType("faraday_singularity._mesh_9e0d8a1")
    mod.__package__ = "faraday_singularity"
    sys.modules[mod.__name__] = mod  # dataclasses resolve annotations through it
    exec(compile(src, "mesh_9e0d8a1.py", "exec"), mod.__dict__)
    base = mod.FaradayCylinderMesh

    def __init__(self, *args, reconnect=False, **kw):
        base.__init__(self, *args, **kw)

    return type("FaradayCylinderMesh", (base,), {"__init__": __init__})


DRIVER_RUN = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from faraday_singularity.driver import CollapseDriver, CollapseSettings
from faraday_singularity.mesh import MeshSettings
from faraday_singularity.parameters import CaseParameters, nondimensionalise
from faraday_singularity.problem import FaradayProblem

out = Path(sys.argv[2])
physics = json.loads(sys.argv[3])
nd = nondimensionalise(CaseParameters.from_dict(physics))
mesh = MeshSettings(h_interface=0.05, h_axis=0.05, h_max=0.4, grading=0.3, interface_points=201)
# a refine ratio below one rebuilds the focus every few steps, so that the run goes through the
# driver's remesh path too; every step is dumped, unless the fourth argument sets another state_dt
state_dt = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
settings = CollapseSettings(t_end=1.0, dt_max=0.02, max_steps=6, refine_ratio=0.5,
                            remesh_backoff=2, state_dt=state_dt, state_keep=3, snapshot_dt=0.03)
pb = FaradayProblem(nd, mesh=mesh)
pb.set_output_directory(str(out))
with pb:
    pb.initialise()
    driver = CollapseDriver(pb, settings, out)
    pb.start_diagnostics(out / "diagnostics.txt")
    result = driver.run(1e-8)
    driver.close()
    pb.close_diagnostics()
(out / "result.json").write_text(json.dumps(result))
"""


def _close(a, b, path=""):
    """``a`` and ``b`` equal in structure and keys, their numbers to round-off."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), (path, sorted(set(a) ^ set(b)))
        for k in a:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for k, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{k}]")
    elif isinstance(a, float) and not isinstance(a, bool):
        assert isinstance(b, (int, float)), path
        assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, rel=1e-10, abs=1e-13), \
            (path, a, b)
    else:
        assert a == b, (path, a, b)


def _table(path: Path):
    lines = path.read_text().splitlines()
    return lines[0], np.loadtxt(path, ndmin=2)


def test_the_ordinary_driver_runs_as_at_9e0d8a1(tmp_path):
    """The collapse driver and problem of this commit, run on a small ordinary case (no reconnection
    key) for a few steps with rebuilds and dumps, against those of commit 9e0d8a1, each in a fresh
    process with MKL single-threaded: the same output files, columns, events and dump keys, and the
    same numbers to round-off (the wall time aside); the result only gains the counts of the
    rebuilds element distortion asked for, none in this run. With MKL on one thread they have
    agreed to the byte; the test asks for round-off only."""
    ref = tmp_path / "ref"
    try:
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", "9e0d8a1", "src"],
                                 capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("commit 9e0d8a1 is not available")
    ref.mkdir()
    subprocess.run(["tar", "-x", "-C", str(ref)], input=archive, check=True)
    physics = dict(PHYSICS, forcing={"Gamma": 0.3, "detuning": 1.0, "phase": 0.0},
                   initial={"amplitude": 0.2}, container={"mode": 1, "depth": 2.0})
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    runs = {}
    for name, src in (("9e0d8a1", ref / "src"), ("head", ROOT / "src")):
        out = tmp_path / name
        proc = subprocess.run([sys.executable, "-c", DRIVER_RUN, str(src), str(out),
                               json.dumps(physics)], capture_output=True, text=True, timeout=900,
                              cwd=tmp_path, env=env, check=False)
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
        runs[name] = out
    a, b = runs["9e0d8a1"], runs["head"]
    names = {f.relative_to(a).as_posix() for f in a.rglob("*") if f.is_file()
             and not f.relative_to(a).parts[0].startswith("_")}
    assert names == {f.relative_to(b).as_posix() for f in b.rglob("*") if f.is_file()
                     and not f.relative_to(b).parts[0].startswith("_")}
    assert (a / "collapse_settings.json").read_text() == (b / "collapse_settings.json").read_text()
    result = json.loads((a / "result.json").read_text())
    assert result["steps"] == 6 and result["remesh_count"] >= 2, result
    # the result gains the counts of the rebuilds element distortion asked for: none here
    head = json.loads((b / "result.json").read_text())
    assert (head.pop("distortion_remeshes"), head.pop("pyoomph_remeshes")) == (0, 0)
    _close(result, head, "result")
    head_a, rows_a = _table(a / "collapse.txt")
    head_b, rows_b = _table(b / "collapse.txt")
    assert head_a == head_b
    wall = head_a[2:].split("\t").index("wall")
    keep = [k for k in range(rows_a.shape[1]) if k != wall]
    assert np.allclose(rows_a[:, keep], rows_b[:, keep], rtol=1e-10, atol=1e-13, equal_nan=True)
    events = [[json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
              for run in (a, b)]
    assert [e["event"] for e in events[0]] == [e["event"] for e in events[1]]
    for e, f in zip(*events):
        _close(e, f, e["event"])
    for name in sorted(n for n in names if n.startswith("states/") and n.endswith(".json")):
        _close(json.loads((a / name).read_text()), json.loads((b / name).read_text()), name)
    for name in sorted(n for n in names if n.startswith("profiles/") or n == "diagnostics.txt"):
        assert (a / name).read_text().splitlines()[0] == (b / name).read_text().splitlines()[0]
        assert np.allclose(np.loadtxt(a / name), np.loadtxt(b / name), rtol=1e-10, atol=1e-13), name


def test_pre_cut_runs_are_as_at_4d2ea17(tmp_path):
    """Before a cut nothing has changed: the short ordinary run of the collapse driver above, with
    rebuilds and dumps, gives the same output files, to the byte (the wall time aside), as the
    driver and problem of commit 4d2ea17, each in a fresh process with MKL on one thread."""
    ref = tmp_path / "ref"
    try:
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", "4d2ea17", "src"],
                                 capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("commit 4d2ea17 is not available")
    ref.mkdir()
    subprocess.run(["tar", "-x", "-C", str(ref)], input=archive, check=True)
    physics = dict(PHYSICS, forcing={"Gamma": 0.3, "detuning": 1.0, "phase": 0.0},
                   initial={"amplitude": 0.2}, container={"mode": 1, "depth": 2.0})
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    runs = {}
    for name, src_dir in (("4d2ea17", ref / "src"), ("head", ROOT / "src")):
        out = tmp_path / name
        proc = subprocess.run([sys.executable, "-c", DRIVER_RUN, str(src_dir), str(out),
                               json.dumps(physics)], capture_output=True, text=True, timeout=900,
                              cwd=tmp_path, env=env, check=False)
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
        runs[name] = out
    a, b = runs["4d2ea17"], runs["head"]
    names = sorted(f.relative_to(a).as_posix() for f in a.rglob("*") if f.is_file()
                   and not f.relative_to(a).parts[0].startswith("_"))
    assert names == sorted(f.relative_to(b).as_posix() for f in b.rglob("*") if f.is_file()
                           and not f.relative_to(b).parts[0].startswith("_"))
    assert json.loads((a / "result.json").read_text())["remesh_count"] >= 2
    head_a, rows_a = _table(a / "collapse.txt")
    head_b, rows_b = _table(b / "collapse.txt")
    wall = head_a[2:].split("\t").index("wall")
    keep = [k for k in range(rows_a.shape[1]) if k != wall]
    assert head_a == head_b and np.array_equal(rows_a[:, keep], rows_b[:, keep], equal_nan=True)
    for name in names:
        if name != "collapse.txt" and not name.endswith(".dump"):
            assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_dumps_do_not_change_the_trajectory(tmp_path):
    """A rotating dump writes the state and does not change it: the short run of the driver above
    with a dump at every step and with one at its first step only, each in a fresh process with MKL
    on one thread, gives the same rows (but the wall time), events, profiles and diagnostics, to
    the byte. This is why a derived case may change the cadence and number of dumps
    (``collapse.state_dt``, ``collapse.state_keep``; ``run_case.DERIVED_KEYS``)."""
    physics = dict(PHYSICS, forcing={"Gamma": 0.3, "detuning": 1.0, "phase": 0.0},
                   initial={"amplitude": 0.2}, container={"mode": 1, "depth": 2.0})
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    runs = {}
    for name, state_dt in (("every", "0.0"), ("first", "10.0")):
        out = tmp_path / name
        proc = subprocess.run([sys.executable, "-c", DRIVER_RUN, str(ROOT / "src"), str(out),
                               json.dumps(physics), state_dt], capture_output=True, text=True,
                              timeout=900, cwd=tmp_path, env=env, check=False)
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
        runs[name] = out
    a, b = runs["every"], runs["first"]
    assert len(list((a / "states").glob("state_*.dump"))) == 3  # the last state_keep of six
    assert [p.name for p in (b / "states").glob("state_*.dump")] == ["state_00000001.dump"]
    result = json.loads((a / "result.json").read_text())
    assert result == json.loads((b / "result.json").read_text()) and result["remesh_count"] >= 2
    head_a, rows_a = _table(a / "collapse.txt")
    head_b, rows_b = _table(b / "collapse.txt")
    wall = head_a[2:].split("\t").index("wall")
    keep = [k for k in range(rows_a.shape[1]) if k != wall]
    assert head_a == head_b
    assert np.array_equal(rows_a[:, keep], rows_b[:, keep], equal_nan=True)
    assert (a / "events.jsonl").read_text() == (b / "events.jsonl").read_text()
    names = sorted(f.relative_to(a).as_posix() for f in (a / "profiles").glob("*.txt"))
    assert names == sorted(f.relative_to(b).as_posix() for f in (b / "profiles").glob("*.txt"))
    for name in names + ["diagnostics.txt"]:
        assert (a / name).read_text() == (b / name).read_text(), name


def test_ordinary_meshes_steps_and_dumps_are_unchanged(tmp_path):
    ref_cls = _reference_mesh_class()
    physics = dict(PHYSICS, forcing={"Gamma": 0.3, "detuning": 1.0, "phase": 0.0},
                   initial={"amplitude": 0.2}, container={"mode": 1, "depth": 2.0})
    nd = nondimensionalise(CaseParameters.from_dict(physics))
    mesh = MeshSettings(h_interface=0.05, h_axis=0.05, h_max=0.4, grading=0.3,
                        interface_points=201)

    class RefProblem(FaradayProblem):
        mesh_class = ref_cls

    runs = {}
    for name, cls, kw in (("ref", RefProblem, {}), ("new", FaradayProblem, {}),
                          ("recon", FaradayProblem, {"reconnect": ReconnectSettings(delta=1e-3)})):
        out = tmp_path / name
        pb = cls(nd, mesh=mesh, **kw)
        pb.set_output_directory(str(out))
        with pb:
            pb.initialise()
            first = (out / "_gmsh" / "FaradayCylinderMesh.msh").read_bytes()
            pb.active_mesh_template().focus = Focus(size=0.02, radius=0.5, grading=0.2)
            pb.force_remesh()
            remeshed = max((out / "_gmsh").glob("*_REMESH_*.msh"))
            for _ in range(3):
                _step(pb, 0.02)
            vec = np.concatenate(pb.interface_arrays())
            pb.save_state(str(out / "state.dump"), quiet=True)
            runs[name] = {"first": first, "remeshed": remeshed.read_bytes(), "vec": vec,
                          "dump": out / "state.dump"}
    # The meshes are the same to the byte. The steps are compared to 1e-12 only: unless MKL runs
    # on one thread (MKL_NUM_THREADS=1), even two runs of one problem differ at about 1e-14, in
    # one process or in fresh ones. With MKL on one thread the ordinary problem reproduces the
    # reference to the byte; the reconnection-enabled one, compiled with its extra residual and
    # observables, does not: it departs at round-off from the third step (about 1e-14 relative
    # after 10 fixed steps of this case), deterministically, and the step control of the collapse
    # driver carries that on. This is why a restart may not add, drop or change a reconnect block
    # (run_case.DERIVED_KEYS) unless it cuts.
    scale = float(np.max(np.abs(runs["ref"]["vec"])))
    for name in ("new", "recon"):
        assert runs[name]["first"] == runs["ref"]["first"]
        assert runs[name]["remeshed"] == runs["ref"]["remeshed"]
        assert np.max(np.abs(runs[name]["vec"] - runs["ref"]["vec"])) <= 1e-12 * scale
    # dumps of the ordinary problem load into the reconnection-enabled one, and back
    for src, cls, kw in (("ref", FaradayProblem, {"reconnect": ReconnectSettings(delta=1e-3)}),
                         ("recon", FaradayProblem, {})):
        out = tmp_path / f"load_{src}"
        pb = cls(nd, mesh=mesh, **kw)
        pb.set_output_directory(str(out))
        with pb:
            pb.initialise()
            pb.load_state(str(runs[src]["dump"]))
            assert np.array_equal(np.concatenate(pb.interface_arrays()), runs[src]["vec"])
            _step(pb, 0.02)  # and continues
            assert len(pb.interface_segments()) == 1


def test_after_a_cut_each_step_makes_two_newton_iterations_whatever_its_tolerance(tmp_path):
    """At the steps after a cut (dt about 1e-9) the driver's tolerance, newton_floor/dt, passed the
    first Newton iterate, whose systematic error grew into noise at the scale of the elements. Around
    each step after a cut the driver asks for ``reconnect.newton_min_iterations`` (2) iterations:
    with a tolerance that the first iterate passes, a step then makes two and lands next to the
    converged solve. The tolerance, the iteration rule and the compression of the dumps are given
    back after the step; the snapshot pyoomph keeps before it is uncompressed, and lossless."""
    s = ReconnectSettings(**FAST_CUT)
    with neck_problem(tmp_path, s) as pb:
        pb.initialise()
        impose_closing_flow(pb)
        rec = surgery.cut(pb, s, tmp_path)
        post = str(rec["postcut_state"]["path"])
        # Each call below reloads the post-cut mesh before entering the context.
        driver = types.SimpleNamespace(pb=pb, reconnect=s, _fresh_mesh=lambda: True)
        tol0, level0 = pb.newton_solver_tolerance, pb.states_compression_level

        def step(tol, rule):
            pb.load_state(post)
            t = pb.get_current_time(dimensional=False, as_float=True)
            with CollapseDriver._post_cut_solve(driver) if rule else contextlib.nullcontext():
                if rule:
                    assert pb.min_newton_iterations == 2 and pb.states_compression_level is None
                pb.run(t + 1e-5, timestep=1e-5, maxstep=1e-5, outstep=False, temporal_error=1.0,
                       newton_solver_tolerance=tol)
            return len(pb.get_last_residual_convergence()) - 1, _state_vector(pb)

        one, x_one = step(1e2, rule=False)
        two, x_two = step(1e2, rule=True)
        converged, x_conv = step(1e-11, rule=False)
        assert (one, two) == (1, 2) and converged >= 2
        assert pb.min_newton_iterations == 1 and pb._held_tolerance is None
        assert pb.newton_solver_tolerance == tol0 and pb.states_compression_level == level0
        scale = np.max(np.abs(x_conv))
        err_one = np.max(np.abs(x_one - x_conv)) / scale
        err_two = np.max(np.abs(x_two - x_conv)) / scale
        assert err_two < 1e-2 * err_one, (err_one, err_two)  # quadratic: about err_one ** 2
        # a tolerance the first iterate does not pass: the rule adds no iteration
        assert step(1e-11, rule=True)[0] == converged
        # the snapshot of the state, uncompressed and compressed, holds the same state
        pb.load_state(post)
        x0 = _state_vector(pb)
        packed = pb._snapshot_state()
        pb.states_compression_level = None
        plain = pb._snapshot_state()
        pb.states_compression_level = level0
        assert len(plain) > len(packed)
        for snap in (plain, packed):
            pb._restore_state(snap)
            assert np.array_equal(_state_vector(pb), x0)


def test_the_jets_are_probed_in_one_location_and_each_integral_once_per_state(tmp_path,
                                                                              monkeypatch):
    """pyoomph builds a locator over the whole liquid mesh at each call of evaluate_at_points: the
    driver locates the discs of every jet in one call, with the same values as one call per disc.
    The integrals that depend on the current state only are evaluated once for it, whichever of a
    step's diagnostics, the driver's row and the budgets after a cut asks first; a step, a rebuild
    or a reload evaluates them again."""
    s = ReconnectSettings(**FAST_CUT)
    with neck_problem(tmp_path, s) as pb:
        pb.initialise()
        impose_closing_flow(pb)
        rec = surgery.cut(pb, s, tmp_path)
        geo = rec["geometry"]
        z_mid = 0.5 * (geo["z_bubble_top"] + geo["z_cavity_tip"])
        sets = [[(x, z) for x in np.linspace(0.0, 0.01, 7)]
                for z in (z_mid, geo["z_bubble_top"] - 0.05, geo["z_cavity_tip"] - 0.2)]
        sets.append([(0.0, 50.0)])  # outside the liquid
        batched = pb.axial_velocity_sets(sets)
        single = [pb.axial_velocity_at(p) for p in sets]
        assert [f for _, f in batched] == [f for _, f in single]
        for (w, _), (w1, _) in zip(batched, single):
            np.testing.assert_array_equal(w, w1)  # to the bit, NaN where a point is not found
        assert batched[0][1] and not batched[-1][1]  # the bridge; and outside the liquid
        assert pb.axial_velocity_sets([]) == []

        calls = []
        real = pb.get_mesh

        class Counting:
            def __init__(self, mesh):
                self._mesh = mesh

            def evaluate_observable(self, name):
                calls.append(name)
                return self._mesh.evaluate_observable(name)

        def counting(name):
            return Counting(real(name)) if name in ("liquid", "liquid/interface") else real(name)

        def evaluate_all():
            monkeypatch.setattr(pb, "get_mesh", counting)
            try:
                return pb.observables(), pb.observables(), pb.cut_observables()
            finally:
                monkeypatch.setattr(pb, "get_mesh", real)

        pb._integral_cache = None  # the cut's budgets have evaluated some for this state
        first, again, budgets = evaluate_all()
        assert first == again and calls == ["volume", "kinetic_energy", "dissipation", "area",
                                            "amp", "momentum_z", "divergence_sq",
                                            "projection_change", "pressure_sq"]
        for mesh, name in (("liquid", "volume"), ("liquid", "kinetic_energy"),
                           ("liquid", "dissipation"), ("liquid/interface", "area")):
            assert first[name] == float(real(mesh).evaluate_observable(name))
        assert budgets["volume"] == first["volume"] and budgets["area"] == first["area"]
        for change in (lambda: _step(pb, 1e-5), pb.force_remesh,
                       lambda: pb.load_state(str(rec["postcut_state"]["path"]))):
            calls.clear()
            change()
            now = evaluate_all()[0]
            assert calls[:5] == ["volume", "kinetic_energy", "dissipation", "area", "amp"]
            assert now["kinetic_energy"] == float(real("liquid").evaluate_observable(
                "kinetic_energy"))


def test_after_a_cut_the_step_can_leave_the_rule_of_the_focus_to_every_node(tmp_path):
    """``reconnect.focus_courant`` off: the step after the cut is bounded by every node's own
    Courant limit (and the capillary time of the sharpest feature), not by the finest focus size
    over the largest speed anywhere; the default keeps both."""
    steps = {}
    for flag in (True, False):
        out = tmp_path / str(flag)
        out.mkdir()
        s = ReconnectSettings(**dict(FAST_CUT, focus_courant=flag, dt_growth=4.0))
        # the focus's rule made strict (the collapse's courant), the nodes' own as before
        settings = CollapseSettings(t_end=10.0, dt_max=0.01, max_steps=4, growth=4.0,
                                    courant=0.02)
        with neck_problem(out, s) as pb:
            pb.initialise()
            impose_closing_flow(pb)
            driver = CollapseDriver(pb, settings, out, reconnect=s)
            surgery.cut(pb, s, out, driver=driver, log=lambda *_: None)
            result = driver.run(1e-8)
            driver.close()
        assert result["stop"]["reason"] == "max-steps"
        table = _columns(out / "collapse.txt")
        chains = _columns(out / "chains.txt")
        steps[flag] = (table, chains)
    on, off = steps[True], steps[False]
    # the first step, from the state of the first row (later rows precede a rebuild of the step)
    assert on[0]["dt"][1] <= settings.courant * on[0]["h_focus"][0] / on[0]["U"][0] * (1 + 1e-12)
    assert off[0]["dt"][1] <= off[1]["dt_courant"][0] * (1 + 1e-12)
    assert off[0]["dt"][1:].sum() > on[0]["dt"][1:].sum()
