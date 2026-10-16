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


def test_the_cut_mesh_is_sized_in_cut_radii_whatever_the_landing_focus(tmp_path):
    """The tips, the bridge and the focus of the cut's remesh are fixed fractions of the cut radius:
    two landing states that differ only in the focus their run left are cut onto the same mesh."""
    meshes, records = [], []
    for k, landing in enumerate((Focus(size=R_N / 3, radius=0.1, grading=0.3),
                                 Focus(size=R_N / 12, radius=0.6, grading=0.3))):
        out = tmp_path / f"cut{k}"
        s = ReconnectSettings(**FAST_CUT)
        with neck_problem(out, s) as pb:
            pb.initialise()
            impose_closing_flow(pb)
            pb.active_mesh_template().focus = landing
            rec = surgery.cut(pb, s, out)
            meshes.append(mesh_signature(pb))
            records.append(rec)
            assert pb.active_mesh_template().focus == Focus(**rec["focus"]["cut"])
    assert meshes[0] == meshes[1]
    for rec, landing in zip(records, (R_N / 3, R_N / 12)):
        rmin = rec["rmin"]
        assert rec["focus"]["landing"]["size"] == landing
        assert rec["focus"]["cut"] == {"size": pytest.approx(0.25 * rmin),
                                       "radius": pytest.approx(20 * rmin), "grading": 0.3}
        assert rec["sizing"]["h_tip"] == pytest.approx(0.25 * rmin)
        assert rec["sizing"]["h_bridge"] == pytest.approx(0.25 * rmin)
        assert rec["sizing_in_cut_radii"] == {"h_tip": 0.25, "h_bridge": 0.25, "focus_size": 0.25,
                                              "focus_radius": 20.0}


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


def test_after_a_cut_the_driver_rebuilds_the_mesh_only_when_a_coarsening_takes_effect(tmp_path):
    """Every mesh after the cut keeps the cut's sizing at the tips, here h_tip = delta/32. Read at
    the open chain's tip, the spacing near the collapse stayed at about 1.17 h_tip whatever focus
    the driver set, and the driver rebuilt the whole mesh at each of 6 steps to coarsen it, without
    effect. Read where the focus sets it, the coarsening takes effect and then stops."""
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
    events = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    remesh = [e for e in events if e["event"] == "remesh"]
    assert result["stop"]["reason"] == "max-steps" and result["steps"] == 6
    assert 1 <= len(remesh) <= 2 and result["remesh_count"] == 1 + len(remesh)  # the cut's own
    assert not [e for e in events if e["event"] == "remesh-ineffective"]
    for e in remesh:  # coarsenings that coarsened
        assert e["focus_size"] > e["h_near_before"] and e["h_near_after"] > e["h_near_before"] / 0.9
    table = _columns(tmp_path / "collapse.txt")
    h_tip = rec["sizing"]["h_tip"]
    assert np.all(table["h_near"] > CUT_SIZING_MARGIN * h_tip)  # read away from the tips
    assert table["h_near"][-1] >= settings.coarsen_ratio * table["h_focus"][-1]


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
    same numbers to round-off (the wall time aside). With MKL on one thread they have agreed to the
    byte; the test asks for round-off only."""
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
    _close(result, json.loads((b / "result.json").read_text()), "result")
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
