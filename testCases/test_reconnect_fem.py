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
* the prototype entry point on a landing state of that cavity, with the lineage of its parent run;
* the ordinary path: meshes as with the mesh class of commit 9e0d8a1 (to the byte), steps
  (to round-off), and dumps that load across the two kinds of problem.

Each problem compiles in seconds; the module takes about a minute.
"""

from __future__ import annotations

import hashlib
import json
import math
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
    s = ReconnectSettings(delta=1.05 * R_N, dt0=1e-5)
    res = {}
    with neck_problem(out, s) as pb:
        pb.initialise()
        impose_closing_flow(pb)
        res["neck"] = neck_radius(pb.interface_segments()[0]["r"], pb.interface_segments()[0]["z"])
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
        s2 = ReconnectSettings(delta=1.05 * R_N, dt0=1e-5, fill="zero", restart="impulsive")
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
    assert rec == neck_cut["file"]
    assert [a["outcome"] for a in rec["plan_attempts"]] == ["plan"]
    assert rec["plan"]["events"][0]["kind"] == "pinch"
    chains = rec["mesh"]["chains"]
    assert sorted(c["open"] for c in chains) == [False, True]
    assert len(rec["mesh"]["axis_spans"]) == 2
    for got, want in zip(rec["mesh"]["axis_spans"], rec["mesh"]["expected_axis_spans"]):
        assert got == pytest.approx(want, abs=1e-12)
    g = rec["geometry"]
    assert g["z_bubble_top"] < g["z_cavity_tip"] and g["bridge"] > 2 * rec["delta"]
    for cap in rec["caps"]:
        assert cap["min_angle"] >= 30.0 and cap["r_s"] >= rec["delta"]


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
from faraday_singularity.reconnect import ReconnectSettings

post, out = Path(sys.argv[2]), Path(sys.argv[3])
meta = json.loads(post.with_suffix(".json").read_text())
report = {}
with t.neck_problem(out, ReconnectSettings(delta=1.05 * t.R_N, dt0=1e-5)) as pb:
    pb.initialise()
    for mode in ("plain", "restored"):
        pb.load_state(str(post))
        if mode == "restored":
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
    assert np.max(np.abs(a - b)) <= 1e-10 * np.max(np.abs(a))  # not bit for bit: the .msh text
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


@pytest.mark.parametrize("ratio", [1.2, 2.0])
def test_cut_is_refused_far_above_the_neck(tmp_path, ratio):
    s = ReconnectSettings(delta=ratio * R_N)
    with neck_problem(tmp_path, s) as pb:
        pb.initialise()
        with pytest.raises(surgery.CutRefused, match="above"):
            surgery.cut(pb, s, tmp_path)
        assert pb.cut is None and not pb.get_mesh("liquid")._templatemesh.has_pending_surgery_plan()


def test_cut_is_refused_too_close_to_the_neck(tmp_path):
    s = ReconnectSettings(delta=1.01 * R_N)
    with neck_problem(tmp_path, s) as pb:
        pb.initialise()
        with pytest.raises(surgery.CutRefused, match="below"):
            surgery.cut(pb, s, tmp_path)
        assert pb.cut is None and not pb.get_mesh("liquid")._templatemesh.has_pending_surgery_plan()


# --- the prototype entry point ----------------------------------------------------------------------


def test_prototype_cuts_a_landing_state_and_steps(tmp_path):
    landing = tmp_path / "landing"
    landing.mkdir()
    s = ReconnectSettings(delta=1.05 * R_N)
    with neck_problem(landing, s) as pb:
        pb.initialise()
        impose_closing_flow(pb)
        driver = CollapseDriver(pb, CollapseSettings(t_end=10.0, dt_max=0.01), landing)
        state = driver.keep_state(landing / "states" / "stop_00000000.dump", stop={})
        driver.close()
    parent_case = {"schema": "pyoomph-case-v1", "case_id": "neck-landing",
                   "geometry": {"kind": "axisymmetric"}, "physics": PHYSICS, "mesh": MESH,
                   "tolerances": {"newton": 1e-8},
                   "time_stepping": {"scheme": "BDF2", "steps_per_forcing_period": 100},
                   "collapse": {"keep_stop_state": True},
                   "output": {"cadence": 0.25, "cadence_unit": "forcing periods",
                              "write_states": False},
                   "completion": {"kind": "event", "t_max": 10, "unit": "forcing periods"}}
    # the parent run's records, as run_case writes them (a stand-in commit)
    parent_path = tmp_path / "landing-case.json"
    parent_path.write_text(json.dumps(parent_case))
    digest = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    (landing / "run-manifest.json").write_text(json.dumps(
        {"schema": "pyoomph-runner-manifest-v1", "component_commit": "b" * 40,
         "case_sha256": digest}))
    (landing / "parameters.json").write_text(json.dumps(
        {"case": parent_case, "provenance": {"case_sha256": digest}}))
    case = json.loads(json.dumps(parent_case))
    case["case_id"] = "neck-prototype"
    case["collapse"] = {"reconnect": {"delta": 1.0, "dt0": 1e-5}}
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case))
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, str(ROOT / "simulationCases" / "reconnect_prototype.py"),
                           str(case_path), "--state", str(state), "--outdir", str(out),
                           "--delta-ratio", "1.05", "--steps", "2"],
                          capture_output=True, text=True, timeout=900, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    status = json.loads((out / "run_status.json").read_text())
    assert status["state"] == "completed" and status["steps"] == 2
    cut = json.loads((out / "cut_diagnostics.json").read_text())
    assert cut["delta"] == pytest.approx(1.05 * cut["r_neck"])
    rows = np.loadtxt(out / "steps.txt")
    assert rows.shape[0] == 3
    cols = (out / "steps.txt").read_text().splitlines()[0][2:].split("\t")
    tau, dt = rows[:, cols.index("tau")], rows[:, cols.index("dt")]
    assert tau[0] == 0.0 and dt[1] == pytest.approx(1e-5) and dt[2] == pytest.approx(1.25e-5)
    # the Courant limit uses the smallest element near the cut, at most the cut's tip size
    h_cut, U_cut = rows[1:, cols.index("h_cut")], rows[1:, cols.index("U_cut")]
    assert np.all(h_cut <= cut["sizing"]["h_tip"]) and np.all(h_cut > 0)
    assert np.all(dt[1:] <= 0.2 * h_cut / U_cut * (1 + 1e-12))
    assert np.all(np.isfinite(rows[:, cols.index("momentum_z")]))
    for k in range(3):
        for kind in ("open", "bubble"):
            assert (out / "profiles" / f"postcut_{k:04d}_{kind}.txt").is_file()
    for name in (f"precut_{cut['delta']:.6g}", f"postcut_{cut['delta']:.6g}"):
        assert (out / "states" / f"{name}.dump").is_file()
        assert (out / "states" / f"{name}.json").is_file()
    events = [json.loads(x)["event"] for x in (out / "events.jsonl").read_text().splitlines()]
    assert events[:2] == ["restart", "cut"]
    manifest = json.loads((out / "run-manifest.json").read_text())
    assert manifest["state"]["path"] == str(state.resolve())
    parent = manifest["state"]["parent_record"]
    assert parent["case_sha256"] == digest and parent["component_commit"] == "b" * 40
    assert parent["derived_keys"] == ["case_id", "collapse.keep_stop_state", "collapse.reconnect"]


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
    # The meshes are the same to the byte. The steps are compared to round-off only: with a
    # multithreaded MKL two runs of the same problem in one process already differ at 1e-14
    # (single-threaded, the ordinary problem reproduces the reference bit for bit; the
    # reconnection-enabled one, compiled with its extra residual, to round-off).
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
