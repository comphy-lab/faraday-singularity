"""Finite-element tests of the cut of a liquid neck after the gas cut (``surgery.liquid_cut``).

The synthetic state is the one the gas cut leaves once the jets it launched have formed necks: an
open chain whose tip is the head of a jet rising from the trough of a groove, and a bubble whose
top holds a jet hanging from the apex of the groove around it, each jet a root, a neck of
``R_NECK delta`` and a bulb of radius ``delta/2`` whose tip is on the axis (``test_events_stops.
head_feature``, at ``delta = DELTA``), built as the first mesh of the problem with a synthetic
record of the gas cut. The velocity is imposed (a "kinematic" cut): a solenoidal flow about each
neck that closes it and stretches it axially, so that each head moves away from its stump, or a
radial inflow that is not solenoidal.

* pyoomph's own parking of a plan, which the cut's ``park_plan`` copies (on the necked cavity of
  ``test_reconnect_fem.py``);
* the identity of the box's cutoff in the projection, on the manufactured field of
  ``test_reconnect_fem.py`` (``u~ = u_s + grad(phi)``);
* two cuts, one on each chain, each landed as the events branch requests it: the transfer leaves
  unlocated exactly the interface nodes the geometry predicts, all on the cap, and no other; the
  stump's tip on the axis is the only node without a value; the chains and the axis spans are those
  planned; the budgets close (the removed volume, momentum and kinetic energy against the box's,
  the projection's change of momentum and energy there, the area); steps after both, and a restart
  from the dump after the second in a new process;
* the cap fill against the imposed field: with it the cap's nodes carry the old bulk velocity at
  their places, without it (``cap_fill`` ``transfer``) the old surface's near the neck;
* the hard checks: a projection that changes the momentum in the box by more than its limit fails
  the cut (the non-solenoidal flow), and the identity of the cutoff holds there with a large
  pressure impulse; a transfer that leaves more interface nodes unlocated than the slack of the
  prediction allows fails it after its remesh; requests without a single head, of a class without this handler, pairing a
  kind with another's class, or whose neck does not land under the cut's own keys are refused at
  admission, and a cap, a plan or its parking refused after the state before the cut is kept: each
  refusal leaves the problem unchanged.

Each problem compiles in seconds; the module takes about three minutes.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

pytest.importorskip("pyoomph")

import test_reconnect_fem as fem  # noqa: E402
from test_events_stops import head_feature  # noqa: E402

from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402
from faraday_singularity import surgery  # noqa: E402
from faraday_singularity.mesh import (FaradayCylinderMesh, Focus, FocusRegion,  # noqa: E402
                                      MeshSettings)
from faraday_singularity.parameters import CaseParameters, nondimensionalise  # noqa: E402
from faraday_singularity.problem import FaradayProblem  # noqa: E402

R_C = 3.8317059702075125
DELTA = 0.05  # the cut radius of the synthetic gas cut
Z_BUBBLE, Z_CAVITY = 0.80, 0.90  # the apex of the groove at the bubble's top; the cavity's trough
R_NECK = 0.09  # the necks in units of delta, under heads of radius delta/2: landed at eps_d 0.2
R_BUBBLE = 0.2  # the bubble's radius below its top
EPS = 8.0  # elements per neck radius
A_FLOW = 20.0  # the strength of the imposed flow about each neck
RECON = rc.ReconnectSettings(delta=DELTA, dt0=1e-5, tip_size=0.25, bridge_size=0.25,
                             focus_size=0.25)
EVENTS = ev.EventSettings.from_dict({"liquid": {"enabled": True},
                                     "post": {"open_neck_stop": False, "neck_floor": 1e-5}})
CUT_KW = {"dt_prev": 1e-5, "elements_per_scale": EPS, "h_floor": 1e-6}
MESH = {"h_interface": 0.01, "h_axis": 0.01, "h_max": 0.25, "grading": 0.3,
        "interface_points": 401}


# --- the synthetic state --------------------------------------------------------------------------


def _resample(P, h_fine, h_far, centres, grading=0.15):
    """Points along the dense polyline ``P``, ``h_fine`` apart at the ``centres`` and growing by
    ``grading`` per unit distance from them to ``h_far``."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    u = [0.0]
    while True:
        x = np.interp(u[-1], s, P[:, 0]), np.interp(u[-1], s, P[:, 1])
        d = min(math.hypot(x[0] - c[0], x[1] - c[1]) for c in centres)
        step = min(h_far, h_fine + grading * d)
        if u[-1] + step >= s[-1] - 0.3 * step:
            break
        u.append(u[-1] + step)
    u = np.array(u + [s[-1]])
    return np.column_stack([np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1])])


def jet_chains(delta=DELTA, r_n=R_NECK):
    """The bubble (a sphere of radius ``R_BUBBLE`` below its top's groove and hanging jet) and the
    open chain (the rising jet in the cavity's trough, then out to the wall), and the heights of
    the two necks."""
    loc, _ = head_feature(r_n=r_n, s=0.5, r_w=1.0, r_root=0.45)
    depth = (0.5 * (1.0 - 0.45) + 0.4) * delta  # the neck below the apex of its feature
    feat = np.column_stack([delta * loc[:, 0], Z_BUBBLE + delta * loc[:, 1]])
    r1, z1 = feat[0]
    zs = z1 - math.sqrt(R_BUBBLE**2 - r1**2)
    th = np.linspace(0.0, math.pi - math.asin(r1 / R_BUBBLE), 20000)
    sphere = np.column_stack([R_BUBBLE * np.sin(th), zs - R_BUBBLE * np.cos(th)])
    zb = Z_BUBBLE - depth
    b = _resample(np.vstack([sphere, feat[1:]]), 0.02 * delta, 0.02,
                  [(r_n * delta, zb), (0.0, feat[-1, 1])])
    b[0, 0] = b[-1, 0] = 0.0
    feat = np.column_stack([delta * loc[::-1, 0], Z_CAVITY - delta * loc[::-1, 1]])
    t = np.linspace(0.0, 1.0, 4000)[1:, None]
    far = ((1 - t) ** 2 * feat[-1] + 2 * t * (1 - t) * np.array([0.3, feat[-1, 1]])
           + t**2 * np.array([0.6, 1.1]))
    wall = np.column_stack([np.linspace(0.6, R_C, 400)[1:], np.full(399, 1.1)])
    zo = Z_CAVITY + depth
    o = _resample(np.vstack([feat, far, wall]), 0.02 * delta, 0.05,
                  [(r_n * delta, zo), (0.0, feat[0, 1])])
    o[0, 0] = 0.0
    o[-1] = (R_C, 1.1)
    return rc.Chain(b, ("axis", "axis")), rc.Chain(o, ("axis", "fixed")), (zb, zo)


def jets_focus(necks, r_n=R_NECK * DELTA):
    regions = (FocusRegion("neck_bubble", size=r_n / EPS, r=r_n, z=necks[0], radius=4 * r_n),
               FocusRegion("neck_open", size=r_n / EPS, r=r_n, z=necks[1], radius=4 * r_n),
               FocusRegion("jets", size=0.5 * DELTA / EPS, r=0.0, z=0.5 * (necks[0] + necks[1]),
                           radius=0.25))
    return Focus(size=min(g.size for g in regions), radius=0.25, grading=0.3, regions=regions)


class JetsMesh(FaradayCylinderMesh):
    """The first mesh is built on the chains of :func:`jet_chains`, as a mesh after a cut."""

    def define_geometry(self):
        if not self.is_first_time():
            return super().define_geometry()
        bubble, cavity, necks = jet_chains()
        focus = jets_focus(necks)
        s = self.settings
        self.mesh_mode = "tris"
        self.set_gmsh_parameter("General.NumThreads", 1)
        self.set_gmsh_parameter("Mesh.MeshSizeMin", min(s.h_interface, focus.size))
        self.set_gmsh_parameter("Mesh.MeshSizeMax", s.h_max)
        self.set_gmsh_parameter("Mesh.MeshSizeFromCurvature", 0)
        self.set_gmsh_parameter("Mesh.MeshSizeFromPoints", 0)
        self.set_gmsh_parameter("Mesh.MeshSizeExtendFromBoundary", 0)
        self.set_gmsh_parameter("Mesh.Algorithm", 6)
        self.default_resolution = s.h_max
        return self._define_chain_geometry([bubble, cavity], focus, focus.size)


class JetsProblem(FaradayProblem):
    mesh_class = JetsMesh


def jets_problem(outdir) -> JetsProblem:
    nd = nondimensionalise(CaseParameters.from_dict(fem.PHYSICS))
    pb = JetsProblem(nd, mesh=MeshSettings(**MESH), reconnect=RECON)
    pb.set_output_directory(str(outdir))
    return pb


def gas_cut_record(pb) -> dict:
    """A record of a gas cut at ``DELTA`` made just before the current state (``problem.cut``, with
    the cut's sizing put on the template, as a dump after a cut restores it) and the focus the
    driver keeps on the template."""
    named, _ = surgery._named_chains(pb)
    geom = rc.cut_geometry([c for _, c in named], RECON.reservoir_depth, gas_before=1.0)
    t = pb.get_current_time(dimensional=False, as_float=True)
    record = {"delta": DELTA, "rmin": DELTA, "r_neck": DELTA / 1.05, "z_neck": 0.85, "t_cut": t,
              "pinch": {"t_p": t - 1e-6}, "geometry": geom.to_dict(),
              "sizing": {"h_tip": RECON.tip_size * DELTA, "h_bridge": RECON.bridge_size * DELTA,
                         "curvature": 0, "grading": 0.3},
              "settings": json.loads(json.dumps(RECON.to_dict())), "focus": {"kept": None}}
    surgery.restore_cut(pb, record)
    pb.active_mesh_template().focus = jets_focus(jet_chains()[2])
    return record


def stream_velocity(r, z, necks, A=A_FLOW):
    """The solenoidal flow about each neck, from the stream function ``A r^2 (z - z_n) G``, ``G =
    exp(-((z - z_n)/(2 delta))^2 - (r/delta)^2)``: it closes the neck and stretches it axially."""
    r, z = np.asarray(r, dtype=float), np.asarray(z, dtype=float)
    ur, uz = np.zeros_like(r), np.zeros_like(r)
    w, a = 2.0 * DELTA, DELTA
    for zn in necks:
        zeta = z - zn
        G = np.exp(-(zeta / w) ** 2 - (r / a) ** 2)
        uz += 2.0 * A * zeta * G * (1.0 - r * r / (a * a))
        ur += -A * r * G * (1.0 - 2.0 * zeta * zeta / (w * w))
    return ur, uz


def closing_velocity(r, z, necks, A=A_FLOW):
    """The solenoidal flow of :func:`stream_velocity` and a radial inflow at each neck, which is
    not solenoidal."""
    ur, uz = stream_velocity(r, z, necks, A)
    for zn in necks:
        ur = ur - A * r * np.exp(-((z - zn) / (2.0 * DELTA)) ** 2 - (r / DELTA) ** 2)
    return ur, uz


def impose(pb, velocity, necks):
    m = pb.get_mesh("liquid")
    fi = m.get_nodal_field_indices()
    nodes = list(m.nodes())
    ur, uz = velocity(np.array([n.x(0) for n in nodes]), np.array([n.x(1) for n in nodes]),
                      necks)
    for n, a, b in zip(nodes, ur, uz):
        n.set_value(fi["velocity_x"], float(a))
        n.set_value(fi["velocity_y"], float(b))
    pb.reapply_boundary_conditions()
    pb.assign_initial_values_impulsive()


def settings(**liquid) -> ev.EventSettings:
    """The events block of the tests with the keys ``liquid`` of its ``liquid`` part changed."""
    return ev.EventSettings.from_dict({"liquid": {"enabled": True, **liquid},
                                       "post": {"open_neck_stop": False, "neck_floor": 1e-5}})


def request(pb, events=EVENTS):
    """The request of the events branch on the state, as the driver's stops would make it."""
    named, _ = surgery._named_chains(pb)
    return rc.post_cut_stops([c for _, c in named], rc.CutGeometry(**pb.cut["geometry"]), RECON,
                             pb.cut["sizing"]["h_tip"], events=events)


def cap_nodes(pb, record):
    """The new interface nodes on the planned cap of a made cut: positions and velocities."""
    cap = record["cap"]
    lc = type("Cap", (), {"r_n": cap["r_n"], "z_n": cap["z_n"],
                          "cap": ev.cap_at(cap["r_n"], cap["z_n"], cap["direction"], cap["eta"],
                                           np.linspace(0.0, ev.cap_arc_length(cap["r_n"],
                                                                              cap["eta"]),
                                                       200)[1:])})
    fi = pb.get_mesh("liquid").get_nodal_field_indices()
    nodes = surgery.mesh_nodes(pb.get_mesh(surgery.INTERFACE))
    X = np.array([[n.x(0), n.x(1)] for n in nodes])
    on = surgery._on_cap(lc, X, surgery._named_chains(pb)[1][record["chain"]])
    U = np.array([[n.value(fi["velocity_x"]), n.value(fi["velocity_y"])] for n in nodes])
    return X[on], U[on]


def problem_state(pb) -> tuple:
    """What a refused cut must leave as it was: the degrees of freedom and pinned values with the
    time and the rebuild (``FaradayProblem._fingerprint``), the focus, a parked plan and the
    handler's note of it, the remeshes asked for and the events made."""
    template = pb.get_mesh("liquid")._templatemesh
    focus = template.focus
    return (pb._fingerprint(), None if focus is None else focus.to_dict(),
            template.has_pending_surgery_plan(surgery.INTERFACE), pb.recon._pending_on,
            len(pb._domains_to_remesh), len(pb.events))


def _refuse(*args, **kwargs):
    raise ev.CapRefused("refused for the test", {"failed": ["test"]})


def _unparkable(*args, **kwargs):
    raise RuntimeError("not parked for the test")


CAP_SAMPLES = surgery._cap_samples


def _unsampled(problem, lc, size):  # the old bulk velocity found at every other sample only
    got = CAP_SAMPLES(problem, lc, size)
    found = got["found"].copy()
    found[1::2] = False
    return dict(got, found=found)


def _reload(pb, state: Path, record: dict):
    pb.load_state(str(state))
    surgery.restore_cut(pb, record)
    pb.active_mesh_template().focus = jets_focus(jet_chains()[2])
    pb.events = []


# --- pyoomph's parking, and the identity of the cutoff --------------------------------------------


def test_the_plan_is_parked_as_pyoomphs_own_hook_parks_one(tmp_path):
    """``surgery.park_plan`` parks a plan with the same names, scale and remesh request as
    pyoomph's reconnection handler parks its own (``after_newton_solve``), on the necked cavity of
    ``test_reconnect_fem.py``: the table and the cut radius as given."""
    with fem.neck_problem(tmp_path, RECON) as pb:
        pb.initialise()
        template = pb.get_mesh("liquid")._templatemesh
        assert pb.recon.request_pinch(pb, 1.05 * fem.R_N) == "plan"
        plan, theirs = template._pending_surgery_plans[surgery.INTERFACE]
        assert pb.recon._pending_on == (template, surgery.INTERFACE)
        assert template in pb._domains_to_remesh
        surgery.clear_plan(pb)
        assert not template.has_pending_surgery_plan(surgery.INTERFACE)
        table = np.asarray(theirs["old_zeta_table"])
        surgery.park_plan(pb, plan, table, rmin=0.02)
        got, ours = template._pending_surgery_plans[surgery.INTERFACE]
        assert got is plan and pb.recon._pending_on == (template, surgery.INTERFACE)
        assert template in pb._domains_to_remesh
        assert set(ours) == set(theirs)
        assert str(ours["spatial_scale"]) == str(theirs["spatial_scale"])
        for key in ("interface_name", "axis_name", "opposite_axis_name",
                    "opposite_interface_name", "distmin_nd"):
            assert ours[key] == theirs[key], key
        assert ours["interface_name"] == surgery.INTERFACE and ours["axis_name"] == "axis"
        assert np.array_equal(ours["old_zeta_table"], table) and ours["rmin_nd"] == 0.02
        surgery.clear_plan(pb)


def test_the_identity_of_the_cutoff_holds_in_the_projection_of_a_gradient(tmp_path):
    """On the manufactured field ``u~ = u_s + grad(phi)`` (``phi = 0`` on the free top), with the
    box inside the liquid: ``int (u+ - u~)_z phi_B dV = int Pi dphi_B/dz dV`` to the quadrature of
    the grid and the interpolation of the cutoff, both sides of the size of the impulse, and the
    error falls with the mesh; the momentum and energy the projection changes in the box are those
    of the grid's samples, and without the box nothing is sampled."""
    errors = []
    for N in (8, 16):
        pb = fem.ManufacturedProblem(N)
        pb.set_output_directory(str(tmp_path / f"m{N}"))
        with pb:
            pb.initialise()
            pb.set_initial_condition()
            t0 = pb.get_current_time(dimensional=False, as_float=True)
            s = rc.ReconnectSettings(delta=1.0, dt0=0.25, restore_pressure=False)
            out = surgery.project(pb, s, t0, box=(0.5, 0.35))
            box = out["box"]
            assert box["z"] == 0.5 and box["R"] == 0.35 and box["n"] == surgery.BOX_SAMPLES
            assert box["points"] == surgery.BOX_SAMPLES**2  # the half-disc lies in the liquid
            assert abs(box["phi_lhs"]) > 1e-2 and box["phi_lhs"] * box["phi_rhs"] > 0.0
            assert box["delta_Pz"] == pytest.approx(box["after"]["Pz"] - box["before"]["Pz"])
            assert box["delta_KE"] < 0.0  # the projection takes kinetic energy
            errors.append(abs(box["phi_relative_error"]))
            plain = surgery.project(pb, s, t0)
            assert "box" not in plain
    assert errors[1] < errors[0] and errors[1] < 1e-4, errors


# --- two cuts, one on each chain ------------------------------------------------------------------


@pytest.fixture(scope="module")
def jets(tmp_path_factory):
    out = tmp_path_factory.mktemp("jets")
    res = {"out": out}
    pb = jets_problem(out)
    with pb:
        pb.initialise()
        record = gas_cut_record(pb)
        res["cut"] = record
        necks = jet_chains()[2]
        res["necks"] = necks
        impose(pb, stream_velocity, necks)
        initial = Path(surgery._keep(pb, None, out, "initial", {})["path"])
        res["remesh_count"] = pb.remesh_count
        # the refusals, which change nothing
        req = request(pb)
        res["request1"] = req
        refusals = {}
        a_head = dict(req, kind="a-head", **{"class": "A-head"},
                      a_head={k: req[k] for k in ev.NECK_KEYS if k in req})
        for name, bad, events in (
                ("unhandled", dict(req, **{"class": "L-unhandled"}), EVENTS),
                ("no-neck", dict(req, index=req["index"] + 7), EVENTS),
                ("gas-neck", dict(req, kind="gas-neck", **{"class": "G"}), EVENTS),
                ("crossed", dict(req, **{"class": "A-head"}), EVENTS),
                # a request made under other keys of its class than the cut is given
                ("unlanded", req, settings(eps_d=0.05)),
                ("a-head-above", a_head, settings(eps_d=0.1, eps_max=0.15))):
            with pytest.raises(surgery.CutRefused) as err:
                surgery.liquid_cut(pb, events, RECON, out / f"refused-{name}", bad, k=1,
                                   **CUT_KW)
            refusals[name] = err.value.record
        res["refusals"] = refusals
        res["unchanged"] = (pb.remesh_count, pb.get_mesh("liquid")._templatemesh
                            .has_pending_surgery_plan(surgery.INTERFACE), len(pb.events))
        # the refusals after the state before the cut is kept: of the cap, of the plan, of its
        # parking, which change nothing either
        late = {}
        for name, owner, attr, fake in (("caps", ev, "truncate_chain", _refuse),
                                        ("plan", ev, "build_liquid_plan", _refuse),
                                        ("budgets", surgery, "_cap_samples", _unsampled),
                                        ("park", surgery, "park_plan", _unparkable)):
            before = problem_state(pb)
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(owner, attr, fake)
                with pytest.raises(surgery.CutRefused) as err:
                    surgery.liquid_cut(pb, EVENTS, RECON, out / f"late-{name}", req, k=1,
                                       **CUT_KW)
            late[name] = {"record": err.value.record, "precut": err.value.precut,
                          "same": problem_state(pb) == before}
        res["late"] = late
        # the cut of the first event, then of the next
        cuts = out / "cuts"
        res["record1"] = surgery.liquid_cut(pb, EVENTS, RECON, cuts, req, k=1, **CUT_KW)
        res["file1"] = json.loads((cuts / "events" / "event_001.json").read_text())
        res["segs1"] = surgery._named_chains(pb)[0]
        res["spans1"] = surgery._axis_spans_of_mesh(pb)
        req = request(pb)
        res["request2"] = req
        res["record2"] = surgery.liquid_cut(pb, EVENTS, RECON, cuts, req, k=2, **CUT_KW)
        res["segs2"] = surgery._named_chains(pb)[0]
        res["spans2"] = surgery._axis_spans_of_mesh(pb)
        res["events"] = [dict(x) for x in pb.events]
        res["after_cuts_time"] = pb.get_current_time(dimensional=False, as_float=True)
        res["counter"] = pb.timestepper.get_num_unsteady_steps_done()
        res["dt"] = pb.timestepper.time_pt().dt(0)
        for k in range(3):
            fem._step(pb, 1e-5)
        res["stepped"] = surgery._named_chains(pb)[0]
        res["request3"] = request(pb)
        # the cap fill against the imposed field, without the projection, and the transfer's arm
        res["fill"] = {}
        for fill in ("sample", "transfer"):
            _reload(pb, initial, record)
            events = ev.EventSettings.from_dict({"liquid": {"enabled": True, "cap_fill": fill,
                                                            "projection": False},
                                                 "post": {"open_neck_stop": False,
                                                          "neck_floor": 1e-5}})
            rec = surgery.liquid_cut(pb, events, RECON, out / f"fill-{fill}", request(pb), k=1,
                                     **CUT_KW)
            X, U = cap_nodes(pb, rec)
            ur, uz = stream_velocity(X[:, 0], X[:, 1], necks)
            res["fill"][fill] = {"record": rec, "error": np.hypot(U[:, 0] - ur, U[:, 1] - uz),
                                 "speed": np.hypot(ur, uz), "nodes": len(X)}
        # a flow that is not solenoidal: the projection changes the momentum in the box
        _reload(pb, initial, record)
        impose(pb, closing_velocity, necks)
        with pytest.raises(surgery.CutFailed) as err:
            surgery.liquid_cut(pb, EVENTS, RECON, out / "closing", request(pb), k=1, **CUT_KW)
        res["closing"] = err.value.record
        res["closing_file"] = json.loads((out / "closing" / "events" / "event_001.json")
                                         .read_text())
        # a transfer that leaves more interface nodes unlocated than predicted, beyond the slack
        _reload(pb, initial, record)
        real = surgery.transfer_warnings

        def miscounted(text):
            got = real(text)
            n, total = got["unlocated"].get(surgery.INTERFACE, [0, 0])
            got["unlocated"][surgery.INTERFACE] = [n + surgery.UNLOCATED_SLACK + 1, total]
            return got

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(surgery, "transfer_warnings", miscounted)
            with pytest.raises(surgery.CutFailed) as err:
                surgery.liquid_cut(pb, EVENTS, RECON, out / "miscounted", request(pb), k=1,
                                   **CUT_KW)
        res["miscounted"] = err.value.record
    return res


def test_the_refused_cuts_leave_the_problem_unchanged(jets):
    """A request of a class without this handler, of a kind and a class that do not go together,
    one that names no tracked neck with a single head, or one whose neck does not land under the
    keys of the cut (a landed neck above ``eps_d``, a head's above ``eps_max``) is refused at
    admission: no plan, no remesh, no event, the partial record written."""
    refusals = jets["refusals"]
    for name, rec in refusals.items():
        assert rec["outcome"] == "refused" and rec["failed_at"] == "admission", name
        assert rec["schema"] == surgery.EVENT_SCHEMA and rec["k"] == 1
        path = jets["out"] / f"refused-{name}" / "events" / "event_001.json"
        assert json.loads(path.read_text()) == rec
    assert "no tracked neck" in refusals["no-neck"]["error"]
    assert "no liquid cut for an event of kind 'liquid-neck' and class 'A-head'" in \
        refusals["crossed"]["error"]
    assert "has not landed at liquid.eps_d 0.05" in refusals["unlanded"]["error"]
    assert refusals["unlanded"]["trigger"]["landed"] is False
    assert "above liquid.eps_max 0.15" in refusals["a-head-above"]["error"]
    assert 0.15 < refusals["a-head-above"]["trigger"]["r_n_over_s"] < 0.2
    assert jets["unchanged"] == (jets["remesh_count"], False, 0)


def test_a_cut_refused_after_the_state_before_it_is_kept_leaves_the_problem_unchanged(jets):
    """A cap refused at its height and at the retry's, a plan refused, the old bulk velocity found
    at too few samples along the cap for its fill, a plan that cannot be parked: each refusal comes
    after the state before the cut is kept (``precut_e<k>``, which it names) and leaves the problem
    as it was (its degrees of freedom, its focus, no plan parked, no remesh asked for, no event);
    the record names the stage it stopped at."""
    late = jets["late"]
    for name, got in late.items():
        rec = got["record"]
        assert rec["outcome"] == "refused" and rec["failed_at"] == name, name
        assert got["same"], name
        assert got["precut"] == rec["dumps"]["precut"] and Path(got["precut"]).is_file(), name
    attempts = late["caps"]["record"]["cap"]["attempts"]
    assert [a["eta"] for a in attempts] == [1.0, ev.CAP_RETRY_HEIGHT]
    assert all(a["outcome"] == "refused" for a in attempts)
    assert "refused for the test" in late["plan"]["record"]["error"]
    samples = late["budgets"]["record"]["cap"]["samples"]
    assert samples["found"] < surgery.CAP_FILL_FOUND * samples["count"]
    assert "samples along the cap" in late["budgets"]["record"]["error"]
    assert "the plan could not be parked" in late["park"]["record"]["error"]


def test_each_cut_removes_its_head_and_rebuilds_the_mesh_as_planned(jets):
    """The first request lands the open chain's neck, the second, after it, the bubble's; each cut
    removes the head and caps the stump, the chains' tips on the axis moving to the neck's height
    plus or minus r_n, and the mesh has the chains and the axis spans planned."""
    r1, r2 = jets["record1"], jets["record2"]
    assert (jets["request1"]["kind"], jets["request1"]["chain"]) == ("liquid-neck", "open")
    assert (jets["request2"]["kind"], jets["request2"]["chain"]) == ("liquid-neck", "bubble")
    for rec, chain, tip, sign in ((r1, "open", "cavity_tip", 1.0),
                                  (r2, "bubble", "bubble_top", -1.0)):
        assert rec["outcome"] == "made" and rec["schema"] == surgery.EVENT_SCHEMA
        assert (rec["chain"], rec["tip"], rec["kind"], rec["class"]) == (chain, tip,
                                                                          "liquid-neck", "L")
        cap = rec["cap"]
        assert cap["shape"] == "hemispheroid" and cap["eta"] == 1.0 and not cap["retried"]
        assert cap["z_tip_new"] == pytest.approx(cap["z_n"] + sign * cap["r_n"])
        assert rec["trigger"]["landed"] and rec["trigger"]["r_n_over_s"] < 0.2
        assert rec["dt"]["history"] == CUT_KW["dt_prev"]
        assert cap["r_n"] == pytest.approx(R_NECK * DELTA, rel=0.05)
        assert 3.0 < rec["removed"]["V"] / cap["s"] ** 3 < 10.0
        mesh = rec["mesh"]
        assert np.ravel(mesh["axis_spans"]) == pytest.approx(np.ravel(mesh["expected_axis_spans"]),
                                                             abs=1e-12)
        assert {c["chain"]: c["end_types"] for c in mesh["chains"]} == {
            "bubble": ["axis", "axis"], "open": ["axis", "fixed"]}
        assert mesh["event_cap"]["size"] == pytest.approx(cap["r_n"] / EPS)
    # the cavity's tip and the bubble's top as planned, after both cuts
    named = dict(jets["segs2"])
    assert named["open"].points[0, 1] == pytest.approx(r1["cap"]["z_tip_new"], abs=1e-12)
    assert named["bubble"].points[-1, 1] == pytest.approx(r2["cap"]["z_tip_new"], abs=1e-12)
    assert jets["spans2"][1] == pytest.approx([r2["cap"]["z_tip_new"], r1["cap"]["z_tip_new"]],
                                              abs=1e-12)
    # the removed heads as ghosts to follow, moving away from their stumps, and the totals
    for rec, sign in ((r1, 1.0), (r2, -1.0)):
        g, rem = rec["ghost"], rec["removed"]
        assert g["status"] == "free" and g["direction"] == sign and sign * g["U"] > 0.0
        assert (g["V"], g["Pz"], g["KE"], g["U"]) == (rem["V"], rem["Pz"], rem["KE"], rem["U"])
        assert g["a"] == pytest.approx((0.75 * rem["V"] / math.pi) ** (1.0 / 3.0))
        assert g["v_ret"] == pytest.approx(rec["cap"]["r_n"] ** -0.5)
        assert (g["z_front0"], g["z_rear0"]) == (rec["cap"]["z_tip_old"], rec["cap"]["z_tip_new"])
        assert math.isfinite(g["du_stump"]) and g["t0"] == rec["t"]
    assert r1["removed_total"] == {"events": 1, **{x: r1["removed"][x] for x in ("V", "Pz", "KE")}}
    assert r2["removed_total"]["events"] == 2
    for x in ("V", "Pz", "KE"):
        assert r2["removed_total"][x] == pytest.approx(r1["removed"][x] + r2["removed"][x])
    # the events made, as the dumps keep them, the polygon of the removed head in the file only
    assert [e["k"] for e in jets["events"]] == [1, 2]
    assert ev.events_done(jets["events"]) == {"liquid": 2, "gap": 0}
    assert "polygon" not in r1["removed"] and len(jets["file1"]["removed"]["polygon"]) > 10
    assert jets["file1"]["dumps"]["postcut"] == r1["dumps"]["postcut"]
    for rec in (r1, r2):
        for key in ("precut", "postcut"):
            assert Path(rec["dumps"][key]).is_file() and \
                Path(rec["dumps"][key]).with_suffix(".json").is_file()
    meta = json.loads(Path(r1["dumps"]["precut"]).with_suffix(".json").read_text())
    assert meta["cut_stage"] == ev.PRECUT_EVENT and meta["event"]["k"] == 1


def test_the_transfer_leaves_unlocated_only_the_cap_it_predicts(jets):
    """pyoomph's transfer leaves unlocated exactly the new interface nodes the geometry predicts
    (farther than half an element size from every old element), all of them on the cap, none of
    the bulk or the axis, and only the stump's tip on the axis without a value."""
    for rec in (jets["record1"], jets["record2"]):
        tr = rec["transfer"]
        assert tr["reported_unlocated"] == tr["predicted_unlocated"] > 0
        assert tr["predicted_off_cap"] == 0 and tr["cap_nodes"] > tr["predicted_unlocated"]
        assert set(tr["unlocated"]) == {surgery.INTERFACE}
        assert tr["no_value"] == {surgery.INTERFACE + "/axis": [1, 3]}
        assert "_kin_bc" in tr["corner_fields"]
        fill = rec["cap_fill"]
        assert fill["mode"] == "sample" and fill["nodes"] == tr["cap_nodes"]
        assert fill["samples_found"] == fill["samples"]
        assert rec["cap"]["samples"] == {"count": fill["samples"], "found": fill["samples"]}


def test_a_transfer_not_as_predicted_fails_the_cut_after_its_remesh(jets):
    """pyoomph's count of unlocated interface nodes beyond the slack of the prediction fails the
    cut at the check of its mesh, after the remesh, before the fill and the projection."""
    rec = jets["miscounted"]
    assert rec["outcome"] == "failed" and rec["failed_at"] == "mesh-check"
    assert "interface nodes unlocated" in rec["error"] and "projection" not in rec
    tr = rec["transfer"]
    assert tr["reported_unlocated"] == tr["predicted_unlocated"] + surgery.UNLOCATED_SLACK + 1


def test_the_budgets_of_each_cut_close(jets):
    """The liquid volume in the box falls by the removed head's (F1, polylines), the axial momentum
    by the head's (F2, the grid; the head moves away from the stump), the projection changes the
    momentum and the kinetic energy there by little (F3, F4: the flow is solenoidal), and the area
    of the interface by the head's less the cap's (F5); the cutoff's identity is recorded."""
    for rec, sign in ((jets["record1"], 1.0), (jets["record2"], -1.0)):
        cl, rem = rec["closures"], rec["removed"]
        assert cl["flags"] == [] and cl["hard"] == [] and cl["sigma"] == 0.0
        assert cl["relative"]["F1"] < 2e-3 and cl["relative"]["F2"] < 1e-2
        assert cl["relative"]["F3"] < 2e-3 and abs(cl["relative"]["F4"]) < 1e-3
        assert cl["relative"]["F5"] < 2e-3
        assert sign * rem["Pz"] > 0.0 and rem["KE"] > 0.0
        assert rem["V_grid"] == pytest.approx(rem["V"], rel=0.01)
        assert rem["U"] == pytest.approx(rem["Pz"] / rem["V"])
        box = rec["projection"]["box"]
        assert box["R"] == rec["budgets"]["box"]["R"] == max(4.0 * DELTA, 2.0 * rem["length"])
        assert set(cl["F3_phi"]) == {"phi_lhs", "phi_rhs", "phi_error", "phi_relative_error"}
        g = rec["budgets"]["grid"]
        assert g["after_transfer"] == box["before"] and g["after_projection"] == box["after"]
        assert rec["projection"]["newton_residuals"][-1] < RECON.projection_tolerance
        assert abs(rec["projection"]["energy_identity_error"]) < 1e-14


def test_after_the_cuts_the_run_steps_and_the_time_is_kept(jets):
    """The cut keeps the time and restarts the history with the step before it (BDF1 next); three
    steps follow on the two chains, and the state they reach lands no other neck."""
    assert jets["after_cuts_time"] == pytest.approx(jets["cut"]["t_cut"], abs=1e-15)
    assert jets["counter"] == 0 and jets["dt"] == CUT_KW["dt_prev"]
    named = dict(jets["stepped"])
    assert set(named) == {"bubble", "open"}
    req = jets["request3"]
    assert req is None or req.get("kind") not in ("liquid-neck", "a-head")


def test_the_cap_carries_the_old_bulk_velocity_at_its_nodes(jets):
    """Without the projection the cap's nodes carry the old bulk velocity at their places, the
    imposed flow to the interpolation of the old mesh; the transfer alone (``cap_fill``
    ``transfer``) gives most of them the old surface's near the neck, many times farther off."""
    sample, transfer = jets["fill"]["sample"], jets["fill"]["transfer"]
    assert sample["nodes"] == transfer["nodes"] > 10
    scale = float(sample["speed"].max())
    assert sample["error"].max() < 2e-3 * scale
    assert transfer["error"].max() > 0.1 * scale > 50.0 * sample["error"].max()
    assert transfer["record"]["cap_fill"]["mode"] == "transfer"
    assert "box" not in sample["record"]["projection"]  # the projection was off


def test_a_projection_that_changes_the_boxs_momentum_fails_the_cut(jets):
    """With a flow that is not solenoidal the projection does work: the cutoff's identity holds with
    a pressure impulse of the size of the flow, and the momentum it changes in the box is beyond the
    limit of the budget (F3), which fails the cut after the remesh, its record written."""
    rec = jets["closing"]
    assert rec["outcome"] == "failed" and rec["failed_at"] == "postcut"
    assert "F3" in rec["closures"]["hard"] and rec == jets["closing_file"]
    box = rec["projection"]["box"]
    assert box["max_abs_Pi"] > 1e-3 and abs(box["phi_lhs"]) > 1e-5
    assert abs(box["phi_relative_error"]) < 5e-3
    assert rec["closures"]["relative"]["F3"] > 1.0  # the momentum of the head, many times


RESTART = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import test_liquid_cut_fem as T
from faraday_singularity import surgery
state, out = Path(sys.argv[3]), Path(sys.argv[4])
pb = T.jets_problem(out)
with pb:
    pb.initialise()
    pb.load_state(str(state))
    meta = json.loads(state.with_suffix(".json").read_text())
    surgery.restore_cut(pb, meta["cut"])
    named, _ = surgery._named_chains(pb)
    before = {name: [float(c.points[0, 1]), float(c.points[-1, 1])] for name, c in named}
    for k in range(2):
        t = pb.get_current_time(dimensional=False, as_float=True)
        pb.run(t + 1e-5, timestep=1e-5, maxstep=1e-5, outstep=False, temporal_error=1.0,
               newton_solver_tolerance=1e-8)
    named, _ = surgery._named_chains(pb)
    print(json.dumps({"before": before, "time": pb.get_current_time(dimensional=False,
                                                                     as_float=True),
                      "chains": sorted(name for name, _ in named)}))
"""


def test_a_restart_from_the_dump_after_the_cuts_steps_in_a_new_process(jets, tmp_path):
    """The dump after the second cut, loaded in a new process (the cut's sizing restored from its
    record), has both chains with the cut tips, and takes steps."""
    state = Path(jets["record2"]["dumps"]["postcut"])
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-c", RESTART, str(ROOT / "src"),
                           str(ROOT / "testCases"), str(state), str(tmp_path / "again")],
                          capture_output=True, text=True, timeout=900, check=False, env=env)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["chains"] == ["bubble", "open"]
    assert got["before"]["open"][0] == pytest.approx(jets["record1"]["cap"]["z_tip_new"],
                                                     abs=1e-12)
    assert got["before"]["bubble"][1] == pytest.approx(jets["record2"]["cap"]["z_tip_new"],
                                                       abs=1e-12)
    assert got["time"] > jets["after_cuts_time"]
