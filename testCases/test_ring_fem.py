"""Finite-element tests of the ring of a pierced bubble: a closed loop of the free surface's own
boundary, on ``FaradayProblem``.

* F0, the pyoomph facts the ring stage relies on: a problem with equations on a boundary of its own
  for the ring refuses a mesh without it (so the ring lives on ``interface``); the private names
  the chart and the transfer use exist; pyoomph's own arclength chart still refuses a loop beside an
  open segment (so the project's chart, ``problem.RingTolerantArclength``, is still needed).
* F1, a ring meshed as a hole: the synthetic necked cavity of ``test_reconnect_fem.py`` after its
  gas cut, with an elliptic ring added in the liquid through the template's planned geometry: the
  liquid volume falls by the ring's volume of revolution (Pappus), the loop reads back closed,
  ordered and seamed on a vertex, the chains are the open chain and the bubble; three steps, an
  ordinary rebuild, three steps; a dump restarted in two new processes (the problem made as the
  runner makes it, no hook before ``load_state``, the stage put back after it,
  ``ringsurgery.restore_ring``) steps the same in both to the bit, and as the process that wrote it
  to within what a reload changes (the mesh read back from its file); the ring removed, three steps.
* F2 and F3, the ring cut and the ring's removal on the synthetic pierced state of
  ``ring_fem_state.py``, made by the driver: the refusals, the rebuilt mesh, the transfer, the
  closures, the steps, rows and dumps of the stage, its restart in new processes, the removal; and
  each hard closure violated in turn (R1-R3 and ``R3_box`` of the cut, Q1-Q3 and ``Q3_box`` of the
  removal: the filled volume misread, momentum added between the transfer and the fill, the impulse
  the box's balance reads shifted, a flow that is not solenoidal), each a stop ``event-unclosed``
  with the state before the event kept.

Each problem compiles in seconds; the module takes a few minutes.
"""

from __future__ import annotations

import hashlib
import inspect
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

from faraday_singularity import ring, ringsurgery, surgery  # noqa: E402
from faraday_singularity.mesh import RingSizing  # noqa: E402
from faraday_singularity.problem import RingTolerantArclength  # noqa: E402
from faraday_singularity.reconnect import ReconnectSettings  # noqa: E402

C, A, B = (1.2, 0.4), 0.08, 0.05  # the ring of F1: an ellipse in the liquid
DT = 1e-5


def ellipse(n=96):
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([C[0] + A * np.cos(t), C[1] + B * np.sin(t)])


def impulsive(pb):
    pb.assign_initial_values_impulsive(DT)
    pb.timestepper.set_num_unsteady_steps_done(0)


def state_vector(pb) -> np.ndarray:
    """Every degree of freedom and the positions of the interface's nodes, chains and loop."""
    dofs = np.asarray(pb.get_current_dofs()[0], dtype=float)
    parts = [dofs] + [np.concatenate([c["r"], c["z"]]) for c in pb.interface_segments()]
    parts += [np.concatenate([x["r"], x["z"]]) for x in pb.interface_loops()]
    return np.concatenate(parts)


RESTART = """
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1] + "/src")
sys.path.insert(0, sys.argv[1] + "/testCases")
import test_reconnect_fem as fem
import test_ring_fem as t
from faraday_singularity import ringsurgery, surgery
from faraday_singularity.reconnect import ReconnectSettings

dump, out = Path(sys.argv[2]), Path(sys.argv[3])
meta = json.loads(dump.with_suffix(".json").read_text())
with fem.neck_problem(out, ReconnectSettings(**fem.FAST_CUT)) as pb:
    pb.initialise()
    pb.load_state(str(dump))
    surgery.restore_cut(pb, meta["cut"])
    ringsurgery.restore_ring(pb, meta["ring"])
    for k in range(2):
        fem._step(pb, t.DT)
    v = t.state_vector(pb)
    np.save(out / "state.npy", v)
    print("LOOPS", len(pb.interface_loops()), pb.active_mesh_template().ring_stage)
"""


def _restart(dump: Path, out: Path) -> np.ndarray:
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-c", RESTART, str(ROOT), str(dump), str(out)],
                          capture_output=True, text=True, timeout=900, env=env, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "LOOPS 1 True" in proc.stdout
    return np.load(out / "state.npy")


@pytest.fixture(scope="module")
def ringed(tmp_path_factory):
    out = tmp_path_factory.mktemp("ring")
    res = {"out": out}
    s = ReconnectSettings(**fem.FAST_CUT)
    with fem.neck_problem(out / "pb", s) as pb:
        pb.initialise()
        fem.impose_closing_flow(pb)
        surgery.cut(pb, s, out / "cut", log=lambda *_: None)
        # F0: pyoomph's own chart, on the cut's two open-chart chains, then with the ring
        res["volume_before"] = pb.cut_observables()["volume"]
        tmpl = pb.active_mesh_template()
        named, _ = surgery._named_chains(pb)
        tmpl.ring_stage = True
        tmpl.ring_sizing = RingSizing(h_ring=0.01, grading=0.3)
        tmpl.pending_ring_geometry = {"chains": [c for _, c in named], "loops": [ellipse()]}
        pb.force_remesh({tmpl})
        pb._domains_to_remesh.clear()
        impulsive(pb)
        res["volume_ring"] = pb.cut_observables()["volume"]
        res["segs"] = [(c["open"], tuple(c["end_types"])) for c in pb.interface_segments()]
        res["loops"] = pb.interface_loops()
        res["pending_after"] = tmpl.pending_ring_geometry
        # pyoomph's stock chart on the ring's interface: refused
        from pyoomph.meshes.meshdatacache import MeshDataCacheEntry, MeshDataCacheKey
        from pyoomph.meshes.zeta import AssignZetaCoordinatesByArclength

        iface = pb.get_mesh("liquid/interface")
        cache = MeshDataCacheEntry(iface, MeshDataCacheKey(nondimensional=True,
                                                           tesselate_tri=True))
        try:
            AssignZetaCoordinatesByArclength._compute_zetas(pb.interface_zeta, iface, cache, True)
            res["stock"] = None
        except RuntimeError as e:
            res["stock"] = str(e)
        _, zetas = pb.interface_zeta._compute_zetas(iface, cache, True)
        res["zetas"] = zetas
        for _ in range(3):
            fem._step(pb, DT)
        pb.force_remesh({tmpl})  # an ordinary rebuild in the ring stage
        res["rebuilt_loops"] = pb.interface_loops()
        for _ in range(3):
            fem._step(pb, DT)
        dump = out / "ring.dump"
        pb.save_state(str(dump))
        record = {"sizing": {"h_ring": 0.01, "grading": 0.3}, "removed": None}
        dump.with_suffix(".json").write_text(json.dumps({"cut": pb.cut, "ring": record}))
        res["dump"] = dump
        for _ in range(2):
            fem._step(pb, DT)
        res["continued"] = state_vector(pb)
        # the removal
        v = pb.cut_observables()["volume"]
        tmpl.pending_ring_geometry = {"chains": [c for _, c in surgery._named_chains(pb)[0]],
                                      "loops": []}
        pb.force_remesh({tmpl})
        pb._domains_to_remesh.clear()
        impulsive(pb)
        res["removed_dV"] = pb.cut_observables()["volume"] - v
        res["loops_after_removal"] = pb.interface_loops()
        for _ in range(3):
            fem._step(pb, DT)
        res["segs_after_removal"] = len(pb.interface_segments())
        res["t_end"] = pb.get_current_time(dimensional=False, as_float=True)
    return res


def test_pyoomph_refuses_a_boundary_of_the_rings_own_and_has_the_names_the_stage_uses(tmp_path):
    from pyoomph.equations.generic import DirichletBC
    from pyoomph.meshes import interpolator, meshdatacache, zeta

    class OwnBoundary(fem.NeckProblem):
        def define_problem(self):
            super().define_problem()
            self.add_equations((DirichletBC(velocity_x=0) @ "ring") @ "liquid")

    nd = fem.nondimensionalise(fem.CaseParameters.from_dict(fem.PHYSICS))
    pb = OwnBoundary(nd, mesh=fem.MeshSettings(**fem.MESH))
    pb.set_output_directory(str(tmp_path / "own"))
    with pytest.raises(Exception, match="ring"):
        pb.initialise()
    assert callable(zeta._find_closed_segments)
    src = inspect.getsource(meshdatacache.MeshDataCacheEntry)
    assert "interface_lines_segs" in src and "interface_lines_segs_ninter" in src
    src = inspect.getsource(interpolator)
    for name in ("try_to_use_zeta_on_boundary", "bulk_locate_boundaries",
                 "boundary_max_distances"):
        assert name in src, name
    assert issubclass(RingTolerantArclength, zeta.AssignZetaCoordinatesByArclength)


def test_the_stock_chart_refuses_the_ring_and_the_projects_charts_it_beyond_the_chains(ringed):
    assert ringed["stock"] is not None and "closed loop" in ringed["stock"]
    z = np.array([x for _, x in ringed["zetas"]])
    assert np.all(np.isfinite(z)) and len(z) == len(set(i for i, _ in ringed["zetas"]))


def test_a_ring_is_meshed_as_a_hole_and_read_back_as_a_loop(ringed):
    dV = ringed["volume_ring"] - ringed["volume_before"]
    pappus = 2 * math.pi * C[0] * math.pi * A * B
    assert dV == pytest.approx(-pappus, rel=1e-4)
    assert sorted(ringed["segs"]) == [(False, ("axis", "axis")), (True, ("axis", "fixed"))]
    assert ringed["pending_after"] is None  # consumed by the build
    for loops in (ringed["loops"], ringed["rebuilt_loops"]):
        assert len(loops) == 1
        L = np.column_stack([loops[0]["r"], loops[0]["z"]])
        ring.validate_loop_order(L)
        assert ring.signed_area(L) > 0.0 and len(L) % 2 == 0 and loops[0]["vertex"][0]
        assert np.all(loops[0]["vertex"][::2]) and not np.any(loops[0]["vertex"][1::2])
        assert L[0, 0] == pytest.approx(C[0] + A, rel=1e-4)  # its seam, the outermost vertex
        assert ring.loop_edges(L)["edges"] is not None
        assert rc_volume(L) == pytest.approx(pappus, rel=1e-3)


def rc_volume(L):
    from faraday_singularity.reconnect import revolved_volume

    return revolved_volume(L)


def test_a_dump_of_the_ring_stage_restarts_in_new_processes(ringed, tmp_path):
    a = _restart(ringed["dump"], tmp_path / "a")
    b = _restart(ringed["dump"], tmp_path / "b")
    assert a.shape == b.shape and np.array_equal(a, b)
    c = ringed["continued"]
    assert c.shape == a.shape
    assert np.max(np.abs(a - c)) <= 1e-10 * np.max(np.abs(c))
    assert hashlib.sha256(a.tobytes()).hexdigest() == hashlib.sha256(b.tobytes()).hexdigest()


def test_the_ring_is_removed_and_the_run_steps_on(ringed):
    pappus = 2 * math.pi * C[0] * math.pi * A * B
    assert ringed["removed_dV"] == pytest.approx(pappus, rel=2e-3)
    assert ringed["loops_after_removal"] == [] and ringed["segs_after_removal"] == 2
    assert ringed["t_end"] > 0.0


def test_restore_ring_needs_its_sizing():
    from types import SimpleNamespace

    template = SimpleNamespace(ring_stage=False, ring_sizing=None, pending_ring_geometry=1)
    pb = SimpleNamespace(active_mesh_template=lambda: template, ring=None)
    with pytest.raises(RuntimeError, match="ring sizing"):
        ringsurgery.restore_ring(pb, {"sizing": {}})
    ringsurgery.restore_ring(pb, {"sizing": {"h_ring": 1e-5, "grading": 0.3}})
    assert template.ring_stage and template.ring_sizing.h_ring == 1e-5
    assert template.pending_ring_geometry is None and pb.ring["sizing"]["h_ring"] == 1e-5


# --- F2 and F3: the ring cut and the ring's removal on a synthetic pierced state ------------------

import ring_fem_state as S  # noqa: E402

from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402
from faraday_singularity.driver import CollapseDriver, CollapseSettings  # noqa: E402

DRIVER = {"t_end": 10.0, "dt_max": 0.01, "state_dt": 0.0, "state_keep": 50}


def problem_state(pb) -> tuple:
    """What a refused surgery must leave as it was (``test_liquid_cut_fem.problem_state``): the
    degrees of freedom with the time and the rebuild, the focus, the template's ring stage, the
    remeshes asked for, the events made and the ring."""
    template = pb.get_mesh("liquid")._templatemesh
    focus = template.focus
    return (pb._fingerprint(), None if focus is None else focus.to_dict(),
            getattr(template, "ring_stage", False), getattr(template, "ring_sizing", None),
            len(pb._domains_to_remesh), len(pb.events), pb.ring)


def ring_block(noise: dict | None, **ring_keys) -> ev.EventSettings:
    sigma = {} if noise is None else {"sigma_V": noise["V"], "sigma_Pz": noise["Pz"],
                                      "sigma_KE": noise["KE"]}
    return ev.EventSettings.from_dict({"ring": {"enabled": True, **sigma, **ring_keys}})


def request(pb, events):
    named, _ = surgery._named_chains(pb)
    return rc.post_cut_stops([c for _, c in named], rc.CutGeometry(**pb.cut["geometry"]),
                             S.RECON, pb.cut["sizing"]["h_tip"], events=events)


def reload(pb, state: Path, record: dict):
    pb.load_state(str(state))
    surgery.restore_cut(pb, record)
    pb.active_mesh_template().focus = S.focus()
    pb.events, pb.ring = [], None
    pb.invalidate_cached_mesh_data()


CUT_KW = {"dt_prev": 1e-5, "elements_per_scale": 8.0, "h_floor": 1e-5}


@pytest.fixture(scope="module")
def pierced(tmp_path_factory):
    out = tmp_path_factory.mktemp("pierced")
    res = {"out": out}
    with S.pierce_problem(out / "pb") as pb:
        pb.initialise()
        record = S.gas_cut_record(pb)
        res["record"] = record
        S.impose(pb)
        initial = Path(surgery._keep(pb, None, out, "initial", {})["path"])
        res["initial"] = initial
        # the noise of the cut's box in an ordinary rebuild of the state, the floors of its closures
        g = ring.ring_cut_geometry(*[dict(surgery._named_chains(pb)[0])[x]
                                     for x in ("bubble", "open")], ev.RingEventSettings(),
                                   elements_per_scale=8.0, min_angle=30.0, reservoir_depth=3.0,
                                   V_b=record["geometry"]["bubble_volume"], bubble_stop=0.01)
        noise = S.rebuild_noise(pb, g.z_mid, max(4.0 * S.DELTA, 2.0 * g.L_fill))
        res["noise"], res["geometry"] = noise, g
        reload(pb, initial, record)
        events = ring_block(noise)
        res["request"] = request(pb, events)
        # the refusals, which leave the problem as it was
        refused = {}
        for name, kw, block in (("dt", dict(CUT_KW, dt_prev=0.0), events),
                                ("waist", CUT_KW, ring_block(noise, waist=0.05)),
                                ("disabled", CUT_KW, ev.EventSettings())):
            before = problem_state(pb)
            with pytest.raises(surgery.CutRefused) as err:
                ringsurgery.ring_cut(pb, block, S.RECON, out / f"refused-{name}",
                                     res["request"], k=1, **kw)
            refused[name] = {"record": err.value.record, "precut": err.value.precut,
                             "same": problem_state(pb) == before}
        res["refused"] = refused
        # at the zero floors the volume closure lies below the noise of this coarse mesh
        with pytest.raises(surgery.BudgetUnclosed) as err:
            ringsurgery.ring_cut(pb, ring_block(None), S.RECON, out / "unclosed",
                                 res["request"], k=1, **CUT_KW)
        res["unclosed"] = err.value.record
        reload(pb, initial, record)
        # the cut, met by the driver: the piercing at the loop's top, then five steps
        run = out / "run"
        driver = CollapseDriver(pb, CollapseSettings(**DRIVER, max_steps=5), run,
                                reconnect=S.RECON, events=events)
        driver.reset_after_cut(pb.cut)
        res["result"] = driver.run(1e-8)
        driver.close()
        res["events"] = [dict(x) for x in pb.events]
        res["ring_stage"] = dict(pb.ring) if pb.ring is not None else None
        res["loops"] = pb.interface_loops()
        res["segs"] = pb.interface_segments()
        res["spans"] = surgery._axis_spans_of_mesh(pb)
        res["run"] = run
    return res


def test_the_piercing_is_requested_and_the_refused_cuts_change_nothing(pierced):
    req = pierced["request"]
    assert (req["kind"], req["class"], req["handler"], req["gap"]) == ("piercing", "W1", "ring",
                                                                       1.0)
    assert req["bubble_length"] == pytest.approx(S.GAP, rel=1e-6)
    refused = pierced["refused"]
    for name, got in refused.items():
        assert got["same"], name
        assert got["record"]["outcome"] == "refused"
    assert refused["dt"]["record"]["failed_at"] == "admission" and refused["dt"]["precut"] is None
    assert refused["disabled"]["record"]["failed_at"] == "admission"
    waist = refused["waist"]
    assert waist["record"]["failed_at"] == "geometry" and Path(waist["precut"]).is_file()
    assert waist["record"]["geometry"]["checks"]["failed"] == ["closure"]
    # the zero floors: a hard closure misses, by less than the floor the rebuild noise of the box
    # sets (on this coarse mesh the noise is of the order of the filled region's own budgets)
    unclosed = pierced["unclosed"]
    assert unclosed["outcome"] == "unclosed" and unclosed["failed_at"] == "postcut"
    hard = unclosed["closures"]["hard"]
    assert hard and set(hard) <= {"R1", "R2", "R3", "R3_box"}
    noise = {"R1": "V", "R2": "Pz", "R3": "Pz", "R3_box": "Pz"}
    for x in hard:
        assert unclosed["closures"][x] < 5.0 * pierced["noise"][noise[x]], x


def test_the_ring_cut_rebuilds_the_mesh_as_planned_and_its_budgets_close(pierced):
    result = pierced["result"]
    assert result["stop"]["reason"] == "max-steps" and result["steps"] == 5, json.dumps(
        result["stop"], default=float)
    assert result["events"]["made"] == 1 and result["events"]["ring"]["cut"] == 1
    rec = pierced["events"][0]
    assert (rec["kind"], rec["class"], rec["handler"], rec["outcome"]) == (
        "piercing", "W1", "ring", "made")
    g = pierced["geometry"]
    geo = rec["geometry"]
    assert geo["r_w"] == pytest.approx(g.r_w) and geo["V_fill"] == pytest.approx(g.V_fill)
    # one open chain and one ring, the axis one span up to the cavity's tip
    assert len(pierced["segs"]) == 1 and pierced["segs"][0]["open"]
    assert len(pierced["loops"]) == 1
    assert len(pierced["spans"]) == 1 and pierced["spans"][0][0] == 0.0
    assert pierced["spans"][0][1] == pytest.approx(float(pierced["segs"][0]["z"][0]), abs=1e-9)
    tr = rec["transfer"]
    assert tr["reported_unlocated"] == tr["predicted_unlocated"] > 0
    assert tr["predicted_off_closure"] == 0
    gas = tr["former_gas"]
    assert gas["axis"] == gas["predicted_axis"] > 0
    assert abs(gas["bulk"] - gas["predicted_bulk"]) <= max(0.1 * gas["predicted_bulk"], 5)
    c = rec["closures"]
    assert c["hard"] == [] and set(c["limits"]) == {f"R{i}" for i in range(1, 7)} | {"R3_box"}
    # the cutoff identity against its own scale, the larger of its sides, floored at the box's
    # noise, and the box's momentum the projection changed against P_ref (here the solenoidal flow
    # leaves the projection little to change in the box)
    phi = c["R3_phi"]
    assert c["R3"] <= c["limits"]["R3"] and math.isfinite(phi["phi_relative_error"])
    assert c["relative"]["R3"] == pytest.approx(
        c["R3"] / max(abs(phi["phi_lhs"]), abs(phi["phi_rhs"])))
    assert c["R3_box"] <= c["limits"]["R3_box"] and c["R3_box"] <= 0.05 * c["scales"]["P_ref"]
    assert tr["closure_nodes"] > 0 and tr["corners"]["new"] == 0 and tr["no_value"] == {}
    assert c["scales"]["U"] == pytest.approx(abs(S.W_JET - S.W_PLUG), rel=0.1)
    assert rec["filled"]["V"] == pytest.approx(g.V_fill)
    stage = pierced["ring_stage"]
    assert stage["k_cut"] == 1 and stage["removed"] is None
    assert stage["V0"] == pytest.approx(g.V_ring) and stage["sizing"]["h_col"] > 0.0
    assert [x["name"] for x in stage["sizing"]["frozen"]] == ["ring_bubble_top"]


def test_the_ring_stage_steps_and_writes_its_rows_and_dumps(pierced):
    run = pierced["run"]
    rows = np.loadtxt(run / "ring.txt", ndmin=2)
    assert len(rows) >= 5
    head = (run / "ring.txt").read_text().splitlines()[0][2:].split("\t")
    V = rows[:, head.index("ring_volume")]
    assert np.all(V > 0.0) and np.all(np.isfinite(rows[:, head.index("ring_courant")]))
    meta = json.loads((run / "states" / "postcut_e001.json").read_text())
    assert meta["ring"]["k_cut"] == 1 and "resume_at" not in meta
    assert meta["cut_stage"] == "postcut-event"
    file = json.loads((run / "events" / "event_001.json").read_text())
    assert len(file["fill_polygon"]) > 10 and len(file["ring_loop"]) > 10
    events = (run / "events.txt").read_text().splitlines()
    assert len(events) == 2 and "\tring\t" in events[1]
    assert list((run / "profiles").glob("profile_*_ring.txt"))


RING_RESTART = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1] + "/src")
sys.path.insert(0, sys.argv[1] + "/testCases")
import ring_fem_state as S
from faraday_singularity import events as ev
from faraday_singularity.driver import CollapseDriver, CollapseSettings

state, out, block = Path(sys.argv[2]), Path(sys.argv[3]), json.loads(sys.argv[4])
meta = json.loads(state.with_suffix(".json").read_text())
with S.pierce_problem(out / "pb") as pb:
    pb.initialise()
    pb.load_state(str(state))
    drv = CollapseDriver(pb, CollapseSettings(t_end=10.0, dt_max=0.01, state_dt=0.0,
                                              state_keep=50, max_steps=meta["step"] + 2),
                         out / "run", reconnect=S.RECON, events=ev.EventSettings.from_dict(block))
    drv.restore(meta)
    result = drv.run(1e-8)
    drv.close()
(out / "result.json").write_text(json.dumps(result, default=float))
"""


def _ring_restart(state: Path, out: Path, block: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-c", RING_RESTART, str(ROOT), str(state), str(out),
                           json.dumps(block)], capture_output=True, text=True, timeout=1200,
                          env=env, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads((out / "result.json").read_text())


def test_the_state_after_the_ring_cut_is_admitted_and_continues_in_new_processes(pierced,
                                                                                  tmp_path):
    import run_case

    run = pierced["run"]
    postcut = run / "states" / "postcut_e001.dump"
    meta = json.loads(postcut.with_suffix(".json").read_text())
    block = meta["events_settings"]
    from test_runner import CASE

    case = json.loads(json.dumps(CASE))
    case["collapse"].update(reconnect=S.RECON.to_dict(), reconnect_events=block)
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case))
    digest = hashlib.sha256(case_path.read_bytes()).hexdigest()
    (run / "run-manifest.json").write_text(json.dumps(
        {"schema": "pyoomph-runner-manifest-v1", "component_commit": "a" * 40,
         "case_sha256": digest}))
    (run / "parameters.json").write_text(json.dumps(
        {"case": case, "provenance": {"case_sha256": digest}}))
    record, mode = run_case.restart_admission(postcut, case, case_path, S.RECON,
                                              events=ev.EventSettings.from_dict(block))
    assert mode == "post-cut"
    with pytest.raises(SystemExit, match="ring.enabled"):
        run_case.restart_admission(postcut, case, case_path, S.RECON,
                                   events=ev.EventSettings.from_dict({}))
    a = _ring_restart(postcut, tmp_path / "a", block)
    b = _ring_restart(postcut, tmp_path / "b", block)
    assert a["stop"]["reason"] == "max-steps" and a["steps"] == meta["step"] + 2 == b["steps"]
    rows = {x: (tmp_path / x / "run" / "chains.txt").read_text() for x in ("a", "b")}
    assert rows["a"] == rows["b"]  # two new processes, to the bit
    ring_rows = {x: (tmp_path / x / "run" / "ring.txt").read_text() for x in ("a", "b")}
    assert ring_rows["a"] == ring_rows["b"] and len(ring_rows["a"].splitlines()) >= 3
    # and as the run that wrote the state, to within what a reload changes
    head = (run / "chains.txt").read_text().splitlines()[0][2:].split("\t")
    k = head.index("step")
    mine = {int(float(r.split("\t")[k])): np.array(r.split("\t"), dtype=float)
            for r in (run / "chains.txt").read_text().splitlines()[1:]}
    theirs = {int(float(r.split("\t")[k])): np.array(r.split("\t"), dtype=float)
              for r in rows["a"].splitlines()[1:]}
    step = meta["step"] + 2
    assert step in mine and step in theirs
    x, y = mine[step], theirs[step]
    finite = np.isfinite(x) & np.isfinite(y)
    cols = [head.index(c) for c in ("z_cavity_tip", "bubble_volume", "volume", "momentum_z")]
    assert np.allclose(x[cols], y[cols], rtol=1e-8, atol=1e-14), (x[cols], y[cols])
    assert finite.sum() > 10


def test_the_ring_is_removed_below_bubble_stop_and_the_window_ends_or_the_run_goes_on(pierced,
                                                                                     tmp_path):
    """The end state of the ring stage with the bubble's volume at the gas cut taken four times as
    large (the ring then below ``bubble_stop`` of it, as it would be after its collapse): the
    removal is requested and made at the loop's top, the ring gone, the closures Q1-Q3 closed; with
    ``after_removal`` ``stop`` the window ends there, with ``continue`` the run takes three steps on
    the open chain alone, the cut's sizing kept."""
    run = pierced["run"]
    end = Path(pierced["result"]["stop"]["end_state"])
    end = end if end.is_absolute() else run.parent / end
    meta = json.loads(end.with_suffix(".json").read_text())
    meta["cut"]["geometry"]["bubble_volume"] *= 4.0
    meta["ring"]["V_b"] *= 4.0
    outs = {}
    for mode in ("stop", "continue"):
        block = json.loads(json.dumps(meta["events_settings"]))
        block["ring"]["after_removal"] = mode
        out = tmp_path / mode
        with S.pierce_problem(out / "pb") as pb:
            pb.initialise()
            pb.load_state(str(end))
            drv = CollapseDriver(pb, CollapseSettings(**DRIVER, max_steps=meta["step"] + 3),
                                 out / "run", reconnect=S.RECON,
                                 events=ev.EventSettings.from_dict(block))
            drv.restore(json.loads(json.dumps(meta)))
            result = drv.run(1e-8)
            drv.close()
            outs[mode] = {"result": result, "events": [dict(x) for x in pb.events],
                          "ring": dict(pb.ring), "loops": pb.interface_loops(),
                          "sizing": pb.active_mesh_template().cut_sizing,
                          "stage": pb.active_mesh_template().ring_stage,
                          "segs": len(pb.interface_segments())}
    for mode, got in outs.items():
        rec = got["events"][-1]
        assert (rec["kind"], rec["class"], rec["handler"], rec["outcome"]) == (
            "ring-collapse", "W2", "ring-removal", "made"), mode
        assert rec["closures"]["hard"] == [] and "Q3_box" in rec["closures"]
        assert rec["trigger"]["ratio"] < 0.01 and rec["after_removal"] == mode
        assert got["ring"]["removed"]["k"] == rec["k"] and got["loops"] == []
        assert got["segs"] == 1 and got["stage"] is True and got["sizing"] is not None
    stop = outs["stop"]["result"]
    assert stop["stop"]["reason"] == "bubble-vanishing" and stop["stop"]["window_end"]
    assert stop["stop"]["ring_removed"] == outs["stop"]["events"][-1]["k"]
    assert stop["window_end"]["reason"] == "bubble-vanishing"
    go = outs["continue"]["result"]
    assert go["stop"]["reason"] == "max-steps" and go["steps"] == meta["step"] + 3
    assert go["events"]["ring"]["removed"] == outs["continue"]["events"][-1]["k"]


# --- the hard closures fire: a violation of each stops the run event-unclosed --------------------


def _add_velocity(pb, region, du_r, du_z):
    """Add ``(du_r(r, z), du_z(r, z))`` to the velocity (current level) of the liquid's nodes where
    ``region(r, z)`` holds; ``u_r`` stays pinned on the axis."""
    m = pb.get_mesh("liquid")
    fi = m.get_nodal_field_indices()
    iu, iw = fi["velocity_x"], fi["velocity_y"]
    n_set = 0
    for n in m.nodes():
        r, z = n.x(0), n.x(1)
        if region(r, z):
            if not n.is_pinned(iu):
                n.set_value(iu, n.value(iu) + du_r(r, z))
            n.set_value(iw, n.value(iw) + du_z(r, z))
            n_set += 1
    pb.invalidate_cached_mesh_data()
    return n_set


def radial_inflow(r, z) -> float:
    """The part of ``ring_fem_state.closing_flow`` that is not solenoidal, its radial inflow."""
    ur_open, _ = S.closing_flow(np.array([r]), np.array([z]), solenoidal=False)
    ur, _ = S.closing_flow(np.array([r]), np.array([z]))
    return float(ur_open[0] - ur[0])


def _transfer_then(action):
    """``ringsurgery._transfer_check`` that, once the transfer has passed its checks, applies
    ``action()``: a field changed between the transfer and the fill, as a broken transfer would
    leave it."""
    original = ringsurgery._transfer_check

    def check(*a, **kw):
        problems = original(*a, **kw)
        if not problems:
            action()
        return problems

    return check


def _impulse_then(pb, region, dPi):
    """``surgery.sample_fields`` that, when the projection reads its pressure impulse on the box
    (the only reading with the pressure), first adds ``dPi`` to the impulse at the pressure nodes
    where ``region(r, z)`` holds: an impulse that is not the one the velocity changed by."""
    original = surgery.sample_fields

    def sample(problem, r, z, names):
        if "pressure" in names:
            ip, nodes = surgery._pressures(pb)
            for n, v in nodes:
                if region(n.x(0), n.x(1)):
                    n.set_value(ip, v + dPi)
            pb.invalidate_cached_mesh_data()
        return original(problem, r, z, names)

    return sample


def _volume_misread(extra_floors: float):
    """``ringsurgery._ring_closures`` with the filled volume read ``extra_floors`` times the limit
    of R1 (floored) too large: the liquid gains less than the surgery says it filled."""
    original = ringsurgery._ring_closures

    def closures(fill, *a, sigma=None, **kw):
        limit = max(ringsurgery.RING_BUDGET_LIMITS["R1"] * fill["V"], rs_floor("R1", sigma))
        return original(dict(fill, V=fill["V"] + extra_floors * limit), *a, sigma=sigma, **kw)

    return closures


def rs_floor(name: str, sigma: dict | None) -> float:
    noise, factor = ringsurgery.RING_NOISE_FLOORS[name]
    return factor * float((sigma or {}).get(noise, 0.0))


def _unclosed_run(pb, out: Path, meta: dict | None, block: ev.EventSettings, max_steps: int):
    """A run of the driver from the problem's state (the gas cut's, or a dump of the ring stage
    restored from ``meta``) up to its first surgery; returns its stop, its result and the record of
    the surgery."""
    driver = CollapseDriver(pb, CollapseSettings(**DRIVER, max_steps=max_steps), out,
                            reconnect=S.RECON, events=block)
    if meta is None:
        driver.reset_after_cut(pb.cut)
    else:
        driver.restore(json.loads(json.dumps(meta)))
    result = driver.run(1e-8)
    driver.close()
    stop = result["stop"]
    path = out / "events" / f"event_{int(stop.get('k', 0)):03d}.json"
    rec = json.loads(path.read_text()) if path.is_file() else {}
    return {"stop": stop, "result": result, "record": rec}


@pytest.fixture(scope="module")
def unclosed(pierced, tmp_path_factory):
    """The ring cut of the pierced state and the ring's removal, each made by the driver with one
    hard closure violated: R1 (Q1) the filled volume read three floors too large; R2 (Q2) axial
    momentum added to the liquid between the transfer and the fill, outside the filled region;
    R3 (Q3) the pressure impulse the box's balance reads shifted in the box's lower shell; R3_box a
    flow that is not solenoidal (the radial inflow of ``ring_fem_state.closing_flow``), whose
    projection moves momentum out of the box, and Q3_box the same inflow added after the removal's
    transfer; the removal's violations are sized against its own limits, its scale ``P_ref``
    being that of the ring's volume."""
    out = tmp_path_factory.mktemp("unclosed")
    noise = pierced["noise"]
    events = ring_block(noise)
    g = pierced["geometry"]
    z_mid, R_b = g.z_mid, max(4.0 * S.DELTA, 2.0 * g.L_fill)
    res = {}
    with S.pierce_problem(out / "pb") as pb:
        pb.initialise()
        cases = {
            "R1": lambda: {"_ring_closures": _volume_misread(3.0)},
            "R2": lambda: {"_transfer_check": _transfer_then(lambda: _add_velocity(
                pb, lambda r, z: r < 0.06 and 0.47 < z < 0.53, lambda r, z: 0.0,
                lambda r, z: 1e-2))},
            "R3": lambda: {"sample_fields": _impulse_then(
                pb, lambda r, z: z < z_mid and 0.6 * R_b < math.hypot(r, z - z_mid) < R_b,
                1e-3)},
            "R3_box": lambda: {}}
        for name, patches in cases.items():
            reload(pb, pierced["initial"], pierced["record"])
            if name == "R3_box":
                S.impose(pb, solenoidal=False)
            with pytest.MonkeyPatch.context() as mp:
                for attr, fn in patches().items():
                    mp.setattr(surgery if attr == "sample_fields" else ringsurgery, attr, fn)
                res[name] = _unclosed_run(pb, out / name, None, events, 1)
    run = pierced["run"]
    end = Path(pierced["result"]["stop"]["end_state"])
    end = end if end.is_absolute() else run.parent / end
    meta = json.loads(end.with_suffix(".json").read_text())
    meta["cut"]["geometry"]["bubble_volume"] *= 4.0
    meta["ring"]["V_b"] *= 4.0
    block = ev.EventSettings.from_dict(meta["events_settings"])
    for name in ("Q1", "Q2", "Q3", "Q3_box"):
        with S.pierce_problem(out / f"pb-{name}") as pb:
            pb.initialise()
            pb.load_state(str(end))
            loop = pb.interface_loops()[0]
            L = np.column_stack([loop["r"], loop["z"]])
            z_c = float(ring.loop_record(L)["z_c"])  # the removal's box, as ring_removal takes it
            Q = ring.closed(L)
            R_q = max(4.0 * S.DELTA, 2.0 * float(np.max(np.hypot(Q[:, 0], Q[:, 1] - z_c))))
            # the removal's P_ref (the ring's volume times its largest speed) and the limits of Q2
            # and Q3_box it sets, which the violations exceed severalfold
            P_q = rc.revolved_volume(L) * float(np.max(np.hypot(loop["u_r"], loop["u_z"])))
            limit = max(0.05 * P_q, rs_floor("R2", {"Pz": block.ring.sigma_Pz}))
            blob = math.pi * 0.06**2 * 0.06  # the liquid below the plug the momentum is added to
            inflow = 30.0 * limit / 1e-4  # the cut's radial inflow moved about 1e-4 out of its box
            patches = {
                "Q1": {"_ring_closures": _volume_misread(3.0)},
                "Q2": {"_transfer_check": _transfer_then(lambda: _add_velocity(
                    pb, lambda r, z: r < 0.06 and 0.47 < z < 0.53, lambda r, z: 0.0,
                    lambda r, z: 5.0 * limit / blob))},
                "Q3": {"sample_fields": _impulse_then(
                    pb, lambda r, z: z < z_c and 0.6 * R_q < math.hypot(r, z - z_c) < R_q,
                    1e-3)},
                "Q3_box": {"_transfer_check": _transfer_then(lambda: _add_velocity(
                    pb, lambda r, z: True,
                    lambda r, z: inflow * radial_inflow(r, z), lambda r, z: 0.0))}}[name]
            with pytest.MonkeyPatch.context() as mp:
                for attr, fn in patches.items():
                    mp.setattr(surgery if attr == "sample_fields" else ringsurgery, attr, fn)
                res[name] = _unclosed_run(pb, out / name, meta, block, meta["step"] + 1)
    return res


@pytest.mark.parametrize("name", ["R1", "R2", "R3", "R3_box", "Q1", "Q2", "Q3", "Q3_box"])
def test_a_violated_hard_closure_stops_the_run_event_unclosed_with_the_state_before_it(unclosed,
                                                                                        name):
    """Each violation is a hard miss of its closure beyond its limit, floored at the box's noise:
    the surgery is not accepted, the run stops ``event-unclosed`` with the state before the event
    kept (``precut_e<k>``) and the record ``unclosed``, its closures and its hard misses."""
    got = unclosed[name]
    stop, rec = got["stop"], got["record"]
    assert stop["reason"] == "event-unclosed", json.dumps(stop, default=float)
    assert name in stop["hard"] and name in rec["closures"]["hard"], rec["closures"]
    assert rec["outcome"] == "unclosed" and rec["failed_at"] == "postcut"
    c = rec["closures"]
    floor = rs_floor("R" + name[1:], c["sigma"])
    assert c[name] > c["limits"][name] >= floor > 0.0
    precut = Path(stop["precut"])
    assert precut.is_file() and precut.name == f"precut_e{int(stop['k']):03d}.dump"
    assert rec["dumps"]["precut"] == stop["precut"]
    assert "postcut" not in rec["dumps"]
