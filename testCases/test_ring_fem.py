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
    assert hard and set(hard) <= {"R1", "R2", "R3"}
    noise = {"R1": "V", "R2": "Pz", "R3": "Pz"}
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
    assert c["hard"] == [] and set(c["limits"]) == {f"R{i}" for i in range(1, 7)}
    # the cutoff identity against its scale, the larger of its sides and the column's impulse
    # (here the solenoidal flow leaves the projection little to change in the box)
    assert c["R3"] <= c["limits"]["R3"] and c["R3"] <= 1e-2 * c["scales"]["P_ref"]
    assert math.isfinite(c["R3_phi"]["phi_relative_error"]) and c["R3_box"] >= 0.0
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
