"""Finite-element tests of the events the collapse driver makes within a run after the cut, with
the events block (``collapse.reconnect_events``), on the synthetic state of
``test_liquid_cut_fem.py``: an open chain whose tip is the head of a jet rising from a groove and a
bubble whose top holds a hanging jet, each with a landed neck, built as the first mesh of the
problem with a synthetic record of the gas cut and an imposed solenoidal flow about each neck.

* two events in one run: the driver meets each landed neck at the check of its stops, cuts it,
  re-measures the fresh mesh, checks its stops as those of a mesh freshly built and goes on to its
  step limit, with its records, rows, ghosts, budget and dumps, the meshes no dump refers to pruned;
* a cut refused before anything changed stops the run resumably and leaves the problem as it was;
* a run whose driver makes no liquid cut stops on the first event with the state before it, from
  which, in a new process, a driver that makes the cut makes both events and goes on; in another
  process the state that run ended on, its mesh kept by the pruning, restarts and steps.

Each problem compiles in seconds; the module takes a few minutes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

pytest.importorskip("pyoomph")

import test_liquid_cut_fem as jets  # noqa: E402

from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity.driver import CollapseDriver, CollapseSettings  # noqa: E402

DRIVER = {"t_end": 10.0, "dt_max": 0.01, "max_steps": 3, "h_floor": 1e-6}


def _rows(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    head = lines[0][2:].split("\t")
    return [dict(zip(head, line.split("\t"))) for line in lines[1:]]


def _jets_driver(out: Path, events=jets.EVENTS):
    """The synthetic state of ``test_liquid_cut_fem.py`` after its gas cut, with the imposed flow,
    and a driver in the problem's own output directory, as the runner makes it."""
    pb = jets.jets_problem(out)
    pb.__enter__()
    pb.initialise()
    jets.gas_cut_record(pb)
    jets.impose(pb, jets.stream_velocity, jets.jet_chains()[2])
    drv = CollapseDriver(pb, CollapseSettings(**DRIVER), out, reconnect=jets.RECON, events=events,
                         leg_info={"commit": "f" * 40})
    drv.reset_after_cut(pb.cut)
    return pb, drv


def _finish(pb, drv):
    drv.close()
    pb.__exit__(None, None, None)


def _mesh_files(run: Path) -> tuple[set, set]:
    from pyoomph.output.states import get_state_file_mesh_files

    meshes = {p.resolve() for p in (run / "_gmsh").glob("*.msh")}
    referred = set()
    for dump in (run / "states").glob("*.dump"):
        referred |= {Path(f).resolve() for f in get_state_file_mesh_files(str(dump))}
    return meshes, referred


@pytest.fixture(scope="module")
def two_events(tmp_path_factory):
    out = tmp_path_factory.mktemp("two-events")
    pb, drv = _jets_driver(out)
    try:
        result = drv.run(1e-8)
        events = [dict(e) for e in pb.events]
    finally:
        _finish(pb, drv)
    return {"out": out, "result": result, "events": events}


def test_two_events_in_one_run_are_made_and_the_run_goes_on(two_events):
    """The open chain's landed neck first (the nearest its landing), then the bubble's, each cut at
    the check of the stops at the cut's time, the second on the fresh mesh of the first; the run
    then takes its steps to its limit. Each record has the driver's bookkeeping (the jet's base
    before the cut, the generation at its tip, the cascade's verdict, the step, the leg, the
    commit, the ghost's start, the jet after it), each event a row of events.txt and of drops.txt,
    events.jsonl a liquid-cut line each, and the run's result the summary of its events."""
    out, result, made = two_events["out"], two_events["result"], two_events["events"]
    assert result["stop"]["reason"] == "max-steps" and result["steps"] == DRIVER["max_steps"]
    assert [(e["k"], e["chain"], e["tip"]) for e in made] == [(1, "open", "cavity_tip"),
                                                              (2, "bubble", "bubble_top")]
    for e in made:
        assert e["outcome"] == "made" and e["step"] == 0 and e["leg"] == 0
        assert e["generation"] == 1 and e["cascade"]["verdict"] == "first"
        assert e["commit"] == "f" * 40 and e["closures"]["hard"] == []
        assert e["trigger"]["landed"] and e["trigger"]["r_j"] > e["trigger"]["r_n"]
        assert e["trigger"]["r_j_over_r_n"] == pytest.approx(e["trigger"]["r_j"]
                                                             / e["trigger"]["r_n"])
        assert e["ghost"]["tip"] == e["tip"] and e["ghost"]["step0"] == 0
        assert e["ghost"]["far_tip"] == ev.FAR_TIPS[e["tip"]]
        assert e["after"]["r_j"] is not None and e["dt"]["history"] == e["dt"]["before"]
        for stage in ("precut", "postcut"):
            assert Path(e["dumps"][stage]).is_file()
    assert made[1]["t"] == made[0]["t"]  # both at the cut's time, the second on the first's mesh
    rows = _rows(out / "events.txt")
    assert [(r["k"], r["tip"]) for r in rows] == [("1", "cavity_tip"), ("2", "bubble_top")]
    assert [(r["k"], r["status"]) for r in _rows(out / "drops.txt")][:2] == [("1", "free"),
                                                                            ("2", "free")]
    lines = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
    assert [x["k"] for x in lines if x["event"] == "liquid-cut"] == [1, 2]
    summary = result["events"]
    assert summary["made"] == 2 and summary["by_tip"] == {"cavity_tip": 1, "bubble_top": 1}
    assert summary["removed_total"]["V"] == pytest.approx(sum(e["removed"]["V"] for e in made))
    assert result["window_end"] is None


def test_the_budget_of_the_run_with_two_events_holds_their_removals(two_events):
    """budget.txt: the liquid of the mesh of every step after the cut, both removals in it from the
    first step on, and their sums."""
    out, made = two_events["out"], two_events["events"]
    rows = _rows(out / "budget.txt")
    assert len(rows) == DRIVER["max_steps"] + 1
    assert rows[0]["events"] == "0" and all(r["events"] == "2" for r in rows[1:])
    V = sum(e["removed"]["V"] for e in made)
    for r in rows[1:]:
        assert float(r["V_removed"]) == pytest.approx(V, rel=1e-11)
        assert float(r["volume_total"]) == pytest.approx(float(r["volume"]) + V, rel=1e-12)


def test_the_meshes_no_dump_of_the_run_refers_to_are_pruned(two_events):
    """Every mesh left in _gmsh is one a dump of the run refers to, or the one the problem was
    built from at its start; the dumps of both events are kept with theirs."""
    out = two_events["out"]
    meshes, referred = _mesh_files(out)
    first = {p.resolve() for p in (out / "_gmsh").glob("JetsMesh.msh")}
    assert meshes - first <= referred <= meshes
    names = sorted(p.name for p in (out / "states").glob("p*cut_e*.dump"))
    assert names == ["postcut_e001.dump", "postcut_e002.dump", "precut_e001.dump",
                     "precut_e002.dump"]


def test_a_refused_cut_stops_the_run_resumably_and_leaves_the_problem_unchanged(tmp_path,
                                                                                monkeypatch):
    """A cap refused at its height and at the retry's: the run stops at once
    (``liquid-cut-refused``, failed at the caps) with the state before the event, which the cut
    kept, and the problem as it was: no rebuild, no event; the state it ends on is dumped."""
    monkeypatch.setattr(ev, "truncate_chain", jets._refuse)
    pb, drv = _jets_driver(tmp_path)
    try:
        before = pb.remesh_count
        result = drv.run(1e-8)
        assert pb.remesh_count == before and pb.events == []
    finally:
        _finish(pb, drv)
    stop = result["stop"]
    assert stop["reason"] == ev.LIQUID_CUT_REFUSED and stop["k"] == 1 and result["steps"] == 0
    assert stop["failed_at"] == "caps" and stop["record"] == "events/event_001.json"
    assert Path(stop["precut"]).name == "precut_e001.dump"
    assert Path(stop["end_state"]).is_file()
    record = json.loads((tmp_path / "events" / "event_001.json").read_text())
    assert record["outcome"] == "refused" and record["failed_at"] == "caps"


RESUME = """
import json, math, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import test_events_run_fem as T
import test_liquid_cut_fem as J
from faraday_singularity import events as ev
from faraday_singularity.driver import CollapseDriver, CollapseSettings
state, out, steps = Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5])
pb = J.jets_problem(out)
with pb:
    pb.initialise()
    pb.load_state(str(state))
    meta = json.loads(state.with_suffix(".json").read_text())
    drv = CollapseDriver(pb, CollapseSettings(**dict(T.DRIVER, max_steps=steps)), out,
                         reconnect=J.RECON, events=J.EVENTS)
    drv.restore(meta)
    # what the restored state holds of the jets' bases, and their radii on the loaded mesh
    restored = {"apex_history": drv.state.apex_history, "kinks": drv.state.kinks,
                "apexes": {tip: a["R_apex"] for tip, a in drv._post_cut_state()["apexes"].items()
                           if math.isfinite(a["R_apex"])}}
    res = drv.run(1e-8)
    drv.close()
    report = {"stop": res["stop"], "steps": res["steps"], "events": [e["k"] for e in pb.events],
              "tips": [e.get("tip") for e in pb.events], "leg": drv.state.legs[-1],
              "summary": res["events"], "restored": restored}
print(json.dumps(report))
"""


def _resume(state: Path, out: Path, steps: int) -> dict:
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-c", RESUME, str(ROOT / "src"),
                           str(ROOT / "testCases"), str(state), str(out), str(steps)],
                          capture_output=True, text=True, timeout=1500, check=False, env=env)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_a_run_resumes_from_the_dump_of_an_unhandled_event_with_the_handler(tmp_path,
                                                                            monkeypatch):
    """A driver that makes no liquid cut (as before the handler existed) stops on the first landed
    neck at once (``event-unhandled``) with the state before it. In a new process a driver that
    makes the cut restores that state, makes both events and takes its steps (its leg from the
    state before event 1); in another, the state that run ended on, whose mesh the pruning kept,
    restarts and takes two more steps."""
    import faraday_singularity.driver as module

    monkeypatch.setattr(module, "IMPLEMENTED_HANDLERS", ())
    pb, drv = _jets_driver(tmp_path / "first")
    try:
        result = drv.run(1e-8)
    finally:
        _finish(pb, drv)
    stop = result["stop"]
    assert stop["reason"] == ev.EVENT_UNHANDLED and stop["k"] == 1 and result["steps"] == 0
    assert stop["handler"] == "liquid-cut" and stop["chain"] == "open"
    precut = Path(stop["precut"])
    meta = json.loads(precut.with_suffix(".json").read_text())
    assert meta["cut_stage"] == ev.PRECUT_EVENT and meta["mesh_fresh"] is True
    rec = json.loads((tmp_path / "first" / stop["record"]).read_text())
    assert (rec["outcome"], rec["kind"], rec["class"], rec["handler"], rec["tip"]) == (
        "unhandled", "liquid-neck", "L", "liquid-cut", "cavity_tip")
    assert rec["dumps"]["precut"] == stop["precut"] and rec["trigger"]["ratio"] <= 1.0
    again = _resume(precut, tmp_path / "again", DRIVER["max_steps"])
    assert again["stop"]["reason"] == "max-steps" and again["steps"] == DRIVER["max_steps"]
    assert again["events"] == [1, 2] and again["tips"] == ["cavity_tip", "bubble_top"]
    made = json.loads((tmp_path / "again" / "events" / "event_001.json").read_text())
    assert made["outcome"] == "made" and made["dumps"]["precut"] != stop["precut"]
    assert again["leg"]["from_event_state"] == 1 and again["leg"]["changed"] == []
    end = Path(again["stop"]["end_state"])
    later = _resume(end, tmp_path / "later", DRIVER["max_steps"] + 2)
    assert later["stop"]["reason"] == "max-steps" and later["steps"] == DRIVER["max_steps"] + 2
    assert later["events"] == [1, 2] and later["summary"]["made"] == 2


def _numbers(row: dict) -> dict:
    return {k: float(v) for k, v in row.items()}


def _remesh_events(run: Path) -> list[dict]:
    """The rebuilds of a run after its start (events ``remesh`` from step 1 on): the step, the
    regions that asked and each region's name, target and the size the rebuild set."""
    lines = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
    return [{"step": x["step"], "asked_by": x.get("asked_by"),
             "regions": [(g["name"], g["target"], g.get("new_size")) for g in x.get("regions", [])]}
            for x in lines if x["event"] == "remesh" and x["step"] >= 1]


def test_a_restart_from_the_state_after_an_event_goes_on_as_the_run_did(two_events, tmp_path):
    """The dump of the state after an event (``postcut_e<k>``) holds the bases of the jets and the
    kink regions of its fresh mesh, as the run commits them before it steps from it: in a new
    process the restart from the state after the second event restores as the history of each jet's
    base its radius on that mesh, at the dump's time, and takes the steps the run took, row for row
    (``chains.txt``) and rebuild for rebuild (the regions of its ``remesh`` events)."""
    out = two_events["out"]
    state = out / "states" / "postcut_e002.dump"
    meta = json.loads(state.with_suffix(".json").read_text())
    again = _resume(state, tmp_path / "again", DRIVER["max_steps"])
    restored = again["restored"]
    assert restored["apexes"], "the jets of the synthetic state have bases"
    assert restored["apex_history"] == {tip: [meta["time"], R]
                                        for tip, R in restored["apexes"].items()}
    assert again["stop"]["reason"] == "max-steps" and again["events"] == [1, 2]
    rows = {int(r["step"]): _numbers(r) for r in _rows(out / "chains.txt")}
    resumed = {int(r["step"]): _numbers(r) for r in _rows(tmp_path / "again" / "chains.txt")}
    steps = range(1, DRIVER["max_steps"] + 1)
    assert all(k in rows and k in resumed for k in steps)
    for k in steps:
        for name, v in rows[k].items():
            assert resumed[k][name] == pytest.approx(v, rel=1e-9, abs=0.0, nan_ok=True), (k, name)
    ours, theirs = _remesh_events(out), _remesh_events(tmp_path / "again")
    assert [(x["step"], x["asked_by"]) for x in theirs] == [(x["step"], x["asked_by"])
                                                            for x in ours]
    for a, b in zip(ours, theirs):
        assert [g[0] for g in a["regions"]] == [g[0] for g in b["regions"]]
        for (name, target, size), (_, target_b, size_b) in zip(a["regions"], b["regions"]):
            assert target_b == pytest.approx(target, rel=1e-9, abs=0.0), (a["step"], name)
            assert (size is None) == (size_b is None), (a["step"], name)
            if size is not None:
                assert size_b == pytest.approx(size, rel=1e-9, abs=0.0), (a["step"], name)
