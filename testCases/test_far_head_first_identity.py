"""The far-head cut first, before any fill, and the curled head, against cb12382, the commit before
them: the source of this commit and that of cb12382, each in a fresh process with MKL on one
thread. Runs that meet no groove and no unhandled event on a far head, and no curled head, are
cb12382's to the byte:

* the stops of the events branch on every synthetic state of ``test_events_gasneck.py``'s battery;
* the finite-element run after the cut of the synthetic necked cavity with the block and
  ``groove.enabled`` (``groove.sheets`` ``log``), and a restart in a new process from one of its
  dumps: every output and every mesh gmsh wrote;
* the finite-element run of the synthetic pierced state with ``ring.enabled`` and
  ``groove.enabled``: the ring cut made by the driver and the steps of the ring stage;
* the far-head cut of ``test_jetcore_fem.py``'s heads' state made by the driver (a gas neck on the
  bubble's head, ``pinch.far_head``), with and without ``groove.enabled``, and the steps after it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import test_events_fem as fem_pins  # noqa: E402
from test_events_fem import _reference_src  # noqa: E402
from test_events_gasneck import BATTERY, _run  # noqa: E402
from test_groove_identity import RING_RUN_GROOVE  # noqa: E402
from test_jetcore_identity import _run_fem, _spec  # noqa: E402
from test_ring_identity import _drop_wall_column  # noqa: E402

BASE = "cb12382"  # the commit before the far-head cut first and the curled head


def test_the_stops_are_cb12382s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    out = {name: json.loads(_run(BATTERY, src, tmp_path / name).read_text())
           for name, src in (("ref", ref), ("head", ROOT / "src"))}
    assert out["ref"] == out["head"]
    assert len(out["head"]) > 40


def test_a_finite_element_run_with_groove_enabled_and_its_restart_are_cb12382s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    spec = _spec(groove={"enabled": True, "sheets": "log"})
    runs = {name: _run_fem(src, tmp_path / name, spec) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    names = fem_pins._same_outputs(runs["ref"], runs["head"])
    result = json.loads((runs["head"] / "result.json").read_text())
    assert result["steps"] == 6 and result["remesh_count"] >= 2, result
    assert result["events"]["made"] == 0 and "budget.txt" in names
    again = {}
    for name, src in (("ref", ref), ("head", ROOT / "src")):
        r = dict(spec, restart=str(tmp_path / name / "run" / "states" / "state_00000003.dump"))
        again[name] = _run_fem(src, tmp_path / f"{name}-restart", r)
    fem_pins._same_outputs(again["ref"], again["head"])


def test_a_ring_run_with_groove_enabled_is_cb12382s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    runs = {name: _run(RING_RUN_GROOVE, src, tmp_path / name) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    for run in runs.values():
        _drop_wall_column(run / "events.txt")
    names = fem_pins._same_outputs(runs["ref"], runs["head"])
    result = json.loads((runs["head"] / "result.json").read_text())
    assert result["stop"]["reason"] == "max-steps" and result["steps"] == 4, result["stop"]
    assert result["events"]["made"] == 1 and result["events"]["ring"]["cut"] == 1
    assert {"ring.txt", "events.txt", "budget.txt", "events/event_001.json"} <= set(names)


FAR_HEAD_RUN = """
import json, sys
from pathlib import Path
src, tests, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, src)
import faraday_singularity  # the source under test, before the tests put theirs on the path
assert Path(faraday_singularity.__file__).resolve().is_relative_to(Path(src).resolve())
sys.path.insert(1, tests)
import test_liquid_cut_fem as lcf
import test_jetcore_fem as J
from faraday_singularity import events as ev, reconnect as rc, surgery
from faraday_singularity.driver import CollapseDriver, CollapseSettings
assert all(Path(m.__file__).resolve().is_relative_to(Path(src).resolve())
           for m in (ev, rc, surgery, J.surgery))
GROOVE = %s
block = ev.EventSettings.from_dict({**J.FAR_HEAD_BLOCK, **GROOVE})
recon = rc.ReconnectSettings(delta=J.DELTA, dt0=1e-5, tip_size=0.25, bridge_size=0.25,
                             focus_size=0.25)
with J.heads_problem(Path("pb"), recon) as pb:
    pb.initialise()
    J.gas_cut_record(pb, recon)
    lcf.impose(pb, lcf.stream_velocity, list(J._chains()[2]))
    drv = CollapseDriver(pb, CollapseSettings(t_end=10.0, dt_max=0.01, max_steps=2, state_dt=0.0,
                                              state_keep=10), out, reconnect=recon, events=block)
    m = drv.reset_after_cut(pb.cut)
    named = drv._named_post_cut_chains()
    neck = next(x for x in ev.liquid_necks(named, block.liquid)
                if x["chain"] == "bubble" and x["status"] == "head")
    site = neck["head"][1] - 6
    gas = {"reason": ev.EVENT_REQUEST, "kind": "gas-neck", "class": "G",
           "handler": ev.HANDLERS["G"], "chain": "bubble", "index": site,
           "r_neck": float(named[0][1].points[site, 0])}
    m2, stop = drv._meet_event(m, gas)
    assert stop is None, stop
    result = drv.run(1e-8)
    drv.close()
(out / "result.json").write_text(json.dumps(result, default=float, sort_keys=True))
"""


def test_the_far_head_cut_made_by_the_driver_is_cb12382s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    for label, groove in (("plain", "{}"), ("groove", '{"groove": {"enabled": True}}')):
        script = FAR_HEAD_RUN % groove
        runs = {name: _run(script, src, tmp_path / label / name) for name, src in
                (("ref", ref), ("head", ROOT / "src"))}
        for run in runs.values():
            _drop_wall_column(run / "events.txt")
        names = fem_pins._same_outputs(runs["ref"], runs["head"])
        result = json.loads((runs["head"] / "result.json").read_text())
        assert result["events"]["made"] == 1 and result["events"]["by_kind"] == {"far-head": 1}
        assert result["stop"]["reason"] == "max-steps", result["stop"]
        assert {"events.txt", "budget.txt", "events/event_001.json"} <= set(names), label
