"""The cut of a far head whose neck a freshly built mesh does not resolve, against 203c7e7, the
commit before it on phase1f-ring-stage (the commit was made on the gas line, c3c7d5d, against
7efe4df, and picked onto the ring stage; its pin of the cut with a polytropic bubble stays on the gas
line, whose modules this one does not hold): the source of this commit and that of 203c7e7, each in
a fresh process with MKL on one thread. Runs that meet no unresolved neck are 203c7e7's to the byte:

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
from test_far_head_first_identity import FAR_HEAD_RUN  # noqa: E402
from test_groove_identity import RING_RUN_GROOVE  # noqa: E402
from test_jetcore_identity import _run_fem, _spec  # noqa: E402
from test_ring_identity import _drop_wall_column  # noqa: E402

BASE = "203c7e7"  # phase1f-ring-stage before the cut of a far head at an unresolved neck


def test_the_stops_are_203c7e7s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    out = {name: json.loads(_run(BATTERY, src, tmp_path / name).read_text())
           for name, src in (("ref", ref), ("head", ROOT / "src"))}
    assert out["ref"] == out["head"]
    assert len(out["head"]) > 40


def test_a_finite_element_run_with_groove_enabled_and_its_restart_are_203c7e7s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    spec = _spec(groove={"enabled": True, "sheets": "log"})
    runs = {name: _run_fem(src, tmp_path / name, spec) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    names = fem_pins._same_outputs(runs["ref"], runs["head"])
    result = json.loads((runs["head"] / "result.json").read_text())
    assert result["steps"] == 6 and result["events"]["made"] == 0 and "budget.txt" in names
    again = {}
    for name, src in (("ref", ref), ("head", ROOT / "src")):
        r = dict(spec, restart=str(tmp_path / name / "run" / "states" / "state_00000003.dump"))
        again[name] = _run_fem(src, tmp_path / f"{name}-restart", r)
    fem_pins._same_outputs(again["ref"], again["head"])


def test_a_ring_run_with_groove_enabled_is_203c7e7s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    runs = {name: _run(RING_RUN_GROOVE, src, tmp_path / name) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    for run in runs.values():
        _drop_wall_column(run / "events.txt")
    names = fem_pins._same_outputs(runs["ref"], runs["head"])
    result = json.loads((runs["head"] / "result.json").read_text())
    assert result["events"]["made"] == 1 and result["events"]["ring"]["cut"] == 1
    assert {"ring.txt", "events.txt", "budget.txt", "events/event_001.json"} <= set(names)


def test_the_far_head_cut_made_by_the_driver_is_203c7e7s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    for label, groove in (("plain", "{}"), ("groove", '{"groove": {"enabled": True}}')):
        script = FAR_HEAD_RUN % groove
        runs = {name: _run(script, src, tmp_path / label / name) for name, src in
                (("ref", ref), ("head", ROOT / "src"))}
        for run in runs.values():
            _drop_wall_column(run / "events.txt")
        names = fem_pins._same_outputs(runs["ref"], runs["head"])
        result = json.loads((runs["head"] / "result.json").read_text())
        assert result["events"]["by_kind"] == {"far-head": 1}, label
        assert {"events.txt", "budget.txt", "events/event_001.json"} <= set(names), label

