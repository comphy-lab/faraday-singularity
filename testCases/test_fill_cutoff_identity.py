"""The fill's budgets weighted by the box's cutoff against b46713a, the commit before it: the
source of this commit and that of b46713a, each in a fresh process with MKL on one thread. Runs
that make no fill are b46713a's to the byte (the cutoff weights only the box of a groove's fill,
``groovesurgery``; the liquid cut's and the ring surgeries' boxes are unchanged):

* the stops of the events branch on every synthetic state of ``test_events_gasneck.py``'s battery;
* the finite-element run after the cut of the synthetic necked cavity with the block and
  ``groove.enabled`` (``groove.sheets`` ``log``), which meets no groove, and a restart in a new
  process from one of its dumps: every output and every mesh gmsh wrote;
* the finite-element run of the synthetic pierced state with the production's ring stage: the ring
  cut (its box unweighted as before) and the steps of the ring stage;
* the far-head cut of ``test_jetcore_fem.py``'s heads' state made by the driver, with and without
  ``groove.enabled``, and the steps after it.
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
from test_jetcore_identity import _run_fem, _spec  # noqa: E402
from test_ring_identity import _drop_wall_column  # noqa: E402
from test_ring_refine_identity import RING_RUN_STAGE  # noqa: E402

BASE = "b46713a"  # the commit before the cutoff's weight


def test_the_stops_are_b46713as(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    out = {name: json.loads(_run(BATTERY, src, tmp_path / name).read_text())
           for name, src in (("ref", ref), ("head", ROOT / "src"))}
    assert out["ref"] == out["head"]
    assert len(out["head"]) > 40


def test_a_finite_element_run_with_groove_enabled_and_its_restart_are_b46713as(tmp_path):
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


def test_a_ring_run_with_the_production_ring_stage_is_b46713as(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    runs = {name: _run(RING_RUN_STAGE, src, tmp_path / name) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    for run in runs.values():
        _drop_wall_column(run / "events.txt")
    names = fem_pins._same_outputs(runs["ref"], runs["head"])
    result = json.loads((runs["head"] / "result.json").read_text())
    assert result["stop"]["reason"] == "max-steps" and result["steps"] == 4, result["stop"]
    assert result["events"]["made"] == 1 and result["events"]["by_kind"] == {"piercing": 1}
    assert {"ring.txt", "events.txt", "budget.txt", "events/event_001.json"} <= set(names)


def test_the_far_head_cut_made_by_the_driver_is_b46713as(tmp_path):
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
