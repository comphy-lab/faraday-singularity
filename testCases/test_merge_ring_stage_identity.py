"""The merge of phase1f-ring-stage into phase1f-core-res against 8ee5a49, the core's commit before
it: the source of this commit and that of 8ee5a49, each in a fresh process with MKL on one thread.
Runs with the core meshed finer (``jetcore.core_elements_per_scale``) that meet none of the ring
stage's handling (no guard stop on the ring or a folded tail, no fill, no failed step) are
8ee5a49's to the byte:

* the finite-element run after the cut of the synthetic necked cavity under the production block
  with ``core_elements_per_scale`` 16, and a restart in a new process from one of its dumps: every
  output and every mesh gmsh wrote;
* the finite-element run of the synthetic pierced state with ``ring.enabled``, jetcore's far sizing
  and the finer core: the ring cut made by the driver and the steps of the ring stage;
* the far-head cut of ``test_jetcore_fem.py``'s heads' state under jetcore's far sizing and the
  finer core, and the steps after it.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import test_events_fem as fem_pins  # noqa: E402
from test_events_fem import _reference_src  # noqa: E402
from test_events_gasneck import _run  # noqa: E402
from test_jetcore_identity import _run_fem, _spec  # noqa: E402
from test_jetcore_res_identity import FAR_HEAD_FAR, PRODUCTION, RING_RUN_FAR  # noqa: E402
from test_ring_identity import _drop_wall_column  # noqa: E402

BASE = "8ee5a49"  # phase1f-core-res before the merge
FINE = {**PRODUCTION, "jetcore": {**PRODUCTION["jetcore"], "core_elements_per_scale": 16}}
RING_RUN_FINE = RING_RUN_FAR.replace('"far_refine_ratio": 2.0}}',
                                     '"far_refine_ratio": 2.0, "core_elements_per_scale": 16}}')
assert RING_RUN_FINE != RING_RUN_FAR
FAR_HEAD_FINE = FAR_HEAD_FAR.replace("'edge_turn_snapshots': 'core'}",
                                     "'edge_turn_snapshots': 'core', 'core_elements_per_scale': 16}")
assert FAR_HEAD_FINE != FAR_HEAD_FAR


def test_a_finite_element_run_with_the_finer_core_and_its_restart_are_8ee5a49s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    spec = _spec(**FINE)
    runs = {name: _run_fem(src, tmp_path / name, spec) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    names = fem_pins._same_outputs(runs["ref"], runs["head"])
    result = json.loads((runs["head"] / "result.json").read_text())
    assert result["steps"] == 6 and {"budget.txt", "jetcore.txt"} <= set(names), result
    # the restarts from copies of the runs at one relative place, so that the path of the parent's
    # mesh their first dumps name is the same in both (the run directories' names differ)
    again = {}
    for name, src in (("ref", ref), ("head", ROOT / "src")):
        home = tmp_path / f"{name}-again"
        shutil.copytree(tmp_path / name / "run", home / "parent" / "run")
        r = dict(spec, restart=str(home / "parent" / "run" / "states" / "state_00000003.dump"))
        again[name] = _run_fem(src, home, r, out="restart")
    fem_pins._same_outputs(again["ref"], again["head"])


def test_a_ring_run_with_the_finer_core_is_8ee5a49s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    runs = {name: _run(RING_RUN_FINE, src, tmp_path / name) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    for run in runs.values():
        _drop_wall_column(run / "events.txt")
    names = fem_pins._same_outputs(runs["ref"], runs["head"])
    result = json.loads((runs["head"] / "result.json").read_text())
    assert result["stop"]["reason"] == "max-steps" and result["steps"] == 4, result["stop"]
    assert result["events"]["made"] == 1 and {"ring.txt", "jetcore.txt"} <= set(names)


def test_the_far_head_cut_with_the_finer_core_is_8ee5a49s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    script = FAR_HEAD_FINE % '{"groove": {"enabled": True}}'
    runs = {name: _run(script, src, tmp_path / name) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    for run in runs.values():
        _drop_wall_column(run / "events.txt")
    names = fem_pins._same_outputs(runs["ref"], runs["head"])
    result = json.loads((runs["head"] / "result.json").read_text())
    assert result["events"]["by_kind"] == {"far-head": 1}
    assert {"events.txt", "jetcore.txt", "events/event_001.json"} <= set(names)
