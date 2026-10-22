"""The stabilisation of the run after the cut against 4a6c999, the commit before it: the source of
this commit and that of 4a6c999, each in a fresh process with MKL on one thread.

Without the new parts (the step timing off, no ``jetcore`` or ``pinch`` part in the events block)
every run is 4a6c999's to the byte:

* the stops of the events branch on every synthetic state of ``test_events_gasneck.py``'s battery,
  the block's settings, their hash and the cut's keys;
* the finite-element run after the cut of the synthetic necked cavity with the block
  (``test_events_fem.py``: rebuilds by the rule at every step it may, a dump at every step) and a
  restart in a new process from one of its dumps: every output and every mesh gmsh wrote;
* the finite-element run of the synthetic pierced state with ``ring.enabled``
  (``test_ring_identity.py``): the ring cut made by the driver and the steps of the ring stage.

With the step timing (the runner's ``--step-timing``) the same finite-element run writes the same
files, ``step_timing.txt`` besides.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import test_events_fem as fem_pins  # noqa: E402
from test_events_fem import _reference_src, _without_wall  # noqa: E402
from test_events_gasneck import BATTERY, _run  # noqa: E402
from test_ring_identity import RING_RUN, _drop_wall_column  # noqa: E402

BASE = "4a6c999"  # the commit before the stabilisation

# the script of test_events_fem with the driver's keywords of the spec (``driver``), which the
# source of 4a6c999 is never given
POST_CUT_RUN = fem_pins._script().replace(
    "driver = CollapseDriver(pb, settings, out, reconnect=recon, **events)",
    "driver = CollapseDriver(pb, settings, out, reconnect=recon, **events, "
    "**spec.get('driver', {}))")
assert POST_CUT_RUN != fem_pins._script()


def _run_fem(src: Path, cwd: Path, spec: dict, out: str = "run") -> Path:
    """:data:`POST_CUT_RUN` in a fresh process (``test_events_fem._run`` with its script)."""
    cwd.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-c", POST_CUT_RUN, str(src), out, json.dumps(spec)],
                          capture_output=True, text=True, timeout=900, cwd=cwd, env=env,
                          check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return cwd / out


def _spec(**events) -> dict:
    return {"physics": fem_pins.fem.PHYSICS, "mesh": fem_pins.fem.MESH, "cut": fem_pins.CUT,
            "collapse": fem_pins.COLLAPSE, "events": events}


def test_the_stops_are_4a6c999s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    out = {name: json.loads(_run(BATTERY, src, tmp_path / name).read_text())
           for name, src in (("ref", ref), ("head", ROOT / "src"))}
    assert out["ref"] == out["head"]
    assert len(out["head"]) > 40


def test_a_finite_element_run_with_the_block_is_4a6c999s_and_so_is_its_step_timing(tmp_path):
    """The run after the cut with the block, and a restart from its dump of step 3, against
    4a6c999's; the same run with the step timing on writes the same files and
    ``step_timing.txt``, a row per step."""
    ref = _reference_src(tmp_path, BASE)
    spec = _spec()
    runs = {name: _run_fem(src, tmp_path / name, spec) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    names = fem_pins._same_outputs(runs["ref"], runs["head"])
    result = json.loads((runs["head"] / "result.json").read_text())
    assert result["steps"] == 6 and result["remesh_count"] >= 2, result
    assert "budget.txt" in names and "step_timing.txt" not in names
    again = {}
    for name, src in (("ref", ref), ("head", ROOT / "src")):
        r = dict(spec, restart=str(tmp_path / name / "run" / "states" / "state_00000003.dump"))
        again[name] = _run_fem(src, tmp_path / f"{name}-restart", r)
    fem_pins._same_outputs(again["ref"], again["head"])
    timed = _run_fem(ROOT / "src", tmp_path / "timed", dict(spec, driver={"step_timing": True}))
    rows = (timed / "step_timing.txt").read_text().splitlines()
    assert len(rows) == 1 + result["steps"]
    (timed / "step_timing.txt").unlink()
    fem_pins._same_outputs(runs["ref"], timed)


def test_a_ring_run_is_4a6c999s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    runs = {name: _run(RING_RUN, src, tmp_path / name) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    for run in runs.values():
        _drop_wall_column(run / "events.txt")
    names = fem_pins._same_outputs(runs["ref"], runs["head"])
    result = json.loads((runs["head"] / "result.json").read_text())
    assert result["stop"]["reason"] == "max-steps" and result["steps"] == 4, result["stop"]
    assert result["events"]["made"] == 1 and result["events"]["ring"]["cut"] == 1
    assert {"ring.txt", "events.txt", "budget.txt", "events/event_001.json"} <= set(names)
    assert _without_wall is not None
