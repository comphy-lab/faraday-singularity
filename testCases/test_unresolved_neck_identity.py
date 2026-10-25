"""The cut of a far head whose neck a freshly built mesh does not resolve, against 7efe4df, the
commit before it (the merge of the ring stage into the polytropic bubble after the cut): the source
of this commit and that of 7efe4df, each in a fresh process with MKL on one thread. Runs that meet
no unresolved neck are 7efe4df's to the byte:

* the stops of the events branch on every synthetic state of ``test_events_gasneck.py``'s battery;
* the finite-element run after the cut of the synthetic necked cavity with the block and
  ``groove.enabled`` (``groove.sheets`` ``log``), and a restart in a new process from one of its
  dumps: every output and every mesh gmsh wrote;
* the finite-element run of the synthetic pierced state with ``ring.enabled`` and
  ``groove.enabled``: the ring cut made by the driver and the steps of the ring stage;
* the far-head cut of ``test_jetcore_fem.py``'s heads' state made by the driver (a gas neck on the
  bubble's head, ``pinch.far_head``), with and without ``groove.enabled``, and the steps after it;
* the cut of the synthetic necked cavity's landing state of a two-way gas with a polytropic bubble
  (``test_bubble_gas_fem.py``'s, from one landing state) and its steps.
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
from test_events_fem import _reference_src  # noqa: E402
from test_events_gasneck import BATTERY, _run  # noqa: E402
from test_far_head_first_identity import FAR_HEAD_RUN  # noqa: E402
from test_groove_identity import RING_RUN_GROOVE  # noqa: E402
from test_jetcore_identity import _run_fem, _spec  # noqa: E402
from test_ring_identity import _drop_wall_column  # noqa: E402

BASE = "7efe4df"  # the commit before the cut of a far head at an unresolved neck


def test_the_stops_are_7efe4dfs(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    out = {name: json.loads(_run(BATTERY, src, tmp_path / name).read_text())
           for name, src in (("ref", ref), ("head", ROOT / "src"))}
    assert out["ref"] == out["head"]
    assert len(out["head"]) > 40


def test_a_finite_element_run_with_groove_enabled_and_its_restart_are_7efe4dfs(tmp_path):
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


def test_a_ring_run_with_groove_enabled_is_7efe4dfs(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    runs = {name: _run(RING_RUN_GROOVE, src, tmp_path / name) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    for run in runs.values():
        _drop_wall_column(run / "events.txt")
    names = fem_pins._same_outputs(runs["ref"], runs["head"])
    result = json.loads((runs["head"] / "result.json").read_text())
    assert result["events"]["made"] == 1 and result["events"]["ring"]["cut"] == 1
    assert {"ring.txt", "events.txt", "budget.txt", "events/event_001.json"} <= set(names)


def test_the_far_head_cut_made_by_the_driver_is_7efe4dfs(tmp_path):
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


def _gas_run(src: Path, cwd: Path, spec: dict) -> Path:
    """``test_neckgas_restart_fem.py``'s runner script in a fresh process with the package at
    ``src``, MKL on one thread; its outputs at the relative ``run``."""
    import test_neckgas_restart_fem as R

    cwd.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-c", R._script(), str(src), "run", json.dumps(spec)],
                          capture_output=True, text=True, timeout=1800, cwd=cwd, env=env,
                          check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return cwd / "run"


def test_the_cut_with_a_polytropic_bubble_is_7efe4dfs(tmp_path):
    import test_neckgas_restart_fem as R
    from test_bubble_gas_fem import CUT

    ref = _reference_src(tmp_path, BASE)
    landing = _gas_run(ROOT / "src", tmp_path / "landing", R.SPEC) / "states" / \
        "landing_0.0153.dump"
    assert landing.is_file()
    spec = dict(R.SPEC, restart=str(landing), cut=dict(CUT, bubble_model="polytropic"),
                traction_out=True)
    spec["collapse"] = dict(R.SPEC["collapse"], max_steps=9)
    runs = {name: _gas_run(src, tmp_path / name, spec) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    names = fem_pins._same_outputs(runs["ref"], runs["head"])
    assert {"bubble_gas.txt", "gas_cut.json", "traction_end.json"} <= set(names)
