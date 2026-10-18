"""Finite-element tests of the run after a cut with and without the events block
(``collapse.reconnect_events``).

* without the block a run after a cut is that of commit 1b65916, to the byte: the cut of the
  synthetic necked cavity of ``test_reconnect_fem.py`` and a few steps of the driver after it, with
  rebuilds and a dump at every step, and a restart in a new process from one of those dumps, each
  run by the source of this commit and by that of 1b65916 in fresh processes with MKL on one thread:
  the same files with the same bytes, but for the wall times;

Each run compiles its problem in seconds; the module takes about two minutes.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("pyoomph")

import test_reconnect_fem as fem  # noqa: E402

REFERENCE = "1b65916"  # the commit the runs after a cut without the block must reproduce

POST_CUT_RUN = """
import json, math, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np
from faraday_singularity import surgery
from faraday_singularity.driver import CollapseDriver, CollapseSettings
from faraday_singularity.mesh import FaradayCylinderMesh, MeshSettings
from faraday_singularity.parameters import CaseParameters, nondimensionalise
from faraday_singularity.problem import FaradayProblem
from faraday_singularity.reconnect import ReconnectSettings, neck_radius
R_N = %(R_N)r
%(neck_interface)s
%(impose_closing_flow)s

class NeckMesh(FaradayCylinderMesh):
    def _interface_points(self, segs=None):
        if self.is_first_time():
            return neck_interface(self.radius)
        return super()._interface_points(segs)


class NeckProblem(FaradayProblem):
    mesh_class = NeckMesh


out, spec = Path(sys.argv[2]), json.loads(sys.argv[3])
nd = nondimensionalise(CaseParameters.from_dict(spec["physics"]))
recon = ReconnectSettings(**spec["cut"])
settings = CollapseSettings(t_end=10.0, dt_max=0.01, **spec["collapse"])
events = {}
if spec.get("events") is not None:
    from faraday_singularity.events import EventSettings
    events = {"events": EventSettings.from_dict(spec["events"])}
pb = NeckProblem(nd, mesh=MeshSettings(**spec["mesh"]), reconnect=recon)
pb.set_output_directory(str(out))
with pb:
    pb.initialise()
    driver = CollapseDriver(pb, settings, out, reconnect=recon, **events)
    if spec.get("restart") is None:
        impose_closing_flow(pb)
        surgery.cut(pb, recon, out, driver=driver, log=lambda *_: None)
    else:
        state = Path(spec["restart"])
        pb.load_state(str(state))
        driver.restore(json.loads(state.with_suffix(".json").read_text()))
    pb.start_diagnostics(out / "diagnostics.txt")
    result = driver.run(1e-8)
    driver.close()
    pb.close_diagnostics()
(out / "result.json").write_text(json.dumps(result))
"""


def _script() -> str:
    return POST_CUT_RUN % {"R_N": fem.R_N, "neck_interface": inspect.getsource(fem.neck_interface),
                           "impose_closing_flow": inspect.getsource(fem.impose_closing_flow)}


def _run(src: Path, cwd: Path, spec: dict, out: str = "run") -> Path:
    """The script in a fresh process from ``cwd`` with the package at ``src``, MKL on one thread;
    its outputs under the relative ``out``, so that the paths they record are the same for every
    source."""
    cwd.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-c", _script(), str(src), out, json.dumps(spec)],
                          capture_output=True, text=True, timeout=900, cwd=cwd, env=env,
                          check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return cwd / out


def _reference_src(tmp_path: Path) -> Path:
    ref = tmp_path / "ref"
    try:
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", REFERENCE, "src"],
                                 capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"commit {REFERENCE} is not available")
    ref.mkdir()
    subprocess.run(["tar", "-x", "-C", str(ref)], input=archive, check=True)
    return ref / "src"


def _without_wall(x):
    """A JSON value without its wall times (``wall_seconds``), which no two runs share."""
    if isinstance(x, dict):
        return {k: _without_wall(v) for k, v in x.items() if k != "wall_seconds"}
    if isinstance(x, list):
        return [_without_wall(v) for v in x]
    return x


def _same_outputs(a: Path, b: Path) -> list[str]:
    """Every file of two run directories (but pyoomph's own, under ``_*``) the same: to the byte,
    but the wall-time column of ``collapse.txt`` and the wall times of the JSON records."""
    names = sorted(f.relative_to(a).as_posix() for f in a.rglob("*") if f.is_file()
                   and not f.relative_to(a).parts[0].startswith("_"))
    assert names == sorted(f.relative_to(b).as_posix() for f in b.rglob("*") if f.is_file()
                           and not f.relative_to(b).parts[0].startswith("_"))
    for name in names:
        x, y = a / name, b / name
        if name == "collapse.txt":
            hx, *rx = x.read_text().splitlines()
            hy, *ry = y.read_text().splitlines()
            wall = hx[2:].split("\t").index("wall")
            assert hx == hy and len(rx) == len(ry), name
            for u, v in zip(rx, ry):
                cu, cv = u.split("\t"), v.split("\t")
                assert cu[:wall] + cu[wall + 1:] == cv[:wall] + cv[wall + 1:], name
        elif name.endswith(".json"):
            assert _without_wall(json.loads(x.read_text())) == _without_wall(
                json.loads(y.read_text())), name
        else:
            assert x.read_bytes() == y.read_bytes(), name
    return names


CUT = dict(fem.FAST_CUT)
# a refine ratio below one makes the rule rebuild after the cut, so that the run goes through the
# rebuilds of the post-cut mode too; a dump at every step
COLLAPSE = {"max_steps": 6, "refine_ratio": 0.5, "remesh_backoff": 2, "state_dt": 0.0,
            "state_keep": 10, "snapshot_dt": 0.0}


def test_runs_after_a_cut_without_the_block_are_as_at_1b65916(tmp_path):
    """Without the events block the run after a cut, its rebuilds, dumps and a restart from one of
    them in a new process, writes what the source of 1b65916 writes, to the byte (the wall times
    aside)."""
    ref = _reference_src(tmp_path)
    spec = {"physics": fem.PHYSICS, "mesh": fem.MESH, "cut": CUT, "collapse": COLLAPSE}
    runs = {name: _run(src, tmp_path / name, spec) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    names = _same_outputs(runs["ref"], runs["head"])
    result = json.loads((runs["head"] / "result.json").read_text())
    assert result["steps"] == 6 and result["remesh_count"] >= 2, result
    assert "states/precut_0.021.dump" in names and "chains.txt" in names
    events = [json.loads(x) for x in (runs["head"] / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "remesh" for e in events)
    assert not any(e["event"] == "leg" for e in events)
    meta = json.loads((runs["head"] / "states" / "state_00000003.json").read_text())
    assert not {"events_settings", "events", "legs"} & set(meta)
    # a restart in a new process from the dump of step 3 to step 6
    again = {}
    for name, src in (("ref", ref), ("head", ROOT / "src")):
        r = dict(spec, restart=str(tmp_path / name / "run" / "states" / "state_00000003.dump"))
        again[name] = _run(src, tmp_path / f"{name}-restart", r)
    _same_outputs(again["ref"], again["head"])
    assert json.loads((again["head"] / "result.json").read_text())["steps"] == 6
    assert np.isfinite(np.loadtxt(again["head"] / "chains.txt", ndmin=2)).any()
