"""Finite-element tests of the run after a cut with and without the events block
(``collapse.reconnect_events``).

* without the block a run after a cut is that of commit 1b65916, to the byte: the cut of the
  synthetic necked cavity of ``test_reconnect_fem.py`` and a few steps of the driver after it, with
  rebuilds and a dump at every step, and a restart in a new process from one of those dumps, each
  run by the source of this commit and by that of 1b65916 in fresh processes with MKL on one thread:
  the same files with the same bytes, but for the wall times;
* with the block the same run goes through the stops of the events branch, and oomph-lib's
  smallest step is lowered after the cut (``post.minimum_dt``), but not before it; a restart from
  its postcut dump checks the loaded mesh as a fresh one and takes its steps;
* with the block the meshes no dump refers to are pruned (``post.prune_gmsh``), the rows of the
  run unchanged, and the dumps kept restart with their meshes in new processes;
* oomph-lib's smallest step (``DTSF_minimum_dt``) on a one-unknown problem: at its default of
  1e-12 an accepted step of 1e-13 ends the solve (its next candidate, four times the step, lies
  below it), and so does a step of 1e-12 that fails, which is not halved; lowered, the first is
  accepted and the second rescued by halving.

Each run compiles its problem in seconds; the module takes about four minutes.
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
    result["minimum_dt"] = pb.DTSF_minimum_dt
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


def test_with_the_block_the_run_after_a_cut_lowers_oomph_libs_smallest_step(tmp_path):
    """The run after the cut of the synthetic cavity with the block: the stops of the events branch
    let it take its steps, and oomph-lib's smallest step is post.minimum_dt after the cut, at
    the default (1e-12) before it; without the block it stays at the default."""
    spec = {"physics": fem.PHYSICS, "mesh": fem.MESH, "cut": CUT,
            "collapse": dict(COLLAPSE, max_steps=3, refine_ratio=1.5),
            "events": {"post": {"minimum_dt": 1e-16}}}
    out = _run(ROOT / "src", tmp_path / "block", spec)
    result = json.loads((out / "result.json").read_text())
    assert result["stop"]["reason"] == "max-steps" and result["steps"] == 3
    assert result["minimum_dt"] == 1e-16
    events = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
    leg = next(e for e in events if e["event"] == "leg")
    assert leg["start"] == "cut" and leg["minimum_dt"] == 1e-16
    meta = json.loads((out / "states" / "precut_0.021.json").read_text())
    assert "events_settings" not in meta  # the state before the cut carries no block
    plain = _run(ROOT / "src", tmp_path / "plain", dict(spec, events=None))
    assert json.loads((plain / "result.json").read_text())["minimum_dt"] == 1e-12
    # a restart in a new process from the postcut dump, whose mesh no step has run on: checked as
    # a fresh mesh (the curved elements of the loaded mesh included), it takes its steps
    (postcut,) = (out / "states").glob("postcut_*.dump")
    assert json.loads(postcut.with_suffix(".json").read_text())["mesh_fresh"] is True
    again = _run(ROOT / "src", tmp_path / "again", dict(spec, restart=str(postcut)))
    result = json.loads((again / "result.json").read_text())
    assert result["stop"]["reason"] == "max-steps" and result["steps"] == 3
    assert result["minimum_dt"] == 1e-16


def _mesh_files(run: Path) -> tuple[set, set]:
    """The meshes in a run's ``_gmsh`` and those its dumps refer to (pyoomph's
    ``get_state_file_mesh_files``), resolved."""
    from pyoomph.output.states import get_state_file_mesh_files

    meshes = {p.resolve() for p in (run / "_gmsh").glob("*.msh")}
    referred = set()
    for dump in (run / "states").glob("*.dump"):
        referred |= {Path(f).resolve() for f in get_state_file_mesh_files(str(dump))}
    return meshes, referred


def test_with_the_block_the_meshes_no_dump_refers_to_are_pruned_and_every_dump_restarts(tmp_path):
    """With the block (``post.prune_gmsh``, the default) the run after the cut of the synthetic
    cavity, rebuilt at every step and keeping its two latest rotating dumps, deletes the meshes of
    the dumps rotated out: every mesh left is one a dump refers to (or the one the problem was
    built from at the start), and fewer are left than the run built; the rows of the run are those
    of the same run without pruning, which keeps every mesh. In a new process the state the run
    ended on and the state after the cut, each with its mesh, restart and take their steps."""
    collapse = dict(COLLAPSE, state_keep=2)
    spec = {"physics": fem.PHYSICS, "mesh": fem.MESH, "cut": CUT, "collapse": collapse,
            "events": {}}
    runs = {name: _run(ROOT / "src", tmp_path / name, dict(spec, events=block))
            for name, block in (("pruned", {}), ("kept", {"post": {"prune_gmsh": False}}))}
    results = {name: json.loads((run / "result.json").read_text()) for name, run in runs.items()}
    for result in results.values():
        assert result["stop"]["reason"] == "max-steps" and result["steps"] == 6
        assert result["remesh_count"] >= 3  # the cut's and at least two rebuilds
    assert results["kept"]["events"]["meshes_pruned"] == 0
    assert results["pruned"]["events"]["meshes_pruned"] > 0
    meshes, referred = _mesh_files(runs["pruned"])
    first = (runs["pruned"] / "_gmsh" / "FaradayCylinderMesh.msh").resolve()
    assert meshes - {first} <= referred and referred <= meshes
    all_meshes, _ = _mesh_files(runs["kept"])
    assert len(meshes) < len(all_meshes)
    assert not [p for p in (runs["pruned"] / "_gmsh").iterdir()
                if p.with_suffix(".msh").resolve() not in meshes]  # no sidecar left alone
    for name in ("chains.txt", "budget.txt"):
        assert (runs["pruned"] / name).read_text() == (runs["kept"] / name).read_text(), name
    a, b = ((runs[x] / "collapse.txt").read_text().splitlines() for x in ("pruned", "kept"))
    wall = a[0][2:].split("\t").index("wall")
    assert [r.split("\t")[:wall] for r in a] == [r.split("\t")[:wall] for r in b]
    # every dump kept restarts in a new process: the end state and the state after the cut
    end = Path(results["pruned"]["stop"]["end_state"])
    (postcut,) = (runs["pruned"] / "states").glob("postcut_*.dump")
    for name, state in (("end", end if end.is_absolute() else runs["pruned"].parent / end),
                        ("postcut", postcut)):
        more = dict(collapse, max_steps=json.loads(state.with_suffix(".json").read_text())["step"]
                    + 2)
        again = _run(ROOT / "src", tmp_path / f"again-{name}",
                     dict(spec, collapse=more, restart=str(state.resolve())))
        result = json.loads((again / "result.json").read_text())
        assert result["stop"]["reason"] == "max-steps" and result["steps"] == more["max_steps"]


ODE_RUN = """
import json, sys
from pyoomph import Problem
from pyoomph.equations.generic import InitialCondition
from pyoomph.expressions import partial_t, testfunction, var
from pyoomph.generic.codegen import ODEEquations

spec = json.loads(sys.argv[1])


class Stiff(ODEEquations):
    def define_fields(self):
        self.define_ode_variable("y")

    def define_residuals(self):
        y = var("y")
        self.add_residual((partial_t(y) + spec["K"] * y**3) * testfunction("y"))


class P(Problem):
    def define_problem(self):
        self.add_equations((Stiff() + InitialCondition(y=1.0)) @ "ode")


pb = P()
pb.set_output_directory(spec["out"])
out = {}
with pb:
    pb.initialise()
    out["default"] = pb.DTSF_minimum_dt
    if spec.get("max_residuals") is not None:
        pb.max_residuals = spec["max_residuals"]
    if spec.get("minimum_dt") is not None:
        pb.DTSF_minimum_dt = spec["minimum_dt"]
    dt = spec["dt"]
    try:
        pb.run(dt, timestep=dt, maxstep=dt, outstep=False, temporal_error=1.0,
               newton_solver_tolerance=spec["tol"])
        out["ok"] = True
    except RuntimeError as exc:
        out["ok"], out["error"] = False, str(exc)
    out["time"] = pb.get_current_time(dimensional=False, as_float=True)
    out["steps"] = pb.timestepper.get_num_unsteady_steps_done()
log = open(spec["out"] + "/_pyoomph_logfile.txt").read()
out["rejected"] = log.count("TIMESTEP REJECTED")
out["minimum_dt_message"] = [x for x in log.splitlines() if "less than the minimum dt" in x]
print(json.dumps(out))
"""


def _ode(tmp_path, name, **spec):
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    spec = {"out": str(tmp_path / name), "tol": 1e-2, **spec}
    proc = subprocess.run([sys.executable, "-c", ODE_RUN, json.dumps(spec)], capture_output=True,
                          text=True, timeout=600, cwd=tmp_path, env=env, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_oomph_libs_smallest_step_ends_accepted_and_failed_steps_unless_lowered(tmp_path):
    """dy/dt = -K y^3 in one unknown. An accepted step of 1e-13 (K = 1): at the default smallest
    step (1e-12) oomph-lib gives up after it, since its next candidate, 4e-13, lies below; lowered
    to 1e-15 the step is taken. A step of 1e-12 whose Newton solve fails (K = 1e9, the residual of
    its first iterate, about 3e3, beyond max_residuals 1e3): at the default it is not halved, the
    solve gives up; lowered it is halved once and taken in two steps of about 5e-13."""
    accepted = {"K": 1.0, "dt": 1e-13}
    out = _ode(tmp_path, "accepted-default", **accepted)
    assert out["default"] == 1e-12 and not out["ok"] and out["rejected"] == 0
    assert "reduce dt to 4e-13 which is less than the minimum dt (1e-12)" in \
        out["minimum_dt_message"][-1]
    out = _ode(tmp_path, "accepted-lowered", minimum_dt=1e-15, **accepted)
    assert out["ok"] and out["steps"] == 1
    assert out["time"] == pytest.approx(1e-13, rel=1e-12, abs=0.0)
    failing = {"K": 1e9, "dt": 1e-12, "max_residuals": 1e3}
    out = _ode(tmp_path, "failing-default", **failing)
    assert not out["ok"] and out["rejected"] == 1
    assert "reduce dt to 5e-13 which is less than the minimum dt (1e-12)" in \
        out["minimum_dt_message"][-1]
    out = _ode(tmp_path, "failing-lowered", minimum_dt=1e-15, **failing)
    assert out["ok"] and out["rejected"] == 1 and out["steps"] == 2
    assert out["time"] == pytest.approx(1e-12, rel=1e-4, abs=0.0) and not out["minimum_dt_message"]
