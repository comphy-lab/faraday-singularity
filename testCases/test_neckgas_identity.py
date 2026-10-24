"""Without a ``collapse.gas`` block nothing has changed (gate G0 of the neck gas): a short collapse
run of the production runner, with rebuilds and a dump at every step, and a restart from one of
its dumps, give the same outputs as the runner, driver and problem of commit 2f3ac25, to the byte
(the wall times, the provenance and the run's own times aside), each in a fresh process with MKL
on one thread and in the same directories."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "2f3ac25"
CASE = {
    "schema": "pyoomph-case-v1", "case_id": "identity",
    "geometry": {"kind": "axisymmetric"},
    "physics": {"fluid": {"Oh": 0.05, "Bo": 0.0}, "container": {"mode": 1, "depth": 2.0},
                "forcing": {"Gamma": 0.3, "detuning": 1.0, "phase": 0.0},
                "initial": {"amplitude": 0.2}},
    "tolerances": {"newton": 1e-8},
    "mesh": {"h_interface": 0.05, "h_axis": 0.05, "h_max": 0.4, "grading": 0.3,
             "interface_points": 201},
    "time_stepping": {"scheme": "BDF2", "steps_per_forcing_period": 160},
    "collapse": {"max_steps": 6, "refine_ratio": 0.5, "remesh_backoff": 2, "state_dt": 0.0,
                 "state_keep": 3, "snapshot_dt": 0.03},
    "output": {"cadence": 0.25, "cadence_unit": "forcing periods", "write_states": False},
    "completion": {"kind": "event", "t_max": 1.0, "unit": "forcing periods"},
}
# fields that record when and where a run ran, not what it computed
VOLATILE = {"started", "ended", "wall_seconds", "wall_time", "provenance", "argv", "runner",
            "component_commit", "case_file", "pyoomph", "detail", "parent_record"}


def _tree(tmp_path: Path) -> Path:
    try:
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", REFERENCE, "src",
                                  "run_case.py", "simulationCases"], capture_output=True,
                                 check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"commit {REFERENCE} is not available")
    ref = tmp_path / f"code-{REFERENCE}"
    ref.mkdir()
    subprocess.run(["tar", "-x", "-C", str(ref)], input=archive, check=True)
    return ref


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k not in VOLATILE}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def _files(root: Path) -> list[str]:
    return sorted(f.relative_to(root).as_posix() for f in root.rglob("*") if f.is_file()
                  and not any(p.startswith("_") for p in f.relative_to(root).parts)
                  and not f.name.endswith(".dump"))


def _without_wall(text: str) -> list[str]:
    lines = text.splitlines()
    wall = lines[0][2:].split("\t").index("wall")
    return [lines[0]] + ["\t".join(x for k, x in enumerate(line.split("\t")) if k != wall)
                         for line in lines[1:]]


def _run(tree: Path, work: Path) -> None:
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    case = work / "case.json"
    case.write_text(json.dumps(CASE))
    runner = tree / "simulationCases" / "run_faraday.py"
    for k, args in enumerate(([str(case), "--out", str(work / "out")],
                              [str(case), "--out", str(work / "again"), "--restart-from",
                               str(work / "out" / "states" / "state_00000004.dump")])):
        if k == 1:
            # the extracted reference tree is no Git checkout, and a tree under development may be
            # dirty: both parents record the same clean commit, which a restart needs
            path = work / "out" / "run-manifest.json"
            manifest = json.loads(path.read_text())
            manifest["component_commit"] = "a" * 40
            path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        proc = subprocess.run([sys.executable, str(runner), *args], capture_output=True, text=True,
                              timeout=900, cwd=work, env=env, check=False)
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


def test_runs_and_restarts_without_the_block_are_as_at_2f3ac25(tmp_path):
    pytest.importorskip("pyoomph")
    work = tmp_path / "work"
    runs = {}
    for name, tree in (("ref", _tree(tmp_path)), ("head", ROOT)):
        work.mkdir()
        _run(tree, work)
        runs[name] = tmp_path / name
        work.rename(runs[name])
    a, b = runs["ref"], runs["head"]
    for leg in ("out", "again"):
        names = _files(a / leg)
        assert names == _files(b / leg), leg
        assert "collapse.txt" in names and "states/state_00000006.json" in names
        for name in names:
            x, y = (a / leg / name).read_text(), (b / leg / name).read_text()
            if name == "collapse.txt":
                assert _without_wall(x) == _without_wall(y), (leg, name)
            elif name.endswith(".json"):
                assert _strip(json.loads(x)) == _strip(json.loads(y)), (leg, name)
            elif name == "progress.jsonl":
                assert [_strip(json.loads(r)) for r in x.splitlines()] == \
                    [_strip(json.loads(r)) for r in y.splitlines()]
            else:
                assert x == y, (leg, name)
        for dump in sorted((a / leg / "states").glob("*.dump")):  # the states, to the byte
            assert dump.read_bytes() == (b / leg / "states" / dump.name).read_bytes(), dump.name
    shutil.rmtree(tmp_path / f"code-{REFERENCE}", ignore_errors=True)
