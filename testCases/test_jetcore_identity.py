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
files, ``step_timing.txt`` besides. With the part ``jetcore`` enabled at its defaults (the far
Courant number the core's) the finite-element run after the cut, its restart and the ring run have
the same solution to the byte: only ``jetcore.txt`` is added and the records hold the part.
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


# --- with jetcore at its defaults: the same solution, jetcore.txt besides -------------------------


def _without_jetcore(x):
    """A record without what the part ``jetcore`` adds to it by design: its settings (the block
    holds the part, ``jetcore`` wherever it is), the hash of the block and the resolved keys of
    each leg (``limits.jetcore``)."""
    if isinstance(x, dict):
        return {k: _without_jetcore(v) for k, v in x.items()
                if k not in ("jetcore", "settings_sha256", "wall_seconds")}
    if isinstance(x, list):
        return [_without_jetcore(v) for v in x]
    return x


def _same_solution(ref: Path, head: Path) -> list[str]:
    """``test_events_fem._same_outputs`` for a run of ``head`` with jetcore at its defaults against
    one of ``ref`` without it: every file of ``ref`` the same, the JSON records and the lines of
    ``events.jsonl`` but for :func:`_without_jetcore`, the meshes gmsh wrote by name and hash, and
    ``jetcore.txt`` in ``head`` only."""
    assert fem_pins._gmsh_files(ref) == fem_pins._gmsh_files(head)

    def files(run):
        return sorted(f.relative_to(run).as_posix() for f in run.rglob("*") if f.is_file()
                      and not f.relative_to(run).parts[0].startswith("_"))

    names = files(ref)
    assert files(head) == sorted(names + ["jetcore.txt"])
    for name in names:
        x, y = ref / name, head / name
        if name == "collapse.txt":
            hx, *rx = x.read_text().splitlines()
            hy, *ry = y.read_text().splitlines()
            wall = hx[2:].split("\t").index("wall")
            assert hx == hy and len(rx) == len(ry), name
            for u, v in zip(rx, ry):
                cu, cv = u.split("\t"), v.split("\t")
                assert cu[:wall] + cu[wall + 1:] == cv[:wall] + cv[wall + 1:], name
        elif name.endswith(".json"):
            assert _without_jetcore(json.loads(x.read_text())) == _without_jetcore(
                json.loads(y.read_text())), name
        elif name == "events.jsonl":
            a = [_without_jetcore(json.loads(u)) for u in x.read_text().splitlines()]
            b = [_without_jetcore(json.loads(u)) for u in y.read_text().splitlines()]
            assert a == b, name
        else:
            assert x.read_bytes() == y.read_bytes(), name
    return names


def test_with_jetcore_at_its_defaults_the_finite_element_run_is_4a6c999s(tmp_path):
    """The run after the cut with the block and ``jetcore`` enabled at its defaults (the far
    Courant number the core's, no far floor), and a restart from its dump of step 3, against
    4a6c999's runs without the part: the same solution to the byte (every row, profile, dump and
    mesh), only ``jetcore.txt`` added and the block's part in the records."""
    ref = _reference_src(tmp_path, BASE)
    runs = {"ref": _run_fem(ref, tmp_path / "ref", _spec()),
            "head": _run_fem(ROOT / "src", tmp_path / "head", _spec(jetcore={"enabled": True}))}
    _same_solution(runs["ref"], runs["head"])
    rows = (runs["head"] / "jetcore.txt").read_text().splitlines()
    result = json.loads((runs["head"] / "result.json").read_text())
    assert len(rows) == 2 + result["steps"]  # the header, the cut's state and a row per step
    again = {}
    for name, src, spec in (("ref", ref, _spec()),
                            ("head", ROOT / "src", _spec(jetcore={"enabled": True}))):
        r = dict(spec, restart=str(tmp_path / name / "run" / "states" / "state_00000003.dump"))
        again[name] = _run_fem(src, tmp_path / f"{name}-restart", r)
    _same_solution(again["ref"], again["head"])


RING_RUN_JETCORE = RING_RUN.replace(
    '''block = ev.EventSettings.from_dict({"ring": {"enabled": True, "sigma_V": 1e-6, "sigma_Pz": 1e-6,
                                             "sigma_KE": 1e-6}})''',
    '''block = ev.EventSettings.from_dict({"ring": {"enabled": True, "sigma_V": 1e-6, "sigma_Pz": 1e-6,
                                             "sigma_KE": 1e-6},
                                    **({"jetcore": {"enabled": True}}
                                       if hasattr(ev, "JetCoreSettings") else {})})''')
assert RING_RUN_JETCORE != RING_RUN


def test_with_jetcore_at_its_defaults_a_ring_run_is_4a6c999s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    runs = {name: _run(RING_RUN_JETCORE, src, tmp_path / name) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    for run in runs.values():
        _drop_wall_column(run / "events.txt")
    names = _same_solution(runs["ref"], runs["head"])
    assert {"ring.txt", "events.txt", "budget.txt", "events/event_001.json"} <= set(names)
